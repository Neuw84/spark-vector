package org.apache.spark.sql.vector.shuffle.flight

import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.util.concurrent.{ConcurrentHashMap, Executors}
import scala.jdk.CollectionConverters._

import io.sparkvector.spark.arrow.VectorAllocators
import org.apache.arrow.flight._
import org.apache.arrow.flight.auth2.{Auth2Constants, BearerCredentialWriter, CallHeaderAuthenticator}
import org.apache.arrow.flight.grpc.CredentialCallOption
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.ArrowStreamReader
import org.apache.spark.{SparkConf, SparkEnv}
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.IndexShuffleBlockResolver
import org.apache.spark.storage.ShuffleBlockId

/**
 * The Arrow Flight data plane of the columnar shuffle (#288, slice 3). Every executor runs one
 * [[FlightServer]] on the block manager's host and an ephemeral port; a reducer's `DoGet` names one
 * map output's partition -- `(shuffleId, mapId, reducePartition)` -- and the server streams that
 * block's IPC record batches straight from the shuffle file the slice-2 writer produced. Each block
 * is one Flight stream because Flight sends a stream's dictionaries once at its start while our
 * blocks carry their own (possibly different) dictionaries; coalescing an executor's blocks into one
 * stream is a follow-up that needs stable dictionaries.
 *
 * Locations travel through the plugin: the executor plugin starts the server and registers
 * `executorId -> host:port` with the driver plugin; reducers look a location up once per executor
 * and cache it ([[FlightRegistry]]).
 *
 * Security: with `spark.authenticate` on, every call must carry Spark's shuffle secret as a bearer
 * token and an unauthenticated `DoGet` is refused; the server refuses to start when auth is on but
 * no secret can be read. TLS follows `spark.ssl.rpc.enabled` (slice 3 documents the gap: the
 * server needs PEM material, Spark configures JKS; until a converter lands, TLS-on refuses to start
 * so nothing runs in the clear by accident). Without `spark.authenticate` the endpoint is as open as
 * Spark's own block transfer in that configuration.
 */
object FlightShuffle extends Logging {

  val BackendKey = "spark.vector.shuffle.backend"
  val BindHostKey = "spark.vector.shuffle.flight.bindHost"
  val ThreadsKey = "spark.vector.shuffle.flight.threads"

  def backend(conf: SparkConf): String = conf.get(BackendKey, "flight").trim.toLowerCase

  /** The ticket of one block: 4-byte shuffleId, 8-byte mapId, 4-byte reducePartition, big-endian. */
  def ticket(shuffleId: Int, mapId: Long, reduce: Int): Ticket = {
    val b = ByteBuffer.allocate(16)
    b.putInt(shuffleId).putLong(mapId).putInt(reduce)
    new Ticket(b.array())
  }

  def parseTicket(t: Ticket): (Int, Long, Int) = {
    val b = ByteBuffer.wrap(t.getBytes)
    require(b.remaining() == 16, s"malformed shuffle ticket (${b.remaining()} bytes)")
    (b.getInt, b.getLong, b.getInt)
  }

  /** Spark's shuffle secret when `spark.authenticate` is on, else None. */
  def secret(conf: SparkConf): Option[String] =
    if (!conf.getBoolean("spark.authenticate", false)) None
    else {
      val sm = SparkEnv.get.securityManager
      Option(sm.getSecretKey()).filter(_.nonEmpty)
    }

  /** Bearer-token check against the shuffle secret: constant-time compare, one identity. */
  final class SecretAuthenticator(expected: String) extends CallHeaderAuthenticator {
    override def authenticate(incomingHeaders: CallHeaders): CallHeaderAuthenticator.AuthResult = {
      val header = Option(incomingHeaders.get(Auth2Constants.AUTHORIZATION_HEADER)).getOrElse("")
      val prefix = Auth2Constants.BEARER_PREFIX
      if (!header.startsWith(prefix) || !java.security.MessageDigest.isEqual(
          header.substring(prefix.length).getBytes("UTF-8"), expected.getBytes("UTF-8"))) {
        throw CallStatus.UNAUTHENTICATED.withDescription("spark.authenticate is on: a valid bearer token is required").toRuntimeException
      }
      () => "spark"
    }
  }

  /** The producer: one block's IPC stream per `DoGet`, served from the shuffle file, decoded once and re-framed by Flight. */
  final class Producer(resolver: () => IndexShuffleBlockResolver, allocator: BufferAllocator) extends NoOpFlightProducer {
    override def getStream(context: FlightProducer.CallContext, ticket: Ticket, listener: FlightProducer.ServerStreamListener): Unit = {
      val (shuffleId, mapId, reduce) = parseTicket(ticket)
      val buf = resolver().getBlockData(ShuffleBlockId(shuffleId, mapId, reduce), None)
      val in = Channels.newChannel(buf.createInputStream())
      val reader = new ArrowStreamReader(in, allocator)
      try {
        var started = false
        while (reader.loadNextBatch()) {
          val root: VectorSchemaRoot = reader.getVectorSchemaRoot
          if (!started) {
            listener.start(root, reader)
            started = true
          }
          listener.putNext()
        }
        if (!started) {
          // An empty block still has a schema: send it so the client sees a well-formed empty stream.
          listener.start(reader.getVectorSchemaRoot, reader)
        }
        listener.completed()
      } catch {
        case e: Exception =>
          logWarning(s"flight shuffle: serving $shuffleId/$mapId/$reduce failed", e)
          listener.error(CallStatus.INTERNAL.withCause(e).withDescription(e.toString).toRuntimeException)
      } finally {
        reader.close()
        buf.release()
      }
    }
  }

  /** The server of this executor; started once, stopped by the executor plugin. */
  final class Service(conf: SparkConf, hostname: String, resolver: () => IndexShuffleBlockResolver) extends AutoCloseable {
    private val allocator = VectorAllocators.newChild("flight-shuffle-server")
    private val host = conf.get(BindHostKey, hostname)
    private val threads = conf.getInt(ThreadsKey, math.max(4, Runtime.getRuntime.availableProcessors()))
    private val executor = Executors.newFixedThreadPool(threads, r => { val t = new Thread(r, "flight-shuffle"); t.setDaemon(true); t })
    private val server: FlightServer = {
      if (conf.getBoolean("spark.ssl.rpc.enabled", false)) {
        throw new IllegalStateException(
          "spark.ssl.rpc.enabled is on but the Flight shuffle server has no TLS material yet (#288): " +
            "use spark.vector.shuffle.backend=block or turn RPC TLS off")
      }
      val builder = FlightServer.builder(allocator, Location.forGrpcInsecure(host, 0), new Producer(resolver, allocator)).executor(executor)
      secret(conf) match {
        case Some(s) => builder.headerAuthenticator(new SecretAuthenticator(s))
        case None if conf.getBoolean("spark.authenticate", false) =>
          throw new IllegalStateException("spark.authenticate is on but no shuffle secret is available to the Flight shuffle server")
        case None =>
      }
      builder.build()
    }
    server.start()
    logInfo(s"flight shuffle server listening on $host:${server.getPort}")

    def location: FlightLocation = FlightLocation(host, server.getPort)

    override def close(): Unit = {
      try server.close() finally {
        executor.shutdownNow()
        allocator.close()
      }
    }
  }

  /** Client side: one client per remote executor, pooled per JVM; the token attached to every call. */
  object Clients {
    private val allocator = VectorAllocators.newChild("flight-shuffle-client")
    private val clients = new ConcurrentHashMap[FlightLocation, FlightClient]()

    def client(loc: FlightLocation): FlightClient =
      clients.computeIfAbsent(loc, l => FlightClient.builder(allocator, Location.forGrpcInsecure(l.host, l.port)).build())

    def callOptions(conf: SparkConf): Array[CallOption] = secret(conf) match {
      case Some(s) => Array(new CredentialCallOption(new BearerCredentialWriter(s)))
      case None => Array.empty
    }

    def allocatorForReads: BufferAllocator = allocator
  }
}

final case class FlightLocation(host: String, port: Int) extends Serializable

/** Messages between the executor plugins and the driver plugin. */
sealed trait FlightMessage extends Serializable
final case class RegisterFlight(executorId: String, location: FlightLocation) extends FlightMessage
final case class LookupFlight(executorId: String) extends FlightMessage

/**
 * The location registry: the driver plugin holds it and answers lookups; each executor caches what
 * it has asked for. Hooked into `VectorPlugin` reflectively ([[org.apache.spark.sql.vector.VectorShuffle]]).
 */
object FlightRegistry extends Logging {
  private val locations = new ConcurrentHashMap[String, FlightLocation]()
  private val cache = new ConcurrentHashMap[String, FlightLocation]()
  @volatile private var service: FlightShuffle.Service = _
  @volatile private var pluginContext: PluginContext = _

  /** Driver plugin: a registration or a lookup. */
  def driverReceive(message: AnyRef): AnyRef = message match {
    case RegisterFlight(executorId, location) =>
      locations.put(executorId, location)
      logInfo(s"flight shuffle: executor $executorId serves at ${location.host}:${location.port}")
      java.lang.Boolean.TRUE
    case LookupFlight(executorId) => Option(locations.get(executorId)).orNull
    case _ => null
  }

  /** Executor plugin init: start the server when the manager is ours and the backend is Flight. */
  @volatile private var initFailure: Throwable = _

  def executorInit(ctx: PluginContext): Unit =
    try executorInit0(ctx) catch {
      case t: Throwable =>
        // The executor must come up: the failure surfaces on the first fetch that needs this server.
        initFailure = t
        logError("flight shuffle: the server did not start on this executor", t)
    }

  private def executorInit0(ctx: PluginContext): Unit = {
    val conf = ctx.conf()
    pluginContext = ctx
    if (org.apache.spark.sql.vector.shuffle.VectorShuffleManager.isConfigured(conf) && FlightShuffle.backend(conf) == "flight") {
      // The plugin initialises before the executor's block manager: the host comes from the plugin
      // context and the block resolver is looked up when the first DoGet arrives.
      val resolver = () => SparkEnv.get.shuffleManager.shuffleBlockResolver.asInstanceOf[IndexShuffleBlockResolver]
      // PluginContext.hostname() reads the executor RpcEnv's address, which a client-mode RpcEnv does
      // not have; the executor's own advertised host name is what the block manager will use too.
      service = new FlightShuffle.Service(conf, org.apache.spark.util.Utils.localHostName(), resolver)
      val loc = service.location
      // In local mode the executor is the driver: register directly.
      if (ctx.executorID() == "driver") locations.put("driver", loc) else ctx.send(RegisterFlight(ctx.executorID(), loc))
      cache.put(ctx.executorID(), loc)
    }
  }

  def executorShutdown(): Unit = {
    if (service != null) { service.close(); service = null }
  }

  /** Where `executorId`'s Flight server is, asking the driver once. */
  def locationOf(executorId: String): FlightLocation = {
    val cached = cache.get(executorId)
    if (cached != null) return cached
    val answer = Option(locations.get(executorId)).orElse {
      Option(pluginContext).flatMap(ctx => Option(ctx.ask(LookupFlight(executorId))).map(_.asInstanceOf[FlightLocation]))
    }
    val loc = answer.getOrElse(throw new IllegalStateException(
      s"no Flight shuffle server registered for executor $executorId" +
        (if (initFailure != null) s"; this executor's own server failed to start: $initFailure" else "")))
    cache.put(executorId, loc)
    loc
  }

  def isServing: Boolean = service != null

  /** Any registered server (the driver's view), for tests and diagnostics. */
  def anyLocation: Option[FlightLocation] = locations.values().asScala.headOption
}

/**
 * One remote block over Flight: a `DoGet` for `(shuffleId, mapId, reduce)` against the executor's
 * server, each arriving root turned into a batch that owns its memory; the previous batch is closed
 * when the next is produced (the consumers do not close their input).
 */
final class FlightBlockStream(
    location: FlightLocation,
    shuffleId: Int,
    mapId: Long,
    reduce: Int,
    conf: SparkConf,
    allocator: BufferAllocator,
    metrics: org.apache.spark.shuffle.ShuffleReadMetricsReporter) extends Iterator[org.apache.spark.sql.vectorized.ColumnarBatch] with AutoCloseable {

  private val stream: FlightStream = {
    val start = System.nanoTime()
    val s = FlightShuffle.Clients.client(location).getStream(FlightShuffle.ticket(shuffleId, mapId, reduce), FlightShuffle.Clients.callOptions(conf): _*)
    metrics.incFetchWaitTime((System.nanoTime() - start) / 1000000)
    s
  }
  private var nextBatch: org.apache.spark.sql.vectorized.ColumnarBatch = _
  private var last: org.apache.spark.sql.vectorized.ColumnarBatch = _
  private var done = false

  private def advance(): Unit = if (!done && nextBatch == null) {
    if (!stream.next()) done = true
    else {
      val root = stream.getRoot
      metrics.incRemoteBytesRead(root.getFieldVectors.asScala.map(_.getBufferSize.toLong).sum)
      nextBatch = io.sparkvector.shuffle.PartitionedIpcFile.toBatch(
        root, id => stream.getDictionaryProvider.lookup(id).getVector.asInstanceOf[org.apache.arrow.vector.VarCharVector], allocator)
    }
  }

  override def hasNext: Boolean = { advance(); nextBatch != null }

  override def next(): org.apache.spark.sql.vectorized.ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException
    val b = nextBatch
    nextBatch = null
    if (last != null) last.close()
    last = b
    b
  }

  override def close(): Unit = {
    if (nextBatch != null) { nextBatch.close(); nextBatch = null }
    if (last != null) { last.close(); last = null }
    stream.close()
  }
}
