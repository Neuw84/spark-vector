package org.apache.spark.sql.vector.shuffle.flight

import java.nio.ByteBuffer
import java.util.concurrent.{ConcurrentHashMap, Executors}
import scala.jdk.CollectionConverters._

import io.sparkvector.spark.arrow.VectorAllocators
import org.apache.arrow.flight._
import org.apache.arrow.flight.auth2.{Auth2Constants, BearerCredentialWriter, CallHeaderAuthenticator}
import org.apache.arrow.flight.grpc.CredentialCallOption
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.spark.{SparkConf, SparkEnv}
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.IndexShuffleBlockResolver
import org.apache.spark.storage.ShuffleBlockId

/**
 * The Arrow Flight data plane of the columnar shuffle (#288, slice 3). Every executor runs one
 * [[FlightServer]] on the block manager's host and an ephemeral port; a reducer's `DoGet` names one
 * map output's partition -- `(shuffleId, mapId, reducePartition)` -- and the server streams that
 * block's IPC bytes straight from the shuffle file the slice-2 writer produced, as chunks of a
 * one-column binary stream (#338: Flight's own record-batch framing sends dictionaries once per
 * stream, and our blocks carry a replacement dictionary per record batch). Each block is one Flight
 * stream; coalescing an executor's blocks into one stream is a follow-up.
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

  val BackendKey: String = org.apache.spark.sql.vector.shuffle.VectorShuffleBackend.Key
  val BindHostKey = "spark.vector.shuffle.flight.bindHost"
  val ThreadsKey = "spark.vector.shuffle.flight.threads"

  def backend(conf: SparkConf): String = org.apache.spark.sql.vector.shuffle.VectorShuffleBackend.backendName(conf).toLowerCase

  /**
   * The ticket of one reduce task's blocks on one executor (#347, #411): 4-byte shuffleId, 4-byte
   * startReduce, 4-byte endReduce (exclusive), 4-byte count, then count 8-byte mapIds, big-endian.
   * One `DoGet` per (executor, reduce task): a reduce task's input is a block per map task, and a
   * round trip per block made the reducers latency-bound (#347); with AQE coalescing several
   * partitions into a task, a `DoGet` per partition multiplied the streams by the coalescing factor
   * and the index lookups and file opens with them (#411) -- the server now reads, per map, the
   * contiguous byte range of the task's partitions in one lookup.
   */
  def ticket(shuffleId: Int, startReduce: Int, endReduce: Int, mapIds: Seq[Long]): Ticket = {
    require(endReduce > startReduce, s"empty reduce range [$startReduce, $endReduce)")
    val b = ByteBuffer.allocate(16 + 8 * mapIds.size)
    b.putInt(shuffleId).putInt(startReduce).putInt(endReduce).putInt(mapIds.size)
    mapIds.foreach(b.putLong)
    new Ticket(b.array())
  }

  /** The ticket of one reduce partition's blocks on one executor. */
  def ticket(shuffleId: Int, reduce: Int, mapIds: Seq[Long]): Ticket = ticket(shuffleId, reduce, reduce + 1, mapIds)

  /** The ticket of a single block. */
  def ticket(shuffleId: Int, mapId: Long, reduce: Int): Ticket = ticket(shuffleId, reduce, reduce + 1, Seq(mapId))

  /** `(shuffleId, startReduce, endReduce, mapIds)`. */
  def parseTicket(t: Ticket): (Int, Int, Int, Seq[Long]) = {
    val b = ByteBuffer.wrap(t.getBytes)
    require(b.remaining() >= 16 && (b.remaining() - 16) % 8 == 0, s"malformed shuffle ticket (${b.remaining()} bytes)")
    val shuffleId = b.getInt; val start = b.getInt; val end = b.getInt; val count = b.getInt
    require(end > start, s"malformed shuffle ticket: reduce range [$start, $end)")
    require(count >= 0 && b.remaining() == 8L * count, s"malformed shuffle ticket: $count blocks, ${b.remaining()} bytes left")
    (shuffleId, start, end, Seq.fill(count)(b.getLong))
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

  /**
   * The producer: one block per `DoGet`, its IPC stream sent **as bytes** -- a one-column binary
   * root, one row per chunk of [[ChunkBytes]]. Not as Flight record batches: Flight writes a
   * stream's dictionaries once, at its start, while our blocks carry one *replacement* dictionary
   * per record batch (the slice's distinct strings), so a block decoded and re-framed by Flight
   * reached the client with batches 2..n indexed against batch 1's dictionary -- out-of-bounds reads
   * or the wrong string silently (#338). As bytes, the client reads the block with the same
   * [[io.sparkvector.shuffle.PartitionedIpcFile.StreamReader]] a local block goes through, and the
   * server neither decodes nor re-encodes anything.
   */
  final class Producer(blockData: (Int, Long, Int, Int) => org.apache.spark.network.buffer.ManagedBuffer, allocator: BufferAllocator) extends NoOpFlightProducer {
    /** A producer serving single partitions: `blockData(shuffleId, mapId, reduce)`. */
    def this(single: (Int, Long, Int) => org.apache.spark.network.buffer.ManagedBuffer, allocator: BufferAllocator) =
      this((shuffleId: Int, mapId: Long, start: Int, end: Int) => {
        require(end == start + 1, s"a single-partition producer asked for [$start, $end)")
        single(shuffleId, mapId, start)
      }, allocator)

    override def getStream(context: FlightProducer.CallContext, ticket: Ticket, listener: FlightProducer.ServerStreamListener): Unit = {
      val (shuffleId, reduce, endReduce, mapIds) = parseTicket(ticket)
      val root = VectorSchemaRoot.create(BytesSchema, allocator)
      val vector = root.getVector(0).asInstanceOf[org.apache.arrow.vector.VarBinaryVector]
      var current = -1L
      try {
        listener.start(root)
        val chunk = new Array[Byte](ChunkBytes)
        // The blocks back to back: each is a sequence of IPC streams and the client's reader decodes
        // concatenated streams, so where one block ends and the next begins needs no marker.
        mapIds.foreach { mapId =>
          current = mapId
          // One lookup and one open per map for the task's whole partition range (#411): the
          // partitions are consecutive in the data file, and an empty one is a zero-length span.
          val buf = blockData(shuffleId, mapId, reduce, endReduce)
          val in = buf.createInputStream()
          try {
            var n = readFully(in, chunk)
            while (n > 0) {
              vector.reset()
              vector.setSafe(0, chunk, 0, n)
              vector.setValueCount(1)
              root.setRowCount(1)
              listener.putNext()
              n = readFully(in, chunk)
            }
          } finally {
            in.close()
            buf.release()
          }
        }
        listener.completed()
      } catch {
        case e: Exception =>
          logWarning(s"flight shuffle: serving $shuffleId/$current/[$reduce, $endReduce) (${mapIds.size} map outputs) failed", e)
          listener.error(CallStatus.INTERNAL.withCause(e).withDescription(e.toString).toRuntimeException)
      } finally {
        root.close()
      }
    }

    /** Fills `dst` as far as the stream goes; the count read (0 at end of stream). */
    private def readFully(in: java.io.InputStream, dst: Array[Byte]): Int = {
      var off = 0
      while (off < dst.length) {
        val r = in.read(dst, off, dst.length - off)
        if (r < 0) return off
        off += r
      }
      off
    }
  }

  /** The wire schema of a block: its IPC bytes, chunked. */
  val BytesSchema: org.apache.arrow.vector.types.pojo.Schema = new org.apache.arrow.vector.types.pojo.Schema(
    java.util.List.of(org.apache.arrow.vector.types.pojo.Field.nullable("ipc", org.apache.arrow.vector.types.pojo.ArrowType.Binary.INSTANCE)))
  /** One Flight message per this many bytes of the block. */
  val ChunkBytes: Int = 4 << 20

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
      val builder = FlightServer.builder(allocator, Location.forGrpcInsecure(host, 0), new Producer((s: Int, m: Long, start: Int, end: Int) =>
        if (end == start + 1) resolver().getBlockData(ShuffleBlockId(s, m, start), None)
        else resolver().getBlockData(org.apache.spark.storage.ShuffleBlockBatchId(s, m, start, end), None), allocator)).executor(executor)
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
 * One reducer's blocks on one remote executor over Flight (#347): a single `DoGet` for
 * `(shuffleId, reduce, mapIds)` against the executor's server, the blocks' IPC bytes arriving back to
 * back and decoded by the local path's reader; each batch owns its memory and the previous batch is
 * closed when the next is produced (the consumers do not close their input). The stream is opened at
 * construction, so the streams to several executors, built together, transfer concurrently.
 */
final class FlightBlockStream(
    location: FlightLocation,
    shuffleId: Int,
    mapIds: Seq[Long],
    reduce: Int,
    endReduce: Int,
    conf: SparkConf,
    allocator: BufferAllocator,
    metrics: org.apache.spark.shuffle.ShuffleReadMetricsReporter) extends Iterator[org.apache.spark.sql.vectorized.ColumnarBatch] with AutoCloseable {

  /** One reduce partition's blocks on the executor. */
  def this(location: FlightLocation, shuffleId: Int, mapIds: Seq[Long], reduce: Int, conf: SparkConf, allocator: BufferAllocator,
      metrics: org.apache.spark.shuffle.ShuffleReadMetricsReporter) = this(location, shuffleId, mapIds, reduce, reduce + 1, conf, allocator, metrics)

  /** A single block. */
  def this(location: FlightLocation, shuffleId: Int, mapId: Long, reduce: Int, conf: SparkConf, allocator: BufferAllocator,
      metrics: org.apache.spark.shuffle.ShuffleReadMetricsReporter) = this(location, shuffleId, Seq(mapId), reduce, reduce + 1, conf, allocator, metrics)

  private val stream: FlightStream = {
    val start = System.nanoTime()
    val s = FlightShuffle.Clients.client(location).getStream(FlightShuffle.ticket(shuffleId, reduce, endReduce, mapIds), FlightShuffle.Clients.callOptions(conf): _*)
    metrics.incFetchWaitTime((System.nanoTime() - start) / 1000000)
    s
  }

  /**
   * The block's IPC bytes, pulled from the Flight stream one chunk at a time as the reader asks for
   * them (#338: the block travels as bytes, so every record batch keeps its own dictionary).
   */
  private final class ChunkChannel extends java.nio.channels.ReadableByteChannel {
    private var current: java.nio.ByteBuffer = _
    private var ended = false
    private var open = true
    private def fill(): Boolean = {
      while ((current == null || !current.hasRemaining) && !ended) {
        if (!stream.next()) ended = true
        else {
          val root = stream.getRoot
          if (root.getRowCount > 0) {
            val v = root.getVector(0).asInstanceOf[org.apache.arrow.vector.VarBinaryVector]
            val bytes = v.get(0) // a copy: the root's buffers are reused by the next message
            metrics.incRemoteBytesRead(bytes.length.toLong)
            current = java.nio.ByteBuffer.wrap(bytes)
          }
        }
      }
      current != null && current.hasRemaining
    }
    override def read(dst: java.nio.ByteBuffer): Int = {
      if (!fill()) return -1
      val n = math.min(dst.remaining(), current.remaining())
      val lim = current.limit()
      current.limit(current.position() + n)
      dst.put(current)
      current.limit(lim)
      n
    }
    /** Whether any byte is left, without consuming one. */
    def isEmpty: Boolean = !fill()
    override def isOpen: Boolean = open
    override def close(): Unit = open = false
  }

  private val channel = new ChunkChannel
  /** The local path's decoder over the remote bytes; None for an empty block. */
  private val reader: Option[io.sparkvector.shuffle.PartitionedIpcFile.StreamReader] =
    if (channel.isEmpty) None else Some(new io.sparkvector.shuffle.PartitionedIpcFile.StreamReader(channel, allocator))

  override def hasNext: Boolean = reader.exists(_.hasNext)

  override def next(): org.apache.spark.sql.vectorized.ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException
    reader.get.next()
  }

  override def close(): Unit = {
    reader.foreach(_.close())
    channel.close()
    stream.close()
  }
}
