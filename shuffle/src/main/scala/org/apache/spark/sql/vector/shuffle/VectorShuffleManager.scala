package org.apache.spark.sql.vector.shuffle

import java.io.File
import java.nio.channels.Channels
import java.util.concurrent.LinkedBlockingQueue
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import io.sparkvector.kernels.PartitionKernels
import io.sparkvector.kernels.PartitionKernels.KeyKind
import io.sparkvector.shuffle.{PartitionedIpcFile, PartitionedIpcWriter}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.arrow.VectorAllocators
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.{Partitioner, ShuffleDependency, SparkConf, SparkEnv, TaskContext}
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.network.shuffle.{BlockFetchingListener, DownloadFileManager}
import org.apache.spark.rdd.RDD
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.shuffle._
import org.apache.spark.shuffle.sort.SortShuffleManager
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, UnsafeProjection}
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{BlockId, BlockManagerId, ShuffleBlockId}
import org.apache.spark.util.Utils

/** How a map task assigns rows to reduce partitions; serialised into the dependency. */
sealed trait VectorPartitioning extends Serializable {
  def numPartitions: Int
}

object VectorPartitioning {
  /** Spark's `HashPartitioning`: `Pmod(Murmur3Hash(keys), n)` over key columns of the batch. */
  final case class Hash(keyOrdinals: Array[Int], kinds: Array[KeyKind], numPartitions: Int) extends VectorPartitioning
  final case class RoundRobin(numPartitions: Int) extends VectorPartitioning
  case object Single extends VectorPartitioning { val numPartitions = 1 }
  /**
   * Spark's `RangePartitioning`: the bounds are Spark's own `RangePartitioner` (sampled by the
   * exchange the way Spark samples), applied to a projection of the sort keys row by row -- the
   * kernel-side binary search over the sort's normalised keys is a follow-up.
   */
  final case class Range(partitioner: Partitioner, sortKeys: Seq[Expression], output: Seq[Attribute]) extends VectorPartitioning {
    def numPartitions: Int = partitioner.numPartitions
  }

  def keyKind(dt: DataType): KeyKind = {
    import org.apache.spark.sql.types._
    dt match {
      case IntegerType | DateType => KeyKind.INT
      case LongType | TimestampType => KeyKind.LONG
      case d: DecimalType if d.precision <= io.sparkvector.spark.adapter.TypeMapping.MAX_DECIMAL_PRECISION => KeyKind.LONG
      case _: DecimalType => KeyKind.DECIMAL128
      case DoubleType => KeyKind.DOUBLE
      case BooleanType => KeyKind.BOOL
      case StringType => KeyKind.UTF8
      case other => throw new IllegalArgumentException(s"no shuffle key kind for $other")
    }
  }
}

/** The dependency our exchange registers; the manager recognises it by type. */
final class VectorShuffleDependency(
    rdd: RDD[Product2[Int, ColumnarBatch]],
    partitioner: Partitioner,
    val schema: StructType,
    val partitioning: VectorPartitioning,
    writeProcessor: ShuffleWriteProcessor)
  extends ShuffleDependency[Int, ColumnarBatch, ColumnarBatch](
    rdd, partitioner, SparkEnv.get.serializer, None, None, false, writeProcessor)

final class VectorShuffleHandle(shuffleId: Int, val dependency: VectorShuffleDependency)
  extends ShuffleHandle(shuffleId)

/**
 * `spark.shuffle.manager` for the columnar shuffle (#288): our dependencies get the Arrow IPC writer
 * and reader below, every other shuffle is Spark's sort shuffle manager untouched -- the pattern of
 * Comet's manager. Map outputs live in the block resolver's data/index files, so `MapStatus`, the
 * `MapOutputTracker` and Spark's own block transfer see ordinary shuffle blocks; slice 3 puts the
 * Flight service in front of the same files for the remote fetch.
 */
final class VectorShuffleManager(conf: SparkConf) extends ShuffleManager {
  private val sort = new SortShuffleManager(conf)

  override def registerShuffle[K, V, C](shuffleId: Int, dependency: ShuffleDependency[K, V, C]): ShuffleHandle =
    dependency match {
      case v: VectorShuffleDependency => new VectorShuffleHandle(shuffleId, v)
      case other => sort.registerShuffle(shuffleId, other)
    }

  override def getWriter[K, V](handle: ShuffleHandle, mapId: Long, context: TaskContext, metrics: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] =
    handle match {
      case v: VectorShuffleHandle =>
        new VectorShuffleWriter(v, mapId, context, metrics, sort.shuffleBlockResolver.asInstanceOf[IndexShuffleBlockResolver]).asInstanceOf[ShuffleWriter[K, V]]
      case other => sort.getWriter(other, mapId, context, metrics)
    }

  override def getReader[K, C](
      handle: ShuffleHandle, startMapIndex: Int, endMapIndex: Int, startPartition: Int, endPartition: Int,
      context: TaskContext, metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] =
    handle match {
      case v: VectorShuffleHandle =>
        new VectorShuffleReader(v, startMapIndex, endMapIndex, startPartition, endPartition, context, metrics).asInstanceOf[ShuffleReader[K, C]]
      case other => sort.getReader(other, startMapIndex, endMapIndex, startPartition, endPartition, context, metrics)
    }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    VectorShuffleBackend(conf).unregisterShuffle(shuffleId)
    sort.unregisterShuffle(shuffleId)
  }
  override def shuffleBlockResolver: ShuffleBlockResolver = sort.shuffleBlockResolver
  override def stop(): Unit = sort.stop()
}

object VectorShuffleManager {
  val ClassName: String = classOf[VectorShuffleManager].getName
  def isConfigured(conf: SparkConf): Boolean = conf.get("spark.shuffle.manager", "sort") == ClassName
}

/**
 * The map task: partition ids from the kernels, batches into per-partition IPC streams
 * ([[PartitionedIpcWriter]]), the streams into the resolver's data file with Spark's index file
 * alongside, so the block manager serves `ShuffleBlockId(shuffle, map, reduce)` as the partition's
 * IPC bytes.
 */
final class VectorShuffleWriter(
    handle: VectorShuffleHandle,
    mapId: Long,
    context: TaskContext,
    metrics: ShuffleWriteMetricsReporter,
    resolver: IndexShuffleBlockResolver) extends ShuffleWriter[Int, ColumnarBatch] {

  private val dep = handle.dependency
  private val numPartitions = dep.partitioner.numPartitions
  private val blockManager = SparkEnv.get.blockManager
  private val dataFile: File = resolver.getDataFile(handle.shuffleId, mapId)
  private val tmp: File = Utils.tempFileWith(dataFile)
  private val allocator: BufferAllocator = VectorAllocators.newChild(s"shuffle-write-${handle.shuffleId}-$mapId")
  private val writer = new PartitionedIpcWriter(dep.schema, numPartitions, allocator, tmp.toPath, VectorShuffleWriter.flushBytes(SparkEnv.get.conf))
  private var lengths: Array[Long] = _
  private var stopped = false
  private var rows = 0L
  private var roundRobinNext = VectorShuffleWriter.roundRobinStart(context, numPartitions)
  private lazy val rangeProjection: UnsafeProjection = dep.partitioning match {
    case r: VectorPartitioning.Range => UnsafeProjection.create(r.sortKeys, r.output)
    case _ => null
  }

  override def write(records: Iterator[Product2[Int, ColumnarBatch]]): Unit = {
    val start = System.nanoTime()
    while (records.hasNext) {
      val batch = records.next()._2
      val n = batch.numRows()
      if (n > 0) {
        val ids = partitionIds(batch, n)
        writer.write(batch, ids)
        rows += n
      }
    }
    metrics.incWriteTime(System.nanoTime() - start)
  }

  private def partitionIds(batch: ColumnarBatch, n: Int): Array[Int] = {
    val ids = new Array[Int](n)
    dep.partitioning match {
      case VectorPartitioning.Hash(ordinals, kinds, num) =>
        val arena = java.lang.foreign.Arena.ofConfined()
        try {
          val keys = ordinals.map(o => ColumnVectorAdapters.adapt(batch.column(o), n, arena))
          PartitionKernels.hashPartitionIds(keys, kinds, n, num, new Array[Int](n), ids)
        } finally arena.close()
      case VectorPartitioning.RoundRobin(num) =>
        roundRobinNext = PartitionKernels.roundRobinIds(n, num, roundRobinNext, ids)
      case VectorPartitioning.Single =>
        java.util.Arrays.fill(ids, 0)
      case r: VectorPartitioning.Range =>
        val it = batch.rowIterator()
        var i = 0
        while (it.hasNext) {
          ids(i) = r.partitioner.getPartition(rangeProjection(it.next()))
          i += 1
        }
    }
    ids
  }

  override def stop(success: Boolean): Option[MapStatus] = {
    if (stopped) return None
    stopped = true
    try {
      if (!success) {
        writer.close()
        tmp.delete()
        None
      } else {
        val index = writer.finish(withFooter = false)
        writer.close()
        lengths = index.lengths
        resolver.writeMetadataFileAndCommit(handle.shuffleId, mapId, lengths, Array.emptyLongArray, tmp)
        VectorShuffleBackend(SparkEnv.get.conf).mapOutputCommitted(handle.shuffleId, mapId, dataFile, lengths)
        metrics.incBytesWritten(lengths.sum)
        metrics.incRecordsWritten(rows)
        Some(MapStatus(blockManager.shuffleServerId, lengths, mapId))
      }
    } finally {
      allocator.close()
      if (tmp.exists()) tmp.delete()
    }
  }

  override def getPartitionLengths(): Array[Long] = lengths
}

object VectorShuffleWriter {
  val FlushBytesKey = "spark.vector.shuffle.flushBytes"
  def flushBytes(conf: SparkConf): Long = conf.getSizeAsBytes(FlushBytesKey, "1m")
  /** Spark's round robin starts each task at a random partition: same here, seeded by the partition id. */
  def roundRobinStart(context: TaskContext, numPartitions: Int): Int =
    new java.util.Random(context.partitionId()).nextInt(numPartitions)
}

/**
 * The reduce side: the map outputs' non-empty blocks for the reduce partitions, each an IPC stream
 * decoded by [[PartitionedIpcFile.StreamReader]], from wherever the configured
 * [[VectorShuffleBackend]] gets them -- the local file when the map ran here, a Flight `DoGet` or
 * Spark's block transfer otherwise.
 */
final class VectorShuffleReader(
    handle: VectorShuffleHandle,
    startMapIndex: Int,
    endMapIndex: Int,
    startPartition: Int,
    endPartition: Int,
    context: TaskContext,
    metrics: ShuffleReadMetricsReporter) extends ShuffleReader[Int, ColumnarBatch] {

  override def read(): Iterator[Product2[Int, ColumnarBatch]] = {
    val env = SparkEnv.get
    val blocksByAddress = env.mapOutputTracker
      .getMapSizesByExecutorId(handle.shuffleId, startMapIndex, endMapIndex, startPartition, endPartition)
      .toSeq
    val allocator = VectorAllocators.newChild(s"shuffle-read-${handle.shuffleId}-${context.partitionId()}")
    val open = new java.util.ArrayList[AutoCloseable]()
    context.addTaskCompletionListener[Unit] { _ =>
      open.asScala.foreach(c => try c.close() catch { case _: Exception => })
      allocator.close()
    }
    val nonEmpty = blocksByAddress.map { case (address, blocks) =>
      address -> blocks.collect { case (id: ShuffleBlockId, size, _) if size > 0 => (id, size) }.toIndexedSeq
    }
    val streams = VectorShuffleBackend(env.conf).read(nonEmpty, allocator, metrics)
    streams.flatMap { reader =>
      open.add(reader)
      new Iterator[Product2[Int, ColumnarBatch]] {
        private var live = true
        override def hasNext: Boolean = {
          if (!live) return false
          val more = reader.hasNext
          if (!more) { reader.close(); open.remove(reader); live = false }
          more
        }
        override def next(): Product2[Int, ColumnarBatch] = {
          val b = reader.next()
          metrics.incRecordsRead(b.numRows())
          (0, b)
        }
      }
    }
  }
}

object VectorShuffleReader {
  /** A fetched or local block (one partition's IPC stream) as batches; the buffer is released with the stream. */
  def blockStream(buf: ManagedBuffer, allocator: BufferAllocator): Iterator[ColumnarBatch] with AutoCloseable =
    new Iterator[ColumnarBatch] with AutoCloseable {
      private val inner = new PartitionedIpcFile.StreamReader(Channels.newChannel(buf.createInputStream()), allocator)
      private var released = false
      override def hasNext: Boolean = inner.hasNext
      override def next(): ColumnarBatch = inner.next()
      override def close(): Unit = { inner.close(); if (!released) { buf.release(); released = true } }
    }

  /** Interim remote path over Spark's block transfer: all of one executor's blocks in one request, collected as they land. */
  def fetchRemote(address: BlockManagerId, ids: Seq[BlockId], metrics: ShuffleReadMetricsReporter): Iterator[(BlockId, ManagedBuffer)] = {
    val client = SparkEnv.get.blockManager.blockStoreClient
    val queue = new LinkedBlockingQueue[Either[Throwable, (BlockId, ManagedBuffer)]]()
    val start = System.nanoTime()
    client.fetchBlocks(address.host, address.port, address.executorId, ids.map(_.toString).toArray, new BlockFetchingListener {
      override def onBlockFetchSuccess(blockId: String, data: ManagedBuffer): Unit = {
        data.retain()
        queue.put(Right((BlockId(blockId), data)))
      }
      override def onBlockFetchFailure(blockId: String, exception: Throwable): Unit = queue.put(Left(exception))
    }, null.asInstanceOf[DownloadFileManager])
    val out = new ArrayBuffer[(BlockId, ManagedBuffer)](ids.size)
    while (out.size < ids.size) {
      queue.take() match {
        case Right(b) =>
          metrics.incRemoteBlocksFetched(1)
          metrics.incRemoteBytesRead(b._2.size())
          out += b
        case Left(e) => throw new org.apache.spark.SparkException(s"columnar shuffle fetch from $address failed", e)
      }
    }
    metrics.incFetchWaitTime((System.nanoTime() - start) / 1000000)
    out.iterator
  }
}
