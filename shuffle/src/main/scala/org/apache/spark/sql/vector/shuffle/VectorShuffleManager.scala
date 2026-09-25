/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.vector.shuffle

import java.io.File
import java.nio.channels.Channels
import java.util.concurrent.LinkedBlockingQueue
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import io.sparkvector.kernels.PartitionKernels
import io.sparkvector.kernels.PartitionKernels.KeyKind
import io.sparkvector.kernels.VectorBuffers
import io.sparkvector.shuffle.{PartitionedIpcFile, PartitionedIpcWriter}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.arrow.VectorAllocators
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.{Partitioner, ShuffleDependency, SparkConf, SparkEnv, SparkException, TaskContext}
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
  final case class Range(partitioner: Partitioner, sortKeys: Seq[Expression], output: Seq[Attribute])
      extends VectorPartitioning {
    def numPartitions: Int = partitioner.numPartitions
  }

  def keyKind(dt: DataType): KeyKind = {
    import org.apache.spark.sql.types._
    dt match {
      case IntegerType | DateType | ByteType | ShortType => KeyKind.INT // Spark hashes a byte or short as its int value
      case LongType | TimestampType => KeyKind.LONG
      case d: DecimalType if d.precision <= io.sparkvector.spark.adapter.TypeMapping.MAX_DECIMAL_PRECISION =>
        KeyKind.LONG
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
    writeProcessor: ShuffleWriteProcessor,
    /** The exchange's `dataSize` metric -- AQE's runtime statistic for the stage; every map task adds its uncompressed Arrow bytes. */
    val dataSize: org.apache.spark.sql.execution.metric.SQLMetric,
    /**
     * Struct columns flattened into lanes ([[io.sparkvector.shuffle.StructFlattening]]): `schema` is
     * then the flat schema the IPC streams carry, and the reader rebuilds the written columns.
     */
    val layout: Option[io.sparkvector.shuffle.StructFlattening.Layout] = None,
    /**
     * Rebalance exchanges only (#20): every map task reports its record count per reduce partition,
     * so AQE can size the partitions by rows ([[RowProportionalSizes]]).
     */
    val recordsByPartition: Option[RecordsByPartitionAccumulator] = None
) extends ShuffleDependency[Int, ColumnarBatch, ColumnarBatch](
      rdd,
      partitioner,
      SparkEnv.get.serializer,
      None,
      None,
      false,
      writeProcessor
    )

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

  /**
   * The map task ids written for each of our shuffles, as `SortShuffleManager` keeps for its own:
   * `unregisterShuffle` removes their data and index files by these. Without it every map output
   * stayed on the executor's disk until the executor died -- the 20 GB node disks of the bench
   * cluster filled within minutes at 1 TB (#358).
   */
  private val taskIdMapsForShuffle =
    new java.util.concurrent.ConcurrentHashMap[Int, org.apache.spark.util.collection.OpenHashSet[Long]]()

  override def registerShuffle[K, V, C](shuffleId: Int, dependency: ShuffleDependency[K, V, C]): ShuffleHandle =
    dependency match {
      case v: VectorShuffleDependency => new VectorShuffleHandle(shuffleId, v)
      case other => sort.registerShuffle(shuffleId, other)
    }

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter
  ): ShuffleWriter[K, V] =
    handle match {
      case v: VectorShuffleHandle =>
        val mapTaskIds = taskIdMapsForShuffle.computeIfAbsent(
          v.shuffleId,
          _ => new org.apache.spark.util.collection.OpenHashSet[Long](16)
        )
        mapTaskIds.synchronized { mapTaskIds.add(mapId) }
        new VectorShuffleWriter(
          v,
          mapId,
          context,
          metrics,
          sort.shuffleBlockResolver.asInstanceOf[IndexShuffleBlockResolver]
        ).asInstanceOf[ShuffleWriter[K, V]]
      case other => sort.getWriter(other, mapId, context, metrics)
    }

  override def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter
  ): ShuffleReader[K, C] =
    handle match {
      case v: VectorShuffleHandle =>
        new VectorShuffleReader(
          v,
          startMapIndex,
          endMapIndex,
          startPartition,
          endPartition,
          context,
          metrics
        ).asInstanceOf[ShuffleReader[K, C]]
      case other => sort.getReader(other, startMapIndex, endMapIndex, startPartition, endPartition, context, metrics)
    }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    VectorShuffleBackend(conf).unregisterShuffle(shuffleId)
    Option(taskIdMapsForShuffle.remove(shuffleId)).foreach { mapTaskIds =>
      val resolver = sort.shuffleBlockResolver.asInstanceOf[IndexShuffleBlockResolver]
      mapTaskIds.iterator.foreach(mapTaskId => resolver.removeDataByMap(shuffleId, mapTaskId))
    }
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
    resolver: IndexShuffleBlockResolver
) extends ShuffleWriter[Int, ColumnarBatch] {

  private val dep = handle.dependency
  private val numPartitions = dep.partitioner.numPartitions
  private val blockManager = SparkEnv.get.blockManager
  private val dataFile: File = resolver.getDataFile(handle.shuffleId, mapId)
  private val tmp: File = Utils.tempFileWith(dataFile)
  // A limit as the backstop behind the writer's own flushing (#340): a runaway task fails on its own
  // instead of exhausting the executor's direct memory for every task on it.
  private val allocator: BufferAllocator = VectorAllocators.root().newChildAllocator(
    s"shuffle-write-${handle.shuffleId}-$mapId",
    0L,
    VectorShuffleWriter.memoryLimit(SparkEnv.get.conf)
  )
  private val writer = {
    val conf = SparkEnv.get.conf
    new PartitionedIpcWriter(
      dep.schema,
      numPartitions,
      allocator,
      tmp.toPath,
      VectorShuffleWriter.flushBytes(conf),
      VectorShuffleWriter.compression(conf),
      conf.getInt(VectorShuffleWriter.BatchRowsKey, 8192),
      conf.getSizeAsBytes(VectorShuffleWriter.BatchBytesKey, "1m"),
      conf.getSizeAsBytes(VectorShuffleWriter.BufferBytesKey, "64m"),
      conf.getDouble(VectorShuffleWriter.DictionaryMaxRatioKey, PartitionedIpcWriter.DefaultDictionaryMaxRatio),
      PartitionedIpcWriter.DictionaryCapBytes,
      // One dictionary per map file (#416) needs the reader to see the file: the Flight producer and the
      // local reader prepend it; Spark's block transfer delivers a block's bytes alone, so it keeps the
      // per-block dictionaries.
      fileDictionary = VectorShuffleBackend.backendName(conf).equalsIgnoreCase("flight")
    )
  }
  private var lengths: Array[Long] = _
  private var stopped = false
  private var rows = 0L
  private val recordCounts: Array[Long] =
    if (dep.recordsByPartition.isDefined) new Array[Long](numPartitions) else null
  private var roundRobinNext = VectorShuffleWriter.roundRobinStart(context, numPartitions)
  private lazy val rangeProjection: UnsafeProjection = dep.partitioning match {
    case r: VectorPartitioning.Range => UnsafeProjection.create(r.sortKeys, r.output)
    case _ => null
  }

  override def write(records: Iterator[Product2[Int, ColumnarBatch]]): Unit =
    try {
      // The write time is the writer's own work per batch -- adapting, partitioning, appending -- not
      // the upstream operators pulled through `records.next()`: timed around the whole loop it read as
      // the entire map stage (15 s against Spark's 81 ms on the same exchange at SF1, #247).
      var written = 0L
      while (records.hasNext) {
        val batch = records.next()._2
        val n = batch.numRows()
        if (n > 0) {
          val start = System.nanoTime()
          val arena = java.lang.foreign.Arena.ofConfined()
          try {
            val buffers = Array.tabulate(batch.numCols()) { c =>
              val cv = batch.column(c)
              // A struct has no lane of its own: StructFlattening writes its validity and leaves.
              if (cv.dataType().isInstanceOf[org.apache.spark.sql.types.StructType]) null
              else ColumnVectorAdapters.adapt(cv, n, arena)
            }
            // Trailing columns beyond the written ones hold materialised hash keys: partitioned on, not written.
            val flat = dep.layout match {
              case Some(l) =>
                val w = l.writtenWidth
                val written = l.flatten(Array.tabulate(w)(batch.column), buffers.take(w), n, arena)
                written ++ buffers.drop(w)
              case None => buffers
            }
            val ids = partitionIds(batch, flat, n)
            if (recordCounts != null) {
              var i = 0
              while (i < n) { recordCounts(ids(i)) += 1; i += 1 }
            }
            writer.write(
              if (flat.length > dep.schema.fields.length) flat.take(dep.schema.fields.length) else flat,
              n,
              ids,
              arena
            )
          } finally arena.close()
          rows += n
          written += System.nanoTime() - start
        }
      }
      metrics.incWriteTime(written)
    } catch {
      case e: org.apache.arrow.memory.OutOfMemoryException =>
        throw VectorShuffleWriter.serializable("write", allocator, e)
    }

  private def partitionIds(batch: ColumnarBatch, buffers: Array[VectorBuffers], n: Int): Array[Int] = {
    val ids = new Array[Int](n)
    dep.partitioning match {
      case VectorPartitioning.Hash(ordinals, kinds, num) =>
        val keys = ordinals.map(o => buffers(o))
        PartitionKernels.hashPartitionIds(keys, kinds, n, num, new Array[Int](n), ids)
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
        val start = System.nanoTime()
        val index = writer.finish(withFooter = false)
        val rawBytes = writer.rawBytes
        writer.close()
        lengths = index.lengths
        resolver.writeMetadataFileAndCommit(handle.shuffleId, mapId, lengths, Array.emptyLongArray, tmp)
        metrics.incWriteTime(
          System.nanoTime() - start
        ) // the last record batches and the file, as Spark counts its merge
        VectorShuffleBackend(SparkEnv.get.conf).mapOutputCommitted(handle.shuffleId, mapId, dataFile, lengths)
        metrics.incBytesWritten(lengths.sum)
        // Pre-compression size, as Spark's dataSize is; never below the file (IPC framing dominates tiny outputs).
        dep.dataSize.add(math.max(rawBytes, lengths.sum))
        metrics.incRecordsWritten(rows)
        dep.recordsByPartition.foreach(_.add((context.partitionId(), recordCounts)))
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

  /** The hard limit of one map task's writer allocator; the writer flushes long before it, this is the backstop. */
  val MemoryLimitKey = "spark.vector.shuffle.writer.memoryLimit"
  def memoryLimit(conf: SparkConf): Long = conf.getSizeAsBytes(MemoryLimitKey, "1g")

  /** A record batch's string column is dictionary-encoded only when distinct/rows is at most this (#356); 0 never, 1 always. */
  val DictionaryMaxRatioKey = "spark.vector.shuffle.writer.dictionaryMaxRatio"

  /**
   * Arrow's `OutOfMemoryException` is not `Serializable` (it carries an `Optional`); a task failing with
   * it cannot be reported and, on Spark 4.1 + JDK 24+, ends the executor instead (SPARK-55679, see
   * `upstream/`). The task fails with this serializable exception carrying the allocator's state.
   */
  def serializable(
      side: String,
      allocator: BufferAllocator,
      e: org.apache.arrow.memory.OutOfMemoryException
  ): SparkException =
    new SparkException(
      s"columnar shuffle $side ran out of Arrow memory: ${e.getMessage}; allocator ${allocator.getName} " +
        s"allocated ${allocator.getAllocatedMemory} peak ${allocator.getPeakMemoryAllocation} limit ${allocator.getLimit}" +
        (if (e.getCause != null) s"; cause: ${e.getCause}" else "")
    )

  /** A partition's held rows / bytes before they become one record batch, and the task-wide cap on held bytes. */
  val BatchRowsKey = "spark.vector.shuffle.batchRows"
  val BatchBytesKey = "spark.vector.shuffle.batchBytes"
  val BufferBytesKey = "spark.vector.shuffle.bufferBytes"
  val FlushBytesKey = "spark.vector.shuffle.flushBytes"
  def flushBytes(conf: SparkConf): Long = conf.getSizeAsBytes(FlushBytesKey, "1m")

  /**
   * `zstd` (default; zstd-jni, native), `lz4` (Arrow's codec is commons-compress pure Java -- an order
   * of magnitude slower, a TPC-H Q3 shuffle crawled under it), or `none`: body compression of the
   * shuffle's record batches.
   */
  val CompressionKey = "spark.vector.shuffle.compression"
  def compression(conf: SparkConf): Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType] =
    conf.get(CompressionKey, "zstd").trim.toLowerCase match {
      case "lz4" => Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.LZ4_FRAME)
      case "zstd" => Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD)
      case "none" | "" => None
      case other => throw new IllegalArgumentException(s"$CompressionKey: lz4, zstd or none, not '$other'")
    }

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
    metrics: ShuffleReadMetricsReporter
) extends ShuffleReader[Int, ColumnarBatch] {

  override def read(): Iterator[Product2[Int, ColumnarBatch]] = {
    val env = SparkEnv.get
    val blocksByAddress = env.mapOutputTracker
      .getMapSizesByExecutorId(handle.shuffleId, startMapIndex, endMapIndex, startPartition, endPartition)
      .toSeq
    val allocator = VectorAllocators.newChild(s"shuffle-read-${handle.shuffleId}-${context.partitionId()}")
    val open = new java.util.ArrayList[AutoCloseable]()
    context.addTaskCompletionListener[Unit] { _ =>
      open.asScala.foreach(c =>
        try c.close()
        catch { case _: Exception => }
      )
      allocator.close()
      // The task-level shuffle read metrics are the reader's to merge (Spark's BlockStoreShuffleReader
      // does it in its completion iterator); the executor only merges them on heartbeats.
      context.taskMetrics().mergeShuffleReadMetrics()
    }
    val nonEmpty = blocksByAddress.map { case (address, blocks) =>
      address -> blocks.collect {
        case (id: ShuffleBlockId, size, mapIndex) if size > 0 => (id, size, mapIndex)
      }.toIndexedSeq
    }
    val streams = VectorShuffleBackend(env.conf).read(
      nonEmpty,
      handle.dependency.schema,
      VectorShuffleWriter.compression(env.conf),
      allocator,
      metrics
    )
    streams.flatMap { reader =>
      open.add(reader)
      new Iterator[Product2[Int, ColumnarBatch]] {
        private var live = true
        override def hasNext: Boolean = {
          if (!live) return false
          val more =
            try reader.hasNext
            catch {
              case e: org.apache.arrow.memory.OutOfMemoryException =>
                throw VectorShuffleWriter.serializable("read", allocator, e)
            }
          if (!more) { reader.close(); open.remove(reader); live = false }
          more
        }
        override def next(): Product2[Int, ColumnarBatch] = {
          val b =
            try reader.next()
            catch {
              case e: org.apache.arrow.memory.OutOfMemoryException =>
                throw VectorShuffleWriter.serializable("read", allocator, e)
            }
          metrics.incRecordsRead(b.numRows())
          (0, handle.dependency.layout.fold(b)(_.unflatten(b)))
        }
      }
    }
  }
}

object VectorShuffleReader {

  /** A fetched or local block (one partition's IPC stream) as batches; the buffer is released with the stream. */
  def blockStream(
      buf: ManagedBuffer,
      allocator: BufferAllocator,
      schema: org.apache.spark.sql.types.StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType]
  ): Iterator[ColumnarBatch] with AutoCloseable =
    new Iterator[ColumnarBatch] with AutoCloseable {
      // A file segment is read with positional reads straight into Arrow memory; an InputStream
      // channel would copy every byte through a heap array first (the GC behind Q19/Q20's 1.15x). The
      // map file's dictionary section comes first (#416).
      private val channel = PartitionedIpcFile.blockChannel(buf)
      private val inner = new PartitionedIpcFile.StreamReader(channel, allocator, schema, compression)
      private var released = false
      override def hasNext: Boolean = inner.hasNext
      override def next(): ColumnarBatch = inner.next()
      override def close(): Unit = { inner.close(); if (!released) { buf.release(); released = true } }
    }

  /** Interim remote path over Spark's block transfer: all of one executor's blocks in one request, collected as they land. */
  def fetchRemote(
      address: BlockManagerId,
      ids: Seq[BlockId],
      metrics: ShuffleReadMetricsReporter
  ): Iterator[(BlockId, ManagedBuffer)] = {
    val client = SparkEnv.get.blockManager.blockStoreClient
    val queue = new LinkedBlockingQueue[Either[Throwable, (BlockId, ManagedBuffer)]]()
    val start = System.nanoTime()
    client.fetchBlocks(
      address.host,
      address.port,
      address.executorId,
      ids.map(_.toString).toArray,
      new BlockFetchingListener {
        override def onBlockFetchSuccess(blockId: String, data: ManagedBuffer): Unit = {
          data.retain()
          queue.put(Right((BlockId(blockId), data)))
        }
        override def onBlockFetchFailure(blockId: String, exception: Throwable): Unit = queue.put(Left(exception))
      },
      null.asInstanceOf[DownloadFileManager]
    )
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
