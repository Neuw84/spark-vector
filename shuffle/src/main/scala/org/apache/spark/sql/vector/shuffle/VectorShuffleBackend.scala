package org.apache.spark.sql.vector.shuffle

import java.io.File

import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.{SparkConf, SparkEnv}
import org.apache.spark.shuffle.ShuffleReadMetricsReporter
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{BlockManagerId, ShuffleBlockId}
import org.apache.spark.util.Utils

/**
 * Where a committed map output goes and where a reducer's blocks come from. The map side is fixed
 * -- every backend gets the slice-1 partitioned IPC file, committed through Spark's resolver so
 * `MapStatus` and the index file work -- and the backends differ in how a reducer reaches a block
 * that another executor wrote:
 *
 *  - [[FlightBackend]] (default): a `DoGet` per block against the executor's Flight server; for
 *    executors that stay up for the job, the direct path with no intermediate copy.
 *  - [[BlockTransferBackend]]: Spark's own block transfer against the same files.
 *
 * Future work, not in this issue: a push-based shuffle service such as Apache Celeborn, which is
 * what makes executors disposable (a lost executor loses no map output, no external shuffle
 * service, no `maps x reduces` small objects). Such a backend implements this trait from its own
 * jar and is named by class in `spark.vector.shuffle.backend`: [[mapOutputCommitted]] is where it
 * pushes each partition's IPC bytes, [[read]] is overridden whole because it reads a reduce
 * partition from the service rather than from executors, and [[io.sparkvector.shuffle.PartitionedIpcFile.StreamReader]]
 * already decodes the concatenation of several map outputs' streams, which is what an aggregated
 * partition file from such a service contains.
 */
trait VectorShuffleBackend {
  def name: String

  /** After the local data and index files are committed: a push-based service ships the partitions from here. */
  def mapOutputCommitted(shuffleId: Int, mapId: Long, dataFile: File, lengths: Array[Long]): Unit = ()

  /** A shuffle is being unregistered: drop whatever this backend keeps for it. */
  def unregisterShuffle(shuffleId: Int): Unit = ()

  /** The non-empty blocks another executor holds for this reducer, as batch streams. */
  def remoteBlocks(address: BlockManagerId, blocks: Seq[(ShuffleBlockId, Long)], schema: org.apache.spark.sql.types.StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable]

  /**
   * The reducer's whole input. The default reads blocks this executor wrote from its own files and
   * hands every other executor's to [[remoteBlocks]]; a service-backed backend overrides it. Each
   * block comes with its map index, so a failed remote fetch can be reported as Spark's
   * `FetchFailedException` (#364): the scheduler then unregisters the executor's map outputs and
   * recomputes them, where a plain task failure would retry against the same dead address and fail
   * the query.
   */
  def read(blocksByAddress: Seq[(BlockManagerId, Seq[(ShuffleBlockId, Long, Int)])], schema: org.apache.spark.sql.types.StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable] = {
    val env = SparkEnv.get
    val local = env.blockManager.blockManagerId.executorId
    // The remote streams are opened up front (one per executor and reducer, #347) so that the
    // executors send concurrently while the reducer decodes the first; gRPC's flow control bounds
    // what each stream buffers. The local blocks are read as they are reached.
    val remote = blocksByAddress.iterator.collect {
      case (address, blocks) if blocks.nonEmpty && address.executorId != local =>
        val (firstId, _, firstIndex) = blocks.head
        def fetchFailed(e: Throwable): Nothing =
          throw new org.apache.spark.shuffle.FetchFailedException(address, firstId.shuffleId, firstId.mapId, firstIndex, firstId.reduceId,
            s"fetch of ${blocks.length} block(s) from ${address.executorId} at ${address.host}:${address.port} failed: $e", e)
        val opened = try remoteBlocks(address, blocks.map { case (id, size, _) => (id, size) }, schema, compression, allocator, metrics).toIndexedSeq
          catch { case e: Exception if !VectorShuffleBackend.isMemory(e) => fetchFailed(e) }
        opened.map(s => VectorShuffleBackend.fetchFailing(s, fetchFailed))
    }.toIndexedSeq.flatten
    // The local map outputs: one range per map for the task's partitions (#411, as the Flight path
    // does) -- the resolver's batch block id is one index lookup and one file segment where a block
    // per partition was a lookup and an open each, times the partitions AQE coalesced into the task.
    val localBlocks = blocksByAddress.iterator.collect {
      case (address, blocks) if blocks.nonEmpty && address.executorId == local => blocks
    }.flatten.toIndexedSeq.groupBy(_._1.mapId).toIndexedSeq.sortBy(_._1).iterator.map { case (mapId, blocks) =>
      metrics.incLocalBlocksFetched(blocks.size)
      val first = blocks.head._1
      val start = blocks.iterator.map(_._1.reduceId).min
      val end = blocks.iterator.map(_._1.reduceId).max + 1
      val id: org.apache.spark.storage.BlockId =
        if (end == start + 1) first else org.apache.spark.storage.ShuffleBlockBatchId(first.shuffleId, mapId, start, end)
      val buf = env.blockManager.getLocalBlockData(id)
      metrics.incLocalBytesRead(buf.size())
      VectorShuffleReader.blockStream(buf, allocator, schema, compression)
    }
    localBlocks ++ remote.iterator
  }
}

object VectorShuffleBackend {
  val Key = "spark.vector.shuffle.backend"

  def backendName(conf: SparkConf): String = conf.get(Key, "flight").trim

  /** Arrow's allocator running out is the reducer's memory, not the remote executor: never a fetch failure. */
  def isMemory(e: Throwable): Boolean = e match {
    case _: org.apache.arrow.memory.OutOfMemoryException | _: OutOfMemoryError => true
    case _ => false
  }

  /** `inner` with every non-memory failure of `hasNext` / `next` turned into the fetch failure `fail` builds (#364). */
  def fetchFailing(inner: Iterator[ColumnarBatch] with AutoCloseable, fail: Throwable => Nothing): Iterator[ColumnarBatch] with AutoCloseable =
    new Iterator[ColumnarBatch] with AutoCloseable {
      override def hasNext: Boolean = try inner.hasNext catch { case e: Exception if !isMemory(e) => fail(e) }
      override def next(): ColumnarBatch = try inner.next() catch { case e: Exception if !isMemory(e) => fail(e) }
      override def close(): Unit = inner.close()
    }

  /** `flight`, `block`, or the class name of a backend with a no-argument constructor. */
  def apply(conf: SparkConf): VectorShuffleBackend = backendName(conf).toLowerCase match {
    case "flight" => FlightBackend
    case "block" => BlockTransferBackend
    case _ => Utils.classForName(backendName(conf)).getConstructor().newInstance().asInstanceOf[VectorShuffleBackend]
  }
}

/**
 * Slice 3: Flight against the executor's server -- one `DoGet` per (executor, reduce task) carrying all
 * of that executor's map outputs for the task's partition range (#347, #411). A reduce task's input is
 * a block per map task; a `DoGet` per block, opened one after another, was a round trip per map output,
 * and a `DoGet` per reduce partition multiplied the streams, index lookups and file opens by the number
 * of partitions AQE coalesced into the task.
 */
object FlightBackend extends VectorShuffleBackend {
  override def name: String = "flight"
  override def remoteBlocks(address: BlockManagerId, blocks: Seq[(ShuffleBlockId, Long)], schema: org.apache.spark.sql.types.StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable] = {
    val conf = SparkEnv.get.conf
    val location = flight.FlightRegistry.locationOf(address.executorId)
    if (blocks.isEmpty) return Iterator.empty
    // The task's partitions are consecutive; a partition with no block here is a zero-length span
    // of the range, so the range is the task's whole [min, max] and the maps are those with any block.
    val shuffleId = blocks.head._1.shuffleId
    val start = blocks.iterator.map(_._1.reduceId).min
    val end = blocks.iterator.map(_._1.reduceId).max + 1
    val mapIds = blocks.iterator.map(_._1.mapId).toIndexedSeq.distinct.sorted
    metrics.incRemoteBlocksFetched(blocks.size)
    Iterator.single(new flight.FlightBlockStream(location, shuffleId, mapIds, start, end, schema, compression, conf, allocator, metrics))
  }
}

/** Slice 2: Spark's block transfer, all of one executor's blocks in one request. */
object BlockTransferBackend extends VectorShuffleBackend {
  override def name: String = "block"
  override def remoteBlocks(address: BlockManagerId, blocks: Seq[(ShuffleBlockId, Long)], schema: org.apache.spark.sql.types.StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable] =
    VectorShuffleReader.fetchRemote(address, blocks.map(_._1), metrics).map { case (_, buf) => VectorShuffleReader.blockStream(buf, allocator, schema, compression) }
}
