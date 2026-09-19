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
  def remoteBlocks(address: BlockManagerId, blocks: Seq[(ShuffleBlockId, Long)], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable]

  /**
   * The reducer's whole input. The default reads blocks this executor wrote from its own files and
   * hands every other executor's to [[remoteBlocks]]; a service-backed backend overrides it.
   */
  def read(blocksByAddress: Seq[(BlockManagerId, Seq[(ShuffleBlockId, Long)])], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable] = {
    val env = SparkEnv.get
    val local = env.blockManager.blockManagerId.executorId
    blocksByAddress.iterator.flatMap { case (address, blocks) =>
      if (blocks.isEmpty) Iterator.empty
      else if (address.executorId == local) blocks.iterator.map { case (id, _) =>
        metrics.incLocalBlocksFetched(1)
        val buf = env.blockManager.getLocalBlockData(id)
        metrics.incLocalBytesRead(buf.size())
        VectorShuffleReader.blockStream(buf, allocator)
      }
      else remoteBlocks(address, blocks, allocator, metrics)
    }
  }
}

object VectorShuffleBackend {
  val Key = "spark.vector.shuffle.backend"

  def backendName(conf: SparkConf): String = conf.get(Key, "flight").trim

  /** `flight`, `block`, or the class name of a backend with a no-argument constructor. */
  def apply(conf: SparkConf): VectorShuffleBackend = backendName(conf).toLowerCase match {
    case "flight" => FlightBackend
    case "block" => BlockTransferBackend
    case _ => Utils.classForName(backendName(conf)).getConstructor().newInstance().asInstanceOf[VectorShuffleBackend]
  }
}

/** Slice 3: one `DoGet` per remote block against the executor's Flight server. */
object FlightBackend extends VectorShuffleBackend {
  override def name: String = "flight"
  override def remoteBlocks(address: BlockManagerId, blocks: Seq[(ShuffleBlockId, Long)], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable] = {
    val conf = SparkEnv.get.conf
    val location = flight.FlightRegistry.locationOf(address.executorId)
    blocks.iterator.map { case (ShuffleBlockId(shuffleId, mapId, reduceId), _) =>
      metrics.incRemoteBlocksFetched(1)
      new flight.FlightBlockStream(location, shuffleId, mapId, reduceId, conf, allocator, metrics)
    }
  }
}

/** Slice 2: Spark's block transfer, all of one executor's blocks in one request. */
object BlockTransferBackend extends VectorShuffleBackend {
  override def name: String = "block"
  override def remoteBlocks(address: BlockManagerId, blocks: Seq[(ShuffleBlockId, Long)], allocator: BufferAllocator,
      metrics: ShuffleReadMetricsReporter): Iterator[Iterator[ColumnarBatch] with AutoCloseable] =
    VectorShuffleReader.fetchRemote(address, blocks.map(_._1), metrics).map { case (_, buf) => VectorShuffleReader.blockStream(buf, allocator) }
}
