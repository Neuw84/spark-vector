package org.apache.spark.sql.vector

import java.io.File
import java.lang.foreign.Arena
import java.nio.channels.FileChannel
import java.nio.file.{Files, StandardOpenOption}

import io.sparkvector.kernels.{Bitmap, PartitionKernels, VecType, VectorBuffers}
import io.sparkvector.kernels.PartitionKernels.KeyKind
import io.sparkvector.spark.adapter.{ColumnVectorAdapters, TypeMapping}
import io.sparkvector.spark.arrow.{ArrowOutput, VectorArrowColumnVector, VectorDecimalColumnVector}
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.{FieldVector, VectorSchemaRoot}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.spark.SparkEnv
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}

/**
 * The disk side of a Final aggregate that outgrows its budget (#363): the group table, emitted as
 * the same key + buffer-slot batches the operator produces, hash-partitioned on the grouping keys
 * into `numBuckets` Arrow IPC streams in the task's local directory. Once the input is consumed,
 * each bucket is read back and merged on its own through the operator's buffer-merge path -- the
 * one its exchange input takes -- so memory is bounded by the largest bucket, not the partition.
 * Grace hash aggregation.
 *
 * `columns` are the batches' (name, Spark type) in the operator's layout; `keyOrdinals` the
 * positions of the grouping keys among them. The buckets are Spark's hash of the keys under `seed`,
 * which must not be the shuffle's (#416): the rows of one reduce task all share their Spark hash
 * modulo the partition count, so bucketing by that same hash fills `buckets / gcd(partitions,
 * buckets)` of the buckets -- two of sixteen at 200 or 1000 partitions, each eight times the size
 * the bucket count was meant to bound.
 */
final class AggregateSpill(
    numBuckets: Int,
    columns: Array[(String, DataType)],
    keyOrdinals: Array[Int],
    allocator: BufferAllocator,
    seed: Int = AggregateSpill.BucketSeed,
    /** The key types when the keys are not columns of the batch but values given to `writeBuffers` (the join, #416). */
    keyTypes: Array[DataType] = null) extends AutoCloseable {

  private val kinds: Array[KeyKind] =
    (if (keyTypes != null) keyTypes else keyOrdinals.map(o => columns(o)._2)).map(AggregateSpill.keyKind)
  private val files = new Array[File](numBuckets)
  private val channels = new Array[FileChannel](numBuckets)
  private val roots = new Array[VectorSchemaRoot](numBuckets)
  private val writers = new Array[ArrowStreamWriter](numBuckets)
  private var closed = false

  var spilledBatches: Long = 0L
  var spilledRows: Long = 0L

  /** Writes `batch` (key then buffer columns, the operator's layout) into its buckets. */
  def write(batch: ColumnarBatch): Unit = {
    val n = batch.numRows()
    if (n == 0) return
    val arena = Arena.ofConfined()
    try {
      val buffers = Array.tabulate(batch.numCols()) { c =>
        val b = ColumnVectorAdapters.adapt(batch.column(c), n, arena)
        // The buckets are written plain: a dictionary-encoded key column is decoded once here.
        if (b.`type`() == VecType.UTF8 && b.isDictionaryEncoded()) ArrowOutput.decodeDictionary(b, arena) else b
      }
      writeBuffers(buffers, keyOrdinals.map(buffers(_)), n, arena)
    } finally arena.close()
  }

  /**
   * Writes `n` rows given as column buffers into their buckets, bucketed by `keys` -- values hashed
   * like the shuffle's partitioning under this spill's seed; for the aggregate they are columns of
   * the batch, for the join (#416) the evaluated key expressions. `arena` holds the scratch (masks).
   * A dictionary-encoded string column among `buffers` must already be decoded (the buckets are plain).
   */
  def writeBuffers(buffers: Array[VectorBuffers], keys: Array[VectorBuffers], n: Int, arena: Arena): Unit = {
    if (n == 0) return
    {
      val ids = new Array[Int](n)
      PartitionKernels.hashPartitionIds(keys, kinds, n, numBuckets, seed, new Array[Int](n), ids)
      // One selection mask per bucket, from one pass over the ids (a fresh segment is all clear).
      val masks = Array.fill(numBuckets)(arena.allocate(Bitmap.bytesFor(n), 8))
      val counts = new Array[Int](numBuckets)
      var i = 0
      while (i < n) { Bitmap.set(masks(ids(i)), i); counts(ids(i)) += 1; i += 1 }
      var b = 0
      while (b < numBuckets) {
        if (counts(b) > 0) {
          open(b)
          val root = roots(b)
          var c = 0
          while (c < columns.length) {
            val (name, dt) = columns(c)
            val col = ArrowOutput.compact(name, dt, buffers(c), masks(b), n, allocator)
            // The compacted vector moves into the bucket's root (the root's previous buffers are released by the transfer).
            try AggregateSpill.vectorOf(col).makeTransferPair(root.getVector(c)).transfer() finally col.close()
            c += 1
          }
          root.setRowCount(counts(b))
          writers(b).writeBatch()
          spilledBatches += 1
          spilledRows += counts(b)
        }
        b += 1
      }
    }
  }

  private def open(b: Int): Unit = if (writers(b) == null) {
    files(b) = AggregateSpill.newFile()
    channels(b) = FileChannel.open(files(b).toPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)
    roots(b) = VectorSchemaRoot.of(columns.map { case (name, dt) => ArrowOutput.newVector(name, dt, allocator) }: _*)
    writers(b) = new ArrowStreamWriter(roots(b), null, channels(b))
    writers(b).start()
  }

  private def endWriting(b: Int): Unit = if (writers(b) != null) {
    writers(b).end(); writers(b).close(); writers(b) = null
    roots(b).close(); roots(b) = null
    channels(b).close(); channels(b) = null
  }

  /** Bucket `b`'s batches, as the operator's column vectors; empty when nothing was spilled into it. */
  def read(b: Int): Iterator[ColumnarBatch] with AutoCloseable = {
    endWriting(b)
    if (files(b) == null) AggregateSpill.empty
    else new Iterator[ColumnarBatch] with AutoCloseable {
      private val in = FileChannel.open(files(b).toPath, StandardOpenOption.READ)
      private val reader = new ArrowStreamReader(in, allocator)
      private var nextBatch: ColumnarBatch = _
      private var last: ColumnarBatch = _
      private var done = false

      private def advance(): Unit = if (!done && nextBatch == null) {
        if (reader.loadNextBatch()) nextBatch = toBatch(reader.getVectorSchemaRoot) else done = true
      }
      private def toBatch(root: VectorSchemaRoot): ColumnarBatch = {
        val out = new Array[ColumnVector](columns.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = columns(c)
          val v = ArrowOutput.newVector(name, dt, allocator)
          root.getVector(c).makeTransferPair(v).transfer()
          out(c) = ArrowOutput.wrap(v, dt)
          c += 1
        }
        new ColumnarBatch(out, root.getRowCount)
      }
      override def hasNext: Boolean = { advance(); nextBatch != null }
      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException
        val b = nextBatch; nextBatch = null
        if (last != null) last.close()
        last = b
        b
      }
      override def close(): Unit = {
        if (nextBatch != null) { nextBatch.close(); nextBatch = null }
        if (last != null) { last.close(); last = null }
        reader.close()
        in.close()
      }
    }
  }

  override def close(): Unit = if (!closed) {
    closed = true
    var b = 0
    while (b < numBuckets) {
      try endWriting(b) catch { case _: Exception => }
      if (files(b) != null) { files(b).delete(); files(b) = null }
      b += 1
    }
  }
}

object AggregateSpill {
  /** The bucket hash's seed: any value but the shuffle's `PartitionKernels.SPARK_SEED` (42) -- see the class note. */
  val BucketSeed: Int = 0x5bd1e995

  /** The same key kinds as the shuffle's hash partitioning, so the buckets are Spark's hashes over the same lanes. */
  def keyKind(dt: DataType): KeyKind = dt match {
    case IntegerType | DateType => KeyKind.INT
    case LongType | TimestampType => KeyKind.LONG
    case d: DecimalType if d.precision <= TypeMapping.MAX_DECIMAL_PRECISION => KeyKind.LONG
    case _: DecimalType => KeyKind.DECIMAL128
    case DoubleType => KeyKind.DOUBLE
    case BooleanType => KeyKind.BOOL
    case StringType => KeyKind.UTF8
    case other => throw new IllegalArgumentException(s"no spill key kind for $other")
  }

  /** Whether every grouping key type can be bucketed. */
  def supportsKeys(types: Seq[DataType]): Boolean = types.forall {
    case IntegerType | DateType | LongType | TimestampType | DoubleType | BooleanType | StringType | _: DecimalType => true
    case _ => false
  }

  private[vector] def vectorOf(col: ColumnVector): FieldVector = col match {
    case v: VectorArrowColumnVector => v.getValueVector.asInstanceOf[FieldVector]
    case d: VectorDecimalColumnVector => d.vector()
    case other => throw new IllegalStateException(s"a spilled column must be a plain vector, not ${other.getClass.getSimpleName}")
  }

  /** A file in the task's local directory (Spark's disk block manager), or a plain temp file outside Spark. */
  private[vector] def newFile(): File = Option(SparkEnv.get) match {
    case Some(env) => env.blockManager.diskBlockManager.createTempLocalBlock()._2
    case None => Files.createTempFile("vector-agg-spill", ".arrow").toFile
  }

  private val empty: Iterator[ColumnarBatch] with AutoCloseable = new Iterator[ColumnarBatch] with AutoCloseable {
    override def hasNext: Boolean = false
    override def next(): ColumnarBatch = throw new NoSuchElementException
    override def close(): Unit = ()
  }
}

/** How a grouped aggregate behaves past its memory budget (#363). */
sealed trait AggSpillPolicy extends Serializable
object AggSpillPolicy {
  /** Everything stays in memory (the budget is off, or the modes cannot re-read their own output). */
  case object InMemory extends AggSpillPolicy
  /**
   * Buffer-emitting modes (Partial, PartialMerge): emit the table as output and start over -- the next stage
   * merges. `passThroughRatio` (#376): once a full table has reduced its input by less than this factor, the
   * rest of the input goes out one batch at a time -- a partial aggregate that does not reduce only costs
   * memory and copies; 0 keeps aggregating whatever the ratio.
   */
  final case class EmitAndReset(thresholdBytes: Long, passThroughRatio: Double = DefaultPassThroughRatio) extends AggSpillPolicy
  /** Result modes merging buffers (Final): spill the table into `buckets` and merge one bucket at a time. */
  final case class GraceHash(thresholdBytes: Long, buckets: Int) extends AggSpillPolicy

  val ThresholdKey = "spark.vector.agg.spillThreshold"
  val BucketsKey = "spark.vector.agg.spillBuckets"
  val PassThroughKey = "spark.vector.agg.passThroughRatio"
  val DefaultThreshold = "512m"
  val DefaultBuckets = 16
  val DefaultPassThroughRatio = 1.5
}
