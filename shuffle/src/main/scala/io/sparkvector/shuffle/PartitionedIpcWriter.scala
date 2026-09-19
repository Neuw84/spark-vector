package io.sparkvector.shuffle

import java.io.ByteArrayOutputStream
import java.lang.foreign.{Arena, MemorySegment}
import java.nio.ByteBuffer
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{Files, Path, StandardOpenOption}
import scala.jdk.CollectionConverters._

import io.sparkvector.kernels.{Bitmap, PartitionKernels, VectorBuffers}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.arrow.{ArrowOutput, VectorArrowColumnVector, VectorDecimalColumnVector, VectorDictionaryColumnVector}
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.{FieldVector, IntVector, VarCharVector, VectorSchemaRoot}
import org.apache.arrow.vector.dictionary.{Dictionary, DictionaryProvider}
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * The write side of the columnar shuffle (#288): every batch of the map task is split by its
 * partition ids into per-partition record batches, each appended to the partition's Arrow IPC
 * stream; `finish` lays the streams out in one file, one per reduce partition, followed by an index
 * of `(offset, length, rows)` per partition ([[PartitionedIpcFile]]). Nothing is decoded on the way:
 * a column is compacted straight from our buffers into the Arrow vector the IPC writer serialises
 * (`ArrowOutput.compact`), dictionary strings stay dictionary-encoded (an IPC dictionary per column,
 * re-sent only when the batch's dictionary differs from the last one written, which the stream
 * format allows as a replacement), and small decimals travel as their unscaled `int64` with the Spark
 * type in the field metadata, which the reader uses to rebuild the same Spark column vectors the
 * operators produce.
 *
 * Memory: a partition's rows are serialised the moment they are compacted, so the live Arrow memory
 * is one batch's worth of vectors, not a batch per partition; the serialised bytes of a partition
 * stay in memory up to `flushBytes` and overflow to a per-partition temporary file, concatenated
 * into the data file at `finish` (the shape of Spark's bypass-merge writer).
 */
final class PartitionedIpcWriter(
    schema: StructType,
    numPartitions: Int,
    allocator: BufferAllocator,
    path: Path,
    flushBytes: Long = 1L << 20) extends AutoCloseable {

  private val arrowSchema: Schema = PartitionedIpcFile.arrowSchema(schema)

  private final class Segment(val partition: Int) {
    val bytes = new ByteArrayOutputStream()
    val provider = new DictionaryProvider.MapDictionaryProvider()
    var root: VectorSchemaRoot = _
    var writer: ArrowStreamWriter = _
    var rows: Long = 0L
    var overflow: FileChannel = _
    var overflowPath: Path = _
    var overflowBytes: Long = 0L

    def start(): Unit = {
      root = VectorSchemaRoot.create(arrowSchema, allocator)
      writer = new ArrowStreamWriter(root, provider, Channels.newChannel(bytes))
      writer.start()
    }

    def spillIfNeeded(): Unit = if (bytes.size() >= flushBytes) {
      if (overflow == null) {
        overflowPath = Files.createTempFile(path.getParent, path.getFileName.toString + ".p" + partition + ".", ".tmp")
        overflow = FileChannel.open(overflowPath, StandardOpenOption.WRITE)
      }
      val buf = ByteBuffer.wrap(bytes.toByteArray)
      while (buf.hasRemaining) overflow.write(buf)
      overflowBytes += buf.limit()
      bytes.reset()
    }

    def end(): Unit = if (writer != null) {
      writer.end()
      writer.close()
      root.close()
      writer = null
    }

    def release(): Unit = {
      if (writer != null) { try writer.close() catch { case _: Exception => }; try root.close() catch { case _: Exception => } }
      if (overflow != null) { try overflow.close() catch { case _: Exception => }; try Files.deleteIfExists(overflowPath) catch { case _: Exception => } }
    }
  }

  private val segments = Array.tabulate(numPartitions)(new Segment(_))

  /** Rows written so far, per partition. */
  def rowsPerPartition: Array[Long] = segments.map(_.rows)

  /** Splits `batch` by `ids` (one partition id per row) and appends each partition's rows to its stream. */
  def write(batch: ColumnarBatch, ids: Array[Int]): Unit = {
    val n = batch.numRows()
    if (n == 0) return
    val scratch = Arena.ofConfined()
    try {
      val masks = Array.tabulate(numPartitions)(_ => scratch.allocate(Bitmap.bytesFor(n), 8))
      val counts = new Array[Int](numPartitions)
      PartitionKernels.partitionMasks(ids, n, masks, counts)
      val buffers: Array[VectorBuffers] =
        Array.tabulate(batch.numCols())(c => ColumnVectorAdapters.adapt(batch.column(c), n, scratch))
      var p = 0
      while (p < numPartitions) {
        if (counts(p) > 0) appendPartition(segments(p), buffers, masks(p), counts(p))
        p += 1
      }
    } finally scratch.close()
  }

  private def appendPartition(seg: Segment, buffers: Array[VectorBuffers], mask: MemorySegment, count: Int): Unit = {
    val columns = new Array[ColumnVector](buffers.length)
    val sources = new Array[FieldVector](buffers.length)
    val extra = scala.collection.mutable.ArrayBuffer.empty[FieldVector]
    try {
      var c = 0
      while (c < buffers.length) {
        val f = schema.fields(c)
        columns(c) = ArrowOutput.compact(f.name, f.dataType, buffers(c), mask, count, allocator)
        sources(c) = columns(c) match {
          case d: VectorDictionaryColumnVector =>
            val encoding = arrowSchema.getFields.get(c).getDictionary
            seg.provider.put(new Dictionary(d.dictionary(), encoding))
            d.indices()
          case d: VectorDecimalColumnVector => d.vector()
          case a: VectorArrowColumnVector if f.dataType == StringType =>
            // A plain string batch (the source was not dictionary encoded): encode it here so the
            // field stays one dictionary-encoded int32 whatever the batch; the dictionary is this
            // batch's distinct values.
            val (indices, dictionary) = PartitionedIpcWriter.encodeStrings(a.getValueVector.asInstanceOf[VarCharVector], f.name, allocator)
            extra += dictionary
            extra += indices
            seg.provider.put(new Dictionary(dictionary, arrowSchema.getFields.get(c).getDictionary))
            indices
          case a: VectorArrowColumnVector => a.getValueVector.asInstanceOf[FieldVector]
          case other => throw new IllegalStateException(s"unexpected compacted column ${other.getClass.getName}")
        }
        c += 1
      }
      // The stream writer converts the schema with the dictionaries' types at construction, so it is
      // created once the first batch's dictionaries are in the provider.
      if (seg.writer == null) seg.start()
      c = 0
      while (c < sources.length) {
        // Buffers move into the root's vector (no copy); the compacted wrapper is left empty and closed below.
        sources(c).makeTransferPair(seg.root.getVector(c)).transfer()
        c += 1
      }
      seg.root.setRowCount(count)
      seg.writer.writeBatch()
      seg.rows += count
      seg.spillIfNeeded()
    } finally {
      // Closes the dictionaries too: the stream writer keeps its own copy of what it sent.
      columns.foreach(cv => if (cv != null) cv.close())
      extra.foreach(_.close())
    }
  }

  /** Ends every stream, writes the data file with its index and footer, and returns the index. */
  def finish(): PartitionedIpcFile.Index = {
    val out = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)
    try {
      val offsets = new Array[Long](numPartitions)
      val lengths = new Array[Long](numPartitions)
      var pos = 0L
      var p = 0
      while (p < numPartitions) {
        val seg = segments(p)
        seg.end()
        offsets(p) = pos
        if (seg.overflow != null) {
          seg.overflow.close()
          val in = FileChannel.open(seg.overflowPath, StandardOpenOption.READ)
          try {
            var copied = 0L
            while (copied < seg.overflowBytes) copied += in.transferTo(copied, seg.overflowBytes - copied, out)
          } finally in.close()
          Files.deleteIfExists(seg.overflowPath)
          seg.overflow = null
          pos += seg.overflowBytes
        }
        val tail = ByteBuffer.wrap(seg.bytes.toByteArray)
        while (tail.hasRemaining) out.write(tail)
        pos += tail.limit()
        lengths(p) = pos - offsets(p)
        p += 1
      }
      val index = PartitionedIpcFile.Index(offsets, lengths, rowsPerPartition)
      val footer = ByteBuffer.wrap(PartitionedIpcFile.encodeIndex(index))
      while (footer.hasRemaining) out.write(footer)
      out.force(false)
      index
    } finally out.close()
  }

  override def close(): Unit = segments.foreach(_.release())
}

object PartitionedIpcWriter {
  /** Dictionary-encodes a plain string vector: the distinct values in first-seen order, int32 ids, nulls kept. */
  def encodeStrings(in: VarCharVector, name: String, allocator: BufferAllocator): (IntVector, VarCharVector) = {
    val n = in.getValueCount
    val ids = new IntVector(name, allocator)
    ids.allocateNew(n)
    val dictionary = new VarCharVector(name + ".dictionary", allocator)
    val seen = new java.util.HashMap[java.nio.ByteBuffer, Integer]()
    var next = 0
    var i = 0
    while (i < n) {
      if (in.isNull(i)) {
        ids.setNull(i)
      } else {
        val bytes = in.get(i)
        val key = java.nio.ByteBuffer.wrap(bytes)
        var id = seen.get(key)
        if (id == null) {
          id = next
          seen.put(key, id)
          dictionary.setSafe(next, bytes)
          next += 1
        }
        ids.set(i, id.intValue())
      }
      i += 1
    }
    ids.setValueCount(n)
    dictionary.setValueCount(next)
    (ids, dictionary)
  }
}
