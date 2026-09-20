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
import org.apache.arrow.vector.compression.CompressionUtil
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.ipc.message.IpcOption
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.arrow.vector.util.VectorAppender
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
 * Record batches are sized on this side, not by the input: a batch of the map task is compacted per
 * partition into slices, and a partition's slices are held until they reach `batchRows` rows or
 * `batchBytes` bytes (or the task's held total passes `bufferBytes`, or `finish`), then written as
 * one record batch -- one IPC message, one compression call per buffer, one dictionary per batch.
 * Without this an input of 4096 rows over 200 partitions would give 20-row record batches, and the
 * per-message costs (Arrow object churn, metadata, compression) were 2x the row shuffle on TPC-H.
 * Slices are concatenated with Arrow's `VectorAppender`; the string slices' dictionaries are
 * concatenated too and each slice's indices shifted by the dictionaries before it, so a record batch
 * carries one replacement dictionary (duplicates across slices are harmless to a reader).
 *
 * Memory: the held slices, at most `bufferBytes` across partitions; the serialised bytes of a
 * partition stay in memory up to `flushBytes` and overflow to a per-partition temporary file,
 * concatenated into the data file at `finish` (the shape of Spark's bypass-merge writer).
 */
final class PartitionedIpcWriter(
    schema: StructType,
    numPartitions: Int,
    allocator: BufferAllocator,
    path: Path,
    flushBytes: Long = 1L << 20,
    /** Body compression of every record batch (`None` = raw). Spark's own shuffle is compressed; raw IPC wrote 1.8x its bytes on TPC-H. */
    compression: Option[CompressionUtil.CodecType] = Some(CompressionUtil.CodecType.ZSTD),
    /** A partition's held rows before they become one record batch. */
    batchRows: Int = 8192,
    /** A partition's held bytes before they become one record batch. */
    batchBytes: Long = 1L << 20,
    /** Held bytes across all partitions before the fullest partitions are written out. */
    bufferBytes: Long = 64L << 20) extends AutoCloseable {

  private val arrowSchema: Schema = PartitionedIpcFile.arrowSchema(schema)

  /** One partition's rows out of one input batch, compacted; owned by the segment until written. */
  private final class Slice(val columns: Array[ColumnVector], val extra: Seq[FieldVector],
      val sources: Array[FieldVector], val dictionaries: Array[VarCharVector], val rows: Int, val bytes: Long) {
    def close(): Unit = { columns.foreach(cv => if (cv != null) cv.close()); extra.foreach(_.close()) }
  }

  private final class Segment(val partition: Int) {
    val bytes = new ByteArrayOutputStream()
    val provider = new DictionaryProvider.MapDictionaryProvider()
    var root: VectorSchemaRoot = _
    var writer: ArrowStreamWriter = _
    var rows: Long = 0L
    val pending = scala.collection.mutable.ArrayBuffer.empty[Slice]
    var pendingRows: Int = 0
    var pendingBytes: Long = 0L
    var overflow: FileChannel = _
    var overflowPath: Path = _
    var overflowBytes: Long = 0L

    /**
     * One IPC stream per record batch (#340): the stream writer keeps a copy of every dictionary it
     * wrote and the root keeps the batch's buffers, both until the writer closes -- across 200
     * partitions that was most of a map task's footprint. A stream per batch frees them with the
     * batch; the reader decodes concatenated streams already. Started by `flush`, ended right after.
     */
    def start(): Unit = {
      root = VectorSchemaRoot.create(arrowSchema, allocator)
      // Up to Spark's bypass-merge threshold of partitions the stream goes straight to its own file,
      // Arrow memory to the page cache with no heap in between; above it, a heap staging buffer up to
      // `flushBytes` keeps the file count down, as Spark's sort-based writer does.
      if (sink == null) sink = new PartitionedIpcWriter.NonClosing(
        if (numPartitions <= PartitionedIpcWriter.DirectFileMaxPartitions) { openOverflow(); overflow }
        else Channels.newChannel(bytes))
      writer = compression match {
        case Some(codec) => new ArrowStreamWriter(root, provider, sink, IpcOption.DEFAULT, io.sparkvector.shuffle.ShuffleCompression.Factory, codec)
        case None => new ArrowStreamWriter(root, provider, sink)
      }
      writer.start()
    }
    private var sink: java.nio.channels.WritableByteChannel = _

    /** Ends the batch's stream and frees the root and the writer's dictionary copies. */
    def endStream(): Unit = if (writer != null) {
      writer.end()
      writer.close()
      root.close()
      writer = null; root = null
      if (overflow != null && overflow.isOpen) overflowBytes = math.max(overflowBytes, overflow.position())
    }

    private def openOverflow(): Unit = if (overflow == null) {
      overflowPath = Files.createTempFile(path.getParent, path.getFileName.toString + ".p" + partition + ".", ".tmp")
      overflow = FileChannel.open(overflowPath, StandardOpenOption.WRITE)
    }

    def spillIfNeeded(): Unit = if (bytes.size() >= flushBytes) {
      openOverflow()
      val buf = ByteBuffer.wrap(bytes.toByteArray)
      while (buf.hasRemaining) overflow.write(buf)
      overflowBytes += buf.limit()
      bytes.reset()
    }

    def end(): Unit = endStream()

    def release(): Unit = {
      pending.foreach(sl => try sl.close() catch { case _: Exception => }); pending.clear()
      if (writer != null) { try writer.close() catch { case _: Exception => }; try root.close() catch { case _: Exception => } }
      if (overflow != null) { try overflow.close() catch { case _: Exception => }; try Files.deleteIfExists(overflowPath) catch { case _: Exception => } }
    }
  }

  private val segments = Array.tabulate(numPartitions)(new Segment(_))
  private var heldBytes = 0L
  private var rawBytesWritten = 0L

  /**
   * Uncompressed Arrow bytes of every record batch written so far -- the exchange's `dataSize`, which
   * AQE compares with the broadcast threshold: Spark's is its rows' pre-compression size, and feeding
   * the compressed file bytes made AQE broadcast sides three times the size it would for Spark.
   */
  def rawBytes: Long = rawBytesWritten

  /** Rows written so far, per partition. */
  def rowsPerPartition: Array[Long] = segments.map(_.rows)

  /** Splits `batch` by `ids` (one partition id per row) and appends each partition's rows to its stream. */
  def write(batch: ColumnarBatch, ids: Array[Int]): Unit = {
    val n = batch.numRows()
    if (n == 0) return
    val scratch = Arena.ofConfined()
    try {
      val buffers: Array[VectorBuffers] =
        Array.tabulate(batch.numCols())(c => ColumnVectorAdapters.adapt(batch.column(c), n, scratch))
      write(buffers, n, ids, scratch)
    } finally scratch.close()
  }

  /**
   * The same over columns already adapted into `scratch` -- the shuffle writer adapts a batch once
   * for the partition ids and the streams (adapting a Parquet dictionary column decodes it).
   */
  def write(buffers: Array[VectorBuffers], n: Int, ids: Array[Int], scratch: Arena): Unit = {
    if (n == 0) return
    val masks = Array.tabulate(numPartitions)(_ => scratch.allocate(Bitmap.bytesFor(n), 8))
    val counts = new Array[Int](numPartitions)
    PartitionKernels.partitionMasks(ids, n, masks, counts)
    var p = 0
    while (p < numPartitions) {
      if (counts(p) > 0) appendPartition(segments(p), buffers, masks(p), counts(p))
      p += 1
    }
  }

  private def appendPartition(seg: Segment, buffers: Array[VectorBuffers], mask: MemorySegment, count: Int): Unit = {
    val columns = new Array[ColumnVector](buffers.length)
    val sources = new Array[FieldVector](buffers.length)
    val dictionaries = new Array[VarCharVector](buffers.length)
    val extra = scala.collection.mutable.ArrayBuffer.empty[FieldVector]
    var ok = false
    try {
      var c = 0
      while (c < buffers.length) {
        val f = schema.fields(c)
        columns(c) = ArrowOutput.compact(f.name, f.dataType, buffers(c), mask, count, allocator)
        sources(c) = columns(c) match {
          case d: VectorDictionaryColumnVector =>
            dictionaries(c) = d.dictionary()
            d.indices()
          case d: VectorDecimalColumnVector => d.vector()
          case a: VectorArrowColumnVector if f.dataType == StringType =>
            // A plain string batch (the source was not dictionary encoded): encode it here so the
            // field stays one dictionary-encoded int32 whatever the batch; the dictionary is this
            // slice's distinct values.
            val (indices, dictionary) = PartitionedIpcWriter.encodeStrings(a.getValueVector.asInstanceOf[VarCharVector], f.name, allocator)
            extra += dictionary
            extra += indices
            dictionaries(c) = dictionary
            indices
          case a: VectorArrowColumnVector => a.getValueVector.asInstanceOf[FieldVector]
          case other => throw new IllegalStateException(s"unexpected compacted column ${other.getClass.getName}")
        }
        c += 1
      }
      ok = true
    } finally if (!ok) { columns.foreach(cv => if (cv != null) cv.close()); extra.foreach(_.close()) }
    var size = 0L
    var c = 0
    while (c < sources.length) {
      size += sources(c).getBufferSize
      if (dictionaries(c) != null) size += dictionaries(c).getBufferSize
      c += 1
    }
    seg.pending += new Slice(columns, extra.toSeq, sources, dictionaries, count, size)
    seg.pendingRows += count
    seg.pendingBytes += size
    heldBytes += size
    if (seg.pendingRows >= batchRows || seg.pendingBytes >= batchBytes) flush(seg)
    // The cap is on what the allocator really holds, not on the slices' used bytes: `setSafe`-grown
    // vectors carry doubled capacity and the per-slice string dictionaries their own, so the estimate
    // ran 10-20x under the truth (#340: a task's writer at 1.1 GB against a 64 MB `bufferBytes`).
    while (heldBytes > bufferBytes || allocator.getAllocatedMemory > bufferBytes) {
      var fullest: Segment = null
      var p = 0
      while (p < numPartitions) {
        val sg = segments(p)
        if (sg.pendingBytes > 0 && (fullest == null || sg.pendingBytes > fullest.pendingBytes)) fullest = sg
        p += 1
      }
      if (fullest == null) { heldBytes = 0L; return } else flush(fullest)
    }
  }

  /** What the writer's allocator holds right now (pending slices; the roots are emptied after each batch). */
  def allocatedBytes: Long = allocator.getAllocatedMemory

  /** The partition's held slices become one record batch of its stream. */
  private def flush(seg: Segment): Unit = if (seg.pending.nonEmpty) {
    val slices = seg.pending
    val single = slices.size == 1
    try {
      // Pass 1: the record batch's dictionaries into the provider -- all of them before the stream
      // writer exists, since it converts the schema with the dictionaries' types at construction.
      var c = 0
      while (c < schema.fields.length) {
        val encoding = arrowSchema.getFields.get(c).getDictionary
        if (encoding != null) {
          if (single) {
            seg.provider.put(new Dictionary(slices.head.dictionaries(c), encoding))
          } else {
            // One dictionary for the record batch: the slices' dictionaries back to back, each slice's
            // indices shifted by the entries before its own.
            val merged = new VarCharVector(schema.fields(c).name + ".dictionary", allocator)
            merged.allocateNew() // the appender reads the target's buffers before growing them
            mergedDictionaries += merged
            val dictAppender = new VectorAppender(merged)
            var offset = 0
            slices.foreach { sl =>
              val ids = sl.sources(c).asInstanceOf[IntVector]
              if (offset > 0) {
                var i = 0
                while (i < ids.getValueCount) { if (!ids.isNull(i)) ids.set(i, ids.get(i) + offset); i += 1 }
              }
              sl.dictionaries(c).accept(dictAppender, null)
              offset += sl.dictionaries(c).getValueCount
            }
            seg.provider.put(new Dictionary(merged, encoding))
          }
        }
        c += 1
      }
      seg.start()
      // Pass 2: the columns into the root -- moved when there is one slice, appended otherwise.
      c = 0
      while (c < schema.fields.length) {
        val target = seg.root.getVector(c)
        if (single) {
          // Buffers move into the root's vector (no copy); the compacted wrapper is left empty and closed below.
          slices.head.sources(c).makeTransferPair(target).transfer()
        } else {
          target.clear()
          target.allocateNew()
          val appender = new VectorAppender(target)
          slices.foreach(sl => sl.sources(c).accept(appender, null))
        }
        c += 1
      }
      seg.root.setRowCount(seg.pendingRows)
      seg.writer.writeBatch()
      seg.endStream()
      seg.rows += seg.pendingRows
      rawBytesWritten += seg.pendingBytes
      seg.spillIfNeeded()
    } finally {
      // Closes the slices' dictionaries too: the stream writer keeps its own copy of what it sent.
      slices.foreach(_.close())
      mergedDictionaries.foreach(_.close()); mergedDictionaries.clear()
      heldBytes -= seg.pendingBytes
      seg.pending.clear(); seg.pendingRows = 0; seg.pendingBytes = 0L
    }
  }

  private val mergedDictionaries = scala.collection.mutable.ArrayBuffer.empty[VarCharVector]

  /**
   * Ends every stream and writes the data file: the streams back to back and, when `withFooter`,
   * the index footer. Under Spark's shuffle the lengths go to the block resolver's index file
   * instead and the data file must be exactly the streams, so the writer there passes `false`.
   */
  def finish(withFooter: Boolean = true): PartitionedIpcFile.Index = {
    val out = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)
    try {
      val offsets = new Array[Long](numPartitions)
      val lengths = new Array[Long](numPartitions)
      var pos = 0L
      var p = 0
      while (p < numPartitions) {
        val seg = segments(p)
        flush(seg)
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
      if (withFooter) {
        val footer = ByteBuffer.wrap(PartitionedIpcFile.encodeIndex(index))
        while (footer.hasRemaining) out.write(footer)
      }
      out.force(false)
      index
    } finally out.close()
  }

  override def close(): Unit = segments.foreach(_.release())
}

object PartitionedIpcWriter {
  /** A channel the stream writer may close without closing the file or buffer behind it (one stream per batch). */
  final class NonClosing(inner: java.nio.channels.WritableByteChannel) extends java.nio.channels.WritableByteChannel {
    override def write(src: ByteBuffer): Int = inner.write(src)
    override def isOpen: Boolean = inner.isOpen
    override def close(): Unit = ()
  }

  /** Partition count up to which each stream is written straight to its own file (Spark's `spark.shuffle.sort.bypassMergeThreshold`). */
  val DirectFileMaxPartitions = 200

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
