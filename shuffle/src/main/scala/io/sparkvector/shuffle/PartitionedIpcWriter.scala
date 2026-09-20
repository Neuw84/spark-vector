package io.sparkvector.shuffle

import java.io.ByteArrayOutputStream
import java.lang.foreign.{Arena, MemorySegment}
import java.nio.ByteBuffer
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{Files, Path, StandardOpenOption}
import scala.jdk.CollectionConverters._

import io.sparkvector.kernels.{Bitmap, CompactKernels, GatherKernels, PartitionKernels, VectorBuffers}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.arrow.{ArrowOutput, ArrowVectorBuffers, VectorArrowColumnVector, VectorDecimalColumnVector, VectorDictionaryColumnVector}
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.{FieldVector, IntVector, VarCharVector, VectorSchemaRoot}
import org.apache.arrow.vector.dictionary.{Dictionary, DictionaryProvider}
import org.apache.arrow.vector.compression.CompressionUtil
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.ipc.message.IpcOption
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.spark.sql.types.{BooleanType, StringType, StructType}
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
 * Record batches are sized on this side, not by the input: every partition holds one builder per
 * column (#351) -- an Arrow vector the input batches' rows for that partition are compacted into at
 * the current row offset, grown by doubling from a small start -- and when a partition reaches
 * `batchRows` rows or `batchBytes` bytes (or the task's held total passes `bufferBytes`, or
 * `finish`) its builders become the record batch as they are: one IPC message, one compression
 * call per buffer, one dictionary per batch. Without sizing, an input of 4096 rows over 200
 * partitions would give 20-row record batches, and the per-message costs (Arrow object churn,
 * metadata, compression) were 2x the row shuffle on TPC-H; with a set of vectors per (input batch,
 * partition) slice, concatenated at the flush, the write was 63% of a string-heavy query's CPU at
 * 200 partitions (#349, #351). Strings are plain in the builders whatever the input (a dictionary
 * column is decoded once per input batch) and dictionary-encoded once per record batch.
 *
 * Memory: the builders, at most about `bufferBytes` across partitions; the serialised bytes of a
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

  /**
   * One column of one partition's pending record batch (#351): an Arrow vector the input batches'
   * rows are compacted into at the current row offset, grown by doubling, and moved into the record
   * batch whole at the flush. Strings are plain here whatever the input (a dictionary column is
   * decoded once per input batch) and encoded once per record batch (#349). Before this every input
   * batch made a set of vectors per partition -- allocated, filled, appended into the batch and freed
   * for ~40 rows at 200 partitions, 3,600 vectors per batch for an 18-column table.
   */
  private final class Builder(val column: Int) {
    private val field = schema.fields(column)
    private val isString = field.dataType == StringType
    private val isBool = field.dataType == BooleanType
    private val width: Int = if (isString || isBool) 0 else PartitionedIpcWriter.byteWidth(field.dataType)
    var vector: FieldVector = _
    var buffers: ArrowVectorBuffers = _
    var rows: Int = 0
    var dataBytes: Long = 0L

    private def allocate(rowCapacity: Int, byteCapacity: Long): Unit = {
      buffers = if (isString) ArrowOutput.allocateUtf8(field.name, rowCapacity, math.max(byteCapacity, 1L), allocator)
        else ArrowOutput.allocateFixed(field.name, field.dataType, rowCapacity, allocator)
      vector = buffers.vector().asInstanceOf[FieldVector]
      if (isString) buffers.offsets().set(VectorBuffers.LE_INT, 0L, 0)
    }

    /** Room for `count` more rows and `bytes` more string bytes, doubling the vector as needed. */
    private def ensure(count: Int, bytes: Long): Unit = {
      if (vector == null) {
        val cap = math.max(PartitionedIpcWriter.InitialRows, Integer.highestOneBit(math.max(count, 1) - 1) << 1)
        allocate(math.min(math.max(cap, count), math.max(batchRows, count)), if (isString) math.max(bytes, PartitionedIpcWriter.InitialBytesPerRow.toLong * cap) else 0L)
        return
      }
      var grown = false
      while (vector.getValueCapacity < rows + count) { vector.reAlloc(); grown = true }
      if (isString) {
        val v = vector.asInstanceOf[VarCharVector]
        while (v.getDataBuffer.capacity() < dataBytes + bytes) { v.reallocDataBuffer(); grown = true }
      }
      if (grown) buffers = ArrowVectorBuffers.forWrite(vector, vector.getValueCapacity, field.dataType)
    }

    /** Appends the rows of `in` selected by `mask` (`count` of them). */
    def append(in: VectorBuffers, mask: MemorySegment, count: Int, scratch: Arena): Long = {
      val bytes = if (isString) CompactKernels.selectedUtf8Bytes(in, mask) else 0L
      ensure(count, bytes)
      val validityScratch = Bitmap.allocate(scratch, count)
      if (isString) {
        val offsets = buffers.offsets().asSlice(rows.toLong << 2)
        CompactKernels.compactUtf8(in, mask, count, offsets, buffers.data().asSlice(dataBytes), if (in.validity() != null) validityScratch else null)
        if (dataBytes > 0) {
          // The kernel's offsets start at zero: rebase them on the bytes already there (count + 1 of them).
          var i = 0
          while (i <= count) { offsets.set(VectorBuffers.LE_INT, i.toLong << 2, offsets.get(VectorBuffers.LE_INT, i.toLong << 2) + dataBytes.toInt); i += 1 }
        }
      } else if (isBool) {
        val bits = Bitmap.allocate(scratch, count)
        CompactKernels.compactFixed(in, mask, count, bits, if (in.validity() != null) validityScratch else null)
        Bitmap.copyBits(bits, buffers.data(), rows, count)
      } else {
        CompactKernels.compactFixed(in, mask, count, buffers.data().asSlice(rows.toLong * width), if (in.validity() != null) validityScratch else null)
      }
      if (in.validity() != null) Bitmap.copyBits(validityScratch, buffers.validity(), rows, count)
      else Bitmap.fillRange(buffers.validity(), rows, count, true)
      rows += count
      dataBytes += bytes
      if (isString) bytes + (count.toLong << 2) else count.toLong * math.max(width, 1)
    }

    /** Appends rows `idx(from until to)` of `in` (the index-list path, #353). */
    def appendIndexed(in: VectorBuffers, idx: Array[Int], from: Int, to: Int, scratch: Arena): Long = {
      val count = to - from
      val bytes = if (isString) GatherKernels.gatherUtf8Bytes(in, idx, from, to) else 0L
      ensure(count, bytes)
      val validityScratch = if (in.validity() != null) Bitmap.allocate(scratch, count) else null
      if (isString) {
        val offsets = buffers.offsets().asSlice(rows.toLong << 2)
        GatherKernels.gatherUtf8(in, idx, from, to, offsets, buffers.data().asSlice(dataBytes), validityScratch)
        if (dataBytes > 0) {
          var i = 0
          while (i <= count) { offsets.set(VectorBuffers.LE_INT, i.toLong << 2, offsets.get(VectorBuffers.LE_INT, i.toLong << 2) + dataBytes.toInt); i += 1 }
        }
      } else if (isBool) {
        val bits = Bitmap.allocate(scratch, count)
        GatherKernels.gatherFixed(in, idx, from, to, bits, validityScratch)
        Bitmap.copyBits(bits, buffers.data(), rows, count)
      } else {
        GatherKernels.gatherFixed(in, idx, from, to, buffers.data().asSlice(rows.toLong * width), validityScratch)
      }
      if (validityScratch != null) Bitmap.copyBits(validityScratch, buffers.validity(), rows, count)
      else Bitmap.fillRange(buffers.validity(), rows, count, true)
      rows += count
      dataBytes += bytes
      if (isString) bytes + (count.toLong << 2) else count.toLong * math.max(width, 1)
    }

    /** The finished vector (value count set), the builder emptied for the next batch. */
    def take(): FieldVector = {
      val v = vector
      v match {
        case vw: VarCharVector => vw.setLastSet(rows - 1); vw.setValueCount(rows)
        case other => other.setValueCount(rows)
      }
      vector = null; buffers = null; rows = 0; dataBytes = 0L
      v
    }

    def close(): Unit = { if (vector != null) vector.close(); vector = null; buffers = null; rows = 0; dataBytes = 0L }
  }

  private final class Segment(val partition: Int) {
    val bytes = new ByteArrayOutputStream()
    val provider = new DictionaryProvider.MapDictionaryProvider()
    var root: VectorSchemaRoot = _
    var writer: ArrowStreamWriter = _
    var rows: Long = 0L
    val builders: Array[Builder] = Array.tabulate(schema.fields.length)(new Builder(_))
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
      builders.foreach(b => try b.close() catch { case _: Exception => })
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
    // A dictionary-encoded string column is decoded once here and appended plain: the batch's strings
    // are dictionary-encoded again at the flush, once per record batch (#349, #351).
    val plain = buffers.map(b => if (b.`type`() == io.sparkvector.kernels.VecType.UTF8 && b.isDictionaryEncoded()) ArrowOutput.decodeDictionary(b, scratch) else b)
    if (numPartitions > PartitionedIpcWriter.IndexListPartitions) {
      // Many partitions: the rows grouped by partition once, a gather per partition of its own rows
      // (#353). A mask per partition cost a scan of the batch's words per partition and column.
      if (order.length < n) order = new Array[Int](n)
      PartitionKernels.partitionOrder(ids, n, numPartitions, starts, order)
      var p = 0
      while (p < numPartitions) {
        if (starts(p + 1) > starts(p)) appendIndexed(segments(p), plain, order, starts(p), starts(p + 1), scratch)
        p += 1
      }
    } else {
      val masks = Array.tabulate(numPartitions)(_ => scratch.allocate(Bitmap.bytesFor(n), 8))
      val counts = new Array[Int](numPartitions)
      PartitionKernels.partitionMasks(ids, n, masks, counts)
      var p = 0
      while (p < numPartitions) {
        if (counts(p) > 0) appendPartition(segments(p), plain, masks(p), counts(p), scratch)
        p += 1
      }
    }
  }

  private val starts = new Array[Int](numPartitions + 1)
  private var order = new Array[Int](0)

  private def appendIndexed(seg: Segment, buffers: Array[VectorBuffers], idx: Array[Int], from: Int, to: Int, scratch: Arena): Unit = {
    var size = 0L
    var c = 0
    while (c < buffers.length) {
      size += seg.builders(c).appendIndexed(buffers(c), idx, from, to, scratch)
      c += 1
    }
    afterAppend(seg, to - from, size)
  }

  private def appendPartition(seg: Segment, buffers: Array[VectorBuffers], mask: MemorySegment, count: Int, scratch: Arena): Unit = {
    var size = 0L
    var c = 0
    while (c < buffers.length) {
      size += seg.builders(c).append(buffers(c), mask, count, scratch)
      c += 1
    }
    afterAppend(seg, count, size)
  }

  private def afterAppend(seg: Segment, count: Int, size: Long): Unit = {
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
  private def flush(seg: Segment): Unit = if (seg.pendingRows > 0) {
    val rows = seg.pendingRows
    val taken = new Array[FieldVector](schema.fields.length)
    try {
      var c = 0
      while (c < schema.fields.length) { taken(c) = seg.builders(c).take(); c += 1 }
      // Pass 1: the record batch's dictionaries into the provider -- all of them before the stream
      // writer exists, since it converts the schema with the dictionaries' types at construction.
      c = 0
      while (c < schema.fields.length) {
        val encoding = arrowSchema.getFields.get(c).getDictionary
        if (encoding != null) {
          val (ids, dictionary) = PartitionedIpcWriter.encodeStrings(taken(c).asInstanceOf[VarCharVector], schema.fields(c).name, allocator)
          taken(c).close()
          taken(c) = ids
          batchDictionaries += dictionary
          seg.provider.put(new Dictionary(dictionary, encoding))
        }
        c += 1
      }
      seg.start()
      // Pass 2: the columns move into the root (no copy).
      c = 0
      while (c < schema.fields.length) {
        taken(c).makeTransferPair(seg.root.getVector(c)).transfer()
        taken(c).close(); taken(c) = null
        c += 1
      }
      seg.root.setRowCount(rows)
      seg.writer.writeBatch()
      seg.endStream()
      seg.rows += rows
      rawBytesWritten += seg.pendingBytes
      seg.spillIfNeeded()
    } finally {
      taken.foreach(v => if (v != null) v.close())
      // The stream writer kept its own copy of what it sent.
      batchDictionaries.foreach(_.close()); batchDictionaries.clear()
      heldBytes -= seg.pendingBytes
      seg.pendingRows = 0; seg.pendingBytes = 0L
    }
  }

  private val batchDictionaries = scala.collection.mutable.ArrayBuffer.empty[VarCharVector]


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
  /** A builder's first capacity in rows, doubled as a partition fills (#351). */
  val InitialRows: Int = 256
  /** A string builder's first data capacity per row, in bytes. */
  val InitialBytesPerRow: Int = 16
  /** Above this many partitions the writer groups rows by index lists and gathers; below, it compacts by masks. */
  val IndexListPartitions: Int = 32

  /** The data width of a fixed-width lane as the writer lays it out (a small decimal is int64). */
  def byteWidth(dt: org.apache.spark.sql.types.DataType): Int = dt match {
    case org.apache.spark.sql.types.IntegerType | org.apache.spark.sql.types.DateType => 4
    case org.apache.spark.sql.types.LongType | org.apache.spark.sql.types.TimestampType | org.apache.spark.sql.types.DoubleType => 8
    case d: org.apache.spark.sql.types.DecimalType if d.precision <= 18 => 8
    case _: org.apache.spark.sql.types.DecimalType => 16
    case other => throw new IllegalArgumentException(s"unsupported shuffle column type $other")
  }

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
    // Sized to the input once (the distinct values are at most all of it): no reallocation per growth.
    dictionary.allocateNew(math.max(if (n == 0) 0L else in.getOffsetBuffer.getInt(n.toLong * 4).toLong, 1L), math.max(n, 1))
    val seen = new java.util.HashMap[java.nio.ByteBuffer, Integer](math.max(16, n * 2))
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
