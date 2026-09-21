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
import org.apache.arrow.vector.compression.CompressionUtil
import org.apache.arrow.vector.ipc.message.IpcOption
import org.apache.arrow.vector.types.pojo.{Field, Schema}
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
    bufferBytes: Long = 64L << 20,
    /**
     * A record batch's string column is dictionary-encoded only when its distinct values are at most
     * this share of its rows (#356): above it -- names, emails, addresses within 8192 rows -- the
     * hash per row buys nothing and the column goes plain. 0 never encodes, 1 always does.
     */
    dictionaryMaxRatio: Double = PartitionedIpcWriter.DefaultDictionaryMaxRatio) extends AutoCloseable {

  private val arrowSchema: Schema = PartitionedIpcFile.arrowSchema(schema)
  /** The plain-UTF8 variant of every string field, for the batches whose dictionary does not pay. */
  private val plainFields: Array[Field] = Array.tabulate(schema.fields.length) { c =>
    val f = schema.fields(c)
    if (f.dataType == StringType) PartitionedIpcFile.arrowField(f.name, f.dataType, c, dictionary = false) else null
  }

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
    var rows: Long = 0L
    val builders: Array[Builder] = Array.tabulate(schema.fields.length)(new Builder(_))
    var pendingRows: Int = 0
    var pendingBytes: Long = 0L
    var overflow: FileChannel = _
    var overflowPath: Path = _
    var overflowBytes: Long = 0L

    /**
     * One IPC stream per record batch (#340): its messages are written directly by `flush` (#411) --
     * the schema message serialised once per distinct batch schema and reused as bytes, the
     * dictionary and record batches through `MessageSerializer`, the end-of-stream marker as two
     * words -- over a root that wraps the builders' vectors. Until #411 every batch built an
     * `ArrowStreamWriter` (a FlatBuffers schema serialisation, a dictionary provider, a root
     * allocated and released): a fifth of the writer's time at 1000 partitions, where a map task
     * writes a thousand small batches over the same bytes it wrote two hundred at 200.
     */
    def sink: java.nio.channels.WritableByteChannel = {
      // Up to Spark's bypass-merge threshold of partitions the stream goes straight to its own file,
      // Arrow memory to the page cache with no heap in between; above it, a heap staging buffer up to
      // `flushBytes` keeps the file count down, as Spark's sort-based writer does.
      if (sinkChannel == null) {
        val raw = new PartitionedIpcWriter.NonClosing(
          if (numPartitions <= PartitionedIpcWriter.DirectFileMaxPartitions) { openOverflow(); overflow }
          else Channels.newChannel(bytes))
        // The partition's bytes are one compressed stream (#411), as Spark's shuffle compresses a
        // partition's segment: one frame per partition instead of one per buffer of every record
        // batch. With a thousand partitions a map task's batches are a few hundred rows, and per-buffer
        // compression was a compressor call and a frame header for every 1-2 KB buffer -- q35 at 1000
        // partitions shuffled 17x Spark's bytes at SF10 -- while a stream over the partition's
        // messages compresses their repeated metadata away and calls the compressor per block.
        compressor = ShuffleCompression.compressing(Channels.newOutputStream(raw), compression)
        sinkChannel = if (compressor == null) raw else Channels.newChannel(compressor)
      }
      sinkChannel
    }
    private var sinkChannel: java.nio.channels.WritableByteChannel = _
    private var compressor: java.io.OutputStream = _

    /** After a stream: the overflow file's high-water mark. */
    def endStream(): Unit =
      if (overflow != null && overflow.isOpen) overflowBytes = math.max(overflowBytes, overflow.position())

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

    /** Ends the partition's compressed stream (its frame end) and takes the overflow's high-water mark. */
    def end(): Unit = {
      if (compressor != null) { compressor.close(); compressor = null; sinkChannel = null }
      endStream()
    }

    def release(): Unit = {
      builders.foreach(b => try b.close() catch { case _: Exception => })
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
      // A string column whose distinct values are too many for the dictionary to pay stays plain
      // (#356), and the batch's schema says which is which: every batch is its own stream.
      var batchFields: java.util.List[Field] = null // built only when a column goes plain
      c = 0
      while (c < schema.fields.length) {
        val encoding = arrowSchema.getFields.get(c).getDictionary
        if (encoding != null) {
          val encoded = PartitionedIpcWriter.encodeStrings(taken(c).asInstanceOf[VarCharVector], schema.fields(c).name, allocator, dictionaryMaxRatio)
          if (encoded != null) {
            val (ids, dictionary) = encoded
            taken(c).close()
            taken(c) = ids
            batchDictionaries += ((dictionary, encoding.getId))
          } else {
            if (batchFields == null) batchFields = new java.util.ArrayList[Field](arrowSchema.getFields)
            batchFields.set(c, plainFields(c))
          }
        }
        c += 1
      }
      val batchSchema = if (batchFields == null) arrowSchema else new Schema(batchFields)
      val out = new org.apache.arrow.vector.ipc.WriteChannel(seg.sink)
      // No schema message and no end-of-stream marker (#411): the reader knows the shuffle's schema
      // from the dependency, and a string column is dictionary-encoded in a batch exactly when a
      // dictionary batch with its id precedes the record batch -- that is the batch's shape signal.
      // The dictionaries, one batch each, then the record batch, over roots that wrap the vectors.
      var d = 0
      while (d < batchDictionaries.length) {
        val (vector, id) = batchDictionaries(d)
        val droot = new VectorSchemaRoot(java.util.List.of(vector.getField), java.util.List.of[FieldVector](vector), vector.getValueCount)
        val dbatch = new org.apache.arrow.vector.ipc.message.ArrowDictionaryBatch(id, new org.apache.arrow.vector.VectorUnloader(droot, true, codec, true).getRecordBatch, false)
        try org.apache.arrow.vector.ipc.message.MessageSerializer.serialize(out, dbatch, IpcOption.DEFAULT) finally dbatch.close()
        d += 1
      }
      val root = new VectorSchemaRoot(batchSchema.getFields, java.util.Arrays.asList(taken: _*), rows)
      val batch = new org.apache.arrow.vector.VectorUnloader(root, true, codec, true).getRecordBatch
      try org.apache.arrow.vector.ipc.message.MessageSerializer.serialize(out, batch, IpcOption.DEFAULT) finally batch.close()
      seg.endStream()
      seg.rows += rows
      rawBytesWritten += seg.pendingBytes
      seg.spillIfNeeded()
    } finally {
      taken.foreach(v => if (v != null) v.close())
      batchDictionaries.foreach(_._1.close()); batchDictionaries.clear()
      heldBytes -= seg.pendingBytes
      seg.pendingRows = 0; seg.pendingBytes = 0L
    }
  }

  private val batchDictionaries = scala.collection.mutable.ArrayBuffer.empty[(VarCharVector, Long)]

  /** No per-buffer body compression: the partition's stream is compressed as a whole (see `Segment.sink`). */
  private val codec: org.apache.arrow.vector.compression.CompressionCodec = org.apache.arrow.vector.compression.NoCompressionCodec.INSTANCE



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
    case org.apache.spark.sql.types.IntegerType | org.apache.spark.sql.types.DateType | org.apache.spark.sql.types.ByteType | org.apache.spark.sql.types.ShortType => 4 // narrow ints ride INT32 lanes (#327)
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

  /** Default share of distinct values per rows above which a batch's string column goes plain (#356). */
  val DefaultDictionaryMaxRatio: Double = 0.5
  /** Rows hashed before the first distinct-ratio check: enough to tell a name column from a state column. */
  val DictionarySampleRows: Int = 512

  /**
   * Dictionary-encodes a plain string vector: the distinct values in first-seen order, int32 ids,
   * nulls kept. Returns `null` -- nothing allocated stays behind -- when the distinct values exceed
   * `maxRatio` of the rows seen, checked after [[DictionarySampleRows]] rows and at the end: the
   * caller then ships the column plain (#356).
   */
  def encodeStrings(in: VarCharVector, name: String, allocator: BufferAllocator,
      maxRatio: Double = DefaultDictionaryMaxRatio): (IntVector, VarCharVector) = {
    val n = in.getValueCount
    if (maxRatio <= 0.0) return null
    val ids = new IntVector(name, allocator)
    ids.allocateNew(n)
    val dictionary = new VarCharVector(name + ".dictionary", allocator)
    val totalBytes = if (n == 0) 0L else in.getOffsetBuffer.getInt(n.toLong * 4).toLong
    // Sized to the input once (the distinct values are at most all of it): no reallocation per growth.
    dictionary.allocateNew(math.max(totalBytes, 1L), math.max(n, 1))
    // The distinct values as an open-addressing table over the input's own bytes (#387): an entry is the
    // row where its value was first seen, hashed and compared in place through the Arrow buffers -- no
    // ByteBuffer per row, no boxing, no HashMap. A HashMap of ByteBuffers here was 11% of an executor's
    // time in q67 at 1 TB (encodeStrings 7%, ByteBuffer.hashCode 4%).
    val offsets = MemorySegment.ofBuffer(in.getOffsetBuffer.nioBuffer(0, (n + 1) * 4))
    val data = if (totalBytes == 0) MemorySegment.NULL else MemorySegment.ofBuffer(in.getDataBuffer.nioBuffer(0, totalBytes.toInt))
    val dataBuf = if (totalBytes == 0) null else in.getDataBuffer.nioBuffer(0, totalBytes.toInt)
    var capacity = 16
    while (capacity < n * 2) capacity <<= 1
    val mask = capacity - 1
    val table = new Array[Int](capacity) // the row of the entry's first occurrence + 1; 0 = empty
    val entryId = new Array[Int](n) // id by first-occurrence row
    var next = 0
    var i = 0
    while (i < n) {
      if (in.isNull(i)) {
        ids.setNull(i)
      } else {
        val start = offsets.get(VectorBuffers.LE_INT, i.toLong * 4)
        val end = offsets.get(VectorBuffers.LE_INT, (i.toLong + 1) * 4)
        var h = (if (end > start) io.sparkvector.kernels.HashKernels.hashBytes(data, start, end - start) else 0) & mask
        var id = -1
        while (id < 0) {
          val slot = table(h)
          if (slot == 0) {
            table(h) = i + 1
            entryId(i) = next
            id = next
            if (end > start) dictionary.setSafe(next, dataBuf, start, end - start) else dictionary.setSafe(next, Array.emptyByteArray)
            next += 1
          } else {
            val row = slot - 1
            val rs = offsets.get(VectorBuffers.LE_INT, row.toLong * 4)
            val re = offsets.get(VectorBuffers.LE_INT, (row.toLong + 1) * 4)
            if (re - rs == end - start && (end == start || MemorySegment.mismatch(data, rs, re, data, start, end) < 0)) id = entryId(row)
            else h = (h + 1) & mask
          }
        }
        ids.set(i, id)
      }
      i += 1
      if ((i == DictionarySampleRows || i == n) && next > (i * maxRatio) && maxRatio < 1.0) {
        ids.close(); dictionary.close()
        return null
      }
    }
    ids.setValueCount(n)
    dictionary.setValueCount(next)
    (ids, dictionary)
  }
}
