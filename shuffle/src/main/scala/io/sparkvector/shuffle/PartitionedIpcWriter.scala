package io.sparkvector.shuffle

import java.io.ByteArrayOutputStream
import java.lang.foreign.{Arena, MemorySegment}
import java.nio.ByteBuffer
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{Files, Path, StandardOpenOption}
import scala.jdk.CollectionConverters._

import io.sparkvector.kernels.{Bitmap, CompactKernels, GatherKernels, PartitionKernels, VectorBuffers}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.arrow.{ArrowOutput, ArrowSegments, ArrowVectorBuffers, VectorArrowColumnVector, VectorDecimalColumnVector, VectorDictionaryColumnVector}
import org.apache.arrow.memory.ArrowBuf
import org.apache.arrow.vector.ipc.WriteChannel
import org.apache.arrow.vector.ipc.message.{ArrowFieldNode, ArrowRecordBatch, MessageSerializer}
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
 * Record batches are sized on this side, not by the input (#416, replacing #349/#351's builder set
 * per partition): the task's rows are staged once, whatever their partition -- one builder per
 * column plus the partition id of every row -- and when the staged data reaches `bufferBytes` (or
 * at `finish`) the rows are grouped by partition and each partition's rows gathered, in slices of
 * at most `batchRows`, into one reusable set of batch vectors that is serialised as the partition's
 * record batch: one IPC message per (partition, slice), one dictionary per batch. A builder set per
 * partition made a map task at 1000 partitions allocate a vector per column per partition -- 18,000
 * for 1.2 MB of output -- and gather every input batch into a thousand slices of eight rows; at 1 TB
 * the map stage carried the whole cost of the partition count. Spark's sort-based writer has this
 * shape: rows appended once, partitioned at the write. Strings are plain in the staging whatever the
 * input (a dictionary column is decoded once per input batch) and dictionary-encoded once per record
 * batch.
 *
 * Memory: the staging, about `bufferBytes` of data plus the spare capacity of its doubled vectors,
 * and one batch's vectors; the serialised bytes of a
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
    /** The most rows of one partition that become one record batch. */
    batchRows: Int = 8192,
    /** A partition's held bytes before they become one record batch (the per-partition path). */
    batchBytes: Long = 1L << 20,
    /** Staged data across all partitions before the staging is written out as record batches. */
    bufferBytes: Long = 64L << 20,
    /**
     * A record batch's string column is dictionary-encoded only when its distinct values are at most
     * this share of its rows (#356): above it -- names, emails, addresses within 8192 rows -- the
     * hash per row buys nothing and the column goes plain. 0 never encodes, 1 always does.
     */
    dictionaryMaxRatio: Double = PartitionedIpcWriter.DefaultDictionaryMaxRatio) extends AutoCloseable {

  private val arrowSchema: Schema = PartitionedIpcFile.arrowSchema(schema)
  /**
   * A builder's first capacity in rows (#416): an input batch's rows spread over the partitions, so
   * at 1000 partitions a partition sees ~8 rows per 8192-row batch and a 256-row first vector per
   * column was 74 KB per partition, 74 MB per map task, allocated and mostly never filled -- a
   * third of the map task's time at 1 TB / 1000 partitions was spent on those. The vectors still
   * double as rows arrive, so a partition that does fill pays only the reallocations it earns.
   */
  private val initialRows: Int = math.max(PartitionedIpcWriter.MinInitialRows,
    math.min(PartitionedIpcWriter.InitialRows, Integer.highestOneBit(math.max(1, 4 * batchRows / math.max(numPartitions, 1)))))
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
  private final class Builder(val column: Int, val shared: Boolean = false) {
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
      if (retained) { retainedBytes -= capacityBytes; retained = false }
      if (vector == null) {
        val cap = math.max(initialRows, Integer.highestOneBit(math.max(count, 1) - 1) << 1)
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

    /**
     * Appends all `n` rows of `in`: the staging path (#416), a bulk copy of each buffer -- no mask,
     * no per-element compaction -- since every row of the input is staged whatever its partition.
     */
    def appendAll(in: VectorBuffers, n: Int): Long = {
      val srcOff = in.offsets()
      val start = if (isString) srcOff.get(VectorBuffers.LE_INT, 0L) else 0
      val bytes = if (isString) (srcOff.get(VectorBuffers.LE_INT, n.toLong << 2) - start).toLong else 0L
      ensure(n, bytes)
      if (isString) {
        val offsets = buffers.offsets()
        val base = dataBytes.toInt - start
        var i = 0
        while (i <= n) { offsets.set(VectorBuffers.LE_INT, (rows + i).toLong << 2, srcOff.get(VectorBuffers.LE_INT, i.toLong << 2) + base); i += 1 }
        if (bytes > 0) MemorySegment.copy(in.data(), start.toLong, buffers.data(), dataBytes, bytes)
      } else if (isBool) {
        Bitmap.copyBits(in.data(), buffers.data(), rows, n)
      } else {
        MemorySegment.copy(in.data(), 0L, buffers.data(), rows.toLong * width, n.toLong * width)
      }
      if (in.validity() != null) Bitmap.copyBits(in.validity(), buffers.validity(), rows, n)
      else Bitmap.fillRange(buffers.validity(), rows, n, true)
      rows += n
      dataBytes += bytes
      if (isString) bytes + (n.toLong << 2) else n.toLong * math.max(width, 1)
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
      val v = finish()
      vector = null; buffers = null; rows = 0; dataBytes = 0L
      v
    }

    /** The staged rows as buffers to gather from (value count set); the builder keeps them. */
    def finished(): VectorBuffers = { finish(); buffers }

    /** The finished vector (value count set), still the builder's: `recycle` or `close` follows the flush. */
    def finish(): FieldVector = {
      val v = vector
      v match {
        case vw: VarCharVector => vw.setLastSet(rows - 1); vw.setValueCount(rows)
        case other => other.setValueCount(rows)
      }
      v
    }

    /** Bytes the vector's buffers hold, used or not. */
    def capacityBytes: Long = if (vector == null) 0L else { var t = 0L; val bs = vector.getBuffers(false); var i = 0; while (i < bs.length) { t += bs(i).capacity(); i += 1 }; t }

    /**
     * After a flush: the vector kept and emptied for the next rows (a reset keeps its capacity). A
     * shared builder (the staging, the batch vectors) always keeps; a partition's own keeps while
     * the writer's retained capacity stays under `retainBudget` (#417) -- the allocator cap in
     * `appendIndexed` measures the allocator's total, and unbounded retention across a few hundred
     * partitions tripped it into flushing on every append (record batches of a few rows, 1.5x the
     * bytes, 2x the time on q67 at 200 partitions).
     */
    def recycle(): Unit = {
      if (vector == null) return
      if (shared) { vector.reset(); rows = 0; dataBytes = 0L; return }
      val cap = capacityBytes
      if (retainedBytes + cap <= retainBudget) {
        vector.reset(); rows = 0; dataBytes = 0L
        retainedBytes += cap
        retained = true
      } else close()
    }
    private var retained = false

    def close(): Unit = {
      if (retained) { retainedBytes -= capacityBytes; retained = false }
      if (vector != null) vector.close()
      vector = null; buffers = null; rows = 0; dataBytes = 0L
    }
  }

  private final class Segment(val partition: Int) {
    val bytes = new ByteArrayOutputStream()
    var rows: Long = 0L
    /** The partition's own builders (the per-partition path, at most `StagingPartitions` partitions); null when staged. */
    val builders: Array[Builder] = if (staged) null else Array.tabulate(schema.fields.length)(new Builder(_))
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
    /**
     * The partition's IPC messages are staged raw on the heap and compressed in frames of at most
     * `flushBytes` (#411): one compressor call per frame through the writer's single reusable context,
     * against a compressing stream per partition -- a native zstd context created for each of a
     * thousand partitions per map task was 5% of an executor's time at 1000 partitions -- and the
     * frames of a partition, and of the partitions of a range, read back as one stream. Frames go to a
     * per-partition overflow file as they fill and the last one straight to the data file at `finish`.
     */
    def sink: java.nio.channels.WritableByteChannel = {
      if (sinkChannel == null) sinkChannel = new PartitionedIpcWriter.NonClosing(Channels.newChannel(bytes))
      sinkChannel
    }
    private var sinkChannel: java.nio.channels.WritableByteChannel = _
    /** The compressed frame of the last staged bytes, written to the data file at `finish`. */
    var tail: Array[Byte] = Array.emptyByteArray

    /** After a record batch: nothing to do until the staged bytes reach `flushBytes`. */
    def endStream(): Unit = ()

    private def openOverflow(): Unit = if (overflow == null) {
      overflowPath = Files.createTempFile(path.getParent, path.getFileName.toString + ".p" + partition + ".", ".tmp")
      overflow = FileChannel.open(overflowPath, StandardOpenOption.WRITE)
    }

    /** Staged bytes past `flushBytes`: one frame to the overflow file. */
    def spillIfNeeded(): Unit = if (bytes.size() >= flushBytes) {
      openOverflow()
      val frame = ByteBuffer.wrap(compressFrame(bytes.toByteArray))
      while (frame.hasRemaining) overflow.write(frame)
      overflowBytes += frame.limit()
      bytes.reset()
    }

    /** The remaining staged bytes as the partition's last frame. */
    def end(): Unit = {
      if (bytes.size() > 0) { tail = compressFrame(bytes.toByteArray); bytes.reset() }
    }

    def release(): Unit = {
      if (builders != null) builders.foreach(b => try b.close() catch { case _: Exception => })
      if (overflow != null) { try overflow.close() catch { case _: Exception => }; try Files.deleteIfExists(overflowPath) catch { case _: Exception => } }
    }
  }

  /**
   * Above `StagingPartitions` partitions the task's rows are staged and partitioned at the flush; at
   * or below, every partition keeps its own builders and each input batch is gathered into them
   * (#349, #351). The staging removes a builder set per partition -- 18,000 vector allocations per
   * map task at 1000 partitions -- but its flush gathers from a staging that no longer fits the
   * cache, which cost q67 8% of executor time at 200 partitions where the per-partition path's
   * gathers read an input batch still in L2. Each shape where it wins.
   */
  private val staged: Boolean = numPartitions > PartitionedIpcWriter.StagingPartitions
  private val segments = Array.tabulate(numPartitions)(new Segment(_))
  /**
   * The task's rows, staged once whatever their partition (#416): one builder per column plus the
   * partition id of every staged row. At the flush the staged rows are grouped by partition and
   * each partition's rows gathered into `batchBuilders` -- one reusable set of vectors -- and
   * serialised as its record batch. Before this every partition had its own builders: at 1000
   * partitions a map task allocated 18,000 vectors (one per column per partition) for 1.2 MB of
   * output and gathered every input batch into a thousand slices of eight rows; at 1 TB the map
   * stage carried the whole partition-count cost. Spark's sort-based writer has this shape.
   */
  private val staging: Array[Builder] = if (staged) Array.tabulate(schema.fields.length)(new Builder(_, shared = true)) else null
  private var stagedIds: Array[Int] = new Array[Int](0)
  private var stagedRows: Int = 0
  /** The vectors a large partition's rows are gathered into for one record batch, reused for every partition and flush. */
  private val batchBuilders: Array[Builder] = if (staged) Array.tabulate(schema.fields.length)(new Builder(_, shared = true)) else null
  /**
   * The staged rows in partition order, one gather per column per flush (#416), from which a small
   * partition's record batch is cut as slices. Gathering per partition per column -- a kernel call
   * for a handful of rows, 18 columns x 1000 partitions per flush -- paid the kernel's fixed cost, not
   * the rows': at 1 TB the rollup partial aggregate's tasks (55 k rows each, 4 strings, 7 DECIMAL128
   * sums, 7 counts) reported ~1 s of shuffle write time each, 20 us a row, against Spark's 7 ms.
   */
  private val orderedBuilders: Array[Builder] = if (staged) Array.tabulate(schema.fields.length)(new Builder(_, shared = true)) else null
  /** Staged data before a flush. A smaller staging (12.8 MB at 200 partitions) gave more, smaller record batches and cost q67 a further 8%. */
  private val stagingBytes: Long = bufferBytes
  private var heldBytes = 0L
  /** Capacity held by emptied per-partition builders kept for their next batch (#417), and its cap: well under `bufferBytes`. */
  private var retainedBytes = 0L
  private val retainBudget: Long = bufferBytes / 4
  /** Per string column, the ids and dictionary vectors of its encoding, reused across blocks (#416). */
  private val dictScratch = new Array[(IntVector, VarCharVector)](schema.fields.length)
  private def scratchFor(c: Int): (IntVector, VarCharVector) = {
    var sc = dictScratch(c)
    if (sc == null) { sc = (new IntVector(schema.fields(c).name, allocator), new VarCharVector(schema.fields(c).name + ".dictionary", allocator)); dictScratch(c) = sc }
    sc
  }
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
    if (!staged) {
      // The per-partition path: the rows grouped by partition once, a gather per partition of its own
      // rows into that partition's builders (#353).
      if (order.length < n) order = new Array[Int](n)
      PartitionKernels.partitionOrder(ids, n, numPartitions, starts, order)
      var p = 0
      while (p < numPartitions) {
        if (starts(p + 1) > starts(p)) appendIndexed(segments(p), plain, order, starts(p), starts(p + 1), scratch)
        p += 1
      }
      return
    }
    if (stagedIds.length < stagedRows + n) stagedIds = java.util.Arrays.copyOf(stagedIds, math.max(stagedRows + n, stagedIds.length * 2))
    System.arraycopy(ids, 0, stagedIds, stagedRows, n)
    var size = 0L
    var c = 0
    while (c < plain.length) {
      size += staging(c).appendAll(plain(c), n)
      c += 1
    }
    stagedRows += n
    heldBytes += size
    // The staging vectors keep their capacity across flushes (at most about twice the data they held,
    // doubling as they grow), so the allocator's total is bounded by a multiple of `bufferBytes`
    // rather than compared with it: the check is a backstop against a growth the estimate misses (#340).
    if (heldBytes > stagingBytes || allocator.getAllocatedMemory > 4 * bufferBytes) flushStaging()
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
    seg.pendingRows += to - from
    seg.pendingBytes += size
    heldBytes += size
    if (seg.pendingRows >= batchRows || seg.pendingBytes >= batchBytes) flushPartition(seg)
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
      if (fullest == null) { heldBytes = 0L; return } else flushPartition(fullest)
    }
  }

  /** The per-partition path's flush: the partition's builders become its record batch. */
  private def flushPartition(seg: Segment): Unit = if (seg.pendingRows > 0) {
    flush(seg, seg.pendingRows, seg.builders)
    heldBytes -= seg.pendingBytes
    seg.pendingRows = 0; seg.pendingBytes = 0L
  }

  /**
   * The staged rows out as record batches: grouped by partition (#353's order kernel), each
   * partition's rows gathered into the batch builders in slices of at most `batchRows` and
   * serialised into the partition's stream.
   */
  private def flushStaging(): Unit = if (stagedRows > 0) {
    val scratch = Arena.ofConfined()
    val n = schema.fields.length
    try {
      if (order.length < stagedRows) order = new Array[Int](stagedRows)
      PartitionKernels.partitionOrder(stagedIds, stagedRows, numPartitions, starts, order)
      val source: Array[VectorBuffers] = Array.tabulate(n)(c => staging(c).finished())
      // Every staged row into partition order, one gather per column (#416).
      var c = 0
      while (c < n) { orderedBuilders(c).appendIndexed(source(c), order, 0, stagedRows, scratch); c += 1 }
      val ordered: Array[FieldVector] = Array.tabulate(n)(c => orderedBuilders(c).finish())
      var p = 0
      while (p < numPartitions) {
        var from = starts(p)
        val end = starts(p + 1)
        if (end - from > 0 && end - from < PartitionedIpcWriter.DictionaryMinRows) {
          // A small partition (its strings stay plain, #426): its record batch is slices of the ordered vectors.
          flushSlice(segments(p), ordered, from, end)
        } else {
          while (from < end) {
            val to = math.min(end, from + batchRows)
            c = 0
            while (c < n) { batchBuilders(c).appendIndexed(source(c), order, from, to, scratch); c += 1 }
            flush(segments(p), to - from, batchBuilders)
            from = to
          }
        }
        p += 1
      }
    } finally {
      scratch.close()
      var c = 0
      while (c < n) { orderedBuilders(c).recycle(); c += 1 }
    }
    var c = 0
    while (c < n) { staging(c).recycle(); c += 1 }
    stagedRows = 0
    heldBytes = 0L
  }

  /** Per column, scratch for a small partition's validity bits (and a BOOL column's data bits, a string column's offsets). */
  private var sliceValidity: Array[ArrowBuf] = _
  private var sliceOffsets: Array[ArrowBuf] = _
  private var sliceBits: Array[ArrowBuf] = _
  private def sliceScratch(): Unit = if (sliceValidity == null) {
    val n = schema.fields.length
    val bitBytes = Bitmap.bytesFor(PartitionedIpcWriter.DictionaryMinRows) + 8 // whole words at the tail
    sliceValidity = Array.fill(n)(allocator.buffer(bitBytes))
    sliceOffsets = Array.tabulate(n)(c => if (schema.fields(c).dataType == StringType) allocator.buffer((PartitionedIpcWriter.DictionaryMinRows + 1).toLong * 4) else null)
    sliceBits = Array.tabulate(n)(c => if (schema.fields(c).dataType == BooleanType) allocator.buffer(bitBytes) else null)
  }

  /** Copies `count` bits from `src` at `from` to the start of `dst`, whole words at a time; returns how many are set. */
  private def copyBitRange(src: MemorySegment, srcBits: Int, from: Int, count: Int, dst: MemorySegment): Int = {
    val shift = from & 63
    var w = from >>> 6
    var k = 0
    var produced = 0
    var set = 0
    while (produced < count) {
      val lo = Bitmap.wordAt(src, w, srcBits)
      var word = if (shift == 0) lo else (lo >>> shift) | (Bitmap.wordAt(src, w + 1, srcBits) << (64 - shift))
      val take = math.min(64, count - produced)
      if (take < 64) word &= Bitmap.lowBits(take)
      Bitmap.setWord(dst, k, count, word)
      set += java.lang.Long.bitCount(word)
      produced += take
      k += 1
      w += 1
    }
    set
  }

  /**
   * A small partition's record batch straight from slices of the ordered vectors (#416): the data
   * buffers are zero-copy slices, the validity bits and string offsets a short copy into scratch,
   * and the batch is serialised as a record batch message of the all-plain shape -- no gather, no
   * vector and no root per partition.
   */
  private def flushSlice(seg: Segment, ordered: Array[FieldVector], from: Int, to: Int): Unit = {
    sliceScratch()
    val rows = to - from
    val n = schema.fields.length
    val nodes = new java.util.ArrayList[ArrowFieldNode](n)
    val buffers = new java.util.ArrayList[ArrowBuf](3 * n)
    var bytes = 0L
    var c = 0
    while (c < n) {
      val v = ordered(c)
      val dt = schema.fields(c).dataType
      val set = copyBitRange(ArrowSegments.of(v.getValidityBuffer), v.getValueCount, from, rows, ArrowSegments.of(sliceValidity(c)))
      val nullCount = rows - set
      nodes.add(new ArrowFieldNode(rows, nullCount))
      buffers.add(if (nullCount == 0) allocator.getEmpty else sliceValidity(c).slice(0, Bitmap.bytesFor(rows)))
      if (nullCount != 0) bytes += Bitmap.bytesFor(rows)
      dt match {
        case StringType =>
          val vc = v.asInstanceOf[VarCharVector]
          val off = vc.getOffsetBuffer
          val start = off.getInt(from.toLong * 4)
          val end = off.getInt(to.toLong * 4)
          val odst = sliceOffsets(c)
          var i = 0
          while (i <= rows) { odst.setInt(i.toLong * 4, off.getInt((from + i).toLong * 4) - start); i += 1 }
          buffers.add(odst.slice(0, (rows + 1).toLong * 4))
          buffers.add(vc.getDataBuffer.slice(start.toLong, (end - start).toLong))
          bytes += (rows + 1).toLong * 4 + (end - start)
        case BooleanType =>
          copyBitRange(ArrowSegments.of(v.getDataBuffer), v.getValueCount, from, rows, ArrowSegments.of(sliceBits(c)))
          buffers.add(sliceBits(c).slice(0, Bitmap.bytesFor(rows)))
          bytes += Bitmap.bytesFor(rows)
        case _ =>
          val w = PartitionedIpcWriter.byteWidth(dt)
          buffers.add(v.getDataBuffer.slice(from.toLong * w, rows.toLong * w))
          bytes += rows.toLong * w
      }
      c += 1
    }
    // No schema message, no dictionaries (the strings are plain): the record batch alone, as `flush` writes one.
    val batch = new ArrowRecordBatch(rows, nodes, buffers)
    try MessageSerializer.serialize(new WriteChannel(seg.sink), batch, IpcOption.DEFAULT) finally batch.close()
    seg.endStream()
    seg.rows += rows
    rawBytesWritten += bytes
    seg.spillIfNeeded()
  }

  /** What the writer's allocator holds right now (pending slices; the roots are emptied after each batch). */
  def allocatedBytes: Long = allocator.getAllocatedMemory

  /** The partition's held slices become one record batch of its stream. */
  private def flush(seg: Segment, rows: Int, builders: Array[Builder]): Unit = if (rows > 0) {
    val taken = new Array[FieldVector](schema.fields.length)
    try {
      var c = 0
      while (c < schema.fields.length) { taken(c) = builders(c).finish(); c += 1 }
      // Pass 1: the record batch's dictionaries into the provider -- all of them before the stream
      // writer exists, since it converts the schema with the dictionaries' types at construction.
      // A string column whose distinct values are too many for the dictionary to pay stays plain
      // (#356), and the batch's schema says which is which: every batch is its own stream.
      var batchFields: java.util.List[Field] = null // built only when a column goes plain
      c = 0
      while (c < schema.fields.length) {
        val encoding = arrowSchema.getFields.get(c).getDictionary
        if (encoding != null) {
          val (sids, sdict) = scratchFor(c)
          // A batch of a few rows keeps its strings plain (#416): its dictionary would cost more than it
          // saves, and only plain batches can be coalesced by the reader before the operators see them.
          val encoded = if (rows < PartitionedIpcWriter.DictionaryMinRows) null
            else PartitionedIpcWriter.encodeStrings(taken(c).asInstanceOf[VarCharVector], schema.fields(c).name, allocator, dictionaryMaxRatio, sids, sdict)
          if (encoded != null) {
            val (ids, dictionary) = encoded
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
      rawBytesWritten += batchBytesOf(taken)
      seg.spillIfNeeded()
    } finally {
      // The builders keep their vectors for the next batch; the encoding scratch is emptied.
      var b = 0
      while (b < schema.fields.length) {
        builders(b).recycle()
        val sc = dictScratch(b)
        if (sc != null) { sc._1.reset(); sc._2.reset() }
        b += 1
      }
      batchDictionaries.clear()
    }
  }

  /** The record batch's data bytes as written (the exchange's `dataSize`, see `rawBytes`). */
  private def batchBytesOf(vectors: Array[FieldVector]): Long = {
    var t = 0L
    var c = 0
    while (c < vectors.length) { t += vectors(c).getBufferSize; c += 1 }
    t
  }

  private val batchDictionaries = scala.collection.mutable.ArrayBuffer.empty[(VarCharVector, Long)]

  /** No per-buffer body compression: the partition's stream is compressed in frames (see `Segment.sink`). */
  private val codec: org.apache.arrow.vector.compression.CompressionCodec = org.apache.arrow.vector.compression.NoCompressionCodec.INSTANCE

  /** The writer's one compression context and its output scratch, reused for every frame of every partition. */
  private val frameCompressor: ShuffleCompression.FrameCompressor = ShuffleCompression.frameCompressor(compression)
  private def compressFrame(raw: Array[Byte]): Array[Byte] = frameCompressor.compress(raw)



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
      flushStaging()
      while (p < numPartitions) {
        val seg = segments(p)
        if (!staged) flushPartition(seg)
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
        val tail = ByteBuffer.wrap(seg.tail)
        while (tail.hasRemaining) out.write(tail)
        pos += tail.limit()
        seg.tail = Array.emptyByteArray
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

  override def close(): Unit = {
    segments.foreach(_.release())
    if (staging != null) staging.foreach(b => try b.close() catch { case _: Exception => })
    if (batchBuilders != null) batchBuilders.foreach(b => try b.close() catch { case _: Exception => })
    if (orderedBuilders != null) orderedBuilders.foreach(b => try b.close() catch { case _: Exception => })
    if (sliceValidity != null) {
      (sliceValidity ++ sliceOffsets ++ sliceBits).foreach(b => if (b != null) try b.close() catch { case _: Exception => })
      sliceValidity = null; sliceOffsets = null; sliceBits = null
    }
    dictScratch.foreach(sc => if (sc != null) { sc._1.close(); sc._2.close() })
    frameCompressor.close()
  }
}

object PartitionedIpcWriter {
  /** A builder's first capacity in rows, doubled as a partition fills (#351). */
  val InitialRows: Int = 256
  /** The floor of a builder's first capacity, however many partitions there are. */
  val MinInitialRows: Int = 16
  /** Above this many partitions the writer stages rows and partitions them at the flush; at or below, a builder set per partition. */
  val StagingPartitions: Int = 256
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


  /** Default share of distinct values per rows above which a batch's string column goes plain (#356). */
  val DefaultDictionaryMaxRatio: Double = 0.5
  /** Below this many rows a record batch's strings stay plain (#416): coalescible by the reader, and no dictionary to pay for. */
  val DictionaryMinRows: Int = 256
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
    if (maxRatio <= 0.0) return null
    val ids = new IntVector(name, allocator)
    val dictionary = new VarCharVector(name + ".dictionary", allocator)
    val r = encodeStrings(in, name, allocator, maxRatio, ids, dictionary)
    if (r == null) { ids.close(); dictionary.close() }
    r
  }

  /**
   * As above into the caller's `ids` and `dictionary` (emptied, grown as needed and kept by the caller
   * across blocks, #416); on `null` they are left empty.
   */
  def encodeStrings(in: VarCharVector, name: String, allocator: BufferAllocator, maxRatio: Double,
      ids: IntVector, dictionary: VarCharVector): (IntVector, VarCharVector) = {
    val n = in.getValueCount
    if (maxRatio <= 0.0) return null
    val totalBytes = if (n == 0) 0L else in.getOffsetBuffer.getInt(n.toLong * 4).toLong
    if (ids.getValueCapacity < n) ids.allocateNew(n) else ids.reset()
    // Sized to the input once (the distinct values are at most all of it): no reallocation per growth.
    if (dictionary.getValueCapacity < math.max(n, 1) || dictionary.getByteCapacity < totalBytes) dictionary.allocateNew(math.max(totalBytes, 1L), math.max(n, 1))
    else dictionary.reset()
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
        ids.reset(); dictionary.reset()
        return null
      }
    }
    ids.setValueCount(n)
    dictionary.setValueCount(next)
    (ids, dictionary)
  }
}
