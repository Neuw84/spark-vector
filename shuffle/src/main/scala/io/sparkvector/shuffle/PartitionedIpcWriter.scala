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
package io.sparkvector.shuffle

import java.io.ByteArrayOutputStream
import java.lang.foreign.{Arena, MemorySegment}
import java.nio.ByteBuffer
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{Files, Path, StandardOpenOption}
import scala.jdk.CollectionConverters._

import io.sparkvector.kernels.{
  Bitmap,
  CompactKernels,
  GatherKernels,
  PartitionKernels,
  ScatterKernels,
  SegmentVectorBuffers,
  StringDictionary,
  VecType,
  VectorBuffers
}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.arrow.{
  ArrowOutput,
  ArrowVectorBuffers,
  VectorArrowColumnVector,
  VectorDecimalColumnVector,
  VectorDictionaryColumnVector
}
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.{FieldVector, IntVector, VarCharVector, VectorSchemaRoot}
import org.apache.arrow.vector.compression.CompressionUtil
import org.apache.arrow.vector.ipc.message.IpcOption
import org.apache.arrow.vector.types.pojo.{Field, Schema}
import org.apache.spark.sql.types.{BooleanType, IntegerType, StringType, StructType}
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
    dictionaryMaxRatio: Double = PartitionedIpcWriter.DefaultDictionaryMaxRatio,
    /** Bytes of distinct values a string column's staging dictionary (#377) holds before the column goes plain for the rest of the task (#416). */
    dictionaryCapBytes: Long = PartitionedIpcWriter.DictionaryCapBytes,
    /**
     * One dictionary per string column per map file (#416): the task's staging dictionary is written
     * once, after the partition streams, and every record batch carries ids over it -- no per-block
     * dictionary, no per-block remap. `false` keeps a dictionary per record batch (the used entries,
     * #345), for a transport that delivers a block's bytes alone (Spark's block transfer).
     */
    fileDictionary: Boolean = true,
    /**
     * The staged flush partitions fixed-width, BOOL and id columns by scatter rather than by a
     * gather through the partition order (#20). `false` is the gather, kept for measuring one
     * against the other.
     */
    scatterFlush: Boolean = true
) extends AutoCloseable {

  private val arrowSchema: Schema = PartitionedIpcFile.arrowSchema(schema)

  /**
   * A builder's first capacity in rows (#416): an input batch's rows spread over the partitions, so
   * at 1000 partitions a partition sees ~8 rows per 8192-row batch and a 256-row first vector per
   * column was 74 KB per partition, 74 MB per map task, allocated and mostly never filled -- a
   * third of the map task's time at 1 TB / 1000 partitions was spent on those. The vectors still
   * double as rows arrive, so a partition that does fill pays only the reallocations it earns.
   */
  private val initialRows: Int = math.max(
    PartitionedIpcWriter.MinInitialRows,
    math.min(
      PartitionedIpcWriter.InitialRows,
      Integer.highestOneBit(math.max(1, 4 * batchRows / math.max(numPartitions, 1)))
    )
  )

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
    private val stringField = field.dataType == StringType

    /** A string column holds plain strings, or staging ids (#377) when its input arrives as INT32 ids; settled by the first rows. */
    private var idsMode = false
    private def isString = stringField && !idsMode
    private val isBool = field.dataType == BooleanType
    private def width: Int =
      if (isString || isBool) 0 else if (stringField) 4 else PartitionedIpcWriter.byteWidth(field.dataType)
    private def settle(in: VectorBuffers): Unit =
      if (stringField && vector == null && rows == 0) idsMode = in.`type`() == VecType.INT32
    var vector: FieldVector = _
    var buffers: ArrowVectorBuffers = _
    var rows: Int = 0
    var dataBytes: Long = 0L

    private def allocate(rowCapacity: Int, byteCapacity: Long): Unit = {
      buffers = if (isString) ArrowOutput.allocateUtf8(field.name, rowCapacity, math.max(byteCapacity, 1L), allocator)
      else
        ArrowOutput.allocateFixed(field.name, if (stringField) IntegerType else field.dataType, rowCapacity, allocator)
      vector = buffers.vector().asInstanceOf[FieldVector]
      if (isString) buffers.offsets().set(VectorBuffers.LE_INT, 0L, 0)
    }

    /** Room for `count` more rows and `bytes` more string bytes, doubling the vector as needed. */
    private def ensure(count: Int, bytes: Long): Unit = {
      if (retained) { retainedBytes -= capacityBytes; retained = false }
      if (vector == null) {
        val cap = math.max(initialRows, Integer.highestOneBit(math.max(count, 1) - 1) << 1)
        allocate(
          math.min(math.max(cap, count), math.max(batchRows, count)),
          if (isString) math.max(bytes, PartitionedIpcWriter.InitialBytesPerRow.toLong * cap) else 0L
        )
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
      settle(in)
      val bytes = if (isString) CompactKernels.selectedUtf8Bytes(in, mask) else 0L
      ensure(count, bytes)
      val validityScratch = Bitmap.allocate(scratch, count)
      if (isString) {
        val offsets = buffers.offsets().asSlice(rows.toLong << 2)
        CompactKernels.compactUtf8(
          in,
          mask,
          count,
          offsets,
          buffers.data().asSlice(dataBytes),
          if (in.validity() != null) validityScratch else null
        )
        if (dataBytes > 0) {
          // The kernel's offsets start at zero: rebase them on the bytes already there (count + 1 of them).
          var i = 0
          while (i <= count) {
            offsets.set(
              VectorBuffers.LE_INT,
              i.toLong << 2,
              offsets.get(VectorBuffers.LE_INT, i.toLong << 2) + dataBytes.toInt
            ); i += 1
          }
        }
      } else if (isBool) {
        val bits = Bitmap.allocate(scratch, count)
        CompactKernels.compactFixed(in, mask, count, bits, if (in.validity() != null) validityScratch else null)
        Bitmap.copyBits(bits, buffers.data(), rows, count)
      } else {
        CompactKernels.compactFixed(
          in,
          mask,
          count,
          buffers.data().asSlice(rows.toLong * width),
          if (in.validity() != null) validityScratch else null
        )
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
      settle(in)
      val srcOff = in.offsets()
      val start = if (isString) srcOff.get(VectorBuffers.LE_INT, 0L) else 0
      val bytes = if (isString) (srcOff.get(VectorBuffers.LE_INT, n.toLong << 2) - start).toLong else 0L
      ensure(n, bytes)
      if (isString) {
        val offsets = buffers.offsets()
        val base = dataBytes.toInt - start
        var i = 0
        while (i <= n) {
          offsets.set(
            VectorBuffers.LE_INT,
            (rows + i).toLong << 2,
            srcOff.get(VectorBuffers.LE_INT, i.toLong << 2) + base
          ); i += 1
        }
        if (bytes > 0) MemorySegment.copy(in.data(), start.toLong, buffers.data(), dataBytes, bytes)
      } else if (isBool) {
        Bitmap.copyBits(in.data(), buffers.data(), rows, n)
      } else {
        MemorySegment.copy(in.data(), 0L, buffers.data(), rows.toLong * width, n.toLong * width)
      }
      if (in.validity() != null) {
        Bitmap.copyBits(in.validity(), buffers.validity(), rows, n)
        if (!mayHaveNulls && !Bitmap.allSet(in.validity(), n)) mayHaveNulls = true
      } else Bitmap.fillRange(buffers.validity(), rows, n, true)
      rows += n
      dataBytes += bytes
      if (isString) bytes + (n.toLong << 2) else n.toLong * math.max(width, 1)
    }

    /**
     * Whether any row appended since the last reset may be null (the staging, #20): a flush scatters
     * the validity of such a column and fills it for every other.
     */
    var mayHaveNulls = false

    /** A column the staged flush partitions by scatter (#20): fixed width, BOOL, or a string column's ids; not plain strings. */
    def scatters: Boolean = !isString

    /**
     * Appends rows `from until from + count` of `in`, a fixed-width, BOOL or id column laid out as
     * this builder's (the scattered staging, #20): contiguous copies. `allValid` skips the validity
     * copy for a column with no nulls.
     */
    def appendRange(in: VectorBuffers, from: Int, count: Int, allValid: Boolean): Long = {
      settle(in)
      ensure(count, 0L)
      if (isBool) Bitmap.copyBitsFrom(in.data(), from, buffers.data(), rows, count)
      else MemorySegment.copy(in.data(), from.toLong * width, buffers.data(), rows.toLong * width, count.toLong * width)
      if (allValid || in.validity() == null) Bitmap.fillRange(buffers.validity(), rows, count, true)
      else Bitmap.copyBitsFrom(in.validity(), from, buffers.validity(), rows, count)
      rows += count
      count.toLong * math.max(width, 1)
    }

    /** Appends rows `idx(from until to)` of `in` (the index-list path, #353). */
    def appendIndexed(in: VectorBuffers, idx: Array[Int], from: Int, to: Int, scratch: Arena): Long = {
      settle(in)
      val count = to - from
      val bytes = if (isString) GatherKernels.gatherUtf8Bytes(in, idx, from, to) else 0L
      ensure(count, bytes)
      val validityScratch = if (in.validity() != null) Bitmap.allocate(scratch, count) else null
      if (isString) {
        val offsets = buffers.offsets().asSlice(rows.toLong << 2)
        GatherKernels.gatherUtf8(in, idx, from, to, offsets, buffers.data().asSlice(dataBytes), validityScratch)
        if (dataBytes > 0) {
          var i = 0
          while (i <= count) {
            offsets.set(
              VectorBuffers.LE_INT,
              i.toLong << 2,
              offsets.get(VectorBuffers.LE_INT, i.toLong << 2) + dataBytes.toInt
            ); i += 1
          }
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
    def capacityBytes: Long = if (vector == null) 0L
    else {
      var t = 0L; val bs = vector.getBuffers(false); var i = 0; while (i < bs.length) { t += bs(i).capacity(); i += 1 };
      t
    }

    /**
     * After a flush: the vector kept and emptied for the next rows (a reset keeps its capacity). A
     * shared builder (the staging, the batch vectors) always keeps; a partition's own keeps while
     * the writer's retained capacity stays under `retainBudget` (#417) -- the allocator cap in
     * `appendIndexed` measures the allocator's total, and unbounded retention across a few hundred
     * partitions tripped it into flushing on every append (record batches of a few rows, 1.5x the
     * bytes, 2x the time on q67 at 200 partitions).
     */
    def recycle(): Unit = {
      mayHaveNulls = false
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
      vector = null; buffers = null; rows = 0; dataBytes = 0L; mayHaveNulls = false
    }

    /** After a column's mode changed (#416: its dictionary froze): the vector dropped so the next rows settle the builder anew. */
    def resettle(): Unit = { close(); idsMode = false }

    /**
     * The pending ids decoded into plain strings in place (#416, item 5): a column frozen before any of
     * its ids reached the file needs no dictionary section -- its pending rows leave as UTF8 like the
     * rows after it. The builder continues as a plain string builder.
     */
    def decodeIds(ic: IdColumn): Unit = {
      if (!idsMode) return
      if (vector == null || rows == 0) { resettle(); return }
      if (retained) { retainedBytes -= capacityBytes; retained = false }
      val decoded = ic.decode(vector.asInstanceOf[IntVector], rows, new VarCharVector(field.name, allocator))
      vector.close()
      vector = decoded
      idsMode = false
      buffers = ArrowVectorBuffers.forWrite(vector, vector.getValueCapacity, field.dataType)
      dataBytes = decoded.getOffsetBuffer.getInt(rows.toLong << 2)
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
      if (builders != null) builders.foreach(b =>
        try b.close()
        catch { case _: Exception => }
      )
      if (overflow != null) {
        try overflow.close()
        catch { case _: Exception => };
        try Files.deleteIfExists(overflowPath)
        catch { case _: Exception => }
      }
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
  private val staging: Array[Builder] =
    if (staged) Array.tabulate(schema.fields.length)(new Builder(_, shared = true)) else null
  private var stagedIds: Array[Int] = new Array[Int](0)
  private var stagedRows: Int = 0

  /** The vectors a partition's rows are gathered into for one record batch, reused for every partition and flush. */
  private val batchBuilders: Array[Builder] =
    if (staged) Array.tabulate(schema.fields.length)(new Builder(_, shared = true)) else null

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
    if (sc == null) {
      sc = (
        new IntVector(schema.fields(c).name, allocator),
        new VarCharVector(schema.fields(c).name + ".dictionary", allocator)
      ); dictScratch(c) = sc
    }
    sc
  }
  private var rawBytesWritten = 0L

  /**
   * A string column in ids mode (#377): its rows are staged as int ids into a dictionary of the
   * distinct values the task has seen, and each record batch's dictionary is built at the flush by
   * remapping the ids the batch uses -- no hash and no compare per row where the input was already
   * dictionary encoded (a Parquet scan, the grouped aggregate's keys). A column takes this mode when
   * its first batch arrives dictionary encoded; a plain batch on such a column is mapped row by row.
   */
  private final class IdColumn(c: Int) {
    val dict = new StringDictionary()

    /** Non-null rows mapped so far: the denominator of the task-level distinct ratio (#416). */
    var rowsSeen: Long = 0L

    /** Whether a record batch carrying this column's ids has been written: then the dictionary section must hold the dictionary. */
    var flushedIds: Boolean = false
    private val scratch = new StringDictionary.Scratch()
    private var rowIds = new Array[Int](0)
    // Input dictionary entries -> staging ids, remembered while the same dictionary keeps arriving --
    // the same buffers object, or the same Arrow vector under a new wrapper (the aggregate's shared
    // dictionary reaches us wrapped anew per batch) -- and mapped when first used, so a batch costs
    // its rows plus its distinct entries. Not by buffer address: a freed dictionary's address is the
    // next batch's dictionary's soon enough.
    private var entryIds = new Array[Int](0)
    private var entryGen = new Array[Int](0)
    private var gen = 0
    private var lastDict: VectorBuffers = _
    private var lastVector: AnyRef = _
    private var lastLength = -1
    private var offs = new Array[Int](0)
    private var bytes = new Array[Byte](0)
    // Flush scratch: staging id -> the batch's dense id, valid for one flush generation.
    private var dense = new Array[Int](0)
    private var denseGen = new Array[Int](0)
    private var flushGen = 0

    private def arrowVector(d: VectorBuffers): AnyRef = d match {
      case a: ArrowVectorBuffers => a.vector()
      case _ => null
    }

    private def sameDictionary(d: VectorBuffers): Boolean = {
      if (d eq lastDict) return true
      val v = arrowVector(d)
      v != null && (v eq lastVector) && d.length() == lastLength
    }

    /** The batch's rows as INT32 staging ids (nulls kept in the validity). */
    def map(in: VectorBuffers, n: Int): VectorBuffers = {
      if (rowIds.length < n) rowIds = new Array[Int](math.max(n, rowIds.length * 2))
      val ids = rowIds
      val validity = in.validity()
      rowsSeen += (if (validity == null) n else Bitmap.popcount(validity, n))
      if (in.isDictionaryEncoded) {
        val d = in.dictionary()
        val m = d.length()
        if (entryIds.length < m) {
          entryIds = new Array[Int](math.max(m, entryIds.length * 2)); entryGen = new Array[Int](entryIds.length);
          gen = 0; lastDict = null; lastLength = -1
        }
        if (!sameDictionary(d)) {
          lastDict = d; lastLength = m; lastVector = arrowVector(d)
          gen += 1
          if (gen == 0) { java.util.Arrays.fill(entryGen, 0); gen = 1 }
        }
        val g = gen
        val dOff = d.offsets(); val dData = d.data(); val dValidity = d.validity()
        val idx = in.data()
        var i = 0
        while (i < n) {
          if (validity != null && !Bitmap.isSet(validity, i)) ids(i) = 0
          else {
            val e = idx.get(VectorBuffers.LE_INT, i.toLong << 2)
            if (entryGen(e) != g) {
              entryIds(e) = if (dValidity != null && !Bitmap.isSet(dValidity, e)) 0
              else {
                val start = dOff.get(VectorBuffers.LE_INT, e.toLong << 2)
                dict.indexOf(dData, start, dOff.get(VectorBuffers.LE_INT, (e + 1).toLong << 2) - start, true, scratch)
              }
              entryGen(e) = g
            }
            ids(i) = entryIds(e)
          }
          i += 1
        }
      } else {
        if (offs.length < n + 1) offs = new Array[Int](math.max(n + 1, offs.length * 2))
        MemorySegment.copy(in.offsets(), VectorBuffers.LE_INT, 0L, offs, 0, n + 1)
        val first = offs(0)
        val total = offs(n) - first
        if (bytes.length < total) bytes = new Array[Byte](math.max(total, bytes.length * 2))
        MemorySegment.copy(in.data(), java.lang.foreign.ValueLayout.JAVA_BYTE, first.toLong, bytes, 0, total)
        var i = 0
        while (i < n) {
          if (validity != null && !Bitmap.isSet(validity, i)) ids(i) = 0
          else {
            val start = offs(i) - first
            val len = offs(i + 1) - offs(i)
            ids(i) = dict.indexOf(StringDictionary.fingerprint(bytes, start, len), len, bytes, start, true)
          }
          i += 1
        }
      }
      SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, MemorySegment.ofArray(ids))
    }

    /**
     * The record batch's dictionary from the staging ids it uses (#345's shape: the used entries
     * only, dense ids in first-seen order), into the column's encoding scratch; null when the
     * distinct values exceed `dictionaryMaxRatio` of the rows (#356) -- checked as `encodeStrings`
     * does, after the sample and at the end -- and the caller ships the column plain.
     */
    def remap(staged: IntVector, n: Int, ids: IntVector, dictionary: VarCharVector): (IntVector, VarCharVector) = {
      if (dictionaryMaxRatio <= 0.0) return null
      val size = dict.size()
      if (dense.length < size) {
        dense = new Array[Int](math.max(size, dense.length * 2)); denseGen = new Array[Int](dense.length); flushGen = 0
      }
      flushGen += 1
      if (flushGen == 0) { java.util.Arrays.fill(denseGen, 0); flushGen = 1 }
      val g = flushGen
      if (ids.getValueCapacity < n) ids.allocateNew(n) else ids.reset()
      if (dictionary.getValueCapacity < math.max(n, 1))
        dictionary.allocateNew(math.max(n.toLong * PartitionedIpcWriter.InitialBytesPerRow, 1L), math.max(n, 1))
      else dictionary.reset()
      val store = dict.bytes()
      var next = 0
      var i = 0
      while (i < n) {
        if (staged.isNull(i)) ids.setNull(i)
        else {
          val id = staged.get(i)
          if (denseGen(id) != g) {
            denseGen(id) = g
            dense(id) = next
            dictionary.setSafe(next, store, dict.offset(id), dict.length(id))
            next += 1
          }
          ids.set(i, dense(id))
        }
        i += 1
        if (
          (i == PartitionedIpcWriter.DictionarySampleRows || i == n) && next > (i * dictionaryMaxRatio) && dictionaryMaxRatio < 1.0
        ) {
          ids.reset(); dictionary.reset()
          return null
        }
      }
      ids.setValueCount(n)
      dictionary.setValueCount(next)
      (ids, dictionary)
    }

    /** The staged ids as a plain string vector (the #356 fallback): one gather from the dictionary. */
    def decode(staged: IntVector, n: Int, out: VarCharVector): VarCharVector = {
      var total = 0L
      var i = 0
      while (i < n) { if (!staged.isNull(i)) total += dict.length(staged.get(i)); i += 1 }
      if (out.getValueCapacity < math.max(n, 1) || out.getByteCapacity < total)
        out.allocateNew(math.max(total, 1L), math.max(n, 1))
      else out.reset()
      val store = dict.bytes()
      i = 0
      while (i < n) {
        if (staged.isNull(i)) out.setNull(i)
        else { val id = staged.get(i); out.setSafe(i, store, dict.offset(id), dict.length(id)) }
        i += 1
      }
      out.setValueCount(n)
      out
    }

    def reset(): Unit = { dict.clear(); gen += 1; lastDict = null; lastVector = null; lastLength = -1 }
  }

  /**
   * Every string column starts in ids mode (#416): its rows become staging ids over the column's
   * dictionary whatever the input's encoding -- a dictionary-encoded batch by its entries, a plain one
   * row by row (the hash `encodeStrings` did per block, into one dictionary per task instead). A
   * column whose dictionary stops paying -- past `dictionaryCapBytes`, or more distinct values than
   * `dictionaryMaxRatio` of its rows once `FreezeSampleRows` have been seen -- is frozen: everything
   * pending is flushed as ids (they stay valid: the dictionary is written whole at `finish`) and the
   * column travels plain for the rest of the task.
   */
  private val idColumns: Array[IdColumn] =
    Array.tabulate(schema.fields.length)(c => if (schema.fields(c).dataType == StringType) new IdColumn(c) else null)
  private val frozen = new Array[Boolean](schema.fields.length)

  /** Per ids-mode string column, the plain vector of a batch that goes plain (#356), reused across blocks. */
  private val plainScratch = new Array[VarCharVector](schema.fields.length)
  private def plainScratchFor(c: Int): VarCharVector = {
    var v = plainScratch(c)
    if (v == null) { v = new VarCharVector(schema.fields(c).name, allocator); plainScratch(c) = v }
    v
  }

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
    // A string column: in ids mode (#377) its rows become staging ids over the column's dictionary,
    // whatever the batch's encoding, and each record batch's dictionary is a remap at the flush. A
    // column whose first batch arrives plain stays plain: a dictionary-encoded batch on it is decoded
    // once here and the strings are dictionary-encoded again at the flush (#349, #351).
    val plain = new Array[VectorBuffers](buffers.length)
    var col = 0
    while (col < buffers.length) {
      val b = buffers(col)
      plain(col) = if (b.`type`() != VecType.UTF8) b
      else {
        if (!frozen(col)) idColumns(col).map(b, n)
        else if (b.isDictionaryEncoded()) ArrowOutput.decodeDictionary(b, scratch)
        else b
      }
      col += 1
    }
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
      capDictionaries()
      return
    }
    if (stagedIds.length < stagedRows + n)
      stagedIds = java.util.Arrays.copyOf(stagedIds, math.max(stagedRows + n, stagedIds.length * 2))
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
    capDictionaries()
  }

  /**
   * A string column whose dictionary stops paying (#416) -- a high-cardinality column: past
   * `dictionaryCapBytes` of distinct values, or more distinct values than `dictionaryMaxRatio` of its
   * rows once `FreezeSampleRows` have been mapped -- is frozen: every partition's pending rows are
   * flushed first (as ids over the dictionary, which stays whole until `finish`), then the column's
   * builders drop their id vectors and its rows travel plain from here on. Rare, and the equivalent
   * of the memory-bound flush the plain path makes anyway.
   */
  private def capDictionaries(): Unit = {
    var c = 0
    var any = false
    while (c < idColumns.length) {
      val ic = idColumns(c)
      if (ic != null && !frozen(c) && mustFreeze(ic)) any = true
      c += 1
    }
    if (!any) return
    // A freezing column whose ids already reached the file keeps its dictionary (the section serves
    // those batches), and its pending rows leave as ids too: everything pending is flushed first. A
    // column frozen before any flush -- the usual case, the sample is smaller than a batch -- has its
    // pending ids decoded in place instead and writes no dictionary section at all (item 5): at 1000
    // partitions every reduce task received every map's dictionary, and for a near-distinct column
    // that section was most of a narrow task's bytes.
    var anyFlushed = false
    c = 0
    while (c < idColumns.length) {
      val ic = idColumns(c); if (ic != null && !frozen(c) && mustFreeze(ic) && ic.flushedIds) anyFlushed = true; c += 1
    }
    if (anyFlushed) {
      if (staged) flushStaging() else { var p = 0; while (p < numPartitions) { flushPartition(segments(p)); p += 1 } }
    }
    c = 0
    while (c < idColumns.length) {
      val ic = idColumns(c)
      if (ic != null && !frozen(c) && mustFreeze(ic)) {
        frozen(c) = true
        if (ic.flushedIds) {
          if (staged) { staging(c).resettle(); batchBuilders(c).resettle() }
          else { var p = 0; while (p < numPartitions) { segments(p).builders(c).resettle(); p += 1 } }
        } else {
          if (staged) { staging(c).decodeIds(ic); batchBuilders(c).resettle() }
          else { var p = 0; while (p < numPartitions) { segments(p).builders(c).decodeIds(ic); p += 1 } }
        }
      }
      c += 1
    }
  }

  private def mustFreeze(ic: IdColumn): Boolean =
    dictionaryMaxRatio <= 0.0 || ic.dict.valueBytes() > dictionaryCapBytes ||
      (dictionaryMaxRatio < 1.0 && ic.rowsSeen >= PartitionedIpcWriter.FreezeSampleRows && ic.dict.size() > ic.rowsSeen * dictionaryMaxRatio)

  private val starts = new Array[Int](numPartitions + 1)
  private var order = new Array[Int](0)
  private var dest = new Array[Int](0)

  private def appendIndexed(
      seg: Segment,
      buffers: Array[VectorBuffers],
      idx: Array[Int],
      from: Int,
      to: Int,
      scratch: Arena
  ): Unit = {
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
      if (fullest == null) { heldBytes = 0L; return }
      else flushPartition(fullest)
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
    // #20: the columns that scatter are partitioned by one sequential pass each into a partition-
    // ordered copy, and each record batch copies a contiguous run of it; only plain string columns
    // still gather through the order (their bytes vary per row).
    val scattered = new Array[org.apache.arrow.vector.FieldVector](schema.fields.length)
    try {
      val n = stagedRows
      val source: Array[VectorBuffers] = Array.tabulate(schema.fields.length)(c => staging(c).finished())
      val scatter = Array.tabulate(schema.fields.length)(c => scatterFlush && staging(c).scatters)
      val gather = scatter.exists(!_)
      if (scatter.exists(identity)) {
        if (dest.length < n) dest = new Array[Int](n)
        PartitionKernels.partitionDestinations(stagedIds, n, numPartitions, starts, dest)
      }
      if (gather) {
        if (order.length < n) order = new Array[Int](n)
        PartitionKernels.partitionOrder(stagedIds, n, numPartitions, starts, order)
      }
      val scatteredBuffers = new Array[VectorBuffers](schema.fields.length)
      val allValid = new Array[Boolean](schema.fields.length)
      var c = 0
      while (c < schema.fields.length) {
        if (scatter(c)) {
          val src = source(c)
          val dt = if (src.`type`() == VecType.INT32 && schema.fields(c).dataType == StringType) IntegerType
          else schema.fields(c).dataType
          val out =
            ArrowOutput.allocateFixed(schema.fields(c).name, dt, n, allocator) // zeroed: the bit scatters need it
          scattered(c) = out.vector().asInstanceOf[org.apache.arrow.vector.FieldVector]
          ScatterKernels.scatterFixed(src.`type`(), src.data(), n, dest, out.data())
          allValid(c) = !staging(c).mayHaveNulls || src.validity() == null
          if (!allValid(c)) ScatterKernels.scatterBits(src.validity(), n, dest, out.validity())
          scatteredBuffers(c) = out
        }
        c += 1
      }
      var p = 0
      while (p < numPartitions) {
        var from = starts(p)
        val end = starts(p + 1)
        while (from < end) {
          val to = math.min(end, from + batchRows)
          c = 0
          while (c < schema.fields.length) {
            if (scatter(c)) batchBuilders(c).appendRange(scatteredBuffers(c), from, to - from, allValid(c))
            else batchBuilders(c).appendIndexed(source(c), order, from, to, scratch)
            c += 1
          }
          flush(segments(p), to - from, batchBuilders)
          from = to
        }
        p += 1
      }
    } finally {
      scratch.close()
      scattered.foreach(v => if (v != null) v.close())
    }
    var c = 0
    while (c < schema.fields.length) { staging(c).recycle(); c += 1 }
    stagedRows = 0
    heldBytes = 0L
  }

  /** What the writer's allocator holds right now (pending slices; the roots are emptied after each batch). */
  def allocatedBytes: Long = allocator.getAllocatedMemory

  /** The partition's held slices become one record batch of its stream. */
  private def flush(seg: Segment, rows: Int, builders: Array[Builder]): Unit = if (rows > 0) {
    val taken = new Array[FieldVector](schema.fields.length)
    try {
      var c = 0
      while (c < schema.fields.length) { taken(c) = builders(c).finish(); c += 1 }
      // The batch's shape (#416): a string column is int32 ids over its dictionary when its builder
      // holds ids (the column is in ids mode), plain UTF8 otherwise (frozen, or a per-block dictionary
      // that did not pay). With `fileDictionary` the ids index the task's dictionary, written once at
      // `finish`; without it each batch's used entries are remapped dense and sent as a dictionary
      // unit ahead of the record batch (#345, #356).
      var batchFields: java.util.List[Field] = null // built only when a column goes plain
      java.util.Arrays.fill(shapeBytes, 0.toByte)
      c = 0
      while (c < schema.fields.length) {
        val encoding = arrowSchema.getFields.get(c).getDictionary
        if (encoding != null) {
          val ic = idColumns(c)
          var encodedColumn = false
          taken(c) match {
            case staged: IntVector =>
              if (fileDictionary) { encodedColumn = true; ic.flushedIds = true }
              else {
                val (sids, sdict) = scratchFor(c)
                val encoded =
                  if (rows < PartitionedIpcWriter.DictionaryMinRows) null else ic.remap(staged, rows, sids, sdict)
                if (encoded != null) {
                  val (ids, dictionary) = encoded
                  taken(c) = ids
                  batchDictionaries += ((dictionary, c))
                  encodedColumn = true
                } else taken(c) = ic.decode(staged, rows, plainScratchFor(c))
              }
            case _ => // plain strings: the column is frozen
          }
          if (encodedColumn) shapeBytes(c >> 3) = (shapeBytes(c >> 3) | (1 << (c & 7))).toByte
          else {
            if (batchFields == null) batchFields = new java.util.ArrayList[Field](arrowSchema.getFields)
            batchFields.set(c, plainFields(c))
          }
        }
        c += 1
      }
      val batchSchema = if (batchFields == null) arrowSchema else new Schema(batchFields)
      val out = new org.apache.arrow.vector.ipc.WriteChannel(seg.sink)
      // No schema message and no end-of-stream marker (#411): the reader knows the shuffle's schema
      // from the dependency. Each IPC message follows a unit header of ours (#416) -- a dictionary unit
      // names its column, a record batch unit carries the shape bitmap.
      var d = 0
      while (d < batchDictionaries.length) {
        val (vector, column) = batchDictionaries(d)
        PartitionedIpcFile.writeUnitHeader(seg.sink, PartitionedIpcFile.UnitDictionary, column)
        writeDictionary(out, vector, PartitionedIpcFile.dictionaryEncoding(column).getId)
        d += 1
      }
      PartitionedIpcFile.writeUnitHeader(seg.sink, PartitionedIpcFile.UnitRecordBatch, shapeBytes.length)
      val shape = ByteBuffer.wrap(shapeBytes)
      while (shape.hasRemaining) seg.sink.write(shape)
      val root = new VectorSchemaRoot(batchSchema.getFields, java.util.Arrays.asList(taken: _*), rows)
      val batch = new org.apache.arrow.vector.VectorUnloader(root, true, codec, true).getRecordBatch
      try org.apache.arrow.vector.ipc.message.MessageSerializer.serialize(out, batch, IpcOption.DEFAULT)
      finally batch.close()
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
        if (plainScratch(b) != null) plainScratch(b).reset()
        b += 1
      }
      batchDictionaries.clear()
    }
  }

  private val shapeBytes = new Array[Byte](math.max(1, (schema.fields.length + 7) / 8))

  /** One dictionary as an IPC `DictionaryBatch` message with `id`, over a root that wraps the vector. */
  private def writeDictionary(out: org.apache.arrow.vector.ipc.WriteChannel, vector: VarCharVector, id: Long): Unit = {
    val droot = new VectorSchemaRoot(
      java.util.List.of(vector.getField),
      java.util.List.of[FieldVector](vector),
      vector.getValueCount
    )
    val dbatch = new org.apache.arrow.vector.ipc.message.ArrowDictionaryBatch(
      id,
      new org.apache.arrow.vector.VectorUnloader(droot, true, codec, true).getRecordBatch,
      false
    )
    try org.apache.arrow.vector.ipc.message.MessageSerializer.serialize(out, dbatch, IpcOption.DEFAULT)
    finally dbatch.close()
  }

  /**
   * The map file's dictionary section (#416): a dictionary unit per string column whose task dictionary
   * has values -- the whole dictionary, dense in first-seen order, exactly what the batches' ids index --
   * as one compressed frame. Empty (no unit) when nothing was encoded or the format is per block.
   */
  private def dictionarySection(): Array[Byte] = {
    val raw = new ByteArrayOutputStream()
    if (fileDictionary) {
      val sink = new PartitionedIpcWriter.NonClosing(Channels.newChannel(raw))
      val out = new org.apache.arrow.vector.ipc.WriteChannel(sink)
      var c = 0
      while (c < idColumns.length) {
        val ic = idColumns(c)
        // Every column whose ids reached the file gets its dictionary, an empty one included: a string
        // column that is null in every row (TPC-DS's c_login) stays in ids mode with an empty dictionary,
        // and a reader meeting its batches without one refuses the stream (#416).
        if (ic != null && ic.flushedIds) {
          val vector = new VarCharVector(schema.fields(c).name + ".dictionary", allocator)
          try {
            val n = ic.dict.size()
            vector.allocateNew(math.max(ic.dict.valueBytes(), 1L), math.max(n, 1))
            val store = ic.dict.bytes()
            var i = 0
            while (i < n) { vector.setSafe(i, store, ic.dict.offset(i), ic.dict.length(i)); i += 1 }
            vector.setValueCount(n)
            PartitionedIpcFile.writeUnitHeader(sink, PartitionedIpcFile.UnitDictionary, c)
            writeDictionary(out, vector, PartitionedIpcFile.dictionaryEncoding(c).getId)
          } finally vector.close()
        }
        c += 1
      }
    }
    if (raw.size() == 0) Array.emptyByteArray else compressFrame(raw.toByteArray)
  }

  /** The record batch's data bytes as written (the exchange's `dataSize`, see `rawBytes`). */
  private def batchBytesOf(vectors: Array[FieldVector]): Long = {
    var t = 0L
    var c = 0
    while (c < vectors.length) { t += vectors(c).getBufferSize; c += 1 }
    t
  }

  private val batchDictionaries = scala.collection.mutable.ArrayBuffer.empty[(VarCharVector, Int)]

  /** No per-buffer body compression: the partition's stream is compressed in frames (see `Segment.sink`). */
  private val codec: org.apache.arrow.vector.compression.CompressionCodec =
    org.apache.arrow.vector.compression.NoCompressionCodec.INSTANCE

  /** The writer's one compression context and its output scratch, reused for every frame of every partition. */
  private val frameCompressor: ShuffleCompression.FrameCompressor = ShuffleCompression.frameCompressor(compression)
  private def compressFrame(raw: Array[Byte]): Array[Byte] = frameCompressor.compress(raw)

  /**
   * Ends every stream and writes the data file: the streams back to back and, when `withFooter`,
   * the index footer. Under Spark's shuffle the lengths go to the block resolver's index file
   * instead and the data file must be exactly the streams, so the writer there passes `false`.
   */
  def finish(withFooter: Boolean = true): PartitionedIpcFile.Index = {
    val out =
      FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)
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
      // The dictionary section and its trailer (#416), then the index footer when there is one.
      val section = ByteBuffer.wrap(dictionarySection())
      val sectionOffset = pos
      while (section.hasRemaining) out.write(section)
      pos += section.limit()
      val trailer = ByteBuffer.wrap(PartitionedIpcFile.encodeDictTrailer(sectionOffset, section.limit()))
      while (trailer.hasRemaining) out.write(trailer)
      pos += trailer.limit()
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
    if (staging != null) staging.foreach(b =>
      try b.close()
      catch { case _: Exception => }
    )
    if (batchBuilders != null) batchBuilders.foreach(b =>
      try b.close()
      catch { case _: Exception => }
    )
    dictScratch.foreach(sc => if (sc != null) { sc._1.close(); sc._2.close() })
    plainScratch.foreach(v => if (v != null) v.close())
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
    case org.apache.spark.sql.types.IntegerType | org.apache.spark.sql.types.DateType | org.apache.spark.sql.types.ByteType | org.apache.spark.sql.types.ShortType =>
      4 // narrow ints ride INT32 lanes (#327)
    case org.apache.spark.sql.types.LongType | org.apache.spark.sql.types.TimestampType | org.apache.spark.sql.types.DoubleType =>
      8
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

  /** Bytes of distinct values a string column's task dictionary (#377) holds before the column is frozen and goes plain (#416). */
  val DictionaryCapBytes: Long = 32L << 20

  /** Non-null rows a string column maps before its task-level distinct ratio is judged against `dictionaryMaxRatio` (#416). */
  val FreezeSampleRows: Long = 4096L

  /**
   * Dictionary-encodes a plain string vector: the distinct values in first-seen order, int32 ids,
   * nulls kept. Returns `null` -- nothing allocated stays behind -- when the distinct values exceed
   * `maxRatio` of the rows seen, checked after [[DictionarySampleRows]] rows and at the end: the
   * caller then ships the column plain (#356).
   */
  def encodeStrings(
      in: VarCharVector,
      name: String,
      allocator: BufferAllocator,
      maxRatio: Double = DefaultDictionaryMaxRatio
  ): (IntVector, VarCharVector) = {
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
  def encodeStrings(
      in: VarCharVector,
      name: String,
      allocator: BufferAllocator,
      maxRatio: Double,
      ids: IntVector,
      dictionary: VarCharVector
  ): (IntVector, VarCharVector) = {
    val n = in.getValueCount
    if (maxRatio <= 0.0) return null
    val totalBytes = if (n == 0) 0L else in.getOffsetBuffer.getInt(n.toLong * 4).toLong
    if (ids.getValueCapacity < n) ids.allocateNew(n) else ids.reset()
    // Sized to the input once (the distinct values are at most all of it): no reallocation per growth.
    if (dictionary.getValueCapacity < math.max(n, 1) || dictionary.getByteCapacity < totalBytes)
      dictionary.allocateNew(math.max(totalBytes, 1L), math.max(n, 1))
    else dictionary.reset()
    // The distinct values as an open-addressing table over the input's own bytes (#387): an entry is the
    // row where its value was first seen, hashed and compared in place through the Arrow buffers -- no
    // ByteBuffer per row, no boxing, no HashMap. A HashMap of ByteBuffers here was 11% of an executor's
    // time in q67 at 1 TB (encodeStrings 7%, ByteBuffer.hashCode 4%).
    val offsets = MemorySegment.ofBuffer(in.getOffsetBuffer.nioBuffer(0, (n + 1) * 4))
    val data = if (totalBytes == 0) MemorySegment.NULL
    else MemorySegment.ofBuffer(in.getDataBuffer.nioBuffer(0, totalBytes.toInt))
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
            if (end > start) dictionary.setSafe(next, dataBuf, start, end - start)
            else dictionary.setSafe(next, Array.emptyByteArray)
            next += 1
          } else {
            val row = slot - 1
            val rs = offsets.get(VectorBuffers.LE_INT, row.toLong * 4)
            val re = offsets.get(VectorBuffers.LE_INT, (row.toLong + 1) * 4)
            if (re - rs == end - start && (end == start || MemorySegment.mismatch(data, rs, re, data, start, end) < 0))
              id = entryId(row)
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
