/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package io.vecruntime.shuffle

import java.nio.ByteBuffer
import java.nio.channels.{FileChannel, ReadableByteChannel}
import java.nio.file.{Path, StandardOpenOption}
import java.util.{HashMap => JHashMap}
import scala.jdk.CollectionConverters._

import io.vecruntime.spark.adapter.TypeMapping
import io.vecruntime.kernels.{Bitmap, VectorBuffers}
import io.vecruntime.spark.arrow.{
  ArrowVectorBuffers,
  VectorArrowColumnVector,
  VectorDecimalColumnVector,
  VectorDictionaryColumnVector,
  VectorNarrowIntColumnVector
}
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.{FieldVector, IntVector, VarCharVector}
import org.apache.arrow.vector.types.{DateUnit, FloatingPointPrecision, TimeUnit}
import org.apache.arrow.vector.types.pojo.{ArrowType, DictionaryEncoding, Field, FieldType, Schema}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * The map output file of the columnar shuffle: `numPartitions` Arrow IPC streams back to back, then
 * the index (`(offset, length, rows)` per partition), its length and a magic word, so a reader seeks
 * to one partition's stream and never touches the others. The Arrow schema is the one
 * `ArrowOutput.newVector` produces for the Spark type, with two conventions the reader needs to
 * rebuild the operators' column vectors: a string column is a dictionary-encoded `int32` field
 * (dictionary id = column ordinal + 1) whatever the batch's own encoding, and a decimal of at most 18
 * digits is an `int64` of unscaled values. Every field carries the Spark type in its metadata under
 * `sparkvector.type`.
 */
object PartitionedIpcFile {

  val Magic: Long = 0x53564950434631L // "SVIPCF1"
  val TypeKey = "sparkvector.type"

  /**
   * Rows a reader accumulates small plain batches up to before handing a batch to the operators (#411).
   * The system property `sparkvector.shuffle.reader.coalesceRows` overrides the default for a JVM
   * (the transport benchmark sweeps it per fork; on a cluster, `spark.executor.extraJavaOptions`).
   */
  val CoalesceRows: Int = Integer.getInteger("sparkvector.shuffle.reader.coalesceRows", 1024)

  /**
   * A small block with a dictionary-encoded column reaches the operators as it is -- ids over its
   * dictionary -- once it has at least this many rows (#416, item 6). Decoding it into the coalesced
   * plain batch trades the per-batch costs coalescing saves for a decode of every row plus the
   * aggregate hashing the strings it would otherwise have consumed as ids; q67's final stage at 1000
   * partitions, whose blocks are a few hundred rows over their map's dictionary, was 10-30% slower
   * for it. Below the floor a block is too small to amortise the kernels' set-up either way.
   */
  val PassEncodedRows: Int = Integer.getInteger("sparkvector.shuffle.reader.passEncodedRows", 128)

  final case class Index(offsets: Array[Long], lengths: Array[Long], rows: Array[Long]) {
    def numPartitions: Int = offsets.length
  }

  /**
   * The block stream's framing (#416): every IPC message is preceded by a unit header of ours --
   * `[UnitMagic:int][kind:int][value:int]` -- so the reader knows what follows without a schema
   * message. `kind` [[UnitDictionary]]: `value` is the string column whose dictionary the next
   * `DictionaryBatch` message carries, current for that column until replaced. `kind`
   * [[UnitRecordBatch]]: `value` is the number of bitmap bytes that follow (`ceil(columns / 8)`),
   * bit `c` set meaning column `c` travels as int32 ids over its current dictionary, clear meaning
   * plain UTF8; then the `RecordBatch` message.
   */
  val UnitMagic: Int = 0x53564231 // "SVB1"
  val UnitDictionary: Int = 1
  val UnitRecordBatch: Int = 2

  def writeUnitHeader(out: java.nio.channels.WritableByteChannel, kind: Int, value: Int): Unit = {
    val b = ByteBuffer.allocate(12).putInt(UnitMagic).putInt(kind).putInt(value)
    b.flip()
    while (b.hasRemaining) out.write(b)
  }

  /** Reads `n` bytes exactly; -1 bytes at a clean end (nothing read), an error on a torn one. */
  def readFully(in: ReadableByteChannel, dst: ByteBuffer): Boolean = {
    val start = dst.position()
    while (dst.hasRemaining) {
      val r = in.read(dst)
      if (r < 0) {
        if (dst.position() == start) return false
        throw new java.io.EOFException(
          s"shuffle stream ended inside a unit header (${dst.position() - start} of ${dst.limit() - start} bytes)"
        )
      }
    }
    true
  }

  /**
   * The map file's dictionary section (#416): after the partition streams, one frame holding a
   * dictionary unit per string column with values, then this trailer -- `[offset:long][length:long]
   * [DictMagic:long]` -- always written, so a reader of a Spark-indexed data file finds the section
   * from the file's end (before the index footer, when there is one).
   */
  val DictMagic: Long = 0x5356444943543131L // "SVDICT11"
  val DictTrailerBytes: Int = 24

  def encodeDictTrailer(offset: Long, length: Long): Array[Byte] =
    ByteBuffer.allocate(DictTrailerBytes).putLong(offset).putLong(length).putLong(DictMagic).array()

  /** The dictionary section `(offset, length)` of a data file whose data (segments + section + trailer) ends at `dataEnd`; `(0, 0)` for a file without one. */
  def readDictSpan(channel: FileChannel, dataEnd: Long): (Long, Long) = {
    if (dataEnd < DictTrailerBytes) return (0L, 0L)
    val buf = ByteBuffer.allocate(DictTrailerBytes)
    channel.read(buf, dataEnd - DictTrailerBytes)
    buf.flip()
    val offset = buf.getLong; val length = buf.getLong; val magic = buf.getLong
    if (magic != DictMagic || offset < 0 || length < 0 || offset + length > dataEnd - DictTrailerBytes) (0L, 0L)
    else (offset, length)
  }

  /** The dictionary span of a Spark-indexed data file (no footer): the trailer sits at the file's end. Cached per path and length. */
  def dictSpanOf(file: java.io.File): (Long, Long) = {
    val key = (file.getPath, file.length())
    var span = dictSpans.get(key)
    if (span == null) {
      val ch = FileChannel.open(file.toPath, StandardOpenOption.READ)
      try span = readDictSpan(ch, ch.size())
      finally ch.close()
      dictSpans.put(key, span)
    }
    span
  }
  private val dictSpans = new java.util.concurrent.ConcurrentHashMap[(String, Long), (Long, Long)]()

  /** A channel over several `(offset, length)` ranges of one file, back to back. */
  final class RangesChannel(file: FileChannel, ranges: Seq[(Long, Long)], closeFile: Boolean = false)
      extends ReadableByteChannel {
    private val live = ranges.filter(_._2 > 0).toIndexedSeq
    private var r = 0
    private var pos = 0L
    override def read(dst: ByteBuffer): Int = {
      while (r < live.length && pos >= live(r)._2) { r += 1; pos = 0L }
      if (r >= live.length) return -1
      val (offset, length) = live(r)
      val remaining = length - pos
      // Bound the read to the range without leaving the caller's limit moved.
      val limit = dst.limit()
      if (dst.remaining() > remaining) dst.limit(dst.position() + remaining.toInt)
      val n =
        try file.read(dst, offset + pos)
        finally dst.limit(limit)
      if (n > 0) pos += n
      n
    }
    override def isOpen: Boolean = file.isOpen
    override def close(): Unit = if (closeFile) file.close()
  }

  /**
   * A map output's block(s) for a reduce range as the reader must see them: the map file's dictionary
   * section first, then the partition range -- for a file segment buffer; any other buffer is read
   * as it is (a self-contained stream). An empty range needs no dictionary and reads as empty.
   */
  def blockChannel(buf: org.apache.spark.network.buffer.ManagedBuffer): ReadableByteChannel = buf match {
    case f: org.apache.spark.network.buffer.FileSegmentManagedBuffer =>
      val file = FileChannel.open(f.getFile.toPath, StandardOpenOption.READ)
      if (f.getLength == 0) new RangesChannel(file, Nil, closeFile = true)
      else new RangesChannel(file, Seq(dictSpanOf(f.getFile), (f.getOffset, f.getLength)), closeFile = true)
    case other => java.nio.channels.Channels.newChannel(other.createInputStream())
  }

  /** The same as an input stream (the Flight producer copies bytes). */
  def blockStream(buf: org.apache.spark.network.buffer.ManagedBuffer): java.io.InputStream = buf match {
    case f: org.apache.spark.network.buffer.FileSegmentManagedBuffer if f.getLength > 0 =>
      java.nio.channels.Channels.newInputStream(blockChannel(f))
    case other => other.createInputStream()
  }

  /**
   * The bytes a transport delivers for partitions `[start, end)` of a footer file (tests, tools): the
   * dictionary section, then the partitions' consecutive streams; empty when they hold no bytes.
   */
  def blockBytes(path: Path, start: Int, end: Int): Array[Byte] = {
    val ch = FileChannel.open(path, StandardOpenOption.READ)
    try {
      val index = readIndex(ch)
      val size = ch.size()
      val tail = ByteBuffer.allocate(12)
      ch.read(tail, size - 12)
      tail.flip()
      val (dOff, dLen) = readDictSpan(ch, size - 12 - tail.getInt)
      val from = index.offsets(start)
      val to = index.offsets(end - 1) + index.lengths(end - 1)
      if (to == from) return Array.emptyByteArray
      val out = ByteBuffer.allocate((dLen + (to - from)).toInt)
      // Each range into its own slice: a read bounded by the buffer alone would run past the section.
      out.limit(dLen.toInt)
      while (out.hasRemaining) require(ch.read(out, dOff + out.position()) > 0, "torn dictionary section")
      out.limit(out.capacity())
      while (out.hasRemaining) require(ch.read(out, from + out.position() - dLen) > 0, "torn partition range")
      out.array()
    } finally ch.close()
  }

  def arrowType(dt: DataType): ArrowType = dt match {
    case IntegerType | ByteType | ShortType => new ArrowType.Int(32, true) // narrow ints ride INT32 lanes (#327)
    case DateType => new ArrowType.Date(DateUnit.DAY)
    case LongType => new ArrowType.Int(64, true)
    case TimestampType => new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC")
    case DoubleType => new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    case BooleanType => ArrowType.Bool.INSTANCE
    case StringType => new ArrowType.Int(32, true) // the indices of the dictionary encoding
    case d: DecimalType if d.precision <= TypeMapping.MAX_DECIMAL_PRECISION => new ArrowType.Int(64, true)
    case d: DecimalType => new ArrowType.Decimal(d.precision, d.scale, 128)
    case other => throw new IllegalArgumentException(s"no shuffle lane for $other")
  }

  def dictionaryEncoding(ordinal: Int): DictionaryEncoding =
    new DictionaryEncoding(ordinal + 1L, false, new ArrowType.Int(32, true))

  /**
   * The field of one column. A string column is dictionary-encoded (int32 ids, the dictionary with
   * id `ordinal + 1`) or, with `dictionary = false`, plain UTF8: the writer decides per record batch
   * whether the dictionary pays (#356), and since every record batch is its own stream (#340) the
   * two encodings may alternate within one partition's bytes -- the reader takes either.
   */
  def arrowField(name: String, dt: DataType, ordinal: Int, dictionary: Boolean = true): Field = {
    val metadata = new JHashMap[String, String]()
    metadata.put(TypeKey, dt.json)
    val (arrow, encoding) =
      if (dt == StringType && dictionary) (arrowType(dt), dictionaryEncoding(ordinal))
      else if (dt == StringType) (ArrowType.Utf8.INSTANCE, null)
      else (arrowType(dt), null)
    new Field(name, new FieldType(true, arrow, encoding, metadata), null)
  }

  def arrowSchema(schema: StructType): Schema =
    new Schema(schema.fields.zipWithIndex.map { case (f, i) => arrowField(f.name, f.dataType, i) }.toSeq.asJava)

  /**
   * The Spark type carried in a field's metadata. Parsed once per distinct JSON: the reader asks for
   * every column of every block, and a block per (map, partition) at 1000 partitions made the JSON
   * parse 3% of an executor's self time (#411).
   */
  def sparkType(field: Field): DataType = {
    val json = field.getMetadata.get(TypeKey)
    var dt = sparkTypeCache.get(json)
    if (dt == null) {
      dt = DataType.fromJson(json)
      sparkTypeCache.put(json, dt)
    }
    dt
  }
  private val sparkTypeCache = new java.util.concurrent.ConcurrentHashMap[String, DataType]()

  /** `[n:int][offsets:long*n][lengths:long*n][rows:long*n][footerLength:int][magic:long]`. */
  def encodeIndex(index: Index): Array[Byte] = {
    val n = index.numPartitions
    val body = 4 + 3 * 8 * n
    val buf = ByteBuffer.allocate(body + 4 + 8)
    buf.putInt(n)
    index.offsets.foreach(buf.putLong)
    index.lengths.foreach(buf.putLong)
    index.rows.foreach(buf.putLong)
    buf.putInt(body)
    buf.putLong(Magic)
    buf.array()
  }

  def readIndex(channel: FileChannel): Index = {
    val size = channel.size()
    val tail = ByteBuffer.allocate(12)
    channel.read(tail, size - 12)
    tail.flip()
    val body = tail.getInt
    val magic = tail.getLong
    require(magic == Magic, s"not a partitioned IPC file (magic $magic)")
    val buf = ByteBuffer.allocate(body)
    channel.read(buf, size - 12 - body)
    buf.flip()
    val n = buf.getInt
    val offsets = Array.fill(n)(buf.getLong)
    val lengths = Array.fill(n)(buf.getLong)
    val rows = Array.fill(n)(buf.getLong)
    Index(offsets, lengths, rows)
  }

  /** A channel over `[offset, offset + length)` of a file. */
  /** `length` bytes of `file` from `offset`, read with positional reads straight into the caller's buffers; closes the file. */
  final class RangeChannel(file: FileChannel, offset: Long, length: Long) extends ReadableByteChannel {
    private var pos = 0L
    override def read(dst: ByteBuffer): Int = {
      if (pos >= length) return -1
      val remaining = length - pos
      if (dst.remaining() > remaining) dst.limit(dst.position() + remaining.toInt)
      val n = file.read(dst, offset + pos)
      if (n > 0) pos += n
      n
    }
    override def isOpen: Boolean = file.isOpen
    override def close(): Unit = {}
  }

  /**
   * Reads one partition's stream back as the column vectors the operators produce:
   * `VectorDictionaryColumnVector` for strings, `VectorDecimalColumnVector` for small decimals,
   * `VectorArrowColumnVector` otherwise. Each batch owns its memory (the reader's root is reused, so
   * its buffers are transferred out; the dictionary, which the reader keeps for later batches, is
   * copied -- it is small by construction).
   */
  final class PartitionReader(
      path: Path,
      partition: Int,
      allocator: BufferAllocator,
      schema: StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType] =
        Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD)
  ) extends Iterator[ColumnarBatch] with AutoCloseable {
    private val file = FileChannel.open(path, StandardOpenOption.READ)
    private val index = readIndex(file)

    /** The dictionary section sits before the index footer (#416): the data ends where the footer's body begins. */
    private val dictSpan: (Long, Long) = {
      val size = file.size()
      val tail = ByteBuffer.allocate(12)
      file.read(tail, size - 12)
      tail.flip()
      readDictSpan(file, size - 12 - tail.getInt)
    }
    private val inner =
      if (index.lengths(partition) == 0) null
      else new StreamReader(
        new RangesChannel(file, Seq(dictSpan, (index.offsets(partition), index.lengths(partition)))),
        allocator,
        schema,
        compression
      )

    def rows: Long = index.rows(partition)
    override def hasNext: Boolean = inner != null && inner.hasNext
    override def next(): ColumnarBatch = { if (inner == null) throw new NoSuchElementException; inner.next() }
    override def close(): Unit = {
      if (inner != null) inner.close()
      file.close()
    }
  }

  /**
   * A loaded root (an IPC reader's or a Flight stream's, both reused across batches) as a batch that
   * owns its memory: the vectors are transferred out into `allocator`, the dictionary -- kept by the
   * reader for later batches -- is copied (small by construction), and the wrappers are the operators'
   * own column vector classes.
   */
  /** The Spark type of every field, parsed once per stream (`toBatch` takes it: parsing the JSON per batch showed in profiles). */
  def sparkTypes(root: org.apache.arrow.vector.VectorSchemaRoot): Array[DataType] =
    root.getSchema.getFields.asScala.map(sparkType).toArray

  def toBatch(
      root: org.apache.arrow.vector.VectorSchemaRoot,
      dictionary: Long => VarCharVector,
      allocator: BufferAllocator,
      types: Array[DataType] = null
  ): ColumnarBatch = {
    val n = root.getRowCount
    val fields = root.getSchema.getFields
    val columns = new Array[ColumnVector](fields.size())
    var c = 0
    while (c < columns.length) {
      val field = fields.get(c)
      val dt = if (types != null) types(c) else sparkType(field)
      val source = root.getVector(c)
      val moved = source.getField.createVector(allocator)
      source.makeTransferPair(moved).transfer()
      columns(c) = dt match {
        case StringType if field.getDictionary != null =>
          val dict = dictionary(field.getDictionary.getId)
          val copy = new VarCharVector(field.getName + ".dictionary", allocator)
          dict.makeTransferPair(copy).splitAndTransfer(0, dict.getValueCount)
          new VectorDictionaryColumnVector(moved.asInstanceOf[IntVector], copy)
        case StringType =>
          new VectorArrowColumnVector(moved) // plain UTF8: the writer found no dictionary worth sending
        case d: DecimalType if d.precision <= TypeMapping.MAX_DECIMAL_PRECISION =>
          new VectorDecimalColumnVector(moved.asInstanceOf[org.apache.arrow.vector.BigIntVector], d)
        case ByteType | ShortType => new VectorNarrowIntColumnVector(moved.asInstanceOf[IntVector], dt) // #327
        case _ => new VectorArrowColumnVector(moved)
      }
      c += 1
    }
    new ColumnarBatch(columns, n)
  }

  /**
   * A partition range's bytes (a file range, a fetched block, a Flight stream) read back as the
   * operators' column vectors. The bytes are IPC messages without a schema message or an
   * end-of-stream marker (#411): the shuffle's schema comes from the dependency, and a string column
   * is dictionary-encoded in a record batch exactly when a dictionary batch with its id precedes it.
   * The reader's roots -- one per batch shape, which columns are encoded -- and its dictionary vectors
   * are created once and reused across every batch of every map output in the range; a batch costs
   * its record-batch message and nothing else. Every batch handed out owns its memory: the vectors
   * are transferred out of the root, and an encoded column's dictionary, loaded for that batch alone,
   * is transferred with it.
   */
  final class StreamReader(
      channel: ReadableByteChannel,
      allocator: BufferAllocator,
      schema: StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType] =
        Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD)
  ) extends Iterator[ColumnarBatch] with AutoCloseable {

    /** The block's bytes decompressed as one stream (several frames back to back read as one), then read as IPC messages. */
    private val input: ReadableByteChannel = compression match {
      case None => channel
      case codec => java.nio.channels.Channels.newChannel(io.vecruntime.shuffle.ShuffleCompression.decompressing(
          java.nio.channels.Channels.newInputStream(channel),
          codec
        ))
    }
    private val messages = new org.apache.arrow.vector.ipc.message.MessageChannelReader(
      new org.apache.arrow.vector.ipc.ReadChannel(input),
      allocator
    )
    private val fields: Array[Field] = arrowSchema(schema).getFields.asScala.toArray
    private val plain: Array[Field] = Array.tabulate(fields.length) { c =>
      if (schema.fields(c).dataType == StringType) arrowField(schema.fields(c).name, StringType, c, dictionary = false)
      else null
    }
    private val types: Array[DataType] = schema.fields.map(_.dataType)

    /** Column ordinal of a dictionary id (#dictionaryEncoding: id = ordinal + 1), or -1. */
    private def columnOf(id: Long): Int =
      if (id >= 1 && id <= fields.length && plain((id - 1).toInt) != null) (id - 1).toInt else -1

    /**
     * The current dictionary of each string column (#416): loaded by a dictionary unit -- once per map
     * output in the file-dictionary format, once per block in the per-block one -- and current until
     * the next unit for that column replaces it. Shared by reference count between the reader and every
     * batch handed out over it, so a replacement does not disturb a batch still in the consumer's hands
     * and a large per-map dictionary is never copied per batch.
     */
    private val dictionaries: Array[SharedDictionary] = new Array[SharedDictionary](fields.length)
    private def dictionary(c: Int): SharedDictionary = {
      val d = dictionaries(c)
      require(
        d != null,
        s"shuffle stream: column ${fields(c).getName} is dictionary-encoded but no dictionary preceded it"
      )
      d
    }
    private final class SharedDictionary(val vector: VarCharVector) {
      private var refs = 1
      def retain(): VarCharVector = { refs += 1; vector }
      def release(): Unit = { refs -= 1; if (refs == 0) vector.close() }

      /**
       * The dictionary's offsets and bytes as heap arrays, built on first use (#416, item 6): the reader's
       * decode of a small encoded block reads them per row, and a heap read is a plain load where a
       * `MemorySegment` read carries a session and alignment check the JIT did not hoist -- a quarter of
       * a 1000-partition reduce task's samples on q67 sat in those checks under `appendEncoded`.
       */
      var starts: Array[Int] = _
      var bytes: Array[Byte] = _
      def heap(): Unit = if (starts == null) {
        val buffers = ArrowVectorBuffers.forRead(vector)
        // Bounded by the offsets buffer present, not the value count alone: a dictionary buffer can be
        // sized to the entries the stream references, and the original per-row reads never went past it.
        // An empty dictionary (a string column null in every row of the map task, #447) arrives with no
        // offsets buffer at all: no entries, every id null.
        val n = math.max(0, math.min(vector.getValueCount, (buffers.offsets().byteSize() >> 2).toInt - 1))
        starts = new Array[Int](n + 1)
        if (n > 0) java.lang.foreign.MemorySegment.copy(buffers.offsets(), VectorBuffers.LE_INT, 0L, starts, 0, n + 1)
        // Likewise the bytes: the data buffer carries the entries the stream references, which can end
        // before the last offset (#345's slices); a referenced entry is always within it.
        val dataLen = math.min(starts(n).toLong, buffers.data().byteSize()).toInt
        bytes = new Array[Byte](dataLen)
        if (dataLen > 0) java.lang.foreign.MemorySegment.copy(
          buffers.data(),
          java.lang.foreign.ValueLayout.JAVA_BYTE,
          0L,
          bytes,
          0,
          dataLen
        )
      }
    }

    /** The shape of the record batch being read: bit `c` set = column `c` as ids over its dictionary. */
    private val encoded = new java.util.BitSet(fields.length)
    private val unitHeader = ByteBuffer.allocate(12)
    private var shapeBytes = new Array[Byte](math.max(1, (fields.length + 7) / 8))

    /** One root and loader per batch shape, keyed by the encoded-columns set. */
    private val roots =
      new JHashMap[java.util.BitSet, (org.apache.arrow.vector.VectorSchemaRoot, org.apache.arrow.vector.VectorLoader)]()
    private val factory = io.vecruntime.shuffle.ShuffleCompression.Factory
    private var nextBatch: ColumnarBatch = _

    /** A full-size batch that arrived while small ones were pending: handed out right after them. */
    private var held: ColumnarBatch = _

    /** The batch last handed out: the consumers do not close their input, so it is closed when the next one is produced (or at close). */
    private var last: ColumnarBatch = _
    private var done = false

    /** Small plain batches accumulated until `CoalesceRows` (all-plain shape, so every string column is UTF8). */
    private val pending: Array[Pending] = Array.tabulate(fields.length)(c => new Pending(c))
    private var pendingRows = 0

    /** Heap scratch for `Pending.appendEncoded` (one block at a time, so shared by the columns). */
    private var ids = new Array[Int](CoalesceRows)
    private var validBytes = new Array[Byte](CoalesceRows / 8 + 1)
    private var outBytes = new Array[Byte](32 * CoalesceRows)
    private var outOffsets = new Array[Int](CoalesceRows)

    /**
     * One accumulating column of the pending batch (#416): each small batch is appended as a bulk copy
     * of its buffers -- data, offsets rebased, validity bits -- the way the writer's staging copies
     * its input. Arrow's `VectorSchemaRootAppender`, used first, copies value by value: at 1 TB and
     * 1000 partitions a reduce task reading a thousand 50-row blocks of 18 columns spent ~1 ms per
     * block in it, more than decoding the block cost.
     */
    private final class Pending(c: Int) {
      private val dt = types(c)
      private val isString = dt == StringType
      private val isBool = dt == BooleanType
      private val width: Int = if (isString || isBool) 0 else PartitionedIpcWriter.byteWidth(dt)
      private val field: Field = if (plain(c) != null) plain(c) else fields(c)
      var vector: FieldVector = _
      private var buffers: ArrowVectorBuffers = _
      private var rows = 0
      private var dataBytes = 0L

      private def ensure(n: Int, bytes: Long): Unit = {
        if (vector == null) {
          val cap = math.max(CoalesceRows, n)
          vector = field.createVector(allocator)
          vector match {
            case v: VarCharVector => v.allocateNew(math.max(bytes, 32L * cap), cap)
            case v => v.setInitialCapacity(cap); v.allocateNew()
          }
          buffers = ArrowVectorBuffers.forWrite(vector, vector.getValueCapacity, dt)
          if (isString) buffers.offsets().set(VectorBuffers.LE_INT, 0L, 0)
          return
        }
        var grown = false
        while (vector.getValueCapacity < rows + n) { vector.reAlloc(); grown = true }
        if (isString) {
          val v = vector.asInstanceOf[VarCharVector]
          while (v.getDataBuffer.capacity() < dataBytes + bytes) { v.reallocDataBuffer(); grown = true }
        }
        if (grown) buffers = ArrowVectorBuffers.forWrite(vector, vector.getValueCapacity, dt)
      }

      /** Appends all `n` rows of `src` (a vector of the loader's root, plain). */
      def append(src: FieldVector, n: Int): Unit = {
        val in = ArrowVectorBuffers.forRead(src)
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
          if (bytes > 0) java.lang.foreign.MemorySegment.copy(in.data(), start.toLong, buffers.data(), dataBytes, bytes)
        } else if (isBool) {
          Bitmap.copyBits(in.data(), buffers.data(), rows, n)
        } else {
          java.lang.foreign.MemorySegment.copy(in.data(), 0L, buffers.data(), rows.toLong * width, n.toLong * width)
        }
        if (in.validity() != null) Bitmap.copyBits(in.validity(), buffers.validity(), rows, n)
        else Bitmap.fillRange(buffers.validity(), rows, n, true)
        rows += n
        dataBytes += bytes
      }

      /**
       * Appends all `n` rows of a dictionary-encoded string column (`indices` over `dictionary`), decoded
       * into the pending plain column: a block of a few hundred rows that the writer still encoded (#356's
       * ratio) joins the coalesced batch like a plain one instead of reaching the operators on its own.
       */
      def appendEncoded(indices: IntVector, dictionary: SharedDictionary, n: Int): Unit = {
        dictionary.heap()
        val dStarts = dictionary.starts
        val dBytes = dictionary.bytes
        val idx = ArrowVectorBuffers.forRead(indices)
        // The block's ids and validity words on the heap: one bulk copy each, then plain array reads.
        if (ids.length < n) ids = new Array[Int](math.max(n, ids.length * 2))
        java.lang.foreign.MemorySegment.copy(idx.data(), VectorBuffers.LE_INT, 0L, ids, 0, n)
        val valid = idx.validity()
        val validLen = (n + 7) >>> 3
        if (valid != null) {
          if (validBytes.length < validLen) validBytes = new Array[Byte](math.max(validLen, validBytes.length * 2))
          java.lang.foreign.MemorySegment.copy(
            valid,
            java.lang.foreign.ValueLayout.JAVA_BYTE,
            0L,
            validBytes,
            0,
            math.min(validLen.toLong, valid.byteSize()).toInt
          )
        }
        var bytes = 0L
        var i = 0
        while (i < n) {
          if (valid == null || ((validBytes(i >>> 3) >>> (i & 7)) & 1) != 0) {
            val e = ids(i)
            bytes += dStarts(e + 1) - dStarts(e)
          }
          i += 1
        }
        ensure(n, bytes)
        // The block's strings assembled in heap scratch, then written with one copy each for bytes and offsets.
        if (outBytes.length < bytes) outBytes = new Array[Byte](math.max(bytes.toInt, outBytes.length * 2))
        if (outOffsets.length < n) outOffsets = new Array[Int](math.max(n, outOffsets.length * 2))
        var out = 0
        val hasNulls = valid != null
        i = 0
        while (i < n) {
          if (!hasNulls || ((validBytes(i >>> 3) >>> (i & 7)) & 1) != 0) {
            val e = ids(i)
            val start = dStarts(e)
            val len = dStarts(e + 1) - start
            System.arraycopy(dBytes, start, outBytes, out, len)
            out += len
          } else Bitmap.setTo(buffers.validity(), rows + i, false)
          outOffsets(i) = dataBytes.toInt + out
          i += 1
        }
        if (out > 0) java.lang.foreign.MemorySegment.copy(
          outBytes,
          0,
          buffers.data(),
          java.lang.foreign.ValueLayout.JAVA_BYTE,
          dataBytes,
          out
        )
        java.lang.foreign.MemorySegment.copy(
          outOffsets,
          0,
          buffers.offsets(),
          VectorBuffers.LE_INT,
          (rows + 1).toLong << 2,
          n
        )
        if (hasNulls) {
          // Valid bits set per row only where the block has nulls; otherwise the range is filled at once.
          i = 0
          while (i < n) {
            if (((validBytes(i >>> 3) >>> (i & 7)) & 1) != 0) Bitmap.setTo(buffers.validity(), rows + i, true); i += 1
          }
        } else Bitmap.fillRange(buffers.validity(), rows, n, true)
        rows += n
        dataBytes += out
      }

      /** The accumulated vector (value count set), the column emptied for the next accumulation. */
      def take(): FieldVector = {
        val v = vector
        v match {
          case vw: VarCharVector => vw.setLastSet(rows - 1); vw.setValueCount(rows)
          case other => other.setValueCount(rows)
        }
        vector = null; buffers = null; rows = 0; dataBytes = 0L
        v
      }

      def close(): Unit = if (vector != null) { vector.close(); vector = null }
    }

    private def append(root: org.apache.arrow.vector.VectorSchemaRoot): Unit = {
      val n = root.getRowCount
      var c = 0
      while (c < fields.length) {
        if (encoded.get(c)) pending(c).appendEncoded(root.getVector(c).asInstanceOf[IntVector], dictionary(c), n)
        else pending(c).append(root.getVector(c), n)
        c += 1
      }
      pendingRows += n
      root.clear()
    }

    private def takePending(): ColumnarBatch = {
      val columns = new Array[ColumnVector](fields.length)
      var c = 0
      while (c < columns.length) { columns(c) = wrap(pending(c).take(), c, dictionaryEncoded = false); c += 1 }
      val b = new ColumnarBatch(columns, pendingRows)
      pendingRows = 0
      b
    }

    private def rootFor(shape: java.util.BitSet)
        : (org.apache.arrow.vector.VectorSchemaRoot, org.apache.arrow.vector.VectorLoader) = {
      var r = roots.get(shape)
      if (r == null) {
        val fs = new java.util.ArrayList[Field](fields.length)
        var c = 0
        while (c < fields.length) { fs.add(if (plain(c) != null && !shape.get(c)) plain(c) else fields(c)); c += 1 }
        val root = org.apache.arrow.vector.VectorSchemaRoot.create(new Schema(fs), allocator)
        r = (root, new org.apache.arrow.vector.VectorLoader(root, factory))
        roots.put(shape.clone().asInstanceOf[java.util.BitSet], r)
      }
      r
    }

    private def advance(): Unit = while (!done && nextBatch == null) {
      if (held != null) { nextBatch = held; held = null; return }
      // Our unit header first (#416): what the next IPC message is and, for a record batch, its shape.
      unitHeader.clear()
      if (!readFully(input, unitHeader)) {
        done = true
        if (pendingRows > 0) nextBatch = takePending()
        return
      }
      unitHeader.flip()
      val magic = unitHeader.getInt
      require(magic == UnitMagic, f"shuffle stream: bad unit header 0x$magic%08x (a writer of another format?)")
      val kind = unitHeader.getInt
      val value = unitHeader.getInt
      kind match {
        case UnitDictionary =>
          val c = value
          require(
            c >= 0 && c < fields.length && plain(c) != null,
            s"shuffle stream: dictionary unit for column $c, which is not a string column"
          )
          val result = messages.readNext()
          require(
            result != null && result.getMessage.headerType() == org.apache.arrow.flatbuf.MessageHeader.DictionaryBatch,
            s"shuffle stream: a dictionary unit must be followed by a DictionaryBatch message"
          )
          val body = if (result.getBodyBuffer == null) allocator.getEmpty else result.getBodyBuffer
          val batch =
            org.apache.arrow.vector.ipc.message.MessageSerializer.deserializeDictionaryBatch(result.getMessage, body)
          try {
            require(
              columnOf(batch.getDictionaryId) == c,
              s"shuffle stream: dictionary ${batch.getDictionaryId} in the unit of column $c"
            )
            // A fresh vector per dictionary: batches handed out over the previous one keep it until they close.
            val vector = new VarCharVector(fields(c).getName + ".dictionary", allocator)
            new org.apache.arrow.vector.VectorLoader(
              new org.apache.arrow.vector.VectorSchemaRoot(
                java.util.List.of(vector.getField),
                java.util.List.of[FieldVector](vector),
                0
              ),
              factory
            )
              .load(batch.getDictionary)
            if (dictionaries(c) != null) dictionaries(c).release()
            dictionaries(c) = new SharedDictionary(vector)
          } finally batch.close()
        case UnitRecordBatch =>
          val nbytes = value
          require(
            nbytes == shapeBytes.length,
            s"shuffle stream: shape bitmap of $nbytes bytes for ${fields.length} columns"
          )
          val shape = ByteBuffer.wrap(shapeBytes)
          require(readFully(input, shape), "shuffle stream ended inside a record batch's shape")
          encoded.clear()
          var c = 0
          while (c < fields.length) { if ((shapeBytes(c >> 3) & (1 << (c & 7))) != 0) encoded.set(c); c += 1 }
          val result = messages.readNext()
          require(
            result != null && result.getMessage.headerType() == org.apache.arrow.flatbuf.MessageHeader.RecordBatch,
            s"shuffle stream: a record batch unit must be followed by a RecordBatch message"
          )
          val body = if (result.getBodyBuffer == null) allocator.getEmpty else result.getBodyBuffer
          val batch =
            org.apache.arrow.vector.ipc.message.MessageSerializer.deserializeRecordBatch(result.getMessage, body)
          try {
            val (root, loader) = rootFor(encoded)
            loader.load(batch)
            if (root.getRowCount < CoalesceRows && (encoded.isEmpty || root.getRowCount < PassEncodedRows)) {
              // A small batch (a block of a few dozen rows at 1000 partitions) is appended to the
              // pending batch instead of reaching the operators on its own: their per-batch costs --
              // kernel set-up, a hash table's probe round, an output batch per input batch -- were
              // most of a reduce task's time over ten-row blocks (#411). A dictionary-encoded column
              // of a tiny block is decoded into the pending plain column (#416); an encoded block of
              // `PassEncodedRows` or more goes through as ids, the shape the aggregate is fastest on.
              append(root)
              if (pendingRows >= CoalesceRows) nextBatch = takePending()
            } else {
              if (pendingRows > 0) { nextBatch = takePending(); held = take(root) }
              else nextBatch = take(root)
            }
          } finally batch.close()
        case other =>
          throw new IllegalStateException(s"shuffle stream: unexpected unit kind $other")
      }
    }

    /** The loaded root as a batch owning its memory (the root is reused for the next batch). */
    private def take(root: org.apache.arrow.vector.VectorSchemaRoot): ColumnarBatch = {
      val n = root.getRowCount
      val columns = new Array[ColumnVector](fields.length)
      var c = 0
      while (c < columns.length) {
        val source = root.getVector(c)
        val moved = source.getField.createVector(allocator)
        source.makeTransferPair(moved).transfer()
        columns(c) = wrap(moved, c, encoded.get(c))
        c += 1
      }
      new ColumnarBatch(columns, n)
    }

    /** The Spark column over a vector of ours (moved out of a root or taken from the pending batch). */
    private def wrap(moved: FieldVector, c: Int, dictionaryEncoded: Boolean): ColumnVector = types(c) match {
      case StringType if dictionaryEncoded =>
        // The batch shares the column's current dictionary (#416): one reference, released when the batch closes.
        val shared = dictionary(c)
        new VectorDictionaryColumnVector(moved.asInstanceOf[IntVector], shared.retain(), () => shared.release())
      case StringType => new VectorArrowColumnVector(moved) // plain UTF8: the writer found no dictionary worth sending
      case d: DecimalType if d.precision <= TypeMapping.MAX_DECIMAL_PRECISION =>
        new VectorDecimalColumnVector(moved.asInstanceOf[org.apache.arrow.vector.BigIntVector], d)
      case ByteType | ShortType => new VectorNarrowIntColumnVector(moved.asInstanceOf[IntVector], types(c)) // #327
      case _ => new VectorArrowColumnVector(moved)
    }

    override def hasNext: Boolean = { advance(); nextBatch != null }

    override def next(): ColumnarBatch = {
      if (!hasNext) throw new NoSuchElementException
      val b = nextBatch
      nextBatch = null
      if (last != null) last.close()
      last = b
      b
    }

    override def close(): Unit = {
      if (nextBatch != null) { nextBatch.close(); nextBatch = null }
      if (last != null) { last.close(); last = null }
      if (held != null) { held.close(); held = null }
      roots.values().forEach(r => r._1.close())
      roots.clear()
      pending.foreach(_.close())
      var c = 0
      while (c < dictionaries.length) {
        if (dictionaries(c) != null) { dictionaries(c).release(); dictionaries(c) = null }; c += 1
      }
      messages.close()
    }
  }
}
