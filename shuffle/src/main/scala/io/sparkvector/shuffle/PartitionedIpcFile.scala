package io.sparkvector.shuffle

import java.nio.ByteBuffer
import java.nio.channels.{FileChannel, ReadableByteChannel}
import java.nio.file.{Path, StandardOpenOption}
import java.util.{HashMap => JHashMap}
import scala.jdk.CollectionConverters._

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{VectorArrowColumnVector, VectorDecimalColumnVector, VectorDictionaryColumnVector, VectorNarrowIntColumnVector}
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
  /** Rows a reader accumulates small plain batches up to before handing a batch to the operators (#411). */
  val CoalesceRows: Int = 1024

  final case class Index(offsets: Array[Long], lengths: Array[Long], rows: Array[Long]) {
    def numPartitions: Int = offsets.length
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
  final class PartitionReader(path: Path, partition: Int, allocator: BufferAllocator, schema: StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType] = Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD))
    extends Iterator[ColumnarBatch] with AutoCloseable {
    private val file = FileChannel.open(path, StandardOpenOption.READ)
    private val index = readIndex(file)
    private val inner =
      if (index.lengths(partition) == 0) null
      else new StreamReader(new RangeChannel(file, index.offsets(partition), index.lengths(partition)), allocator, schema, compression)

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

  def toBatch(root: org.apache.arrow.vector.VectorSchemaRoot, dictionary: Long => VarCharVector, allocator: BufferAllocator,
      types: Array[DataType] = null): ColumnarBatch = {
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
        case StringType => new VectorArrowColumnVector(moved) // plain UTF8: the writer found no dictionary worth sending
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
  final class StreamReader(channel: ReadableByteChannel, allocator: BufferAllocator, schema: StructType,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType] = Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD))
    extends Iterator[ColumnarBatch] with AutoCloseable {
    /** The block's bytes decompressed as one stream (several frames back to back read as one), then read as IPC messages. */
    private val input: ReadableByteChannel = compression match {
      case None => channel
      case codec => java.nio.channels.Channels.newChannel(io.sparkvector.shuffle.ShuffleCompression.decompressing(java.nio.channels.Channels.newInputStream(channel), codec))
    }
    private val messages = new org.apache.arrow.vector.ipc.message.MessageChannelReader(new org.apache.arrow.vector.ipc.ReadChannel(input), allocator)
    private val fields: Array[Field] = arrowSchema(schema).getFields.asScala.toArray
    private val plain: Array[Field] = Array.tabulate(fields.length) { c =>
      if (schema.fields(c).dataType == StringType) arrowField(schema.fields(c).name, StringType, c, dictionary = false) else null
    }
    private val types: Array[DataType] = schema.fields.map(_.dataType)
    /** Column ordinal of a dictionary id (#dictionaryEncoding: id = ordinal + 1), or -1. */
    private def columnOf(id: Long): Int = if (id >= 1 && id <= fields.length && plain((id - 1).toInt) != null) (id - 1).toInt else -1
    /** The dictionary vector of each string column, loaded by the batch's dictionary message. */
    private val dictionaries: Array[VarCharVector] = Array.tabulate(fields.length)(c => if (plain(c) != null) new VarCharVector(fields(c).getName + ".dictionary", allocator) else null)
    private val encoded = new java.util.BitSet(fields.length)
    /** One root and loader per batch shape, keyed by the encoded-columns set. */
    private val roots = new JHashMap[java.util.BitSet, (org.apache.arrow.vector.VectorSchemaRoot, org.apache.arrow.vector.VectorLoader)]()
    private val factory = io.sparkvector.shuffle.ShuffleCompression.Factory
    private var nextBatch: ColumnarBatch = _
    /** A full-size batch that arrived while small ones were pending: handed out right after them. */
    private var held: ColumnarBatch = _
    /** The batch last handed out: the consumers do not close their input, so it is closed when the next one is produced (or at close). */
    private var last: ColumnarBatch = _
    private var done = false
    /** Small plain batches accumulated until `CoalesceRows` (all-plain shape, so every string column is UTF8). */
    private var pending: org.apache.arrow.vector.VectorSchemaRoot = _

    private def append(root: org.apache.arrow.vector.VectorSchemaRoot): Unit = {
      if (pending == null) {
        val fs = new java.util.ArrayList[Field](fields.length)
        var c = 0
        while (c < fields.length) { fs.add(if (plain(c) != null) plain(c) else fields(c)); c += 1 }
        pending = org.apache.arrow.vector.VectorSchemaRoot.create(new Schema(fs), allocator)
        pending.allocateNew() // the appender reads the target's buffers: they must exist, empty
        pending.setRowCount(0)
      }
      org.apache.arrow.vector.util.VectorSchemaRootAppender.append(false, pending, root)
      root.clear()
    }

    private def takePending(): ColumnarBatch = {
      val b = take(pending) // the vectors moved out; fresh empty buffers for the next accumulation
      pending.allocateNew()
      pending.setRowCount(0)
      b
    }

    private def rootFor(shape: java.util.BitSet): (org.apache.arrow.vector.VectorSchemaRoot, org.apache.arrow.vector.VectorLoader) = {
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
      val result = messages.readNext()
      if (result == null) {
        done = true
        if (pending != null && pending.getRowCount > 0) nextBatch = takePending()
      } else {
        val message = result.getMessage
        // A message with no body (an empty dictionary, a batch of zero-length buffers) carries a null buffer.
        val body = if (result.getBodyBuffer == null) allocator.getEmpty else result.getBodyBuffer
        message.headerType() match {
          case org.apache.arrow.flatbuf.MessageHeader.DictionaryBatch =>
            val batch = org.apache.arrow.vector.ipc.message.MessageSerializer.deserializeDictionaryBatch(message, body)
            try {
              val c = columnOf(batch.getDictionaryId)
              require(c >= 0, s"shuffle stream: dictionary ${batch.getDictionaryId} matches no string column")
              val vector = dictionaries(c)
              new org.apache.arrow.vector.VectorLoader(
                new org.apache.arrow.vector.VectorSchemaRoot(java.util.List.of(vector.getField), java.util.List.of[FieldVector](vector), 0), factory)
                .load(batch.getDictionary)
              encoded.set(c)
            } finally batch.close()
          case org.apache.arrow.flatbuf.MessageHeader.RecordBatch =>
            val batch = org.apache.arrow.vector.ipc.message.MessageSerializer.deserializeRecordBatch(message, body)
            try {
              val (root, loader) = rootFor(encoded)
              loader.load(batch)
              if (root.getRowCount < CoalesceRows && encoded.isEmpty) {
                // A small, plain batch (a block of a few dozen rows at 1000 partitions) is appended to
                // the pending batch instead of reaching the operators on its own: their per-batch
                // costs -- kernel set-up, a hash table's probe round, an output batch per input batch
                // -- were most of a reduce task's time over ten-row blocks (#411).
                append(root)
                if (pending.getRowCount >= CoalesceRows) nextBatch = takePending()
              } else {
                if (pending != null && pending.getRowCount > 0) { nextBatch = takePending(); held = take(root) }
                else nextBatch = take(root)
              }
            } finally {
              batch.close()
              encoded.clear()
            }
          case other =>
            throw new IllegalStateException(s"shuffle stream: unexpected IPC message type $other")
        }
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
        columns(c) = types(c) match {
          case StringType if (root ne pending) && encoded.get(c) =>
            val dict = dictionaries(c)
            val movedDict = new VarCharVector(dict.getName, allocator)
            dict.makeTransferPair(movedDict).transfer()
            new VectorDictionaryColumnVector(moved.asInstanceOf[IntVector], movedDict)
          case StringType => new VectorArrowColumnVector(moved) // plain UTF8: the writer found no dictionary worth sending
          case d: DecimalType if d.precision <= TypeMapping.MAX_DECIMAL_PRECISION =>
            new VectorDecimalColumnVector(moved.asInstanceOf[org.apache.arrow.vector.BigIntVector], d)
          case ByteType | ShortType => new VectorNarrowIntColumnVector(moved.asInstanceOf[IntVector], types(c)) // #327
          case _ => new VectorArrowColumnVector(moved)
        }
        c += 1
      }
      new ColumnarBatch(columns, n)
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
      if (pending != null) { pending.close(); pending = null }
      dictionaries.foreach(d => if (d != null) d.close())
      messages.close()
    }
  }
}
