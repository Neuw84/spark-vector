package io.sparkvector.shuffle

import java.nio.ByteBuffer
import java.nio.channels.{FileChannel, ReadableByteChannel}
import java.nio.file.{Path, StandardOpenOption}
import java.util.{HashMap => JHashMap}
import scala.jdk.CollectionConverters._

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{VectorArrowColumnVector, VectorDecimalColumnVector, VectorDictionaryColumnVector}
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.{FieldVector, IntVector, VarCharVector}
import org.apache.arrow.compression.CommonsCompressionFactory
import org.apache.arrow.vector.ipc.ArrowStreamReader
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

  final case class Index(offsets: Array[Long], lengths: Array[Long], rows: Array[Long]) {
    def numPartitions: Int = offsets.length
  }

  def arrowType(dt: DataType): ArrowType = dt match {
    case IntegerType => new ArrowType.Int(32, true)
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

  def arrowField(name: String, dt: DataType, ordinal: Int): Field = {
    val metadata = new JHashMap[String, String]()
    metadata.put(TypeKey, dt.json)
    val encoding = if (dt == StringType) dictionaryEncoding(ordinal) else null
    new Field(name, new FieldType(true, arrowType(dt), encoding, metadata), null)
  }

  def arrowSchema(schema: StructType): Schema =
    new Schema(schema.fields.zipWithIndex.map { case (f, i) => arrowField(f.name, f.dataType, i) }.toSeq.asJava)

  def sparkType(field: Field): DataType = DataType.fromJson(field.getMetadata.get(TypeKey))

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
  private final class RangeChannel(file: FileChannel, offset: Long, length: Long) extends ReadableByteChannel {
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
  final class PartitionReader(path: Path, partition: Int, allocator: BufferAllocator) extends Iterator[ColumnarBatch] with AutoCloseable {
    private val file = FileChannel.open(path, StandardOpenOption.READ)
    private val index = readIndex(file)
    private val inner =
      if (index.lengths(partition) == 0) null
      else new StreamReader(new RangeChannel(file, index.offsets(partition), index.lengths(partition)), allocator)

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
        case StringType =>
          val dict = dictionary(field.getDictionary.getId)
          val copy = new VarCharVector(field.getName + ".dictionary", allocator)
          dict.makeTransferPair(copy).splitAndTransfer(0, dict.getValueCount)
          new VectorDictionaryColumnVector(moved.asInstanceOf[IntVector], copy)
        case d: DecimalType if d.precision <= TypeMapping.MAX_DECIMAL_PRECISION =>
          new VectorDecimalColumnVector(moved.asInstanceOf[org.apache.arrow.vector.BigIntVector], d)
        case _ => new VectorArrowColumnVector(moved)
      }
      c += 1
    }
    new ColumnarBatch(columns, n)
  }

  /**
   * One IPC stream (a partition's bytes, wherever they come from: a file range, a fetched block, a
   * Flight stream's file) read back as the operators' column vectors.
   */
  final class StreamReader(channel: ReadableByteChannel, allocator: BufferAllocator) extends Iterator[ColumnarBatch] with AutoCloseable {
    private val input = new PeekableChannel(channel)
    private var reader: ArrowStreamReader = new ArrowStreamReader(input, allocator, CommonsCompressionFactory.INSTANCE)
    private var nextBatch: ColumnarBatch = _
    private var types: Array[DataType] = _
    /** The batch last handed out: the consumers do not close their input, so it is closed when the next one is produced (or at close). */
    private var last: ColumnarBatch = _
    private var done = false

    /**
     * Past one stream's end-of-stream marker, another stream may follow: the channel is the
     * concatenation of several map outputs' streams when a shuffle service aggregated a partition
     * (future work) -- each carries its own schema and dictionaries, so a fresh reader starts there.
     */
    private def advance(): Unit = while (!done && nextBatch == null) {
      if (reader.loadNextBatch()) {
        if (types == null) types = sparkTypes(reader.getVectorSchemaRoot)
        nextBatch = toBatch(reader.getVectorSchemaRoot, id => reader.lookup(id).getVector.asInstanceOf[VarCharVector], allocator, types)
      } else if (input.atEnd) {
        done = true
      } else {
        reader.close(false)
        reader = new ArrowStreamReader(input, allocator, CommonsCompressionFactory.INSTANCE)
      }
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
      reader.close()
    }
  }

  /** A channel that can tell whether any byte is left, by reading one ahead. */
  private final class PeekableChannel(inner: ReadableByteChannel) extends ReadableByteChannel {
    private var peeked: Int = -1 // -1 none, 0..255 a byte held back
    private var eof = false

    def atEnd: Boolean = {
      if (peeked < 0 && !eof) {
        val one = ByteBuffer.allocate(1)
        var n = 0
        while (n == 0) n = inner.read(one)
        if (n < 0) eof = true else peeked = one.get(0) & 0xff
      }
      peeked < 0
    }

    override def read(dst: ByteBuffer): Int = {
      if (!dst.hasRemaining) return 0
      if (peeked >= 0) {
        dst.put(peeked.toByte)
        peeked = -1
        val more = if (dst.hasRemaining) inner.read(dst) else 0
        1 + (if (more < 0) 0 else more)
      } else inner.read(dst)
    }

    override def isOpen: Boolean = inner.isOpen
    override def close(): Unit = inner.close()
  }
}
