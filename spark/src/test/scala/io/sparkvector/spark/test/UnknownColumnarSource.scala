package io.sparkvector.spark.test

import java.util

import scala.jdk.CollectionConverters._

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.connector.read.{Batch, InputPartition, PartitionReader, PartitionReaderFactory, Scan, ScanBuilder}
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.vectorized.{ColumnarArray, ColumnarBatch, ColumnarMap, ColumnVector}
import org.apache.spark.unsafe.types.UTF8String

/**
 * A minimal DSv2 source that only reads columnar, with batches made of [[UnknownColumnVector]]s --
 * a `ColumnVector` class no spark-vector adapter recognises, so every column must go through the
 * copy fallback of the adapter seam. Deterministic: row `r` (global, 3 partitions x 2 batches x
 * 600 rows) carries `i = r`, `l = 3r` (null when `r % 7 = 3`), `d = r / 4` (null when `r % 50 = 0`),
 * `b = (r % 3 = 0)`, `dt = 2020-01-01 + r % 730`, `ts = r hours since the epoch` (null when
 * `r % 9 = 0`), `dec = (37r - 5000) / 100` as decimal(12,2), `s = 's' || r % 50`. With the option
 * `struct=true` the schema gains `st STRUCT<a: INT>` with `a = r % 5`, for the negative case.
 *
 * Use: `spark.read.format(classOf[UnknownColumnarSource].getName).load()`.
 */
class UnknownColumnarSource extends TableProvider {
  override def inferSchema(options: CaseInsensitiveStringMap): StructType =
    UnknownColumnarSource.schema(options.getBoolean("struct", false))

  override def getTable(schema: StructType, partitioning: Array[Transform], properties: util.Map[String, String]): Table =
    new UnknownColumnarSource.UnknownTable(schema)

  override def supportsExternalMetadata(): Boolean = true
}

object UnknownColumnarSource {
  val Partitions = 3
  val BatchesPerPartition = 2
  val RowsPerBatch = 600

  def schema(withStruct: Boolean): StructType = {
    val base = StructType(Seq(
      StructField("i", IntegerType, nullable = false),
      StructField("l", LongType),
      StructField("d", DoubleType),
      StructField("b", BooleanType, nullable = false),
      StructField("dt", DateType, nullable = false),
      StructField("ts", TimestampType),
      StructField("dec", DecimalType(12, 2), nullable = false),
      StructField("s", StringType, nullable = false)))
    if (withStruct) base.add(StructField("st", StructType(Seq(StructField("a", IntegerType, nullable = false))), nullable = false)) else base
  }

  /** The value of column `name` for global row `r`, boxed; `null` for SQL null. */
  def value(name: String, r: Int): Any = name match {
    case "i" => Int.box(r)
    case "l" => if (r % 7 == 3) null else Long.box(3L * r)
    case "d" => if (r % 50 == 0) null else Double.box(r / 4.0)
    case "b" => Boolean.box(r % 3 == 0)
    case "dt" => Int.box(18262 + r % 730)
    case "ts" => if (r % 9 == 0) null else Long.box(r * 3600L * 1000000L)
    case "dec" => Decimal(37L * r - 5000L, 12, 2)
    case "s" => UTF8String.fromString("s" + (r % 50))
    case "st" => Array[Any](Int.box(r % 5))
  }

  final class UnknownTable(tableSchema: StructType) extends Table with SupportsRead {
    override def name(): String = "unknown_columnar"
    override def schema(): StructType = tableSchema
    override def capabilities(): util.Set[TableCapability] = Set(TableCapability.BATCH_READ).asJava
    override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder = () => new UnknownScan(tableSchema)
  }

  final class UnknownScan(tableSchema: StructType) extends Scan with Batch {
    override def readSchema(): StructType = tableSchema
    override def toBatch: Batch = this
    override def planInputPartitions(): Array[InputPartition] = Array.tabulate(Partitions)(UnknownPartition(_))
    override def createReaderFactory(): PartitionReaderFactory = new UnknownReaderFactory(tableSchema)
  }

  final case class UnknownPartition(index: Int) extends InputPartition

  final class UnknownReaderFactory(tableSchema: StructType) extends PartitionReaderFactory {
    override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
      throw new UnsupportedOperationException("this source only reads columnar")
    override def supportColumnarReads(partition: InputPartition): Boolean = true
    override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] =
      new UnknownReader(tableSchema, partition.asInstanceOf[UnknownPartition].index)
  }

  final class UnknownReader(tableSchema: StructType, partition: Int) extends PartitionReader[ColumnarBatch] {
    private var batch = -1
    private var current: ColumnarBatch = _

    override def next(): Boolean = {
      if (current != null) { current.close(); current = null }
      batch += 1
      if (batch >= BatchesPerPartition) return false
      val first = (partition * BatchesPerPartition + batch) * RowsPerBatch
      val columns: Array[ColumnVector] = tableSchema.fields.map { f =>
        new UnknownColumnVector(f.dataType, RowsPerBatch, k => value(f.name, first + k))
      }
      current = new ColumnarBatch(columns, RowsPerBatch)
      true
    }

    override def get(): ColumnarBatch = current
    override def close(): Unit = if (current != null) { current.close(); current = null }
  }
}

/**
 * A `ColumnVector` backed by a plain array of boxed values that no adapter knows: the adapter
 * seam can only read it through the public getters. Struct children are further such vectors.
 */
final class UnknownColumnVector(dt: DataType, n: Int, value: Int => Any) extends ColumnVector(dt) {
  private val values: Array[Any] = Array.tabulate(n)(value)
  private lazy val children: Array[UnknownColumnVector] = dt match {
    case st: StructType =>
      st.fields.zipWithIndex.map { case (f, j) =>
        new UnknownColumnVector(f.dataType, n, i => if (values(i) == null) null else values(i).asInstanceOf[Array[Any]](j))
      }
    case _ => Array.empty
  }

  override def close(): Unit = {}
  override def hasNull: Boolean = values.exists(_ == null)
  override def numNulls: Int = values.count(_ == null)
  override def isNullAt(rowId: Int): Boolean = values(rowId) == null
  override def getBoolean(rowId: Int): Boolean = values(rowId).asInstanceOf[Boolean]
  override def getByte(rowId: Int): Byte = values(rowId).asInstanceOf[Byte]
  override def getShort(rowId: Int): Short = values(rowId).asInstanceOf[Short]
  override def getInt(rowId: Int): Int = values(rowId).asInstanceOf[Int]
  override def getLong(rowId: Int): Long = values(rowId).asInstanceOf[Long]
  override def getFloat(rowId: Int): Float = values(rowId).asInstanceOf[Float]
  override def getDouble(rowId: Int): Double = values(rowId).asInstanceOf[Double]
  override def getArray(rowId: Int): ColumnarArray = throw new UnsupportedOperationException("arrays")
  override def getMap(ordinal: Int): ColumnarMap = throw new UnsupportedOperationException("maps")
  override def getDecimal(rowId: Int, precision: Int, scale: Int): Decimal =
    if (values(rowId) == null) null else values(rowId).asInstanceOf[Decimal]
  override def getUTF8String(rowId: Int): UTF8String =
    if (values(rowId) == null) null else values(rowId).asInstanceOf[UTF8String]
  override def getBinary(rowId: Int): Array[Byte] = throw new UnsupportedOperationException("binary")
  override def getChild(ordinal: Int): ColumnVector = children(ordinal)
}
