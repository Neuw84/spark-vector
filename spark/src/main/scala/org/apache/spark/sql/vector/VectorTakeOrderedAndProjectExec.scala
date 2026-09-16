package org.apache.spark.sql.vector

import scala.jdk.CollectionConverters._

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.expr.VectorExpr
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, NamedExpression, NullsFirst, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, SinglePartition}
import org.apache.spark.sql.execution.{ShuffledRowRDD, SparkPlan, TakeOrderedAndProjectExec, UnsafeRowSerializer}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics, SQLShuffleReadMetricsReporter, SQLShuffleWriteMetricsReporter}
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.catalyst.types.DataTypeUtils
import org.apache.spark.sql.types.{BooleanType, DateType, DecimalType, DoubleType, IntegerType, LongType, StringType, StructType, TimestampType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.apache.spark.util.collection.Utils

/**
 * Columnar replacement for TakeOrderedAndProjectExec (`ORDER BY ... LIMIT n`).
 *
 * Two stages, like Spark's operator. Per partition the child's batches are drained, sorted with the
 * kernels ([[VectorSortIterator]] with a limit) and only the first `limit` rows are gathered -- the
 * part that touches every row stays columnar. Those at most `limit` rows per partition are then
 * sent as rows through Spark's own single-partition shuffle, where Spark's ordering takes the final
 * top `limit`, Spark's projection applies `projectList`, and the result is materialised as one
 * columnar batch of on-heap vectors. The row stages see at most `limit x partitions` rows, which is
 * what the issue calls the driver-side merge that can stay row based.
 *
 * `offset` is not supported (the rule falls back). The partition-local stage holds the whole
 * partition in memory like [[VectorSortExec]]; partitions that need a spill should keep Spark's
 * operator (`spark.vector.exec.takeOrdered.enabled=false`).
 */
case class VectorTakeOrderedAndProjectExec(
    limit: Int,
    sortOrder: Seq[SortOrder],
    projectList: Seq[NamedExpression],
    child: SparkPlan)
    extends VectorExec {

  override def output: Seq[Attribute] = projectList.map(_.toAttribute)
  override def outputOrdering: Seq[SortOrder] = sortOrder
  override def outputPartitioning: Partitioning = SinglePartition

  private lazy val writeMetrics = SQLShuffleWriteMetricsReporter.createShuffleWriteMetrics(sparkContext)
  private lazy val readMetrics = SQLShuffleReadMetricsReporter.createShuffleReadMetrics(sparkContext)
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "time" -> SQLMetrics.createNanoTimingMetric(sparkContext, "time in spark-vector kernels")) ++ readMetrics ++ writeMetrics

  @transient private lazy val compiledKeys: Array[VectorExpr] = sortOrder.map { o =>
    VectorSortPlanner.compileKey(o, child.output) match {
      case Right(k) => k
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize sort key ${o.sql}: $reason")
    }
  }.toArray

  /** Per partition: the sorted head of the partition, as rows for the exchange. */
  private def localTopK(): RDD[InternalRow] = {
    val keys = compiledKeys
    val ascending = sortOrder.map(_.direction == Ascending).toArray
    val nullsFirst = sortOrder.map(_.nullOrdering == NullsFirst).toArray
    val childAttrs = child.output.map(a => (a.name, a.dataType)).toArray
    val childOutput = child.output
    val n = limit
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      val sorted = new VectorSortIterator(iter, keys, ascending, nullsFirst, childAttrs, m, n)
      val toUnsafe = UnsafeProjection.create(childOutput, childOutput)
      // The shuffle serialiser needs UnsafeRows; a batch is released when the next is requested,
      // so every row is copied out of it.
      sorted.flatMap(batch => batch.rowIterator().asScala.map(row => toUnsafe(row).copy()))
    }
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val ord = new LazilyGeneratedOrdering(sortOrder, child.output)
    val childOutput = child.output
    val project = projectList
    val n = limit
    val schema = DataTypeUtils.fromAttributes(output)
    val numOutputBatches = longMetric("numOutputBatches")
    val numOutputRows = longMetric("numOutputRows")
    val shuffled = new ShuffledRowRDD(
      ShuffleExchangeExec.prepareShuffleDependency(
        localTopK(), childOutput, SinglePartition, new UnsafeRowSerializer(childOutput.size), writeMetrics),
      readMetrics)
    shuffled.mapPartitionsInternal { iter =>
      val topK = Utils.takeOrdered(iter.map(_.copy()), n)(ord)
      val projected: Iterator[InternalRow] =
        if (project == childOutput) topK
        else {
          val proj = UnsafeProjection.create(project, childOutput)
          topK.map(r => proj(r).copy()) // the projection reuses one buffer; the rows are collected below
        }
      val rows = projected.toArray
      if (rows.isEmpty) Iterator.empty
      else {
        numOutputBatches += 1
        numOutputRows += rows.length
        Iterator.single(VectorTakeOrderedAndProjectExec.toBatch(schema, rows))
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def simpleString(maxFields: Int): String =
    s"VectorTakeOrderedAndProject(limit=$limit, orderBy=${sortOrder.map(_.sql).mkString(", ")}, output=${output.map(_.name).mkString(", ")})"

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Limit: $limit
       |Sort: ${sortOrder.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

object VectorTakeOrderedAndProjectExec {

  /**
   * One columnar batch of on-heap vectors holding `rows`. Covers exactly the Spark types
   * `TypeMapping` supports (the planner refuses any other output type), so every case is a plain
   * `WritableColumnVector` put.
   */
  private[vector] def toBatch(schema: StructType, rows: Array[InternalRow]): ColumnarBatch = {
    val n = rows.length
    val vectors: Array[OnHeapColumnVector] = OnHeapColumnVector.allocateColumns(n, schema)
    var c = 0
    while (c < vectors.length) {
      val v = vectors(c)
      val dt = schema(c).dataType
      var i = 0
      while (i < n) {
        val row = rows(i)
        if (row.isNullAt(c)) v.putNull(i)
        else dt match {
          case BooleanType => v.putBoolean(i, row.getBoolean(c))
          case IntegerType | DateType => v.putInt(i, row.getInt(c))
          case LongType | TimestampType => v.putLong(i, row.getLong(c))
          case DoubleType => v.putDouble(i, row.getDouble(c))
          case d: DecimalType => v.putDecimal(i, row.getDecimal(c, d.precision, d.scale), d.precision)
          case StringType => val b = row.getUTF8String(c).getBytes; v.putByteArray(i, b, 0, b.length)
          case other => throw new IllegalStateException(s"unsupported output type ${other.simpleString}")
        }
        i += 1
      }
      c += 1
    }
    new ColumnarBatch(vectors.map(v => v: ColumnVector), n)
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorTakeOrderedPlanner {

  /** Attempts to convert a Spark TakeOrderedAndProjectExec; Left explains the fallback. */
  def plan(t: TakeOrderedAndProjectExec): Either[String, VectorTakeOrderedAndProjectExec] = {
    val keyFailures = t.sortOrder.flatMap(o => VectorSortPlanner.compileKey(o, t.child.output).left.toOption.map(r => s"${o.sql}: $r"))
    val outputFailures = t.projectList.filterNot(e => TypeMapping.isSupported(e.dataType)).map(e => s"unsupported output type ${e.dataType.simpleString} for ${e.name}")
    if (t.offset != 0) Left(s"offset ${t.offset} not supported")
    else if (t.sortOrder.isEmpty) Left("sort without keys")
    else if (keyFailures.nonEmpty) Left(keyFailures.mkString("; "))
    else if (outputFailures.nonEmpty) Left(outputFailures.mkString("; "))
    else Right(VectorTakeOrderedAndProjectExec(t.limit, t.sortOrder, t.projectList, t.child))
  }
}
