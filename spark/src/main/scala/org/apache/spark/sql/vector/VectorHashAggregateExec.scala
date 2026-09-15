package org.apache.spark.sql.vector

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.agg.{AggState, VectorAggFunction, VectorAggregates}
import io.sparkvector.spark.arrow.{ArrowOutput, VectorAllocators}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for a Partial-mode HashAggregateExec. It emits the same aggregation buffer
 * schema as Spark (`resultExpressions` are the grouping attributes followed by the functions'
 * `inputAggBufferAttributes`), so Spark's exchange and Final aggregate consume it unchanged.
 *
 * This version handles the ungrouped case: one buffer row per partition, emitted even for empty
 * input, exactly like Spark.
 */
case class VectorHashAggregateExec(
    requiredChildDistributionExpressions: Option[Seq[Expression]],
    groupingExpressions: Seq[NamedExpression],
    aggregateExpressions: Seq[AggregateExpression],
    aggregateAttributes: Seq[Attribute],
    resultExpressions: Seq[NamedExpression],
    child: SparkPlan)
    extends VectorExec {

  require(groupingExpressions.isEmpty, "grouped aggregation is not supported yet")

  override def output: Seq[Attribute] = resultExpressions.map(_.toAttribute)
  override def outputPartitioning: Partitioning = child.outputPartitioning

  /** Buffer and result attributes originate here (mirrors HashAggregateExec.producedAttributes). */
  override def producedAttributes: AttributeSet =
    AttributeSet(aggregateAttributes) ++
      AttributeSet(resultExpressions.diff(groupingExpressions).map(_.toAttribute)) ++
      AttributeSet(aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)) ++
      AttributeSet(aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes))

  /** For each output column: (aggregate index, buffer slot). */
  @transient private lazy val layout: Array[(Int, Int)] =
    VectorAggregatePlanner.bufferLayout(aggregateExpressions, resultExpressions) match {
      case Right(l) => l.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate: $reason")
    }

  @transient private lazy val compiled: Array[VectorAggFunction] = aggregateExpressions.map { agg =>
    VectorAggregates.compile(agg, child.output) match {
      case Right(f) => f
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate ${agg.sql}: $reason")
    }
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val aggs = compiled
    val l = layout
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorUngroupedAggregateIterator(iter, aggs, l, outputAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Functions: ${aggregateExpressions.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

/** Drains the partition, then emits exactly one buffer row. */
private[vector] class VectorUngroupedAggregateIterator(
    input: Iterator[ColumnarBatch],
    aggs: Array[VectorAggFunction],
    layout: Array[(Int, Int)],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorHashAggregateExec")
  private var emitted = false
  private var result: ColumnarBatch = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = !emitted

  override def next(): ColumnarBatch = {
    if (emitted) throw new NoSuchElementException("aggregate already emitted")
    emitted = true
    val states: Array[AggState] = aggs.map(_.newState())
    while (input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) {
        metrics.timed {
          metrics.numInputBatches += 1
          EvalContexts.withBatch(batch) { ctx =>
            var i = 0
            while (i < states.length) { states(i).update(ctx); i += 1 }
          }
        }
      }
    }
    val buffers = states.map(_.bufferValues)
    val columns = new Array[ColumnVector](layout.length)
    var c = 0
    while (c < columns.length) {
      val (aggIdx, slot) = layout(c)
      val (name, dt) = outputAttrs(c)
      columns(c) = ArrowOutput.scalarColumn(name, dt, buffers(aggIdx)(slot), allocator)
      c += 1
    }
    metrics.numOutputBatches += 1
    metrics.numOutputRows += 1
    result = new ColumnarBatch(columns, 1)
    result
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      if (result != null) { result.close(); result = null }
      allocator.close()
    }
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorAggregatePlanner {

  /** Maps each result attribute to the (aggregate, buffer slot) that produces it. */
  def bufferLayout(
      aggregateExpressions: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression]): Either[String, Seq[(Int, Int)]] = {
    val slots: Map[org.apache.spark.sql.catalyst.expressions.ExprId, (Int, Int)] =
      aggregateExpressions.zipWithIndex.flatMap { case (agg, i) =>
        agg.aggregateFunction.inputAggBufferAttributes.zipWithIndex.map { case (a, slot) => a.exprId -> (i, slot) }
      }.toMap
    val mapped = resultExpressions.map {
      case a: Attribute => slots.get(a.exprId).toRight(s"result attribute ${a.name} is not an aggregation buffer")
      case other => Left(s"result expression ${other.sql} is not a plain buffer attribute")
    }
    mapped.collectFirst { case Left(r) => r } match {
      case Some(reason) => Left(reason)
      case None => Right(mapped.collect { case Right(x) => x })
    }
  }

  /** Attempts to convert a Spark HashAggregateExec; Left explains the fallback. */
  def plan(a: HashAggregateExec): Either[String, VectorHashAggregateExec] = {
    if (a.groupingExpressions.nonEmpty) Left("grouped aggregation not supported yet")
    else if (a.aggregateExpressions.isEmpty) Left("aggregate without functions")
    else if (a.aggregateExpressions.exists(_.mode != Partial)) Left("only Partial aggregation is vectorized; Final runs in Spark")
    else {
      val failures = a.aggregateExpressions.flatMap(agg => VectorAggregates.compile(agg, a.child.output).left.toOption.map(r => s"${agg.sql}: $r"))
      if (failures.nonEmpty) Left(failures.mkString("; "))
      else bufferLayout(a.aggregateExpressions, a.resultExpressions).flatMap { _ =>
        a.resultExpressions.map(_.toAttribute).find(attr => !TypeMapping.isSupported(attr.dataType)) match {
          case Some(attr) => Left(s"unsupported buffer type ${attr.dataType.simpleString} for ${attr.name}")
          case None =>
            Right(VectorHashAggregateExec(
              a.requiredChildDistributionExpressions,
              a.groupingExpressions,
              a.aggregateExpressions,
              a.aggregateAttributes,
              a.resultExpressions,
              a.child))
        }
      }
    }
  }
}
