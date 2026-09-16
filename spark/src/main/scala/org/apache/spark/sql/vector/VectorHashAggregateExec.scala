package org.apache.spark.sql.vector

import io.sparkvector.kernels.{GroupAssignment, GroupKeyTable, VecType, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.agg.{AggState, GroupedAggState, VectorAggFunction, VectorAggregates}
import io.sparkvector.spark.arrow.{ArrowOutput, ArrowVectorBuffers, VectorAllocators}
import io.sparkvector.spark.expr.{ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, AttributeSet, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateMode, Complete, DeclarativeAggregate, Final, Partial, PartialMerge}
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning, UnspecifiedDistribution}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/** Where an output column of the partial aggregate comes from. */
sealed trait OutputSlot extends Serializable
final case class KeySlot(key: Int) extends OutputSlot
final case class BufferSlot(agg: Int, slot: Int) extends OutputSlot

/**
 * Columnar replacement for a HashAggregateExec in Partial or Final mode.
 *
 * Partial mode emits the same aggregation buffer schema as Spark (`resultExpressions` are the
 * grouping attributes followed by the functions' `inputAggBufferAttributes`), so any exchange and
 * Final aggregate consume it unchanged. Final mode merges those buffers per group (sum of sums,
 * sum of counts, min of mins...) and then evaluates the `resultExpressions`, with each aggregate's
 * result attribute replaced by the function's `evaluateExpression` over the merged buffer (for
 * `avg`, `sum / count`), through the same expression kernels as a projection.
 *
 * Without grouping keys one row is emitted per partition, even for empty input, exactly like
 * Spark. With grouping keys, rows are assigned dense group ids through a [[GroupKeyTable]] and
 * reduced per group (masked SIMD reductions while there are very few groups, scalar scatter
 * beyond); one row per group is emitted at the end.
 */
case class VectorHashAggregateExec(
    requiredChildDistributionExpressions: Option[Seq[Expression]],
    groupingExpressions: Seq[NamedExpression],
    aggregateExpressions: Seq[AggregateExpression],
    aggregateAttributes: Seq[Attribute],
    resultExpressions: Seq[NamedExpression],
    child: SparkPlan)
    extends VectorExec {

  override def output: Seq[Attribute] = resultExpressions.map(_.toAttribute)
  override def outputPartitioning: Partitioning = child.outputPartitioning

  /** Same contract as HashAggregateExec, so adaptive execution treats the exchange below alike. */
  override def requiredChildDistribution: List[Distribution] = requiredChildDistributionExpressions match {
    case Some(exprs) if exprs.isEmpty => AllTuples :: Nil
    case Some(exprs) => ClusteredDistribution(exprs) :: Nil
    case None => UnspecifiedDistribution :: Nil
  }

  def isFinal: Boolean = aggregateExpressions.nonEmpty && aggregateExpressions.forall(_.mode == Final)

  /** The modes present, in plan order. */
  def modes: Seq[AggregateMode] = aggregateExpressions.map(_.mode).distinct

  /**
   * `Final` and `Complete` evaluate the result expressions over the aggregated state; `Partial` and
   * `PartialMerge` emit the aggregation buffers for a later stage. The planner refuses any other mix.
   */
  def emitsResults: Boolean = VectorAggregatePlanner.emitsResults(modes)

  /** Buffer and result attributes originate here (mirrors HashAggregateExec.producedAttributes). */
  override def producedAttributes: AttributeSet =
    AttributeSet(aggregateAttributes) ++
      AttributeSet(resultExpressions.diff(groupingExpressions).map(_.toAttribute)) ++
      AttributeSet(aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)) ++
      AttributeSet(aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes))

  /** Buffer modes: the emitted columns. Result modes: the aggregated-state batch the result projection reads. */
  @transient private lazy val layout: Array[OutputSlot] =
    if (emitsResults) VectorAggregatePlanner.bufferLayout(groupingExpressions, aggregateExpressions).toArray
    else VectorAggregatePlanner.outputLayout(groupingExpressions, aggregateExpressions, resultExpressions) match {
      case Right(l) => l.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate: $reason")
    }

  @transient private lazy val bufferAttributes: Seq[Attribute] =
    VectorAggregatePlanner.bufferAttributes(groupingExpressions, aggregateExpressions)

  @transient private lazy val resultProjection: Array[VectorExpr] =
    VectorAggregatePlanner.compileFinalResults(groupingExpressions, aggregateExpressions, resultExpressions) match {
      case Right(exprs) => exprs.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate results: $reason")
    }

  @transient private lazy val compiledKeys: Array[VectorExpr] = groupingExpressions.map { e =>
    VectorAggregatePlanner.compileKey(e, child.output) match {
      case Right(k) => k
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize grouping key ${e.sql}: $reason")
    }
  }.toArray

  @transient private lazy val compiled: Array[VectorAggFunction] = aggregateExpressions.map { agg =>
    VectorAggregates.compile(agg, child.output) match {
      case Right(f) => f
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate ${agg.sql}: $reason")
    }
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val aggs = compiled
    val keys = compiledKeys
    val l = layout
    val finalMode = emitsResults
    val bufferAttrs = (if (finalMode) bufferAttributes else output).map(a => (a.name, a.dataType)).toArray
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val results = if (finalMode) resultProjection else Array.empty[VectorExpr]
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      val buffers: Iterator[ColumnarBatch] =
        if (keys.isEmpty) new VectorUngroupedAggregateIterator(iter, aggs, l, bufferAttrs, m)
        else new VectorGroupedAggregateIterator(iter, keys, aggs, l, bufferAttrs, m)
      if (finalMode) {
        // The projection's own bookkeeping goes to unregistered metrics so rows are not counted twice.
        val scratch = new VectorMetrics(new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("timing"))
        new VectorProjectIterator(buffers, results, identity = false, outputAttrs, emitSelection = false, scratch)
      } else buffers
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Mode: ${modes.mkString(", ")}
       |Keys: ${groupingExpressions.map(_.sql).mkString(", ")}
       |Functions: ${aggregateExpressions.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

/** Drains the partition, then emits exactly one buffer row. */
private[vector] class VectorUngroupedAggregateIterator(
    input: Iterator[ColumnarBatch],
    aggs: Array[VectorAggFunction],
    layout: Array[OutputSlot],
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
      val (name, dt) = outputAttrs(c)
      columns(c) = layout(c) match {
        case BufferSlot(aggIdx, slot) => ArrowOutput.scalarColumn(name, dt, buffers(aggIdx)(slot), allocator)
        case KeySlot(_) => throw new IllegalStateException("key slot in ungrouped aggregate")
      }
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

/** Drains the partition into a group table, then emits the groups in batches. */
private[vector] class VectorGroupedAggregateIterator(
    input: Iterator[ColumnarBatch],
    keyExprs: Array[VectorExpr],
    aggs: Array[VectorAggFunction],
    layout: Array[OutputSlot],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val OutputBatchSize = 4096

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorHashAggregateExec")
  private val table = new GroupKeyTable(keyExprs.map(_.vecType))
  private val states: Array[GroupedAggState] = aggs.map(_.newGroupedState())
  private var idScratch = new Array[Int](0)
  private var drained = false
  private var emittedGroups = 0
  private var current: ColumnarBatch = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  private def drain(): Unit = {
    if (drained) return
    drained = true
    val keys = new Array[VectorBuffers](keyExprs.length)
    while (input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) {
        metrics.timed {
          metrics.numInputBatches += 1
          EvalContexts.withBatch(batch) { ctx =>
            // Physical rows: a normalized foreign batch reports its live count as numRows.
            val n = ctx.numRows
            if (idScratch.length < n) idScratch = new Array[Int](n)
            var k = 0
            while (k < keys.length) { keys(k) = keyExprs(k).eval(ctx); k += 1 }
            val numGroups = table.assign(keys, n, idScratch, ctx.selection)
            val assignment = GroupAssignment.of(idScratch, n, numGroups, ctx.arena, ctx.selection)
            var i = 0
            while (i < states.length) { states(i).update(ctx, assignment); i += 1 }
          }
        }
      }
    }
  }

  override def hasNext: Boolean = {
    drain()
    emittedGroups < table.size()
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more groups")
    releaseCurrent()
    val from = emittedGroups
    val to = math.min(table.size(), from + OutputBatchSize)
    val count = to - from
    val columns = new Array[ColumnVector](layout.length)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = outputAttrs(c)
      columns(c) = layout(c) match {
        case KeySlot(k) => keyColumn(name, dt, k, from, to)
        case BufferSlot(aggIdx, slot) => bufferColumn(name, dt, states(aggIdx), slot, from, to)
      }
      c += 1
    }
    emittedGroups = to
    metrics.numOutputBatches += 1
    metrics.numOutputRows += count
    current = new ColumnarBatch(columns, count)
    current
  }

  private def keyColumn(name: String, dt: DataType, k: Int, from: Int, to: Int): ColumnVector = {
    val count = to - from
    if (table.`type`(k) == VecType.UTF8) {
      val out = ArrowOutput.allocateUtf8(name, count, table.utf8Bytes(k, from, to), allocator)
      table.writeKeys(k, from, to, out.validity(), out.data(), out.offsets())
      ArrowOutput.finish(out, count, false)
    } else {
      val out = ArrowOutput.allocateFixed(name, dt, count, allocator)
      table.writeKeys(k, from, to, out.validity(), out.data(), null)
      ArrowOutput.finish(out, count, false)
    }
  }

  private def bufferColumn(name: String, dt: DataType, state: GroupedAggState, slot: Int, from: Int, to: Int): ColumnVector = {
    val count = to - from
    val out: ArrowVectorBuffers = ArrowOutput.allocateFixed(name, dt, count, allocator)
    val data = out.data()
    val validity = out.validity()
    var o = 0
    while (o < count) {
      state.bufferValue(from + o, slot) match {
        case null => io.sparkvector.kernels.Bitmap.clear(validity, o)
        case v: java.lang.Double =>
          io.sparkvector.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_DOUBLE, o.toLong << 3, v.doubleValue())
        case v: java.lang.Long =>
          io.sparkvector.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_LONG, o.toLong << 3, v.longValue())
        case v: java.lang.Integer =>
          io.sparkvector.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_INT, o.toLong << 2, v.intValue())
        case v: java.lang.Boolean =>
          io.sparkvector.kernels.Bitmap.set(validity, o); io.sparkvector.kernels.Bitmap.setTo(data, o, v.booleanValue())
        case other => throw new IllegalStateException(s"unexpected buffer value $other")
      }
      o += 1
    }
    ArrowOutput.finish(out, count, false)
  }

  private def releaseCurrent(): Unit = {
    if (current != null) { current.close(); current = null }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseCurrent()
      allocator.close()
    }
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorAggregatePlanner {

  private val keyTypes: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.BOOL, VecType.UTF8)

  /**
   * Grouping keys: ints, longs, booleans and strings. Doubles are excluded because Spark wraps them
   * in NormalizeNaNAndZero, which is not compiled (and would need its own equality semantics).
   */
  def compileKey(e: NamedExpression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(e, input).flatMap {
      case _: LiteralExpr => Left("literal grouping key")
      case k if !keyTypes.contains(k.vecType) => Left(s"grouping key type ${e.dataType.simpleString} not supported")
      case k => Right(k)
    }

  /** The merged-buffer batch of a Final aggregate: keys, then every function's buffer slots. */
  def bufferLayout(groupingExpressions: Seq[NamedExpression], aggregateExpressions: Seq[AggregateExpression]): Seq[OutputSlot] =
    groupingExpressions.indices.map(KeySlot(_): OutputSlot) ++
      aggregateExpressions.zipWithIndex.flatMap { case (agg, i) =>
        agg.aggregateFunction.aggBufferAttributes.indices.map(slot => BufferSlot(i, slot): OutputSlot)
      }

  /** Attributes of [[bufferLayout]]'s columns, the input of a Final aggregate's result projection. */
  def bufferAttributes(groupingExpressions: Seq[NamedExpression], aggregateExpressions: Seq[AggregateExpression]): Seq[Attribute] =
    groupingExpressions.map(_.toAttribute) ++ aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)

  /**
   * Final-mode result expressions with each aggregate's result attribute replaced by the
   * function's `evaluateExpression`, compiled against the merged-buffer batch.
   */
  def compileFinalResults(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression]): Either[String, Seq[VectorExpr]] = {
    val input = bufferAttributes(groupingExpressions, aggregateExpressions)
    val evaluate: Map[org.apache.spark.sql.catalyst.expressions.ExprId, Expression] = aggregateExpressions.flatMap { agg =>
      agg.aggregateFunction match {
        case d: DeclarativeAggregate => Some(agg.resultAttribute.exprId -> d.evaluateExpression)
        case _ => None
      }
    }.toMap
    val compiled = resultExpressions.map { e =>
      val substituted = e.transform { case a: AttributeReference if evaluate.contains(a.exprId) => evaluate(a.exprId) }
      ExpressionCompiler.compile(substituted, input).flatMap {
        case _: LiteralExpr => Left(s"literal result ${e.sql}")
        case v if !TypeMapping.isSupported(e.dataType) => Left(s"unsupported result type ${e.dataType.simpleString} for ${e.name}")
        case v => Right(v)
      }.left.map(r => s"${e.sql}: $r")
    }
    compiled.collectFirst { case Left(r) => r } match {
      case Some(reason) => Left(reason)
      case None => Right(compiled.collect { case Right(v) => v })
    }
  }

  /** Maps each result attribute to the grouping key or the (aggregate, buffer slot) producing it. */
  def outputLayout(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression]): Either[String, Seq[OutputSlot]] = {
    val keySlots: Map[org.apache.spark.sql.catalyst.expressions.ExprId, OutputSlot] =
      groupingExpressions.zipWithIndex.map { case (g, i) => g.toAttribute.exprId -> (KeySlot(i): OutputSlot) }.toMap
    val bufferSlots: Map[org.apache.spark.sql.catalyst.expressions.ExprId, OutputSlot] =
      aggregateExpressions.zipWithIndex.flatMap { case (agg, i) =>
        agg.aggregateFunction.inputAggBufferAttributes.zipWithIndex.map { case (a, slot) => a.exprId -> (BufferSlot(i, slot): OutputSlot) }
      }.toMap
    val mapped = resultExpressions.map {
      case a: Attribute =>
        keySlots.get(a.exprId).orElse(bufferSlots.get(a.exprId)).toRight(s"result attribute ${a.name} is neither a grouping key nor an aggregation buffer")
      case other => Left(s"result expression ${other.sql} is not a plain attribute")
    }
    mapped.collectFirst { case Left(r) => r } match {
      case Some(reason) => Left(reason)
      case None => Right(mapped.collect { case Right(x) => x })
    }
  }

  /** Every mode evaluates result expressions (`Final`, `Complete`). */
  def emitsResults(modes: Seq[AggregateMode]): Boolean = modes.nonEmpty && modes.forall(m => m == Final || m == Complete)

  /** Every mode emits aggregation buffers for a later stage (`Partial`, `PartialMerge`). */
  def emitsBuffers(modes: Seq[AggregateMode]): Boolean = modes.nonEmpty && modes.forall(m => m == Partial || m == PartialMerge)

  /** Every mode merges buffers (`PartialMerge`, `Final`): the child is an exchange whose types alone matter. */
  def mergesBuffers(modes: Seq[AggregateMode]): Boolean = modes.nonEmpty && modes.forall(VectorAggregates.merges)

  /**
   * Attempts to convert a Spark HashAggregateExec; Left explains the fallback. Two independent
   * decisions: each aggregate expression's mode says whether its state is updated from the input
   * (`Partial`, `Complete`) or merged from buffers (`PartialMerge`, `Final`); the operator's mode set
   * says whether it emits buffers (`Partial` / `PartialMerge`, as Spark's distinct rewrite mixes
   * them) or results (`Final` / `Complete`). `finalEnabled` gates the modes that read an exchange.
   */
  def plan(a: HashAggregateExec, finalEnabled: Boolean): Either[String, VectorHashAggregateExec] = {
    val modes = a.aggregateExpressions.map(_.mode).distinct
    val results = emitsResults(modes)
    if (a.aggregateExpressions.isEmpty) Left("aggregate without functions (distinct-style) not supported")
    else if (!results && !emitsBuffers(modes)) Left(s"aggregation modes ${modes.mkString(", ")} mix buffer and result output")
    else if (mergesBuffers(modes) && !finalEnabled) Left("merging aggregation stages disabled by configuration")
    else {
      val keyFailures = a.groupingExpressions.flatMap(g => compileKey(g, a.child.output).left.toOption.map(r => s"${g.sql}: $r"))
      val aggFailures = a.aggregateExpressions.flatMap(agg => VectorAggregates.compile(agg, a.child.output).left.toOption.map(r => s"${agg.sql}: $r"))
      val failures = keyFailures ++ aggFailures
      val layoutCheck: Either[String, Any] =
        if (results) compileFinalResults(a.groupingExpressions, a.aggregateExpressions, a.resultExpressions)
        else outputLayout(a.groupingExpressions, a.aggregateExpressions, a.resultExpressions)
      if (failures.nonEmpty) Left(failures.mkString("; "))
      else layoutCheck.flatMap { _ =>
        a.resultExpressions.map(_.toAttribute).find(attr => !TypeMapping.isSupported(attr.dataType)) match {
          case Some(attr) => Left(s"unsupported output type ${attr.dataType.simpleString} for ${attr.name}")
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
