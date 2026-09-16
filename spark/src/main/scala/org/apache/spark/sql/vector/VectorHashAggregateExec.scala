package org.apache.spark.sql.vector

import io.sparkvector.kernels.{GroupAssignment, GroupKeyTable, VecType, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.agg.{AggState, GroupedAggState, VectorAggFunction, VectorAggregates}
import io.sparkvector.spark.arrow.{ArrowOutput, ArrowVectorBuffers, VectorAllocators}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference, AttributeSet, Cast, CheckOverflowInSum, DecimalDivideWithOverflowCheck, EqualTo, Expression, If, Literal, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateMode, Average, Complete, DeclarativeAggregate, Final, Partial, PartialMerge, Sum}
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning, UnspecifiedDistribution}
import org.apache.spark.sql.execution.{PartitioningPreservingUnaryExecNode, SparkPlan}
import org.apache.spark.sql.execution.aggregate.{BaseAggregateExec, HashAggregateExec}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types.{DataType, DecimalType}
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
    child: SparkPlan,
    strictFloatingPoint: Boolean = true)
    extends VectorExec with PartitioningPreservingUnaryExecNode {

  override def output: Seq[Attribute] = resultExpressions.map(_.toAttribute)

  /**
   * The child's partitioning expressed over this operator's output, through the result aliases, as
   * Spark's HashAggregateExec reports it: `GROUP BY d_year` with `d_year AS year` in the result is
   * partitioned by `year`, not by the input attribute `d_year` this operator no longer outputs. A
   * union above compares its children's partitionings over their outputs -- an expression naming an
   * attribute the child does not output makes the union's partitioning unknown at execution, and its
   * plain concatenation then emits every key once per child (TPC-DS q66: 10 rows for Spark's 5, #162).
   */
  override protected def outputExpressions: Seq[NamedExpression] = resultExpressions

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
    VectorAggregatePlanner.compileFinalResults(groupingExpressions, aggregateExpressions, aggregateAttributes, resultExpressions) match {
      case Right(exprs) => exprs.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate results: $reason")
    }

  @transient private lazy val compiledKeys: Array[VectorExpr] = groupingExpressions.map { e =>
    VectorAggregatePlanner.compileKey(e, child.output) match {
      case Right(k) => k
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize grouping key ${e.sql}: $reason")
    }
  }.toArray

  // A merging stage's input is its grouping columns then the buffers (Spark's initialInputBufferOffset).
  @transient private lazy val compiled: Array[VectorAggFunction] = aggregateExpressions.zip(VectorAggregates.bufferOffsets(groupingExpressions.length, aggregateExpressions)).map { case (agg, offset) =>
    VectorAggregates.compile(agg, child.output, offset, strictFloatingPoint) match {
      case Right(f) => f
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate ${agg.sql}: $reason")
    }
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val aggs = compiled
    val keys = compiledKeys
    val l = layout
    val finalMode = emitsResults
    // In result modes a function may emit its result in a buffer slot under the result's type
    // (the wide decimal average), so the batch's column types are the functions' emitted types.
    val bufferAttrs =
      if (finalMode) {
        val declared = aggregateExpressions.map(_.aggregateFunction.aggBufferAttributes.map(_.dataType))
        val types = groupingExpressions.map(_.dataType) ++ aggs.zip(declared).flatMap { case (f, d) => f.emittedTypes(d) }
        bufferAttributes.map(_.name).zip(types).toArray
      } else output.map(a => (a.name, a.dataType)).toArray
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
       |Floating point: ${if (strictFloatingPoint) "strict (Spark's rounding)" else "fast"}
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

  private def bufferColumn(name: String, dt: DataType, state: GroupedAggState, slot: Int, from: Int, to: Int): ColumnVector =
    AggBufferColumns.column(name, dt, state, slot, from, to, allocator)

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
      aggregateAttributes: Seq[Attribute],
      resultExpressions: Seq[NamedExpression]): Either[String, Seq[VectorExpr]] = {
    val input = bufferAttributes(groupingExpressions, aggregateExpressions)
    // The result expressions reference the operator's `aggregateAttributes`, positionally aligned with
    // its aggregate expressions -- not necessarily each expression's own `resultAttribute`: Spark's
    // distinct rewrite builds the Final distinct expression afresh but keeps the original attribute.
    val evaluate: Map[org.apache.spark.sql.catalyst.expressions.ExprId, Expression] = aggregateExpressions.zip(aggregateAttributes).flatMap { case (agg, attr) =>
      agg.aggregateFunction match {
        case d: DeclarativeAggregate => Seq(attr.exprId -> d.evaluateExpression, agg.resultAttribute.exprId -> d.evaluateExpression)
        case _ => Nil
      }
    }.toMap
    val compiled = resultExpressions.map { e =>
      val substituted = e.transform { case a: AttributeReference if evaluate.contains(a.exprId) => evaluate(a.exprId) }
      (wideResult(substituted, input) match {
        // The merge already applied isEmpty / the count test and the overflow rule at emission: forward the column.
        case Some(ordinal) => Right(ColumnRef(ordinal, e.dataType))
        // A literal result (`'store' AS channel` beside the aggregates) is a constant column the
        // result projection materialises like any other.
        case None => ExpressionCompiler.compile(substituted, input).flatMap {
          case v if !TypeMapping.isSupported(e.dataType) => Left(s"unsupported result type ${e.dataType.simpleString} for ${e.name}")
          case v => Right(v)
        }
      }).left.map(r => s"${e.sql}: $r")
    }
    compiled.collectFirst { case Left(r) => r } match {
      case Some(reason) => Left(reason)
      case None => Right(compiled.collect { case Right(v) => v })
    }
  }

  /**
   * A wide decimal aggregate's result over its own buffer attributes, in the shape and only the
   * shape Spark builds: a decimal `Sum`'s `If(isEmpty, null, CheckOverflowInSum(sum, ...))`, or a
   * decimal `Average`'s `If(count = 0, null, DecimalDivideWithOverflowCheck(sum, count, ...))`. The
   * function emits the result in the sum slot already, so the projection is the ordinal of that
   * slot. Returns None for anything else.
   */
  private def wideResult(e: Expression, input: Seq[Attribute]): Option[Int] = {
    val body = e match { case Alias(child, _) => child; case other => other }
    def slot(sum: AttributeReference, other: AttributeReference): Option[Int] = {
      val ordinal = input.indexWhere(_.exprId == sum.exprId)
      if (ordinal >= 0 && input.exists(_.exprId == other.exprId)) Some(ordinal) else None
    }
    body match {
      case If(isEmpty: AttributeReference, Literal(null, _), CheckOverflowInSum(sum: AttributeReference, dt: DecimalType, _, _))
          if dt.precision > TypeMapping.MAX_DECIMAL_PRECISION && sum.dataType == dt =>
        slot(sum, isEmpty)
      case If(EqualTo(count: AttributeReference, Literal(0L, _)), Literal(null, _), DecimalDivideWithOverflowCheck(sum: AttributeReference, Cast(count2: AttributeReference, _, _, _), _, _, _))
          if count.exprId == count2.exprId && sum.dataType.isInstanceOf[DecimalType] && sum.dataType.asInstanceOf[DecimalType].precision > TypeMapping.MAX_DECIMAL_PRECISION =>
        slot(sum, count)
      case _ => None
    }
  }

  /** Wide decimal sum and average buffers a merging aggregate reads: the one kind of wide input the operator accepts. */
  def wideBuffers(a: BaseAggregateExec): Set[org.apache.spark.sql.catalyst.expressions.ExprId] =
    a.aggregateExpressions.zip(VectorAggregates.bufferOffsets(a)).collect {
      case (agg, offset) if VectorAggregates.merges(agg.mode) && wideDecimalBuffer(agg.aggregateFunction) =>
        // By id, and by Spark's position when the Final's function instance carries fresh ids.
        Seq(agg.aggregateFunction.inputAggBufferAttributes.head.exprId) ++ a.child.output.lift(offset).map(_.exprId)
    }.flatten.toSet

  private def wideDecimalBuffer(f: org.apache.spark.sql.catalyst.expressions.aggregate.AggregateFunction): Boolean = f match {
    case s: Sum => s.dataType.isInstanceOf[DecimalType]
    case a: Average => a.child.dataType.isInstanceOf[DecimalType]
    case _ => false
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
    // A buffer-emitting stage's result attributes are its keys then its buffers in order, so a result
    // attribute whose id matches nothing (a function instance rewritten between planning and this
    // check) is still bound by that position, as Spark binds it.
    val positional: Seq[OutputSlot] =
      groupingExpressions.indices.map(KeySlot(_): OutputSlot) ++
        aggregateExpressions.zipWithIndex.flatMap { case (agg, i) => agg.aggregateFunction.inputAggBufferAttributes.indices.map(BufferSlot(i, _): OutputSlot) }
    val mapped = resultExpressions.zipWithIndex.map {
      case (a: Attribute, pos) =>
        keySlots.get(a.exprId).orElse(bufferSlots.get(a.exprId)).orElse(if (resultExpressions.length == positional.length) positional.lift(pos) else None)
          .toRight(s"result attribute ${a.name} is neither a grouping key nor an aggregation buffer")
      case (other, _) => Left(s"result expression ${other.sql} is not a plain attribute")
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
   * Whether the operator reads an exchange (so only its input types matter): the merge modes, or a
   * keys-only aggregate (`SELECT DISTINCT`, the first aggregate of Spark's distinct rewrite) whose
   * required child distribution is set -- Spark plans those as a Partial with no distribution and a
   * Final over the exchange with the keys as distribution, and both merely emit the keys.
   */
  def readsExchange(a: BaseAggregateExec): Boolean =
    if (a.aggregateExpressions.isEmpty) a.requiredChildDistributionExpressions.isDefined
    else mergesBuffers(a.aggregateExpressions.map(_.mode).distinct)

  /**
   * Attempts to convert a Spark HashAggregateExec; Left explains the fallback. Two independent
   * decisions: each aggregate expression's mode says whether its state is updated from the input
   * (`Partial`, `Complete`) or merged from buffers (`PartialMerge`, `Final`); the operator's mode set
   * says whether it emits buffers (`Partial` / `PartialMerge`, as Spark's distinct rewrite mixes
   * them) or results (`Final` / `Complete`). `finalEnabled` gates the modes that read an exchange.
   */
  def plan(a: BaseAggregateExec, finalEnabled: Boolean, strictFloatingPoint: Boolean = true): Either[String, VectorHashAggregateExec] = {
    val modes = a.aggregateExpressions.map(_.mode).distinct
    val keysOnly = a.aggregateExpressions.isEmpty
    // A keys-only aggregate emits its keys in both of Spark's stages: the buffer layout fits both.
    val results = !keysOnly && emitsResults(modes)
    if (keysOnly && a.groupingExpressions.isEmpty) Left("aggregate without keys or functions")
    else if (!keysOnly && !results && !emitsBuffers(modes)) Left(s"aggregation modes ${modes.mkString(", ")} mix buffer and result output")
    else if (readsExchange(a) && !finalEnabled) Left("merging aggregation stages disabled by configuration")
    else {
      val keyFailures = a.groupingExpressions.flatMap(g => compileKey(g, a.child.output).left.toOption.map(r => s"${g.sql}: $r"))
      val aggFailures = a.aggregateExpressions.zip(VectorAggregates.bufferOffsets(a)).flatMap { case (agg, offset) =>
        VectorAggregates.compile(agg, a.child.output, offset).left.toOption.map(r => s"${agg.sql}: $r")
      }
      val failures = keyFailures ++ aggFailures
      val layoutCheck: Either[String, Seq[Any]] =
        if (results) compileFinalResults(a.groupingExpressions, a.aggregateExpressions, a.aggregateAttributes, a.resultExpressions)
        else outputLayout(a.groupingExpressions, a.aggregateExpressions, a.resultExpressions)
      if (failures.nonEmpty) Left(failures.mkString("; "))
      else layoutCheck.flatMap { compiled =>
        // A wide decimal (p > 18) has no lane, but an aggregate may output one in two places: a
        // buffer-emitting operator's `sum` buffer of a decimal sum (Spark's Decimal(p + 10, s)), and
        // a Final's result of that sum, which the merge emits ready-made and the projection forwards.
        val wideOutputs: Set[org.apache.spark.sql.catalyst.expressions.ExprId] =
          if (results) a.resultExpressions.zip(compiled).collect {
            case (e, ColumnRef(_, dt: DecimalType)) if dt.precision > TypeMapping.MAX_DECIMAL_PRECISION => e.toAttribute.exprId
          }.toSet
          else a.aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes).collect {
            case attr if attr.dataType.isInstanceOf[DecimalType] => attr.exprId
          }.toSet
        a.resultExpressions.map(_.toAttribute).find(attr => !TypeMapping.isSupported(attr.dataType) && !wideOutputs.contains(attr.exprId)) match {
          case Some(attr) => Left(s"unsupported output type ${attr.dataType.simpleString} for ${attr.name}")
          case None =>
            Right(VectorHashAggregateExec(
              a.requiredChildDistributionExpressions,
              a.groupingExpressions,
              a.aggregateExpressions,
              a.aggregateAttributes,
              a.resultExpressions,
              a.child,
              strictFloatingPoint))
        }
      }
    }
  }
}

/** Boxed buffer values of a run of groups `[from, to)` as one Arrow column; shared with the window aggregate. */
object AggBufferColumns {
  def column(name: String, dt: DataType, state: GroupedAggState, slot: Int, from: Int, to: Int, allocator: BufferAllocator): ColumnVector =
    values(name, dt, to - from, o => state.bufferValue(from + o, slot), allocator)

  /** Boxed values `get(0 until count)` (Spark's internal representation, or null) as one Arrow column. */
  def values(name: String, dt: DataType, count: Int, get: Int => Any, allocator: BufferAllocator): ColumnVector = {
    dt match {
      case d: DecimalType if d.precision > TypeMapping.MAX_DECIMAL_PRECISION =>
        // The sum buffer of a wide decimal sum: boxed exact totals into Arrow's 128-bit vector.
        val values = new Array[java.math.BigDecimal](count)
        var o = 0
        while (o < count) { values(o) = get(o).asInstanceOf[java.math.BigDecimal]; o += 1 }
        return ArrowOutput.decimalColumn(name, d, values, allocator)
      case org.apache.spark.sql.types.StringType =>
        // min/max, first/last, min_by/max_by over strings: boxed UTF8Strings into a varchar vector.
        val values = new Array[Array[Byte]](count)
        var o = 0
        while (o < count) {
          val v = get(o)
          values(o) = if (v == null) null else v.asInstanceOf[org.apache.spark.unsafe.types.UTF8String].getBytes
          o += 1
        }
        return ArrowOutput.utf8Column(name, values, allocator)
      case _ =>
    }
    val out: ArrowVectorBuffers = ArrowOutput.allocateFixed(name, dt, count, allocator)
    val data = out.data()
    val validity = out.validity()
    var o = 0
    while (o < count) {
      get(o) match {
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
}
