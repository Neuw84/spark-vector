package org.apache.spark.sql.vector

import io.sparkvector.kernels.{Bitmap, GroupAssignment, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.agg.{GroupedAggState, Rows, VectorAggFunction, VectorAggregates}
import io.sparkvector.spark.arrow.{ArrowOutput, BorrowedColumnVector, SelectedColumnarBatch, VectorAllocators}
import io.sparkvector.spark.expr.{ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, DenseRank, Expression, NamedExpression, Rank, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedFollowing, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Average, Complete, Count, First, Last, Max, Min, Sum}
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, EmptyRow, FrameLessOffsetWindowFunction, Literal, NthValue}
import org.apache.spark.sql.types.{DateType, IntegerType, StringType, TimestampType, BooleanType, DecimalType => SparkDecimalType}
import org.apache.spark.sql.catalyst.expressions.{CurrentRow, EvalMode, RangeFrame, RowFrame}
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.window.{Final, Partial, WindowExec, WindowGroupLimitExec, WindowGroupLimitMode}
import org.apache.spark.sql.types.{DataType, DecimalType, DoubleType, LongType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

import scala.collection.mutable

/**
 * Columnar replacement for WindowExec in two shapes (#58). Both walk the child's rows in the order
 * Spark already established -- the child arrives sorted by the partition keys and then the order keys,
 * because this operator requires exactly what `WindowExec` does -- and detect a partition boundary
 * where a partition key differs from the previous row's (null-safe equality, as Spark's window
 * functions compare with `<=>`).
 *
 * Ranking (layer 1): `row_number`, `rank` and `dense_rank`. A row starts a new peer group when an
 * order key differs from the previous row's; `row_number` counts rows since the partition began,
 * `rank` is the row number at which the current peer group began, `dense_rank` counts peer groups.
 * The previous row's keys and the counters carry across batches, so a partition longer than a batch
 * is one partition. Streaming: every input batch is one output batch, the input columns borrowed,
 * the ranks new INT32 columns.
 *
 * Whole-partition aggregates (layer 2): `sum`, `avg`, `count`, `min`, `max` and the rest of the
 * grouped aggregate family over the frame `UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` -- Spark's
 * default frame for a spec without `ORDER BY`, the shape of TPC-DS q12, q20, q47, q53, q57, q63, q89,
 * q98. Each partition is a group: the aggregate functions of the grouped hash aggregate run over the
 * batch with the partition ordinal as the group id, and the partition's value is the function's
 * `evaluateExpression` over its buffers, exactly as the Final aggregate computes it. Rows are held
 * (copied, in memory, no spill -- like the sort) until their partition ends, since the value is
 * known only then, and are emitted batch by batch with the values gathered per row.
 *
 * Running frames (layer 2b): `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and `RANGE BETWEEN
 * UNBOUNDED PRECEDING AND CURRENT ROW` -- the latter Spark's default for an aggregate over a spec WITH
 * `ORDER BY`, where every row of a peer group gets the value at the end of its group. Each row (ROWS)
 * or peer group (RANGE) is a group; the same grouped states accumulate them, and at emission a group's
 * value is its partition's prefix: the previous group's running buffers combined with its own, slot by
 * slot -- sums and counts add (a bigint sum with Spark's ANSI overflow), `min`/`max` compare -- so the
 * function's `evaluateExpression` over the prefixed buffers is the running value. `sum`, `avg`,
 * `count`, `min` and `max` have that prefix form; the other functions do not and are refused.
 *
 * The child may be a row operator: Spark plans `Window` above `Sort` above an exchange, and without
 * Comet's shuffle that sort stays Spark's, so `RowToColumnarExec` is inserted below us; from here up
 * the chain (a filter on the rank, a projection, a limit) is columnar again.
 *
 * Offset functions (layer 3): `lag`, `lead`, `first_value`, `last_value` and `nth_value` are one row
 * of the partition each -- a row shifted by a literal offset (the default outside the partition), the
 * partition's first row, the frame's last row (the partition; the current row for `ROWS ... CURRENT
 * ROW`; the end of the peer group for the `RANGE` default), the frame's n-th row -- read from the held
 * copies of the partition's rows, across batch boundaries; `IGNORE NULLS` is refused.
 *
 * Sliding frames (`n PRECEDING`, suffixes), `percent_rank`, `cume_dist` and `ntile` are later layers;
 * each is refused with a reason naming the function and frame.
 */
case class VectorWindowExec(
    windowExpression: Seq[NamedExpression],
    partitionSpec: Seq[Expression],
    orderSpec: Seq[SortOrder],
    child: SparkPlan)
    extends VectorExec {

  override def output: Seq[Attribute] = child.output ++ windowExpression.map(_.toAttribute)

  /** Same contracts as WindowExec, so the exchange and sort Spark planned below are the right ones. */
  override def requiredChildDistribution: Seq[Distribution] =
    if (partitionSpec.isEmpty) AllTuples :: Nil else ClusteredDistribution(partitionSpec) :: Nil
  override def requiredChildOrdering: Seq[Seq[SortOrder]] = Seq(partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec)
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  private def compileKey(e: Expression): VectorExpr = ExpressionCompiler.compile(e, child.output) match {
    case Right(v) => v
    case Left(reason) => throw new IllegalStateException(s"cannot vectorize window key ${e.sql}: $reason")
  }

  // A constant key never separates two rows (`ORDER BY length('abc')` folds to a literal: every row is a
  // peer), so literals are dropped rather than evaluated as columns.
  @transient private lazy val partitionKeys: Array[VectorExpr] = partitionSpec.map(compileKey).filterNot(_.isInstanceOf[LiteralExpr]).toArray
  @transient private lazy val orderKeys: Array[VectorExpr] = orderSpec.map(o => compileKey(o.child)).filterNot(_.isInstanceOf[LiteralExpr]).toArray

  private def aggregateMode: Boolean = VectorWindowPlanner.aggregateWindows(windowExpression).isDefined
  private def offsetMode: Boolean = !aggregateMode && VectorWindowPlanner.offsetWindows(windowExpression, child.output).isRight
  @transient private lazy val offsets: Array[VectorWindowPlanner.OffsetFunction] =
    VectorWindowPlanner.offsetWindows(windowExpression, child.output).getOrElse(throw new IllegalStateException("window expressions are not offset functions")).toArray

  @transient private lazy val kinds: Array[Int] = windowExpression.map(e => VectorWindowPlanner.rankKind(e).getOrElse(
    throw new IllegalStateException(s"cannot vectorize window function ${e.sql}"))).toArray

  @transient private lazy val (aggregates: Seq[AggregateExpression], frame: Int) = VectorWindowPlanner.aggregateWindows(windowExpression).getOrElse(
    throw new IllegalStateException("window expressions are not aggregates over one supported frame"))
  @transient private lazy val combiners: Array[Array[VectorWindowPlanner.Combine]] =
    if (frame == VectorWindowPlanner.WholePartition) null
    else aggregates.map(a => VectorWindowPlanner.prefixCombiners(a).getOrElse(throw new IllegalStateException(s"no running frame for ${a.sql}"))).toArray
  @transient private lazy val aggFunctions: Array[VectorAggFunction] = aggregates.map { a =>
    VectorAggregates.compile(a, child.output) match {
      case Right(f) => f
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize window aggregate ${a.sql}: $reason")
    }
  }.toArray
  @transient private lazy val aggResults: Array[VectorExpr] = {
    val attrs = aggregates.map(_.resultAttribute)
    VectorAggregatePlanner.compileFinalResults(Nil, aggregates, attrs, attrs) match {
      case Right(exprs) => exprs.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize window aggregate results: $reason")
    }
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val pk = partitionKeys
    val childAttrs = child.output.map(a => (a.name, a.dataType)).toArray
    val windowAttrs = windowExpression.map(e => (e.name, e.dataType)).toArray
    val m = vectorMetrics
    if (aggregateMode) {
      val aggs = aggFunctions
      val results = aggResults
      val layout = VectorAggregatePlanner.bufferLayout(Nil, aggregates).toArray
      val ok = orderKeys
      val fr = frame
      val cb = combiners
      val bufferAttrs = VectorAggregatePlanner.bufferAttributes(Nil, aggregates).map(a => (a.name, a.dataType)).toArray
      child.executeColumnar().mapPartitionsInternal { iter =>
        new VectorWindowAggregateIterator(iter, pk, ok, fr, cb, aggs, layout, bufferAttrs, results, childAttrs, windowAttrs, m)
      }
    } else if (offsetMode) {
      val ok = orderKeys
      val fs = offsets
      child.executeColumnar().mapPartitionsInternal { iter =>
        new VectorWindowOffsetIterator(iter, pk, ok, fs, childAttrs, windowAttrs, m)
      }
    } else {
      val ok = orderKeys
      val ks = kinds
      child.executeColumnar().mapPartitionsInternal { iter =>
        new VectorWindowIterator(iter, pk, ok, ks, childAttrs, windowAttrs, m)
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Window: ${windowExpression.map(_.sql).mkString(", ")}
       |Partition: ${partitionSpec.map(_.sql).mkString(", ")}
       |Order: ${orderSpec.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

object VectorWindowPlanner {
  val RowNumberKind = 0
  val RankKind = 1
  val DenseRankKind = 2

  /** Which ranking function a window expression is, or why it is not one this layer computes. */
  def rankKind(e: NamedExpression): Either[String, Int] = e match {
    case Alias(WindowExpression(f, _), _) => f match {
      case _: RowNumber => Right(RowNumberKind)
      case _: Rank => Right(RankKind)
      case _: DenseRank => Right(DenseRankKind)
      // Only reached beside a ranking function: the aggregate frames themselves are judged first.
      case a: AggregateExpression => Left(s"window aggregate ${a.aggregateFunction.prettyName} beside a ranking function in one operator not supported")
      case other => Left(s"window function ${other.prettyName} not supported (row_number, rank and dense_rank are)")
    }
    case other => Left(s"window expression ${other.sql} not supported")
  }

  /** The ranking kind of a bare ranking function (WindowGroupLimitExec's `rankLikeFunction`). */
  def rankLikeKind(f: Expression): Either[String, Int] = f match {
    case _: RowNumber => Right(RowNumberKind)
    case _: Rank => Right(RankKind)
    case _: DenseRank => Right(DenseRankKind)
    case other => Left(s"window group limit function ${other.prettyName} not supported")
  }

  /** Aggregate frames this operator computes. */
  val WholePartition = 0
  /** `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: a prefix over rows. */
  val RunningRows = 1
  /** `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` (Spark's default with `ORDER BY`): a prefix over peer groups. */
  val RunningRange = 2

  /** Combines the running value so far with the next group's buffer slot; either side may be null. */
  type Combine = (Any, Any) => Any

  private def frameKind(f: Expression): Option[Int] = f match {
    case SpecifiedWindowFrame(_, UnboundedPreceding, UnboundedFollowing) => Some(WholePartition)
    case SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow) => Some(RunningRows)
    case SpecifiedWindowFrame(RangeFrame, UnboundedPreceding, CurrentRow) => Some(RunningRange)
    case _ => None
  }

  /** The aggregate and frame kind of `e`, if it is a Complete aggregate without FILTER over a frame we compute. */
  private def aggregateWindow(e: NamedExpression): Option[(AggregateExpression, Int)] = e match {
    // first_value / last_value are positions in the frame, not prefixes: the offset family computes them.
    case Alias(WindowExpression(a: AggregateExpression, _), _) if a.aggregateFunction.isInstanceOf[First] || a.aggregateFunction.isInstanceOf[Last] => None
    case Alias(WindowExpression(a: AggregateExpression, WindowSpecDefinition(_, _, f)), _) if a.mode == Complete && a.filter.isEmpty =>
      frameKind(f).map(k => (a, k))
    case _ => None
  }

  /** Offset-family kinds: a shifted row, the frame's first row, the frame's last row, the frame's n-th row. */
  val Shift = 0
  val FirstValue = 1
  val LastValue = 2
  val NthValueKind = 3

  /**
   * A window function whose value is one row of the partition, read from the held rows: `kind` says
   * which row, `frame` (an aggregate frame kind) where the frame ends for the position-based ones,
   * `default` (Spark's internal representation) what a shifted row outside the partition yields.
   */
  final case class OffsetFunction(kind: Int, inputOrdinal: Int, dataType: DataType, offset: Int, default: Any, frame: Int)

  private def literalInt(e: Expression): Option[Int] = e match {
    case l: Literal if l.foldable && l.value != null && (l.dataType == IntegerType) => Some(l.value.asInstanceOf[Int])
    case other if other.foldable && other.dataType == IntegerType => Option(other.eval(EmptyRow)).map(_.asInstanceOf[Int])
    case _ => None
  }

  /** A foldable default in Spark's internal representation, with decimals as their unscaled long (the INT64 lane). */
  private def literalDefault(e: Expression, dt: DataType): Either[String, Any] =
    if (!e.foldable) Left("default value must be a literal")
    else e.eval(EmptyRow) match {
      case null => Right(null)
      case d: org.apache.spark.sql.types.Decimal => Right(java.lang.Long.valueOf(d.toUnscaledLong))
      case v => Right(v)
    }

  private def inputOrdinal(input: Expression, output: Seq[Attribute]): Either[String, Int] = input match {
    case a: AttributeReference =>
      val i = output.indexWhere(_.exprId == a.exprId)
      if (i < 0) Left(s"input ${a.sql} is not a column of the child")
      else if (!TypeMapping.isSupported(a.dataType)) Left(s"unsupported column type ${a.dataType.simpleString} for ${a.name}")
      else Right(i)
    case other => Left(s"input ${other.sql} is not a column (Spark projects complex inputs below the window)")
  }

  /** `e` as an offset-family function, or why it is not one. */
  private def offsetWindow(e: NamedExpression, output: Seq[Attribute]): Either[String, OffsetFunction] = e match {
    case Alias(WindowExpression(f: FrameLessOffsetWindowFunction, _), _) =>
      if (f.ignoreNulls) Left(s"${f.prettyName} IGNORE NULLS not supported")
      else literalInt(f.offset) match {
        case None => Left(s"${f.prettyName} offset must be an integer literal")
        case Some(off) =>
          for (ord <- inputOrdinal(f.input, output); dflt <- literalDefault(f.default, f.dataType).left.map(r => s"${f.prettyName}: $r"))
            yield OffsetFunction(Shift, ord, f.dataType, off, dflt, WholePartition)
      }
    case Alias(WindowExpression(n: NthValue, WindowSpecDefinition(_, _, frame)), _) =>
      if (n.ignoreNulls) Left("nth_value IGNORE NULLS not supported")
      else (literalInt(n.offset), frameKind(frame)) match {
        case (None, _) => Left("nth_value offset must be an integer literal")
        case (_, None) => Left(s"nth_value over frame ${frame.sql} not supported")
        case (Some(k), Some(fk)) => inputOrdinal(n.input, output).map(ord => OffsetFunction(NthValueKind, ord, n.dataType, k, null, fk))
      }
    case Alias(WindowExpression(a: AggregateExpression, WindowSpecDefinition(_, _, frame)), _) if a.aggregateFunction.isInstanceOf[First] || a.aggregateFunction.isInstanceOf[Last] =>
      val (child, ignoreNulls, isFirst) = a.aggregateFunction match {
        case fst: First => (fst.child, fst.ignoreNulls, true)
        case lst: Last => (lst.child, lst.ignoreNulls, false)
      }
      val name = a.aggregateFunction.prettyName
      if (ignoreNulls) Left(s"$name IGNORE NULLS not supported")
      else if (a.filter.nonEmpty) Left(s"$name with FILTER not supported")
      else frameKind(frame) match {
        case None => Left(s"$name over frame ${frame.sql} not supported")
        case Some(fk) => inputOrdinal(child, output).map(ord => OffsetFunction(if (isFirst) FirstValue else LastValue, ord, a.dataType, 0, null, fk))
      }
    case Alias(WindowExpression(f, _), _) => Left(s"window function ${f.prettyName} is not an offset function")
    case other => Left(s"window expression ${other.sql} not supported")
  }

  /** All of the operator's expressions as offset-family functions, or the first reason one is not. */
  def offsetWindows(es: Seq[NamedExpression], output: Seq[Attribute]): Either[String, Seq[OffsetFunction]] = {
    val all = es.map(offsetWindow(_, output))
    all.collectFirst { case Left(r) => r }.toLeft(all.collect { case Right(f) => f })
  }

  /** All of the operator's expressions as aggregates over ONE frame kind, or None when any is something else. */
  def aggregateWindows(es: Seq[NamedExpression]): Option[(Seq[AggregateExpression], Int)] = {
    val aggs = es.map(aggregateWindow)
    if (es.nonEmpty && aggs.forall(_.isDefined) && aggs.flatten.map(_._2).distinct.length == 1) Some((aggs.flatten.map(_._1), aggs.flatten.head._2))
    else None
  }

  private def nullSafe(op: (Any, Any) => Any): Combine = (p, c) => if (p == null) c else if (c == null) p else op(p, c)
  private def addLong(checked: Boolean, context: org.apache.spark.QueryContext): Combine = nullSafe { (p, c) =>
    val a = p.asInstanceOf[java.lang.Long].longValue(); val b = c.asInstanceOf[java.lang.Long].longValue()
    if (checked) {
      try java.lang.Long.valueOf(Math.addExact(a, b))
      catch { case _: ArithmeticException => throw VectorErrors.arithmeticOverflow("long overflow", "try_sum", context) }
    } else java.lang.Long.valueOf(a + b)
  }
  private val addDouble: Combine = nullSafe((p, c) => java.lang.Double.valueOf(p.asInstanceOf[java.lang.Double].doubleValue() + c.asInstanceOf[java.lang.Double].doubleValue()))
  private def extreme(isMin: Boolean): Combine = nullSafe { (p, c) =>
    // Boxed Integer / Long / Double / Boolean / UTF8String: Comparable with Spark's ordering (doubles total, strings binary).
    val cmp = p.asInstanceOf[Comparable[Any]].compareTo(c)
    if ((cmp <= 0) == isMin) p else c
  }

  /** How a running frame carries each buffer slot forward, or why this function has no running form here. */
  def prefixCombiners(a: AggregateExpression): Either[String, Array[Combine]] = a.aggregateFunction match {
    case s: Sum if s.evalContext.evalMode == EvalMode.TRY => Left("running try_sum not supported")
    case s: Sum if s.dataType == LongType => Right(Array(addLong(s.evalContext.evalMode == EvalMode.ANSI, s.origin.context)))
    case s: Sum if s.dataType == DoubleType => Right(Array(addDouble))
    case _: Count => Right(Array(addLong(checked = false, null)))
    case av: Average if av.dataType == DoubleType => Right(Array(addDouble, addLong(checked = false, null)))
    case _: Min => Right(Array(extreme(isMin = true)))
    case _: Max => Right(Array(extreme(isMin = false)))
    case other => Left(s"running frame for ${other.prettyName} not supported (sum, avg, count, min and max are)")
  }

  /** Why a window aggregate is not computed: the frame, or the function itself. */
  private def aggregateReason(e: NamedExpression, input: Seq[Attribute]): Option[String] = e match {
    case Alias(WindowExpression(a: AggregateExpression, _), _) if a.aggregateFunction.isInstanceOf[First] || a.aggregateFunction.isInstanceOf[Last] => None
    case Alias(WindowExpression(a: AggregateExpression, WindowSpecDefinition(_, _, frame)), _) =>
      frameKind(frame) match {
        case Some(kind) =>
          // A decimal sum's buffer is Spark's Decimal(p + 10): the 128-bit lane (#28) before this can gather it.
          if (a.dataType.isInstanceOf[DecimalType] || a.aggregateFunction.children.exists(_.dataType.isInstanceOf[DecimalType]))
            Some(s"window aggregate ${a.aggregateFunction.prettyName} over decimals not supported (its buffer is a wide decimal)")
          else VectorAggregates.compile(a, input).left.toOption.map(r => s"window aggregate ${a.aggregateFunction.prettyName}: $r")
            .orElse(if (kind == WholePartition) None else prefixCombiners(a).left.toOption.map(r => s"window aggregate ${a.aggregateFunction.prettyName}: $r"))
        case None => Some(s"window aggregate ${a.aggregateFunction.prettyName} over frame ${frame.sql} not supported (whole-partition and running frames are)")
      }
    case _ => None
  }

  def plan(w: WindowExec): Either[String, VectorWindowExec] = {
    val keys = w.partitionSpec ++ w.orderSpec.map(_.child)
    val keyFailures = keys.flatMap { k =>
      if (k.dataType == DoubleType) Some(s"window key ${k.sql}: double keys not supported (Spark compares them after NaN and zero normalisation)")
      else if (!TypeMapping.isSupported(k.dataType)) Some(s"window key type ${k.dataType.simpleString} not supported")
      else ExpressionCompiler.compile(k, w.child.output).left.toOption.map(r => s"window key ${k.sql}: $r")
    }
    aggregateWindows(w.windowExpression) match {
      case Some((aggs, _)) =>
        val reasons = w.windowExpression.flatMap(aggregateReason(_, w.child.output))
        val attrs = aggs.map(_.resultAttribute)
        val resultReason = if (reasons.nonEmpty) None else VectorAggregatePlanner.compileFinalResults(Nil, aggs, attrs, attrs).left.toOption.map(r => s"window aggregate result: $r")
        (reasons ++ resultReason ++ keyFailures).headOption.toLeft(VectorWindowExec(w.windowExpression, w.partitionSpec, w.orderSpec, w.child))
      case None =>
        val kinds = w.windowExpression.map(rankKind)
        kinds.collectFirst { case Left(reason) => reason } match {
          case Some(reason) =>
            offsetWindows(w.windowExpression, w.child.output) match {
              case Right(_) => keyFailures.headOption.toLeft(VectorWindowExec(w.windowExpression, w.partitionSpec, w.orderSpec, w.child))
              case Left(offsetReason) =>
                // A ranking function beside an aggregate in one spec, an aggregate over another frame, two
                // frame kinds in one operator, or an offset function we cannot read: say which.
                val mixed = w.windowExpression.map(aggregateWindow).flatten.map(_._2).distinct.length > 1
                val offsetLike = w.windowExpression.exists {
                  case Alias(WindowExpression(f, _), _) => f.isInstanceOf[FrameLessOffsetWindowFunction] || f.isInstanceOf[NthValue] ||
                    (f match { case a: AggregateExpression => a.aggregateFunction.isInstanceOf[First] || a.aggregateFunction.isInstanceOf[Last]; case _ => false })
                  case _ => false
                }
                Left(w.windowExpression.flatMap(aggregateReason(_, w.child.output)).headOption.getOrElse(
                  if (mixed) "window aggregates over different frames in one operator not supported"
                  else if (offsetLike && !offsetReason.contains("is not an offset function")) offsetReason
                  else if (offsetLike) "offset functions beside other window functions in one operator not supported"
                  else reason))
            }
          case None if w.orderSpec.isEmpty => Left("ranking window without an ORDER BY")
          case None => keyFailures.headOption.toLeft(VectorWindowExec(w.windowExpression, w.partitionSpec, w.orderSpec, w.child))
        }
    }
  }
}

/** Detects partition (or peer) boundaries: null-safe equality of a row's keys with the previous row's. */
private[vector] final class KeyTracker(keys: Array[VectorExpr]) {
  private val previous = new Array[Any](keys.length)
  private var lanes: Array[VectorBuffers] = _
  var hasPrevious = false

  def startBatch(ctx: io.sparkvector.spark.expr.EvalContext): Unit = lanes = keys.map(_.eval(ctx))

  /** True when row `i` differs from the remembered row (or none is remembered). */
  def changed(i: Int): Boolean = {
    if (!hasPrevious) return true
    var k = 0
    while (k < lanes.length) {
      val current = if (Rows.valid(lanes(k), i)) Rows.box(lanes(k), i) else null
      val before = previous(k)
      if (!(if (before == null) current == null else before.equals(current))) return true
      k += 1
    }
    false
  }

  def remember(i: Int): Unit = {
    var k = 0
    while (k < lanes.length) {
      previous(k) = if (Rows.valid(lanes(k), i)) Rows.box(lanes(k), i) else null
      k += 1
    }
    hasPrevious = true
  }
}

/** The ranking walk over sorted batches; the previous row's keys and the counters carry across batches. */
private[vector] class VectorWindowIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    orderKeys: Array[VectorExpr],
    kinds: Array[Int],
    childAttrs: Array[(String, DataType)],
    windowAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorWindowExec") {

  private val partition = new KeyTracker(partitionKeys)
  private val order = new KeyTracker(orderKeys)
  private var rowNumber = 0L
  private var rank = 0L
  private var denseRank = 0L

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      val outRows = if (live == null) n else ctx.selectedCount
      partition.startBatch(ctx)
      order.startBatch(ctx)
      val results = new Array[Array[Int]](kinds.length)
      var f = 0
      while (f < kinds.length) { results(f) = new Array[Int](outRows); f += 1 }
      var out = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = partition.changed(i)
          if (newPartition) {
            rowNumber = 0L
            rank = 0L
            denseRank = 0L
          }
          rowNumber += 1
          val peer = !newPartition && !order.changed(i)
          if (!peer) {
            rank = rowNumber
            denseRank += 1
          }
          partition.remember(i)
          order.remember(i)
          f = 0
          while (f < kinds.length) {
            results(f)(out) = kinds(f) match {
              case VectorWindowPlanner.RowNumberKind => rowNumber.toInt
              case VectorWindowPlanner.RankKind => rank.toInt
              case _ => denseRank.toInt
            }
            f += 1
          }
          out += 1
        }
        i += 1
      }
      val columns = new Array[ColumnVector](childAttrs.length + kinds.length)
      var c = 0
      while (c < childAttrs.length) {
        val (name, dt) = childAttrs(c)
        // Forwarded columns are never copied: the child keeps them alive until its next batch.
        columns(c) = if (live == null) BorrowedColumnVector.of(batch.column(c)) else ArrowOutput.compact(name, dt, ctx.input(c), live, outRows, allocator)
        c += 1
      }
      f = 0
      while (f < kinds.length) {
        val (name, dt) = windowAttrs(f)
        val buffers = ArrowOutput.allocateFixed(name, dt, outRows, allocator)
        val data = buffers.data()
        var r = 0
        while (r < outRows) { data.setAtIndex(VectorBuffers.LE_INT, r, results(f)(r)); r += 1 }
        columns(childAttrs.length + f) = ArrowOutput.finish(buffers, outRows, true)
        f += 1
      }
      metrics.numOutputBatches += 1
      metrics.numOutputRows += outRows
      new ColumnarBatch(columns, outRows)
    }
  }
}

/**
 * Whole-partition aggregates over sorted batches. Partitions are groups numbered in arrival order;
 * the grouped aggregate states accumulate every batch with the partition ordinal as the group id.
 * A batch is held (its live rows copied) until the partition its last row belongs to has ended --
 * the next batch starts a new one, or the input ends -- and is then emitted with the values of its
 * partitions gathered per row.
 */
private[vector] class VectorWindowAggregateIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    orderKeys: Array[VectorExpr],
    frame: Int,
    combiners: Array[Array[VectorWindowPlanner.Combine]],
    aggs: Array[VectorAggFunction],
    layout: Array[OutputSlot],
    bufferAttrs: Array[(String, DataType)],
    results: Array[VectorExpr],
    childAttrs: Array[(String, DataType)],
    windowAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch] with AutoCloseable {

  /** A held batch: owned copies of the live rows and each row's partition ordinal. */
  private final class Held(val columns: Array[ColumnVector], val numRows: Int, val groups: Array[Int])

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorWindowExec")
  private val states: Array[GroupedAggState] = aggs.map(_.newGroupedState())
  private val partition = new KeyTracker(partitionKeys)
  private val order = new KeyTracker(orderKeys)
  private var numGroups = 0
  // Running frames: groups are rows or peer groups, and a group's value is the prefix of its partition's
  // buffers up to it. `partitionStarts` marks the groups that begin a partition; `lastPrefix` is the
  // prefix of group `prefixDone - 1`, the only earlier group a later batch can still reference.
  private val partitionStarts = mutable.BitSet.empty
  private var prefixDone = 0
  private var lastPrefix: Array[Array[Any]] = _
  private val held = mutable.Queue.empty[Held]
  private val ready = mutable.Queue.empty[ColumnarBatch]
  private var idScratch = new Array[Int](0)
  private var inputDone = false
  private var current: ColumnarBatch = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  private def consume(batch: ColumnarBatch): Unit = metrics.timed {
    metrics.numInputBatches += 1
    EvalContexts.withBatch(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      val outRows = if (live == null) n else ctx.selectedCount
      if (idScratch.length < n) idScratch = new Array[Int](n)
      partition.startBatch(ctx)
      order.startBatch(ctx)
      val groups = new Array[Int](outRows)
      var out = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = partition.changed(i)
          val newGroup = frame match {
            case VectorWindowPlanner.RunningRows => true
            case VectorWindowPlanner.RunningRange => newPartition || order.changed(i)
            case _ => newPartition
          }
          if (newGroup) {
            numGroups += 1
            if (newPartition) partitionStarts += numGroups - 1
          }
          partition.remember(i)
          order.remember(i)
          idScratch(i) = numGroups - 1
          groups(out) = numGroups - 1
          out += 1
        } else idScratch(i) = -1
        i += 1
      }
      val assignment = GroupAssignment.of(idScratch, n, numGroups, ctx.arena, live)
      var s = 0
      while (s < states.length) { states(s).update(ctx, assignment); s += 1 }
      val columns = new Array[ColumnVector](childAttrs.length)
      var c = 0
      while (c < childAttrs.length) {
        val (name, dt) = childAttrs(c)
        // The child reuses or releases its batch once we pull the next one: the held rows are copies.
        columns(c) = if (live == null) ArrowOutput.copy(name, dt, ctx.input(c), allocator) else ArrowOutput.compact(name, dt, ctx.input(c), live, outRows, allocator)
        c += 1
      }
      held.enqueue(new Held(columns, outRows, groups))
    }
  }

  /** Emits every held batch whose partitions have all ended (all of them once the input is done). */
  private def release(): Unit = metrics.timed {
    while (held.nonEmpty && (inputDone || held.head.groups(held.head.numRows - 1) < numGroups - 1)) {
      val h = held.dequeue()
      val from = h.groups(0)
      val to = h.groups(h.numRows - 1) + 1
      val idx = new Array[Int](h.numRows)
      var r = 0
      while (r < h.numRows) { idx(r) = h.groups(r) - from; r += 1 }
      // The groups' buffers as a batch of `to - from` rows -- for a running frame the prefix within the
      // partition -- the results evaluated over it and gathered per row.
      val prefixes = if (combiners == null) null else prefixOf(from, to)
      val bufferColumns = new Array[ColumnVector](layout.length)
      var c = 0
      while (c < layout.length) {
        val (name, dt) = bufferAttrs(c)
        bufferColumns(c) = layout(c) match {
          case BufferSlot(aggIdx, slot) if prefixes == null => AggBufferColumns.column(name, dt, states(aggIdx), slot, from, to, allocator)
          case BufferSlot(aggIdx, slot) => AggBufferColumns.values(name, dt, to - from, o => prefixes(o)(aggIdx)(slot), allocator)
          case KeySlot(_) => throw new IllegalStateException("key slot in a window aggregate")
        }
        c += 1
      }
      val buffers = new ColumnarBatch(bufferColumns, to - from)
      val columns = new Array[ColumnVector](childAttrs.length + results.length)
      try {
        EvalContexts.withBatch(buffers) { ctx =>
          var f = 0
          while (f < results.length) {
            val (name, dt) = windowAttrs(f)
            columns(childAttrs.length + f) = results(f) match {
              case lit: LiteralExpr => ArrowOutput.constant(name, dt, lit.value, h.numRows, allocator)
              case e => ArrowOutput.gather(name, dt, e.eval(ctx), idx, 0, h.numRows, allocator)
            }
            f += 1
          }
        }
      } finally buffers.close()
      System.arraycopy(h.columns, 0, columns, 0, childAttrs.length)
      metrics.numOutputBatches += 1
      metrics.numOutputRows += h.numRows
      ready.enqueue(new ColumnarBatch(columns, h.numRows))
    }
  }

  /** The running values of groups `[from, to)`: each group's buffers combined with the prefix before it in its partition. */
  private def prefixOf(from: Int, to: Int): Array[Array[Array[Any]]] = {
    val out = new Array[Array[Array[Any]]](to - from)
    var g = from
    while (g < to) {
      if (g == prefixDone - 1) {
        out(g - from) = lastPrefix
      } else {
        require(g == prefixDone, s"running frame groups released out of order: $g after $prefixDone")
        val value = new Array[Array[Any]](aggs.length)
        var a = 0
        while (a < aggs.length) {
          val slots = new Array[Any](combiners(a).length)
          var k = 0
          while (k < slots.length) {
            val current = states(a).bufferValue(g, k)
            slots(k) = if (partitionStarts.contains(g)) current else combiners(a)(k)(lastPrefix(a)(k), current)
            k += 1
          }
          value(a) = slots
          a += 1
        }
        lastPrefix = value
        prefixDone = g + 1
        out(g - from) = value
      }
      g += 1
    }
    out
  }

  override def hasNext: Boolean = {
    while (ready.isEmpty && !inputDone) {
      if (input.hasNext) {
        val batch = input.next()
        if (batch.numRows() > 0) consume(batch)
      } else inputDone = true
      release()
    }
    ready.nonEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseCurrent()
    current = ready.dequeue()
    current
  }

  private def releaseCurrent(): Unit = if (current != null) { current.close(); current = null }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseCurrent()
      ready.foreach(_.close())
      ready.clear()
      held.foreach(_.columns.foreach(_.close()))
      held.clear()
      allocator.close()
    }
  }
}

/**
 * Columnar replacement for WindowGroupLimitExec (#58, layer 4): Spark's per-partition top-k below a
 * ranking window, inserted for a filter `rank <= k` (k up to `spark.sql.optimizer.windowGroupLimitThreshold`)
 * -- in Partial mode below the exchange, over whatever produced the rows, and in Final mode above the
 * sort. Both modes require the same ordering as the window (partition keys, then order keys) and the
 * Final mode the window's distribution, so the walk is the ranking walk of [[VectorWindowIterator]]
 * with a filter: a row is kept while the ranking function's value at that row is at most `limit`.
 * Spark's own operator also lets the first row of the peer group past the limit through (it checks the
 * previous row's rank); both are pre-filters under the window that computes the real ranks, so the
 * tighter set is exact for the query and smaller for the shuffle.
 *
 * Output rows are compacted (a top-k keeps a few rows per partition); a batch with every row kept is
 * forwarded, one with none is dropped.
 */
case class VectorWindowGroupLimitExec(
    partitionSpec: Seq[Expression],
    orderSpec: Seq[SortOrder],
    rankLikeFunction: Expression,
    limit: Int,
    mode: WindowGroupLimitMode,
    child: SparkPlan)
    extends VectorExec {

  override def output: Seq[Attribute] = child.output

  override def requiredChildDistribution: Seq[Distribution] = mode match {
    case Partial => super.requiredChildDistribution
    case Final => if (partitionSpec.isEmpty) AllTuples :: Nil else ClusteredDistribution(partitionSpec) :: Nil
  }
  override def requiredChildOrdering: Seq[Seq[SortOrder]] = Seq(partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec)
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  private def compileKey(e: Expression): VectorExpr = ExpressionCompiler.compile(e, child.output) match {
    case Right(v) => v
    case Left(reason) => throw new IllegalStateException(s"cannot vectorize window group limit key ${e.sql}: $reason")
  }
  @transient private lazy val partitionKeys: Array[VectorExpr] = partitionSpec.map(compileKey).filterNot(_.isInstanceOf[LiteralExpr]).toArray
  @transient private lazy val orderKeys: Array[VectorExpr] = orderSpec.map(o => compileKey(o.child)).filterNot(_.isInstanceOf[LiteralExpr]).toArray
  @transient private lazy val kind: Int = VectorWindowPlanner.rankLikeKind(rankLikeFunction).getOrElse(
    throw new IllegalStateException(s"cannot vectorize window group limit function ${rankLikeFunction.sql}"))

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val pk = partitionKeys
    val ok = orderKeys
    val k = kind
    val lim = limit
    val childAttrs = child.output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorWindowGroupLimitIterator(iter, pk, ok, k, lim, childAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Function: ${rankLikeFunction.sql} <= $limit ($mode)
       |Partition: ${partitionSpec.map(_.sql).mkString(", ")}
       |Order: ${orderSpec.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

object VectorWindowGroupLimitPlanner {
  def plan(g: WindowGroupLimitExec): Either[String, VectorWindowGroupLimitExec] = {
    VectorWindowPlanner.rankLikeKind(g.rankLikeFunction) match {
      case Left(reason) => Left(reason)
      case Right(_) =>
        val keys = g.partitionSpec ++ g.orderSpec.map(_.child)
        val keyFailures = keys.flatMap { k =>
          if (k.dataType == DoubleType) Some(s"window key ${k.sql}: double keys not supported (Spark compares them after NaN and zero normalisation)")
          else if (!TypeMapping.isSupported(k.dataType)) Some(s"window key type ${k.dataType.simpleString} not supported")
          else ExpressionCompiler.compile(k, g.child.output).left.toOption.map(r => s"window key ${k.sql}: $r")
        }
        keyFailures.headOption.toLeft(VectorWindowGroupLimitExec(g.partitionSpec, g.orderSpec, g.rankLikeFunction, g.limit, g.mode, g.child))
    }
  }
}

/** The ranking walk with a filter: rows whose ranking value is at most `limit` survive. */
private[vector] class VectorWindowGroupLimitIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    orderKeys: Array[VectorExpr],
    kind: Int,
    limit: Int,
    childAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorWindowGroupLimitExec") {

  private val partition = new KeyTracker(partitionKeys)
  private val order = new KeyTracker(orderKeys)
  private var rowNumber = 0L
  private var rank = 0L
  private var denseRank = 0L

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      partition.startBatch(ctx)
      order.startBatch(ctx)
      val kept = Bitmap.allocate(ctx.arena, n)
      var keptCount = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = partition.changed(i)
          if (newPartition) {
            rowNumber = 0L
            rank = 0L
            denseRank = 0L
          }
          rowNumber += 1
          val peer = !newPartition && !order.changed(i)
          if (!peer) {
            rank = rowNumber
            denseRank += 1
          }
          partition.remember(i)
          order.remember(i)
          val value = kind match {
            case VectorWindowPlanner.RowNumberKind => rowNumber
            case VectorWindowPlanner.RankKind => rank
            case _ => denseRank
          }
          if (value <= limit) { Bitmap.set(kept, i); keptCount += 1 }
        }
        i += 1
      }
      if (keptCount == 0) {
        null
      } else if (live == null && keptCount == n) {
        metrics.numOutputBatches += 1
        metrics.numOutputRows += n
        batch
      } else {
        val columns = new Array[ColumnVector](childAttrs.length)
        var c = 0
        while (c < childAttrs.length) {
          val (name, dt) = childAttrs(c)
          columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), kept, keptCount, allocator)
          c += 1
        }
        metrics.numOutputBatches += 1
        metrics.numOutputRows += keptCount
        new ColumnarBatch(columns, keptCount)
      }
    }
  }
}

/**
 * Offset functions over the held partition: `lag` / `lead` (a row shifted by a literal offset, the default
 * outside the partition), `first_value` (the partition's first row), `last_value` (the last row of the
 * frame: the partition, the current row for `ROWS ... CURRENT ROW`, the end of the peer group for the
 * `RANGE` default) and `nth_value` (the frame's n-th row, null when the frame is shorter). Rows are held
 * as copies until their partition has ended; a released batch stays readable (its columns are forwarded
 * borrowed) until no later batch can still address a row of one of its partitions, since `lag` reads
 * backwards across batch boundaries.
 */
private[vector] class VectorWindowOffsetIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    orderKeys: Array[VectorExpr],
    functions: Array[VectorWindowPlanner.OffsetFunction],
    childAttrs: Array[(String, DataType)],
    windowAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch] with AutoCloseable {

  /** A held batch: owned copies of the live rows, each row's partition ordinal, position in it, and peer ordinal. */
  private final class Held(val columns: Array[ColumnVector], val numRows: Int, val firstGlobal: Long,
      val partOf: Array[Int], val posOf: Array[Int], val peerOf: Array[Int]) {
    /** The consumer has moved past this batch's output (whose columns are borrowed from here). */
    var consumed = false
    def close(): Unit = columns.foreach(_.close())
  }

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorWindowExec")
  private val partition = new KeyTracker(partitionKeys)
  private val order = new KeyTracker(orderKeys)
  private var numPartitions = 0
  private var numPeers = 0
  private var posInPartition = 0
  private var globalRows = 0L
  // Filled when a partition / peer group ends: its length / its last position.
  private val partitionLength = mutable.ArrayBuffer.empty[Int]
  private val peerEnd = mutable.ArrayBuffer.empty[Int]
  private val held = mutable.Queue.empty[Held]
  private val released = mutable.ArrayBuffer.empty[Held]
  // No longer addressable by any unreleased row, closed once the consumer is past their output too.
  private val retired = mutable.ArrayBuffer.empty[Held]
  private val ready = mutable.Queue.empty[(ColumnarBatch, Held)]
  private var inputDone = false
  private var current: ColumnarBatch = _
  private var currentHeld: Held = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  private def pruneRetired(): Unit = {
    val (done, waiting) = retired.partition(_.consumed)
    done.foreach(_.close())
    retired.clear(); retired ++= waiting
  }

  private def endPeer(): Unit = if (numPeers > 0) peerEnd += posInPartition - 1
  private def endPartition(): Unit = if (numPartitions > 0) partitionLength += posInPartition

  private def consume(batch: ColumnarBatch): Unit = metrics.timed {
    metrics.numInputBatches += 1
    EvalContexts.withBatch(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      val outRows = if (live == null) n else ctx.selectedCount
      partition.startBatch(ctx)
      order.startBatch(ctx)
      val partOf = new Array[Int](outRows)
      val posOf = new Array[Int](outRows)
      val peerOf = new Array[Int](outRows)
      var out = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = partition.changed(i)
          if (newPartition) {
            endPeer(); endPartition()
            numPartitions += 1; numPeers += 1; posInPartition = 0
          } else if (order.changed(i)) {
            endPeer(); numPeers += 1
          }
          partition.remember(i)
          order.remember(i)
          partOf(out) = numPartitions - 1
          posOf(out) = posInPartition
          peerOf(out) = numPeers - 1
          posInPartition += 1
          out += 1
        }
        i += 1
      }
      val columns = new Array[ColumnVector](childAttrs.length)
      var c = 0
      while (c < childAttrs.length) {
        val (name, dt) = childAttrs(c)
        columns(c) = if (live == null) ArrowOutput.copy(name, dt, ctx.input(c), allocator) else ArrowOutput.compact(name, dt, ctx.input(c), live, outRows, allocator)
        c += 1
      }
      held.enqueue(new Held(columns, outRows, globalRows, partOf, posOf, peerOf))
      globalRows += outRows
    }
  }

  /** The held batch (released or not) holding global row `g`. */
  private def batchOf(g: Long): Held = {
    var i = released.length - 1
    while (i >= 0) { val h = released(i); if (g >= h.firstGlobal && g < h.firstGlobal + h.numRows) return h; i -= 1 }
    val it = held.iterator
    while (it.hasNext) { val h = it.next(); if (g >= h.firstGlobal && g < h.firstGlobal + h.numRows) return h }
    throw new IllegalStateException(s"row $g is no longer held")
  }

  /** Spark's internal representation of column `c` at row `r` of `h`, or null; decimals as their unscaled long. */
  private def valueAt(h: Held, c: Int, r: Int, dt: DataType): Any = {
    val col = h.columns(c)
    if (col.isNullAt(r)) null
    else dt match {
      case IntegerType | DateType => java.lang.Integer.valueOf(col.getInt(r))
      case LongType | TimestampType => java.lang.Long.valueOf(col.getLong(r))
      case DoubleType => java.lang.Double.valueOf(col.getDouble(r))
      case BooleanType => java.lang.Boolean.valueOf(col.getBoolean(r))
      case StringType => col.getUTF8String(r).clone()
      case d: SparkDecimalType => java.lang.Long.valueOf(col.getDecimal(r, d.precision, d.scale).toUnscaledLong)
      case other => throw new IllegalStateException(s"offset window over $other")
    }
  }

  /** Emits every held batch whose partitions have all ended; keeps released batches readable while needed. */
  private def release(): Unit = metrics.timed {
    while (held.nonEmpty && (inputDone || held.head.partOf(held.head.numRows - 1) < numPartitions - 1)) {
      val h = held.dequeue()
      // Rows of partitions before this batch's first one can no longer be addressed by anything unreleased.
      val firstPart = h.partOf(0)
      val (dead, alive) = released.partition(e => e.partOf(e.numRows - 1) < firstPart)
      retired ++= dead
      released.clear(); released ++= alive; released += h
      pruneRetired()
      val columns = new Array[ColumnVector](childAttrs.length + functions.length)
      var c = 0
      while (c < childAttrs.length) { columns(c) = BorrowedColumnVector.of(h.columns(c)); c += 1 }
      var f = 0
      while (f < functions.length) {
        val fn = functions(f)
        val (name, dt) = windowAttrs(f)
        columns(childAttrs.length + f) = AggBufferColumns.values(name, dt, h.numRows, { r =>
          val part = h.partOf(r); val pos = h.posOf(r); val len = partitionLength(part)
          val frameEnd = fn.frame match {
            case VectorWindowPlanner.WholePartition => len - 1
            case VectorWindowPlanner.RunningRows => pos
            case _ => peerEnd(h.peerOf(r))
          }
          val target: Int = fn.kind match {
            case VectorWindowPlanner.Shift => pos + fn.offset
            case VectorWindowPlanner.FirstValue => 0
            case VectorWindowPlanner.LastValue => frameEnd
            case _ => if (fn.offset - 1 <= frameEnd) fn.offset - 1 else -1
          }
          if (target < 0 || target >= len) { if (fn.kind == VectorWindowPlanner.Shift) fn.default else null }
          else {
            val g = h.firstGlobal + r + (target - pos)
            val src = if (g >= h.firstGlobal && g < h.firstGlobal + h.numRows) h else batchOf(g)
            valueAt(src, fn.inputOrdinal, (g - src.firstGlobal).toInt, fn.dataType)
          }
        }, allocator)
        f += 1
      }
      metrics.numOutputBatches += 1
      metrics.numOutputRows += h.numRows
      ready.enqueue((new ColumnarBatch(columns, h.numRows), h))
    }
  }

  override def hasNext: Boolean = {
    while (ready.isEmpty && !inputDone) {
      if (input.hasNext) {
        val batch = input.next()
        if (batch.numRows() > 0) consume(batch)
      } else {
        inputDone = true
        endPeer(); endPartition()
      }
      release()
    }
    ready.nonEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseCurrent()
    val (batch, h) = ready.dequeue()
    current = batch
    currentHeld = h
    current
  }

  private def releaseCurrent(): Unit = if (current != null) {
    current.close()
    current = null
    currentHeld.consumed = true
    currentHeld = null
    pruneRetired()
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      if (current != null) { current.close(); current = null }
      ready.foreach(_._1.close()); ready.clear()
      held.foreach(_.close()); held.clear()
      released.foreach(_.close()); released.clear()
      retired.foreach(_.close()); retired.clear()
      allocator.close()
    }
  }
}
