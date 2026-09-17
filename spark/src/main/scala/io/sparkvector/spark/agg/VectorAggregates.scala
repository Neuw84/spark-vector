package io.sparkvector.spark.agg

import io.sparkvector.kernels.{AggKernels, Bitmap, CompareOp, GroupAssignment, GroupedAccumulators, VecType, VectorBuffers}
import io.sparkvector.spark.expr.{CastExpr, EvalContext, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.spark.sql.catalyst.expressions.{Attribute, EvalMode, Expression, Literal}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.types.{BooleanType, DataType, DecimalType, DoubleType, LongType, StringType}

/**
 * Running state of one aggregate function within one task. `bufferValues` yields the partial
 * aggregation buffer in Spark's internal representation (boxed, `null` for SQL null), one entry
 * per `aggBufferAttribute` of the corresponding Catalyst function.
 */
trait AggState {
  def update(ctx: EvalContext): Unit
  def bufferValues: Array[Any]
}

/** Running state of one aggregate function over many groups (see [[GroupAssignment]]). */
trait GroupedAggState {
  def update(ctx: EvalContext, groups: GroupAssignment): Unit
  /** Buffer slot value of group `g` in Spark's internal representation, or `null`. */
  def bufferValue(g: Int, slot: Int): Any
}

/** Serializable description of a supported aggregate function; states are created per task. */
trait VectorAggFunction extends Serializable {
  def bufferTypes: Seq[DataType]
  def newState(): AggState
  def newGroupedState(): GroupedAggState
}

/** SUM over doubles: buffer `sum` is null until the first non-null input. */
final case class SumDoubleAgg(input: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType)
  override def newState(): AggState = new AggState {
    private var sum = 0.0
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      val c = AggKernels.countValid(v)
      if (c > 0) { sum += AggKernels.sumDouble(v); count += c }
    }
    override def bufferValues: Array[Any] = Array(if (count == 0) null else java.lang.Double.valueOf(sum))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.DoubleSum
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
    override def bufferValue(g: Int, slot: Int): Any = if (acc.count(g) == 0) null else java.lang.Double.valueOf(acc.sum(g))
  }
}

/**
 * SUM over ints or longs into a long buffer. `checked` is Spark's ANSI mode for a bigint sum: an
 * overflow raises ARITHMETIC_OVERFLOW instead of wrapping.
 */
final case class SumLongAgg(input: VectorExpr, checked: Boolean, queryContext: org.apache.spark.QueryContext) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType)
  private def overflow(): Nothing = throw org.apache.spark.sql.vector.VectorErrors.arithmeticOverflow("long overflow", "try_sum", queryContext)
  override def newState(): AggState = new AggState {
    private var sum = 0L
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      val c = AggKernels.countValid(v)
      if (c > 0) {
        try {
          val s = if (v.`type`() == VecType.INT32) AggKernels.sumInt(v) else if (checked) AggKernels.sumLongExact(v) else AggKernels.sumLong(v)
          sum = if (checked) Math.addExact(sum, s) else sum + s
        } catch { case _: ArithmeticException => overflow() }
        count += c
      }
    }
    override def bufferValues: Array[Any] = Array(if (count == 0) null else java.lang.Long.valueOf(sum))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.LongSum(checked)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit =
      try acc.update(input.eval(ctx), groups) catch { case _: ArithmeticException => overflow() }
    override def bufferValue(g: Int, slot: Int): Any =
      if (acc.count(g) == 0) null
      else try java.lang.Long.valueOf(acc.sum(g)) catch { case _: ArithmeticException => overflow() }
  }
}

/**
 * SUM over a decimal whose sum type is wider than 18 digits -- Spark's `Decimal(p + 10, s)` buffer
 * for an input of more than 8 digits, the range `DecimalAggregates` does not rewrite to a long sum.
 * The update modes only (the merge is #87): the INT64 lane's unscaled values go into a 128-bit
 * accumulator per group, and the buffer is Spark's own two columns -- `sum` as a wide decimal and
 * `isEmpty`. A group without a non-null input has `sum = 0, isEmpty = true` like Spark's initial
 * buffer; a sum that leaves the buffer precision is `null` with `isEmpty = false`, which is what
 * Spark's non-ANSI decimal add leaves behind and what its Final turns into a null or an overflow error.
 */
final case class WideDecimalSumAgg(input: VectorExpr, bufferType: DecimalType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(bufferType, BooleanType)
  private val limit = java.math.BigInteger.TEN.pow(bufferType.precision)

  /** The buffer's `sum` for a group's 128-bit total, or `null` past the buffer precision. */
  private def sumValue(count: Long, total: => java.math.BigInteger): Any =
    if (count == 0) java.math.BigDecimal.valueOf(0L, bufferType.scale)
    else {
      val t = total
      if (t.abs.compareTo(limit) >= 0) null else new java.math.BigDecimal(t, bufferType.scale)
    }

  override def newState(): AggState = new AggState {
    private val acc = new GroupedAccumulators.WideLongSum
    override def update(ctx: EvalContext): Unit = acc.updateAll(ctx.masked(input.eval(ctx)))
    override def bufferValues: Array[Any] = Array(sumValue(acc.count(0), acc.sum(0)), java.lang.Boolean.valueOf(acc.count(0) == 0))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.WideLongSum
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
    override def bufferValue(g: Int, slot: Int): Any =
      if (slot == 0) sumValue(acc.count(g), acc.sum(g)) else java.lang.Boolean.valueOf(acc.count(g) == 0)
  }
}

/**
 * The merge modes of the wide decimal sum: Spark's `(sum: Decimal(p + 10, s), isEmpty)` buffer rows
 * combined per group with Spark's own rules -- `isEmpty = isEmpty && other.isEmpty`, sums added
 * exactly, a null `sum` on a non-empty row meaning "overflowed earlier" and poisoning the group. The
 * wide buffer has no lane: a merge stage sees one buffer row per partition per group, so the column
 * is read row by row from the batch (`getDecimal`, whatever vector the shuffle delivered it in) into
 * an exact per-group total, and none is introduced.
 *
 * In `Final` mode the first slot is already the result -- Spark's `evaluateExpression` is
 * `If(isEmpty, null, CheckOverflowInSum(sum, resultType, nullOnOverflow))`, and the result type is
 * the buffer type -- so the result projection forwards the column: null when the group is empty,
 * null (non-ANSI) or Spark's precision error (ANSI) when the total leaves the declared precision,
 * the total otherwise. In `PartialMerge` mode the slot is the merged buffer.
 */
final case class WideDecimalSumMergeAgg(
    sumOrdinal: Int,
    isEmpty: VectorExpr,
    bufferType: DecimalType,
    finalResult: Boolean,
    nullOnOverflow: Boolean,
    queryContext: org.apache.spark.QueryContext) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(bufferType, BooleanType)
  private val limit = java.math.BigInteger.TEN.pow(bufferType.precision)

  /** Per-group merge state. */
  private final class State(var groups: Int) {
    var total = Array.fill[java.math.BigInteger](groups)(java.math.BigInteger.ZERO)
    var nonEmpty = new Array[Boolean](groups)
    var overflowed = new Array[Boolean](groups)
    def ensure(n: Int): Unit = if (n > groups) {
      total = java.util.Arrays.copyOf(total, n); java.util.Arrays.fill(total.asInstanceOf[Array[AnyRef]], groups, n, java.math.BigInteger.ZERO)
      nonEmpty = java.util.Arrays.copyOf(nonEmpty, n); overflowed = java.util.Arrays.copyOf(overflowed, n); groups = n
    }
    /** Folds the buffer rows of `ctx` in: `groupOf(i)` is the row's group, or -1 to skip it. */
    def merge(ctx: EvalContext, groupOf: Int => Int): Unit = {
      val column = ctx.column(sumOrdinal)
      require(column != null, "the wide decimal sum buffer needs the batch's column")
      val empty = isEmpty.eval(ctx)
      val n = ctx.numRows
      var i = 0
      while (i < n) {
        val g = groupOf(i)
        if (g >= 0 && !(empty.validity() != null && !Bitmap.isSet(empty.validity(), i)) && !Bitmap.isSet(empty.data(), i)) {
          nonEmpty(g) = true
          if (column.isNullAt(i)) overflowed(g) = true
          else if (!overflowed(g)) total(g) = total(g).add(column.getDecimal(i, bufferType.precision, bufferType.scale).toJavaBigDecimal.unscaledValue())
        }
        i += 1
      }
    }
    def value(g: Int, slot: Int): Any =
      if (slot == 1) java.lang.Boolean.valueOf(!nonEmpty(g))
      else if (finalResult) {
        if (!nonEmpty(g)) null
        else if (overflowed(g)) {
          // A partial that overflowed left a null sum: Spark's CheckOverflowInSum raises on it in ANSI.
          if (nullOnOverflow) null else throw org.apache.spark.sql.vector.VectorErrors.overflowInSumOfDecimal(queryContext)
        } else if (total(g).abs.compareTo(limit) >= 0) {
          if (nullOnOverflow) null
          else throw org.apache.spark.sql.vector.VectorErrors.decimalPrecisionOverflow(
            org.apache.spark.sql.types.Decimal(new java.math.BigDecimal(total(g), bufferType.scale)), bufferType.precision, bufferType.scale, queryContext)
        } else new java.math.BigDecimal(total(g), bufferType.scale)
      } else {
        // The merged buffer, as Spark's Sum holds it: zero while empty, null once overflowed.
        if (!nonEmpty(g)) java.math.BigDecimal.valueOf(0L, bufferType.scale)
        else if (overflowed(g) || total(g).abs.compareTo(limit) >= 0) null
        else new java.math.BigDecimal(total(g), bufferType.scale)
      }
  }

  override def newState(): AggState = new AggState {
    private val state = new State(1)
    override def update(ctx: EvalContext): Unit =
      state.merge(ctx, i => if (ctx.selection == null || Bitmap.isSet(ctx.selection, i)) 0 else -1)
    override def bufferValues: Array[Any] = Array(state.value(0, 0), state.value(0, 1))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val state = new State(64)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      state.ensure(groups.numGroups())
      val ids = groups.ids()
      state.merge(ctx, i => ids(i))
    }
    override def bufferValue(g: Int, slot: Int): Any = state.value(g, slot)
  }
}

/** COUNT(*) when `input` is None, otherwise COUNT of non-null values of the expression. */
final case class CountAgg(input: Option[VectorExpr]) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType)
  override def newState(): AggState = new AggState {
    private var count = 0L
    override def update(ctx: EvalContext): Unit = input match {
      case None => count += ctx.selectedCount
      case Some(e) => count += AggKernels.countValid(ctx.masked(e.eval(ctx)))
    }
    override def bufferValues: Array[Any] = Array(java.lang.Long.valueOf(count))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.Count
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = input match {
      case None => acc.updateAll(groups)
      case Some(e) => acc.updateNonNull(e.eval(ctx), groups)
    }
    override def bufferValue(g: Int, slot: Int): Any = java.lang.Long.valueOf(acc.count(g))
  }
}

/** MIN / MAX over Int (incl. Date), Long (incl. Timestamp) or Double, with Spark's NaN ordering. */
final case class MinMaxAgg(input: VectorExpr, isMin: Boolean, dataType: DataType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType)
  override def newState(): AggState = new AggState {
    private var any = false
    private var bestLong = 0L
    private var bestDouble = 0.0
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      if (AggKernels.countValid(v) > 0) {
        v.`type`() match {
          case VecType.FLOAT64 =>
            val m = if (isMin) AggKernels.minDouble(v) else AggKernels.maxDouble(v)
            if (!any) bestDouble = m
            else {
              val cmp = CompareOp.nanSafeCompare(m, bestDouble)
              if ((isMin && cmp < 0) || (!isMin && cmp > 0)) bestDouble = m
            }
          case VecType.INT32 =>
            val m: Long = if (isMin) AggKernels.minInt(v) else AggKernels.maxInt(v)
            if (!any) bestLong = m else bestLong = if (isMin) math.min(bestLong, m) else math.max(bestLong, m)
          case VecType.INT64 =>
            val m = if (isMin) AggKernels.minLong(v) else AggKernels.maxLong(v)
            if (!any) bestLong = m else bestLong = if (isMin) math.min(bestLong, m) else math.max(bestLong, m)
          case other => throw new IllegalStateException(s"min/max on $other")
        }
        any = true
      }
    }
    override def bufferValues: Array[Any] =
      if (!any) Array(null)
      else input.vecType match {
        case VecType.FLOAT64 => Array(java.lang.Double.valueOf(bestDouble))
        case VecType.INT32 => Array(java.lang.Integer.valueOf(bestLong.toInt))
        case _ => Array(java.lang.Long.valueOf(bestLong))
      }
  }
  override def newGroupedState(): GroupedAggState = input.vecType match {
    case VecType.FLOAT64 =>
      new GroupedAggState {
        private val acc = new GroupedAccumulators.DoubleMinMax(isMin)
        override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
        override def bufferValue(g: Int, slot: Int): Any = if (acc.hasValue(g)) java.lang.Double.valueOf(acc.value(g)) else null
      }
    case VecType.INT32 =>
      new GroupedAggState {
        private val acc = new GroupedAccumulators.LongMinMax(isMin)
        override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
        override def bufferValue(g: Int, slot: Int): Any = if (acc.hasValue(g)) java.lang.Integer.valueOf(acc.value(g).toInt) else null
      }
    case _ =>
      new GroupedAggState {
        private val acc = new GroupedAccumulators.LongMinMax(isMin)
        override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
        override def bufferValue(g: Int, slot: Int): Any = if (acc.hasValue(g)) java.lang.Long.valueOf(acc.value(g)) else null
      }
  }
}

/** AVG over a double-typed input (integers are cast first): buffer is (sum, count). */
final case class AverageAgg(input: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType, LongType)
  override def newState(): AggState = new AggState {
    private var sum = 0.0
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      val c = AggKernels.countValid(v)
      if (c > 0) { sum += AggKernels.sumDouble(v); count += c }
    }
    override def bufferValues: Array[Any] = Array(java.lang.Double.valueOf(sum), java.lang.Long.valueOf(count))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.DoubleSum
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
    override def bufferValue(g: Int, slot: Int): Any =
      if (slot == 0) java.lang.Double.valueOf(acc.sum(g)) else java.lang.Long.valueOf(acc.count(g))
  }
}

/** COUNT in Final mode: the partial counts are summed; never null, 0 for empty input. */
final case class CountMergeAgg(count: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType)
  override def newState(): AggState = new AggState {
    private var total = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(count.eval(ctx))
      if (AggKernels.countValid(v) > 0) total += AggKernels.sumLong(v)
    }
    override def bufferValues: Array[Any] = Array(java.lang.Long.valueOf(total))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.LongSum
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(count.eval(ctx), groups)
    override def bufferValue(g: Int, slot: Int): Any = java.lang.Long.valueOf(acc.sum(g))
  }
}

/**
 * AVG in Final mode: partial (sum, count) buffers are summed component-wise. Empty input yields
 * (0.0, 0), Spark's initial buffer, so `sum / count` evaluates to null.
 */
final case class AverageMergeAgg(sum: VectorExpr, count: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType, LongType)
  override def newState(): AggState = new AggState {
    private var s = 0.0
    private var c = 0L
    override def update(ctx: EvalContext): Unit = {
      val sv = ctx.masked(sum.eval(ctx))
      if (AggKernels.countValid(sv) > 0) s += AggKernels.sumDouble(sv)
      val cv = ctx.masked(count.eval(ctx))
      if (AggKernels.countValid(cv) > 0) c += AggKernels.sumLong(cv)
    }
    override def bufferValues: Array[Any] = Array(java.lang.Double.valueOf(s), java.lang.Long.valueOf(c))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val sums = new GroupedAccumulators.DoubleSum
    private val counts = new GroupedAccumulators.LongSum
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      sums.update(sum.eval(ctx), groups)
      counts.update(count.eval(ctx), groups)
    }
    override def bufferValue(g: Int, slot: Int): Any =
      if (slot == 0) java.lang.Double.valueOf(sums.sum(g)) else java.lang.Long.valueOf(counts.sum(g))
  }
}

object VectorAggregates {

  private val numeric: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.FLOAT64)

  /**
   * Compiles an aggregate expression, or explains why it cannot be vectorized. The update modes
   * (`Partial`, `Complete`) read the function's input; the merge modes (`PartialMerge`, `Final`)
   * merge the partial buffers (`inputAggBufferAttributes`) found in `input`. What the operator then
   * emits -- buffers or results -- is the planner's decision, not the function's.
   */
  def compile(agg: AggregateExpression, input: Seq[Attribute]): Either[String, VectorAggFunction] =
    agg.mode match {
      // `isDistinct` is only a marker in a physical plan: Spark's distinct rewrites have already
      // grouped by the distinct column below, so the function runs over deduplicated input as is.
      case Partial | Complete =>
        compileFunction(agg.aggregateFunction, input).flatMap { f =>
          agg.filter match {
            case None => Right(f)
            case Some(p) => FilteredAgg.predicate(p, input).map(FilteredAgg(f, _))
          }
        }
      // The FILTER clause is applied while updating; merging buffers does not see it (Spark drops it).
      case PartialMerge | Final => compileMerge(agg.aggregateFunction, input, finalResult = agg.mode == Final)
    }

  /** Whether `mode` advances the state by merging buffers rather than by reading the function's input. */
  def merges(mode: AggregateMode): Boolean = mode == PartialMerge || mode == Final

  private def compileMerge(f: AggregateFunction, input: Seq[Attribute], finalResult: Boolean): Either[String, VectorAggFunction] = {
    val buffers = f.inputAggBufferAttributes
    def ref(i: Int): Either[String, VectorExpr] = ExpressionCompiler.compile(buffers(i), input)
    f match {
      case s: Sum if s.dataType.isInstanceOf[DecimalType] && buffers.length == 2 =>
        // The wide decimal sum's (sum, isEmpty) buffer: the sum column has no lane and is read from
        // the batch by ordinal; isEmpty is an ordinary boolean.
        val sumOrdinal = input.indexWhere(_.exprId == buffers(0).exprId)
        if (sumOrdinal < 0) Left(s"sum buffer ${buffers(0).name} not found in the input")
        else ref(1).map(empty => WideDecimalSumMergeAgg(sumOrdinal, empty, s.dataType.asInstanceOf[DecimalType], finalResult,
          nullOnOverflow = s.evalContext.evalMode != EvalMode.ANSI, s.origin.context))
      case s: Sum if buffers.length != 1 => Left(s"sum with a ${buffers.length}-column buffer not supported")
      case s: Sum =>
        ref(0).flatMap { b =>
          (s.dataType, b.vecType) match {
            case (DoubleType, VecType.FLOAT64) => Right(SumDoubleAgg(b))
            case (LongType, VecType.INT64) => Right(SumLongAgg(b, s.evalContext.evalMode == EvalMode.ANSI, s.origin.context))
            case (dt, _) => Left(s"merging sum buffers of ${dt.simpleString} not supported")
          }
        }
      case _: Count => ref(0).map(CountMergeAgg.apply)
      case m: Min if orderedLane(m.dataType) => ref(0).map(b => OrderedMinMaxAgg(b, isMin = true, m.dataType))
      case m: Max if orderedLane(m.dataType) => ref(0).map(b => OrderedMinMaxAgg(b, isMin = false, m.dataType))
      case m: Min => ref(0).flatMap(numericBuffer(m.dataType)).map(b => MinMaxAgg(b, isMin = true, m.dataType))
      case m: Max => ref(0).flatMap(numericBuffer(m.dataType)).map(b => MinMaxAgg(b, isMin = false, m.dataType))
      case b: BitAggregate if buffers.length == 1 && integralLane(b.dataType) =>
        ref(0).map(v => BitAgg(v, bitOp(b), b.dataType))
      case l: Last if buffers.length == 2 && FirstAgg.supports(l.dataType) =>
        for (last <- ref(0); valueSet <- ref(1)) yield LastAgg(last, l.dataType, l.ignoreNulls, Some(valueSet))
      case l: Last => Left(s"last over ${l.dataType.simpleString} not supported")
      case m: MaxMinBy if buffers.length == 2 && FirstAgg.supports(m.valueExpr.dataType) && Rows.supportsOrdering(TypeMapping.vecTypeOf(m.orderingExpr.dataType)) =>
        for (value <- ref(0); ordering <- ref(1)) yield MaxMinByAgg(value, ordering, isMax = m.isInstanceOf[MaxBy], m.valueExpr.dataType, m.orderingExpr.dataType)
      case m: MaxMinBy => Left(s"${m.prettyName} over ${m.valueExpr.dataType.simpleString} by ${m.orderingExpr.dataType.simpleString} not supported")
      case a: Average =>
        if (a.dataType != DoubleType || buffers.length != 2) Left(s"avg producing ${a.dataType.simpleString} not supported")
        else for (sum <- ref(0); count <- ref(1)) yield AverageMergeAgg(sum, count)
      case f: First if buffers.length == 2 && FirstAgg.supports(f.dataType) =>
        for (first <- ref(0); valueSet <- ref(1)) yield FirstMergeAgg(first, valueSet, f.dataType)
      case f: First => Left(s"first over ${f.dataType.simpleString} not supported")
      case other => Left(s"unsupported aggregate function ${other.getClass.getSimpleName}: ${other.sql}")
    }
  }

  /** min/max lanes handled by a comparison per row rather than the numeric kernels. */
  private def orderedLane(dt: DataType): Boolean = dt == BooleanType || dt == StringType
  private def integralLane(dt: DataType): Boolean = dt == org.apache.spark.sql.types.IntegerType || dt == LongType
  private def bitOp(b: BitAggregate): BitAgg.Op = b match {
    case _: BitAndAgg => BitAgg.And
    case _: BitOrAgg => BitAgg.Or
    case _ => BitAgg.Xor
  }
  private def orderedChild(e: Expression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(e, input).flatMap {
      case _: LiteralExpr => Left("min/max of a literal")
      case c => Right(c)
    }

  private def numericBuffer(dt: DataType)(b: VectorExpr): Either[String, VectorExpr] =
    if (numeric.contains(b.vecType)) Right(b) else Left(s"min/max over ${dt.simpleString} not supported")

  private def compileFunction(f: AggregateFunction, input: Seq[Attribute]): Either[String, VectorAggFunction] = f match {
    case s: Sum if s.dataType.isInstanceOf[DecimalType] =>
      // Only reached for decimals of more than 8 digits (the optimizer rewrites smaller ones to a
      // long sum): the buffer is Spark's (sum: Decimal(p + 10, s), isEmpty) pair, the sum wider than
      // 18 digits, accumulated in 128 bits from the unscaled lane values.
      numericChild(s.child, input).flatMap { child =>
        if (child.vecType == VecType.FLOAT64) Left(s"sum over ${s.child.dataType.simpleString} producing ${s.dataType.simpleString} not supported")
        else Right(WideDecimalSumAgg(child, s.dataType.asInstanceOf[DecimalType]))
      }
    case s: Sum =>
      numericChild(s.child, input).flatMap { child =>
        (s.dataType, child.vecType) match {
          case (DoubleType, VecType.FLOAT64) => Right(SumDoubleAgg(child))
          case (LongType, VecType.INT32) => Right(SumLongAgg(child, checked = false, s.origin.context))
          case (LongType, VecType.INT64) => Right(SumLongAgg(child, s.evalContext.evalMode == EvalMode.ANSI, s.origin.context))
          case (dt, _) => Left(s"sum over ${s.child.dataType.simpleString} producing ${dt.simpleString} not supported")
        }
      }

    case c: Count =>
      c.children match {
        case Seq(Literal(v, _)) if v != null => Right(CountAgg(None))
        case Seq(child) =>
          ExpressionCompiler.compile(child, input).flatMap {
            case _: LiteralExpr => Right(CountAgg(None))
            case e => Right(CountAgg(Some(e)))
          }
        case _ => Left("count with several arguments not supported")
      }

    case m: Min if orderedLane(m.dataType) => orderedChild(m.child, input).map(child => OrderedMinMaxAgg(child, isMin = true, m.dataType))
    case m: Max if orderedLane(m.dataType) => orderedChild(m.child, input).map(child => OrderedMinMaxAgg(child, isMin = false, m.dataType))
    case m: Min => numericChild(m.child, input).map(child => MinMaxAgg(child, isMin = true, m.dataType))
    case m: Max => numericChild(m.child, input).map(child => MinMaxAgg(child, isMin = false, m.dataType))
    case b: BitAggregate if integralLane(b.dataType) =>
      ExpressionCompiler.compile(b.child, input).flatMap {
        case _: LiteralExpr => Left(s"${b.prettyName} of a literal")
        case e => Right(BitAgg(e, bitOp(b), b.dataType))
      }
    case b: BitAggregate => Left(s"${b.prettyName} over ${b.dataType.simpleString} not supported")
    case l: Last if FirstAgg.supports(l.dataType) =>
      ExpressionCompiler.compile(l.child, input).flatMap {
        case _: LiteralExpr => Left("last of a literal")
        case e => Right(LastAgg(e, l.dataType, l.ignoreNulls, None))
      }
    case l: Last => Left(s"last over ${l.dataType.simpleString} not supported")
    case m: MaxMinBy if FirstAgg.supports(m.valueExpr.dataType) && Rows.supportsOrdering(TypeMapping.vecTypeOf(m.orderingExpr.dataType)) =>
      for {
        value <- ExpressionCompiler.compile(m.valueExpr, input)
        ordering <- ExpressionCompiler.compile(m.orderingExpr, input)
        _ <- if (ordering.isInstanceOf[LiteralExpr]) Left(s"${m.prettyName} by a literal") else Right(())
      } yield MaxMinByAgg(value, ordering, isMax = m.isInstanceOf[MaxBy], m.valueExpr.dataType, m.orderingExpr.dataType)
    case m: MaxMinBy => Left(s"${m.prettyName} over ${m.valueExpr.dataType.simpleString} by ${m.orderingExpr.dataType.simpleString} not supported")

    case a: Average if a.child.dataType.isInstanceOf[DecimalType] =>
      Left(s"avg buffer ${a.aggBufferAttributes.head.dataType.simpleString} exceeds ${TypeMapping.MAX_DECIMAL_PRECISION} digits")
    case a: Average =>
      if (a.dataType != DoubleType) Left(s"avg producing ${a.dataType.simpleString} not supported")
      else numericChild(a.child, input).map { child =>
        AverageAgg(if (child.vecType == VecType.FLOAT64) child else CastExpr(child, DoubleType))
      }

    case f: First if FirstAgg.supports(f.dataType) =>
      ExpressionCompiler.compile(f.child, input).flatMap {
        case _: LiteralExpr => Left("first of a literal")
        case e => Right(FirstAgg(e, f.dataType, f.ignoreNulls))
      }
    case f: First => Left(s"first over ${f.dataType.simpleString} not supported")

    case other => Left(s"unsupported aggregate function ${other.getClass.getSimpleName}: ${other.sql}")
  }

  private def numericChild(child: org.apache.spark.sql.catalyst.expressions.Expression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(child, input).flatMap {
      case _: LiteralExpr => Left("aggregate over a literal")
      case e if !numeric.contains(e.vecType) => Left(s"aggregate over ${child.dataType.simpleString} not supported")
      case e => Right(e)
    }

  /** Bridges a kernel-level buffers object to a Spark-side accessor, for tests. */
  def countValid(v: VectorBuffers): Long = AggKernels.countValid(v)
}
