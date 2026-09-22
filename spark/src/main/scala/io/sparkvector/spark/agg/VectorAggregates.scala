package io.sparkvector.spark.agg

import io.sparkvector.kernels.{AggKernels, Bitmap, CompareOp, Decimal128, GroupAssignment, GroupedAccumulators, VecType, VectorBuffers}
import io.sparkvector.spark.expr.{CastExpr, EvalContext, ExpressionCompiler, LiteralExpr, SpeculativeDecimalExpr, SpeculativeDecimals, VectorExpr}
import org.apache.spark.sql.catalyst.expressions.{Attribute, EvalMode, Expression, Literal}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.ArrowVectorBuffers
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
  /**
   * Writes slot `slot` of groups `[from, to)` straight into `out` -- a lane of the slot's emitted
   * type, validity bits included -- and answers true; false leaves the column to the boxed path
   * (#416: the rollup partial aggregate of q18 at 1 TB emitted 5.5 M rows x 7 wide averages through a
   * BigDecimal, a BigInteger and a Long per value, most of that stage's excess over Spark).
   */
  def writeBuffer(slot: Int, from: Int, to: Int, out: ArrowVectorBuffers): Boolean = false
  /** Buffer slot value of group `g` in Spark's internal representation, or `null`. */
  def bufferValue(g: Int, slot: Int): Any
}

/** Serializable description of a supported aggregate function; states are created per task. */
trait VectorAggFunction extends Serializable {
  def bufferTypes: Seq[DataType]
  def newState(): AggState
  def newGroupedState(): GroupedAggState
  /**
   * The types of the columns this function's slots are emitted as, given Spark's declared buffer
   * types. A function whose result mode emits the result ready-made in a buffer slot (the wide
   * decimal average: `Decimal(p + 4, s + 4)` in the `Decimal(p + 10, s)` sum slot) says so here.
   */
  def emittedTypes(declared: Seq[DataType]): Seq[DataType] = declared
}

/**
 * SUM over doubles: buffer `sum` is null until the first non-null input. `strict` selects Spark's
 * rounding (rows added in order into one accumulator) over the faster lane-parallel and interleaved
 * partial sums; see `spark.vector.exec.strictFloatingPoint`.
 */
final case class SumDoubleAgg(input: VectorExpr, strict: Boolean) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType)
  override def newState(): AggState = new AggState {
    private var sum = 0.0
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      val c = AggKernels.countValid(v)
      if (c > 0) { sum = AggKernels.sumDoubleFrom(v, sum, strict); count += c }
    }
    override def bufferValues: Array[Any] = Array(if (count == 0) null else java.lang.Double.valueOf(sum))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.DoubleSum(strict)
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
/**
 * The exact side of a speculative narrow input (#26): the rows whose product left 64 bits are added
 * here as exact integers, per group, beside the 128-bit lane accumulator. Nothing to do for an
 * ordinary lane input. Shared by the wide decimal sum and average.
 */
private[agg] final class Escalation {
  private var totals = new Array[java.math.BigInteger](0)
  private var counts = new Array[Long](0)
  private def ensure(g: Int): Unit = if (g >= totals.length) {
    val n = math.max(g + 1, totals.length * 2)
    totals = java.util.Arrays.copyOf(totals, n); counts = java.util.Arrays.copyOf(counts, n)
  }
  def add(g: Int, v: java.math.BigInteger): Unit = {
    ensure(g)
    totals(g) = if (totals(g) == null) v else totals(g).add(v)
    counts(g) += 1
  }
  def count(g: Int): Long = if (g < counts.length) counts(g) else 0L
  def total(g: Int, base: java.math.BigInteger): java.math.BigInteger = if (g < totals.length && totals(g) != null) base.add(totals(g)) else base
  /** Whether any row of group `g` was escalated (its total then needs the `BigInteger` path). */
  def hasTotal(g: Int): Boolean = g < totals.length && totals(g) != null
}

/**
 * Writes of a wide decimal buffer straight into its DECIMAL128 lane (#416): the group's 128-bit total
 * as two limbs when it is within the buffer precision, null past it -- the limit compared as limbs, so
 * no `BigInteger` is made for the common group -- and the count and empty flags into their lanes.
 */
private[agg] final class WideLimbs(limit: java.math.BigInteger) extends Serializable {
  private val limitHi = Decimal128.hiOf(limit)
  private val limitLo = Decimal128.loOf(limit)
  private val negLimit = limit.negate()
  private val negHi = Decimal128.hiOf(negLimit)
  private val negLo = Decimal128.loOf(negLimit)

  /** `(hi, lo)` at `o`, or null when `|total| >= limit`. */
  def checked(out: ArrowVectorBuffers, o: Int, hi: Long, lo: Long): Unit =
    if (Decimal128.compare(hi, lo, limitHi, limitLo) >= 0 || Decimal128.compare(hi, lo, negHi, negLo) <= 0) nul(out, o)
    else { Bitmap.set(out.validity(), o); Decimal128.set(out.data(), o, hi, lo) }

  /** An exact total that needed a `BigInteger` (escalated rows, or past 128 bits), the same check. */
  def big(out: ArrowVectorBuffers, o: Int, t: java.math.BigInteger): Unit =
    if (t.abs.compareTo(limit) >= 0) nul(out, o)
    else { Bitmap.set(out.validity(), o); Decimal128.set(out.data(), o, Decimal128.hiOf(t), Decimal128.loOf(t)) }

  def zero(out: ArrowVectorBuffers, o: Int): Unit = { Bitmap.set(out.validity(), o); Decimal128.set(out.data(), o, 0L, 0L) }
  def nul(out: ArrowVectorBuffers, o: Int): Unit = { Bitmap.clear(out.validity(), o); Decimal128.set(out.data(), o, 0L, 0L) }
}

private[agg] object WideLimbs {
  def long(out: ArrowVectorBuffers, o: Int, v: Long): Unit = { Bitmap.set(out.validity(), o); out.data().setAtIndex(VectorBuffers.LE_LONG, o.toLong, v) }
  def bool(out: ArrowVectorBuffers, o: Int, v: Boolean): Unit = { Bitmap.set(out.validity(), o); Bitmap.setTo(out.data(), o, v) }
}

private[agg] object Escalation {
  /**
   * The group's exact total: the 128-bit accumulator plus the escalated rows. An accumulator that
   * left 128 bits (a DECIMAL128 input, #259) is reported as 10^38, past every decimal precision, so the
   * callers' limit check turns it into the null or the ANSI error Spark produces for an overflowed sum.
   */
  private val pastAnyPrecision = java.math.BigInteger.TEN.pow(38)
  def total(acc: GroupedAccumulators.WideLongSum, extra: Escalation, g: Int): java.math.BigInteger =
    if (acc.overflowed(g)) pastAnyPrecision else extra.total(g, acc.sum(g))

  /** Evaluates `input` for one batch: the lane to accumulate, and the escalated rows folded into `extra`. */
  def evalInput(input: VectorExpr, ctx: EvalContext, extra: Escalation, groupOf: Int => Int): VectorBuffers = input match {
    case s: SpeculativeDecimalExpr =>
      val checked = s.evalChecked(ctx)
      var k = 0
      var added = 0
      while (k < checked.rows.length) {
        val g = groupOf(checked.rows(k))
        if (g >= 0) { extra.add(g, checked.exact(k)); added += 1 }
        k += 1
      }
      if (added > 0) SpeculativeDecimals.addEscalated(added)
      checked.lane
    case other => other.eval(ctx)
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
 *
 * In `Complete` mode (the streaming planners' single stage; Spark 4.1's batch planner never emits
 * it) there is no merge to apply Spark's `If(isEmpty, null, CheckOverflowInSum(sum))`, so the sum
 * slot carries the result itself: null for an empty group, null (non-ANSI) or Spark's precision
 * error (ANSI) past the declared precision.
 */
final case class WideDecimalSumAgg(
    input: VectorExpr,
    bufferType: DecimalType,
    finalResult: Boolean,
    nullOnOverflow: Boolean,
    queryContext: org.apache.spark.QueryContext) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(bufferType, BooleanType)
  private val limit = java.math.BigInteger.TEN.pow(bufferType.precision)

  /** The buffer's `sum` for a group's 128-bit total, or `null` past the buffer precision. */
  private def sumValue(count: Long, total: => java.math.BigInteger): Any =
    if (finalResult) {
      if (count == 0) null
      else {
        val t = total
        if (t.abs.compareTo(limit) >= 0) {
          if (nullOnOverflow) null
          else throw org.apache.spark.sql.vector.VectorErrors.decimalPrecisionOverflow(
            org.apache.spark.sql.types.Decimal(new java.math.BigDecimal(t, bufferType.scale)), bufferType.precision, bufferType.scale, queryContext)
        } else new java.math.BigDecimal(t, bufferType.scale)
      }
    } else if (count == 0) java.math.BigDecimal.valueOf(0L, bufferType.scale)
    else {
      val t = total
      if (t.abs.compareTo(limit) >= 0) null else new java.math.BigDecimal(t, bufferType.scale)
    }

  override def newState(): AggState = new AggState {
    private val acc = new GroupedAccumulators.WideLongSum
    private val extra = new Escalation
    override def update(ctx: EvalContext): Unit = {
      val sel = ctx.selection
      val lane = Escalation.evalInput(input, ctx, extra, r => if (sel == null || Bitmap.isSet(sel, r)) 0 else -1)
      acc.updateAll(ctx.masked(lane))
    }
    override def bufferValues: Array[Any] = {
      val count = acc.count(0) + extra.count(0)
      Array(sumValue(count, Escalation.total(acc, extra, 0)), java.lang.Boolean.valueOf(count == 0))
    }
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.WideLongSum
    private val extra = new Escalation
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      val ids = groups.ids()
      val lane = Escalation.evalInput(input, ctx, extra, r => ids(r))
      acc.update(lane, groups)
    }
    override def bufferValue(g: Int, slot: Int): Any = {
      val count = acc.count(g) + extra.count(g)
      if (slot == 0) sumValue(count, Escalation.total(acc, extra, g)) else java.lang.Boolean.valueOf(count == 0)
    }
    /** The buffer as lanes (#416): the sum as limbs -- zero while empty, null past the precision -- and the empty flag. */
    override def writeBuffer(slot: Int, from: Int, to: Int, out: ArrowVectorBuffers): Boolean = {
      if (finalResult) return false
      var g = from
      while (g < to) {
        val o = g - from
        val count = acc.count(g) + extra.count(g)
        if (slot == 1) WideLimbs.bool(out, o, count == 0)
        else if (count == 0) limbs.zero(out, o)
        else if (acc.overflowed(g) || extra.hasTotal(g)) limbs.big(out, o, Escalation.total(acc, extra, g))
        else limbs.checked(out, o, acc.hi(g), acc.lo(g))
        g += 1
      }
      true
    }
  }
  private val limbs = new WideLimbs(limit)
}

/**
 * The result of a decimal average from its merged `(sum, count)` -- Spark's own
 * `If(count = 0, null, DecimalDivideWithOverflowCheck(sum, count, Decimal(p + 4, s + 4)))`, bound to
 * the two buffer slots and evaluated per group, so the division's 39-digit half-up quotient, its
 * rounding to the result scale, the null or `NUMERIC_VALUE_OUT_OF_RANGE` past the result precision
 * and the `ARITHMETIC_OVERFLOW` on a sum the buffer could not hold are Spark's to the last digit
 * and the last error message. One evaluation per group, as Spark's own Final does.
 */
final case class DecimalAvgResult(expression: Expression, resultType: DecimalType) extends Serializable {
  private val wide = resultType.precision > TypeMapping.MAX_DECIMAL_PRECISION
  @transient private lazy val row = new org.apache.spark.sql.catalyst.expressions.GenericInternalRow(2)

  /** `sum` is the buffer's exact total, or `null` when the buffer overflowed; boxed as the result column holds it. */
  def value(sum: java.math.BigDecimal, count: Long): Any = {
    row.update(0, if (sum == null) null else org.apache.spark.sql.types.Decimal(sum))
    row.update(1, count)
    expression.eval(row) match {
      case null => null
      case d: org.apache.spark.sql.types.Decimal => if (wide) d.toJavaBigDecimal else java.lang.Long.valueOf(d.toUnscaledLong)
    }
  }
}

object DecimalAvgResult {
  def apply(a: Average): DecimalAvgResult =
    DecimalAvgResult(
      org.apache.spark.sql.catalyst.expressions.BindReferences.bindReference(a.evaluateExpression, a.aggBufferAttributes),
      a.dataType.asInstanceOf[DecimalType])
}

/**
 * AVG over a decimal of more than 11 digits -- the range `DecimalAggregates` does not rewrite to a
 * double average -- whose buffer is Spark's `(sum: Decimal(p + 10, s), count: bigint)`, the sum wider
 * than 18 digits. The update modes: the INT64 lane's unscaled values go into a 128-bit accumulator
 * per group beside the count of non-null rows, and the buffer is Spark's own two columns. Spark's
 * update adds without an overflow check and lets the buffer writer null a sum past the buffer
 * precision, a null that poisons the group; the same total here is null past that precision. A
 * declared-wide product under the average (#26) is speculative like the sum's.
 *
 * In `Complete` mode the sum slot carries the result -- [[DecimalAvgResult]] over the group's own
 * sum and count -- as the merge emits it in `Final` mode.
 */
final case class WideDecimalAvgAgg(input: VectorExpr, bufferType: DecimalType, result: DecimalAvgResult, finalResult: Boolean) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(bufferType, LongType)
  override def emittedTypes(declared: Seq[DataType]): Seq[DataType] =
    if (finalResult) Seq(result.resultType, declared(1)) else declared
  private val limit = java.math.BigInteger.TEN.pow(bufferType.precision)

  /**
   * The buffer's `sum`: zero while empty, null past the buffer precision -- except an ungrouped
   * result, where Spark's generated code keeps the sum in a local variable nothing re-checks.
   */
  private def sumValue(count: Long, total: => java.math.BigInteger, ungrouped: Boolean): Any = {
    val buffer: java.math.BigDecimal =
      if (count == 0) java.math.BigDecimal.valueOf(0L, bufferType.scale)
      else {
        val t = total
        if (!(finalResult && ungrouped) && t.abs.compareTo(limit) >= 0) null else new java.math.BigDecimal(t, bufferType.scale)
      }
    if (finalResult) result.value(buffer, count) else buffer
  }

  override def newState(): AggState = new AggState {
    private val acc = new GroupedAccumulators.WideLongSum
    private val extra = new Escalation
    override def update(ctx: EvalContext): Unit = {
      val sel = ctx.selection
      val lane = Escalation.evalInput(input, ctx, extra, r => if (sel == null || Bitmap.isSet(sel, r)) 0 else -1)
      acc.updateAll(ctx.masked(lane))
    }
    override def bufferValues: Array[Any] = {
      val count = acc.count(0) + extra.count(0)
      Array(sumValue(count, Escalation.total(acc, extra, 0), ungrouped = true), java.lang.Long.valueOf(count))
    }
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.WideLongSum
    private val extra = new Escalation
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      val ids = groups.ids()
      val lane = Escalation.evalInput(input, ctx, extra, r => ids(r))
      acc.update(lane, groups)
    }
    override def bufferValue(g: Int, slot: Int): Any = {
      val count = acc.count(g) + extra.count(g)
      if (slot == 0) sumValue(count, Escalation.total(acc, extra, g), ungrouped = false) else java.lang.Long.valueOf(count)
    }
    /** The buffer as lanes (#416): the sum as limbs -- zero while empty, null past the precision -- and the count. */
    override def writeBuffer(slot: Int, from: Int, to: Int, out: ArrowVectorBuffers): Boolean = {
      if (finalResult) return false
      var g = from
      while (g < to) {
        val o = g - from
        val count = acc.count(g) + extra.count(g)
        if (slot == 1) WideLimbs.long(out, o, count)
        else if (count == 0) limbs.zero(out, o)
        else if (acc.overflowed(g) || extra.hasTotal(g)) limbs.big(out, o, Escalation.total(acc, extra, g))
        else limbs.checked(out, o, acc.hi(g), acc.lo(g))
        g += 1
      }
      true
    }
  }
  private val limbs = new WideLimbs(limit)
}

/**
 * The merge modes of the wide decimal average: Spark's `(sum, count)` buffer rows combined per group
 * with Spark's own rules -- sums added exactly (a null `sum` on a row meaning "overflowed earlier"
 * and poisoning the group, as `DecimalAddNoOverflowCheck` propagates it), counts added. The wide
 * sum column has no lane and is read row by row from the batch like the wide sum's; the count is
 * an ordinary bigint lane. `PartialMerge` emits the merged buffer; `Final` emits the result in the
 * sum slot ([[DecimalAvgResult]]) and the projection forwards it.
 */
final case class WideDecimalAvgMergeAgg(
    sumOrdinal: Int,
    count: VectorExpr,
    bufferType: DecimalType,
    result: DecimalAvgResult,
    finalResult: Boolean) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(bufferType, LongType)
  override def emittedTypes(declared: Seq[DataType]): Seq[DataType] =
    if (finalResult) Seq(result.resultType, declared(1)) else declared
  private val limit = java.math.BigInteger.TEN.pow(bufferType.precision)

  /**
   * Per-group merge state: the total as two 64-bit limbs, as the sum's merge keeps it (#388). A
   * `BigInteger` per row -- an allocation and an add for each of 5.5 M rows x 7 columns in q18's final
   * aggregate at 1 TB -- was most of a reduce task's time (#416). A signed 128-bit overflow is past the
   * 38-digit limit, so it marks the group overflowed; the exact ungrouped final, which reports such a
   * total rather than null, carries it on in a `BigInteger` from that point. `BigInteger` otherwise
   * appears once per group, at the end.
   */
  private final class State(var groups: Int, ungrouped: Boolean) {
    var hi = new Array[Long](groups)
    var lo = new Array[Long](groups)
    /** Exact totals of the groups whose 128 bits overflowed (the exact ungrouped final only); null otherwise. */
    var wide: Array[java.math.BigInteger] = _
    var counts = new Array[Long](groups)
    var overflowed = new Array[Boolean](groups)
    def ensure(needed: Int): Unit = if (needed > groups) {
      val n = math.max(needed, groups * 2) // geometric: a copy per batch as groups trickle in was 8% of an executor (#388)
      hi = java.util.Arrays.copyOf(hi, n); lo = java.util.Arrays.copyOf(lo, n)
      if (wide != null) wide = java.util.Arrays.copyOf(wide, n)
      counts = java.util.Arrays.copyOf(counts, n); overflowed = java.util.Arrays.copyOf(overflowed, n); groups = n
    }
    def merge(ctx: EvalContext, groupOf: Int => Int): Unit = {
      // The partial's sum buffer is a DECIMAL128 lane (ours in place, Spark's copied once): two limbs per row.
      val sums = ctx.input(sumOrdinal)
      val data = sums.data()
      val counted = count.eval(ctx)
      val n = ctx.numRows
      var i = 0
      while (i < n) {
        val g = groupOf(i)
        if (g >= 0) {
          // Spark's count buffer is never null (initial 0); a null here is a defensive skip.
          if (counted.validity() == null || Bitmap.isSet(counted.validity(), i)) counts(g) += counted.data().getAtIndex(VectorBuffers.LE_LONG, i)
          if (sums.isNull(i)) overflowed(g) = true
          else if (!overflowed(g)) add(g, Decimal128.hi(data, i), Decimal128.lo(data, i))
        }
        i += 1
      }
    }
    private def add(g: Int, rhi: Long, rlo: Long): Unit = {
      if (wide != null && wide(g) != null) { wide(g) = wide(g).add(Decimal128.toBigInteger(rhi, rlo)); return }
      val l = lo(g) + rlo
      val carry = if (java.lang.Long.compareUnsigned(l, lo(g)) < 0) 1L else 0L
      val h = hi(g) + rhi + carry
      // Signed overflow of the 128-bit total: both operands of one sign, the result of the other.
      if (((hi(g) ^ h) & (rhi ^ h)) < 0) {
        if (finalResult && ungrouped) {
          if (wide == null) wide = new Array[java.math.BigInteger](groups)
          wide(g) = Decimal128.toBigInteger(hi(g), lo(g)).add(Decimal128.toBigInteger(rhi, rlo))
        } else overflowed(g) = true
      } else { hi(g) = h; lo(g) = l }
    }
    private def total(g: Int): java.math.BigInteger =
      if (wide != null && wide(g) != null) wide(g) else Decimal128.toBigInteger(hi(g), lo(g))
    /**
     * The merged sum as Spark's buffer holds it: null once a partial arrived null. A total past the
     * buffer precision is null in a grouped result -- Spark's hash-map buffer rows re-check the
     * precision on every merge -- but exact in an ungrouped one, where Spark's generated Final keeps
     * the sum in a local variable that nothing re-checks and the division sees the full total.
     */
    private def sum(g: Int): java.math.BigDecimal =
      if (overflowed(g) || (!(finalResult && ungrouped) && total(g).abs.compareTo(limit) >= 0)) null
      else new java.math.BigDecimal(total(g), bufferType.scale)
    def value(g: Int, slot: Int): Any =
      if (slot == 1) java.lang.Long.valueOf(counts(g))
      else if (finalResult) result.value(sum(g), counts(g))
      else sum(g)
  }

  override def newState(): AggState = new AggState {
    private val state = new State(1, ungrouped = true)
    override def update(ctx: EvalContext): Unit =
      state.merge(ctx, i => if (ctx.selection == null || Bitmap.isSet(ctx.selection, i)) 0 else -1)
    override def bufferValues: Array[Any] = Array(state.value(0, 0), state.value(0, 1))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val state = new State(64, ungrouped = false)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      state.ensure(groups.numGroups())
      val ids = groups.ids()
      state.merge(ctx, i => ids(i))
    }
    override def bufferValue(g: Int, slot: Int): Any = state.value(g, slot)
    /** The merged buffer as lanes (#416): the sum as limbs -- null once overflowed or past the precision -- and the count. */
    override def writeBuffer(slot: Int, from: Int, to: Int, out: ArrowVectorBuffers): Boolean = {
      if (finalResult) return false
      var g = from
      while (g < to) {
        val o = g - from
        if (slot == 1) WideLimbs.long(out, o, state.counts(g))
        else if (state.overflowed(g)) limbs.nul(out, o)
        else if (state.wide != null && state.wide(g) != null) limbs.big(out, o, state.wide(g))
        else limbs.checked(out, o, state.hi(g), state.lo(g))
        g += 1
      }
      true
    }
  }
  private val limbs = new WideLimbs(limit)
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

  /**
   * Per-group merge state: the total as two 64-bit limbs (#388 -- a `BigInteger` per row was 4% of an
   * executor's time in q67 at 1 TB), a signed 128-bit overflow marking the group overflowed (any such
   * total is past the 38-digit limit anyway). `BigInteger` appears once per group, at the end.
   */
  private final class State(var groups: Int) {
    var hi = new Array[Long](groups)
    var lo = new Array[Long](groups)
    var nonEmpty = new Array[Boolean](groups)
    var overflowed = new Array[Boolean](groups)
    def ensure(needed: Int): Unit = if (needed > groups) {
      val n = math.max(needed, groups * 2) // geometric: a copy per batch as groups trickle in was 8% of an executor (#388)
      hi = java.util.Arrays.copyOf(hi, n); lo = java.util.Arrays.copyOf(lo, n)
      nonEmpty = java.util.Arrays.copyOf(nonEmpty, n); overflowed = java.util.Arrays.copyOf(overflowed, n); groups = n
    }
    /** Folds the buffer rows of `ctx` in: `groupOf(i)` is the row's group, or -1 to skip it. */
    def merge(ctx: EvalContext, groupOf: Int => Int): Unit = {
      // The partial's sum buffer is a DECIMAL128 lane (ours in place, Spark's copied once): two limbs per row.
      val sums = ctx.input(sumOrdinal)
      val data = sums.data()
      val empty = isEmpty.eval(ctx)
      val emptyValidity = empty.validity()
      val emptyBits = empty.data()
      val n = ctx.numRows
      var i = 0
      while (i < n) {
        val g = groupOf(i)
        if (g >= 0 && !(emptyValidity != null && !Bitmap.isSet(emptyValidity, i)) && !Bitmap.isSet(emptyBits, i)) {
          nonEmpty(g) = true
          if (sums.isNull(i)) overflowed(g) = true
          else if (!overflowed(g)) {
            val rhi = Decimal128.hi(data, i)
            val rlo = Decimal128.lo(data, i)
            val l = lo(g) + rlo
            val carry = if (java.lang.Long.compareUnsigned(l, lo(g)) < 0) 1L else 0L
            val h = hi(g) + rhi + carry
            // Signed overflow of the 128-bit total: both operands of one sign, the result of the other.
            if (((hi(g) ^ h) & (rhi ^ h)) < 0) overflowed(g) = true
            else { hi(g) = h; lo(g) = l }
          }
        }
        i += 1
      }
    }
    private def total(g: Int): java.math.BigInteger = Decimal128.toBigInteger(hi(g), lo(g))
    def value(g: Int, slot: Int): Any =
      if (slot == 1) java.lang.Boolean.valueOf(!nonEmpty(g))
      else if (finalResult) {
        if (!nonEmpty(g)) null
        else if (overflowed(g)) {
          // A partial that overflowed left a null sum: Spark's CheckOverflowInSum raises on it in ANSI.
          if (nullOnOverflow) null else throw org.apache.spark.sql.vector.VectorErrors.overflowInSumOfDecimal(queryContext)
        } else {
          val t = total(g)
          if (t.abs.compareTo(limit) >= 0) {
            if (nullOnOverflow) null
            else throw org.apache.spark.sql.vector.VectorErrors.decimalPrecisionOverflow(
              org.apache.spark.sql.types.Decimal(new java.math.BigDecimal(t, bufferType.scale)), bufferType.precision, bufferType.scale, queryContext)
          } else new java.math.BigDecimal(t, bufferType.scale)
        }
      } else {
        // The merged buffer, as Spark's Sum holds it: zero while empty, null once overflowed.
        if (!nonEmpty(g)) java.math.BigDecimal.valueOf(0L, bufferType.scale)
        else if (overflowed(g)) null
        else {
          val t = total(g)
          if (t.abs.compareTo(limit) >= 0) null else new java.math.BigDecimal(t, bufferType.scale)
        }
      }
    /** The merged buffer as lanes (#416): the sum as limbs -- zero while empty, null once overflowed or past the precision -- and the empty flag. */
    def write(slot: Int, from: Int, to: Int, out: ArrowVectorBuffers): Unit = {
      var g = from
      while (g < to) {
        val o = g - from
        if (slot == 1) WideLimbs.bool(out, o, !nonEmpty(g))
        else if (!nonEmpty(g)) limbs.zero(out, o)
        else if (overflowed(g)) limbs.nul(out, o)
        else limbs.checked(out, o, hi(g), lo(g))
        g += 1
      }
    }
  }
  private val limbs = new WideLimbs(limit)

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
    override def writeBuffer(slot: Int, from: Int, to: Int, out: ArrowVectorBuffers): Boolean = {
      if (finalResult) return false
      state.write(slot, from, to, out)
      true
    }
  }
}

/**
 * `try_sum` over an integral input -- Spark's `Sum` in TRY mode: the buffer is `(sum: bigint, isEmpty)`
 * like the decimal sum's, the rows are added exactly, and the first overflow poisons the group -- its
 * `sum` is null from then on and stays null through every later row and merge, exactly as Spark's
 * `Add(sum, child, TRY)` leaves a null that no later add revives. Spark's Final result,
 * `If(isEmpty, null, sum)`, compiles over the emitted buffer as an ordinary expression.
 */
final case class TrySumLongAgg(input: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType, BooleanType)

  private final class State(var groups: Int) {
    var sum = new Array[Long](groups)
    var nonEmpty = new Array[Boolean](groups)
    var poisoned = new Array[Boolean](groups)
    def ensure(needed: Int): Unit = if (needed > groups) {
      val n = math.max(needed, groups * 2) // geometric: a copy per batch as groups trickle in was 8% of an executor (#388)
      sum = java.util.Arrays.copyOf(sum, n); nonEmpty = java.util.Arrays.copyOf(nonEmpty, n); poisoned = java.util.Arrays.copyOf(poisoned, n); groups = n
    }
    def add(v: VectorBuffers, validity: java.lang.foreign.MemorySegment, n: Int, groupOf: Int => Int): Unit = {
      val wide = v.`type`() == VecType.INT64
      var i = 0
      while (i < n) {
        val g = groupOf(i)
        if (g >= 0 && (validity == null || Bitmap.isSet(validity, i))) {
          nonEmpty(g) = true
          if (!poisoned(g)) {
            val x = if (wide) v.data().getAtIndex(VectorBuffers.LE_LONG, i) else v.data().getAtIndex(VectorBuffers.LE_INT, i).toLong
            val r = sum(g) + x
            // Overflow iff both operands share a sign the result does not (Math.addExact's test).
            if (((sum(g) ^ r) & (x ^ r)) < 0) poisoned(g) = true else sum(g) = r
          }
        }
        i += 1
      }
    }
    def value(g: Int, slot: Int): Any =
      if (slot == 1) java.lang.Boolean.valueOf(!nonEmpty(g))
      else if (poisoned(g)) null
      else java.lang.Long.valueOf(sum(g))
  }

  override def newState(): AggState = new AggState {
    private val state = new State(1)
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      state.add(v, v.validity(), ctx.numRows, _ => 0)
    }
    override def bufferValues: Array[Any] = Array(state.value(0, 0), state.value(0, 1))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val state = new State(64)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      state.ensure(groups.numGroups())
      val v = input.eval(ctx)
      val ids = groups.ids()
      state.add(v, groups.effectiveValidity(v), groups.numRows(), i => ids(i))
    }
    override def bufferValue(g: Int, slot: Int): Any = state.value(g, slot)
  }
}

/** The merge modes of `try_sum`: Spark's rules -- a null `sum` on a non-empty row poisons the group, sums add exactly, `isEmpty = isEmpty && other.isEmpty`. */
final case class TrySumLongMergeAgg(sum: VectorExpr, isEmpty: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType, BooleanType)

  private final class State(var groups: Int) {
    var total = new Array[Long](groups)
    var nonEmpty = new Array[Boolean](groups)
    var poisoned = new Array[Boolean](groups)
    def ensure(needed: Int): Unit = if (needed > groups) {
      val n = math.max(needed, groups * 2) // geometric: a copy per batch as groups trickle in was 8% of an executor (#388)
      total = java.util.Arrays.copyOf(total, n); nonEmpty = java.util.Arrays.copyOf(nonEmpty, n); poisoned = java.util.Arrays.copyOf(poisoned, n); groups = n
    }
    def merge(ctx: EvalContext, groupOf: Int => Int): Unit = {
      val sv = sum.eval(ctx)
      val empty = isEmpty.eval(ctx)
      val n = ctx.numRows
      var i = 0
      while (i < n) {
        val g = groupOf(i)
        if (g >= 0 && !(empty.validity() != null && !Bitmap.isSet(empty.validity(), i)) && !Bitmap.isSet(empty.data(), i)) {
          nonEmpty(g) = true
          if (sv.validity() != null && !Bitmap.isSet(sv.validity(), i)) poisoned(g) = true
          else if (!poisoned(g)) {
            val x = sv.data().getAtIndex(VectorBuffers.LE_LONG, i)
            val r = total(g) + x
            if (((total(g) ^ r) & (x ^ r)) < 0) poisoned(g) = true else total(g) = r
          }
        }
        i += 1
      }
    }
    def value(g: Int, slot: Int): Any =
      if (slot == 1) java.lang.Boolean.valueOf(!nonEmpty(g))
      else if (poisoned(g)) null
      else java.lang.Long.valueOf(total(g))
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
    private var bestHi = 0L
    private var bestLo = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      if (AggKernels.countValid(v) > 0) {
        v.`type`() match {
          case VecType.DECIMAL128 =>
            // The wide lane is scalar (#28): a two-limb compare per valid row.
            val data = v.data()
            var i = 0
            val n = v.length()
            while (i < n) {
              if (!v.isNull(i)) {
                val hi = Decimal128.hi(data, i)
                val lo = Decimal128.lo(data, i)
                if (!any) { bestHi = hi; bestLo = lo; any = true }
                else {
                  val c = Decimal128.compare(hi, lo, bestHi, bestLo)
                  if ((isMin && c < 0) || (!isMin && c > 0)) { bestHi = hi; bestLo = lo }
                }
              }
              i += 1
            }
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
        case VecType.DECIMAL128 => Array(new java.math.BigDecimal(Decimal128.toBigInteger(bestHi, bestLo), scale))
        case _ => Array(java.lang.Long.valueOf(bestLong))
      }
  }
  /** The wide decimal's scale, for the boxed value the DECIMAL128 buffer column is built from. */
  private def scale: Int = dataType.asInstanceOf[DecimalType].scale
  override def newGroupedState(): GroupedAggState = input.vecType match {
    case VecType.DECIMAL128 =>
      new GroupedAggState {
        private val acc = new GroupedAccumulators.Decimal128MinMax(isMin)
        override def update(ctx: EvalContext, groups: GroupAssignment): Unit = acc.update(input.eval(ctx), groups)
        override def bufferValue(g: Int, slot: Int): Any = if (acc.hasValue(g)) new java.math.BigDecimal(acc.value(g), scale) else null
      }
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
final case class AverageAgg(input: VectorExpr, strict: Boolean) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType, LongType)
  override def newState(): AggState = new AggState {
    private var sum = 0.0
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = ctx.masked(input.eval(ctx))
      val c = AggKernels.countValid(v)
      if (c > 0) { sum = AggKernels.sumDoubleFrom(v, sum, strict); count += c }
    }
    override def bufferValues: Array[Any] = Array(java.lang.Double.valueOf(sum), java.lang.Long.valueOf(count))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val acc = new GroupedAccumulators.DoubleSum(strict)
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
final case class AverageMergeAgg(sum: VectorExpr, count: VectorExpr, strict: Boolean) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType, LongType)
  override def newState(): AggState = new AggState {
    private var s = 0.0
    private var c = 0L
    override def update(ctx: EvalContext): Unit = {
      val sv = ctx.masked(sum.eval(ctx))
      if (AggKernels.countValid(sv) > 0) s = AggKernels.sumDoubleFrom(sv, s, strict)
      val cv = ctx.masked(count.eval(ctx))
      if (AggKernels.countValid(cv) > 0) c += AggKernels.sumLong(cv)
    }
    override def bufferValues: Array[Any] = Array(java.lang.Double.valueOf(s), java.lang.Long.valueOf(c))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val sums = new GroupedAccumulators.DoubleSum(strict)
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
   * The position in a merging stage's input of each aggregate's first buffer column: the grouping
   * columns first, then every function's buffers in order -- Spark binds buffers by position, and the
   * exprIds only happen to agree when the Partial's and the Final's function instances are the same
   * object (a subquery rewrite between the two stages gives the Final a fresh instance).
   */
  def bufferOffsets(a: org.apache.spark.sql.execution.aggregate.BaseAggregateExec): Seq[Int] =
    bufferOffsets(a.initialInputBufferOffset, a.aggregateExpressions)

  def bufferOffsets(initialOffset: Int, aggregateExpressions: Seq[AggregateExpression]): Seq[Int] =
    aggregateExpressions.scanLeft(initialOffset)((off, agg) => off + agg.aggregateFunction.inputAggBufferAttributes.length).init

  /**
   * Compiles an aggregate expression, or explains why it cannot be vectorized. The update modes
   * (`Partial`, `Complete`) read the function's input; the merge modes (`PartialMerge`, `Final`)
   * merge the partial buffers (`inputAggBufferAttributes`) found in `input`. What the operator then
   * emits -- buffers or results -- is the planner's decision, not the function's. `strict` is
   * `spark.vector.exec.strictFloatingPoint` (default on): double sums round exactly like Spark's.
   */
  def compile(agg: AggregateExpression, input: Seq[Attribute], bufferOffset: Int = -1, strict: Boolean = true): Either[String, VectorAggFunction] =
    agg.mode match {
      // `isDistinct` is only a marker in a physical plan: Spark's distinct rewrites have already
      // grouped by the distinct column below, so the function runs over deduplicated input as is.
      case Partial | Complete =>
        compileFunction(agg.aggregateFunction, input, complete = agg.mode == Complete, strict).flatMap { f =>
          agg.filter match {
            case None => Right(f)
            case Some(p) => FilteredAgg.predicate(p, input).map(FilteredAgg(f, _))
          }
        }
      // The FILTER clause is applied while updating; merging buffers does not see it (Spark drops it).
      case PartialMerge | Final => compileMerge(agg.aggregateFunction, input, finalResult = agg.mode == Final, bufferOffset, strict)
    }

  /** Whether `mode` advances the state by merging buffers rather than by reading the function's input. */
  def merges(mode: AggregateMode): Boolean = mode == PartialMerge || mode == Final

  private def compileMerge(f: AggregateFunction, input: Seq[Attribute], finalResult: Boolean, bufferOffset: Int, strict: Boolean): Either[String, VectorAggFunction] = {
    val buffers = f.inputAggBufferAttributes
    /** The input column holding buffer `i`: by exprId when the ids agree, by Spark's position otherwise. */
    def bufferOrdinal(i: Int): Int = {
      val byId = input.indexWhere(_.exprId == buffers(i).exprId)
      if (byId >= 0) byId
      else if (bufferOffset >= 0 && bufferOffset + i < input.length && input(bufferOffset + i).dataType == buffers(i).dataType) bufferOffset + i
      else -1
    }
    def ref(i: Int): Either[String, VectorExpr] = {
      val ordinal = bufferOrdinal(i)
      if (ordinal < 0) Left(s"unbound attribute ${buffers(i).name}") else ExpressionCompiler.compileLaneColumn(input(ordinal), input)
    }
    f match {
      case s: Sum if s.evalContext.evalMode == EvalMode.TRY && s.dataType == LongType && buffers.length == 2 =>
        for (sum <- ref(0); empty <- ref(1)) yield TrySumLongMergeAgg(sum, empty)
      case s: Sum if s.evalContext.evalMode == EvalMode.TRY && s.dataType.isInstanceOf[DecimalType] => Left("try_sum over a decimal not supported")
      case s: Sum if s.dataType.isInstanceOf[DecimalType] && buffers.length == 2 =>
        // The wide decimal sum's (sum, isEmpty) buffer: the sum column has no lane and is read from
        // the batch by ordinal; isEmpty is an ordinary boolean.
        val sumOrdinal = bufferOrdinal(0)
        if (sumOrdinal < 0) Left(s"sum buffer ${buffers(0).name} not found in the input")
        else ref(1).map(empty => WideDecimalSumMergeAgg(sumOrdinal, empty, s.dataType.asInstanceOf[DecimalType], finalResult,
          nullOnOverflow = s.evalContext.evalMode != EvalMode.ANSI, s.origin.context))
      case s: Sum if buffers.length != 1 => Left(s"sum with a ${buffers.length}-column buffer not supported")
      case s: Sum =>
        ref(0).flatMap { b =>
          (s.dataType, b.vecType) match {
            case (DoubleType, VecType.FLOAT64) => Right(SumDoubleAgg(b, strict))
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
      case l: Last if buffers.length == 2 && FirstAgg.supportsWide(l.dataType) =>
        for (last <- ref(0); valueSet <- ref(1)) yield LastAgg(last, l.dataType, l.ignoreNulls, Some(valueSet))
      case l: Last => Left(s"last over ${l.dataType.simpleString} not supported")
      case m: MaxMinBy if buffers.length == 2 && FirstAgg.supports(m.valueExpr.dataType) && Rows.supportsOrdering(TypeMapping.vecTypeOf(m.orderingExpr.dataType)) =>
        for (value <- ref(0); ordering <- ref(1)) yield MaxMinByAgg(value, ordering, isMax = m.isInstanceOf[MaxBy], m.valueExpr.dataType, m.orderingExpr.dataType)
      case m: MaxMinBy => Left(s"${m.prettyName} over ${m.valueExpr.dataType.simpleString} by ${m.orderingExpr.dataType.simpleString} not supported")
      case a: Average if a.child.dataType.isInstanceOf[DecimalType] && buffers.length == 2 =>
        // The wide decimal average's (sum, count) buffer: the sum column has no lane and is read
        // from the batch by ordinal; the count is an ordinary bigint.
        val bufferType = buffers(0).dataType.asInstanceOf[DecimalType]
        val sumOrdinal = bufferOrdinal(0)
        if (bufferType.precision <= TypeMapping.MAX_DECIMAL_PRECISION) Left(s"avg buffer ${bufferType.simpleString} within 18 digits not supported")
        else if (sumOrdinal < 0) Left(s"avg buffer ${buffers(0).name} not found in the input")
        else ref(1).flatMap { count =>
          if (count.vecType != VecType.INT64) Left(s"avg count buffer ${buffers(1).dataType.simpleString} is not a bigint")
          else Right(WideDecimalAvgMergeAgg(sumOrdinal, count, bufferType, DecimalAvgResult(a), finalResult))
        }
      case a: Average =>
        if (a.dataType != DoubleType || buffers.length != 2) Left(s"avg producing ${a.dataType.simpleString} not supported")
        else for (sum <- ref(0); count <- ref(1)) yield AverageMergeAgg(sum, count, strict)
      case f: First if buffers.length == 2 && FirstAgg.supportsWide(f.dataType) =>
        for (first <- ref(0); valueSet <- ref(1)) yield FirstMergeAgg(first, valueSet, f.dataType)
      case f: First => Left(s"first over ${f.dataType.simpleString} not supported")
      case m: CentralMomentAgg if buffers.length == m.aggBufferAttributes.length =>
        momentBuffers(buffers.indices.map(ref)).map(MomentsAgg(_, MomentsAgg.Central(buffers.length - 1), merge = true))
      case _: Covariance if buffers.length == 4 =>
        momentBuffers(buffers.indices.map(ref)).map(MomentsAgg(_, MomentsAgg.Covariance, merge = true))
      case _: PearsonCorrelation if buffers.length == 6 =>
        momentBuffers(buffers.indices.map(ref)).map(MomentsAgg(_, MomentsAgg.Correlation, merge = true))
      case _: RegrSlope | _: RegrIntercept if buffers.length == 7 =>
        momentBuffers(buffers.indices.map(ref)).map(MomentsAgg(_, MomentsAgg.Regression, merge = true))
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
    if (numeric.contains(b.vecType) || b.vecType == VecType.DECIMAL128) Right(b) else Left(s"min/max over ${dt.simpleString} not supported")

  /** The arguments of a moment statistic as double lanes. */
  private def momentArgs(args: Seq[Expression], input: Seq[Attribute]): Either[String, Seq[VectorExpr]] =
    args.foldRight[Either[String, List[VectorExpr]]](Right(Nil)) { (a, acc) =>
      for (rest <- acc; child <- numericChild(a, input)) yield (if (child.vecType == VecType.FLOAT64) child else CastExpr(child, DoubleType)) :: rest
    }

  /** The buffer columns of a moment statistic, every one a double. */
  private def momentBuffers(refs: Seq[Either[String, VectorExpr]]): Either[String, Seq[VectorExpr]] =
    refs.foldRight[Either[String, List[VectorExpr]]](Right(Nil)) { (r, acc) =>
      for (rest <- acc; b <- r) yield {
        if (b.vecType != VecType.FLOAT64) return Left(s"moment buffer ${b.dataType.simpleString} is not a double")
        b :: rest
      }
    }

  private def compileFunction(f: AggregateFunction, input: Seq[Attribute], complete: Boolean, strict: Boolean): Either[String, VectorAggFunction] = f match {
    // try_sum over an integral input: Spark's (sum, isEmpty) buffer with the whole group nulled on overflow.
    case s: Sum if s.evalContext.evalMode == EvalMode.TRY && s.dataType == LongType =>
      numericChild(s.child, input).flatMap { child =>
        if (child.vecType == VecType.INT32 || child.vecType == VecType.INT64) Right(TrySumLongAgg(child))
        else Left(s"try_sum over ${s.child.dataType.simpleString} not supported")
      }
    case s: Sum if s.evalContext.evalMode == EvalMode.TRY && s.dataType.isInstanceOf[DecimalType] => Left("try_sum over a decimal not supported")
    case s: Sum if s.dataType.isInstanceOf[DecimalType] =>
      // Only reached for decimals of more than 8 digits (the optimizer rewrites smaller ones to a
      // long sum): the buffer is Spark's (sum: Decimal(p + 10, s), isEmpty) pair, the sum wider than
      // 18 digits, accumulated in 128 bits from the unscaled lane values. A declared-wide product of
      // narrow operands under the sum (TPC-H's `sum(l_extendedprice * (1 - l_discount))`) is computed
      // speculatively in 64 bits, its overflowing rows added exactly (#26).
      val bufferType = s.dataType.asInstanceOf[DecimalType]
      def agg(child: VectorExpr) = WideDecimalSumAgg(child, bufferType, complete, nullOnOverflow = s.evalContext.evalMode != EvalMode.ANSI, s.origin.context)
      ExpressionCompiler.speculativeDecimalArithmetic(s.child, input) match {
        case Some(speculative) => speculative.map(agg)
        case None =>
          numericChild(s.child, input, wide = true).flatMap { child =>
            if (child.vecType == VecType.FLOAT64) Left(s"sum over ${s.child.dataType.simpleString} producing ${s.dataType.simpleString} not supported")
            else Right(agg(child))
          }
      }
    case s: Sum =>
      numericChild(s.child, input).flatMap { child =>
        (s.dataType, child.vecType) match {
          case (DoubleType, VecType.FLOAT64) => Right(SumDoubleAgg(child, strict))
          case (LongType, VecType.INT32) => Right(SumLongAgg(child, checked = false, s.origin.context))
          case (LongType, VecType.INT64) => Right(SumLongAgg(child, s.evalContext.evalMode == EvalMode.ANSI, s.origin.context))
          case (dt, _) => Left(s"sum over ${s.child.dataType.simpleString} producing ${dt.simpleString} not supported")
        }
      }

    case c: Count =>
      c.children match {
        case Seq(Literal(v, _)) if v != null => Right(CountAgg(None))
        case Seq(child) =>
          ExpressionCompiler.compileLaneColumn(child, input).flatMap {
            case _: LiteralExpr => Right(CountAgg(None))
            case e => Right(CountAgg(Some(e)))
          }
        case several =>
          // count(a, b): the rows where every argument is non-null (regr_count's replacement).
          several.foldRight[Either[String, List[VectorExpr]]](Right(Nil)) { (a, acc) =>
            for (rest <- acc; e <- ExpressionCompiler.compile(a, input)) yield e :: rest
          }.map { inputs =>
            // A non-null literal argument (count(DISTINCT 3, 2) after the distinct rewrite) is never
            // null, so only the lanes decide; all literals is count(*).
            val lanes = inputs.filterNot(_.isInstanceOf[LiteralExpr])
            if (lanes.isEmpty) CountAgg(None) else CountAllAgg(lanes)
          }
      }

    case m: Min if orderedLane(m.dataType) => orderedChild(m.child, input).map(child => OrderedMinMaxAgg(child, isMin = true, m.dataType))
    case m: Max if orderedLane(m.dataType) => orderedChild(m.child, input).map(child => OrderedMinMaxAgg(child, isMin = false, m.dataType))
    case m: Min => numericChild(m.child, input, wide = true).map(child => MinMaxAgg(child, isMin = true, m.dataType))
    case m: Max => numericChild(m.child, input, wide = true).map(child => MinMaxAgg(child, isMin = false, m.dataType))
    case b: BitAggregate if integralLane(b.dataType) =>
      ExpressionCompiler.compile(b.child, input).flatMap {
        case _: LiteralExpr => Left(s"${b.prettyName} of a literal")
        case e => Right(BitAgg(e, bitOp(b), b.dataType))
      }
    case b: BitAggregate => Left(s"${b.prettyName} over ${b.dataType.simpleString} not supported")
    case l: Last if FirstAgg.supportsWide(l.dataType) =>
      ExpressionCompiler.compileLaneColumn(l.child, input).flatMap {
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
      // Only reached for decimals of more than 11 digits (the optimizer rewrites smaller ones to a
      // double average of the unscaled value): the buffer is Spark's (sum: Decimal(p + 10, s), count)
      // pair, the sum wider than 18 digits, accumulated in 128 bits; the result is Spark's own
      // expression over the merged buffer (#26).
      val bufferType = a.aggBufferAttributes.head.dataType.asInstanceOf[DecimalType]
      if (bufferType.precision <= TypeMapping.MAX_DECIMAL_PRECISION) Left(s"avg buffer ${bufferType.simpleString} within 18 digits not supported")
      else {
        def agg(child: VectorExpr) = WideDecimalAvgAgg(child, bufferType, DecimalAvgResult(a), complete)
        ExpressionCompiler.speculativeDecimalArithmetic(a.child, input) match {
          case Some(speculative) => speculative.map(agg)
          case None =>
            numericChild(a.child, input, wide = true).flatMap { child =>
              if (child.vecType == VecType.FLOAT64) Left(s"avg over ${a.child.dataType.simpleString} producing ${a.dataType.simpleString} not supported")
              else Right(agg(child))
            }
        }
      }
    case a: Average =>
      if (a.dataType != DoubleType) Left(s"avg producing ${a.dataType.simpleString} not supported")
      else numericChild(a.child, input).map { child =>
        AverageAgg(if (child.vecType == VecType.FLOAT64) child else CastExpr(child, DoubleType), strict)
      }

    // The one-pass moment statistics: Spark's Welford step per row over doubles (the analyzer has
    // cast the arguments), the buffers emitted as Spark's doubles, the result Spark's own expression.
    case m: CentralMomentAgg =>
      momentArgs(Seq(m.child), input).map(MomentsAgg(_, MomentsAgg.Central(m.aggBufferAttributes.length - 1), merge = false))
    case c: Covariance =>
      momentArgs(Seq(c.left, c.right), input).map(MomentsAgg(_, MomentsAgg.Covariance, merge = false))
    case c: PearsonCorrelation =>
      momentArgs(Seq(c.left, c.right), input).map(MomentsAgg(_, MomentsAgg.Correlation, merge = false))
    // regr_slope / regr_intercept: a covariance and a variance of the independent variable (the right argument) over the same pairs.
    case r: RegrSlope => momentArgs(Seq(r.right, r.left), input).map(MomentsAgg(_, MomentsAgg.Regression, merge = false))
    case r: RegrIntercept => momentArgs(Seq(r.right, r.left), input).map(MomentsAgg(_, MomentsAgg.Regression, merge = false))

    case f: First if FirstAgg.supportsWide(f.dataType) =>
      ExpressionCompiler.compileLaneColumn(f.child, input).flatMap {
        case _: LiteralExpr => Left("first of a literal")
        case e => Right(FirstAgg(e, f.dataType, f.ignoreNulls))
      }
    case f: First => Left(s"first over ${f.dataType.simpleString} not supported")

    case other => Left(s"unsupported aggregate function ${other.getClass.getSimpleName}: ${other.sql}")
  }

  /** A numeric lane operand; `wide` also admits a DECIMAL128 lane (the decimal sum, average, min and max, #259). */
  private def numericChild(child: org.apache.spark.sql.catalyst.expressions.Expression, input: Seq[Attribute], wide: Boolean = false): Either[String, VectorExpr] =
    ExpressionCompiler.compileLaneColumn(child, input).flatMap {
      case _: LiteralExpr => Left("aggregate over a literal")
      case e if !numeric.contains(e.vecType) && !(wide && e.vecType == VecType.DECIMAL128) => Left(s"aggregate over ${child.dataType.simpleString} not supported")
      case e => Right(e)
    }

  /** Bridges a kernel-level buffers object to a Spark-side accessor, for tests. */
  def countValid(v: VectorBuffers): Long = AggKernels.countValid(v)
}
