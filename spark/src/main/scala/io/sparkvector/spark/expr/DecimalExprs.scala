package io.sparkvector.spark.expr

import java.lang.foreign.MemorySegment

import io.sparkvector.kernels._
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.QueryContext
import org.apache.spark.sql.types._

/**
 * Decimals of at most 18 digits travel through the kernels as unscaled INT64 lanes; these nodes
 * carry the scale bookkeeping that the physical type does not.
 */
object DecimalExprs {

  /** Rows of `bits` (within `validity` and `ctx.active`) that would be visible to Spark. */
  private[expr] def affected(ctx: EvalContext, bits: MemorySegment, validity: MemorySegment): (MemorySegment, Int) = {
    val n = ctx.numRows
    if (bits == null) return (null, 0)
    val out = ctx.bitmap()
    if (validity == null) BitmapKernels.copy(bits, out, n) else BitmapKernels.and(bits, validity, out, n)
    if (ctx.active != null) BitmapKernels.and(out, ctx.active, out, n)
    (out, Bitmap.popcount(out, n))
  }

  /** `validity` minus `bits`. */
  private[expr] def without(ctx: EvalContext, validity: MemorySegment, bits: MemorySegment): MemorySegment = {
    val n = ctx.numRows
    val out = ctx.bitmap()
    if (validity == null) BitmapKernels.not(bits, out, n) else BitmapKernels.andNot(validity, bits, out, n)
    out
  }

  private[expr] def firstSet(bits: MemorySegment, n: Int): Int = {
    var w = 0
    val words = Bitmap.wordsFor(n)
    while (w < words) {
      val word = Bitmap.wordAt(bits, w, n)
      if (word != 0L) return (w << 6) + java.lang.Long.numberOfTrailingZeros(word)
      w += 1
    }
    -1
  }

  /** The unscaled long of a decimal literal, rescaled to `scale` (exact: the literal has at most that scale). */
  def unscaled(value: Any, from: DecimalType, scale: Int): Long = {
    val d = value.asInstanceOf[Decimal]
    d.toUnscaledLong * DecimalKernels.POW10(scale - from.scale)
  }

  /** `a` brought to `scale` (a copy when the scale changes, `a` itself otherwise). */
  private[expr] def rescaled(a: VectorBuffers, from: Int, to: Int, ctx: EvalContext): VectorBuffers = {
    if (from == to) a
    else {
      val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, ctx.numRows)
      DecimalKernels.mulPow10(a, to - from, data)
      SegmentVectorBuffers.fixedWidth(VecType.INT64, ctx.numRows, a.validity(), data)
    }
  }
}

/**
 * Decimal `+ - * /` with Spark's result type. Addition and subtraction rescale both operands to the
 * result scale; multiplication multiplies the unscaled values (the result scale is the sum of the
 * operand scales). Their result precision, as Spark computes it, always has room for the result, so
 * none of them can overflow. Division rounds half up like Spark and can exceed the result precision,
 * which yields null in legacy mode and raises in ANSI mode; a zero divisor behaves like `ArithExpr`.
 */
final case class DecimalArithExpr(
    op: ArithOp,
    left: VectorExpr,
    right: VectorExpr,
    leftType: DecimalType,
    rightType: DecimalType,
    dataType: DecimalType,
    ansi: Boolean,
    queryContext: QueryContext)
    extends VectorExpr {

  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    op match {
      case ArithOp.ADD | ArithOp.SUB =>
        val s = dataType.scale
        (left, right) match {
          case (l, lit: LiteralExpr) =>
            val a = DecimalExprs.rescaled(l.eval(ctx), leftType.scale, s, ctx)
            ArithKernels.arithScalar(op, a, java.lang.Long.valueOf(DecimalExprs.unscaled(lit.value, rightType, s)), data)
            SegmentVectorBuffers.fixedWidth(VecType.INT64, n, a.validity(), data)
          case (lit: LiteralExpr, r) =>
            val b = DecimalExprs.rescaled(r.eval(ctx), rightType.scale, s, ctx)
            ArithKernels.scalarArith(op, java.lang.Long.valueOf(DecimalExprs.unscaled(lit.value, leftType, s)), b, data)
            SegmentVectorBuffers.fixedWidth(VecType.INT64, n, b.validity(), data)
          case (l, r) =>
            val a = DecimalExprs.rescaled(l.eval(ctx), leftType.scale, s, ctx)
            val b = DecimalExprs.rescaled(r.eval(ctx), rightType.scale, s, ctx)
            ArithKernels.arith(op, a, b, data)
            SegmentVectorBuffers.fixedWidth(VecType.INT64, n, combined(ctx, a.validity(), b.validity()), data)
        }
      case ArithOp.MUL =>
        (left, right) match {
          case (l, lit: LiteralExpr) =>
            val a = l.eval(ctx)
            ArithKernels.arithScalar(op, a, java.lang.Long.valueOf(lit.value.asInstanceOf[Decimal].toUnscaledLong), data)
            SegmentVectorBuffers.fixedWidth(VecType.INT64, n, a.validity(), data)
          case (lit: LiteralExpr, r) =>
            val b = r.eval(ctx)
            ArithKernels.arithScalar(op, b, java.lang.Long.valueOf(lit.value.asInstanceOf[Decimal].toUnscaledLong), data)
            SegmentVectorBuffers.fixedWidth(VecType.INT64, n, b.validity(), data)
          case (l, r) =>
            val a = l.eval(ctx)
            val b = r.eval(ctx)
            ArithKernels.arith(op, a, b, data)
            SegmentVectorBuffers.fixedWidth(VecType.INT64, n, combined(ctx, a.validity(), b.validity()), data)
        }
      case ArithOp.DIV => divide(ctx, data)
    }
  }

  private def combined(ctx: EvalContext, a: MemorySegment, b: MemorySegment): MemorySegment =
    if (a == null && b == null) null
    else {
      val v = ctx.bitmap()
      BitmapKernels.combineValidity(a, b, v, ctx.numRows)
      v
    }

  private def divide(ctx: EvalContext, data: MemorySegment): VectorBuffers = {
    val n = ctx.numRows
    val overflow = ctx.bitmap()
    var validity: MemorySegment = null
    var divisorZero: MemorySegment = null
    val s1 = leftType.scale
    val s2 = rightType.scale
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        val d = lit.value.asInstanceOf[Decimal].toUnscaledLong
        DecimalKernels.divideScalar(a, d, s1, s2, dataType.scale, dataType.precision, data, overflow)
        validity = a.validity()
        if (d == 0L) { divisorZero = ctx.bitmap(); Bitmap.fill(divisorZero, n, true) }
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        DecimalKernels.scalarDivide(lit.value.asInstanceOf[Decimal].toUnscaledLong, b, s1, s2, dataType.scale, dataType.precision, data, overflow)
        validity = b.validity()
        divisorZero = zeroMask(b, ctx)
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        DecimalKernels.divide(a, b, s1, s2, dataType.scale, dataType.precision, data, overflow)
        validity = combined(ctx, a.validity(), b.validity())
        divisorZero = zeroMask(b, ctx)
    }
    val (zeroRows, zeroCount) = DecimalExprs.affected(ctx, divisorZero, validity)
    if (zeroCount > 0) {
      if (ansi) throw org.apache.spark.sql.vector.VectorErrors.divideByZero(queryContext)
      validity = DecimalExprs.without(ctx, validity, divisorZero)
    }
    val (overflowRows, overflowCount) = DecimalExprs.affected(ctx, overflow, validity)
    if (overflowCount > 0) {
      if (ansi) {
        // Recompute the offending quotient with Spark's own arithmetic for the error message.
        val i = DecimalExprs.firstSet(overflowRows, n)
        val a = left match { case lit: LiteralExpr => lit.value.asInstanceOf[Decimal]; case e => Decimal.createUnsafe(e.eval(ctx).getLong(i), leftType.precision, s1) }
        val b = right match { case lit: LiteralExpr => lit.value.asInstanceOf[Decimal]; case e => Decimal.createUnsafe(e.eval(ctx).getLong(i), rightType.precision, s2) }
        throw org.apache.spark.sql.vector.VectorErrors.decimalPrecisionOverflow(a / b, dataType.precision, dataType.scale, queryContext)
      }
      validity = DecimalExprs.without(ctx, validity, overflow)
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, validity, data)
  }

  private def zeroMask(b: VectorBuffers, ctx: EvalContext): MemorySegment = {
    val zero = ctx.bitmap()
    CompareKernels.compareScalar(b, java.lang.Long.valueOf(0L), CompareOp.EQ, zero)
    zero
  }
}

/**
 * Casts involving a decimal: decimal to decimal (rescale, half-up), integral to decimal, decimal to
 * long/int (truncation), decimal to double and double to decimal. A value that does not fit the
 * target is null in legacy mode and an error in ANSI mode, as in Spark's `Cast`.
 */
final case class DecimalCastExpr(child: VectorExpr, from: DataType, dataType: DataType, ansi: Boolean, queryContext: QueryContext)
    extends VectorExpr {

  override def children: Seq[VectorExpr] = Seq(child)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    (from, dataType) match {
      case (f: DecimalType, t: DecimalType) =>
        val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
        if (t.scale >= f.scale) {
          // Range check on the input so the multiplication cannot wrap.
          val invalid = ctx.bitmap()
          val bound = DecimalKernels.maxUnscaled(t.precision) / DecimalKernels.POW10(t.scale - f.scale)
          DecimalKernels.outOfRange(a, bound, invalid)
          DecimalKernels.mulPow10(a, t.scale - f.scale, data)
          finishDecimal(ctx, a, data, invalid, f, t)
        } else {
          DecimalKernels.divPow10HalfUp(a, f.scale - t.scale, data)
          val invalid = ctx.bitmap()
          val out = SegmentVectorBuffers.fixedWidth(VecType.INT64, n, a.validity(), data)
          DecimalKernels.outOfRange(out, DecimalKernels.maxUnscaled(t.precision), invalid)
          finishDecimal(ctx, a, data, invalid, f, t)
        }
      case (IntegerType | LongType | DateType, t: DecimalType) =>
        val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
        val invalid = ctx.bitmap()
        DecimalKernels.fromIntegral(a, t.precision, t.scale, data, invalid)
        finishDecimal(ctx, a, data, invalid, from, t)
      case (DoubleType, t: DecimalType) =>
        val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
        val invalid = ctx.bitmap()
        DecimalKernels.fromDouble(a, t.precision, t.scale, data, invalid)
        finishDecimal(ctx, a, data, invalid, from, t)
      case (f: DecimalType, DoubleType) =>
        val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, n)
        DecimalKernels.toDouble(a, f.scale, data)
        SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, n, a.validity(), data)
      case (f: DecimalType, LongType) =>
        val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
        DecimalKernels.toLong(a, f.scale, data)
        SegmentVectorBuffers.fixedWidth(VecType.INT64, n, a.validity(), data)
      case (f: DecimalType, IntegerType) =>
        val data = ArrowLayout.allocateData(ctx.arena, VecType.INT32, n)
        val invalid = ctx.bitmap()
        DecimalKernels.toInt(a, f.scale, data, invalid)
        val (rows, count) = DecimalExprs.affected(ctx, invalid, a.validity())
        if (count > 0 && ansi) {
          val i = DecimalExprs.firstSet(rows, n)
          throw org.apache.spark.sql.vector.VectorErrors.castOverflow(Decimal.createUnsafe(a.getLong(i), f.precision, f.scale), f, IntegerType)
        }
        // Legacy mode wraps, exactly what the kernel wrote.
        SegmentVectorBuffers.fixedWidth(VecType.INT32, n, a.validity(), data)
      case other => throw new IllegalStateException(s"unsupported decimal cast $other")
    }
  }

  /** Applies the invalid-row bitmap: ANSI raises for the first affected row, legacy nulls them. */
  private def finishDecimal(ctx: EvalContext, in: VectorBuffers, data: MemorySegment, invalid: MemorySegment, f: DataType, t: DecimalType): VectorBuffers = {
    val n = ctx.numRows
    val (rows, count) = DecimalExprs.affected(ctx, invalid, in.validity())
    var validity = in.validity()
    if (count > 0) {
      if (ansi) {
        val i = DecimalExprs.firstSet(rows, n)
        val value: Decimal = f match {
          case fd: DecimalType => Decimal.createUnsafe(in.getLong(i), fd.precision, fd.scale)
          case IntegerType | DateType => Decimal(in.getInt(i))
          case LongType => Decimal(in.getLong(i))
          case DoubleType =>
            val d = in.getDouble(i)
            if (d.isNaN || d.isInfinite) throw org.apache.spark.sql.vector.VectorErrors.castOverflow(d, DoubleType, t)
            Decimal(d)
          case _ => throw new IllegalStateException(s"unexpected source type $f")
        }
        throw org.apache.spark.sql.vector.VectorErrors.decimalPrecisionOverflow(value, t.precision, t.scale, queryContext)
      }
      validity = DecimalExprs.without(ctx, validity, invalid)
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, validity, data)
  }
}

/**
 * `UnscaledValue(decimal)`: the lanes already hold the unscaled long, so only the logical type
 * changes. Spark's optimizer produces it (with [[MakeDecimalExpr]]) when it rewrites a sum or an
 * average over a small decimal into one over longs.
 */
final case class UnscaledValueExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = LongType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = child.eval(ctx)
}

/**
 * `MakeDecimal(long, precision, scale)`: the lanes are reused as the decimal's unscaled value;
 * values beyond the precision are null (`nullOnOverflow`, legacy mode) or raise
 * NUMERIC_VALUE_OUT_OF_RANGE (ANSI).
 */
final case class MakeDecimalExpr(child: VectorExpr, dataType: DecimalType, nullOnOverflow: Boolean) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    val invalid = ctx.bitmap()
    if (DecimalKernels.outOfRange(a, DecimalKernels.maxUnscaled(dataType.precision), invalid) == 0) a
    else {
      val (rows, count) = DecimalExprs.affected(ctx, invalid, a.validity())
      if (count == 0) a
      else if (nullOnOverflow) SegmentVectorBuffers.fixedWidth(VecType.INT64, n, DecimalExprs.without(ctx, a.validity(), invalid), a.data())
      else {
        // Decimal.set raises Spark's own error for the value.
        Decimal(a.getLong(DecimalExprs.firstSet(rows, n)), dataType.precision, dataType.scale)
        throw new IllegalStateException("unreachable: Decimal.set accepted an out-of-range value")
      }
    }
  }
}


/** Rows an operator had to recompute exactly because the speculative narrow result overflowed 64 bits (#26). */
object SpeculativeDecimals {
  private val escalated = new java.util.concurrent.atomic.LongAdder
  def escalatedRows(): Long = escalated.sum()
  private[spark] def addEscalated(n: Int): Unit = escalated.add(n)
}

/**
 * A decimal expression whose *declared* result is wider than 18 digits, computed speculatively in
 * the INT64 unscaled lane with the rows that do not fit escalated exactly (#26). Consumed only by the
 * wide decimal sum and average through [[evalChecked]]; [[VectorExpr.eval]] is never called on one.
 */
trait SpeculativeDecimalExpr extends VectorExpr {
  def dataType: DecimalType
  def evalChecked(ctx: EvalContext): SpeculativeChecked
  override def eval(ctx: EvalContext): VectorBuffers =
    throw new IllegalStateException(s"speculative narrow decimal ${dataType.simpleString} is consumed only by the wide decimal aggregates")
}

/** One operand as a speculative loop reads it: a literal, a lane, or a speculative child with its escalated rows. */
private[expr] final class SpeculativeOperand(e: VectorExpr, ctx: EvalContext) {
  val literal: Boolean = e.isInstanceOf[LiteralExpr]
  val lit: Long = if (literal) e.asInstanceOf[LiteralExpr].value.asInstanceOf[Decimal].toUnscaledLong else 0L
  private val checked: SpeculativeChecked = e match { case s: SpeculativeDecimalExpr => s.evalChecked(ctx); case _ => null }
  private val lane: VectorBuffers = if (literal) null else if (checked != null) checked.lane else e.eval(ctx)
  private val validity = if (lane == null) null else lane.validity()
  private var k = 0 // cursor into the child's escalated rows (rows ascend with i)
  /** Row `i` is null (a real null, not an escalation). */
  def isNull(i: Int): Boolean = !literal && validity != null && !Bitmap.isSet(validity, i) && !escalated(i)
  /** Row `i` was escalated by the child: its exact value is `exactAt`. */
  def escalated(i: Int): Boolean = {
    if (checked == null) return false
    while (k < checked.rows.length && checked.rows(k) < i) k += 1
    k < checked.rows.length && checked.rows(k) == i
  }
  def exactAt(i: Int): java.math.BigInteger = checked.exact(k)
  def narrow(i: Int): Long = if (literal) lit else lane.data().get(VectorBuffers.LE_LONG, i.toLong << 3)
  def wide(i: Int): java.math.BigInteger = if (escalated(i)) exactAt(i) else java.math.BigInteger.valueOf(narrow(i))
}

/** The escalated rows of one batch being collected, with Spark's range check against the declared precision. */
private[expr] final class Escalations(dataType: DecimalType, ansi: Boolean, queryContext: QueryContext) {
  private val limit = java.math.BigInteger.TEN.pow(dataType.precision)
  private var rows: Array[Int] = null
  private var exact: Array[java.math.BigInteger] = null
  private var count = 0
  def add(i: Int, v: java.math.BigInteger): Unit = {
    if (v.abs.compareTo(limit) >= 0) {
      // Past the declared precision: Spark's operator yields null (legacy) or raises (ANSI) for this row.
      if (ansi) throw org.apache.spark.sql.vector.VectorErrors.decimalPrecisionOverflow(Decimal(new java.math.BigDecimal(v, dataType.scale)), dataType.precision, dataType.scale, queryContext)
    } else {
      if (rows == null) { rows = new Array[Int](8); exact = new Array[java.math.BigInteger](8) }
      if (count == rows.length) { rows = java.util.Arrays.copyOf(rows, count * 2); exact = java.util.Arrays.copyOf(exact, count * 2) }
      rows(count) = i; exact(count) = v; count += 1
    }
  }
  def result(lane: VectorBuffers): SpeculativeChecked =
    if (count == 0) new SpeculativeChecked(lane, Array.emptyIntArray, Array.empty)
    else new SpeculativeChecked(lane, java.util.Arrays.copyOf(rows, count), java.util.Arrays.copyOf(exact, count))
}

/**
 * `a * b` over decimals whose *declared* result is wider than 18 digits (`decimal(12,2) * decimal(14,2)`
 * is `decimal(27,4)`): Spark's result type is a static rule, not a statement about the data, so the
 * product is computed speculatively in the INT64 unscaled lane and checked per row with
 * `Math.multiplyHigh` -- the high word of the 128-bit product must be the sign extension of the low
 * word. Rows whose product does not fit are *escalated*: their exact product is returned beside the
 * lane (as `BigInteger`, at the declared scale `s1 + s2`, which Spark keeps whenever the declared
 * precision is at most 38) and the lane's validity is cleared for them, so a consumer that can take
 * exact values row by row -- the 128-bit decimal sum -- adds them exactly, and no wrong value ever
 * leaves.
 *
 * An operand may itself be a speculative expression (`(price * (1 - disc)) * (1 + tax)`, declared
 * `decimal(38,6)`): its escalated rows are multiplied exactly, so the whole tree escalates as one.
 * Where Spark capped the declared precision at 38 the product may not fit it: like Spark's
 * `Multiply`, such a row is null in legacy mode and `NUMERIC_VALUE_OUT_OF_RANGE` in ANSI mode --
 * only an escalated (exact) value can be that large.
 */
final case class SpeculativeDecimalMulExpr(
    left: VectorExpr,
    right: VectorExpr,
    leftType: DecimalType,
    rightType: DecimalType,
    dataType: DecimalType,
    ansi: Boolean,
    queryContext: QueryContext)
    extends SpeculativeDecimalExpr {

  override def children: Seq[VectorExpr] = Seq(left, right)

  def evalChecked(ctx: EvalContext): SpeculativeChecked = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    val validity = ctx.bitmap()
    val a = new SpeculativeOperand(left, ctx)
    val b = new SpeculativeOperand(right, ctx)
    val out = new Escalations(dataType, ansi, queryContext)
    var i = 0
    while (i < n) {
      if (!a.isNull(i) && !b.isNull(i)) {
        if (a.escalated(i) || b.escalated(i)) out.add(i, a.wide(i).multiply(b.wide(i)))
        else {
          val x = a.narrow(i); val y = b.narrow(i)
          val lo = x * y
          if (Math.multiplyHigh(x, y) == (lo >> 63)) { data.set(VectorBuffers.LE_LONG, i.toLong << 3, lo); Bitmap.set(validity, i) }
          else out.add(i, java.math.BigInteger.valueOf(x).multiply(java.math.BigInteger.valueOf(y)))
        }
      }
      i += 1
    }
    out.result(SegmentVectorBuffers.fixedWidth(VecType.INT64, n, validity, data))
  }
}

/**
 * `a + b` / `a - b` over decimals whose *declared* result is wider than 18 digits (`decimal(18,4) + 1`
 * is `decimal(19,4)`; `(a * b) + c` under TPC-H-style sums): Spark rescales both operands to the
 * common scale `max(s1, s2)`, adds exactly and checks the declared precision. Here each operand is
 * rescaled in the INT64 lane (the shift by `10^(s - si)` checked with `Math.multiplyHigh`) and added
 * with an overflow check; a row that leaves 64 bits at either step is escalated exactly, and an
 * operand's own escalated rows are combined exactly. Only the shape whose declared scale *is*
 * `max(s1, s2)` is taken -- when Spark's precision cap lowers the scale it also rounds, which is not
 * mirrored here. Past the declared precision a row is null (legacy) or `NUMERIC_VALUE_OUT_OF_RANGE`
 * (ANSI), like Spark's `Add` / `Subtract`.
 */
final case class SpeculativeDecimalAddExpr(
    left: VectorExpr,
    right: VectorExpr,
    subtract: Boolean,
    leftType: DecimalType,
    rightType: DecimalType,
    dataType: DecimalType,
    ansi: Boolean,
    queryContext: QueryContext)
    extends SpeculativeDecimalExpr {
  require(dataType.scale == math.max(leftType.scale, rightType.scale), "the declared scale must be the common scale")

  override def children: Seq[VectorExpr] = Seq(left, right)

  private val leftPow: Long = pow10(dataType.scale - leftType.scale)
  private val rightPow: Long = pow10(dataType.scale - rightType.scale)
  private val leftPowBig = java.math.BigInteger.valueOf(leftPow)
  private val rightPowBig = java.math.BigInteger.valueOf(rightPow)
  private def pow10(d: Int): Long = { var p = 1L; var k = 0; while (k < d) { p *= 10; k += 1 }; p } // d <= 18: fits

  private def exact(x: java.math.BigInteger, y: java.math.BigInteger): java.math.BigInteger = {
    val xs = x.multiply(leftPowBig); val ys = y.multiply(rightPowBig)
    if (subtract) xs.subtract(ys) else xs.add(ys)
  }

  def evalChecked(ctx: EvalContext): SpeculativeChecked = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    val validity = ctx.bitmap()
    val a = new SpeculativeOperand(left, ctx)
    val b = new SpeculativeOperand(right, ctx)
    val out = new Escalations(dataType, ansi, queryContext)
    var i = 0
    while (i < n) {
      if (!a.isNull(i) && !b.isNull(i)) {
        if (a.escalated(i) || b.escalated(i)) out.add(i, exact(a.wide(i), b.wide(i)))
        else {
          val x = a.narrow(i); val y = b.narrow(i)
          val xs = x * leftPow; val ys = y * rightPow
          val r = if (subtract) xs - ys else xs + ys
          // Both rescales fit (high word is the sign extension) and the add/subtract did not overflow (sign trick).
          val fits = Math.multiplyHigh(x, leftPow) == (xs >> 63) && Math.multiplyHigh(y, rightPow) == (ys >> 63) &&
            (if (subtract) ((xs ^ ys) & (xs ^ r)) >= 0 else ((xs ^ r) & (ys ^ r)) >= 0)
          if (fits) { data.set(VectorBuffers.LE_LONG, i.toLong << 3, r); Bitmap.set(validity, i) }
          else out.add(i, exact(java.math.BigInteger.valueOf(x), java.math.BigInteger.valueOf(y)))
        }
      }
      i += 1
    }
    out.result(SegmentVectorBuffers.fixedWidth(VecType.INT64, n, validity, data))
  }
}

/** The narrow products as an INT64 lane (overflowing rows invalid) plus the escalated rows (ascending) and their exact products. */
final class SpeculativeChecked(val lane: VectorBuffers, val rows: Array[Int], val exact: Array[java.math.BigInteger])
