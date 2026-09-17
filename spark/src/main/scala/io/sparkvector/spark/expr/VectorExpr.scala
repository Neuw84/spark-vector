package io.sparkvector.spark.expr

import java.lang.foreign.Arena

import io.sparkvector.kernels._
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.types.{BooleanType, DataType}

/**
 * Per-batch evaluation context. Input columns are adapted lazily so that a predicate touching two
 * columns out of twenty never pays for adapting the other eighteen unless the batch survives.
 *
 * `selection` (null for all rows) qualifies the batch's rows when the producer emitted a
 * [[io.sparkvector.spark.arrow.SelectedColumnarBatch]]; `active` is the subset of rows whose result
 * anybody will read while evaluating the current sub-expression. It starts as the selection and is
 * narrowed by `AND`/`OR` for their right operands, so kernels can skip 64-row blocks with no live
 * row and ANSI errors are only raised for rows that survive.
 */
final class EvalContext(
    val arena: Arena,
    val numRows: Int,
    adaptColumn: Int => VectorBuffers,
    val selection: java.lang.foreign.MemorySegment,
    val selectedCount: Int,
    rawColumn: Int => org.apache.spark.sql.vectorized.ColumnVector) {
  private val adapted = new java.util.HashMap[Int, VectorBuffers]()

  /** Rows whose values matter for the sub-expression under evaluation (null = all). */
  var active: java.lang.foreign.MemorySegment = selection

  def this(arena: Arena, numRows: Int, adaptColumn: Int => VectorBuffers, selection: java.lang.foreign.MemorySegment, selectedCount: Int) =
    this(arena, numRows, adaptColumn, selection, selectedCount, null)

  def this(arena: Arena, numRows: Int, adaptColumn: Int => VectorBuffers) =
    this(arena, numRows, adaptColumn, null, numRows, null)

  def hasSelection: Boolean = selection != null

  /**
   * The batch's own column, for the one value that has no lane: the wide decimal sum buffer a
   * merging aggregate reads row by row. Null for a context not built over a batch.
   */
  def column(ordinal: Int): org.apache.spark.sql.vectorized.ColumnVector =
    if (rawColumn == null) null else rawColumn(ordinal)

  /** `v` with the batch selection folded into its validity, so reductions skip unselected rows. */
  def masked(v: VectorBuffers): VectorBuffers = {
    if (selection == null) v
    else if (v.validity() == null) new SegmentVectorBuffers(v.`type`(), numRows, selection, v.data(), v.offsets(), v.dictionary())
    else {
      val combined = bitmap()
      BitmapKernels.and(v.validity(), selection, combined, numRows)
      new SegmentVectorBuffers(v.`type`(), numRows, combined, v.data(), v.offsets(), v.dictionary())
    }
  }

  /** Evaluates `f` with `active` narrowed to `rows`, restoring it afterwards. */
  def withActive[T](rows: java.lang.foreign.MemorySegment)(f: => T): T = {
    val saved = active
    active = rows
    try f finally active = saved
  }

  def input(ordinal: Int): VectorBuffers = {
    var v = adapted.get(ordinal)
    if (v == null) {
      v = adaptColumn(ordinal)
      adapted.put(ordinal, v)
    }
    v
  }

  def bitmap(): java.lang.foreign.MemorySegment = ArrowLayout.allocateBitmap(arena, numRows)
}

/**
 * Compiled, serializable expression tree evaluated column-at-a-time with the kernels. Built by
 * [[ExpressionCompiler]] on the driver and shipped inside the physical operator.
 */
trait VectorExpr extends Serializable {
  def dataType: DataType
  final def vecType: VecType = TypeMapping.vecTypeOf(dataType)
  def eval(ctx: EvalContext): VectorBuffers
  def children: Seq[VectorExpr]
}

/** Reference to an input column by ordinal in the child's output. */
final case class ColumnRef(ordinal: Int, dataType: DataType) extends VectorExpr {
  override def eval(ctx: EvalContext): VectorBuffers = ctx.input(ordinal)
  override def children: Seq[VectorExpr] = Nil
}

/**
 * A non-null literal in Spark's internal representation (Int for dates, Long for timestamps).
 * Literals are only valid as operands of a [[CompareExpr]] (and later arithmetic); they cannot be
 * materialised as a column on their own, which is why [[eval]] throws.
 */
final case class LiteralExpr(value: Any, dataType: DataType) extends VectorExpr {
  /** The literal as a lane value: decimals are their unscaled long. */
  def number: Number = value match {
    case d: org.apache.spark.sql.types.Decimal => java.lang.Long.valueOf(d.toUnscaledLong)
    case n: Number => n
  }
  /** A string literal's UTF-8 bytes. */
  def utf8Bytes: Array[Byte] = value match {
    case s: org.apache.spark.unsafe.types.UTF8String => s.getBytes
    case s: String => s.getBytes(java.nio.charset.StandardCharsets.UTF_8)
  }
  override def eval(ctx: EvalContext): VectorBuffers =
    throw new UnsupportedOperationException("literal cannot be evaluated as a column")
  override def children: Seq[VectorExpr] = Nil
}

/**
 * Comparison of two same-typed operands, at most one of which is a literal. Numbers (and the types
 * carried as numeric lanes) go through `CompareKernels`; strings through `StringCompareKernels`, in
 * Spark's default UTF8_BINARY order.
 */
final case class CompareExpr(op: CompareOp, left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)

  private def isString: Boolean = left.vecType == VecType.UTF8

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val bits = ctx.bitmap()
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        if (isString) StringCompareKernels.compareScalar(a, lit.utf8Bytes, op, ctx.active, bits)
        else CompareKernels.compareScalar(a, lit.number, op, ctx.active, bits)
        // Result is null exactly where the column is null: share its validity.
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, a.validity(), bits)
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        if (isString) StringCompareKernels.compareScalar(b, lit.utf8Bytes, op.flip(), ctx.active, bits)
        else CompareKernels.compareScalar(b, lit.number, op.flip(), ctx.active, bits)
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, b.validity(), bits)
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        if (isString) StringCompareKernels.compare(a, b, op, ctx.active, bits)
        else CompareKernels.compare(a, b, op, ctx.active, bits)
        val validity =
          if (a.validity() == null && b.validity() == null) null
          else {
            val v = ctx.bitmap()
            BitmapKernels.combineValidity(a.validity(), b.validity(), v, n)
            v
          }
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
    }
  }
}

/**
 * Spark's three-valued AND. The right operand is only evaluated where the left one is not
 * definitely false (true or null): a false left operand decides the row on its own. Rows outside
 * that set get whatever the skipped kernels left (zero bits), which the Kleene combination ignores.
 */
final case class AndExpr(left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = left.eval(ctx)
    val b = ctx.withActive(LogicalExprs.narrow(ctx, a, keepTrue = true))(right.eval(ctx))
    val bits = ctx.bitmap()
    if (a.validity() == null && b.validity() == null) {
      BitmapKernels.and(a.data(), b.data(), bits, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
    } else {
      val validity = ctx.bitmap()
      BitmapKernels.kleeneAnd(a.data(), a.validity(), b.data(), b.validity(), bits, validity, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
    }
  }
}

/** Spark's three-valued OR; the right operand is only evaluated where the left is not true. */
final case class OrExpr(left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = left.eval(ctx)
    val b = ctx.withActive(LogicalExprs.narrow(ctx, a, keepTrue = false))(right.eval(ctx))
    val bits = ctx.bitmap()
    if (a.validity() == null && b.validity() == null) {
      BitmapKernels.or(a.data(), b.data(), bits, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
    } else {
      val validity = ctx.bitmap()
      BitmapKernels.kleeneOr(a.data(), a.validity(), b.data(), b.validity(), bits, validity, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
    }
  }
}

/**
 * `child IN (v1, ..., vN)` over non-null literals of the child's type: true where any literal
 * matches, null exactly where the child is null (with no null in the list, that is Spark's `In`).
 * The child is evaluated once and each literal costs one equality pass -- through the dictionary
 * for a dictionary-encoded string column, so TPC-H's `l_shipmode IN ('MAIL', 'SHIP')` compares
 * two strings per distinct value, not per row.
 */
final case class InExpr(child: VectorExpr, values: Seq[LiteralExpr]) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = child +: values
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    Bitmap.fill(bits, n, false)
    val hit = ctx.bitmap()
    val isString = child.vecType == VecType.UTF8
    values.foreach { lit =>
      if (isString) StringCompareKernels.compareScalar(a, lit.utf8Bytes, CompareOp.EQ, ctx.active, hit)
      else CompareKernels.compareScalar(a, lit.number, CompareOp.EQ, ctx.active, hit)
      BitmapKernels.or(bits, hit, bits, n)
    }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, a.validity(), bits)
  }
}

/**
 * `startswith(child, p)`, `endswith(child, p)` and `contains(child, p)` against a string literal
 * -- what Spark's `LikeSimplification` makes of `LIKE 'p%'`, `LIKE '%p'` and `LIKE '%p%'`. Null
 * exactly where the child is null (the pattern is a non-null literal), so the child's validity is
 * shared; the match runs once per dictionary entry on a dictionary-encoded column.
 */
final case class StringMatchExpr(kind: StringMatchKernels.Kind, child: VectorExpr, pattern: LiteralExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child, pattern)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    StringMatchKernels.`match`(kind, a, pattern.utf8Bytes, ctx.active, bits)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, a.validity(), bits)
  }
}

final case class NotExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    BitmapKernels.not(a.data(), bits, ctx.numRows)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, a.validity(), bits)
  }
}

final case class IsNullExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    BitmapKernels.isNull(a.validity(), bits, ctx.numRows)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, null, bits)
  }
}

final case class IsNotNullExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    BitmapKernels.isNotNull(a.validity(), bits, ctx.numRows)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, null, bits)
  }
}

private[expr] object LogicalExprs {
  /**
   * Rows still undecided after seeing `a` as the left operand of AND (`keepTrue`: rows where `a`
   * is true or null) or OR (rows where `a` is false or null), intersected with the current active
   * set. Returns null when nothing narrows.
   */
  def narrow(ctx: EvalContext, a: VectorBuffers, keepTrue: Boolean): java.lang.foreign.MemorySegment = {
    val n = ctx.numRows
    val undecided = ctx.bitmap()
    if (keepTrue) {
      // true or null == not (false and valid) == data | ~validity
      if (a.validity() == null) BitmapKernels.copy(a.data(), undecided, n)
      else { BitmapKernels.andNot(a.validity(), a.data(), undecided, n); BitmapKernels.not(undecided, undecided, n) }
    } else {
      // false or null == not (true and valid)
      if (a.validity() == null) BitmapKernels.not(a.data(), undecided, n)
      else { BitmapKernels.and(a.data(), a.validity(), undecided, n); BitmapKernels.not(undecided, undecided, n) }
    }
    if (ctx.active != null) BitmapKernels.and(undecided, ctx.active, undecided, n)
    undecided
  }
}

object VectorExpr {

  /** All column ordinals an expression reads. */
  def referencedColumns(e: VectorExpr): Set[Int] = e match {
    case ColumnRef(o, _) => Set(o)
    case other => other.children.flatMap(referencedColumns).toSet
  }

  /**
   * Selection bitmap for a boolean result: true and not null. Returns the number of selected rows.
   */
  def selection(pred: VectorBuffers, ctx: EvalContext): (java.lang.foreign.MemorySegment, Int) = {
    val sel = ctx.bitmap()
    BitmapKernels.selection(pred.data(), pred.validity(), sel, ctx.numRows)
    if (ctx.selection != null) BitmapKernels.and(sel, ctx.selection, sel, ctx.numRows)
    (sel, Bitmap.popcount(sel, ctx.numRows))
  }
}

/**
 * Binary arithmetic on same-typed operands (at most one literal). Division is only defined for
 * doubles; a zero divisor yields null in legacy mode and raises DIVIDE_BY_ZERO in ANSI mode,
 * exactly like Spark's `Divide`.
 */
final case class ArithExpr(
    op: ArithOp,
    left: VectorExpr,
    right: VectorExpr,
    dataType: DataType,
    ansiDivideByZero: Boolean,
    queryContext: org.apache.spark.QueryContext,
    nullOnOverflow: Boolean = false)
    extends VectorExpr {

  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, vecType, n)
    var validity: java.lang.foreign.MemorySegment = null
    var divisorZero: java.lang.foreign.MemorySegment = null
    // ANSI mode: integer results that did not fit their lane raise for active rows (below); the
    // try_* forms (nullOnOverflow) null those rows instead.
    val checkOverflow = (ansiDivideByZero || nullOnOverflow) && (vecType == VecType.INT32 || vecType == VecType.INT64)
    var overflow: java.lang.foreign.MemorySegment = null
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        ArithKernels.arithScalar(op, a, lit.number, data)
        validity = a.validity()
        if (op == ArithOp.DIV && lit.number.doubleValue() == 0.0) {
          divisorZero = ctx.bitmap()
          Bitmap.fill(divisorZero, n, true)
        }
        if (checkOverflow) { overflow = ctx.bitmap(); OverflowKernels.overflowScalar(op, a, lit.number, data, overflow) }
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        ArithKernels.scalarArith(op, lit.number, b, data)
        validity = b.validity()
        if (op == ArithOp.DIV) divisorZero = zeroMask(b, ctx)
        if (checkOverflow) { overflow = ctx.bitmap(); OverflowKernels.scalarOverflow(op, lit.number, b, data, overflow) }
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        ArithKernels.arith(op, a, b, data)
        if (a.validity() != null || b.validity() != null) {
          validity = ctx.bitmap()
          BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n)
        }
        if (op == ArithOp.DIV) divisorZero = zeroMask(b, ctx)
        if (checkOverflow) { overflow = ctx.bitmap(); OverflowKernels.overflow(op, a, b, data, overflow) }
    }
    if (overflow != null && nullOnOverflow) {
      val newValidity = ctx.bitmap()
      if (validity == null) BitmapKernels.not(overflow, newValidity, n)
      else BitmapKernels.andNot(validity, overflow, newValidity, n)
      validity = newValidity
    } else if (overflow != null && ArithExpr.anyActive(overflow, validity, ctx)) {
      throw org.apache.spark.sql.vector.VectorErrors.arithmeticOverflow(ArithExpr.overflowMessage(vecType), ArithExpr.hint(op), queryContext)
    }
    if (divisorZero != null) {
      // Lanes that are otherwise valid but divide by zero.
      val affected = ctx.bitmap()
      if (validity == null) BitmapKernels.copy(divisorZero, affected, n)
      else BitmapKernels.and(divisorZero, validity, affected, n)
      // Rows nobody will read (filtered out, or decided by the other side of a conjunction) must
      // not raise, exactly as Spark never evaluates them.
      if (ctx.active != null) BitmapKernels.and(affected, ctx.active, affected, n)
      val count = Bitmap.popcount(affected, n)
      if (count > 0) {
        if (ansiDivideByZero) {
          throw org.apache.spark.sql.vector.VectorErrors.divideByZero(queryContext)
        }
        val newValidity = ctx.bitmap()
        if (validity == null) BitmapKernels.not(divisorZero, newValidity, n)
        else BitmapKernels.andNot(validity, divisorZero, newValidity, n)
        validity = newValidity
      }
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, validity, data)
  }

  private def zeroMask(b: VectorBuffers, ctx: EvalContext): java.lang.foreign.MemorySegment = {
    val zero = ctx.bitmap()
    CompareKernels.compareScalar(b, java.lang.Double.valueOf(0.0), CompareOp.EQ, zero)
    zero
  }
}

object ArithExpr {
  /**
   * Whether any lane flagged in `mask` is both valid and active. Rows a filter removed or an earlier
   * conjunct decided must not raise, exactly as Spark never evaluates them.
   */
  private[expr] def anyActive(mask: java.lang.foreign.MemorySegment, validity: java.lang.foreign.MemorySegment, ctx: EvalContext): Boolean = {
    val n = ctx.numRows
    if (validity != null) BitmapKernels.and(mask, validity, mask, n)
    if (ctx.active != null) BitmapKernels.and(mask, ctx.active, mask, n)
    Bitmap.popcount(mask, n) > 0
  }

  /** Spark raises `Math.addExact`'s own message: `integer overflow` or `long overflow`. */
  private[expr] def overflowMessage(vecType: VecType): String =
    if (vecType == VecType.INT32) "integer overflow" else "long overflow"

  /** The `try_*` function Spark suggests in the error. */
  private[expr] def hint(op: ArithOp): String = op match {
    case ArithOp.ADD => "try_add"
    case ArithOp.SUB => "try_subtract"
    case ArithOp.MUL => "try_multiply"
    case ArithOp.DIV => "try_divide"
  }
}

/** Widening numeric cast; validity is shared with the child. */
final case class CastExpr(child: VectorExpr, dataType: DataType) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    CastKernels.cast(a, vecType, data)
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), data)
  }
}

/**
 * Unary minus. In ANSI mode an integer `MIN_VALUE` cannot be negated and raises Spark's overflow
 * error for active rows (`Math.negateExact`'s message, no `try_*` hint); doubles and decimals of at
 * most 18 digits never overflow here.
 */
final case class NegateExpr(child: VectorExpr, ansi: Boolean = false, queryContext: org.apache.spark.QueryContext = null) extends VectorExpr {
  override def dataType: DataType = child.dataType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    ArithKernels.negate(a, data)
    if (ansi && (vecType == VecType.INT32 || vecType == VecType.INT64)) {
      val overflow = ctx.bitmap()
      OverflowKernels.negateOverflow(a, overflow)
      if (ArithExpr.anyActive(overflow, a.validity(), ctx)) {
        throw org.apache.spark.sql.vector.VectorErrors.arithmeticOverflow(ArithExpr.overflowMessage(vecType), "", queryContext)
      }
    }
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), data)
  }
}
