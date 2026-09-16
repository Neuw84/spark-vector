package io.sparkvector.spark.expr

import java.lang.foreign.MemorySegment

import io.sparkvector.kernels.{Bitmap, BitmapKernels, CompareOp, PredicateKernels, SegmentVectorBuffers, VecType, VectorBuffers}
import org.apache.spark.sql.types.{BooleanType, DataType}

/**
 * `a <=> b`: true where both are null or both are non-null and equal, false otherwise, never null.
 * The equality is the ordinary comparison node's; this node only fixes up its result with the two
 * validity bitmaps: `(eq AND valid) OR (aNull AND bNull)`.
 */
final case class NullSafeEqExpr(left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    // Each side is evaluated once; the ordinary comparison node then dispatches on lane type / literal.
    val a = left match { case _: LiteralExpr => null; case c => c.eval(ctx) }
    val b = right match { case _: LiteralExpr => null; case c => c.eval(ctx) }
    val eq =
      if (left.vecType == VecType.BOOL) BoolCompareExpr(CompareOp.EQ, NullSafeEqExpr.wrap(left, a), NullSafeEqExpr.wrap(right, b)).eval(ctx)
      else CompareExpr(CompareOp.EQ, NullSafeEqExpr.wrap(left, a), NullSafeEqExpr.wrap(right, b)).eval(ctx)
    if (eq.validity() == null) {
      // Neither side can be null: plain equality.
      return SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, eq.data())
    }
    val bits = ctx.bitmap()
    BitmapKernels.and(eq.data(), eq.validity(), bits, n)
    if (a != null && b != null && a.validity() != null && b.validity() != null) {
      val aNull = ctx.bitmap()
      val bothNull = ctx.bitmap()
      BitmapKernels.isNull(a.validity(), aNull, n)
      BitmapKernels.isNull(b.validity(), bothNull, n)
      BitmapKernels.and(aNull, bothNull, bothNull, n)
      BitmapKernels.or(bits, bothNull, bits, n)
    }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
  }
}

object NullSafeEqExpr {
  /** An already-evaluated column, so a node can be re-used on buffers without re-evaluating its child. */
  private[expr] final case class Evaluated(buffers: VectorBuffers, dataType: DataType) extends VectorExpr {
    override def children: Seq[VectorExpr] = Nil
    override def eval(ctx: EvalContext): VectorBuffers = buffers
  }

  private def wrap(e: VectorExpr, evaluated: VectorBuffers): VectorExpr =
    if (evaluated == null) e else Evaluated(evaluated, e.dataType)
}

/** `isnan(x)` over doubles; false (not null) where `x` is null, as Spark's. */
final case class IsNaNExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    PredicateKernels.isNaN(a, bits)
    if (a.validity() != null) BitmapKernels.and(bits, a.validity(), bits, n)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
  }
}

/** Comparison of two BOOL columns (`false < true`), null where either is. */
final case class BoolCompareExpr(op: CompareOp, left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = left.eval(ctx)
    val b = right.eval(ctx)
    val bits = ctx.bitmap()
    PredicateKernels.compareBool(a.data(), b.data(), op, bits, n)
    var validity: MemorySegment = null
    if (a.validity() != null || b.validity() != null) {
      validity = ctx.bitmap()
      BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n)
    }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
  }
}

/** Comparison of a BOOL column against a boolean literal; null where the column is. */
final case class BoolCompareScalarExpr(op: CompareOp, child: VectorExpr, value: Boolean) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    PredicateKernels.compareBoolScalar(a.data(), value, op, bits, n)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, a.validity(), bits)
  }
}

/**
 * `child IN set` for a numeric lane and a literal set without nulls (Spark's `InSet`, the optimizer's
 * rewrite of a long `IN` list): one binary search per row over the sorted keys, null exactly where
 * the child is null. Doubles are matched on their bit images, which is Spark's boxed-set semantics.
 */
final case class InSetExpr(child: VectorExpr, keys: Array[Long]) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    PredicateKernels.inSet(a, keys, bits)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, a.validity(), bits)
  }
  override def toString: String = s"InSetExpr($child, ${keys.length} keys)"
}
