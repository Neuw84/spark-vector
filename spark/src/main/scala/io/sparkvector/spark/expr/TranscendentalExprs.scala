package io.sparkvector.spark.expr

import java.lang.foreign.MemorySegment

import io.sparkvector.kernels.{
  ArrowLayout,
  BitmapKernels,
  SegmentVectorBuffers,
  TranscendentalKernels,
  VecType,
  VectorBuffers
}
import io.sparkvector.kernels.TranscendentalKernels.{Fn, Fn2}
import org.apache.spark.sql.types.{DataType, DoubleType}

/**
 * A unary transcendental function over a double lane: `exp`, the log family, the trigonometric,
 * hyperbolic and inverse hyperbolic functions, `cot`/`sec`/`csc`, `degrees`/`radians`. The kernel makes
 * exactly Spark's call per lane; the log family is null at or below its asymptote (Spark's
 * `UnaryLogExpression`), everything else keeps the argument's validity.
 */
final case class UnaryMathExpr(fn: Fn, child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(child)
  private val asymptote = TranscendentalKernels.asymptote(fn)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, n)
    TranscendentalKernels.unary(fn, a, data)
    val validity =
      if (asymptote.isNaN) a.validity()
      else {
        val inDomain = ctx.bitmap()
        TranscendentalKernels.above(a, asymptote, inDomain)
        if (a.validity() != null) BitmapKernels.and(inDomain, a.validity(), inDomain, n)
        inDomain
      }
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, n, validity, data)
  }
}

/**
 * `pow`, `atan2`, `hypot` and `log(base, x)` over double lanes, a literal on either side allowed.
 * `log(base, x)` is null when either operand is non-positive, as Spark's `Logarithm`.
 */
final case class BinaryMathExpr(fn: Fn2, left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, n)
    var validity: MemorySegment = null
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        TranscendentalKernels.binaryScalar(fn, a, lit.number.doubleValue(), false, data)
        validity = a.validity()
        if (fn == Fn2.LOG_BASE) validity = positive(a, validity, ctx, !(lit.number.doubleValue() <= 0.0))
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        TranscendentalKernels.binaryScalar(fn, b, lit.number.doubleValue(), true, data)
        validity = b.validity()
        if (fn == Fn2.LOG_BASE) validity = positive(b, validity, ctx, !(lit.number.doubleValue() <= 0.0))
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        TranscendentalKernels.binary(fn, a, b, data)
        if (a.validity() != null || b.validity() != null) {
          validity = ctx.bitmap()
          BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n)
        }
        if (fn == Fn2.LOG_BASE)
          validity = positive(b, positive(a, validity, ctx, literalPositive = true), ctx, literalPositive = true)
    }
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, n, validity, data)
  }

  /** The validity narrowed to the lanes where `!(v <= 0)` (NaN kept, as Spark); all null when the literal side is non-positive. */
  private def positive(
      v: VectorBuffers,
      validity: MemorySegment,
      ctx: EvalContext,
      literalPositive: Boolean
  ): MemorySegment = {
    val n = ctx.numRows
    val out = ctx.bitmap()
    if (!literalPositive) return out // freshly allocated bitmaps are all clear: every lane null
    TranscendentalKernels.above(v, 0.0, out)
    if (validity != null) BitmapKernels.and(out, validity, out, n)
    out
  }
}
