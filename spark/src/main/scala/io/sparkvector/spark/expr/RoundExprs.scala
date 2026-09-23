package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArrowLayout, RoundKernels, SegmentVectorBuffers, VecType, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{DataType, DecimalType, DoubleType}
import org.apache.spark.sql.vector.VectorErrors

/**
 * `ceil` / `floor` of a double (to a long, Java's cast) or of a decimal (an unscaled divide with the
 * directed mode, to Spark's `bounded(p - s + 1, 0)` type). A long child is the identity and never
 * reaches this node.
 */
final case class CeilFloorExpr(child: VectorExpr, ceil: Boolean, dataType: DataType) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, ctx.numRows)
    child.dataType match {
      case d: DecimalType =>
        RoundKernels.roundDecimal(a, d.scale, 0, if (ceil) RoundKernels.Mode.CEILING else RoundKernels.Mode.FLOOR, data)
      case _ => RoundKernels.ceilFloorToLong(a, ceil, data)
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, ctx.numRows, a.validity(), data)
  }
}

/** `rint`: the nearest integral double, ties to even. */
final case class RintExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, ctx.numRows)
    RoundKernels.rint(a, data)
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, ctx.numRows, a.validity(), data)
  }
}

/**
 * Spark's `RoundBase` family -- `round` (half up), `bround` (half even) and the two-argument
 * `ceil` / `floor` -- to a literal `scale`:
 *
 *  - doubles: `BigDecimal(Double.toString(d)).setScale(scale, mode).doubleValue()`, NaN and the
 *    infinities pass through;
 *  - integers: identity for `scale >= 0` (the compiler short-circuits it), otherwise rounded to a
 *    power of ten; a result outside the type raises Spark's `ARITHMETIC_OVERFLOW` (`Overflow`, no
 *    hint) in ANSI mode for active rows, and wraps like `BigDecimal.intValue()` otherwise;
 *  - decimals: an unscaled-value operation to the result type Spark computed (`dataType`), whose
 *    precision always holds the exact result.
 */
final case class RoundExpr(
    child: VectorExpr,
    dataType: DataType,
    mode: RoundKernels.Mode,
    scale: Int,
    ansi: Boolean,
    queryContext: QueryContext
) extends VectorExpr {

  override def children: Seq[VectorExpr] = Seq(child)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, n)
    (child.dataType, dataType) match {
      case (from: DecimalType, to: DecimalType) =>
        if (scale >= 0) RoundKernels.roundDecimal(a, from.scale - to.scale, 0, mode, data)
        else RoundKernels.roundDecimal(a, from.scale - scale, -scale, mode, data)
      case _ if vecType == VecType.FLOAT64 =>
        RoundKernels.roundDouble(a, scale, mode, data)
      case _ =>
        val overflow = ctx.bitmap()
        RoundKernels.roundIntegral(a, -scale, mode, data, overflow)
        if (ansi && ArithExpr.anyActive(overflow, a.validity(), ctx)) {
          throw VectorErrors.arithmeticOverflow("Overflow", "", queryContext)
        }
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, a.validity(), data)
  }
}

object RoundExpr {

  /** Whether a child / result pair is one this node handles (the result must be a 64-bit lane). */
  def supports(from: DataType, to: DataType): Boolean = (from, to) match {
    case (f: DecimalType, t: DecimalType) => TypeMapping.isSupported(f) && TypeMapping.isSupported(t)
    case (f, t) if f == t && !TypeMapping.isDecimal(f) =>
      TypeMapping.isSupported(f) && (TypeMapping.vecTypeOf(f) match {
        case VecType.INT32 | VecType.INT64 | VecType.FLOAT64 => true
        case _ => false
      })
    case _ => false
  }
}
