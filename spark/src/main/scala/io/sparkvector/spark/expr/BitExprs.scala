package io.sparkvector.spark.expr

import java.lang.foreign.MemorySegment

import io.sparkvector.kernels.{ArrowLayout, BitKernels, BitmapKernels, SegmentVectorBuffers, VecType, VectorBuffers}
import org.apache.spark.sql.types.{DataType, IntegerType}

/**
 * `&`, `|`, `^` and the three shifts over INT32 / INT64 lanes with Java's semantics (Spark's). A
 * literal on either side of and/or/xor and a literal on either side of a shift are handled without
 * materialising a constant column. Validity is the operands' conjunction.
 */
final case class BitBinaryExpr(op: BitKernels.BitOp, left: VectorExpr, right: VectorExpr, dataType: DataType)
    extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, vecType, n)
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        BitKernels.binaryScalar(op, a, lit.number.longValue(), data)
        SegmentVectorBuffers.fixedWidth(vecType, n, a.validity(), data)
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        if (op.isShift) BitKernels.shiftScalarValue(op, vecType, lit.number.longValue(), b, data)
        else BitKernels.binaryScalar(op, b, lit.number.longValue(), data)
        SegmentVectorBuffers.fixedWidth(vecType, n, b.validity(), data)
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        BitKernels.binary(op, a, b, data)
        var validity: MemorySegment = null
        if (a.validity() != null || b.validity() != null) {
          validity = ctx.bitmap()
          BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n)
        }
        SegmentVectorBuffers.fixedWidth(vecType, n, validity, data)
    }
  }
}

/** `~x`. */
final case class BitNotExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = child.dataType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    BitKernels.not(a, data)
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), data)
  }
}

/** `bit_count(x)`: an int, counted on the value widened to a long as Spark does. */
final case class BitCountExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = IntegerType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT32, ctx.numRows)
    BitKernels.bitCount(a, data)
    SegmentVectorBuffers.fixedWidth(VecType.INT32, ctx.numRows, a.validity(), data)
  }
}
