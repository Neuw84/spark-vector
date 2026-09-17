package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArrowLayout, Bitmap, CastKernels, SegmentVectorBuffers, StringConcatKernels, VecType, VectorBuffers}
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{BooleanType, DataType, DoubleType, IntegerType, LongType, StringType, TimestampType}
import org.apache.spark.sql.vector.VectorErrors
import org.apache.spark.unsafe.types.UTF8String

/**
 * `bigint -> int`, `double -> int / bigint`: Java's narrowing (a long wraps; a double truncates toward
 * zero, saturates, NaN is 0), and under ANSI Spark's `CAST_OVERFLOW` for the first out-of-range value
 * among the batch's active rows -- a filtered row never raises.
 */
final case class NarrowCastExpr(child: VectorExpr, dataType: DataType, ansi: Boolean, queryContext: QueryContext) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val n = ctx.numRows
    val out = ArrowLayout.allocateData(ctx.arena, vecType, n)
    val overflow = if (ansi) ctx.bitmap() else null
    CastKernels.narrow(a, vecType, out, overflow)
    if (ansi && ArithExpr.anyActive(overflow, a.validity(), ctx)) {
      var i = 0
      while (i < n) {
        if (Bitmap.isSet(overflow, i) && (a.validity() == null || Bitmap.isSet(a.validity(), i)) && (ctx.active == null || Bitmap.isSet(ctx.active, i))) {
          val value: Any = if (a.`type`() == VecType.INT64) a.data().getAtIndex(VectorBuffers.LE_LONG, i) else a.data().getAtIndex(VectorBuffers.LE_DOUBLE, i)
          throw VectorErrors.castOverflow(value, child.dataType, dataType)
        }
        i += 1
      }
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, a.validity(), out)
  }
}

/** `int / bigint / double -> boolean`: `v != 0`, so NaN is true. */
final case class ToBooleanExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ArrowLayout.allocateBitmap(ctx.arena, ctx.numRows)
    CastKernels.toBool(a, bits)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, a.validity(), bits)
  }
}

/** `boolean -> int / bigint / double`: 1 or 0. */
final case class FromBooleanExpr(child: VectorExpr, dataType: DataType) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val out = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    CastKernels.fromBool(a, vecType, out)
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), out)
  }
}

/** `date -> timestamp` under a UTC or fixed-offset session zone: local midnight as micros. */
final case class DateToTimestampExpr(child: VectorExpr, offsetMicros: Long) extends VectorExpr {
  override def dataType: DataType = TimestampType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT64, ctx.numRows)
    CastKernels.daysToMicros(a, offsetMicros, out)
    SegmentVectorBuffers.fixedWidth(VecType.INT64, ctx.numRows, a.validity(), out)
  }
}

/** `int / bigint / double / boolean -> string`: Java's `toString`, which is what Spark's cast calls, through the row writer. */
final case class ToStringExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = CaseWhenExpr.materialise(child, ctx)
    val n = ctx.numRows
    val rows = new Array[Array[Byte]](n)
    val t = a.`type`()
    var i = 0
    while (i < n) {
      if (a.validity() == null || Bitmap.isSet(a.validity(), i)) {
        val s = t match {
          case VecType.INT32 => Integer.toString(a.data().getAtIndex(VectorBuffers.LE_INT, i))
          case VecType.INT64 => java.lang.Long.toString(a.data().getAtIndex(VectorBuffers.LE_LONG, i))
          case VecType.FLOAT64 => java.lang.Double.toString(a.data().getAtIndex(VectorBuffers.LE_DOUBLE, i))
          case VecType.BOOL => if (Bitmap.isSet(a.data(), i)) "true" else "false"
          case other => throw new IllegalArgumentException(s"to string of $other")
        }
        rows(i) = s.getBytes(java.nio.charset.StandardCharsets.US_ASCII)
      }
      i += 1
    }
    StringConcatKernels.fromRows(rows, a.validity(), ctx.arena)
  }
}

/** `string -> boolean`: Spark's accepted spellings; anything else is null, or under ANSI Spark's error for an active row. */
final case class StringToBooleanExpr(child: VectorExpr, ansi: Boolean, queryContext: QueryContext) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val n = ctx.numRows
    val bits = ArrowLayout.allocateBitmap(ctx.arena, n)
    val validity = ArrowLayout.allocateBitmap(ctx.arena, n)
    var i = 0
    while (i < n) {
      if (a.validity() == null || Bitmap.isSet(a.validity(), i)) {
        val s = UTF8String.fromBytes(a.getUtf8Bytes(i))
        if (VectorErrors.isTrueString(s)) { Bitmap.set(bits, i); Bitmap.set(validity, i) }
        else if (VectorErrors.isFalseString(s)) Bitmap.set(validity, i)
        else if (ansi && (ctx.active == null || Bitmap.isSet(ctx.active, i))) throw VectorErrors.invalidBooleanInput(s, queryContext)
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
  }
}

object CastExprs {
  /** The lane types the first cast slice writes strings for. */
  def stringable(dt: DataType): Boolean = dt == IntegerType || dt == LongType || dt == DoubleType || dt == BooleanType
}
