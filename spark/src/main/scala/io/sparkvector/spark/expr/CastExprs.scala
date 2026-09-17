package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArrowLayout, Bitmap, CastKernels, SegmentVectorBuffers, StringConcatKernels, VecType, VectorBuffers}
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{BooleanType, DataType, DoubleType, IntegerType, LongType, StringType, TimestampType}
import org.apache.spark.sql.vector.{SparkCasts, SparkFormatters, VectorErrors}
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

/**
 * `string -> int / bigint / double` with Spark's own parsers per row: legacy `UTF8String.toInt/toLong`
 * (trimmed, optional sign, an all-digit fraction dropped, null on anything else or overflow) and
 * `Double.parseDouble` with Spark's special literals (`NaN`, `Infinity`, `inf`, ...); under ANSI the
 * exact parsers raise Spark's CAST_INVALID_INPUT, checked for active rows only.
 */
final case class StringToNumberExpr(child: VectorExpr, dataType: DataType, ansi: Boolean, queryContext: QueryContext) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val n = ctx.numRows
    val out = ArrowLayout.allocateData(ctx.arena, vecType, n)
    val validity = ArrowLayout.allocateBitmap(ctx.arena, n)
    val intBox = new UTF8String.IntWrapper
    val longBox = new UTF8String.LongWrapper
    var i = 0
    while (i < n) {
      if (a.validity() == null || Bitmap.isSet(a.validity(), i)) {
        val s = UTF8String.fromBytes(a.getUtf8Bytes(i))
        val active = ctx.active == null || Bitmap.isSet(ctx.active, i)
        vecType match {
          case VecType.INT32 =>
            if (ansi && active) { out.setAtIndex(VectorBuffers.LE_INT, i, SparkCasts.toIntExact(s, queryContext)); Bitmap.set(validity, i) }
            else if (s.toInt(intBox)) { out.setAtIndex(VectorBuffers.LE_INT, i, intBox.value); Bitmap.set(validity, i) }
          case VecType.INT64 =>
            if (ansi && active) { out.setAtIndex(VectorBuffers.LE_LONG, i, SparkCasts.toLongExact(s, queryContext)); Bitmap.set(validity, i) }
            else if (s.toLong(longBox)) { out.setAtIndex(VectorBuffers.LE_LONG, i, longBox.value); Bitmap.set(validity, i) }
          case _ =>
            val str = s.toString
            val d: java.lang.Double =
              try java.lang.Double.parseDouble(str)
              catch { case _: NumberFormatException => SparkCasts.specialDouble(str) }
            if (d != null) { out.setAtIndex(VectorBuffers.LE_DOUBLE, i, d.doubleValue()); Bitmap.set(validity, i) }
            else if (ansi && active) throw SparkCasts.invalidNumber(DoubleType, s, queryContext)
        }
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, validity, out)
  }
}

/** `date / timestamp -> string`: the formatters Spark's cast uses (ISO date; fraction-trimmed timestamp in the session zone), per row. */
final case class DateTimeToStringExpr(child: VectorExpr, isDate: Boolean, timeZoneId: String) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(child)
  @transient private lazy val dates = SparkCasts.dateFormatter()
  @transient private lazy val timestamps = SparkCasts.fractionFormatter(SparkFormatters.zoneId(timeZoneId))
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val n = ctx.numRows
    val rows = new Array[Array[Byte]](n)
    var i = 0
    while (i < n) {
      if (a.validity() == null || Bitmap.isSet(a.validity(), i)) {
        val str = if (isDate) dates.format(a.data().getAtIndex(VectorBuffers.LE_INT, i)) else timestamps.format(a.data().getAtIndex(VectorBuffers.LE_LONG, i))
        rows(i) = UTF8String.fromString(str).getBytes
      }
      i += 1
    }
    StringConcatKernels.fromRows(rows, a.validity(), ctx.arena)
  }
}

/** `string -> date / timestamp` with Spark's own parser per row (every form Spark accepts, any zone); null or, under ANSI, Spark's error for an active row. */
final case class StringToDateTimeExpr(child: VectorExpr, toTimestamp: Boolean, timeZoneId: String, ansi: Boolean, queryContext: QueryContext) extends VectorExpr {
  override def dataType: DataType = if (toTimestamp) TimestampType else org.apache.spark.sql.types.DateType
  override def children: Seq[VectorExpr] = Seq(child)
  @transient private lazy val zone = SparkFormatters.zoneId(timeZoneId)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val n = ctx.numRows
    val out = ArrowLayout.allocateData(ctx.arena, vecType, n)
    val validity = ArrowLayout.allocateBitmap(ctx.arena, n)
    var i = 0
    while (i < n) {
      if (a.validity() == null || Bitmap.isSet(a.validity(), i)) {
        val s = UTF8String.fromBytes(a.getUtf8Bytes(i))
        val active = ctx.active == null || Bitmap.isSet(ctx.active, i)
        if (toTimestamp) {
          if (ansi && active) { out.setAtIndex(VectorBuffers.LE_LONG, i, SparkCasts.stringToTimestampAnsi(s, zone, queryContext)); Bitmap.set(validity, i) }
          else SparkCasts.stringToTimestamp(s, zone).foreach { v => out.setAtIndex(VectorBuffers.LE_LONG, i, v); Bitmap.set(validity, i) }
        } else {
          if (ansi && active) { out.setAtIndex(VectorBuffers.LE_INT, i, SparkCasts.stringToDateAnsi(s, queryContext)); Bitmap.set(validity, i) }
          else SparkCasts.stringToDate(s).foreach { v => out.setAtIndex(VectorBuffers.LE_INT, i, v); Bitmap.set(validity, i) }
        }
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, validity, out)
  }
}

object CastExprs {
  /** The lane types the first cast slice writes strings for. */
  def stringable(dt: DataType): Boolean = dt == IntegerType || dt == LongType || dt == DoubleType || dt == BooleanType
}
