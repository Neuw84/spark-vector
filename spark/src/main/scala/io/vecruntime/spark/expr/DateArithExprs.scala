/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.sparkvector.spark.expr

import io.sparkvector.kernels.{
  ArrowLayout,
  Bitmap,
  BitmapKernels,
  DateKernels,
  SegmentVectorBuffers,
  VecType,
  VectorBuffers
}
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{DataType, DateType, DoubleType, IntegerType, LongType, TimestampType}
import org.apache.spark.sql.vector.VectorErrors

/** The child's lane under another Spark type -- `unix_date`, `date_from_unix_date`, `timestamp_micros`, `unix_micros`. */
final case class RelabelExpr(child: VectorExpr, dataType: DataType) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = child.eval(ctx)
}

/** Seconds or millis to micros (Spark's `multiplyExact`: `long overflow` on an active row) and back (`floorDiv`). */
final case class EpochScaleExpr(child: VectorExpr, factor: Long, toMicros: Boolean, dataType: DataType)
    extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val v = CaseWhenExpr.materialise(child, ctx)
    val n = ctx.numRows
    val wide = v.`type`() == VecType.INT64
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    var i = 0
    while (i < n) {
      if (v.validity() == null || Bitmap.isSet(v.validity(), i)) {
        val x = if (wide) v.data().getAtIndex(VectorBuffers.LE_LONG, i)
        else v.data().getAtIndex(VectorBuffers.LE_INT, i).toLong
        val r =
          if (toMicros) {
            val hi = Math.multiplyHigh(x, factor)
            val lo = x * factor
            if (hi != (lo >> 63)) {
              if (ctx.active == null || Bitmap.isSet(ctx.active, i)) throw new ArithmeticException("long overflow")
              0L
            } else lo
          } else Math.floorDiv(x, factor)
        out.setAtIndex(VectorBuffers.LE_LONG, i, r)
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, v.validity(), out)
  }
}

/** `last_day`, `weekofyear`, `next_day(date, literal day)` and `add_months(date, months)`: a date in, a date or int out. */
final case class DateScalarExpr(
    kind: DateScalarExpr.Kind,
    child: VectorExpr,
    arg: Option[VectorExpr],
    dataType: DataType
) extends VectorExpr {
  override def children: Seq[VectorExpr] = child +: arg.toSeq
  override def eval(ctx: EvalContext): VectorBuffers = {
    val d = child.eval(ctx)
    val a = arg.map(CaseWhenExpr.materialise(_, ctx)).orNull
    val n = ctx.numRows
    val validity = if (a == null) d.validity() else StringSlices.validity(ctx, d, a)
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT32, n)
    var i = 0
    while (i < n) {
      if (validity == null || Bitmap.isSet(validity, i)) {
        val days = d.data().getAtIndex(VectorBuffers.LE_INT, i)
        val r = kind match {
          case DateScalarExpr.LastDay => DateKernels.lastDay(days)
          case DateScalarExpr.WeekOfYear => DateKernels.weekOfYear(days)
          case DateScalarExpr.NextDay(code) => DateKernels.nextDay(days, code)
          case DateScalarExpr.AddMonths => DateKernels.addMonths(days, a.data().getAtIndex(VectorBuffers.LE_INT, i))
        }
        out.setAtIndex(VectorBuffers.LE_INT, i, r)
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, out)
  }
}

object DateScalarExpr {
  sealed trait Kind
  case object LastDay extends Kind
  case object WeekOfYear extends Kind
  final case class NextDay(code: Int) extends Kind
  case object AddMonths extends Kind
}

/**
 * `months_between(a, b[, roundOff])` under a fixed offset. Each side is a timestamp lane or a date lane
 * (Spark casts dates to timestamps; a date's instant is midnight in the session zone).
 */
final case class MonthsBetweenExpr(
    a: VectorExpr,
    b: VectorExpr,
    aIsDate: Boolean,
    bIsDate: Boolean,
    roundOff: Boolean,
    offsetMicros: Long
) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(a, b)
  private val MicrosPerDay = 86400000000L

  private def micros(v: VectorBuffers, isDate: Boolean, i: Int): Long =
    if (isDate) v.data().getAtIndex(VectorBuffers.LE_INT, i).toLong * MicrosPerDay - offsetMicros
    else v.data().getAtIndex(VectorBuffers.LE_LONG, i)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val va = CaseWhenExpr.materialise(a, ctx)
    val vb = CaseWhenExpr.materialise(b, ctx)
    val n = ctx.numRows
    val validity = StringSlices.validity(ctx, va, vb)
    val out = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, n)
    var i = 0
    while (i < n) {
      if (validity == null || Bitmap.isSet(validity, i)) {
        out.setAtIndex(
          VectorBuffers.LE_DOUBLE,
          i,
          DateKernels.monthsBetween(micros(va, aIsDate, i), micros(vb, bIsDate, i), roundOff, offsetMicros)
        )
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, n, validity, out)
  }
}

/** `make_date(y, m, d)`: null (or Spark's out-of-range error under ANSI, active rows only) for an invalid civil date. */
final case class MakeDateExpr(
    year: VectorExpr,
    month: VectorExpr,
    day: VectorExpr,
    ansi: Boolean,
    queryContext: QueryContext
) extends VectorExpr {
  override def dataType: DataType = DateType
  override def children: Seq[VectorExpr] = Seq(year, month, day)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val y = CaseWhenExpr.materialise(year, ctx)
    val m = CaseWhenExpr.materialise(month, ctx)
    val d = CaseWhenExpr.materialise(day, ctx)
    val n = ctx.numRows
    val in = StringSlices.validity(ctx, y, m, d)
    val validity = ArrowLayout.allocateBitmap(ctx.arena, n)
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT32, n)
    var i = 0
    while (i < n) {
      if (in == null || Bitmap.isSet(in, i)) {
        val yy = y.data().getAtIndex(VectorBuffers.LE_INT, i)
        val mm = m.data().getAtIndex(VectorBuffers.LE_INT, i)
        val dd = d.data().getAtIndex(VectorBuffers.LE_INT, i)
        val r = DateKernels.makeDate(yy, mm, dd)
        if (r != Integer.MIN_VALUE) {
          Bitmap.set(validity, i)
          out.setAtIndex(VectorBuffers.LE_INT, i, r)
        } else if (ansi && (ctx.active == null || Bitmap.isSet(ctx.active, i))) {
          // Spark's own exception for exactly these arguments.
          val e =
            try { java.time.LocalDate.of(yy, mm, dd); null }
            catch { case t: java.time.DateTimeException => t }
          throw VectorErrors.dateTimeArgumentOutOfRange(if (e != null) e
          else new java.time.DateTimeException(s"Invalid date $yy-$mm-$dd"))
        }
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, out)
  }
}
