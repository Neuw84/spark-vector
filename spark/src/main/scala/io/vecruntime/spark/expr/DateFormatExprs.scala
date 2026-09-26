/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package io.vecruntime.spark.expr

import io.vecruntime.kernels.{
  ArrowLayout,
  Bitmap,
  DateKernels,
  SegmentVectorBuffers,
  StringConcatKernels,
  VecType,
  VectorBuffers
}
import org.apache.spark.sql.types.{DataType, LongType, StringType, TimestampType}
import org.apache.spark.sql.vecruntime.SparkFormatters
import org.apache.spark.unsafe.types.UTF8String

/** Shared micros-of-row reading for a timestamp lane or a date lane behind Spark's date -> timestamp cast. */
private[expr] object Instants {
  val MicrosPerDay = 86400000000L
  def micros(v: VectorBuffers, isDate: Boolean, offsetMicros: Long, i: Int): Long =
    if (isDate) v.data().getAtIndex(VectorBuffers.LE_INT, i).toLong * MicrosPerDay - offsetMicros
    else v.data().getAtIndex(VectorBuffers.LE_LONG, i)
}

/**
 * `date_format(ts, literal pattern)` and `from_unixtime(seconds, literal pattern)`: Spark's own
 * `TimestampFormatter` for the session zone, applied per row, the results written as one UTF8 lane.
 * A date side needs a fixed offset (its instant is our arithmetic); a timestamp side takes any zone.
 */
final case class FormatInstantExpr(
    child: VectorExpr,
    childIsDate: Boolean,
    secondsIn: Boolean,
    pattern: String,
    timeZoneId: String,
    offsetMicros: Long
) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(child)
  @transient private lazy val formatter =
    SparkFormatters.timestampFormatter(pattern, SparkFormatters.zoneId(timeZoneId))

  override def eval(ctx: EvalContext): VectorBuffers = {
    val v = CaseWhenExpr.materialise(child, ctx)
    val n = ctx.numRows
    val rows = new Array[Array[Byte]](n)
    var i = 0
    while (i < n) {
      if (v.validity() == null || Bitmap.isSet(v.validity(), i)) {
        val micros =
          if (secondsIn)
            v.data().getAtIndex(VectorBuffers.LE_LONG, i) * 1000000L // Spark multiplies without an overflow check
          else Instants.micros(v, childIsDate, offsetMicros, i)
        rows(i) = UTF8String.fromString(formatter.format(micros)).getBytes
      }
      i += 1
    }
    StringConcatKernels.fromRows(rows, v.validity(), ctx.arena)
  }
}

/** `unix_timestamp(ts|date)` / `to_unix_timestamp`: Spark truncates micros toward zero into seconds. */
final case class UnixTimestampExpr(child: VectorExpr, childIsDate: Boolean, offsetMicros: Long) extends VectorExpr {
  override def dataType: DataType = LongType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val v = CaseWhenExpr.materialise(child, ctx)
    val n = ctx.numRows
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    var i = 0
    while (i < n) {
      if (v.validity() == null || Bitmap.isSet(v.validity(), i)) {
        out.setAtIndex(VectorBuffers.LE_LONG, i, Instants.micros(v, childIsDate, offsetMicros, i) / 1000000L)
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, v.validity(), out)
  }
}

/** `date_trunc(literal unit, ts)` under a fixed offset: Spark's `truncTimestamp` -- floor in local time. */
final case class TruncTimestampExpr(
    unit: TruncTimestampExpr.Unit,
    child: VectorExpr,
    childIsDate: Boolean,
    offsetMicros: Long
) extends VectorExpr {
  override def dataType: DataType = TimestampType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val v = CaseWhenExpr.materialise(child, ctx)
    val n = ctx.numRows
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    var i = 0
    while (i < n) {
      if (v.validity() == null || Bitmap.isSet(v.validity(), i)) {
        val micros = Instants.micros(v, childIsDate, offsetMicros, i)
        val r = unit match {
          case TruncTimestampExpr.Micros => micros
          case TruncTimestampExpr.Sub(u) => micros - Math.floorMod(micros, u) // offsets are whole seconds, so zone-free
          case TruncTimestampExpr.Local(u) => Math.floorDiv(micros + offsetMicros, u) * u - offsetMicros
          case TruncTimestampExpr.Date(du) =>
            val days = Math.floorDiv(micros + offsetMicros, Instants.MicrosPerDay).toInt
            DateKernels.trunc(du, days).toLong * Instants.MicrosPerDay - offsetMicros
        }
        out.setAtIndex(VectorBuffers.LE_LONG, i, r)
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, v.validity(), out)
  }
}

object TruncTimestampExpr {
  sealed trait Unit
  case object Micros extends Unit

  /** Millisecond / second: independent of the zone. */
  final case class Sub(micros: Long) extends Unit

  /** Minute / hour / day: a floor in local time. */
  final case class Local(micros: Long) extends Unit

  /** Week / month / quarter / year: the date kernel's truncation of the local day. */
  final case class Date(unit: DateKernels.TruncUnit) extends Unit

  /** Spark's `date_trunc` formats (case-insensitive). */
  def of(format: String): Option[Unit] = format.toUpperCase(java.util.Locale.ROOT) match {
    case "MICROSECOND" => Some(Micros)
    case "MILLISECOND" => Some(Sub(1000L))
    case "SECOND" => Some(Sub(1000000L))
    case "MINUTE" => Some(Local(60000000L))
    case "HOUR" => Some(Local(3600000000L))
    case "DAY" | "DD" => Some(Local(Instants.MicrosPerDay))
    case other => DateExprs.truncUnit(other).map(Date)
  }
}
