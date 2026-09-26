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

import java.time.ZoneId

import io.vecruntime.kernels.{ArrowLayout, DateKernels, SegmentVectorBuffers, VecType, VectorBuffers}
import org.apache.spark.sql.types.{DataType, DateType, IntegerType}

/**
 * `year`, `month`, `dayofmonth`, `dayofyear`, `quarter`, `dayofweek`, `weekday` of a date column
 * (INT32 days) -- integer arithmetic per lane (`DateKernels.field`), validity shared with the child.
 */
final case class DateFieldExpr(field: DateKernels.Field, child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = IntegerType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT32, ctx.numRows)
    DateKernels.field(field, a, data)
    SegmentVectorBuffers.fixedWidth(VecType.INT32, ctx.numRows, a.validity(), data)
  }
}

/** `trunc(date, unit)` for year / quarter / month / week; a date in, a date out. */
final case class DateTruncExpr(unit: DateKernels.TruncUnit, child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DateType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT32, ctx.numRows)
    DateKernels.trunc(unit, a, data)
    SegmentVectorBuffers.fixedWidth(VecType.INT32, ctx.numRows, a.validity(), data)
  }
}

/**
 * `cast(timestamp AS date)` under a session zone with a fixed offset: the local day of each
 * instant is `floorDiv(micros + offset, micros per day)`. Zones with rules are not compiled.
 */
final case class TimestampToDateExpr(child: VectorExpr, offsetMicros: Long) extends VectorExpr {
  override def dataType: DataType = DateType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT32, ctx.numRows)
    DateKernels.timestampToDate(a, offsetMicros, data)
    SegmentVectorBuffers.fixedWidth(VecType.INT32, ctx.numRows, a.validity(), data)
  }
}

/** `hour`, `minute`, `second` of a timestamp under a fixed-offset session zone. */
final case class TimeFieldExpr(field: DateKernels.TimeField, child: VectorExpr, offsetMicros: Long) extends VectorExpr {
  override def dataType: DataType = IntegerType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT32, ctx.numRows)
    DateKernels.timeField(field, a, offsetMicros, data)
    SegmentVectorBuffers.fixedWidth(VecType.INT32, ctx.numRows, a.validity(), data)
  }
}

object DateExprs {

  /**
   * The fixed offset of a session zone in microseconds, or None when the zone has rules (DST or
   * historical changes), in which case timestamp functions fall back rather than guess.
   */
  def fixedOffsetMicros(zoneId: Option[String]): Option[Long] =
    zoneId.flatMap { id =>
      val rules = ZoneId.of(id).getRules
      if (rules.isFixedOffset) Some(rules.getOffset(java.time.Instant.EPOCH).getTotalSeconds * 1000000L) else None
    }

  /** Spark's `trunc` formats (case-insensitive) for the units the kernel implements. */
  def truncUnit(format: String): Option[DateKernels.TruncUnit] = format.toUpperCase(java.util.Locale.ROOT) match {
    case "YEAR" | "YYYY" | "YY" => Some(DateKernels.TruncUnit.YEAR)
    case "QUARTER" => Some(DateKernels.TruncUnit.QUARTER)
    case "MONTH" | "MON" | "MM" => Some(DateKernels.TruncUnit.MONTH)
    case "WEEK" => Some(DateKernels.TruncUnit.WEEK)
    case _ => None
  }
}
