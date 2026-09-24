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
package org.apache.spark.sql.vector

import java.time.ZoneId

import org.apache.spark.QueryContext
import org.apache.spark.sql.catalyst.expressions.Cast
import org.apache.spark.sql.catalyst.util.{DateFormatter, DateTimeUtils, TimestampFormatter, UTF8StringUtils}
import org.apache.spark.sql.errors.QueryExecutionErrors
import org.apache.spark.sql.types.DataType
import org.apache.spark.unsafe.types.UTF8String

/** Exactly the helpers Spark's `Cast` calls for the string casts, so our per-row results are Spark's. */
object SparkCasts {

  /** ANSI `string -> int / bigint`: Spark's exact parsers, raising CAST_INVALID_INPUT themselves. */
  def toIntExact(s: UTF8String, context: QueryContext): Int = UTF8StringUtils.toIntExact(s, context)
  def toLongExact(s: UTF8String, context: QueryContext): Long = UTF8StringUtils.toLongExact(s, context)

  /** `NaN`, `Infinity`, `inf`, signed forms, any case, trimmed -- Spark's fallback after `parseDouble` fails; null otherwise. */
  def specialDouble(str: String): java.lang.Double =
    Cast.processFloatingPointSpecialLiterals(str, false).asInstanceOf[java.lang.Double]

  def invalidNumber(to: DataType, s: UTF8String, context: QueryContext): RuntimeException =
    QueryExecutionErrors.invalidInputInCastToNumberError(to, s, context)

  /** The formatters `castToString` uses for dates and timestamps. */
  def dateFormatter(): DateFormatter = DateFormatter()
  def fractionFormatter(zone: ZoneId): TimestampFormatter = TimestampFormatter.getFractionFormatter(zone)

  /** Spark's own string parsers for dates and timestamps, legacy (`None` on failure) and ANSI (raising). */
  def stringToDate(s: UTF8String): Option[Int] = DateTimeUtils.stringToDate(s)
  def stringToDateAnsi(s: UTF8String, context: QueryContext): Int = DateTimeUtils.stringToDateAnsi(s, context)
  def stringToTimestamp(s: UTF8String, zone: ZoneId): Option[Long] = DateTimeUtils.stringToTimestamp(s, zone)
  def stringToTimestampAnsi(s: UTF8String, zone: ZoneId, context: QueryContext): Long =
    DateTimeUtils.stringToTimestampAnsi(s, zone, context)
}
