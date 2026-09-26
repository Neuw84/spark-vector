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
package org.apache.spark.sql.vecruntime

import java.time.ZoneId

import org.apache.spark.sql.catalyst.util.{DateTimeUtils, LegacyDateFormats, TimestampFormatter}

/** Spark's own datetime formatter, built exactly as `DateFormatClass` / `FromUnixTime` build theirs. */
object SparkFormatters {
  def zoneId(timeZoneId: String): ZoneId = DateTimeUtils.getZoneId(timeZoneId)

  def timestampFormatter(pattern: String, zone: ZoneId): TimestampFormatter =
    TimestampFormatter(pattern, zone, LegacyDateFormats.SIMPLE_DATE_FORMAT, isParsing = false)
}
