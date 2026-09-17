package org.apache.spark.sql.vector

import java.time.ZoneId

import org.apache.spark.sql.catalyst.util.{DateTimeUtils, LegacyDateFormats, TimestampFormatter}

/** Spark's own datetime formatter, built exactly as `DateFormatClass` / `FromUnixTime` build theirs. */
object SparkFormatters {
  def zoneId(timeZoneId: String): ZoneId = DateTimeUtils.getZoneId(timeZoneId)

  def timestampFormatter(pattern: String, zone: ZoneId): TimestampFormatter =
    TimestampFormatter(pattern, zone, LegacyDateFormats.SIMPLE_DATE_FORMAT, isParsing = false)
}
