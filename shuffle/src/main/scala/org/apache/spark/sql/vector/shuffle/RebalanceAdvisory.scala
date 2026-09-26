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
package org.apache.spark.sql.vector.shuffle

import org.apache.spark.sql.types._

/**
 * The advisory size of a rebalance exchange, on our shuffle's scale (#20).
 *
 * A data source that asks for a rebalance ahead of its write sizes the advisory partition size for
 * Spark's shuffle -- Iceberg multiplies its target file size by the compression it expects of
 * Spark's shuffle -- and AQE packs and splits the partitions to that size in the shuffle's bytes.
 * Our columnar batches take fewer bytes per row than Spark's `UnsafeRow`s (8 bytes per field, null
 * or not, where our lanes take their width and a validity bit), while both compress about as well
 * (measured on the CDC MERGE's write exchange: 0.180 ours, 0.178 Spark's). So the same advisory size
 * holds ~1.8x Spark's rows in ours, and a skewed partition was split into pieces of 6.2M rows where
 * Spark's were 3.9M -- the write, row by row, is as long as its largest piece.
 *
 * [[scale]] maps the size to ours by the ratio of our uncompressed bytes per row to an estimate of
 * Spark's `UnsafeRow` bytes for the same rows: the schema's fixed part is exact, the variable part
 * (string bytes) is taken from ours.
 */
object RebalanceAdvisory {

  /** Smallest factor applied, so a mis-estimate cannot shrink the size to nothing. */
  val MinFactor = 1.0 / 16

  /** An `UnsafeRow`'s null bit set and fixed-width words for `fields`. */
  private def unsafeFixed(fields: Seq[DataType]): Double = ((fields.size + 63) / 64) * 8.0 + 8.0 * fields.size

  /**
   * Bytes of Spark's `UnsafeRow` beyond ours for one value of `dt`, other than the variable data both
   * carry: a string's data is padded to a word (4 bytes on average); a wide decimal's 16 bytes live
   * past the fixed region; a struct is a nested row.
   */
  private def unsafeExtra(dt: DataType): Double = dt match {
    case s: StructType => unsafeFixed(s.fields.map(_.dataType).toSeq) + s.fields.map(f => unsafeExtra(f.dataType)).sum
    case _: StringType | _: BinaryType => 4.0
    case d: DecimalType if d.precision > Decimal.MAX_LONG_DIGITS => 16.0
    case _ => 0.0
  }

  /** Our fixed bytes per row for one column: its validity bit and its lane (strings: the offset). */
  private def arrowFixed(dt: DataType): Double = 1.0 / 8 + (dt match {
    case s: StructType => s.fields.map(f => arrowFixed(f.dataType)).sum
    case BooleanType => 1.0 / 8
    case LongType | TimestampType | DoubleType => 8.0
    case d: DecimalType => if (d.precision > Decimal.MAX_LONG_DIGITS) 16.0 else 8.0
    case _: StringType | _: BinaryType => 4.0
    case _ => 4.0 // INT32 lanes: int, date, byte, short
  })

  /** Estimated `UnsafeRow` bytes per row for `schema`, given our uncompressed bytes per row. */
  def unsafeRowBytes(schema: Seq[DataType], ourBytesPerRow: Double): Double = {
    val variable = math.max(0.0, ourBytesPerRow - schema.map(arrowFixed).sum)
    unsafeFixed(schema) + schema.map(unsafeExtra).sum + variable
  }

  /**
   * `size` on our shuffle's scale, given the exchange's written `rows` and uncompressed `bytes`
   * (its `dataSize`). The size is never raised, and never cut below [[MinFactor]]; with no rows the
   * size is returned as is.
   */
  def scale(size: Long, schema: Seq[DataType], rows: Long, bytes: Long): Long = {
    if (rows <= 0 || bytes <= 0 || size <= 0) return size
    val ours = bytes.toDouble / rows
    val factor = math.min(1.0, math.max(MinFactor, ours / unsafeRowBytes(schema, ours)))
    math.max(1L, math.round(size * factor))
  }
}
