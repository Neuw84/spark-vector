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
   * Estimated `UnsafeRow` bytes per row for `schema` when the string values' padded bytes per row are
   * known ([[unsafeStringBytes]]): the fixed part and the non-string extras from the schema, the
   * strings as measured. Dictionary-encoded columns keep their strings out of our `dataSize`, so the
   * estimate from our bytes alone misses them (the CDC MERGE's `_file` path: 236 estimated against
   * 304 bytes per row in Spark's own shuffle).
   */
  def unsafeRowBytesWithStrings(schema: Seq[DataType], stringBytesPerRow: Double): Double =
    unsafeFixed(schema) + schema.map(unsafeExtraNonString).sum + stringBytesPerRow

  /** [[unsafeExtra]] without the strings' padding, which [[unsafeStringBytes]] counts exactly. */
  private def unsafeExtraNonString(dt: DataType): Double = dt match {
    case s: StructType =>
      unsafeFixed(s.fields.map(_.dataType).toSeq) + s.fields.map(f => unsafeExtraNonString(f.dataType)).sum
    case _: StringType | _: BinaryType => 0.0
    case other => unsafeExtra(other)
  }

  /**
   * The bytes `UnsafeRow` gives the string values of these written columns over `n` rows: each non-null
   * value's UTF-8 bytes rounded up to a word, dictionary-encoded columns resolved through their
   * dictionary. Non-string columns count nothing.
   */
  def unsafeStringBytes(columns: Array[io.sparkvector.kernels.VectorBuffers], n: Int): Long = {
    import io.sparkvector.kernels.{Bitmap, VecType, VectorBuffers}
    var total = 0L
    var c = 0
    while (c < columns.length) {
      val b = columns(c)
      if (b != null && b.`type`() == VecType.UTF8) {
        val validity = b.validity()
        val dict = b.dictionary()
        val off = if (dict != null) dict.offsets() else b.offsets()
        val ids = if (dict != null) b.data() else null
        var i = 0
        while (i < n) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            val e = if (ids != null) ids.get(VectorBuffers.LE_INT, i.toLong << 2) else i
            val len = off.get(VectorBuffers.LE_INT, (e + 1).toLong << 2) - off.get(VectorBuffers.LE_INT, e.toLong << 2)
            total += (len + 7) & ~7
          }
          i += 1
        }
      }
      c += 1
    }
    total
  }

  /**
   * `size` on our shuffle's scale, given the exchange's written `rows` and uncompressed `bytes`
   * (its `dataSize`). The size is never raised, and never cut below [[MinFactor]]; with no rows the
   * size is returned as is.
   */
  def scale(size: Long, schema: Seq[DataType], rows: Long, bytes: Long, stringBytes: Option[Long] = None): Long = {
    if (rows <= 0 || bytes <= 0 || size <= 0) return size
    val ours = bytes.toDouble / rows
    val spark = stringBytes match {
      case Some(s) if s >= 0 => unsafeRowBytesWithStrings(schema, s.toDouble / rows)
      case _ => unsafeRowBytes(schema, ours)
    }
    val factor = math.min(1.0, math.max(MinFactor, ours / spark))
    math.max(1L, math.round(size * factor))
  }

  /** Bounds of [[mapSizeFactor]], so a mis-estimate cannot blow the sizes up or shrink them to nothing. */
  val MaxMapSizeFactor = 16.0

  /**
   * The factor that puts an exchange's map output sizes -- our on-disk bytes, which AQE packs, splits
   * and tests for skew against sizes set for Spark's shuffle -- on Spark's scale (#511): Spark's
   * estimated on-disk bytes per row over ours. Spark's are the estimated `UnsafeRow` bytes per row
   * ([[unsafeRowBytes]], or [[unsafeRowBytesWithStrings]] with measured string bytes) divided by
   * `sparkCompression`, the compression expected of Spark's shuffle (TPC-DS 1 TB, 2026-09-26: 2.57
   * over the whole run, 1.8-3.4 per query); `sparkCompression <= 0` assumes Spark's shuffle
   * compresses as well as ours, and takes the uncompressed ratio. `rows`/`bytes` are what the
   * exchange wrote (records and uncompressed `dataSize`), `onDisk` the sum of its map output sizes.
   * Bounded to [1/16, 16]; `1.0` when anything is missing.
   */
  def mapSizeFactor(
      schema: Seq[DataType],
      rows: Long,
      bytes: Long,
      onDisk: Long,
      sparkCompression: Double,
      stringBytes: Option[Long] = None
  ): Double = {
    if (rows <= 0 || bytes <= 0 || onDisk <= 0) return 1.0
    val oursRow = bytes.toDouble / rows
    val sparkRow = stringBytes match {
      case Some(s) if s >= 0 => unsafeRowBytesWithStrings(schema, s.toDouble / rows)
      case _ => unsafeRowBytes(schema, oursRow)
    }
    val raw =
      if (sparkCompression > 0) (sparkRow / sparkCompression) / (onDisk.toDouble / rows)
      else sparkRow / oursRow
    math.min(MaxMapSizeFactor, math.max(1.0 / MaxMapSizeFactor, raw))
  }

  /** `sizes` times `factor`, rounded; a non-empty partition stays non-empty. */
  def scaleSizes(sizes: Array[Long], factor: Double): Array[Long] =
    sizes.map(s => if (s <= 0) s else math.max(1L, math.round(s * factor)))
}
