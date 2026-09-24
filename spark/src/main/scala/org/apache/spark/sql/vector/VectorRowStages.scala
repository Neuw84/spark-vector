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

import scala.jdk.CollectionConverters._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans.physical.SinglePartition
import org.apache.spark.sql.execution.{ShuffledRowRDD, UnsafeRowSerializer}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.types.{
  BooleanType,
  ByteType,
  DateType,
  DecimalType,
  DoubleType,
  IntegerType,
  LongType,
  ShortType,
  StringType,
  StructType,
  TimestampType
}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * The row-based tail shared by the operators whose final step is a bounded merge of a few rows per
 * partition (`ORDER BY ... LIMIT`, `LIMIT`): batches become `UnsafeRow`s for Spark's own
 * single-partition shuffle, and the handful of surviving rows become one columnar batch again for
 * the `ColumnarToRowExec` above. Everything here sees at most `limit x partitions` rows.
 */
object VectorRowStages {

  /**
   * Every row of every batch as an `UnsafeRow` copy. A batch is released when the next is requested,
   * so the rows must be copied out of it.
   */
  def toUnsafeRows(batches: Iterator[ColumnarBatch], attrs: Seq[Attribute]): Iterator[InternalRow] = {
    val toUnsafe = UnsafeProjection.create(attrs, attrs)
    batches.flatMap(batch => batch.rowIterator().asScala.map(row => toUnsafe(row).copy()))
  }

  /** `rows` shuffled into one partition through Spark's exchange machinery. */
  def singlePartition(
      rows: RDD[InternalRow],
      attrs: Seq[Attribute],
      writeMetrics: Map[String, SQLMetric],
      readMetrics: Map[String, SQLMetric]
  ): RDD[InternalRow] =
    new ShuffledRowRDD(
      ShuffleExchangeExec.prepareShuffleDependency(
        rows,
        attrs,
        SinglePartition,
        new UnsafeRowSerializer(attrs.size),
        writeMetrics
      ),
      readMetrics
    )

  /**
   * One columnar batch of on-heap vectors holding `rows`. Covers exactly the Spark types
   * `TypeMapping` supports (the planners refuse any other output type), so every case is a plain
   * `WritableColumnVector` put.
   */
  def toBatch(schema: StructType, rows: Array[InternalRow]): ColumnarBatch = {
    val n = rows.length
    val vectors: Array[OnHeapColumnVector] = OnHeapColumnVector.allocateColumns(n, schema)
    var c = 0
    while (c < vectors.length) {
      val v = vectors(c)
      val dt = schema(c).dataType
      var i = 0
      while (i < n) {
        val row = rows(i)
        if (row.isNullAt(c)) v.putNull(i)
        else dt match {
          case BooleanType => v.putBoolean(i, row.getBoolean(c))
          case IntegerType | DateType => v.putInt(i, row.getInt(c))
          case ByteType => v.putByte(i, row.getByte(c))
          case ShortType => v.putShort(i, row.getShort(c))
          case LongType | TimestampType => v.putLong(i, row.getLong(c))
          case DoubleType => v.putDouble(i, row.getDouble(c))
          case d: DecimalType => v.putDecimal(i, row.getDecimal(c, d.precision, d.scale), d.precision)
          case StringType => val b = row.getUTF8String(c).getBytes; v.putByteArray(i, b, 0, b.length)
          case other => throw new IllegalStateException(s"unsupported output type ${other.simpleString}")
        }
        i += 1
      }
      c += 1
    }
    new ColumnarBatch(vectors.map(v => v: ColumnVector), n)
  }
}
