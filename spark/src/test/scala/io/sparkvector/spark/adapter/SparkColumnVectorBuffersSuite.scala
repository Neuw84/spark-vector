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
package io.sparkvector.spark.adapter

import java.lang.foreign.Arena
import io.sparkvector.kernels.{Bitmap, VecType}
import io.sparkvector.spark.test.SparkVectorFunSuite
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.execution.vectorized.{OffHeapColumnVector, OnHeapColumnVector, WritableColumnVector}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

class SparkColumnVectorBuffersSuite extends SparkVectorFunSuite {

  private val n = 101

  private def fill(cv: WritableColumnVector, dt: DataType, withNulls: Boolean): Unit = {
    (0 until n).foreach { i =>
      if (withNulls && i % 7 == 3) cv.putNull(i)
      else dt match {
        case IntegerType | DateType => cv.putInt(i, i * 3 - 50)
        case LongType => cv.putLong(i, i.toLong * 123456789L)
        case DoubleType => cv.putDouble(i, i / 3.0)
        case BooleanType => cv.putBoolean(i, i % 2 == 0)
        case StringType => cv.putByteArray(i, s"s$i".getBytes("UTF-8"))
        case d: DecimalType if d.precision <= 9 => cv.putInt(i, i * 7 - 300) // Spark keeps small decimals as ints
        case _: DecimalType => cv.putLong(i, i.toLong * 98765432101L - 5)
        case other => fail(s"unexpected $other")
      }
    }
  }

  private def check(cv: WritableColumnVector, dt: DataType, withNulls: Boolean): Unit = {
    val arena = Arena.ofConfined()
    try {
      val vb = SparkColumnVectorBuffers.copy(cv, n, arena)
      assert(vb.`type`() === TypeMapping.vecTypeOf(dt))
      assert(vb.length() === n)
      assert(vb.hasNulls === withNulls)
      (0 until n).foreach { i =>
        assert(vb.isNull(i) === cv.isNullAt(i), s"$dt row $i null flag")
        if (!cv.isNullAt(i)) dt match {
          case IntegerType | DateType => assert(vb.getInt(i) === cv.getInt(i))
          case LongType => assert(vb.getLong(i) === cv.getLong(i))
          case DoubleType => assert(vb.getDouble(i) === cv.getDouble(i))
          case BooleanType => assert(vb.getBoolean(i) === cv.getBoolean(i))
          case StringType => assert(vb.getString(i) === cv.getUTF8String(i).toString)
          case d: DecimalType => assert(vb.getLong(i) === cv.getDecimal(i, d.precision, d.scale).toUnscaledLong)
          case _ =>
        }
      }
    } finally arena.close()
  }

  private val types =
    Seq(IntegerType, DateType, LongType, DoubleType, BooleanType, StringType, DecimalType(7, 2), DecimalType(15, 3))

  test("OnHeapColumnVector copies into Arrow layout, with and without nulls") {
    for (dt <- types; withNulls <- Seq(true, false)) {
      val cv = new OnHeapColumnVector(n, dt)
      try { fill(cv, dt, withNulls); check(cv, dt, withNulls) }
      finally cv.close()
    }
  }

  test("OffHeapColumnVector copies into Arrow layout, with and without nulls") {
    for (dt <- types; withNulls <- Seq(true, false)) {
      val cv = new OffHeapColumnVector(n, dt)
      try { fill(cv, dt, withNulls); check(cv, dt, withNulls) }
      finally cv.close()
    }
  }

  test("dictionary-encoded string vectors copy as indices plus the referenced dictionary values") {
    val values = Array("A", "N", "R", "unused")
    val dictionary = new org.apache.spark.sql.execution.vectorized.Dictionary {
      override def decodeToInt(id: Int): Int = fail("not an int dictionary")
      override def decodeToLong(id: Int): Long = fail("not a long dictionary")
      override def decodeToFloat(id: Int): Float = fail("not a float dictionary")
      override def decodeToDouble(id: Int): Double = fail("not a double dictionary")
      override def decodeToBinary(id: Int): Array[Byte] = values(id).getBytes("UTF-8")
    }
    for (withNulls <- Seq(true, false)) {
      val cv = new OnHeapColumnVector(n, StringType)
      try {
        cv.setDictionary(dictionary)
        val ids = cv.reserveDictionaryIds(n)
        (0 until n).foreach { i =>
          if (withNulls && i % 7 == 3) cv.putNull(i) else ids.putInt(i, i % 3)
        }
        val arena = Arena.ofConfined()
        try {
          val vb = SparkColumnVectorBuffers.copy(cv, n, arena)
          assert(vb.isDictionaryEncoded)
          assert(vb.dictionary().length() === 3, "only referenced values are kept")
          assert(vb.hasNulls === withNulls)
          (0 until n).foreach { i =>
            assert(vb.isNull(i) === cv.isNullAt(i))
            if (!cv.isNullAt(i)) assert(vb.getString(i) === cv.getUTF8String(i).toString)
          }
        } finally arena.close()
      } finally cv.close()
    }
  }

  test("dictionary-encoded numeric vectors decode to plain values on both heap kinds (#398, #403)") {
    // Ints, longs and doubles through decodeDictionaryInto's bulk path: the on-heap vector's own id
    // and null arrays, or the off-heap vector's native ones copied out once. Values are compared
    // with the vector's own decoding reads.
    val dictionary = new org.apache.spark.sql.execution.vectorized.Dictionary {
      override def decodeToInt(id: Int): Int = id * 11 - 7
      override def decodeToLong(id: Int): Long = id.toLong * 9876543210L - 3
      override def decodeToFloat(id: Int): Float = fail("not a float dictionary")
      override def decodeToDouble(id: Int): Double = id / 4.0 + 0.5
      override def decodeToBinary(id: Int): Array[Byte] = fail("not a binary dictionary")
    }
    for (dt <- Seq(IntegerType, LongType, DoubleType); offHeap <- Seq(false, true); withNulls <- Seq(true, false)) {
      val cv: WritableColumnVector = if (offHeap) new OffHeapColumnVector(n, dt) else new OnHeapColumnVector(n, dt)
      try {
        cv.setDictionary(dictionary)
        val ids = cv.reserveDictionaryIds(n)
        (0 until n).foreach { i =>
          if (withNulls && i % 7 == 3) cv.putNull(i) else ids.putInt(i, i % 5)
        }
        val arena = Arena.ofConfined()
        try {
          val vb = SparkColumnVectorBuffers.copy(cv, n, arena)
          assert(!vb.isDictionaryEncoded, s"$dt offHeap=$offHeap")
          assert(vb.hasNulls === withNulls, s"$dt offHeap=$offHeap")
          (0 until n).foreach { i =>
            assert(vb.isNull(i) === cv.isNullAt(i), s"$dt offHeap=$offHeap row $i")
            if (!cv.isNullAt(i)) dt match {
              case IntegerType => assert(vb.getInt(i) === cv.getInt(i), s"offHeap=$offHeap row $i")
              case LongType => assert(vb.getLong(i) === cv.getLong(i), s"offHeap=$offHeap row $i")
              case _ => assert(vb.getDouble(i) === cv.getDouble(i), s"offHeap=$offHeap row $i")
            }
          }
        } finally arena.close()
      } finally cv.close()
    }
  }

  test("unsupported Spark types are rejected") {
    val cv = new OnHeapColumnVector(4, FloatType)
    try {
      assert(!TypeMapping.isSupported(FloatType))
      val arena = Arena.ofConfined()
      try intercept[UnsupportedOperationException](SparkColumnVectorBuffers.copy(cv, 4, arena))
      finally arena.close()
    } finally cv.close()
  }

  test("batches from Spark's vectorized Parquet scan adapt column by column") {
    val path = newTempPath("adapter/parquet")
    spark
      .range(0, 5000)
      .selectExpr(
        "cast(id as int) as i",
        "if(id % 4 = 0, null, id) as l",
        "cast(id as double) / 7 as d",
        "date_add(date '2020-01-01', cast(id % 365 as int)) as dt",
        "id % 2 = 0 as b",
        "if(id % 10 = 0, null, concat('s', id)) as s"
      )
      .write
      .parquet(path)

    val df = spark.read.parquet(path)
    val scan = df.queryExecution.executedPlan.collect { case s: FileSourceScanExec => s }.head
    assert(scan.supportsColumnar)

    // Adapt inside the task (batches are not serializable) and ship back small summaries.
    val summaries = scan
      .executeColumnar()
      .mapPartitions { batches =>
        batches.flatMap { batch =>
          val arena = Arena.ofConfined()
          try {
            (0 until batch.numCols()).map { c =>
              val cv = batch.column(c)
              val vb = ColumnVectorAdapters.adapt(cv, batch.numRows(), arena)
              val validCount = if (vb.validity() == null) vb.length() else Bitmap.popcount(vb.validity(), vb.length())
              val checksum = vb.`type`() match {
                case VecType.INT32 => (0 until vb.length()).filterNot(vb.isNull).map(vb.getInt(_).toLong).sum
                case VecType.INT64 => (0 until vb.length()).filterNot(vb.isNull).map(vb.getLong).sum
                case VecType.FLOAT64 => (0 until vb.length()).filterNot(vb.isNull).map(vb.getDouble).sum.round
                case VecType.BOOL => (0 until vb.length()).filterNot(vb.isNull).count(vb.getBoolean).toLong
                case VecType.UTF8 => (0 until vb.length()).filterNot(vb.isNull).map(vb.getString(_).length.toLong).sum
              }
              (c, vb.`type`().name(), vb.length(), validCount, vb.data().byteSize(), checksum)
            }
          } finally arena.close()
        }
      }
      .collect()

    val byCol = summaries.groupBy(_._1).view.mapValues(_.toSeq).toMap
    assert(byCol.keySet === (0 until 6).toSet)
    val rows = byCol.map { case (_, s) => s.map(_._3).sum }
    assert(rows.forall(_ === 5000L), s"every column should see 5000 rows: $rows")

    // Expected values computed by Spark itself.
    val expected = df
      .selectExpr(
        "sum(i)",
        "count(i)",
        "sum(l)",
        "count(l)",
        "round(sum(d))",
        "count(d)",
        "sum(datediff(dt, date '1970-01-01'))",
        "count(dt)",
        "sum(if(b, 1, 0))",
        "count(b)",
        "sum(length(s))",
        "count(s)"
      )
      .collect()
      .head
    def sumOf(c: Int) = byCol(c).map(_._6).sum
    def validOf(c: Int) = byCol(c).map(_._4.toLong).sum
    assert(byCol(0).head._2 === "INT32"); assert(sumOf(0) === expected.getLong(0));
    assert(validOf(0) === expected.getLong(1))
    assert(byCol(1).head._2 === "INT64"); assert(sumOf(1) === expected.getLong(2));
    assert(validOf(1) === expected.getLong(3))
    assert(byCol(2).head._2 === "FLOAT64"); assert(math.abs(sumOf(2) - expected.getDouble(4).round) <= 1);
    assert(validOf(2) === expected.getLong(5))
    assert(byCol(3).head._2 === "INT32"); assert(sumOf(3) === expected.getLong(6));
    assert(validOf(3) === expected.getLong(7))
    assert(byCol(4).head._2 === "BOOL"); assert(sumOf(4) === expected.getLong(8));
    assert(validOf(4) === expected.getLong(9))
    assert(byCol(5).head._2 === "UTF8"); assert(sumOf(5) === expected.getLong(10));
    assert(validOf(5) === expected.getLong(11))

    info(summaries.sortBy(s => (s._1, -s._3)).take(6).map { case (c, t, len, valid, bytes, _) =>
      s"col $c $t rows=$len valid=$valid dataBytes=$bytes"
    }.mkString("\n"))
  }
}
