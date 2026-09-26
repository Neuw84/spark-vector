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
package io.sparkvector.spark.shuffle

import java.lang.foreign.Arena
import java.math.BigInteger
import scala.util.Random

import io.sparkvector.kernels.{ArrowLayout, Bitmap, PartitionKernels, SegmentVectorBuffers, VectorBuffers}
import io.sparkvector.kernels.PartitionKernels.KeyKind
import org.apache.spark.sql.catalyst.expressions.Murmur3HashFunction
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
import org.scalatest.funsuite.AnyFunSuite

/**
 * #288: the shuffle's hash partitioning must be bit-identical to Spark's
 * `Pmod(Murmur3Hash(keys, 42), n)`, else AQE's exchange reuse and Spark's co-partitioning
 * assumptions break silently. Property test: random rows of every lane type, with nulls, with
 * dictionary-encoded strings, in several key orders, against Spark's own `Murmur3HashFunction`.
 */
class PartitionKernelsSuite extends AnyFunSuite {

  private val rnd = new Random(288)

  private def nulls(n: Int, fraction: Double): Array[Boolean] = Array.fill(n)(rnd.nextDouble() < fraction)

  private def sparkHash(value: Any, dt: DataType, seed: Int): Int =
    if (value == null) seed else Murmur3HashFunction.hash(value, dt, seed).toInt

  /** A key column: our buffers, Spark's values and type, our kind. */
  private case class Key(buffers: VectorBuffers, values: Array[Any], dt: DataType, kind: KeyKind)

  private def intKey(arena: Arena, n: Int, dt: DataType): Key = {
    val nl = nulls(n, 0.2)
    val v = Array.fill(n)(rnd.nextInt())
    Key(
      ArrowLayout.ofInts(arena, v, nl),
      v.indices.map(i => if (nl(i)) null else Int.box(v(i))).toArray,
      dt,
      KeyKind.INT
    )
  }

  private def longKey(arena: Arena, n: Int, dt: DataType, toSpark: Long => Any, bound: Long = Long.MaxValue): Key = {
    val nl = nulls(n, 0.2)
    val v = Array.fill(n)(if (rnd.nextBoolean()) rnd.nextLong() % bound else rnd.nextInt(1000).toLong - 500)
    Key(
      ArrowLayout.ofLongs(arena, v, nl),
      v.indices.map(i => if (nl(i)) null else toSpark(v(i))).toArray,
      dt,
      KeyKind.LONG
    )
  }

  private def doubleKey(arena: Arena, n: Int): Key = {
    val nl = nulls(n, 0.2)
    val specials = Array(
      0.0,
      -0.0,
      Double.NaN,
      java.lang.Double.longBitsToDouble(0x7ff8000000000123L),
      Double.PositiveInfinity,
      -1.5
    )
    val v = Array.fill(n)(if (rnd.nextInt(4) == 0) specials(rnd.nextInt(specials.length)) else rnd.nextGaussian() * 1e6)
    Key(
      ArrowLayout.ofDoubles(arena, v, nl),
      v.indices.map(i => if (nl(i)) null else Double.box(v(i))).toArray,
      DoubleType,
      KeyKind.DOUBLE
    )
  }

  private def boolKey(arena: Arena, n: Int): Key = {
    val nl = nulls(n, 0.2)
    val v = Array.fill(n)(rnd.nextBoolean())
    Key(
      ArrowLayout.ofBooleans(arena, v, nl),
      v.indices.map(i => if (nl(i)) null else Boolean.box(v(i))).toArray,
      BooleanType,
      KeyKind.BOOL
    )
  }

  private def randomString(): String = {
    val len = rnd.nextInt(12)
    val sb = new StringBuilder
    for (_ <- 0 until len) {
      sb.append(rnd.nextInt(5) match {
        case 0 => ('a' + rnd.nextInt(26)).toChar
        case 1 => "é"
        case 2 => "€"
        case 3 => "😀"
        case _ => ('A' + rnd.nextInt(26)).toChar
      })
    }
    sb.toString
  }

  private def stringKey(arena: Arena, n: Int): Key = {
    val v = Array.fill[String](n)(if (rnd.nextDouble() < 0.2) null else randomString())
    Key(
      ArrowLayout.ofStrings(arena, v),
      v.map(s => if (s == null) null else UTF8String.fromString(s)),
      StringType,
      KeyKind.UTF8
    )
  }

  private def dictionaryStringKey(arena: Arena, n: Int): Key = {
    val dict = Array.fill(7)(randomString())
    val nl = nulls(n, 0.2)
    val ids = Array.fill(n)(rnd.nextInt(dict.length))
    val dictBuffers = ArrowLayout.ofStrings(arena, dict)
    val idBuffers = ArrowLayout.ofInts(arena, ids, nl)
    val buffers = SegmentVectorBuffers.dictionaryUtf8(n, idBuffers.validity(), idBuffers.data(), dictBuffers)
    Key(
      buffers,
      ids.indices.map(i => if (nl(i)) null else UTF8String.fromString(dict(ids(i)))).toArray,
      StringType,
      KeyKind.UTF8
    )
  }

  private def wideDecimalKey(arena: Arena, n: Int): Key = {
    val dt = DecimalType(30, 4)
    val nl = nulls(n, 0.2)
    val v = Array.fill(n) {
      val bits = rnd.nextInt(100)
      val mag = new BigInteger(bits, rnd.self)
      if (rnd.nextBoolean()) mag.negate() else mag
    }
    val spark = v.indices.map(i =>
      if (nl(i)) null else org.apache.spark.sql.types.Decimal(new java.math.BigDecimal(v(i), 4), 30, 4)
    ).toArray[Any]
    Key(ArrowLayout.ofDecimal128(arena, v, nl), spark, dt, KeyKind.DECIMAL128)
  }

  private def check(keys: Seq[Key], numPartitions: Int): Unit = {
    val n = keys.head.values.length
    val hashes = new Array[Int](n)
    val ids = new Array[Int](n)
    PartitionKernels.hashPartitionIds(
      keys.map(_.buffers).toArray,
      keys.map(_.kind).toArray,
      n,
      numPartitions,
      hashes,
      ids
    )
    var i = 0
    while (i < n) {
      var h = PartitionKernels.SPARK_SEED
      keys.foreach(k => h = sparkHash(k.values(i), k.dt, h))
      val expected = { val r = h % numPartitions; if (r < 0) r + numPartitions else r }
      assert(hashes(i) === h, s"row $i hash, keys ${keys.map(_.dt).mkString(",")}")
      assert(ids(i) === expected, s"row $i partition")
      i += 1
    }
  }

  test("hash partition ids equal Spark's Pmod(Murmur3Hash) for every lane type, nulls included") {
    val n = 2000
    val arena = Arena.ofConfined()
    try {
      val all = Seq(
        intKey(arena, n, IntegerType),
        intKey(arena, n, DateType),
        longKey(arena, n, LongType, identity),
        longKey(arena, n, TimestampType, identity),
        longKey(
          arena,
          n,
          DecimalType(18, 3),
          v => org.apache.spark.sql.types.Decimal(v, 18, 3),
          bound = 1000000000000000000L
        ),
        doubleKey(arena, n),
        boolKey(arena, n),
        stringKey(arena, n),
        dictionaryStringKey(arena, n),
        wideDecimalKey(arena, n)
      )
      all.foreach(k => check(Seq(k), 8))
      check(all, 200)
      check(all.reverse, 7)
      check(Seq(all(7), all(2), all(0)), 1)
      check(Seq(all(8), all(5)), 1023)
    } finally arena.close()
  }

  test("round-robin ids continue across batches and wrap") {
    val ids = new Array[Int](10)
    val next = PartitionKernels.roundRobinIds(10, 4, 2, ids)
    assert(ids.toSeq === Seq(2, 3, 0, 1, 2, 3, 0, 1, 2, 3))
    assert(next === 0)
  }

  test("partition masks split the rows by id") {
    val arena = Arena.ofConfined()
    try {
      val n = 100
      val ids = Array.tabulate(n)(i => (i * 7) % 3)
      val masks = Array.fill(3)(arena.allocate(Bitmap.bytesFor(n), 8))
      val counts = new Array[Int](3)
      PartitionKernels.partitionMasks(ids, n, masks, counts)
      assert(counts.sum === n)
      for (i <- 0 until n; p <- 0 until 3) {
        assert(Bitmap.isSet(masks(p), i) === (ids(i) == p))
      }
    } finally arena.close()
  }
}
