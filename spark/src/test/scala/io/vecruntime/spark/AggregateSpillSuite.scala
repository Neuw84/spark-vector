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
package io.vecruntime.spark

import java.lang.foreign.Arena

import io.vecruntime.kernels.{ArrowLayout, Bitmap, PartitionKernels}
import io.vecruntime.spark.arrow.{ArrowOutput, VectorAllocators}
import org.apache.spark.sql.types.{IntegerType, LongType}
import org.apache.spark.sql.vector.AggregateSpill
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}
import org.scalatest.funsuite.AnyFunSuite

/**
 * The aggregate's grace-hash buckets (#363) over the rows one reduce task actually holds (#416): every
 * row of a task agrees on Spark's hash of the keys modulo the shuffle's partition count, so bucketing
 * by the same hash spreads the rows over `buckets / gcd(partitions, buckets)` buckets only. With its own
 * seed the spill fills every bucket.
 */
class AggregateSpillSuite extends AnyFunSuite {

  /** Keys a reduce task at `partitions` partitions receives: ints whose Spark hash lands in partition 7. */
  private def taskKeys(n: Int, partitions: Int): Array[Int] = {
    val out = new Array[Int](n)
    var i = 0; var v = 1
    while (i < n) {
      if (PartitionKernels.pmod(PartitionKernels.hashInt(v, PartitionKernels.SPARK_SEED), partitions) == 7) {
        out(i) = v; i += 1
      }
      v += 1
    }
    out
  }

  private def bucketsFilled(seed: Int, partitions: Int, buckets: Int): Int = {
    val allocator = VectorAllocators.newChild("AggregateSpillSuite")
    val arena = Arena.ofConfined()
    val columns = Array(
      ("k", IntegerType: org.apache.spark.sql.types.DataType),
      ("sum", LongType: org.apache.spark.sql.types.DataType)
    )
    val spill = new AggregateSpill(buckets, columns, Array(0), allocator, seed)
    try {
      val n = 8192
      val keys = taskKeys(n, partitions)
      val all = arena.allocate(Bitmap.bytesFor(n), 8)
      Bitmap.fill(all, n, true)
      val cols: Array[ColumnVector] = Array(
        ArrowOutput.compact("k", IntegerType, ArrowLayout.ofInts(arena, keys, Array.fill(n)(false)), all, n, allocator),
        ArrowOutput.compact(
          "sum",
          LongType,
          ArrowLayout.ofLongs(arena, keys.map(_.toLong), Array.fill(n)(false)),
          all,
          n,
          allocator
        )
      )
      val batch = new ColumnarBatch(cols, n)
      try spill.write(batch)
      finally batch.close()
      var filled = 0
      var rows = 0L
      (0 until buckets).foreach { b =>
        val it = spill.read(b)
        try {
          var any = false
          while (it.hasNext) { rows += it.next().numRows(); any = true }
          if (any) filled += 1
        } finally it.close()
      }
      assert(rows === n)
      filled
    } finally { spill.close(); arena.close(); allocator.close() }
  }

  test("#416: under the shuffle's seed a task's rows fill buckets / gcd(partitions, buckets) buckets") {
    assert(bucketsFilled(PartitionKernels.SPARK_SEED, 1000, 16) === 2)
    assert(bucketsFilled(PartitionKernels.SPARK_SEED, 300, 16) === 4)
    assert(bucketsFilled(PartitionKernels.SPARK_SEED, 200, 16) === 2)
  }

  test("#416: under the spill's own seed a task's rows fill every bucket") {
    Seq(1000, 300, 200).foreach { p =>
      assert(bucketsFilled(AggregateSpill.BucketSeed, p, 16) === 16, s"$p partitions")
    }
  }
}
