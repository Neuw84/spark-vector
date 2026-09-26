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
package org.apache.spark.sql.vecruntime.shuffle

import org.apache.spark.{MapOutputTrackerMaster, SparkEnv}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.vecruntime.VectorShuffleExchangeExec
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** #20: the row-proportional partition sizes a rebalance exchange hands AQE. */
class RowProportionalSizesSuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private var tempDir: java.nio.file.Path = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[4]")
      .appName("RowProportionalSizesSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.plugins", "io.vecruntime.spark.VectorPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager")
      .config("spark.vector.shuffle.enabled", "true")
      .getOrCreate()
    tempDir = java.nio.file.Files.createTempDirectory("vecruntime-rowsizes")
    // Two kinds of rows: a constant string (tiny once dictionary-encoded) and a near-unique one, on
    // different keys so they land in different partitions under the rebalance.
    spark.sql(
      "select cast(id % 3 as int) g, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' v from range(0, 30000) " +
        "union all select cast(3 + id % 3 as int) g, cast(id * 7919 as string) || 'bbbbbbbbbbbbbbbb' v from range(0, 3000)"
    ).write.parquet(tempDir.resolve("mixed").toString)
    spark.read.parquet(tempDir.resolve("mixed").toString).createOrReplaceTempView("mixed")
    spark.range(0, 20000).selectExpr("cast(id % 97 as int) k").write.parquet(tempDir.resolve("keys").toString)
    spark.read.parquet(tempDir.resolve("keys").toString).createOrReplaceTempView("keys")
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()
  }

  private def nodes(plan: SparkPlan): Seq[SparkPlan] = plan +: (plan match {
    case q: QueryStageExec => nodes(q.plan)
    case p => p.children.flatMap(nodes)
  })

  /** The first of our exchanges in the executed plan of `sql`. */
  private def ourExchange(sql: String): VectorShuffleExchangeExec = {
    val df = spark.sql(sql)
    df.collect()
    val plan = df.queryExecution.executedPlan match {
      case a: AdaptiveSparkPlanExec => a.executedPlan
      case p => p
    }
    nodes(
      plan
    ).collectFirst { case e: VectorShuffleExchangeExec => e }.getOrElse(fail(s"no exchange of ours in\n$plan"))
  }

  private def aqeSizes(ex: VectorShuffleExchangeExec): Array[Long] =
    ex.mapOutputStatisticsFuture.value.get.get.bytesByPartitionId

  private def realSizes(ex: VectorShuffleExchangeExec): Array[Long] =
    SparkEnv.get.mapOutputTracker.asInstanceOf[MapOutputTrackerMaster].getStatistics(ex.shuffleDependency)
      .bytesByPartitionId

  private val rebalance = "select /*+ REBALANCE(g) */ g, v from mixed"

  test("a rebalance exchange hands AQE row-proportional sizes over the real total") {
    val ex = ourExchange(rebalance)
    assert(ex.shuffleOrigin.toString.startsWith("REBALANCE"), ex.shuffleOrigin.toString)
    val counts = ex.recordsByPartition.get.value
    val rows = Array.tabulate(ex.numPartitions)(p => (0 until ex.numMappers).map(m => counts.get(m)(p)).sum)
    assert(rows.sum === 33000L)
    val sized = aqeSizes(ex)
    val real = realSizes(ex)
    assert(!(sized sameElements real), s"not reweighted: ${sized.toSeq} vs ${real.toSeq}")
    assert(math.abs(sized.sum - real.sum) <= ex.numPartitions, s"total ${sized.sum} vs ${real.sum}")
    val perRow = real.sum.toDouble / rows.sum
    sized.indices.foreach { p =>
      val expected = if (rows(p) == 0) 0L else math.max(1L, math.round(rows(p) * perRow))
      assert(sized(p) === expected, s"partition $p")
    }
    assert(spark.sql(rebalance).count() === 33000L)
  }

  test("with the switch off a rebalance keeps the real sizes") {
    spark.conf.set(VectorShuffleExchangeExec.RebalanceRowSizingKey, "false")
    try {
      val ex = ourExchange(rebalance)
      assert(ex.recordsByPartition.isEmpty)
      assert(aqeSizes(ex) sameElements realSizes(ex))
    } finally spark.conf.unset(VectorShuffleExchangeExec.RebalanceRowSizingKey)
  }

  test("an aggregate's exchange (ENSURE_REQUIREMENTS) keeps the real sizes with AQE map-size scaling off") {
    // Row sizing is a rebalance's only; with #511's scaling (on by default) the aggregate's sizes are
    // multiplied instead -- RebalanceAdvisorySuite covers that.
    spark.conf.set(VectorShuffleExchangeExec.MapSizeScalingKey, "false")
    try {
      val ex = ourExchange("select k, count(*) c from keys group by k")
      assert(ex.recordsByPartition.isEmpty)
      assert(aqeSizes(ex) sameElements realSizes(ex))
    } finally spark.conf.unset(VectorShuffleExchangeExec.MapSizeScalingKey)
  }

  private def counts(byMap: (Int, Array[Long])*): java.util.Map[Integer, Array[Long]] = {
    val acc = new RecordsByPartitionAccumulator
    byMap.foreach(acc.add)
    acc.value
  }

  test("sizes follow rows, keep the total, and leave empty partitions empty") {
    // The cluster MERGE's shape in small: light rows at ~2.5 B, heavy rows at ~60 B.
    val bytes = Array(25L, 600L, 0L, 250L)
    val records = counts(0 -> Array(6L, 5L, 0L, 60L), 1 -> Array(4L, 5L, 0L, 40L))
    val sized = RowProportionalSizes.reweight(bytes, records, numMappers = 2).get
    assert(math.abs(sized.sum - bytes.sum) <= 2L, s"total ${sized.sum}")
    assert(sized(2) === 0L)
    // 10 : 10 : 0 : 100 rows.
    assert(sized(0) === sized(1))
    assert(math.abs(sized(3) - 10 * sized(0)) <= 10L, sized.toSeq.toString)
  }

  test("a map's missing or malformed counts keep the real sizes") {
    val bytes = Array(10L, 20L)
    assert(RowProportionalSizes.reweight(bytes, counts(0 -> Array(1L, 2L)), numMappers = 2).isEmpty)
    assert(RowProportionalSizes.reweight(bytes, counts(0 -> Array(1L, 2L), 1 -> Array(1L)), numMappers = 2).isEmpty)
    assert(RowProportionalSizes.reweight(bytes, counts(0 -> Array(0L, 0L)), numMappers = 1).isEmpty)
  }

  test("a partition with rows never sizes to zero") {
    val sized = RowProportionalSizes.reweight(Array(1L, 1000000L), counts(0 -> Array(1L, 10000000L)), 1).get
    assert(sized(0) === 1L)
  }

  test("a retried map's counts replace the first attempt's; merging keeps the last report per map") {
    val driver = new RecordsByPartitionAccumulator
    val first = new RecordsByPartitionAccumulator
    first.add(0 -> Array(5L, 5L))
    val retry = new RecordsByPartitionAccumulator
    retry.add(0 -> Array(5L, 5L))
    val other = new RecordsByPartitionAccumulator
    other.add(1 -> Array(1L, 9L))
    driver.merge(first)
    driver.merge(retry)
    driver.merge(other)
    val v = driver.value
    assert(v.size === 2)
    assert(v.get(0).toSeq === Seq(5L, 5L))
    assert(v.get(1).toSeq === Seq(1L, 9L))
    assert(!driver.isZero)
    assert(driver.copy().value.size === 2)
    driver.reset()
    assert(driver.isZero)
  }
}
