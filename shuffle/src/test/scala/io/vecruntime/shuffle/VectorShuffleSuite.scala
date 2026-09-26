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
package io.vecruntime.shuffle

import java.nio.file.{Files, Path}

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.execution.{ColumnarToRowExec, RowToColumnarExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, AQEShuffleReadExec, QueryStageExec}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.vecruntime.VectorShuffleExchangeExec
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * #288 slice 2: the columnar exchange under our shuffle manager, `local[4]`. Hash, round-robin,
 * single and range partitionings go through `VectorShuffleExchangeExec`; every result equals the
 * same query without the plugin; AQE coalesces over our exchange; no row conversion sits between the
 * exchange and the columnar operators above it.
 */
class VectorShuffleSuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private var plain: SparkSession = _
  private var tempDir: Path = _

  override def beforeAll(): Unit = {
    tempDir = Files.createTempDirectory("vecruntime-shuffle")
    spark = SparkSession.builder()
      .master("local[4]")
      .appName("VectorShuffleSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "6")
      .config("spark.sql.warehouse.dir", tempDir.resolve("wh").toString)
      .config("spark.plugins", "io.vecruntime.spark.VectorPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager")
      .config("spark.vector.shuffle.enabled", "true")
      .config("spark.vector.exec.strictFloatingPoint", "true")
      .getOrCreate()
    val data = spark.range(0, 20000).selectExpr(
      "id",
      "cast(id % 97 as int) as k",
      "cast(id % 1000 as string) as s",
      "cast(id % 7 as double) / 3 as x",
      "date_add(date '2020-01-01', cast(id % 400 as int)) as d",
      "cast(id % 13 as decimal(12,2)) / 7 as dec",
      "id % 3 = 0 as b"
    )
    data.write.mode("overwrite").parquet(tempDir.resolve("t").toString)
    spark.read.parquet(tempDir.resolve("t").toString).createOrReplaceTempView("t")
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()
  }

  private def collectPlan(df: DataFrame): SparkPlan = {
    df.collect()
    df.queryExecution.executedPlan match {
      case a: AdaptiveSparkPlanExec => a.executedPlan
      case p => p
    }
  }

  /** Every node, looking through the query stages (leaves in AQE's final plan) into their plans. */
  private def nodes(plan: SparkPlan): Seq[SparkPlan] = plan +: (plan match {
    case q: QueryStageExec => nodes(q.plan)
    case p => p.children.flatMap(nodes)
  })

  private def exchanges(plan: SparkPlan): Seq[SparkPlan] = nodes(plan).collect {
    case e: VectorShuffleExchangeExec => e
    case e: ShuffleExchangeExec => e
  }

  /** Rows of `sql` with the plugin's shuffle against the same query with the plugin off, order-insensitive. */
  private def checkAgainstSpark(sql: String): Unit = {
    val ours = spark.sql(sql).collect().toSeq.sortBy(_.toString)
    val theirs = spark.sessionState.conf.setConfString("spark.vector.enabled", "false")
    try {
      val expected = spark.sql(sql).collect().toSeq.sortBy(_.toString)
      assert(ours === expected, sql)
    } finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
  }

  private def assertOurExchange(df: DataFrame, expectedCount: Int = 1): SparkPlan = {
    val plan = collectPlan(df)
    val ex = exchanges(plan)
    assert(ex.nonEmpty, s"no exchange in\n$plan")
    assert(ex.forall(_.isInstanceOf[VectorShuffleExchangeExec]), s"Spark's exchange survived in\n$plan")
    assert(ex.size === expectedCount, s"expected $expectedCount exchanges in\n$plan")
    plan
  }

  test("hash partitioning: a grouped aggregate's final over our exchange, without row conversions") {
    val sql = "select k, count(*) c, sum(x) sx, max(s) ms from t group by k"
    val plan = assertOurExchange(spark.sql(sql))
    assert(nodes(plan).collect { case r: RowToColumnarExec => r }.isEmpty, s"RowToColumnar in\n$plan")
    checkAgainstSpark(sql)
  }

  test("hash partitioning on several keys of every lane type") {
    val sql = "select k, s, d, dec, b, count(*) c from t group by k, s, d, dec, b"
    assertOurExchange(spark.sql(sql))
    checkAgainstSpark(sql)
  }

  /**
   * Struct columns (Iceberg's `_partition` on a MERGE) cross our exchange flattened into lanes and come
   * back as structs: nested structs, nulls at the struct and at the field level, and
   * a struct as a hash key -- every row lands in the partition Spark's exchange puts it in.
   */
  test("struct columns and struct hash keys cross our exchange as lanes") {
    val path = tempDir.resolve("structs").toString
    spark.range(0, 5000).selectExpr(
      "cast(id as int) as k",
      "case when id % 17 = 0 then null else named_struct(" +
        "'a', cast(id % 11 as int), " +
        "'b', case when id % 5 = 0 then null else cast(id % 23 as string) end, " +
        "'c', named_struct('x', cast(id % 3 as bigint), 'y', cast(id % 7 as decimal(9,2)))) end as st"
    ).write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("st")
    val df = spark.table("st").repartition(
      5,
      org.apache.spark.sql.functions.col("st"),
      org.apache.spark.sql.functions.col("k")
    )
    val plan = collectPlan(df)
    assert(exchanges(plan).exists(_.isInstanceOf[VectorShuffleExchangeExec]), s"expected our exchange in\n$plan")
    assert(exchanges(plan).forall(_.isInstanceOf[VectorShuffleExchangeExec]), s"Spark's exchange survived in\n$plan")
    def rows(enabled: Boolean): Seq[String] = {
      spark.sessionState.conf.setConfString("spark.vector.enabled", enabled.toString)
      try
        spark.table("st").repartition(
          5,
          org.apache.spark.sql.functions.col("st"),
          org.apache.spark.sql.functions.col("k")
        )
          .selectExpr("spark_partition_id() as p", "k", "st").collect().map(_.toString).toSeq.sorted
      finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
    }
    assert(rows(enabled = true) === rows(enabled = false))
  }

  test("hash keys that are expressions (q47's self-join on rn + 1) are materialised under our exchange") {
    // A sort-merge self-join whose one side partitions on `k + 1`: without the materialised key the
    // exchange stayed Spark's, with a row Sort and a RowToColumnar over it.
    val sql =
      "select a.k, a.c, b.c from (select k, count(*) c from t group by k) a join (select k, count(*) c from t group by k) b on a.k = b.k + 1"
    spark.sessionState.conf.setConfString("spark.sql.autoBroadcastJoinThreshold", "-1")
    try {
      val plan =
        assertOurExchange(spark.sql(sql), expectedCount = 2) // the aggregate's exchange is reused for the other side
      assert(nodes(plan).collect { case r: RowToColumnarExec => r }.isEmpty, s"RowToColumnar in\n$plan")
      assert(
        nodes(plan).collect { case s: org.apache.spark.sql.execution.SortExec => s }.isEmpty,
        s"Spark's Sort in\n$plan"
      )
      // The exchange declares the original keys (what the join required); the projection under it holds the value.
      val computed = nodes(
        plan
      ).collect { case e: VectorShuffleExchangeExec => e }.filter(_.outputPartitioning.toString.contains("+ 1"))
      assert(computed.size === 1, s"expected one exchange on k + 1 in\n$plan")
      assert(
        computed.head.child.output.exists(_.name.startsWith("_shuffle_key_")),
        s"no materialised key under\n${computed.head}"
      )
      checkAgainstSpark(sql)
      // The same key with the plain side of the join the other way round, and a wider expression.
      val sql2 =
        "select a.k, b.c from (select k, count(*) c from t group by k) a join (select k, count(*) c from t group by k) b on a.k - 2 = b.k * 1"
      assertOurExchange(
        spark.sql(sql2),
        expectedCount = 3
      ) // b.k * 1 folds to b.k: its aggregate's exchange serves the join
      checkAgainstSpark(sql2)
    } finally spark.sessionState.conf.unsetConf("spark.sql.autoBroadcastJoinThreshold")
  }

  test("single partition: a global aggregate") {
    val sql = "select count(*), sum(x), min(d), max(dec) from t"
    assertOurExchange(spark.sql(sql))
    checkAgainstSpark(sql)
  }

  test("round robin: repartition(n) keeps every row exactly once") {
    val df = spark.table("t").repartition(5)
    assertOurExchange(df.select("id"))
    assert(df.rdd.getNumPartitions === 5)
    val ids = df.select("id").collect().map(_.getLong(0)).sorted
    assert(ids.toSeq === (0L until 20000L))
  }

  test("range partitioning: a global order by is Spark's rows in Spark's order") {
    val sql = "select s, x, k from t order by s desc, x, k limit 5000"
    val ours = spark.sql(sql).collect().toSeq
    spark.sessionState.conf.setConfString("spark.vector.enabled", "false")
    try assert(ours === spark.sql(sql).collect().toSeq)
    finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
    val plan = collectPlan(spark.sql("select s, x, k from t order by s desc, x, k"))
    assert(exchanges(plan).exists(_.isInstanceOf[VectorShuffleExchangeExec]), s"$plan")
  }

  test("AQE coalesces small partitions over our exchange") {
    spark.sessionState.conf.setConfString("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.sessionState.conf.setConfString("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
    spark.sessionState.conf.setConfString("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1MB")
    try {
      val df = spark.sql("select k, count(*) c from t group by k")
      val plan = collectPlan(df)
      assert(nodes(plan).collect { case r: AQEShuffleReadExec => r }.nonEmpty, s"no coalesced read in\n$plan")
      assertOurExchange(df)
      checkAgainstSpark("select k, count(*) c from t group by k")
    } finally {
      spark.sessionState.conf.unsetConf("spark.sql.adaptive.coalescePartitions.enabled")
      spark.sessionState.conf.unsetConf("spark.sql.adaptive.advisoryPartitionSizeInBytes")
      spark.sessionState.conf.unsetConf("spark.sql.adaptive.coalescePartitions.minPartitionSize")
    }
  }

  test("the stage metrics see our shuffle: bytes and records written, local blocks and bytes read, AQE's data size") {
    val stages = scala.collection.mutable.ArrayBuffer.empty[org.apache.spark.scheduler.StageInfo]
    val listener = new org.apache.spark.scheduler.SparkListener {
      override def onStageCompleted(e: org.apache.spark.scheduler.SparkListenerStageCompleted): Unit =
        stages.synchronized { stages += e.stageInfo }
    }
    spark.sparkContext.addSparkListener(listener)
    try {
      val df = spark.sql("select k, count(*) c, sum(x) sx from t group by k")
      val plan = collectPlan(df)
      val deadline = System.nanoTime() + 10000000000L
      while (
        System.nanoTime() < deadline && stages.synchronized(
          stages.map(_.taskMetrics.shuffleReadMetrics.localBlocksFetched).sum
        ) == 0
      ) Thread.sleep(50)
      val ex = exchanges(plan).collectFirst { case e: VectorShuffleExchangeExec => e }.get
      val write = stages.synchronized(stages.map(_.taskMetrics.shuffleWriteMetrics.bytesWritten).sum)
      val read = stages.synchronized(stages.map(_.taskMetrics.shuffleReadMetrics))
      assert(write > 0, "shuffle bytes written")
      val sql = ex.metrics.filter(_._1.matches(".*(Blocks|Bytes|records|Wait|Time|Size|dataSize).*")).map {
        case (k, m) => s"$k=${m.value}"
      }.toSeq.sorted.mkString(", ")
      assert(
        read.map(_.localBlocksFetched).sum > 0,
        s"local blocks fetched; SQL metrics: $sql; stage read totals: blocks=${read.map(_.localBlocksFetched).sum}/${read.map(_.remoteBlocksFetched).sum} bytes=${read.map(_.totalBytesRead).sum}"
      )
      assert(
        read.map(_.totalBytesRead).sum === write,
        s"bytes read ${read.map(_.totalBytesRead).sum} vs written $write"
      )
      assert(read.map(_.recordsRead).sum > 0, "records read")
      // dataSize is the pre-compression Arrow size, as Spark's is its rows' pre-compression size: at least the compressed bytes.
      assert(
        ex.metrics("dataSize").value >= write,
        s"AQE's data size ${ex.metrics("dataSize").value} vs bytes written $write"
      )
      assert(ex.runtimeStatistics.sizeInBytes.toLong === ex.metrics("dataSize").value)
    } finally spark.sparkContext.removeSparkListener(listener)
  }

  test("#358: unregistering a shuffle deletes our map outputs' data and index files") {
    val df = spark.sql("select k, count(*) c from t group by k")
    val plan = collectPlan(df)
    val ex = exchanges(plan).collectFirst { case e: VectorShuffleExchangeExec => e }.get
    val shuffleId = ex.shuffleDependency.shuffleId
    val disk = org.apache.spark.SparkEnv.get.blockManager.diskBlockManager
    def files() = disk.getAllFiles().filter(_.getName.startsWith(s"shuffle_${shuffleId}_")).map(_.getName).sorted
    val before = files()
    assert(before.exists(_.endsWith(".data")) && before.exists(_.endsWith(".index")), s"map outputs on disk: $before")
    // What the ContextCleaner does on every executor once the dependency is unreachable.
    assert(org.apache.spark.SparkEnv.get.shuffleManager.unregisterShuffle(shuffleId))
    assert(files().isEmpty, s"left behind: ${files()}")
  }

  test("a shuffled hash join and a merge join read both sides from our exchanges") {
    spark.sessionState.conf.setConfString("spark.sql.autoBroadcastJoinThreshold", "-1")
    try {
      val sql =
        "select a.k, count(*) from t a join (select k, count(*) n from t group by k) b on a.k = b.k group by a.k"
      val plan = collectPlan(spark.sql(sql))
      val ex = exchanges(plan)
      assert(ex.nonEmpty && ex.forall(_.isInstanceOf[VectorShuffleExchangeExec]), s"$plan")
      checkAgainstSpark(sql)
    } finally spark.sessionState.conf.unsetConf("spark.sql.autoBroadcastJoinThreshold")
  }
}
