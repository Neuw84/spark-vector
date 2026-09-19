package io.sparkvector.shuffle

import java.nio.file.{Files, Path}

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.execution.{ColumnarToRowExec, RowToColumnarExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, AQEShuffleReadExec, QueryStageExec}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.vector.VectorShuffleExchangeExec
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
    tempDir = Files.createTempDirectory("spark-vector-shuffle")
    spark = SparkSession.builder()
      .master("local[4]")
      .appName("VectorShuffleSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "6")
      .config("spark.sql.warehouse.dir", tempDir.resolve("wh").toString)
      .config("spark.plugins", "io.sparkvector.spark.VectorPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.sql.vector.shuffle.VectorShuffleManager")
      .config("spark.vector.shuffle.enabled", "true")
      .config("spark.vector.exec.strictFloatingPoint", "true")
      .getOrCreate()
    val data = spark.range(0, 20000).selectExpr(
      "id", "cast(id % 97 as int) as k", "cast(id % 1000 as string) as s",
      "cast(id % 7 as double) / 3 as x", "date_add(date '2020-01-01', cast(id % 400 as int)) as d",
      "cast(id % 13 as decimal(12,2)) / 7 as dec", "id % 3 = 0 as b")
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
    try assert(ours === spark.sql(sql).collect().toSeq) finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
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

  test("a shuffled hash join and a merge join read both sides from our exchanges") {
    spark.sessionState.conf.setConfString("spark.sql.autoBroadcastJoinThreshold", "-1")
    try {
      val sql = "select a.k, count(*) from t a join (select k, count(*) n from t group by k) b on a.k = b.k group by a.k"
      val plan = collectPlan(spark.sql(sql))
      val ex = exchanges(plan)
      assert(ex.nonEmpty && ex.forall(_.isInstanceOf[VectorShuffleExchangeExec]), s"$plan")
      checkAgainstSpark(sql)
    } finally spark.sessionState.conf.unsetConf("spark.sql.autoBroadcastJoinThreshold")
  }
}
