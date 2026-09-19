package io.sparkvector.spark.comet

import io.sparkvector.spark.VectorPlugin
import io.sparkvector.spark.test.{CometTest, TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.vector.{CometShuffle, PlanUtils, VectorFilterExec, VectorHashAggregateExec, VectorToCometExec}

/**
 * Comet scan and Comet native shuffle around spark-vector operators: partial aggregates hand their
 * batches to the shuffle writer through the Arrow C Data bridge, the Final aggregate reads Comet's
 * shuffle output zero copy. Requires the Comet jar (`mvn -Pcomet`).
 */
class CometShuffleSuite extends VectorQuerySuite {
  private val Filter = classOf[VectorFilterExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def extraSparkConf: Map[String, String] = Map(
    "spark.plugins" -> s"org.apache.spark.CometPlugin,${classOf[VectorPlugin].getName}",
    "spark.shuffle.manager" -> "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager",
    "spark.comet.enabled" -> "true",
    "spark.comet.scan.enabled" -> "true",
    "spark.comet.exec.enabled" -> "true",
    "spark.comet.exec.shuffle.enabled" -> "true",
    "spark.comet.exec.project.enabled" -> "false",
    "spark.comet.exec.filter.enabled" -> "false",
    "spark.comet.exec.aggregate.enabled" -> "false",
    "spark.comet.exec.sort.enabled" -> "false",
    "spark.comet.exec.localLimit.enabled" -> "false",
    "spark.comet.exec.globalLimit.enabled" -> "false",
    "spark.comet.exec.takeOrderedAndProject.enabled" -> "false",
    "spark.comet.exec.hashJoin.enabled" -> "false",
    "spark.comet.exec.sortMergeJoin.enabled" -> "false",
    "spark.comet.exec.broadcastHashJoin.enabled" -> "false",
    "spark.comet.exec.broadcastExchange.enabled" -> "false",
    "spark.comet.exec.expand.enabled" -> "false",
    "spark.comet.exec.union.enabled" -> "false",
    "spark.comet.exec.window.enabled" -> "false",
    "spark.comet.exec.coalesce.enabled" -> "false",
    "spark.comet.exec.collectLimit.enabled" -> "false",
    "spark.comet.exec.explode.enabled" -> "false",
    "spark.comet.exec.sample.enabled" -> "false",
    "spark.memory.offHeap.enabled" -> "true",
    "spark.memory.offHeap.size" -> "1g",
    "spark.comet.explainFallback.enabled" -> "false",
    "spark.sql.adaptive.enabled" -> "true")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet-shuffle/t"))
    TestTables.createLineitem(spark, newTempPath("comet-shuffle/lineitem"))
    TestTables.mixedDataFrame(spark).selectExpr("i", "s", "named_struct('a', i, 'b', s) AS st")
      .repartition(2).write.mode("overwrite").parquet(newTempPath("comet-shuffle/t_struct"))
    spark.read.parquet(newTempPath("comet-shuffle/t_struct")).createOrReplaceTempView("t_struct")
  }

  private def cometExchanges(plan: SparkPlan): Seq[SparkPlan] = PlanUtils.allNodes(plan).filter(CometShuffle.isCometExchange)

  private def assertBridgedShuffle(df: org.apache.spark.sql.DataFrame): Unit = {
    val plan = finalPlan(df)
    val exchanges = cometExchanges(plan)
    assert(exchanges.nonEmpty, s"expected a Comet exchange:\n${plan.treeString}")
    val bridged = exchanges.filter(e => e.children.head.isInstanceOf[VectorToCometExec])
    assert(bridged.nonEmpty, s"expected a Comet exchange over VectorToComet:\n${plan.treeString}")
    bridged.foreach(e => assert(CometShuffle.isNative(e), s"expected the native shuffle:\n${e.treeString}"))
    assert(nodesOf[HashAggregateExec](df).isEmpty, s"both aggregate stages should be ours:\n${plan.treeString}")
  }

  private def assertNoLeak(): Unit = {
    val deadline = System.nanoTime() + 10_000_000_000L
    while (ArrowCData.liveExports() > 0 && System.nanoTime() < deadline) Thread.sleep(50)
    assert(ArrowCData.liveExports() === 0, "every exported column should have been released by Comet")
  }

  test("partial aggregate feeds Comet's native shuffle through the C Data bridge", CometTest) {
    val df = checkVectorized("SELECT count(*), sum(d2), avg(d), min(i), max(l) FROM t WHERE i > 10", Seq(Filter, Agg))
    assertBridgedShuffle(df)
    assertNoLeak()
  }

  test("wide decimals cross the native shuffle as 128-bit lanes (#281)", CometTest) {
    // sum(decimal(16,6)) is decimal(26,6), avg decimal(20,10): the partial's results and a wide key
    // are DECIMAL128 lanes; the export once widened them word by word, so every value became two rows.
    spark.range(0, 20000).selectExpr("cast(id as int) as i", "cast(id % 997 as decimal(12,2)) / 7 as m",
      "cast(id % 40 as decimal(22,4)) * 1000000000 as w", "if(id % 10 = 0, null, concat('g', id % 40)) as g")
      .repartition(3).write.mode("overwrite").parquet(newTempPath("comet-shuffle/dec"))
    spark.read.parquet(newTempPath("comet-shuffle/dec")).createOrReplaceTempView("dec")
    val df = checkVectorized("SELECT g, sum(m), avg(m), max(w) FROM dec WHERE i > 100 GROUP BY g", Seq(Filter, Agg))
    assertBridgedShuffle(df)
    val keyed = checkVectorized("SELECT w, count(*), sum(m) FROM dec WHERE i > 100 GROUP BY w", Seq(Filter, Agg))
    assertBridgedShuffle(keyed)
    assertNoLeak()
  }

  test("grouped keys of every supported type cross the bridge, nulls included", CometTest) {
    assertBridgedShuffle(checkVectorized("SELECT s, count(*), sum(d2) FROM t GROUP BY s", Seq(Agg)))
    assertBridgedShuffle(checkVectorized("SELECT b, dt, i > 100 AS big, count(*), max(d) FROM t GROUP BY b, dt, i > 100", Seq(Agg)))
    assertBridgedShuffle(checkVectorized("SELECT l, s, count(*), avg(d2) FROM t WHERE i < 3000 GROUP BY l, s", Seq(Filter, Agg)))
    assertNoLeak()
  }

  test("TPC-H Q1 and Q6 with Comet scan and shuffle", CometTest) {
    val q6 = checkVectorized(TestTables.TpchQ6, Seq(Filter, Agg))
    assertBridgedShuffle(q6)
    val q1 = checkVectorized(TestTables.TpchQ1, Seq(Filter, Agg))
    assertBridgedShuffle(q1)
    info(finalPlan(q1).treeString)
    assertNoLeak()
  }

  test("range-partitioned exchange above our operators takes the native shuffle; sort above it is ours", CometTest) {
    val sql = "SELECT s, count(*) AS c, sum(d2) AS total FROM t GROUP BY s ORDER BY s, c DESC"
    val df = checkVectorized(sql, Seq(Agg, classOf[org.apache.spark.sql.vector.VectorSortExec]))
    val plan = finalPlan(df)
    val exchanges = cometExchanges(plan)
    val range = exchanges.filter(_.outputPartitioning.isInstanceOf[org.apache.spark.sql.catalyst.plans.physical.RangePartitioning])
    assert(range.nonEmpty, s"expected a Comet range exchange:\n${plan.treeString}")
    range.foreach { e =>
      assert(e.children.head.isInstanceOf[VectorToCometExec], s"range exchange should sit on the bridge:\n${plan.treeString}")
      assert(CometShuffle.isNative(e), s"range exchange should be native:\n${plan.treeString}")
    }
    // The rows come back in order: the sort ran over Comet's columnar shuffle output.
    val keys = df.collect().map(r => (Option(r.getString(0)), -r.getLong(1)))
    assert(keys.toSeq === keys.sortBy(k => (k._1.isDefined, k._1.getOrElse(""), k._2)).toSeq) // ASC NULLS FIRST
    assertNoLeak()
    withConf(io.sparkvector.spark.VectorConf.CometRangeShuffleEnabled -> "false") {
      val plain = checkVectorized(sql, Seq(Agg))
      val ranges = cometExchanges(finalPlan(plain)).filter(_.outputPartitioning.isInstanceOf[org.apache.spark.sql.catalyst.plans.physical.RangePartitioning])
      assert(ranges.forall(e => !e.children.head.isInstanceOf[VectorToCometExec]), finalPlan(plain).treeString)
    }
  }

  test("a passed-through nested column keeps the exchange off the bridge", CometTest) {
    // A struct has no lane and cannot cross the C Data interface: the project is ours, the shuffle above it is not bridged.
    val df = checkVectorized("SELECT st, i + 1 AS n FROM t_struct WHERE i % 3 = 0 DISTRIBUTE BY i", Seq.empty)
    assert(nodesOf[VectorToCometExec](df).isEmpty, finalPlan(df).treeString)
    assertNoLeak()
  }

  test("bridge can be disabled, leaving Comet's row-based columnar shuffle", CometTest) {
    withConf(io.sparkvector.spark.VectorConf.CometShuffleEnabled -> "false") {
      val df = checkVectorized("SELECT s, count(*) FROM t GROUP BY s", Seq(Agg))
      assert(nodesOf[VectorToCometExec](df).isEmpty)
    }
  }
}
