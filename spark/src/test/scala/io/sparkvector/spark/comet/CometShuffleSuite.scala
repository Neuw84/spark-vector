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

  test("bridge can be disabled, leaving Comet's row-based columnar shuffle", CometTest) {
    withConf(io.sparkvector.spark.VectorConf.CometShuffleEnabled -> "false") {
      val df = checkVectorized("SELECT s, count(*) FROM t GROUP BY s", Seq(Agg))
      assert(nodesOf[VectorToCometExec](df).isEmpty)
    }
  }
}
