package io.sparkvector.spark.comet

import io.sparkvector.spark.{VectorConf, VectorPlugin}
import io.sparkvector.spark.test.{CometTest, TestTables, VectorQuerySuite}
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorProjectExec}

/**
 * Comet in scan-only mode feeding spark-vector operators. Requires the Comet jar on the test
 * classpath (`mvn -Pcomet`); on macOS that means a Comet built from source, since the published
 * jar only bundles Linux native libraries. Excluded by default through the CometTest tag.
 */
class CometScanSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def extraSparkConf: Map[String, String] = Map(
    // Both plugins register their session extensions, Comet first so its scan rule runs first.
    "spark.plugins" -> s"org.apache.spark.CometPlugin,${classOf[VectorPlugin].getName}",
    "spark.comet.enabled" -> "true",
    "spark.comet.scan.enabled" -> "true",
    "spark.comet.exec.enabled" -> "false",
    "spark.comet.explainFallback.enabled" -> "false",
    "spark.sql.parquet.enableVectorizedReader" -> "true",
    "spark.sql.adaptive.enabled" -> "true")

  private def cometScans(df: org.apache.spark.sql.DataFrame) =
    org.apache.spark.sql.vector.PlanUtils.allNodes(finalPlan(df)).filter(_.getClass.getSimpleName.startsWith("CometScan"))

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet/t"))
    TestTables.createLineitem(spark, newTempPath("comet/lineitem"))
  }

  test("Comet scan is the columnar source and its vectors are adapted zero-copy", CometTest) {
    assert(CometVectorAdapter.isRegistered, "Comet adapter should be registered when Comet is on the classpath")
    val df = checkVectorized("SELECT i, l, d, dt, b, s FROM t WHERE i > 100 AND d IS NOT NULL", Seq(Filter))
    assert(cometScans(df).nonEmpty, s"expected CometScanExec:\n${finalPlan(df).treeString}")
    val filter = nodesOf[VectorFilterExec](df).head
    assert(filter.child.getClass.getSimpleName.startsWith("CometScan"), filter.child.nodeName)
  }

  test("filter, project and aggregate over Comet scan", CometTest) {
    checkVectorized("SELECT i, d * 2.0 AS x, s FROM t WHERE d2 > 0.5 AND l IS NOT NULL", Seq(Filter, Project))
    checkVectorized("SELECT s, count(*), sum(d2), avg(d2), min(i), max(l) FROM t GROUP BY s", Seq(Agg))
    checkVectorized("SELECT count(*), sum(d2), max(d), min(dt) FROM t WHERE i > 10", Seq(Filter, Agg))
    checkVectorized("SELECT i, count(*), sum(d2) FROM t GROUP BY i", Seq(Agg))
  }

  test("dictionary-encoded string keys from Comet group correctly", CometTest) {
    checkVectorized("SELECT s, b, count(*), sum(d2) FROM t WHERE i > 5 GROUP BY s, b", Seq(Filter, Agg))
    checkVectorized("SELECT s FROM t WHERE i < 100", Seq(Filter))
  }

  test("TPC-H Q1 and Q6 over Comet scan", CometTest) {
    val q6 = checkVectorized(TestTables.TpchQ6, Seq(Filter, Agg))
    assert(cometScans(q6).nonEmpty)
    val q1 = checkVectorized(TestTables.TpchQ1, Seq(Filter, Agg))
    assert(cometScans(q1).nonEmpty)
    info(finalPlan(q1).treeString)
  }

  test("plugin disabled leaves Comet scan feeding Spark operators", CometTest) {
    withConf(VectorConf.Enabled -> "false") {
      val df = spark.sql("SELECT count(*) FROM t WHERE i > 10")
      df.collect()
      assert(cometScans(df).nonEmpty)
      assert(nodesOf[VectorFilterExec](df).isEmpty)
    }
  }
}
