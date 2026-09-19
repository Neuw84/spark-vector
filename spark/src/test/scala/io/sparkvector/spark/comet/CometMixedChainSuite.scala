package io.sparkvector.spark.comet

import io.sparkvector.spark.VectorConf
import io.sparkvector.spark.test.{CometTest, TestTables, VectorQuerySuite}
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec, VectorProjectExec, VectorToCometExec}

/**
 * Mixed chains (#280): Comet's native operators above ours through the sink leaf, and ours above
 * Comet's. Requires the Comet jar on the test classpath (`mvn -Pcomet`); excluded by default through
 * the CometTest tag. Comet's project is on and its filter off, so a filter is ours and a projection
 * above it is Comet's when mixing is on.
 */
class CometMixedChainSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]

  override protected def extraSparkConf: Map[String, String] = CometTestConf.scanOnly ++ Map(
    "spark.comet.exec.project.enabled" -> "true",
    VectorConf.CometMixedEnabled -> "true",
    VectorConf.CometPreferComet -> "all")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet-mixed/t"))
  }

  private def nodesNamed(df: org.apache.spark.sql.DataFrame, prefix: String) =
    PlanUtils.allNodes(finalPlan(df)).filter(_.getClass.getSimpleName.startsWith(prefix))

  private def reasonsOf(df: org.apache.spark.sql.DataFrame): String =
    org.apache.spark.sql.vector.VectorFallback.reasons(finalPlan(df)).map { case (n, r) => s"${n.nodeName}: $r" }.mkString("; ")

  private def awaitReleased(): Unit = {
    val deadline = System.nanoTime() + 5_000_000_000L
    while (ArrowCData.liveExports() > 0 && System.nanoTime() < deadline) Thread.sleep(50)
    assert(ArrowCData.liveExports() === 0, "every exported column should have been released by Comet")
  }

  test("Comet's projection above our filter through the sink leaf", CometTest) {
    // Our filter (Comet's is off) below a projection Comet plans natively over the leaf.
    withConf(VectorConf.ProjectEnabled -> "false") {
      val df = checkVectorized("SELECT i * 2 AS ii, d + 1.0 AS dd, s FROM t WHERE i > 100 AND d IS NOT NULL", Seq(Filter))
      val plan = finalPlan(df)
      assert(nodesNamed(df, "CometProject").nonEmpty, s"expected Comet's projection above ours:\n${plan.treeString}")
      // Comet's block pass unwraps the sink placeholder; the pass-through union stays as the block's input.
      assert(nodesNamed(df, "CometUnion").nonEmpty, s"expected the pass-through sink under Comet's block:\n${plan.treeString}")
      assert(nodesOf[VectorToCometExec](df).nonEmpty, s"expected our export node under the leaf:\n${plan.treeString}")
      assert(nodesOf[VectorProjectExec](df).isEmpty)
      awaitReleased()
    }
  }

  test("our projection above Comet's filter (Comet below ours needs no leaf)", CometTest) {
    withConf("spark.comet.exec.filter.enabled" -> "true", "spark.comet.exec.project.enabled" -> "false", VectorConf.FilterEnabled -> "false") {
      val df = checkVectorized("SELECT i * 2 AS ii, s FROM t WHERE i > 100", Seq(Project))
      assert(nodesNamed(df, "CometFilter").nonEmpty, s"expected Comet's filter below ours:\n${finalPlan(df).treeString}")
      assert(nodesOf[VectorFilterExec](df).isEmpty)
    }
  }

  test("with mixing off the plan is today's: ours, then Spark's projection", CometTest) {
    withConf(VectorConf.CometMixedEnabled -> "false", VectorConf.ProjectEnabled -> "false") {
      val df = checkVectorized("SELECT i * 2 AS ii FROM t WHERE i > 100", Seq(Filter))
      assert(nodesNamed(df, "CometProject").isEmpty, finalPlan(df).treeString)
      assert(nodesNamed(df, "CometUnion").isEmpty)
    }
  }

  test("Comet's partial aggregate above our filter when the buffers are compatible", CometTest) {
    withConf("spark.comet.exec.aggregate.enabled" -> "true", VectorConf.AggregateEnabled -> "false") {
      // sum, min, max and a non-decimal avg lay their buffers out the same way in Spark and in Comet
      // (Comet's own list), so the partial goes to Comet over the leaf and the final, above Spark's
      // shuffle, may be anyone's.
      val df = checkVectorized("SELECT s, sum(d), max(i), min(l), avg(d2) FROM t WHERE i > 100 GROUP BY s", Seq(Filter))
      val reasons = org.apache.spark.sql.vector.VectorFallback.reasons(finalPlan(df)).map { case (n, r) => s"$n: $r" }
      assert(nodesNamed(df, "CometHashAggregate").nonEmpty, s"expected Comet's partial aggregate; reasons: ${reasons.mkString("; ")}\n${finalPlan(df).treeString}")
      awaitReleased()
    }
  }

  test("Comet's local limit, expand and union above our chains", CometTest) {
    withConf("spark.comet.exec.localLimit.enabled" -> "true", VectorConf.LimitEnabled -> "false", VectorConf.AggregateEnabled -> "false") {
      // A bare LIMIT plans as CollectLimit, whose Comet form needs Comet's shuffle (off here); a LIMIT
      // under an aggregate plans a LocalLimit in our filter's stage, and that one Comet takes natively.
      val df = checkVectorized("SELECT count(*) FROM (SELECT i FROM t WHERE i > 100 LIMIT 10)", Seq(Filter))
      assert(nodesNamed(df, "CometLocalLimit").nonEmpty, s"expected Comet's local limit above our filter; reasons: ${reasonsOf(df)}\n${finalPlan(df).treeString}")
    }
    withConf("spark.comet.exec.expand.enabled" -> "true", VectorConf.ExpandEnabled -> "false", VectorConf.AggregateEnabled -> "false") {
      val df = checkVectorized("SELECT s, b, sum(d) FROM t WHERE i > 100 GROUP BY s, b WITH ROLLUP", Seq(Filter))
      assert(nodesNamed(df, "CometExpand").nonEmpty, s"expected Comet's expand above our filter:\n${finalPlan(df).treeString}")
    }
    withConf("spark.comet.exec.union.enabled" -> "true", VectorConf.UnionEnabled -> "false") {
      val df = checkVectorized("SELECT i FROM t WHERE i > 100 UNION ALL SELECT i FROM t WHERE i < 50", Seq(Filter))
      // Comet's union over two leaves (each a pass-through union over our export node).
      assert(nodesOf[VectorToCometExec](df).size == 2, s"expected two leaves under Comet's union; reasons: ${reasonsOf(df)}\n${finalPlan(df).treeString}")
      assert(nodesNamed(df, "CometUnion").nonEmpty, finalPlan(df).treeString)
      awaitReleased()
    }
  }

  test("the acceleration view counts both engines and the leaf as the bridge", CometTest) {
    withConf(VectorConf.ProjectEnabled -> "false") {
      val df = checkVectorized("SELECT i * 2 AS ii FROM t WHERE i > 100", Seq(Filter))
      val accelerated = org.apache.spark.sql.vector.ui.PlanAcceleration.fromPlan(finalPlan(df))
      val engines = accelerated.nodes.map(_.engine).toSet
      assert(engines.contains(org.apache.spark.sql.vector.ui.Engine.Vector) && engines.contains(org.apache.spark.sql.vector.ui.Engine.Comet), engines.toString)
      val bridges = accelerated.nodes.filter(_.engine == org.apache.spark.sql.vector.ui.Engine.Bridge)
      assert(bridges.size == 2, s"the export node and the pass-through union are the bridge: ${accelerated.nodes.map(n => s"${n.name}=${n.engine}")}")
      assert(accelerated.fullyAccelerated, accelerated.nodes.map(n => s"${n.name}=${n.engine}").toString)
    }
  }

  test("a failure inside Comet's block above the leaf still releases every export", CometTest) {
    withConf("spark.sql.ansi.enabled" -> "true", VectorConf.ProjectEnabled -> "false") {
      // Comet's projection divides by zero under ANSI above our filter; whatever raises, our exports go.
      val failed = scala.util.Try(spark.sql("SELECT l DIV (i - i) AS z FROM t WHERE i > 100").collect())
      assert(failed.isFailure, "the ANSI division by zero should fail the query")
      awaitReleased()
    }
  }

  test("an aggregate whose buffers differ between the engines stays where it is, with the reason", CometTest) {
    withConf("spark.comet.exec.aggregate.enabled" -> "true", VectorConf.AggregateEnabled -> "false") {
      // count is not on Comet's list of buffers it will share with Spark: the pair is not split.
      checkFallback("SELECT s, count(*), sum(d) FROM t WHERE i > 100 GROUP BY s", Seq(classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]), "aggregate halves cannot be split across engines")
    }
  }
}
