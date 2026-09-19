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
    VectorConf.CometMixedEnabled -> "true")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet-mixed/t"))
  }

  private def nodesNamed(df: org.apache.spark.sql.DataFrame, prefix: String) =
    PlanUtils.allNodes(finalPlan(df)).filter(_.getClass.getSimpleName.startsWith(prefix))

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

  test("an aggregate pair is not split across the engines", CometTest) {
    withConf("spark.comet.exec.aggregate.enabled" -> "true", VectorConf.AggregateEnabled -> "false") {
      // Both halves stay on one side: Comet's rule never saw our filter, so the pair is Spark's here,
      // and the mixed pass leaves an aggregate alone (Comet's final needs Comet's partial buffers).
      val df = checkVectorized("SELECT s, count(*), sum(d) FROM t WHERE i > 100 GROUP BY s", Seq(Filter))
      assert(nodesNamed(df, "CometHashAggregate").isEmpty, finalPlan(df).treeString)
    }
  }
}
