package io.sparkvector.spark.comet

import io.sparkvector.spark.{VectorConf, VectorPlugin}
import io.sparkvector.spark.test.{CometTest, TestTables, VectorQuerySuite}
import org.apache.spark.sql.vector.{
  PlanUtils,
  PreferComet,
  VectorFallback,
  VectorFilterExec,
  VectorProjectExec,
  VectorToCometExec
}

/**
 * The operator allowlist `spark.vector.comet.preferComet` (#281). Comet's own rule runs first and takes
 * what sits directly on its scan, so the list is about the operators above one of our chains: a listed
 * one is offered to Comet through the #280 leaf and ours steps aside with the reason; an unlisted one
 * stays ours; a listed one Comet declines is converted by ours after all, never left to Spark. Comet's
 * filter is off here so the chain's leaf is ours. Requires the Comet jar (`mvn -Pcomet`).
 */
class CometPreferCometSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]

  override protected def extraSparkConf: Map[String, String] =
    CometTestConf.scanOnly ++ Map(
      "spark.comet.exec.project.enabled" -> "true",
      "spark.comet.exec.expand.enabled" -> "true",
      VectorConf.CometMixedEnabled -> "true"
    )

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet-prefer/t"))
  }

  private def nodesNamed(df: org.apache.spark.sql.DataFrame, prefix: String) =
    PlanUtils.allNodes(finalPlan(df)).filter(_.getClass.getSimpleName.startsWith(prefix))

  private def reasons(df: org.apache.spark.sql.DataFrame): Seq[String] =
    VectorFallback.reasons(finalPlan(df)).map(_._2)

  private val Query = "SELECT i + 1 AS j, s FROM t WHERE i > 100"

  test("parse: kinds, predicates, 'all', refusals", CometTest) {
    val p = PreferComet.parse(" project:wideDecimal, sort , filter:estimatedRows>1000, aggregate, bogus, limit:nope ")
    assert(p.entries.map(_.kind) === Seq("project", "sort", "filter"))
    assert(p.entries.head.predicate === Some(PreferComet.WideDecimal))
    assert(p.entries(2).predicate === Some(PreferComet.EstimatedRows(1000)))
    assert(PreferComet.parse("all").entries === Seq(PreferComet.Entry("all", None)))
    assert(PreferComet.parse("").isEmpty && PreferComet.parse(null).isEmpty)
  }

  test("a listed projection above our filter goes to Comet, with the reason on ours", CometTest) {
    withConf(VectorConf.CometPreferComet -> "project") {
      val df = checkVectorized(Query, Seq(Filter))
      val plan = finalPlan(df)
      assert(nodesNamed(df, "CometProject").nonEmpty, plan.treeString)
      assert(nodesOf[VectorToCometExec](df).nonEmpty, plan.treeString)
      assert(nodesOf[VectorProjectExec](df).isEmpty, plan.treeString)
      assert(
        reasons(df).exists(_ == PreferComet.Reason),
        s"expected the delegation reason: ${reasons(df)}\n${plan.treeString}"
      )
    }
  }

  test("an empty list under mixed keeps every operator ours", CometTest) {
    withConf(VectorConf.CometPreferComet -> "") {
      val df = checkVectorized(Query, Seq(Filter, Project))
      assert(nodesNamed(df, "CometProject").isEmpty && nodesOf[VectorToCometExec](df).isEmpty, finalPlan(df).treeString)
    }
  }

  test("an unlisted kind stays ours while the listed one crosses", CometTest) {
    withConf(VectorConf.CometPreferComet -> "expand") {
      val df = checkVectorized(Query, Seq(Filter, Project))
      assert(nodesNamed(df, "CometProject").isEmpty, finalPlan(df).treeString)
    }
  }

  test("a predicate that does not hold keeps the operator ours", CometTest) {
    withConf(VectorConf.CometPreferComet -> "project:wideDecimal") {
      val df = checkVectorized(Query, Seq(Filter, Project))
      assert(nodesNamed(df, "CometProject").isEmpty, finalPlan(df).treeString)
    }
    withConf(VectorConf.CometPreferComet -> "project:strings") {
      val df = checkVectorized(Query, Seq(Filter))
      assert(nodesNamed(df, "CometProject").nonEmpty, finalPlan(df).treeString)
    }
  }

  test("our operator above a delegated one stays ours (the chain does not break at the swap)", CometTest) {
    // The delegated projection is still Spark's when its parent is planned; the parent must treat it as
    // columnar, or a Spark row join runs above the swap (TPC-H q5 under the allowlist).
    withConf(VectorConf.CometPreferComet -> "project") {
      val df = checkVectorized(
        "SELECT a.j, b.s FROM (SELECT i + 1 AS j FROM t WHERE i > 100) a JOIN t b ON a.j = b.i WHERE b.l IS NOT NULL",
        Seq(Filter)
      )
      val plan = finalPlan(df)
      assert(nodesNamed(df, "CometProject").nonEmpty, plan.treeString)
      val ourJoins = PlanUtils.allNodes(
        plan
      ).filter(_.getClass.getSimpleName.startsWith("Vector")).filter(_.nodeName.contains("Join"))
      assert(
        ourJoins.nonEmpty,
        s"expected our join above the delegated projection; reasons: ${reasons(df)}\n${plan.treeString}"
      )
      assert(
        !reasons(df).exists(_.contains("is not columnar")),
        s"the swap must not break the chain above it: ${reasons(df)}"
      )
    }
  }

  test("a listed operator Comet declines is ours, not Spark's", CometTest) {
    // Comet's project is off: it declines the offered projection, so ours takes it after the decline.
    withConf(VectorConf.CometPreferComet -> "project", "spark.comet.exec.project.enabled" -> "false") {
      val df = checkVectorized(Query, Seq(Filter, Project))
      val plan = finalPlan(df)
      assert(nodesNamed(df, "CometProject").isEmpty, plan.treeString)
      assert(
        !reasons(df).contains(PreferComet.Reason),
        s"the delegation reason must not survive a decline: ${reasons(df)}"
      )
    }
  }
}
