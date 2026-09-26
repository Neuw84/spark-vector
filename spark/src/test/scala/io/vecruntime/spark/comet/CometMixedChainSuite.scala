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
package io.vecruntime.spark.comet

import io.vecruntime.spark.VectorConf
import io.vecruntime.spark.test.{CometTest, TestTables, VectorQuerySuite}
import org.apache.spark.sql.vecruntime.{PlanUtils, VectorFilterExec, VectorProjectExec, VectorToCometExec}

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
    VectorConf.CometPreferComet -> "all"
  )

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet-mixed/t"))
  }

  private def nodesNamed(df: org.apache.spark.sql.DataFrame, prefix: String) =
    PlanUtils.allNodes(finalPlan(df)).filter(_.getClass.getSimpleName.startsWith(prefix))

  private def reasonsOf(df: org.apache.spark.sql.DataFrame): String =
    org.apache.spark.sql.vecruntime.VectorFallback.reasons(finalPlan(df)).map { case (n, r) =>
      s"${n.nodeName}: $r"
    }.mkString("; ")

  private def awaitReleased(): Unit = {
    val deadline = System.nanoTime() + 5_000_000_000L
    while (ArrowCData.liveExports() > 0 && System.nanoTime() < deadline) Thread.sleep(50)
    assert(ArrowCData.liveExports() === 0, "every exported column should have been released by Comet")
  }

  test("Comet's projection above our filter through the sink leaf", CometTest) {
    // Our filter (Comet's is off) below a projection Comet plans natively over the leaf.
    withConf(VectorConf.ProjectEnabled -> "false") {
      val df =
        checkVectorized("SELECT i * 2 AS ii, d + 1.0 AS dd, s FROM t WHERE i > 100 AND d IS NOT NULL", Seq(Filter))
      val plan = finalPlan(df)
      assert(nodesNamed(df, "CometProject").nonEmpty, s"expected Comet's projection above ours:\n${plan.treeString}")
      // Comet's block pass unwraps the sink placeholder; the pass-through union stays as the block's input.
      assert(
        nodesNamed(df, "CometUnion").nonEmpty,
        s"expected the pass-through sink under Comet's block:\n${plan.treeString}"
      )
      assert(nodesOf[VectorToCometExec](df).nonEmpty, s"expected our export node under the leaf:\n${plan.treeString}")
      assert(nodesOf[VectorProjectExec](df).isEmpty)
      awaitReleased()
    }
  }

  test("our projection above Comet's filter (Comet below ours needs no leaf)", CometTest) {
    withConf(
      "spark.comet.exec.filter.enabled" -> "true",
      "spark.comet.exec.project.enabled" -> "false",
      VectorConf.FilterEnabled -> "false"
    ) {
      val df = checkVectorized("SELECT i * 2 AS ii, s FROM t WHERE i > 100", Seq(Project))
      assert(
        nodesNamed(df, "CometFilter").nonEmpty,
        s"expected Comet's filter below ours:\n${finalPlan(df).treeString}"
      )
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
      val reasons = org.apache.spark.sql.vecruntime.VectorFallback.reasons(finalPlan(df)).map { case (n, r) =>
        s"$n: $r"
      }
      assert(
        nodesNamed(df, "CometHashAggregate").nonEmpty,
        s"expected Comet's partial aggregate; reasons: ${reasons.mkString("; ")}\n${finalPlan(df).treeString}"
      )
      awaitReleased()
    }
  }

  test("Comet's local limit, expand and union above our chains", CometTest) {
    withConf(
      "spark.comet.exec.localLimit.enabled" -> "true",
      VectorConf.LimitEnabled -> "false",
      VectorConf.AggregateEnabled -> "false"
    ) {
      // A bare LIMIT plans as CollectLimit, whose Comet form needs Comet's shuffle (off here); a LIMIT
      // under an aggregate plans a LocalLimit in our filter's stage, and that one Comet takes natively.
      val df = checkVectorized("SELECT count(*) FROM (SELECT i FROM t WHERE i > 100 LIMIT 10)", Seq(Filter))
      assert(
        nodesNamed(df, "CometLocalLimit").nonEmpty,
        s"expected Comet's local limit above our filter; reasons: ${reasonsOf(df)}\n${finalPlan(df).treeString}"
      )
    }
    withConf(
      "spark.comet.exec.expand.enabled" -> "true",
      VectorConf.ExpandEnabled -> "false",
      VectorConf.AggregateEnabled -> "false"
    ) {
      val df = checkVectorized("SELECT s, b, sum(d) FROM t WHERE i > 100 GROUP BY s, b WITH ROLLUP", Seq(Filter))
      assert(
        nodesNamed(df, "CometExpand").nonEmpty,
        s"expected Comet's expand above our filter:\n${finalPlan(df).treeString}"
      )
    }
    withConf("spark.comet.exec.union.enabled" -> "true", VectorConf.UnionEnabled -> "false") {
      val df = checkVectorized("SELECT i FROM t WHERE i > 100 UNION ALL SELECT i FROM t WHERE i < 50", Seq(Filter))
      // Comet's union over two leaves (each a pass-through union over our export node).
      assert(
        nodesOf[VectorToCometExec](df).size == 2,
        s"expected two leaves under Comet's union; reasons: ${reasonsOf(df)}\n${finalPlan(df).treeString}"
      )
      assert(nodesNamed(df, "CometUnion").nonEmpty, finalPlan(df).treeString)
      awaitReleased()
    }
  }

  test("a Comet sink executed directly above a chain rooted in our join (two children)", CometTest) {
    // Comet 1.0's CometUnionExec reads its partitioning through originalPlan.withNewChildren(children):
    // the pass-through's originalPlan must take one child, or a chain rooted in a join asserts (TPC-H q2).
    withConf("spark.comet.exec.union.enabled" -> "true", VectorConf.UnionEnabled -> "false") {
      val df = checkVectorized(
        "SELECT a.i FROM t a JOIN t b ON a.i = b.i WHERE a.d > 0.5 AND b.l IS NOT NULL UNION ALL SELECT i FROM t WHERE i < 50",
        Seq(Filter)
      )
      assert(nodesNamed(df, "CometUnion").nonEmpty, s"reasons: ${reasonsOf(df)}\n${finalPlan(df).treeString}")
      assert(nodesOf[VectorToCometExec](df).size == 2, finalPlan(df).treeString)
      awaitReleased()
    }
  }

  test("decimals cross the seam with Spark's results (TPC-H q11/q15/q17/q18 shapes, #281)", CometTest) {
    // decimal(12,2) and decimal(15,2) as in the benchmark's decimal schema: a product projected by Comet above
    // our filter and summed by ours; a Comet filter over a decimal; a decimal compared after the crossing.
    spark.range(0, 20000).selectExpr(
      "cast(id as int) as i",
      "cast(id % 997 as decimal(12,2)) / 7 as m",
      "cast((id % 89) as decimal(15,2)) * 1.5 as n",
      "if(id % 10 = 0, null, concat('g', id % 40)) as g"
    )
      .repartition(3).write.mode("overwrite").parquet(newTempPath("comet-mixed/dec"))
    spark.read.parquet(newTempPath("comet-mixed/dec")).createOrReplaceTempView("dec")
    withConf(VectorConf.ProjectEnabled -> "false") {
      checkVectorized("SELECT g, sum(m * (1 - n / 100)), avg(m), max(n) FROM dec WHERE i > 100 GROUP BY g", Seq(Filter))
      checkVectorized("SELECT sum(m) FROM dec WHERE i > 100 AND m * 2 > 50", Seq(Filter))
      checkVectorized("SELECT g, sum(m) AS s FROM dec WHERE i > 100 GROUP BY g HAVING sum(m) > 1000", Seq(Filter))
    }
    withConf("spark.comet.exec.filter.enabled" -> "true", VectorConf.FilterEnabled -> "false") {
      checkVectorized(
        "SELECT g, sum(m + n), count(*) FROM dec WHERE m > 10 AND i > 100 GROUP BY g",
        Seq(classOf[org.apache.spark.sql.vecruntime.VectorHashAggregateExec])
      )
    }
    // q17 / q11 shapes: Comet's filter, delegated, above our join or aggregate, over WIDE decimals our
    // operators computed and carried through the join's build side -- avg(m) is decimal(20,10),
    // 0.2 * avg(m) decimal(22,11), sum(m) decimal(26,6): DECIMAL128 lanes, which the export once widened
    // word by word so Comet read every value as two rows (TPC-H q11/q15/q17/q18 at SF1 decimals, #281).
    withConf("spark.comet.exec.filter.enabled" -> "true", VectorConf.CometPreferComet -> "filter") {
      val base =
        "FROM dec d JOIN (SELECT g AS g2, avg(m) AS a, 0.2 * avg(m) AS thr, sum(m) AS s FROM dec GROUP BY g) x ON d.g = x.g2"
      val Agg = classOf[org.apache.spark.sql.vecruntime.VectorHashAggregateExec]
      val df1 = checkVectorized(s"SELECT d.i, x.thr $base WHERE x.thr > 14.15 ORDER BY d.i", Seq(Agg))
      assert(
        nodesNamed(df1, "CometFilter").nonEmpty,
        s"expected Comet's delegated filter over the wide decimal:\n${finalPlan(df1).treeString}"
      )
      checkVectorized(s"SELECT d.i, x.a, x.s $base WHERE x.a > 70 AND x.s > 35000 ORDER BY d.i", Seq(Agg))
      checkVectorized(
        "SELECT sum(d.m) / 7.0 FROM dec d JOIN (SELECT g AS g2, 0.2 * avg(m) AS thr FROM dec GROUP BY g) a ON d.g = a.g2 WHERE d.m < a.thr",
        Seq(classOf[org.apache.spark.sql.vecruntime.VectorHashAggregateExec])
      )
      checkVectorized(
        "SELECT g, sum(m * n) AS v FROM dec WHERE i > 100 GROUP BY g HAVING sum(m * n) > (SELECT sum(m * n) * 0.0001 FROM dec WHERE i > 100)",
        Seq(classOf[org.apache.spark.sql.vecruntime.VectorHashAggregateExec])
      )
    }
    awaitReleased()
  }

  test("the acceleration view counts both engines and the leaf as the bridge", CometTest) {
    withConf(VectorConf.ProjectEnabled -> "false") {
      val df = checkVectorized("SELECT i * 2 AS ii FROM t WHERE i > 100", Seq(Filter))
      val accelerated = org.apache.spark.sql.vecruntime.ui.PlanAcceleration.fromPlan(finalPlan(df))
      val engines = accelerated.nodes.map(_.engine).toSet
      assert(
        engines.contains(org.apache.spark.sql.vecruntime.ui.Engine.Vector) && engines.contains(
          org.apache.spark.sql.vecruntime.ui.Engine.Comet
        ),
        engines.toString
      )
      val bridges = accelerated.nodes.filter(_.engine == org.apache.spark.sql.vecruntime.ui.Engine.Bridge)
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
      checkFallback(
        "SELECT s, count(*), sum(d) FROM t WHERE i > 100 GROUP BY s",
        Seq(classOf[org.apache.spark.sql.vecruntime.VectorHashAggregateExec]),
        "aggregate halves cannot be split across engines"
      )
    }
  }
}
