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

import io.vecruntime.spark.{VectorConf, VectorPlugin}
import io.vecruntime.spark.test.{CometTest, TestTables, VectorQuerySuite}
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec, VectorToCometExec}

/**
 * Mixed chains through Comet's native shuffle (#280): a join or a final aggregate of Comet's is
 * reached through the exchange, and the exchange over one of our operators is Comet's native shuffle
 * over our export node already (`useCometShuffle`). Comet's filter is off so the leaves are ours,
 * its shuffle, joins and aggregate on, ours off. Requires the Comet jar (`mvn -Pcomet`).
 */
class CometMixedShuffleSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]

  override protected def extraSparkConf: Map[String, String] = Map(
    "spark.plugins" -> s"org.apache.spark.CometPlugin,${classOf[VectorPlugin].getName}",
    "spark.shuffle.manager" -> "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager",
    "spark.comet.enabled" -> "true",
    "spark.comet.scan.enabled" -> "true",
    "spark.comet.exec.enabled" -> "true",
    "spark.comet.exec.shuffle.enabled" -> "true",
    "spark.comet.exec.filter.enabled" -> "false",
    "spark.comet.exec.project.enabled" -> "true",
    "spark.comet.exec.aggregate.enabled" -> "true",
    "spark.comet.exec.hashJoin.enabled" -> "true",
    "spark.comet.exec.broadcastHashJoin.enabled" -> "true",
    "spark.comet.exec.broadcastExchange.enabled" -> "true",
    "spark.comet.exec.sortMergeJoin.enabled" -> "false",
    "spark.comet.exec.sort.enabled" -> "false",
    "spark.comet.exec.window.enabled" -> "false",
    "spark.memory.offHeap.enabled" -> "true",
    "spark.memory.offHeap.size" -> "1g",
    "spark.comet.explainFallback.enabled" -> "false",
    "spark.sql.parquet.enableVectorizedReader" -> "true",
    "spark.sql.adaptive.enabled" -> "true",
    VectorConf.CometMixedEnabled -> "true",
    VectorConf.CometPreferComet -> "all",
    VectorConf.AggregateEnabled -> "false",
    VectorConf.ProjectEnabled -> "false",
    VectorConf.BroadcastHashJoinEnabled -> "false",
    VectorConf.ShuffledHashJoinEnabled -> "false"
  )

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet-mixed-shuffle/t"))
  }

  private def nodesNamed(df: org.apache.spark.sql.DataFrame, prefix: String) =
    PlanUtils.allNodes(finalPlan(df)).filter(_.getClass.getSimpleName.startsWith(prefix))

  private def awaitReleased(): Unit = {
    val deadline = System.nanoTime() + 5_000_000_000L
    while (ArrowCData.liveExports() > 0 && System.nanoTime() < deadline) Thread.sleep(50)
    assert(ArrowCData.liveExports() === 0, "every exported column should have been released by Comet")
  }

  private val Join = "SELECT a.i, b.s FROM t a JOIN t b ON a.i = b.i WHERE a.d > 0.5 AND b.l IS NOT NULL"

  private def reasonsOf(df: org.apache.spark.sql.DataFrame): String =
    org.apache.spark.sql.vector.VectorFallback.reasons(finalPlan(df)).map { case (n, r) =>
      s"${n.nodeName}: $r"
    }.mkString("; ")

  test("Comet's broadcast join above our filters, its broadcast over our chain (adaptive execution off)", CometTest) {
    // Under adaptive execution the broadcast stage is planned and run before the join's stage, with Spark's
    // exchange over the child as Comet's rule saw it (our filter, not native), so Comet's broadcast is out
    // of reach there; without it the exchange is the join's direct child and Comet converts both.
    withConf("spark.sql.adaptive.enabled" -> "false") {
      val df = checkVectorized(Join, Seq(Filter))
      val plan = finalPlan(df)
      assert(
        nodesNamed(df, "CometBroadcastHashJoin").nonEmpty,
        s"expected Comet's broadcast join above our filters; reasons: ${reasonsOf(df)}\n${plan.treeString}"
      )
      assert(nodesNamed(df, "CometBroadcastExchange").nonEmpty, plan.treeString)
      assert(nodesOf[VectorToCometExec](df).nonEmpty, plan.treeString)
      awaitReleased()
    }
  }

  test("Comet's shuffled hash join above our filters, through its shuffle", CometTest) {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      val df = checkVectorized(Join.replace("SELECT a.i", "SELECT /*+ SHUFFLE_HASH(b) */ a.i"), Seq(Filter))
      val plan = finalPlan(df)
      assert(
        nodesNamed(df, "CometHashJoin").nonEmpty,
        s"expected Comet's hash join above its shuffle over our filters:\n${plan.treeString}"
      )
      assert(nodesOf[VectorToCometExec](df).nonEmpty, plan.treeString)
      awaitReleased()
    }
  }

  test("Comet's final aggregate above its shuffle above its partial above our filter", CometTest) {
    val df = checkVectorized("SELECT s, sum(d), max(i) FROM t WHERE i > 100 GROUP BY s", Seq(Filter))
    val plan = finalPlan(df)
    assert(nodesNamed(df, "CometHashAggregate").size == 2, s"expected both halves Comet's:\n${plan.treeString}")
    assert(
      nodesNamed(df, "CometShuffleExchange").nonEmpty || nodesNamed(df, "CometColumnarExchange").nonEmpty,
      plan.treeString
    )
    awaitReleased()
  }
}
