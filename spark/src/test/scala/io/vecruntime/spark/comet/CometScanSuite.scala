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

  override protected def extraSparkConf: Map[String, String] = CometTestConf.scanOnly

  /** Comet 1.0 plans its scan as CometNativeScanExec (or CometScanExec / CometBatchScanExec). */
  private def cometScans(df: org.apache.spark.sql.DataFrame) =
    org.apache.spark.sql.vector.PlanUtils.allNodes(finalPlan(df)).filter { n =>
      val name = n.getClass.getSimpleName
      name.startsWith("Comet") && name.contains("Scan")
    }

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("comet/t"))
    TestTables.createLineitem(spark, newTempPath("comet/lineitem"))
  }

  test("Comet scan is the columnar source and its vectors are adapted zero-copy", CometTest) {
    val df = checkVectorized("SELECT i, l, d, dt, b, s FROM t WHERE i > 100 AND d IS NOT NULL", Seq(Filter))
    assert(cometScans(df).nonEmpty, s"expected a Comet scan:\n${finalPlan(df).treeString}")
    val filter = nodesOf[VectorFilterExec](df).head
    assert(cometScans(df).contains(filter.child), filter.child.nodeName)
    // The adapter registers itself the first time a batch is adapted (executor side, local here).
    assert(CometVectorAdapter.isRegistered, "Comet adapter should be registered when Comet is on the classpath")
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

  test("the prefetching converter never wraps a Comet scan (#403)", CometTest) {
    withConf(VectorConf.ScanPrefetch -> "2") {
      val df = checkVectorized("SELECT s, count(*), max(d) FROM t WHERE i > 100 GROUP BY s", Seq(Filter, Agg))
      assert(nodesOf[org.apache.spark.sql.vector.VectorPrefetchScanExec](df).isEmpty, finalPlan(df).treeString)
      val filter = nodesOf[VectorFilterExec](df).head
      assert(cometScans(df).contains(filter.child), filter.child.nodeName)
    }
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
