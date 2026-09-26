/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.{CollectLimitExec, GlobalLimitExec, LocalLimitExec}
import org.apache.spark.sql.vector.{
  VectorBroadcastHashJoinExec,
  VectorCollectLimitExec,
  VectorFallback,
  VectorFilterExec,
  VectorGlobalLimitExec,
  VectorHashAggregateExec,
  VectorLocalLimitExec,
  VectorSortExec
}

/**
 * The three limit operators. A `LIMIT` without an `ORDER BY` returns *some* n rows, and which ones
 * depends on shuffle arrival order even in Spark, so where the limit cuts the data the tests check
 * the count and that every row belongs to the input; where the limit exceeds the data the result is
 * compared row for row against Spark as usual.
 */
class VectorLimitSuite extends VectorQuerySuite {

  private val Collect = classOf[VectorCollectLimitExec]
  private val Filter = classOf[VectorFilterExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("limit/t"))
    TestTables.createLineitem(spark, newTempPath("limit/lineitem"))
  }

  /** Runs `sql` with the plugin, asserts `n` rows come back through our operator and Spark's is gone. */
  private def checkCut(
      sql: String,
      n: Int,
      extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil
  ): DataFrame = {
    val df = withPlugin(enabled = true) { val d = spark.sql(sql); d.collect(); d }
    assert(df.collect().length === n, s"row count for: $sql\n${finalPlan(df).treeString}")
    (Collect +: extra).foreach(cls =>
      assert(
        nodesOf(df)(scala.reflect.ClassTag(cls)).nonEmpty,
        s"expected ${cls.getSimpleName} in plan for: $sql\n${finalPlan(df).treeString}"
      )
    )
    assert(nodesOf[CollectLimitExec](df).isEmpty, s"Spark's CollectLimitExec should be gone for: $sql")
    df
  }

  test("LIMIT over the scan, above and below a batch, zero, and larger than the data") {
    // t has 20000 rows in 3 partitions; Parquet batches hold 4096 rows.
    assert(checkCut("SELECT i, s FROM t LIMIT 10", 10).collect().forall(r => r.getInt(0) >= 0 && r.getInt(0) < 20000))
    checkCut("SELECT i FROM t LIMIT 1", 1)
    checkCut("SELECT i FROM t LIMIT 4096", 4096) // exactly one batch per partition
    checkCut("SELECT i FROM t LIMIT 5000", 5000) // crosses a batch boundary
    // LIMIT 0 is folded to an empty relation by the optimizer before any operator runs.
    assert(withPlugin(enabled = true)(spark.sql("SELECT i FROM t LIMIT 0").collect()).isEmpty)
    checkVectorized("SELECT i FROM t WHERE i < 0 LIMIT 10", Seq(Collect, Filter)) // an empty input under a live limit
    // Larger than the data: every row, compared with Spark.
    checkVectorized("SELECT i, s, d FROM t WHERE i < 300 LIMIT 1000", Seq(Collect, Filter))
    checkVectorized("SELECT i FROM t LIMIT 100000", Seq(Collect))
  }

  test("LIMIT above a filter, an aggregate, a join and a local sort") {
    val filtered = checkCut("SELECT i FROM t WHERE i >= 19000 LIMIT 100", 100, Seq(Filter))
    assert(filtered.collect().forall(_.getInt(0) >= 19000))
    // The filter below a limit forwards a selection (both are ours): the boundary batch is compacted.
    val sparse = checkCut("SELECT i, s FROM t WHERE s = 's7' LIMIT 250", 250, Seq(Filter))
    assert(sparse.collect().forall(_.getString(1) == "s7"))
    val grouped = checkCut("SELECT s, count(*) AS c FROM t GROUP BY s LIMIT 7", 7, Seq(Agg))
    assert(grouped.collect().forall(r => r.isNullAt(0) || r.getString(0).startsWith("s")))
    // 50 groups under LIMIT 100: adaptive execution may drop the limit once the stage statistics show it is a no-op.
    checkVectorized("SELECT s, count(*) AS c, sum(l) AS sl FROM t GROUP BY s LIMIT 100", Seq(Agg))
    val joined = checkCut(
      "SELECT t.i, d.k FROM t JOIN (SELECT i AS k FROM t WHERE i < 100) d ON t.i = d.k LIMIT 30",
      30,
      Seq(classOf[VectorBroadcastHashJoinExec])
    )
    assert(joined.collect().forall(r => r.getInt(0) == r.getInt(1) && r.getInt(0) < 100))
    checkVectorized(
      "SELECT t.i, d.k FROM t JOIN (SELECT i AS k FROM t WHERE i < 100) d ON t.i = d.k LIMIT 500",
      Seq(Collect, classOf[VectorBroadcastHashJoinExec])
    )
    checkCut("SELECT i FROM t WHERE i < 5000 SORT BY i LIMIT 12", 12, Seq(classOf[VectorSortExec]))
    checkCut("SELECT l_returnflag, count(*) FROM lineitem GROUP BY l_returnflag LIMIT 2", 2, Seq(Agg))
  }

  test("LOCAL LIMIT in a subquery is ours; the GLOBAL LIMIT above Spark's row shuffle stays Spark's") {
    // LIMIT inside a subquery plans as LocalLimit -> single-partition exchange -> GlobalLimit.
    val df = withPlugin(enabled = true) {
      val d = spark.sql("SELECT i FROM (SELECT i FROM t LIMIT 5000) WHERE i >= 0"); d.collect(); d
    }
    assert(df.collect().length === 5000)
    assert(nodesOf[VectorLocalLimitExec](df).nonEmpty, finalPlan(df).treeString)
    assert(nodesOf[LocalLimitExec](df).isEmpty, "Spark's LocalLimitExec should be gone")
    assert(nodesOf[GlobalLimitExec](df).nonEmpty, "the global limit above the row shuffle stays with Spark")
    assert(nodesOf[VectorGlobalLimitExec](df).isEmpty)
    val reasons = VectorFallback.reasons(finalPlan(df)).map(_._2)
    assert(reasons.exists(_.contains("is not columnar")), reasons.mkString("; "))
  }

  test("LIMIT ... OFFSET and the configuration switch fall back") {
    checkFallback("SELECT i FROM t LIMIT 5 OFFSET 3", Seq(Collect), "offset 3 not supported")
    withConf(VectorConf.LimitEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT i FROM t LIMIT 5"); d.collect(); d }
      assert(nodesOf[VectorCollectLimitExec](df).isEmpty)
      assert(nodesOf[CollectLimitExec](df).nonEmpty)
    }
  }
}
