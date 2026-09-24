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
import org.apache.spark.sql.execution.{LocalTableScanExec, SampleExec}
import org.apache.spark.sql.vector.{
  VectorFilterExec,
  VectorHashAggregateExec,
  VectorLocalTableScanExec,
  VectorProjectExec,
  VectorSampleExec
}

/**
 * Sampling must return exactly Spark's rows for a seed: the operator drives Spark's own Bernoulli
 * sampler over the live rows in order, so every shape below is compared row for row, not by count.
 */
class VectorSampleSuite extends VectorQuerySuite {

  private val Sample = classOf[VectorSampleExec]
  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]
  private val LocalScan = classOf[VectorLocalTableScanExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("sample/t"))
  }

  test("TABLESAMPLE without replacement returns Spark's rows for a seed") {
    for (percent <- Seq(1, 30, 50, 95); seed <- Seq(0L, 42L, 123456789L)) {
      val df = checkVectorized(s"SELECT i, s FROM t TABLESAMPLE ($percent PERCENT) REPEATABLE ($seed)", Seq(Sample))
      assert(nodesOf[SampleExec](df).isEmpty, "Spark's SampleExec should be gone")
      val n = df.count()
      assert(n > 0 && n < 20000, s"$percent percent kept $n of 20000 rows")
    }
    // The bucket form is the same Bernoulli window, shifted.
    checkVectorized("SELECT i FROM t TABLESAMPLE (BUCKET 2 OUT OF 5) REPEATABLE (11)", Seq(Sample))
    // Nulls in the sampled columns travel with their rows.
    checkVectorized("SELECT i, l, d, dt FROM t TABLESAMPLE (40 PERCENT) REPEATABLE (5)", Seq(Sample))
  }

  test("a sample composes with our filter, projection and aggregate on either side") {
    // A filter beneath the sample: the sampler draws only for rows the filter kept, as Spark's does.
    checkVectorized(
      "SELECT i FROM (SELECT i FROM t WHERE i % 3 = 0) TABLESAMPLE (50 PERCENT) REPEATABLE (7) x",
      Seq(Sample, Filter)
    )
    // A filter above the sample consumes the forwarded selection.
    checkVectorized(
      "SELECT i FROM (SELECT i FROM t TABLESAMPLE (60 PERCENT) REPEATABLE (3)) x WHERE i > 100",
      Seq(Sample, Filter)
    )
    // Sparse samples are compacted, dense ones forwarded: both end in the same rows.
    checkVectorized("SELECT i + 1 AS j, s FROM t TABLESAMPLE (2 PERCENT) REPEATABLE (99)", Seq(Sample, Project))
    checkVectorized("SELECT i + 1 AS j, s FROM t TABLESAMPLE (98 PERCENT) REPEATABLE (99)", Seq(Sample, Project))
    checkVectorized(
      "SELECT s, count(*) AS n, sum(i) AS si FROM t TABLESAMPLE (25 PERCENT) REPEATABLE (2024) GROUP BY s",
      Seq(Sample, Agg)
    )
    checkVectorized("SELECT count(*) AS n FROM t TABLESAMPLE (0.5 PERCENT) REPEATABLE (1)", Seq(Sample, Agg))
  }

  test("sampling with replacement falls back; the switch turns the operator off") {
    spark.table("t").sample(withReplacement = true, 0.3, 42L).createOrReplaceTempView("t_poisson")
    checkFallback("SELECT i FROM t_poisson", Seq(Sample), "with replacement")
    withConf(VectorConf.SampleEnabled -> "false") {
      val df = withPlugin(enabled = true) {
        val d = spark.sql("SELECT i FROM t TABLESAMPLE (30 PERCENT) REPEATABLE (42)"); d.collect(); d
      }
      assert(nodesOf[VectorSampleExec](df).isEmpty && nodesOf[SampleExec](df).nonEmpty, finalPlan(df).treeString)
    }
  }

  test("a VALUES relation runs our operators when the local table scan is enabled (off by default)") {
    val sql =
      "SELECT a, count(*) AS n, max(b) AS mb FROM VALUES (1, 'x'), (2, 'y'), (1, 'z'), (3, NULL) AS v(a, b) WHERE a < 3 GROUP BY a"
    // The optimizer normally evaluates filters and projections over a literal relation on the driver
    // (ConvertToLocalRelation); Spark's own operator tests exclude that rule, and so does this one.
    val keepOperators =
      "spark.sql.optimizer.excludedRules" -> "org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation"
    withConf(VectorConf.LocalTableScanEnabled -> "true", keepOperators) {
      val df = checkVectorized(sql, Seq(LocalScan, Filter, Agg))
      assert(nodesOf[LocalTableScanExec](df).isEmpty, finalPlan(df).treeString)
      checkVectorized("SELECT a * 2 AS d, b FROM VALUES (1, 'x'), (2, 'y') AS v(a, b)", Seq(LocalScan, Project))
      checkVectorized("SELECT a FROM VALUES (1), (2) AS v(a) WHERE a > 5", Seq(LocalScan, Filter))
      checkVectorized(
        "SELECT a, b FROM (SELECT * FROM VALUES (1, 'x'), (2, NULL), (3, 'z') AS v(a, b)) TABLESAMPLE (50 PERCENT) REPEATABLE (4) w",
        Seq(LocalScan, Sample)
      )
    }
    withConf(keepOperators) {
      val df = withPlugin(enabled = true) { val d = spark.sql(sql); d.collect(); d }
      assert(
        nodesOf[VectorLocalTableScanExec](df).isEmpty && nodesOf[LocalTableScanExec](df).nonEmpty,
        finalPlan(df).treeString
      )
    }
  }
}
