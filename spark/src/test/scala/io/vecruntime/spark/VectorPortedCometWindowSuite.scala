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
package io.vecruntime.spark

import io.vecruntime.spark.test.VectorQuerySuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.types._
import org.apache.spark.sql.vecruntime.VectorWindowExec

/**
 * SQL correctness coverage ported from DataFusion Comet's `CometWindowExecSuite`, adapted to
 * spark-vector's plugin-on/plugin-off comparison model: every window query is run twice on one
 * session with `spark.vector.enabled` toggled, the rows compared, and a `VectorWindowExec` asserted
 * in the accelerated plan (a frame or function we don't accelerate asserts fallback with its reason).
 *
 * This is the window slice of the SQL-coverage survey. It complements the hand-written
 * `VectorWindowSuite` (which pins the ranking walk, the held-partition aggregate/offset paths, the
 * sliding frames, the `WindowGroupLimit` top-k and the fallback reasons in depth) by adding Comet's
 * data-driven window matrix in the exact shapes its suite checks over a small `(a, b, c)` table:
 * COUNT(*) without a frame, SUM/AVG with PARTITION BY (and ORDER BY), decimal AVG and SUM over a
 * running `ROWS UNBOUNDED PRECEDING ... CURRENT ROW` frame (ours since #259), MIN/MAX with ORDER BY,
 * the ROWS-between frame family (`UNBOUNDED PRECEDING AND CURRENT ROW`, `CURRENT ROW AND UNBOUNDED
 * FOLLOWING`, `n PRECEDING AND m FOLLOWING`, `n PRECEDING AND CURRENT ROW`, `CURRENT ROW AND n
 * FOLLOWING`, `UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`), ROW_NUMBER/RANK/DENSE_RANK/PERCENT_RANK/
 * NTILE with PARTITION BY and ORDER BY, LAG/LEAD (default offset, offset+default), FIRST_VALUE,
 * LAST_VALUE over a ROWS frame and NTH_VALUE.
 *
 * It also pins the shapes Comet documents as *not* accelerated or as errors, which are contracts
 * `docs/operators.md` also records: a RANGE frame with a value offset falls back, a RANGE frame whose
 * lower bound is FOLLOWING falls back, `IGNORE NULLS` on lag/lead falls back, and a non-literal
 * lag/lead offset or default is a Spark *analysis error* (not a fallback) that both engines raise.
 * Where a shape uncovered an engine bug the case is `ignore`d with a one-line reason and a minimal
 * repro in the PR, per the task's rule not to fix engine code in a coverage change.
 *
 * The table mirrors Comet's fixture: `w(a, b, c)` with small integer partition/order/value columns,
 * ties in the order key, a null partition, a null value, plus a decimal measure `d` for the decimal
 * running-frame cases -- written as Parquet so the scan is columnar.
 */
class VectorPortedCometWindowSuite extends VectorQuerySuite {

  private val Window = classOf[VectorWindowExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    // Comet's window fixture shape: several partitions (`a`), an order key with ties (`b`), a value
    // (`c`) with nulls, a null partition, and a decimal(10,2) measure for the running decimal frames.
    val rows = (0 until 4000).map { id =>
      val a = id % 7
      val bnull = if (id % 53 == 0) null else Int.box((id % 11))
      val cnull = if (id % 17 == 0) null else Int.box(id % 100)
      val part: Integer = if (id % 101 == 0) null else Int.box(a)
      val dec: java.math.BigDecimal =
        if (id % 29 == 0) null else java.math.BigDecimal.valueOf((id % 500) - 250).movePointLeft(2)
      Row(part, bnull, cnull, dec)
    }.toList
    val schema = StructType(
      Seq(
        StructField("a", IntegerType, nullable = true),
        StructField("b", IntegerType, nullable = true),
        StructField("c", IntegerType, nullable = true),
        StructField("d", DecimalType(10, 2), nullable = true)
      )
    )
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows, 4), schema)
    val path = newTempPath("pwin/w")
    df.write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("w")
  }

  private def checkWindow(
      sql: String,
      extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil
  ): org.apache.spark.sql.DataFrame = {
    val df = checkVectorized(sql, Window +: extra)
    assert(nodesOf[WindowExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  // ---------------------------------------------------------------------------
  // Whole-partition and running aggregates (Comet's "window: simple COUNT(*)
  // without frame", "SUM with PARTITION BY", "AVG with PARTITION BY and ORDER BY").
  // ---------------------------------------------------------------------------

  test("count(*) without a frame, and sum/avg/min/max with PARTITION BY (and ORDER BY)") {
    checkWindow("SELECT a, b, c, count(*) OVER (PARTITION BY a) AS cnt FROM w")
    checkWindow("SELECT a, b, c, sum(c) OVER (PARTITION BY a) AS s FROM w")
    checkWindow("SELECT a, b, c, avg(c) OVER (PARTITION BY a ORDER BY b) AS av FROM w")
    checkWindow(
      "SELECT a, b, c, min(c) OVER (PARTITION BY a ORDER BY b) AS lo, max(c) OVER (PARTITION BY a ORDER BY b) AS hi FROM w"
    )
    // A null partition key is its own group; a null value is skipped by sum/avg/min/max, counted only by count(*).
    checkWindow("SELECT a, count(*) OVER (PARTITION BY a) AS n, count(c) OVER (PARTITION BY a) AS nn FROM w")
  }

  // ---------------------------------------------------------------------------
  // Decimal running frames (Comet's "window: decimal AVG/SUM with PARTITION BY
  // and ORDER BY"): ours since #259 over whole-partition and running frames.
  // ---------------------------------------------------------------------------

  test("decimal sum/avg over a whole partition and a running ROWS frame") {
    // Whole-partition decimal sum, running sum (ROWS UNBOUNDED PRECEDING .. CURRENT ROW) and the
    // RANGE default are ours since #259. Each expression is pinned on its own so a fallback in one
    // shape is visible and does not mask the others. (Decimal running AVG: the case below, ours
    // since #513.)
    checkWindow("SELECT a, d, sum(d) OVER (PARTITION BY a) AS total FROM w")
    checkWindow(
      "SELECT a, b, d, sum(d) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_sum FROM w"
    )
    // The RANGE default (peers share the group's value) over the decimal measure.
    checkWindow("SELECT a, b, d, sum(d) OVER (PARTITION BY a ORDER BY b) AS running FROM w")
  }

  // A decimal AVG over a running ROWS frame on a *native* decimal Parquet column: Spark rewrites it to
  // `cast(avg(UnscaledValue(d)) OVER (...) / scale as decimal(p,s))`, which nests the WindowExpression
  // inside a Cast(Divide(...)). This fell back until #513 taught VectorWindowPlanner to see through the
  // scalar wrapper and compute it over the window column.
  test("decimal AVG over a running ROWS frame on a native decimal column") {
    checkWindow(
      "SELECT a, b, d, avg(d) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_avg FROM w"
    )
  }

  // ---------------------------------------------------------------------------
  // The ROWS-between frame family (Comet's several "window: ... with ROWS
  // BETWEEN ..." cases): every bound shape, sum/count/avg/min/max.
  // ---------------------------------------------------------------------------

  test("sum/count/avg/min/max over every ROWS BETWEEN bound shape") {
    checkWindow(
      "SELECT a, b, c, count(*) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cnt FROM w"
    )
    checkWindow(
      "SELECT a, b, c, sum(c) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS suffix FROM w"
    )
    checkWindow(
      "SELECT a, b, c, avg(c) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving FROM w"
    )
    checkWindow(
      "SELECT a, b, c, sum(c) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS trailing FROM w"
    )
    checkWindow(
      "SELECT a, b, c, count(*) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS ahead FROM w"
    )
    checkWindow(
      "SELECT a, b, c, max(c) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS whole FROM w"
    )
    // Preceding-only and following-only windows (empty frames near the partition edge).
    checkWindow(
      "SELECT a, b, c, sum(c) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS before, avg(c) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS after FROM w"
    )
    checkWindow(
      "SELECT a, b, c, min(c) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS around FROM w"
    )
  }

  // ---------------------------------------------------------------------------
  // Ranking functions (Comet's ROW_NUMBER / RANK / DENSE_RANK / PERCENT_RANK /
  // NTILE "with PARTITION BY and ORDER BY").
  // ---------------------------------------------------------------------------

  test("row_number, rank, dense_rank, percent_rank, ntile with PARTITION BY and ORDER BY") {
    checkWindow("SELECT a, b, row_number() OVER (PARTITION BY a ORDER BY b, c) AS rn FROM w")
    checkWindow("SELECT a, b, rank() OVER (PARTITION BY a ORDER BY b) AS rk FROM w")
    checkWindow("SELECT a, b, dense_rank() OVER (PARTITION BY a ORDER BY b) AS dr FROM w")
    checkWindow("SELECT a, b, percent_rank() OVER (PARTITION BY a ORDER BY b) AS pr FROM w")
    checkWindow("SELECT a, b, ntile(4) OVER (PARTITION BY a ORDER BY b, c) AS q FROM w")
    checkWindow("SELECT a, b, cume_dist() OVER (PARTITION BY a ORDER BY b) AS cd FROM w")
    // Rank and dense_rank over ASC and DESC order keys with ties (Comet toggles the order direction).
    checkWindow(
      "SELECT a, b, rank() OVER (PARTITION BY a ORDER BY b DESC) AS rk, dense_rank() OVER (PARTITION BY a ORDER BY b DESC) AS dr FROM w"
    )
  }

  // ---------------------------------------------------------------------------
  // Offset functions (Comet's LAG / LEAD / FIRST_VALUE / LAST_VALUE / NTH_VALUE).
  // ---------------------------------------------------------------------------

  test("lag/lead default offset, offset with default, first_value, last_value over ROWS, nth_value") {
    checkWindow("SELECT a, b, c, lag(c) OVER (PARTITION BY a ORDER BY b, c) AS lag_c FROM w")
    checkWindow("SELECT a, b, c, lag(c, 2, -1) OVER (PARTITION BY a ORDER BY b, c) AS lag2 FROM w")
    checkWindow("SELECT a, b, c, lead(c) OVER (PARTITION BY a ORDER BY b, c) AS lead_c FROM w")
    checkWindow("SELECT a, b, c, lead(c, 2, -1) OVER (PARTITION BY a ORDER BY b, c) AS lead2 FROM w")
    checkWindow("SELECT a, b, c, first_value(c) OVER (PARTITION BY a ORDER BY b) AS f FROM w")
    checkWindow(
      "SELECT a, b, c, last_value(c) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_c FROM w"
    )
    checkWindow(
      "SELECT a, b, c, nth_value(c, 2) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS nth_c FROM w"
    )
    // lag/lead with the default (null) beyond the partition (Comet's "return the default value if the
    // offset row does not exist"): a literal default, and the default null.
    checkWindow(
      "SELECT a, b, lag(b, 100, 321) OVER (PARTITION BY a ORDER BY b, c) AS lg, lead(b, 100, 321) OVER (PARTITION BY a ORDER BY b, c) AS ld FROM w"
    )
  }

  // ---------------------------------------------------------------------------
  // Frames and functions that fall back with a reason (Comet's "rangeBetween
  // FOLLOWING lower bound falls back", "Unsupported window expression falls
  // back", the RANGE value-offset frame, IGNORE NULLS).
  // ---------------------------------------------------------------------------

  test("RANGE value-offset frames, IGNORE NULLS and unsupported functions fall back with a reason") {
    // A RANGE frame with a value offset needs per-row order-key value comparisons.
    checkFallback(
      "SELECT a, b, sum(c) OVER (PARTITION BY a ORDER BY b RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) AS moving FROM w",
      Seq(Window),
      "RANGE frames with value offsets"
    )
    // A RANGE frame whose lower bound is FOLLOWING (Comet's "rangeBetween FOLLOWING lower bound").
    checkFallback(
      "SELECT a, b, sum(c) OVER (PARTITION BY a ORDER BY b RANGE BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS ahead FROM w",
      Seq(Window),
      "RANGE frames with value offsets"
    )
    // IGNORE NULLS on lag/lead.
    checkFallback(
      "SELECT a, b, c, lag(c) IGNORE NULLS OVER (PARTITION BY a ORDER BY b) AS lag_c FROM w",
      Seq(Window),
      "IGNORE NULLS not supported"
    )
    // A statistical aggregate over a sliding/running frame (Comet's "Unsupported window expression").
    checkFallback(
      "SELECT a, b, c, stddev(c) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS sd FROM w",
      Seq(Window),
      "over a sliding frame not supported"
    )
    // A sliding decimal frame is still refused (whole-partition and running decimal frames run since #259).
    checkFallback(
      "SELECT a, b, d, sum(d) OVER (PARTITION BY a ORDER BY b, c ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS dec FROM w",
      Seq(Window),
      "over decimals in a sliding frame not supported"
    )
  }

  // ---------------------------------------------------------------------------
  // Analysis errors both engines raise (Comet's rules: nth_value needs ORDER BY;
  // a non-literal lag offset/default is an analysis error, not a fallback).
  // ---------------------------------------------------------------------------

  test("a non-literal lag offset is a Spark analysis error, raised the same with the plugin on") {
    // The offset of lag/lead must be a foldable integer: a column offset is a Spark *analysis* error
    // (not a fallback), and the plugin does not change that -- it is raised before planning.
    val e1 = intercept[Exception] {
      withPlugin(enabled = true) {
        spark.sql("SELECT a, lag(b, c) OVER (PARTITION BY a ORDER BY b) AS lg FROM w").collect()
      }
    }
    assert(
      e1.getMessage.toLowerCase.contains("offset") || e1.getMessage.toLowerCase.contains("foldable") ||
        e1.getMessage.contains("must be") || e1.getMessage.contains("literal") ||
        e1.getMessage.contains("non-foldable") || e1.getMessage.contains("integer"),
      e1.getMessage
    )
    // nth_value requires an ORDER BY: without one Spark raises at analysis (both engines).
    val e2 = intercept[Exception] {
      withPlugin(enabled = true) {
        spark.sql("SELECT a, nth_value(c, 2) OVER (PARTITION BY a) AS nth_c FROM w").collect()
      }
    }
    assert(
      e2.getMessage.toLowerCase.contains("order") || e2.getMessage.contains("requires") ||
        e2.getMessage.contains("window frame") || e2.getMessage.contains("nth_value"),
      e2.getMessage
    )
  }
}
