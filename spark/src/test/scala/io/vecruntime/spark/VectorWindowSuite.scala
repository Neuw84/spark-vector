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

import io.vecruntime.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.window.{WindowExec, WindowGroupLimitExec}
import org.apache.spark.sql.vector.{VectorFilterExec, VectorProjectExec, VectorWindowExec, VectorWindowGroupLimitExec}

/** Window functions (#58), first layer: row_number, rank, dense_rank over sorted input. */
class VectorWindowSuite extends VectorQuerySuite {
  private val Window = classOf[VectorWindowExec]

  override def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("window/t"))
  }

  private def checkWindow(
      sql: String,
      extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil
  ): org.apache.spark.sql.DataFrame = {
    val df = checkVectorized(sql, Window +: extra)
    assert(nodesOf[WindowExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  test("row_number, rank and dense_rank over partitions with ties and null keys equal Spark") {
    // Ties in the order key (l % 5), a null partition (s is null for one row in ten), a null order key.
    checkWindow("SELECT i, s, l, row_number() OVER (PARTITION BY s ORDER BY l % 5, i) AS rn FROM t")
    checkWindow(
      "SELECT i, s, l, rank() OVER (PARTITION BY s ORDER BY l % 5) AS rk, dense_rank() OVER (PARTITION BY s ORDER BY l % 5) AS dr FROM t"
    )
    checkWindow(
      "SELECT i, l, rank() OVER (PARTITION BY i % 7 ORDER BY nullif(l % 3, 0) DESC NULLS FIRST, i) AS rk FROM t"
    )
    checkWindow("SELECT i, dt, dense_rank() OVER (PARTITION BY b, dt ORDER BY i) AS dr FROM t")
    // A constant order key (Spark's golden window_part3: rank() OVER (ORDER BY length('abc'))) makes every row a peer.
    checkWindow(
      "SELECT i, rank() OVER (ORDER BY length('abc')) AS rk, dense_rank() OVER (PARTITION BY 1 ORDER BY i % 3, 2) AS dr FROM t WHERE i < 100",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
    // Several specs in one query: Spark stacks one Window operator per spec; every one is ours.
    val stacked = checkWindow(
      "SELECT i, row_number() OVER (PARTITION BY i % 7 ORDER BY l, i) AS rn, rank() OVER (PARTITION BY s ORDER BY l) AS rk, dense_rank() OVER (PARTITION BY s ORDER BY l DESC) AS dr FROM t"
    )
    assert(nodesOf[VectorWindowExec](stacked).length === 3, finalPlan(stacked).treeString)
    // Decimal-free key types: strings, dates, booleans, longs, the same walk.
    checkWindow(
      "SELECT s, dt, row_number() OVER (PARTITION BY s ORDER BY dt, l) AS rn FROM t WHERE i < 5000",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
  }

  test("a partition longer than a batch carries the counters across batches") {
    // No PARTITION BY: one partition of 20000 rows, several batches; rank ties every 100 rows.
    val df = checkWindow(
      "SELECT i, row_number() OVER (ORDER BY i) AS rn, rank() OVER (ORDER BY i % 100) AS rk, dense_rank() OVER (ORDER BY i % 100) AS dr FROM t"
    )
    val last = df.orderBy(df("rn").desc).limit(1).collect().head
    assert(last.getInt(1) === 20000, last.toString)
    // The rank of the row ordered first in its peer group is the row number where the group began.
    assert(df.filter("rk = 1").count() === 200 && df.filter("dr = 100").count() === 200)
  }

  test("the chain above the window is ours: a filter on the rank and a projection") {
    // The TPC-DS shape (q44, q49, q67, q70, q86): rank in a subquery, filtered to the top few per group.
    checkWindow(
      "SELECT s, i, rk FROM (SELECT s, i, rank() OVER (PARTITION BY s ORDER BY l DESC) AS rk FROM t) w WHERE rk <= 3",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec], classOf[VectorProjectExec])
    )
    checkWindow(
      "SELECT s, count(*) AS n FROM (SELECT s, row_number() OVER (PARTITION BY s ORDER BY i) AS rn FROM t) w WHERE rn <= 10 GROUP BY s",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
  }

  test("the per-partition top-k Spark inserts under a ranking window is ours in both modes") {
    def checkTopK(sql: String, modes: Int): Unit = {
      val df = checkVectorized(sql, Seq(Window, classOf[VectorWindowGroupLimitExec]))
      assert(
        nodesOf[WindowGroupLimitExec](df).isEmpty && nodesOf[VectorWindowGroupLimitExec](df).length === modes,
        finalPlan(df).treeString
      )
    }
    // Partial below the exchange over our scan-side operators, Final above Spark's sort: two operators.
    checkTopK(
      "SELECT s, i, rk FROM (SELECT s, i, rank() OVER (PARTITION BY s ORDER BY l DESC) AS rk FROM t) w WHERE rk <= 3",
      2
    )
    checkTopK(
      "SELECT s, i FROM (SELECT s, i, row_number() OVER (PARTITION BY i % 7 ORDER BY l, i) AS rn FROM t) w WHERE rn < 5",
      2
    )
    // Ties at the boundary: rank keeps every tied row at k, dense_rank keeps whole peer groups.
    checkTopK(
      "SELECT i, l, rk FROM (SELECT i, l, rank() OVER (PARTITION BY s ORDER BY l % 5) AS rk FROM t) w WHERE rk <= 2",
      2
    )
    checkTopK(
      "SELECT i, l, dr FROM (SELECT i, l, dense_rank() OVER (PARTITION BY s ORDER BY l % 5 DESC) AS dr FROM t) w WHERE dr <= 2",
      2
    )
    // k at Spark's threshold (past it Spark plans no group limit); a null partition key and a null order key.
    checkTopK(
      "SELECT i, rk FROM (SELECT i, rank() OVER (PARTITION BY s ORDER BY nullif(l % 3, 0)) AS rk FROM t) w WHERE rk <= 1000",
      2
    )
    // No PARTITION BY: one group; the limit reached mid-batch drops the rest of every batch (row_number
    // without a partition becomes Spark's top-N instead, so rank is the shape that keeps the group limit).
    checkTopK("SELECT i, rk FROM (SELECT i, rank() OVER (ORDER BY l DESC) AS rk FROM t) w WHERE rk <= 7", 2)
    // Under the threshold Spark plans no group limit at all: the window alone.
    withConf("spark.sql.optimizer.windowGroupLimitThreshold" -> "0") {
      val df = checkVectorized(
        "SELECT s, i, rk FROM (SELECT s, i, rank() OVER (PARTITION BY s ORDER BY l DESC) AS rk FROM t) w WHERE rk <= 3",
        Seq(Window)
      )
      assert(nodesOf[VectorWindowGroupLimitExec](df).isEmpty, finalPlan(df).treeString)
    }
    withConf("spark.vector.exec.window.enabled" -> "false") {
      val df = withPlugin(enabled = true) {
        val d = spark.sql(
          "SELECT s, i, rk FROM (SELECT s, i, rank() OVER (PARTITION BY s ORDER BY l DESC) AS rk FROM t) w WHERE rk <= 3"
        ); d.collect(); d
      }
      assert(
        nodesOf[WindowGroupLimitExec](df).nonEmpty && nodesOf[VectorWindowGroupLimitExec](df).isEmpty,
        finalPlan(df).treeString
      )
    }
  }

  test("whole-partition aggregates equal Spark: sum, avg, count, min, max over every partition shape") {
    // The default frame without ORDER BY is the whole partition; s is null for one row in ten.
    checkWindow(
      "SELECT i, s, sum(l) OVER (PARTITION BY s) AS total, avg(l) OVER (PARTITION BY s) AS mean, count(*) OVER (PARTITION BY s) AS n FROM t"
    )
    checkWindow(
      "SELECT i, min(d) OVER (PARTITION BY i % 7) AS lo, max(d) OVER (PARTITION BY i % 7) AS hi, count(nullif(l % 3, 0)) OVER (PARTITION BY i % 7) AS nn FROM t"
    )
    checkWindow(
      "SELECT i, sum(i) OVER (PARTITION BY b, dt) AS total, avg(d) OVER (PARTITION BY b, dt) AS mean, max(s) OVER (PARTITION BY b, dt) AS last_s FROM t"
    )
    // The explicit whole-partition frame beside an ORDER BY is the same computation.
    checkWindow(
      "SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total FROM t"
    )
    // No PARTITION BY: one partition of 20000 rows across several held batches, released at the end.
    checkWindow("SELECT i, sum(l) OVER () AS total, count(*) OVER () AS n, avg(i) OVER () AS mean FROM t")
    // Partitions of 4000 rows straddle batch boundaries; a batch is held until its last partition ends.
    checkWindow(
      "SELECT i, sum(l) OVER (PARTITION BY i DIV 4000) AS total, min(i) OVER (PARTITION BY i DIV 4000) AS lo FROM t"
    )
    // Empty-input partition (no rows survive the filter) and the chain above: filter, then group by the value.
    checkWindow(
      "SELECT total, count(*) AS n FROM (SELECT i, sum(l) OVER (PARTITION BY i % 5) AS total FROM t WHERE l % 2 = 0) w GROUP BY total",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
    // Ranking and an aggregate in one query are two operators (different specs), both ours.
    val both = checkWindow(
      "SELECT i, rank() OVER (PARTITION BY s ORDER BY l) AS rk, sum(l) OVER (PARTITION BY s) AS total FROM t"
    )
    assert(nodesOf[VectorWindowExec](both).length === 2, finalPlan(both).treeString)
  }

  test("running aggregates equal Spark: the default RANGE frame with ORDER BY and ROWS UNBOUNDED PRECEDING") {
    // The default frame of an ordered aggregate window is RANGE ... CURRENT ROW: peers share the value at the end of their group.
    checkWindow(
      "SELECT i, l, sum(l) OVER (PARTITION BY s ORDER BY l % 5) AS running, count(*) OVER (PARTITION BY s ORDER BY l % 5) AS n FROM t"
    )
    checkWindow(
      "SELECT i, avg(i) OVER (PARTITION BY i % 7 ORDER BY dt) AS mean, min(l) OVER (PARTITION BY i % 7 ORDER BY dt) AS lo, max(s) OVER (PARTITION BY i % 7 ORDER BY dt) AS hi FROM t"
    )
    // ROWS: a prefix over rows, ties or not; nulls in the values and in the order key.
    checkWindow(
      "SELECT i, sum(nullif(l % 3, 0)) OVER (PARTITION BY s ORDER BY l % 5, i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running FROM t"
    )
    checkWindow(
      "SELECT i, count(nullif(l % 3, 0)) OVER (PARTITION BY s ORDER BY nullif(l % 3, 0) ROWS UNBOUNDED PRECEDING) AS nn, max(d) OVER (PARTITION BY s ORDER BY nullif(l % 3, 0) ROWS UNBOUNDED PRECEDING) AS hi FROM t"
    )
    // One partition of 20000 rows, per-row groups across several held batches; sums of doubles too.
    checkWindow(
      "SELECT i, sum(l) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running, sum(d) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS dsum FROM t WHERE d = d AND abs(d) < 1e300"
    )
    // Partitions straddling batches with the RANGE default; then the chain above.
    checkWindow("SELECT i, sum(l) OVER (PARTITION BY i DIV 4000 ORDER BY i % 100) AS running FROM t")
    checkWindow(
      "SELECT s, max(running) AS m FROM (SELECT s, sum(l) OVER (PARTITION BY s ORDER BY i) AS running FROM t WHERE l % 2 = 0) w GROUP BY s",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
    // ANSI: a running bigint sum that overflows raises like Spark; non-ANSI wraps.
    val overflowing =
      "SELECT i, sum(CASE WHEN i = 1 THEN 9223372036854775807L ELSE l END) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running FROM t WHERE i < 10"
    withConf("spark.sql.ansi.enabled" -> "true") {
      val e = intercept[Exception] { withPlugin(enabled = true) { spark.sql(overflowing).collect() } }
      assert(e.getMessage.contains("ARITHMETIC_OVERFLOW") || e.getMessage.contains("overflow"), e.getMessage)
    }
    withConf("spark.sql.ansi.enabled" -> "false") { checkWindow(overflowing) }
  }

  test("offset functions equal Spark: lag, lead, first_value, last_value, nth_value over the held partition") {
    // lag/lead with the default null, an explicit default, offsets past the partition, a negative lag (a lead).
    checkWindow(
      "SELECT i, l, lag(l) OVER (PARTITION BY s ORDER BY i) AS prev, lead(l) OVER (PARTITION BY s ORDER BY i) AS nxt FROM t"
    )
    checkWindow(
      "SELECT i, lag(s, 2, 'none') OVER (PARTITION BY i % 7 ORDER BY l, i) AS p2, lead(dt, 3) OVER (PARTITION BY i % 7 ORDER BY l, i) AS n3, lag(d, -1) OVER (PARTITION BY i % 7 ORDER BY l, i) AS back FROM t"
    )
    checkWindow(
      "SELECT i, lead(b, 1, true) OVER (PARTITION BY s ORDER BY l % 5, i) AS nb, lag(nullif(l % 3, 0), 1, -1L) OVER (PARTITION BY s ORDER BY l % 5, i) AS pn FROM t"
    )
    // first_value / last_value: whole partition (no ORDER BY), the RANGE default (peers share the group's last row), ROWS.
    checkWindow(
      "SELECT i, first_value(l) OVER (PARTITION BY s) AS f, last_value(s) OVER (PARTITION BY i % 7) AS lst FROM t"
    )
    checkWindow(
      "SELECT i, l, first_value(l) OVER (PARTITION BY s ORDER BY l % 5) AS f, last_value(l) OVER (PARTITION BY s ORDER BY l % 5) AS peer_last, last_value(l) OVER (PARTITION BY s ORDER BY l % 5, i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS self FROM t"
    )
    checkWindow(
      "SELECT i, first(dt) OVER (PARTITION BY s ORDER BY i) AS f, last(nullif(l % 3, 0)) OVER (PARTITION BY s ORDER BY i) AS l3 FROM t"
    )
    // nth_value: null while the frame has fewer than n rows; whole-partition and running frames.
    checkWindow(
      "SELECT i, nth_value(l, 3) OVER (PARTITION BY s ORDER BY i) AS third, nth_value(s, 2) OVER (PARTITION BY i % 7 ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS second FROM t"
    )
    // Offsets across several held batches: one 20000-row partition, lag over 5000 rows and a lead near the end.
    checkWindow(
      "SELECT i, lag(l, 5000) OVER (ORDER BY i) AS far_back, lead(l, 7) OVER (ORDER BY i) AS ahead, first_value(i) OVER (ORDER BY i) AS f, last_value(i) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS lst FROM t"
    )
    // Partitions straddling batches, several functions with different offsets in one operator, the chain above.
    checkWindow(
      "SELECT s, count(*) AS n FROM (SELECT s, l, lag(l, 1) OVER (PARTITION BY i DIV 4000 ORDER BY i) AS p1, lag(l, 2) OVER (PARTITION BY i DIV 4000 ORDER BY i) AS p2 FROM t WHERE l % 2 = 0) w WHERE p1 > p2 GROUP BY s",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
  }

  test("percent_rank, cume_dist and ntile equal Spark, and ranking functions beside offsets share one operator") {
    // Ties share values; a null partition key; single-row partitions (i % 7 with a narrow filter).
    checkWindow(
      "SELECT i, l, percent_rank() OVER (PARTITION BY s ORDER BY l % 5) AS pr, cume_dist() OVER (PARTITION BY s ORDER BY l % 5) AS cd FROM t"
    )
    checkWindow(
      "SELECT i, ntile(4) OVER (PARTITION BY s ORDER BY l, i) AS q4, ntile(1) OVER (PARTITION BY s ORDER BY l, i) AS q1, ntile(1000) OVER (PARTITION BY i % 7 ORDER BY i) AS many FROM t"
    )
    checkWindow(
      "SELECT i, percent_rank() OVER (PARTITION BY i ORDER BY l) AS single, cume_dist() OVER (PARTITION BY i ORDER BY l) AS one, ntile(3) OVER (PARTITION BY i ORDER BY l) AS b FROM t WHERE i < 50",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec])
    )
    // One 20000-row partition across batches; buckets that do not divide the size.
    checkWindow(
      "SELECT i, percent_rank() OVER (ORDER BY i % 100) AS pr, cume_dist() OVER (ORDER BY i % 100) AS cd, ntile(7) OVER (ORDER BY i) AS b FROM t"
    )
    // Ranking functions beside offsets in one spec: the held-partition path computes all of them.
    checkWindow(
      "SELECT i, row_number() OVER (PARTITION BY s ORDER BY l % 5, i) AS rn, rank() OVER (PARTITION BY s ORDER BY l % 5, i) AS rk, dense_rank() OVER (PARTITION BY s ORDER BY l % 5, i) AS dr, lag(l) OVER (PARTITION BY s ORDER BY l % 5, i) AS prev, percent_rank() OVER (PARTITION BY s ORDER BY l % 5, i) AS pr FROM t"
    )
    checkWindow(
      "SELECT i, dense_rank() OVER (PARTITION BY i % 7 ORDER BY l % 3) AS dr, rank() OVER (PARTITION BY i % 7 ORDER BY l % 3) AS rk, first_value(l) OVER (PARTITION BY i % 7 ORDER BY l % 3) AS f FROM t"
    )
  }

  test("sliding frames equal Spark: sum, count, avg, min, max over ROWS BETWEEN with every bound shape") {
    // Bounded both sides, preceding-only, following-only (empty frames near the partition end), and the suffix.
    checkWindow(
      "SELECT i, l, sum(l) OVER (PARTITION BY s ORDER BY l % 5, i ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS moving, count(*) OVER (PARTITION BY s ORDER BY l % 5, i ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS n FROM t"
    )
    checkWindow(
      "SELECT i, sum(i) OVER (PARTITION BY i % 7 ORDER BY l, i ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS before, avg(d) OVER (PARTITION BY i % 7 ORDER BY l, i ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) AS after FROM t WHERE d = d AND abs(d) < 1e300"
    )
    checkWindow(
      "SELECT i, min(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS suffix_min, max(s) OVER (PARTITION BY i % 7 ORDER BY i ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS around, count(nullif(l % 3, 0)) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS nn FROM t"
    )
    // UNBOUNDED PRECEDING with a following bound: advanced in order like Spark; doubles bit-identical; the whole-partition ROWS form.
    checkWindow(
      "SELECT i, sum(d) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND 2 FOLLOWING) AS ahead, avg(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) AS mean, sum(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total FROM t WHERE d = d AND abs(d) < 1e300"
    )
    // A 20000-row partition with bounded frames across batch boundaries, and ties (ROWS frames ignore peers).
    checkWindow(
      "SELECT i, sum(l) OVER (ORDER BY i ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) AS window7, max(i) OVER (ORDER BY l % 5, i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS m FROM t"
    )
    // Sliding aggregates beside a running one and an offset in one operator (all take the held-partition path).
    checkWindow(
      "SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving, sum(l) OVER (PARTITION BY s ORDER BY i) AS running, lag(l) OVER (PARTITION BY s ORDER BY i) AS prev, count(*) OVER (PARTITION BY s) AS n FROM t"
    )
    // ANSI: a sliding bigint sum that overflows raises like Spark; non-ANSI wraps.
    val overflowing =
      "SELECT i, sum(CASE WHEN i = 1 THEN 9223372036854775807L ELSE l END) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS moving FROM t WHERE i < 10"
    withConf("spark.sql.ansi.enabled" -> "true") {
      val e = intercept[Exception] { withPlugin(enabled = true) { spark.sql(overflowing).collect() } }
      assert(e.getMessage.contains("ARITHMETIC_OVERFLOW") || e.getMessage.contains("overflow"), e.getMessage)
    }
    withConf("spark.sql.ansi.enabled" -> "false") { checkWindow(overflowing) }
  }

  test("other window functions and frames fall back with a reason; the operator can be disabled") {
    // RANGE frames with value offsets, sliding frames of functions without a form here, and two frame kinds in one operator.
    checkFallback(
      "SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) AS moving FROM t",
      Seq(Window),
      "RANGE frames with value offsets"
    )
    checkFallback(
      "SELECT i, stddev(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS sd FROM t",
      Seq(Window),
      "over a sliding frame not supported"
    )
    checkWindow(
      "SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i) AS running, first(l) OVER (PARTITION BY s ORDER BY i) AS f FROM t"
    )
    checkFallback(
      "SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i) AS running, sum(cast(l AS decimal(12,2))) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS dec FROM t",
      Seq(Window),
      "over decimals in a sliding frame not supported"
    )
    // A decimal window aggregate over a whole partition or a running frame runs since #259 (the buffer is the 128-bit lane).
    checkVectorized(
      "SELECT i, sum(cast(l AS decimal(12,2))) OVER (PARTITION BY s) AS total, avg(cast(l AS decimal(12,2))) OVER (PARTITION BY s ORDER BY i) AS running FROM t",
      Seq(Window)
    )
    checkFallback(
      "SELECT i, approx_count_distinct(l) OVER (PARTITION BY s) AS n FROM t",
      Seq(Window),
      "window aggregate approx_count_distinct:"
    )
    checkWindow(
      "SELECT i, lag(l) OVER (PARTITION BY s ORDER BY i) AS previous, sum(l) OVER (PARTITION BY s ORDER BY i) AS running FROM t"
    )
    checkFallback(
      "SELECT i, lag(l) OVER (PARTITION BY s ORDER BY i) AS previous, stddev(l) OVER (PARTITION BY s ORDER BY i) AS sd FROM t",
      Seq(Window),
      "running frame for stddev not supported"
    )
    checkFallback(
      "SELECT i, lag(l, 1) IGNORE NULLS OVER (PARTITION BY s ORDER BY i) AS previous FROM t",
      Seq(Window),
      "IGNORE NULLS not supported"
    )
    checkFallback(
      "SELECT i, first_value(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS f FROM t",
      Seq(Window),
      "first_value over frame"
    )
    checkFallback(
      "SELECT i, rank() OVER (PARTITION BY d ORDER BY i) AS rk FROM t",
      Seq(Window),
      "double keys not supported"
    )
    // A ranking function beside an aggregate in the same spec: one operator of ours (the held-partition path computes both);
    // beside an aggregate it cannot compute, the whole operator stays Spark's with that reason.
    checkWindow(
      "SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk, sum(l) OVER (PARTITION BY s ORDER BY i) AS running FROM t"
    )
    checkWindow(
      "SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk, sum(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total FROM t"
    )
    checkFallback(
      "SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk, stddev(l) OVER (PARTITION BY s ORDER BY i) AS sd FROM t",
      Seq(Window),
      "running frame for stddev not supported"
    )
    withConf("spark.vector.exec.window.enabled" -> "false") {
      val df = withPlugin(enabled = true) {
        val d = spark.sql("SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk FROM t"); d.collect(); d
      }
      assert(nodesOf[WindowExec](df).nonEmpty && nodesOf[VectorWindowExec](df).isEmpty, finalPlan(df).treeString)
    }
  }

  test("decimal AVG over a running ROWS frame on a native decimal column equals Spark") {
    // A NATIVE decimal Parquet column (not a cast expression): Spark rewrites its window avg to
    // `cast(avg(UnscaledValue(d)) OVER (...) / scale as decimal(p,s))`, nesting the WindowExpression
    // inside a Cast(Divide(...)) the planner now sees through (it computes the window column, then
    // projects the wrapper). Decimals of several precisions, a null partition key and null values, a
    // running ROWS frame and the RANGE default; the results must equal Spark's exactly.
    val path = newTempPath("window/wd")
    spark
      .range(0, 4000)
      .selectExpr(
        "cast(id as int) as i",
        "if(id % 10 = 0, null, concat('g', id % 40)) as g",
        "cast(if(id % 13 = 0, null, (id % 1000) / 7.0) as decimal(10,2)) as d10",
        "cast(if(id % 17 = 0, null, (id % 100000) / 3.0) as decimal(18,4)) as d18",
        "cast(if(id % 7 = 0, null, (id % 500) / 11.0) as decimal(6,2)) as d6"
      )
      .repartition(3)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("wd")

    // The gap: decimal avg over a running ROWS UNBOUNDED PRECEDING .. CURRENT ROW frame, decimal(10,2).
    // The `cast(avg(UnscaledValue(d)) OVER (...) / scale as decimal)` wrapper is part of the window
    // expression itself (Spark keeps it in the Window node), so the vector window operator produces
    // the decimal column directly -- no separate projection is required.
    checkWindow(
      "SELECT i, g, d10, avg(d10) OVER (PARTITION BY g ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_avg FROM wd"
    )
    // Higher precision, and beside a decimal running SUM (a bare window expression) in one operator.
    checkWindow(
      "SELECT i, avg(d18) OVER (PARTITION BY g ORDER BY i ROWS UNBOUNDED PRECEDING) AS a18, sum(d18) OVER (PARTITION BY g ORDER BY i ROWS UNBOUNDED PRECEDING) AS s18 FROM wd"
    )
    // decimal(6,2), the RANGE default frame with ORDER BY (peers share the group's value).
    checkWindow("SELECT i, avg(d6) OVER (PARTITION BY g ORDER BY i) AS a6 FROM wd")
    // Whole-partition decimal avg (no ORDER BY) already worked; kept as the wrapper's identity edge.
    checkWindow("SELECT i, avg(d10) OVER (PARTITION BY g) AS a FROM wd")
    // No PARTITION BY: one partition across several held batches; a running decimal avg per row.
    checkWindow(
      "SELECT i, avg(d10) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running FROM wd"
    )
    // A sliding decimal frame is still refused. On a native decimal column Spark's avg rewrite makes
    // the window input `UnscaledValue(d)` -- not a bare column -- so the sliding path refuses it there
    // (the sliding kernels read a column input); the window falls back to Spark.
    checkFallback(
      "SELECT i, avg(d10) OVER (PARTITION BY g ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS moving FROM wd",
      Seq(Window),
      "is not a column"
    )
  }
}
