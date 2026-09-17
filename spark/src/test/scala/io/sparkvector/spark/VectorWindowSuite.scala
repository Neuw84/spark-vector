package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.vector.{VectorFilterExec, VectorProjectExec, VectorWindowExec}

/** Window functions (#58), first layer: row_number, rank, dense_rank over sorted input. */
class VectorWindowSuite extends VectorQuerySuite {
  private val Window = classOf[VectorWindowExec]

  override def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("window/t"))
  }

  private def checkWindow(sql: String, extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil): org.apache.spark.sql.DataFrame = {
    val df = checkVectorized(sql, Window +: extra)
    assert(nodesOf[WindowExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  test("row_number, rank and dense_rank over partitions with ties and null keys equal Spark") {
    // Ties in the order key (l % 5), a null partition (s is null for one row in ten), a null order key.
    checkWindow("SELECT i, s, l, row_number() OVER (PARTITION BY s ORDER BY l % 5, i) AS rn FROM t")
    checkWindow("SELECT i, s, l, rank() OVER (PARTITION BY s ORDER BY l % 5) AS rk, dense_rank() OVER (PARTITION BY s ORDER BY l % 5) AS dr FROM t")
    checkWindow("SELECT i, l, rank() OVER (PARTITION BY i % 7 ORDER BY nullif(l % 3, 0) DESC NULLS FIRST, i) AS rk FROM t")
    checkWindow("SELECT i, dt, dense_rank() OVER (PARTITION BY b, dt ORDER BY i) AS dr FROM t")
    // A constant order key (Spark's golden window_part3: rank() OVER (ORDER BY length('abc'))) makes every row a peer.
    checkWindow("SELECT i, rank() OVER (ORDER BY length('abc')) AS rk, dense_rank() OVER (PARTITION BY 1 ORDER BY i % 3, 2) AS dr FROM t WHERE i < 100", Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec]))
    // Several specs in one query: Spark stacks one Window operator per spec; every one is ours.
    val stacked = checkWindow("SELECT i, row_number() OVER (PARTITION BY i % 7 ORDER BY l, i) AS rn, rank() OVER (PARTITION BY s ORDER BY l) AS rk, dense_rank() OVER (PARTITION BY s ORDER BY l DESC) AS dr FROM t")
    assert(nodesOf[VectorWindowExec](stacked).length === 3, finalPlan(stacked).treeString)
    // Decimal-free key types: strings, dates, booleans, longs, the same walk.
    checkWindow("SELECT s, dt, row_number() OVER (PARTITION BY s ORDER BY dt, l) AS rn FROM t WHERE i < 5000", Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec]))
  }

  test("a partition longer than a batch carries the counters across batches") {
    // No PARTITION BY: one partition of 20000 rows, several batches; rank ties every 100 rows.
    val df = checkWindow("SELECT i, row_number() OVER (ORDER BY i) AS rn, rank() OVER (ORDER BY i % 100) AS rk, dense_rank() OVER (ORDER BY i % 100) AS dr FROM t")
    val last = df.orderBy(df("rn").desc).limit(1).collect().head
    assert(last.getInt(1) === 20000, last.toString)
    // The rank of the row ordered first in its peer group is the row number where the group began.
    assert(df.filter("rk = 1").count() === 200 && df.filter("dr = 100").count() === 200)
  }

  test("the chain above the window is ours: a filter on the rank and a projection") {
    // The TPC-DS shape (q44, q49, q67, q70, q86): rank in a subquery, filtered to the top few per group.
    checkWindow(
      "SELECT s, i, rk FROM (SELECT s, i, rank() OVER (PARTITION BY s ORDER BY l DESC) AS rk FROM t) w WHERE rk <= 3",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec], classOf[VectorProjectExec]))
    checkWindow(
      "SELECT s, count(*) AS n FROM (SELECT s, row_number() OVER (PARTITION BY s ORDER BY i) AS rn FROM t) w WHERE rn <= 10 GROUP BY s",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec]))
  }

  test("whole-partition aggregates equal Spark: sum, avg, count, min, max over every partition shape") {
    // The default frame without ORDER BY is the whole partition; s is null for one row in ten.
    checkWindow("SELECT i, s, sum(l) OVER (PARTITION BY s) AS total, avg(l) OVER (PARTITION BY s) AS mean, count(*) OVER (PARTITION BY s) AS n FROM t")
    checkWindow("SELECT i, min(d) OVER (PARTITION BY i % 7) AS lo, max(d) OVER (PARTITION BY i % 7) AS hi, count(nullif(l % 3, 0)) OVER (PARTITION BY i % 7) AS nn FROM t")
    checkWindow("SELECT i, sum(i) OVER (PARTITION BY b, dt) AS total, avg(d) OVER (PARTITION BY b, dt) AS mean, max(s) OVER (PARTITION BY b, dt) AS last_s FROM t")
    // The explicit whole-partition frame beside an ORDER BY is the same computation.
    checkWindow("SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total FROM t")
    // No PARTITION BY: one partition of 20000 rows across several held batches, released at the end.
    checkWindow("SELECT i, sum(l) OVER () AS total, count(*) OVER () AS n, avg(i) OVER () AS mean FROM t")
    // Partitions of 4000 rows straddle batch boundaries; a batch is held until its last partition ends.
    checkWindow("SELECT i, sum(l) OVER (PARTITION BY i DIV 4000) AS total, min(i) OVER (PARTITION BY i DIV 4000) AS lo FROM t")
    // Empty-input partition (no rows survive the filter) and the chain above: filter, then group by the value.
    checkWindow(
      "SELECT total, count(*) AS n FROM (SELECT i, sum(l) OVER (PARTITION BY i % 5) AS total FROM t WHERE l % 2 = 0) w GROUP BY total",
      Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]](classOf[VectorFilterExec]))
    // Ranking and an aggregate in one query are two operators (different specs), both ours.
    val both = checkWindow("SELECT i, rank() OVER (PARTITION BY s ORDER BY l) AS rk, sum(l) OVER (PARTITION BY s) AS total FROM t")
    assert(nodesOf[VectorWindowExec](both).length === 2, finalPlan(both).treeString)
  }

  test("other window functions and frames fall back with a reason; the operator can be disabled") {
    checkFallback("SELECT i, avg(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running FROM t", Seq(Window), "window aggregate avg over frame")
    // The default frame with an ORDER BY is RANGE ... CURRENT ROW: a running aggregate, not a whole-partition one.
    checkFallback("SELECT i, sum(l) OVER (PARTITION BY s ORDER BY i) AS running FROM t", Seq(Window), "window aggregate sum over frame")
    checkFallback("SELECT i, sum(cast(l AS decimal(12,2))) OVER (PARTITION BY s) AS total FROM t", Seq(Window), "window aggregate sum over decimals not supported")
    checkFallback("SELECT i, approx_count_distinct(l) OVER (PARTITION BY s) AS n FROM t", Seq(Window), "window aggregate approx_count_distinct:")
    checkFallback("SELECT i, lag(l) OVER (PARTITION BY s ORDER BY i) AS previous FROM t", Seq(Window), "window function lag not supported")
    checkFallback("SELECT i, percent_rank() OVER (PARTITION BY s ORDER BY i) AS pr FROM t", Seq(Window), "window function percent_rank not supported")
    checkFallback("SELECT i, rank() OVER (PARTITION BY d ORDER BY i) AS rk FROM t", Seq(Window), "double keys not supported")
    // A ranking function beside an aggregate in the same spec keeps the whole operator Spark's.
    checkFallback("SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk, sum(l) OVER (PARTITION BY s ORDER BY i) AS running FROM t", Seq(Window), "window aggregate sum over frame")
    checkFallback("SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk, sum(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total FROM t", Seq(Window), "window aggregate sum over a frame not supported")
    withConf("spark.vector.exec.window.enabled" -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk FROM t"); d.collect(); d }
      assert(nodesOf[WindowExec](df).nonEmpty && nodesOf[VectorWindowExec](df).isEmpty, finalPlan(df).treeString)
    }
  }
}
