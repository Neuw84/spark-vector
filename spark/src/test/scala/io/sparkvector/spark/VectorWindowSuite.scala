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

  test("other window functions and frames fall back with a reason; the operator can be disabled") {
    checkFallback("SELECT i, sum(l) OVER (PARTITION BY s) AS total FROM t", Seq(Window), "window aggregate sum over a frame not supported")
    checkFallback("SELECT i, avg(l) OVER (PARTITION BY s ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running FROM t", Seq(Window), "window aggregate avg")
    checkFallback("SELECT i, lag(l) OVER (PARTITION BY s ORDER BY i) AS previous FROM t", Seq(Window), "window function lag not supported")
    checkFallback("SELECT i, percent_rank() OVER (PARTITION BY s ORDER BY i) AS pr FROM t", Seq(Window), "window function percent_rank not supported")
    checkFallback("SELECT i, rank() OVER (PARTITION BY d ORDER BY i) AS rk FROM t", Seq(Window), "double keys not supported")
    // A ranking function beside an unsupported one in the same spec keeps the whole operator Spark's.
    checkFallback("SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk, sum(l) OVER (PARTITION BY s ORDER BY i) AS running FROM t", Seq(Window), "window aggregate sum")
    withConf("spark.vector.exec.window.enabled" -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT i, rank() OVER (PARTITION BY s ORDER BY i) AS rk FROM t"); d.collect(); d }
      assert(nodesOf[WindowExec](df).nonEmpty && nodesOf[VectorWindowExec](df).isEmpty, finalPlan(df).treeString)
    }
  }
}
