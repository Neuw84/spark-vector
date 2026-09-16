package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.SortExec
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorSortExec}

/**
 * VectorSortExec against Spark's SortExec. Result sets are compared as usual and, on top, the
 * sequence of key values per partition must be identical to Spark's (ties in non-key columns may
 * legitimately differ in order, so only the keys are compared positionally).
 */
class VectorSortSuite extends VectorQuerySuite {

  private val Sort = classOf[VectorSortExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("sort/t"))
    TestTables.createLineitem(spark, newTempPath("sort/lineitem"))
  }

  /** Per-partition sequences of the first `numKeys` columns, with doubles rounded like assertRowsEqual. */
  private def keySequences(sql: String, numKeys: Int, enabled: Boolean): Seq[Seq[Seq[Any]]] =
    withPlugin(enabled) {
      spark.sql(sql).rdd.glom().collect().toSeq.map(_.toSeq.map(r => (0 until numKeys).map { c =>
        r.get(c) match {
          case d: Double if d.isNaN => "NaN"
          case d: Double => f"$d%.6f"
          case v => v
        }
      }))
    }

  /**
   * Runs `sql` (whose leading `numKeys` columns are the sort keys) with and without the plugin,
   * checks the rows, the operator and that every partition's key sequence matches Spark's.
   */
  private def checkSorted(sql: String, numKeys: Int): Unit = {
    checkVectorized(sql, Seq(Sort))
    val expected = keySequences(sql, numKeys, enabled = false)
    val actual = keySequences(sql, numKeys, enabled = true)
    assert(actual.length === expected.length, s"partition count for: $sql")
    expected.zip(actual).zipWithIndex.foreach { case ((e, a), p) =>
      assert(a === e, s"key order differs in partition $p for: $sql")
    }
  }

  test("local sort over the vectorized scan, every key type, both directions") {
    checkSorted("SELECT i, l, d, s FROM t SORT BY i DESC", 1)
    checkSorted("SELECT l, i FROM t SORT BY l", 1)
    checkSorted("SELECT l, i FROM t SORT BY l DESC NULLS FIRST", 1)
    checkSorted("SELECT d, i FROM t SORT BY d", 1) // NaN, infinities and nulls
    checkSorted("SELECT d, i FROM t SORT BY d DESC NULLS LAST", 1)
    checkSorted("SELECT dt, i FROM t SORT BY dt DESC", 1)
    checkSorted("SELECT b, i FROM t SORT BY b", 1)
    checkSorted("SELECT s, i FROM t SORT BY s", 1) // dictionary-encoded strings with nulls
    checkSorted("SELECT s, i FROM t SORT BY s DESC NULLS FIRST", 1)
  }

  test("multiple keys with mixed directions and null orderings") {
    checkSorted("SELECT s, b, d, i FROM t SORT BY s NULLS LAST, b DESC, d", 3)
    checkSorted("SELECT b, l, i FROM t SORT BY b, l DESC NULLS FIRST", 2)
  }

  test("sort over a filter forwarding a selection, and a computed key") {
    val df = checkVectorized("SELECT d, i FROM t WHERE i > 100 SORT BY d DESC", Seq(Sort, classOf[VectorFilterExec]))
    assert(df.count() === 20000 - 101)
    checkSorted("SELECT d, i FROM t WHERE i > 100 SORT BY d DESC", 1)
    // A computed key is evaluated with the kernels and kept alongside the output columns (double
    // arithmetic: integer arithmetic in ANSI mode is a documented fallback).
    checkVectorized("SELECT i, d FROM t SORT BY d * 2.0 + 1 DESC", Seq(Sort))
    checkVectorized("SELECT i, l FROM t SORT BY CAST(l AS DOUBLE) / 3", Seq(Sort))
    checkFallback("SELECT i, l FROM t SORT BY i * 2 + 1", Seq(Sort), "ANSI integer arithmetic")
  }

  test("global sort over a Spark row shuffle stays with Spark") {
    // With shuffle.partitions=4 the global ORDER BY needs a range exchange, which is row based.
    checkFallback("SELECT i, s FROM t ORDER BY s, i", Seq(Sort), "is not columnar")
    checkVectorized("SELECT i, s FROM t ORDER BY s, i", Seq()) // still correct, just Spark's sort
  }

  test("unsupported sort keys fall back with a reason") {
    checkFallback("SELECT i, s FROM t SORT BY s LIKE 'a%'", Seq(Sort), "StartsWith")
  }

  test("TPC-H Q1 shape: sort above our Final aggregate, single partition") {
    // One shuffle partition makes the exchange a no-op distribution wise but it is still Spark's
    // row shuffle, so the global Sort stays with Spark; the sort within partitions variant is ours.
    withConf("spark.sql.shuffle.partitions" -> "1") {
      val local = TestTables.TpchQ1.replace("ORDER BY", "SORT BY")
      // Final aggregate -> Sort: the aggregate's output is columnar, so the sort is ours.
      val df = checkVectorized(local, Seq(Sort, classOf[VectorHashAggregateExec]))
      val rows: Array[Row] = df.collect()
      assert(rows.map(r => (r.getString(0), r.getString(1))).toSeq === rows.map(r => (r.getString(0), r.getString(1))).sorted.toSeq)
      assert(rows.length === 3, "the synthetic lineitem has three flag/status groups")
    }
  }

  test("sort can be disabled") {
    withConf(VectorConf.SortEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT i FROM t SORT BY i"); d.collect(); d }
      assert(nodesOf[VectorSortExec](df).isEmpty)
      assert(nodesOf[SortExec](df).nonEmpty)
    }
  }
}
