package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.{SortExec, TakeOrderedAndProjectExec}
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorSortExec, VectorTakeOrderedAndProjectExec}

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
    // A computed key is evaluated with the kernels and kept alongside the output columns; integer
    // keys are overflow-checked in ANSI mode (the issue's own example).
    checkVectorized("SELECT i, d FROM t SORT BY d * 2.0 + 1 DESC", Seq(Sort))
    checkVectorized("SELECT i, l FROM t SORT BY CAST(l AS DOUBLE) / 3", Seq(Sort))
    checkVectorized("SELECT i, l FROM t SORT BY i * 2 + 1", Seq(Sort))
    // Expression keys of other families: a string built per row, a CASE, a cast, several at once.
    checkVectorized("SELECT i, s FROM t SORT BY concat(s, '-', CAST(i % 7 AS STRING)) DESC, i", Seq(Sort))
    checkVectorized("SELECT i, d2 FROM t SORT BY CASE WHEN d2 > 1 THEN -d2 ELSE d2 END, CAST(i AS BIGINT) * 3 DESC", Seq(Sort))
  }

  test("global sort over a Spark row shuffle stays with Spark") {
    // With shuffle.partitions=4 the global ORDER BY needs a range exchange, which is row based.
    checkFallback("SELECT i, s FROM t ORDER BY s, i", Seq(Sort), "is not columnar")
    checkVectorized("SELECT i, s FROM t ORDER BY s, i", Seq()) // still correct, just Spark's sort
  }

  test("unsupported sort keys fall back with a reason") {
    checkFallback("SELECT i, s FROM t SORT BY s LIKE 'a%b%c'", Seq(Sort), "Like")
    checkVectorized("SELECT i, s FROM t SORT BY s LIKE 's1%', i", Seq(Sort)) // the simplified shape compiles
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

  private val TopN = classOf[VectorTakeOrderedAndProjectExec]

  /** Rows must come back in exactly Spark's order: the keys are compared positionally. */
  private def checkOrdered(sql: String, key: Row => Any, extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil): Array[Row] = {
    val expected = withPlugin(enabled = false)(spark.sql(sql).collect())
    val df = checkVectorized(sql, TopN +: extra)
    val actual = df.collect()
    assert(actual.map(key).toSeq === expected.map(key).toSeq, s"key order differs for: $sql")
    assert(nodesOf[TakeOrderedAndProjectExec](df).isEmpty, "Spark's operator should be gone")
    actual
  }

  test("ORDER BY ... LIMIT over the scan: every key type, both directions, nulls, projections") {
    // t has 3 Parquet partitions, so the per-partition top-N and the final merge both run.
    val top = checkOrdered("SELECT i, s FROM t ORDER BY i DESC LIMIT 10", _.getInt(0))
    assert(top.length === 10 && top.head.getInt(0) === 19999)
    checkOrdered("SELECT i FROM t ORDER BY i LIMIT 1", _.getInt(0))
    checkOrdered("SELECT l, i FROM t ORDER BY l NULLS FIRST, i LIMIT 25", r => (if (r.isNullAt(0)) None else Some(r.getLong(0)), r.getInt(1)))
    checkOrdered("SELECT d, i FROM t ORDER BY d DESC NULLS LAST, i LIMIT 25", r => (if (r.isNullAt(0)) "null" else r.getDouble(0).toString, r.getInt(1))) // NaN sorts greatest; compared as text since NaN != NaN
    checkOrdered("SELECT dt, i FROM t ORDER BY dt, i DESC LIMIT 7", r => (r.getDate(0), r.getInt(1)))
    checkOrdered("SELECT s, i FROM t ORDER BY s DESC, i LIMIT 12", r => (r.getString(0), r.getInt(1)))
    // Projection on top: computed columns and a reordered subset, key not in the output.
    checkOrdered("SELECT i * 2 AS twice, s FROM t ORDER BY i LIMIT 5", _.getInt(0))
    checkOrdered("SELECT s FROM t WHERE i > 100 ORDER BY i LIMIT 5", _.getString(0), Seq(classOf[VectorFilterExec]))
    checkOrdered("SELECT i, b, d2 FROM t ORDER BY i LIMIT 3", _.getInt(0))
    // Limit beyond the partition sizes and beyond the data.
    checkOrdered("SELECT i FROM t WHERE i < 50 ORDER BY i DESC LIMIT 1000", _.getInt(0))
    val none = checkOrdered("SELECT i FROM t WHERE i < 0 ORDER BY i LIMIT 10", _.getInt(0))
    assert(none.isEmpty)
    // A computed key.
    checkOrdered("SELECT i FROM t ORDER BY i * -1 LIMIT 4", _.getInt(0))
  }

  test("TPC-H Q3/Q10 shape: ORDER BY ... LIMIT above our Final aggregate") {
    val q = """SELECT l_returnflag, l_linestatus, sum(l_extendedprice) AS revenue, count(*) AS c
              |FROM lineitem WHERE l_shipdate < DATE '1996-01-01'
              |GROUP BY l_returnflag, l_linestatus ORDER BY revenue DESC LIMIT 2""".stripMargin
    val rows = checkOrdered(q, r => (r.getString(0), r.getString(1)), Seq(classOf[VectorHashAggregateExec]))
    assert(rows.length === 2)
    assert(rows(0).getDouble(2) >= rows(1).getDouble(2))
    // Q10 orders on an aggregate and limits to 20; here every group survives the limit.
    checkOrdered("SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem GROUP BY l_orderkey ORDER BY q DESC, l_orderkey LIMIT 20", _.getLong(0), Seq(classOf[VectorHashAggregateExec]))
  }

  test("ORDER BY ... LIMIT falls back with a reason: offset, unsupported key, disabled") {
    checkFallback("SELECT i FROM t ORDER BY i LIMIT 5 OFFSET 2", Seq(TopN), "offset 2 not supported")
    checkFallback("SELECT i FROM t ORDER BY s LIKE 'a%b%c' LIMIT 5", Seq(TopN), "Like")
    withConf(VectorConf.TakeOrderedEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT i FROM t ORDER BY i LIMIT 5"); d.collect(); d }
      assert(nodesOf[VectorTakeOrderedAndProjectExec](df).isEmpty)
      assert(nodesOf[TakeOrderedAndProjectExec](df).nonEmpty)
    }
  }
}
