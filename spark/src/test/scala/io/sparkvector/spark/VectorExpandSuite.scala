package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.ExpandExec
import org.apache.spark.sql.vector.{VectorExpandExec, VectorFilterExec, VectorHashAggregateExec}

/** ExpandExec: grouping sets and the distinct-aggregate rewrite, row for row against Spark. */
class VectorExpandSuite extends VectorQuerySuite {

  private val Expand = classOf[VectorExpandExec]
  private val Filter = classOf[VectorFilterExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("expand/t"))
    TestTables.createLineitem(spark, newTempPath("expand/lineitem"))
  }

  test("ROLLUP, CUBE and GROUPING SETS with grouping() and grouping_id()") {
    // s has nulls (i % 10 = 0) and b is a boolean key: the expand's own nulls must stay
    // distinguishable from the data's, which grouping()/grouping_id() make visible.
    val rollup = checkVectorized(
      "SELECT s, b, count(*) AS c, sum(l) AS sl, grouping(s) AS gs, grouping(b) AS gb, grouping_id() AS gid FROM t GROUP BY ROLLUP(s, b)",
      Seq(Expand, Agg))
    assert(nodesOf[ExpandExec](rollup).isEmpty, "Spark's ExpandExec should be gone")
    checkVectorized("SELECT s, b, count(*) AS c, min(d) AS md, max(i) AS mi FROM t GROUP BY CUBE(s, b)", Seq(Expand, Agg))
    checkVectorized(
      "SELECT s, dt, count(*) AS c, grouping_id() AS gid FROM t WHERE i > 100 GROUP BY GROUPING SETS ((s), (dt), (s, dt), ())",
      Seq(Expand, Filter, Agg))
    checkVectorized("SELECT year(dt) AS y, s, sum(i) AS si FROM t GROUP BY ROLLUP(year(dt), s)", Seq(Expand, Agg))
    // Grouping sets whose totals include the data's null group as well as the expand's null.
    checkVectorized("SELECT s, count(*) AS c, grouping(s) AS gs FROM t GROUP BY s WITH ROLLUP", Seq(Expand, Agg))
  }

  test("TPC-H-shaped rollups over lineitem") {
    checkVectorized(
      "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS q, sum(l_extendedprice * (1 - l_discount)) AS rev, count(*) AS c FROM lineitem WHERE l_shipdate < DATE '1996-01-01' GROUP BY ROLLUP(l_returnflag, l_linestatus)",
      Seq(Expand, Filter, Agg), tolerance = 1e-6)
    checkVectorized(
      "SELECT l_returnflag, year(l_shipdate) AS y, avg(l_quantity) AS aq, grouping_id() AS gid FROM lineitem GROUP BY CUBE(l_returnflag, year(l_shipdate))",
      Seq(Expand, Agg), tolerance = 1e-9)
  }

  test("count(distinct) lowers to an expand; the expand is ours even where the aggregate is not yet") {
    // RewriteDistinctAggregates: Expand with null literals per distinct group, then a two-level
    // aggregate whose first level is a keys-plus-count shape. The expand converts; whatever the
    // aggregate does today, the results must match Spark.
    val df = checkVectorized("SELECT s, count(DISTINCT i) AS di, count(DISTINCT l) AS dl, sum(i) AS si FROM t GROUP BY s", Seq(Expand))
    assert(nodesOf[ExpandExec](df).isEmpty)
  }

  test("expand can be disabled") {
    withConf(VectorConf.ExpandEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT s, count(*) FROM t GROUP BY ROLLUP(s)"); d.collect(); d }
      assert(nodesOf[VectorExpandExec](df).isEmpty)
      assert(nodesOf[ExpandExec](df).nonEmpty)
    }
  }
}
