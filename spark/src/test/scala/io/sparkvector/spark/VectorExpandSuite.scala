package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.ExpandExec
import org.apache.spark.sql.vector.{VectorExpandExec, VectorFilterExec, VectorHashAggregateExec, VectorRollupExec}

/** ExpandExec: grouping sets and the distinct-aggregate rewrite, row for row against Spark. */
class VectorExpandSuite extends VectorQuerySuite {

  private val Expand = classOf[VectorExpandExec]
  private val Filter = classOf[VectorFilterExec]
  private val Agg = classOf[VectorHashAggregateExec]
  private val Rollup = classOf[VectorRollupExec] // a ROLLUP's Expand and partial aggregate become the chain (#383)

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
      Seq(Rollup, Agg))
    assert(nodesOf[ExpandExec](rollup).isEmpty, "Spark's ExpandExec should be gone")
    checkVectorized("SELECT s, b, count(*) AS c, min(d) AS md, max(i) AS mi FROM t GROUP BY CUBE(s, b)", Seq(Expand, Agg))
    checkVectorized(
      "SELECT s, dt, count(*) AS c, grouping_id() AS gid FROM t WHERE i > 100 GROUP BY GROUPING SETS ((s), (dt), (s, dt), ())",
      Seq(Expand, Filter, Agg))
    checkVectorized("SELECT year(dt) AS y, s, sum(i) AS si FROM t GROUP BY ROLLUP(year(dt), s)", Seq(Rollup, Agg))
    // Grouping sets whose totals include the data's null group as well as the expand's null.
    checkVectorized("SELECT s, count(*) AS c, grouping(s) AS gs FROM t GROUP BY s WITH ROLLUP", Seq(Rollup, Agg))
  }

  test("TPC-H-shaped rollups over lineitem") {
    checkVectorized(
      "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS q, sum(l_extendedprice * (1 - l_discount)) AS rev, count(*) AS c FROM lineitem WHERE l_shipdate < DATE '1996-01-01' GROUP BY ROLLUP(l_returnflag, l_linestatus)",
      Seq(Rollup, Filter, Agg), tolerance = 1e-6)
    checkVectorized(
      "SELECT l_returnflag, year(l_shipdate) AS y, avg(l_quantity) AS aq, grouping_id() AS gid FROM lineitem GROUP BY CUBE(l_returnflag, year(l_shipdate))",
      Seq(Expand, Agg), tolerance = 1e-9)
  }

  test("#383: a rollup aggregates the finest grouping once and rolls the partials up") {
    // The partial over the Expand becomes one chain operator: the finest-set partial, then a merge per coarser set.
    def rollups(df: org.apache.spark.sql.DataFrame) = nodesOf[org.apache.spark.sql.vector.VectorRollupExec](df)
    val rollup = checkVectorized(
      "SELECT s, b, count(*) AS c, sum(l) AS sl, avg(d2) AS ad, min(i) AS mi, grouping_id() AS gid FROM t GROUP BY ROLLUP(s, b)", Seq(Agg))
    assert(rollups(rollup).size === 1 && rollups(rollup).head.levels.size === 3, rollup.queryExecution.executedPlan.treeString)
    // A FILTER clause is applied by the finest-set partial; the merges carry none.
    checkVectorized("SELECT s, b, count(*) FILTER (WHERE i % 2 = 0) AS ce, sum(l) AS sl FROM t GROUP BY ROLLUP(s, b)", Seq(Agg))
    // An expression key is aliased into the project below, so it is still a column reference here.
    checkVectorized("SELECT year(dt) AS y, s, sum(i) AS si, count(*) AS c FROM t GROUP BY ROLLUP(year(dt), s)", Seq(Agg))
    // Nested GROUPING SETS are a rollup too; a CUBE is not nested and keeps Spark's plan.
    assert(rollups(checkVectorized("SELECT s, dt, count(*) AS c FROM t GROUP BY GROUPING SETS ((s, dt), (s), ())", Seq(Agg))).size === 1)
    assert(rollups(checkVectorized("SELECT s, b, count(*) AS c, sum(l) AS sl FROM t GROUP BY CUBE(s, b)", Seq(Expand, Agg))).isEmpty)
    // The distinct rewrite also uses an Expand, but nulls the aggregate inputs per projection: no rewrite, same answer.
    assert(rollups(checkVectorized("SELECT s, count(DISTINCT i) AS di, sum(l) AS sl FROM t GROUP BY s", Seq(Agg))).isEmpty)
    // The switch.
    withConf(io.sparkvector.spark.VectorConf.RollupRewriteEnabled -> "false") {
      val off = checkVectorized("SELECT s, b, count(*) AS c, sum(l) AS sl FROM t GROUP BY ROLLUP(s, b)", Seq(Expand, Agg))
      assert(rollups(off).isEmpty)
    }
  }


  test("count(distinct) lowers to an expand; the expand and every aggregate stage are ours") {
    // RewriteDistinctAggregates: Expand with null literals per distinct group, then a keys-only
    // aggregate and a filtered one (see VectorAggregateSuite). Nothing of Spark's is left.
    val df = checkVectorized("SELECT s, count(DISTINCT i) AS di, count(DISTINCT l) AS dl, sum(i) AS si FROM t GROUP BY s", Seq(Expand))
    assert(nodesOf[ExpandExec](df).isEmpty)
    assert(nodesOf[org.apache.spark.sql.execution.aggregate.HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
  }

  test("expand can be disabled") {
    withConf(VectorConf.ExpandEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT s, count(*) FROM t GROUP BY ROLLUP(s)"); d.collect(); d }
      assert(nodesOf[VectorExpandExec](df).isEmpty)
      assert(nodesOf[ExpandExec](df).nonEmpty)
    }
  }
}
