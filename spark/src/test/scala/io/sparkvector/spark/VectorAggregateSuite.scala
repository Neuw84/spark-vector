package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec}

class VectorAggregateSuite extends VectorQuerySuite {

  private val Agg = classOf[VectorHashAggregateExec]
  private val Filter = classOf[VectorFilterExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("agg/t"))
    TestTables.createLineitem(spark, newTempPath("agg/lineitem"))
  }

  test("count, sum, min, max, avg over doubles, ints and dates") {
    checkVectorized("SELECT count(*), count(d), sum(d2), min(d2), max(d2), avg(d2) FROM t", Seq(Agg))
    checkVectorized("SELECT count(i), sum(i), min(i), max(i), avg(i) FROM t", Seq(Agg))
    checkVectorized("SELECT min(dt), max(dt), count(dt) FROM t", Seq(Agg))
    checkVectorized("SELECT count(l), min(l), max(l), avg(l) FROM t", Seq(Agg))
  }

  test("NaN and infinities follow Spark's ordering in min and max") {
    // d contains NaN, +Inf and -Inf: max is NaN, min is -Inf.
    val df = checkVectorized("SELECT max(d), min(d) FROM t", Seq(Agg))
    val row = df.collect().head
    assert(row.getDouble(0).isNaN)
    assert(row.getDouble(1) === Double.NegativeInfinity)
  }

  test("aggregates over expressions and filtered input") {
    checkVectorized("SELECT sum(d2 * 2.0), avg(d2 - 1.0), count(d2 * d2) FROM t WHERE i > 1000", Seq(Filter, Agg))
    checkVectorized("SELECT sum(d2 * (1.0 - d2) * (1.0 + d2)) FROM t WHERE d2 > 0.25", Seq(Filter, Agg))
  }

  test("all-null and empty inputs produce Spark's results") {
    checkVectorized("SELECT sum(d), count(d), min(d), max(d), avg(d) FROM t WHERE d IS NULL", Seq(Filter, Agg))
    checkVectorized("SELECT sum(d2), count(*), count(d2), min(i), avg(d2) FROM t WHERE i < 0", Seq(Filter, Agg))
    checkVectorized("SELECT count(*) FROM t WHERE i < 0", Seq(Filter, Agg))
  }

  test("count(*) over a scan with no projected columns") {
    checkVectorized("SELECT count(*) FROM t", Seq(Agg))
  }

  test("final aggregation merges the partial buffers over the shuffle") {
    // Ungrouped: one partial row per partition, one final row; avg is sum / count of the merged buffers.
    val df = checkVectorized("SELECT sum(d2), count(*), count(l), min(d), max(i), avg(d2), avg(i) FROM t WHERE i >= 0", Seq(Agg))
    assert(nodesOf[HashAggregateExec](df).isEmpty, df.queryExecution.executedPlan.treeString)
    val aggs = nodesOf[VectorHashAggregateExec](df)
    assert(aggs.size === 2 && aggs.count(_.isFinal) === 1, df.queryExecution.executedPlan.treeString)
    // Grouped, with result expressions that are not plain attributes.
    val grouped = checkVectorized("SELECT s, count(*) AS n, sum(d) + 1.0 AS s1, avg(d2) * 2.0 AS a2, max(dt) FROM t GROUP BY s", Seq(Agg))
    assert(nodesOf[HashAggregateExec](grouped).isEmpty)
    assert(nodesOf[VectorHashAggregateExec](grouped).count(_.isFinal) === 1)
    // Empty input: Spark's initial buffers (count 0, sum null, avg null).
    checkVectorized("SELECT count(*), sum(d), avg(d), min(i) FROM t WHERE i < 0", Seq(Agg))
    // (With AQE on, Spark replaces an empty grouped aggregate by EmptyRelation before we run.)
    withConf("spark.sql.adaptive.enabled" -> "false") {
      checkVectorized("SELECT s, count(*) FROM t WHERE i < 0 GROUP BY s", Seq(Agg))
    }
  }

  test("final aggregation stays in Spark when disabled by configuration") {
    withConf(VectorConf.FinalAggregateEnabled -> "false") {
      val df = checkVectorized("SELECT sum(d2), count(*) FROM t WHERE i >= 0", Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).nonEmpty, "expected Spark's Final HashAggregateExec")
      assert(nodesOf[VectorHashAggregateExec](df).size === 1)
    }
  }

  test("sum of bigint falls back in ANSI mode and is vectorized in legacy mode") {
    checkFallback("SELECT sum(l) FROM t", Seq(Agg), "ANSI sum of bigint")
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT sum(l), sum(i) FROM t", Seq(Agg))
    }
  }

  test("unsupported aggregates fall back") {
    // Spark rewrites DISTINCT into a grouped partial aggregate, which is not vectorized yet.
    checkFallback("SELECT count(DISTINCT i) FROM t", Seq(Agg), "not supported")
    checkFallback("SELECT first(d2) FROM t", Seq(Agg), "unsupported aggregate function")
    // FILTER applies while updating, so only the Partial stage falls back; the Final merge is ours.
    val filtered = checkVectorized("SELECT sum(d2) FILTER (WHERE i > 5), count(*) FROM t", Seq(Agg))
    assert(nodesOf[HashAggregateExec](filtered).exists(_.aggregateExpressions.forall(_.mode == org.apache.spark.sql.catalyst.expressions.aggregate.Partial)))
    assert(nodesOf[VectorHashAggregateExec](filtered).forall(_.isFinal))
    checkFallback("SELECT min(b) FROM t", Seq(Agg), "not supported")
    // min over strings is planned by Spark as SortAggregateExec, which we never touch.
    val df = withPlugin(enabled = true)(spark.sql("SELECT min(s) FROM t"))
    df.collect()
    assert(nodesOf[VectorHashAggregateExec](df).isEmpty)
  }

  test("aggregate conversion can be disabled by configuration") {
    withConf(VectorConf.AggregateEnabled -> "false") {
      val df = withPlugin(enabled = true)(spark.sql("SELECT sum(d2) FROM t"))
      df.collect()
      assert(nodesOf[VectorHashAggregateExec](df).isEmpty)
    }
  }

  test("grouped aggregation with few groups uses the mask path") {
    // s has 49 distinct values plus null: at most 50 groups, below the low-cardinality threshold.
    checkVectorized("SELECT s, count(*), count(d), sum(d2), avg(d2), min(i), max(l), min(dt) FROM t GROUP BY s", Seq(Agg))
    checkVectorized("SELECT b, sum(d2), count(*) FROM t GROUP BY b", Seq(Agg))
    checkVectorized("SELECT s, b, sum(d2 * 2.0), avg(d2) FROM t WHERE i > 100 GROUP BY s, b", Seq(Filter, Agg))
  }

  test("grouped aggregation with many groups uses the scatter path") {
    checkVectorized("SELECT i, sum(d2), count(*), max(d2) FROM t GROUP BY i", Seq(Agg))
    checkVectorized("SELECT l, count(*), sum(d2), min(i) FROM t GROUP BY l", Seq(Agg)) // nullable bigint key
    checkVectorized("SELECT dt, count(*), avg(d2) FROM t GROUP BY dt", Seq(Agg)) // 730 date keys
  }

  test("grouping keys with nulls and NaN values in aggregates") {
    checkVectorized("SELECT s, max(d), min(d), sum(d) FROM t GROUP BY s", Seq(Agg))
    checkVectorized("SELECT l, max(d) FROM t WHERE d IS NOT NULL GROUP BY l", Seq(Filter, Agg))
  }

  test("grouped aggregation over empty input emits no rows") {
    // With AQE the empty partial output makes Spark replace the stage by EmptyRelation, so only
    // check the plan shape with AQE off.
    checkVectorized("SELECT s, count(*) FROM t WHERE i < 0 GROUP BY s", Seq.empty)
    withConf("spark.sql.adaptive.enabled" -> "false") {
      val df = checkVectorized("SELECT s, count(*) FROM t WHERE i < 0 GROUP BY s", Seq(Filter, Agg))
      assert(df.count() === 0)
    }
  }

  test("double grouping keys fall back") {
    checkFallback("SELECT d2, count(*) FROM t GROUP BY d2", Seq(Agg), "grouping key type double not supported")
  }

  test("TPC-H Q1 end to end") {
    val df = checkVectorized(TestTables.TpchQ1, Seq(Filter, Agg), tolerance = 1e-9)
    assert(nodesOf[HashAggregateExec](df).isEmpty, "both aggregation stages should be vectorized")
    val rows = df.collect()
    assert(rows.length === 3, rows.mkString("\n")) // synthetic data yields (A,F), (N,O), (R,F)
    info(s"Q1 rows:\n${rows.mkString("\n")}\n${df.queryExecution.executedPlan.treeString}")
  }

  test("TPC-H Q6 end to end") {
    val df = checkVectorized(TestTables.TpchQ6, Seq(Filter, Agg), tolerance = 1e-9)
    val revenue = df.collect().head.getDouble(0)
    assert(revenue > 0.0)
    info(s"Q6 revenue = $revenue\n${df.queryExecution.executedPlan.treeString}")
  }
}
