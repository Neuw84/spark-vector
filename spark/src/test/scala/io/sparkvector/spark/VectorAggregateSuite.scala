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

  test("final aggregation stays in Spark and consumes the partial buffers") {
    val df = checkVectorized("SELECT sum(d2), count(*) FROM t WHERE i >= 0", Seq(Agg))
    val sparkAggs = nodesOf[HashAggregateExec](df)
    assert(sparkAggs.nonEmpty, "expected Spark's Final HashAggregateExec")
    assert(nodesOf[VectorHashAggregateExec](df).size === 1)
  }

  test("sum of bigint falls back in ANSI mode and is vectorized in legacy mode") {
    checkFallback("SELECT sum(l) FROM t", Seq(Agg), "ANSI sum of bigint")
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT sum(l), sum(i) FROM t", Seq(Agg))
    }
  }

  test("unsupported aggregates fall back") {
    // Spark rewrites DISTINCT into a grouped partial aggregate, which is not vectorized yet.
    checkFallback("SELECT count(DISTINCT i) FROM t", Seq(Agg), "grouped aggregation")
    checkFallback("SELECT first(d2) FROM t", Seq(Agg), "unsupported aggregate function")
    checkFallback("SELECT sum(d2) FILTER (WHERE i > 5) FROM t", Seq(Agg), "FILTER")
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

  test("TPC-H Q6 end to end") {
    val df = checkVectorized(TestTables.TpchQ6, Seq(Filter, Agg), tolerance = 1e-9)
    val revenue = df.collect().head.getDouble(0)
    assert(revenue > 0.0)
    info(s"Q6 revenue = $revenue\n${df.queryExecution.executedPlan.treeString}")
  }
}
