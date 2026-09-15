package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.FilterExec
import org.apache.spark.sql.vector.VectorFilterExec

class VectorFilterSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("filter/t"))
  }

  test("int comparison against literal") {
    val df = checkVectorized("SELECT * FROM t WHERE i > 500", Seq(Filter))
    assert(df.count() === 20000 - 501)
  }

  test("all six operators on int, long, double and date") {
    Seq("=", "<", "<=", ">", ">=", "!=").foreach { op =>
      checkVectorized(s"SELECT i FROM t WHERE i $op 777", Seq(Filter))
      checkVectorized(s"SELECT i, l FROM t WHERE l $op 30000", Seq(Filter))
      checkVectorized(s"SELECT i, d FROM t WHERE d $op 50.5", Seq(Filter))
      checkVectorized(s"SELECT i, dt FROM t WHERE dt $op DATE '2020-06-15'", Seq(Filter))
    }
  }

  test("literal on the left flips the operator") {
    checkVectorized("SELECT i FROM t WHERE 1000 > i", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE 1000 <= i AND 2000 >= i", Seq(Filter))
  }

  test("column versus column comparisons including NaN-safe doubles") {
    checkVectorized("SELECT i, d, d2 FROM t WHERE d < d2", Seq(Filter))
    checkVectorized("SELECT i, d, d2 FROM t WHERE d >= d2", Seq(Filter))
    checkVectorized("SELECT i, d, d2 FROM t WHERE d = d2", Seq(Filter))
    checkVectorized("SELECT i, d FROM t WHERE d > 1e308", Seq(Filter)) // +Inf and NaN survive
    checkVectorized("SELECT i, d FROM t WHERE d = CAST('NaN' AS DOUBLE)", Seq(Filter))
  }

  test("null tests and three-valued logic") {
    checkVectorized("SELECT * FROM t WHERE l IS NOT NULL AND d < 50", Seq(Filter))
    checkVectorized("SELECT * FROM t WHERE l IS NULL OR d IS NULL", Seq(Filter))
    checkVectorized("SELECT * FROM t WHERE i > 100 OR l IS NULL", Seq(Filter))
    checkVectorized("SELECT * FROM t WHERE NOT (d >= 20) AND dt > DATE '2020-06-01'", Seq(Filter))
    checkVectorized("SELECT * FROM t WHERE (d > 10 AND l > 100) OR (d < 5 AND s IS NULL)", Seq(Filter))
    // Null AND false is false, null OR true is true: must match Spark exactly.
    checkVectorized("SELECT * FROM t WHERE (d > 10) AND (i < 0 OR l > 5)", Seq(Filter))
  }

  test("boolean column as predicate") {
    checkVectorized("SELECT i, b FROM t WHERE b", Seq(Filter))
    checkVectorized("SELECT i, b FROM t WHERE NOT b AND i < 5000", Seq(Filter))
  }

  test("string columns pass through compaction with nulls") {
    val df = checkVectorized("SELECT s, i FROM t WHERE i < 1000 AND d IS NOT NULL", Seq(Filter))
    assert(df.schema.fieldNames.toSeq === Seq("s", "i"))
  }

  test("selectivity extremes: pass-through and empty output") {
    checkVectorized("SELECT * FROM t WHERE i >= 0", Seq(Filter))
    val none = checkVectorized("SELECT * FROM t WHERE i < 0", Seq(Filter))
    assert(none.count() === 0)
    checkVectorized("SELECT * FROM t WHERE i = 12345", Seq(Filter))
  }

  test("aggregate above the filter with adaptive execution on and off") {
    Seq("true", "false").foreach { aqe =>
      withConf("spark.sql.adaptive.enabled" -> aqe) {
        checkVectorized("SELECT count(*), sum(l), min(d), max(i) FROM t WHERE i > 10 AND d IS NOT NULL", Seq(Filter))
      }
    }
  }

  test("unsupported expressions fall back with a reason") {
    checkFallback("SELECT * FROM t WHERE s LIKE 's1%'", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE i % 2 = 0", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE s = 's1'", Seq(Filter), "string")
    checkFallback("SELECT * FROM t WHERE s = concat(s, 'x')", Seq(Filter), "unsupported expression")
  }

  test("filter conversion can be disabled by configuration") {
    withConf(VectorConf.FilterEnabled -> "false") {
      val df = withPlugin(enabled = true)(spark.sql("SELECT * FROM t WHERE i > 500"))
      df.collect()
      assert(nodesOf[VectorFilterExec](df).isEmpty)
      assert(nodesOf[FilterExec](df).nonEmpty)
    }
  }

  test("metrics are populated") {
    val df = checkVectorized("SELECT * FROM t WHERE i < 100", Seq(Filter))
    val node = nodesOf[VectorFilterExec](df).head
    assert(node.metrics("numOutputRows").value === 100L)
    assert(node.metrics("numInputBatches").value > 0L)
    assert(node.metrics("numOutputBatches").value > 0L)
    assert(node.metrics("numOutputBatches").value <= node.metrics("numInputBatches").value)
  }

  test("explain shows the vectorized operator") {
    val plan = withPlugin(enabled = true)(spark.sql("SELECT * FROM t WHERE i > 5 AND l IS NOT NULL")).queryExecution.executedPlan
    val text = plan.treeString
    assert(text.contains("VectorFilter"), text)
  }
}
