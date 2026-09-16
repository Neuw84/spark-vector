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

  test("string comparisons against literals, column versus column, and IN") {
    // s is 's0'..'s49' with nulls where i % 10 = 0; byte order puts 's10' before 's2'.
    checkVectorized("SELECT * FROM t WHERE s = 's1'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s <> 's1'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s < 's2'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s <= 's10'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s > 's45'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s >= 's9'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE 's3' = s", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE 's3' < s", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s = ''", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s > ''", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s = 'nowhere'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s IN ('s1', 's17', 's30')", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s NOT IN ('s1', 's17')", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE i IN (1, 2, 300) OR l IN (33, 36)", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s IN ('s1') OR s IS NULL", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s = 's1' AND i > 100 OR s = 's7' AND d IS NULL", Seq(Filter))
    // Column versus column: a computed string next to s.
    checkVectorized("SELECT count(*) FROM (SELECT s, s AS s2 FROM t) WHERE s = s2", Seq(Filter))
    // Nulls: a null string compares to null, so NOT (s = 's1') keeps no null rows.
    val notEq = checkVectorized("SELECT i FROM t WHERE NOT (s = 's1')", Seq(Filter))
    assert(notEq.count() === 20000 - 2000 - 400) // 2000 nulls, 400 rows equal to 's1'
    val in = checkVectorized("SELECT i FROM t WHERE s IN ('s1', 's17', 's30')", Seq(Filter))
    assert(in.count() === 800) // 400 each for s1 and s17; id % 50 = 30 is one of the null rows, so s30 never occurs
  }

  test("LIKE prefix, suffix and contains shapes, and the string match functions") {
    // LikeSimplification rewrites these to StartsWith / EndsWith / Contains; s is 's0'..'s49'.
    checkVectorized("SELECT i, s FROM t WHERE s LIKE 's1%'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s LIKE '%7'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s LIKE '%3%'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s NOT LIKE 's1%'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE startswith(s, 's4') AND i > 100", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE endswith(s, '') OR s IS NULL", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE contains(s, '2') AND NOT contains(s, '4')", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s LIKE 'nowhere%'", Seq(Filter))
    val prefix = checkVectorized("SELECT i FROM t WHERE s LIKE 's1%'", Seq(Filter))
    assert(prefix.count() === 4000) // s1 and s10..s19: 11 values, 400 rows each, minus the null value s10
    val notLike = checkVectorized("SELECT i FROM t WHERE s NOT LIKE '%3%'", Seq(Filter))
    assert(notLike.count() === 20000 - 2000 - 5200) // nulls drop; s3, s13, s23, s43 and s30..s39 minus the null s30 = 13 values
  }

  test("unsupported expressions fall back with a reason") {
    // Inner wildcards are left as Like by the optimizer; 's%1' becomes Length(s) >= 2 AND ... which needs Length.
    checkFallback("SELECT * FROM t WHERE s LIKE 's%1%2'", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE s LIKE 's_'", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE hash(i) = 2", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE s = concat(s, 'x')", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE startswith(s, s)", Seq(Filter), "string pattern is not a literal")
    // The optimizer turns a long IN list into InSet, which is #48.
    checkFallback("SELECT * FROM t WHERE s IN ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12')", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE s IN ('s1', NULL)", Seq(Filter), "NULL in IN list")
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
