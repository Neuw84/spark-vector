package io.sparkvector.spark

import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.vector.{VectorFallback, VectorFilterExec, VectorHashAggregateExec, VectorProjectExec, VectorSortExec}

/**
 * Decimals of up to 18 digits as unscaled long lanes. Results are compared exactly (no double
 * tolerance): decimal arithmetic is integer arithmetic and must match Spark to the last digit.
 */
class VectorDecimalSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]
  private val Sort = classOf[VectorSortExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    // dec7 is int-backed in Spark's vectors (precision <= 9), dec12 and dec18 long-backed.
    val path = newTempPath("decimal/t")
    spark
      .range(0, 20000)
      .selectExpr(
        "cast(id as int) as i",
        "cast(id as double) / 8 as d2",
        "if(id % 7 = 3, null, cast(cast(id % 10007 as double) / 4 - 900 as decimal(7,2))) as dec7",
        "cast(cast(id * 37 % 1000003 as double) / 100 - 3000 as decimal(12,2)) as dec12",
        "if(id % 5 = 0, null, cast(cast(id as double) / 7 as decimal(18,4))) as dec18",
        // 17-digit values of either sign whose sum over the table leaves the long range (~1.8e21 unscaled).
        "if(id % 11 = 0, null, cast((900000000000000 + id) * if(id % 3 = 0, -1, 1) + 0.25 as decimal(18,2))) as big",
        "cast(id % 4 as decimal(3,1)) as k",
        "if(id % 10 = 0, null, concat('s', id % 50)) as s")
      .repartition(3)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("t")
    // Four rows near 10^38 in four partitions: any two of them summed leave the 38-digit result type
    // of sum(decimal(38,0)) (p + 10 capped at 38), so the merge overflows while every partial fits.
    val widePath = newTempPath("decimal/wide")
    spark
      .range(0, 4)
      .selectExpr("cast(id as int) as i", "cast('99999999999999999999999999999999999990' as decimal(38,0)) + cast(id as decimal(38,0)) as v")
      .repartition(4)
      .write
      .mode("overwrite")
      .parquet(widePath)
    spark.read.parquet(widePath).createOrReplaceTempView("wide")
  }

  /** Both runs, compared exactly on the string form of every value, order-insensitively. */
  private def checkExact(sql: String, operators: Seq[Class[_ <: SparkPlan]]): Unit = {
    val expected = withPlugin(enabled = false)(spark.sql(sql).collect()).map(_.toString).sorted
    val df = checkVectorized(sql, operators)
    val actual = df.collect().map(_.toString).sorted
    assert(actual.toSeq === expected.toSeq, s"exact mismatch for: $sql")
  }

  test("decimal columns of every width are read from Spark's vectors and forwarded") {
    checkExact("SELECT dec7, dec12, dec18, k FROM t WHERE i > 100", Seq(Filter))
    val df = checkVectorized("SELECT dec7, dec12 FROM t WHERE i < 10", Seq(Filter))
    assert(df.schema("dec7").dataType.simpleString === "decimal(7,2)")
    assert(df.schema("dec12").dataType.simpleString === "decimal(12,2)")
  }

  test("comparisons against literals and columns, with the analyzer's decimal casts") {
    checkExact("SELECT i FROM t WHERE dec12 > 100.50", Seq(Filter))
    checkExact("SELECT i FROM t WHERE dec7 <= -899.75 OR dec7 = 0", Seq(Filter))
    checkExact("SELECT i FROM t WHERE dec12 > dec7", Seq(Filter)) // dec7 is cast to decimal(12,2)
    checkExact("SELECT i FROM t WHERE dec18 BETWEEN 10 AND 20.5", Seq(Filter))
    checkExact("SELECT i FROM t WHERE dec7 IS NULL AND k <> 1.0", Seq(Filter))
  }

  test("addition, subtraction and multiplication within 18 digits") {
    checkExact("SELECT i, dec12 + dec7 AS a, dec12 - dec7 AS b, dec7 + 1.005 AS c, 10 - dec12 AS d FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i, dec7 * dec7 AS sq, dec7 * 3 AS tri, -dec12 AS neg, dec7 * k AS prod FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i, (dec7 + k) * 2 AS x FROM t WHERE dec7 IS NOT NULL", Seq(Filter, Project))
  }

  test("division rounds half up like Spark, null or error on zero") {
    checkExact("SELECT i, dec7 / 4 AS a, dec7 / k AS b, 100 / dec7 AS c FROM t WHERE dec7 <> 0 AND k <> 0", Seq(Filter, Project))
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkExact("SELECT i, dec7 / k AS b FROM t WHERE i < 400", Seq(Filter, Project))
    }
    withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql("SELECT dec7 / k FROM t WHERE i < 400").collect())
      assert(causes(e).exists(_.getMessage.contains("DIVIDE_BY_ZERO")), s"expected DIVIDE_BY_ZERO, got $e")
    }
  }

  test("casts to and from decimals, ANSI errors and legacy nulls") {
    checkExact("SELECT i, CAST(dec12 AS DOUBLE) AS a, CAST(dec18 AS DOUBLE) AS b, CAST(dec7 AS DECIMAL(10,4)) AS c, CAST(dec12 AS DECIMAL(9,1)) AS d FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i, CAST(i AS DECIMAL(10,2)) AS a, CAST(dec12 AS BIGINT) AS b, CAST(dec12 AS INT) AS c, CAST(d2 AS DECIMAL(12,3)) AS e FROM t WHERE i > 5", Seq(Filter, Project))
    withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql("SELECT CAST(dec12 AS DECIMAL(5,2)) FROM t").collect())
      assert(causes(e).exists(_.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE")), s"expected a precision error, got $e")
    }
    withConf("spark.sql.ansi.enabled" -> "false") {
      val df = checkVectorized("SELECT CAST(dec12 AS DECIMAL(5,2)) AS narrow FROM t", Seq(Project))
      assert(df.filter("narrow IS NULL").count() > 0)
      checkExact("SELECT CAST(dec12 AS DECIMAL(5,2)) AS narrow, CAST(d2 AS DECIMAL(4,1)) AS d FROM t", Seq(Project))
    }
  }

  test("rounding on the unscaled value: ceil, floor, round, bround and their scale forms") {
    // dec7 has exact halves at scale 2 (x.25 / x.75 quarters and x.50), dec18 has scale 4; k is decimal(3,1).
    checkExact("SELECT i, ceil(dec7) AS c7, floor(dec7) AS f7, ceil(dec12) AS c12, floor(dec12) AS f12, ceil(dec18) AS c18, floor(dec18) AS f18, ceil(k) AS ck, floor(k) AS fk FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i, round(dec7) AS r7, bround(dec7) AS b7, round(dec7, 1) AS r71, bround(dec7, 1) AS b71, round(dec7, -1) AS r7m, bround(dec7, -2) AS b7m, round(dec7, 5) AS r7wide FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i, round(dec18, 2) AS r2, bround(dec18, 2) AS b2, round(dec18, 0) AS r0, round(dec18, -3) AS rm, round(dec12, 1) AS r121, bround(dec12, -1) AS b12m, round(k) AS rk, bround(k) AS bk FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i, ceil(dec7, 1) AS c71, floor(dec7, 1) AS f71, ceil(dec7, -1) AS c7m, floor(dec7, -1) AS f7m, ceil(dec18, 3) AS c183, floor(dec18, -2) AS f18m, ceil(i, -2) AS cim, floor(i, -3) AS fim FROM t WHERE i > 5", Seq(Filter, Project))
    checkExact("SELECT i FROM t WHERE round(dec7) = 100 OR floor(k) = 2", Seq(Filter))
    // Spark widens the result by one integral digit: round(dec12, 2) is decimal(13,2) and compiles; round(dec18, 4) is decimal(19,4) and falls back, as does a huge negative scale.
    checkExact("SELECT i, round(dec12, 2) AS same FROM t WHERE i > 5", Seq(Filter, Project))
    checkFallback("SELECT round(dec18, 4) AS wide FROM t", Seq(Project), "not supported")
    checkFallback("SELECT round(dec18, -20) AS wider FROM t", Seq(Project), "not supported")
  }

  test("results wider than 18 digits fall back with a reason") {
    checkFallback("SELECT dec12 * dec12 AS x FROM t WHERE i > 5", Seq(Project), "exceeds 18 digits")
    checkFallback("SELECT dec12 / dec7 AS x FROM t WHERE dec7 > 1", Seq(Project), "exceeds 18 digits")
    // A wide decimal is accepted in exactly two places: the sum buffer a merge reads and the sum's
    // result. Any other wide column still refuses the operator.
    checkFallback("SELECT k, sum(dec12) * 2 AS x FROM t GROUP BY k", Seq(Project), "exceeds 18 digits")
  }

  test("a declared-wide product under a decimal sum is computed in 64 bits, its overflowing rows added exactly (#26)") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    def bothOurs(sql: String): Unit = {
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).isEmpty, "both stages should be ours\n" + finalPlan(df).treeString)
    }
    // Products that fit in 64 bits: no escalation, one lane, the 128-bit sum as before.
    val before = io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows()
    bothOurs("SELECT sum(dec12 * dec12) AS s FROM t")                       // decimal(25,4)
    bothOurs("SELECT sum(k * big) AS s FROM t")                             // decimal(22,3), fits
    bothOurs("SELECT sum(dec12 * 1234567.89) AS s FROM t")                  // literal operand, decimal(22,4)
    bothOurs("SELECT k, sum(dec12 * dec7) AS s FROM t GROUP BY k")           // decimal(20,4), grouped
    assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() == before, "nothing should have escalated")
    // Every product overflows 64 bits: every non-null row is escalated and the total is still exact
    // (a total past the sum's own decimal(38,4) is Spark's null in legacy mode, Spark's error in ANSI).
    withConf("spark.sql.ansi.enabled" -> "false") {
      bothOurs("SELECT sum(big * big) AS s FROM t")                         // decimal(37,4); the total leaves (38,4) -> null
      val nonNullBig = spark.sql("SELECT count(big) FROM t").collect().head.getLong(0)
      assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() == before + nonNullBig)
      bothOurs("SELECT k, sum(big * big) AS s, count(*) AS n FROM t GROUP BY k")
      // Under a selection: rows a filter dropped are neither summed nor escalated.
      val beforeFilter = io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows()
      bothOurs("SELECT sum(big * big) AS s FROM t WHERE i % 2 = 0 AND dec7 IS NOT NULL")
      val selected = spark.sql("SELECT count(big) FROM t WHERE i % 2 = 0 AND dec7 IS NOT NULL").collect().head.getLong(0)
      assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() == beforeFilter + selected, "only the selected rows escalate")
    }
    // Overflow in some rows only, spread across batches; grouped; a literal that overflows; totals that fit (ANSI, Spark 4's default).
    bothOurs("SELECT sum(cast(if(i % 997 = 0, 999999999999999999, i) AS decimal(18,0)) * cast(if(i % 997 = 0, 999999999999999999, i) AS decimal(18,0))) AS s FROM t")
    bothOurs("SELECT i % 3 AS g, sum(cast(if(i % 500 = 7, 999999999999999999, i) AS decimal(18,0)) * dec12) AS s FROM t GROUP BY i % 3")
    bothOurs("SELECT sum(big * 100000) AS s FROM t")                         // decimal(25,2), every row overflows 64 bits
    bothOurs("SELECT sum(dec12 * big) AS s FROM t")                          // decimal(31,4), most rows overflow 64 bits
    // A total past the sum's decimal(38,4) raises in ANSI mode for both engines, with the same error class.
    val ours = intercept[Exception] { withPlugin(enabled = true) { spark.sql("SELECT sum(big * big) AS s FROM t").collect() } }
    assert(ours.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE"), ours.getMessage)
    // Nested products (slice 2): a wide product as an operand of another, escalating as one tree; Q1's shape.
    bothOurs("SELECT sum(dec12 * dec7 * dec12) AS s FROM t")                                     // decimal(33,6), fits
    bothOurs("SELECT k, sum(dec12 * dec7 * dec12) AS s FROM t GROUP BY k")
    bothOurs("SELECT sum(dec12 * (1 - dec7) * (1 + dec7)) AS s FROM t")                           // decimal(30,6): TPC-H Q1's charge
    bothOurs("SELECT sum(dec12 * big * dec7) AS s FROM t")                                        // decimal(38,6): the inner product escalates
    // Past the capped precision: Spark's Multiply nulls the row in legacy mode and raises in ANSI mode; so do we.
    withConf("spark.sql.ansi.enabled" -> "false") { bothOurs("SELECT sum(big * big * big) AS s FROM t") }
    val past = intercept[Exception] { withPlugin(enabled = true) { spark.sql("SELECT sum(big * big * big) AS s FROM t").collect() } }
    assert(past.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE"), past.getMessage)
    // Still refused: a wide product anywhere but under a sum, and a product whose operand is neither a lane nor a product.
    checkFallback("SELECT dec12 * dec12 AS x FROM t WHERE i > 5", Seq(Project), "exceeds 18 digits")
    // (the Final stage may still be ours: it merges the buffer Spark's Partial produced)
    // (a wide sum as an operand became speculative in slice 4; a wide operand that is neither is still refused)
    bothOurs("SELECT sum((dec12 * dec12 + 1) * dec7) AS s FROM t")
    assert(nodesOf[HashAggregateExec](checkFallback("SELECT sum(cast(dec12 * dec12 AS decimal(30,4)) * dec7) AS s FROM t", Seq.empty, "neither a lane nor a speculative product or sum")).nonEmpty)
    assert(nodesOf[HashAggregateExec](checkFallback("SELECT max(dec12 * dec12) AS m FROM t", Seq.empty, "exceeds 18 digits")).nonEmpty)
  }

  test("aggregates over decimals: sum, avg, min, max, count, grouped and not") {
    checkExact("SELECT sum(dec7), avg(dec7), min(dec12), max(dec12), count(dec7), min(dec18) FROM t WHERE i > 3", Seq(Filter, Agg))
    checkExact("SELECT k, sum(dec7), avg(dec7), max(dec7), count(*) FROM t GROUP BY k", Seq(Agg))
    checkExact("SELECT s, sum(dec7) AS total, avg(dec7) AS mean FROM t WHERE dec7 > -500 GROUP BY s", Seq(Filter, Agg))
    // Every partition of one group empty of non-null values: isEmpty must survive the merge.
    checkExact("SELECT k, sum(dec7), avg(dec7) FROM t WHERE i % 7 = 3 GROUP BY k", Seq(Agg))
    // Decimal grouping key and a decimal sort key.
    checkExact("SELECT k, count(*) FROM t GROUP BY k", Seq(Agg))
    checkVectorized("SELECT dec12, i FROM t SORT BY dec12 DESC", Seq(Sort))
  }

  test("wide decimal sums: our Partial in 128 bits, our Final merging the (sum, isEmpty) buffer") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    def partialOurs(sql: String): Unit = {
      val df = checkVectorized(sql, Seq(Agg))
      val ours = nodesOf[VectorHashAggregateExec](df)
      assert(ours.exists(a => !a.isFinal) && ours.exists(_.isFinal), finalPlan(df).treeString)
      assert(nodesOf[HashAggregateExec](df).isEmpty, "both stages should be ours\n" + finalPlan(df).treeString)
      // The buffer the Partial emits is Spark's: a wide sum and isEmpty.
      assert(ours.filter(!_.isFinal).forall(_.output.exists(a => a.dataType.isInstanceOf[org.apache.spark.sql.types.DecimalType] && a.dataType.asInstanceOf[org.apache.spark.sql.types.DecimalType].precision > 18)), finalPlan(df).treeString)
    }
    // Grouped and ungrouped, 12 and 18 digits, a sum past 64 bits (`big`), nulls, empty groups.
    Seq(
      "SELECT sum(dec12) FROM t",
      "SELECT k, sum(dec12), sum(dec18) FROM t GROUP BY k",
      "SELECT sum(big), count(big) FROM t",
      "SELECT k, sum(big) FROM t GROUP BY k",
      "SELECT s, sum(big), sum(dec12) FROM t WHERE i > 100 GROUP BY s",
      // Every partition of some groups without a non-null value: isEmpty must come out of the Partial.
      "SELECT k, sum(big) FROM t WHERE i % 11 = 0 GROUP BY k",
      "SELECT sum(big) FROM t WHERE i % 11 = 0",
      // Mixed with the long-sum rewrite and other functions in the same operator.
      "SELECT k, sum(dec7), sum(dec12), avg(dec7), max(big), count(*) FROM t GROUP BY k",
      // FILTER clauses: groups empty in every partition, and partially filtered ones.
      "SELECT k, sum(big) FILTER (WHERE i < 0) AS none, sum(big) FILTER (WHERE i % 3 = 0) AS some FROM t GROUP BY k",
      "SELECT sum(big) FILTER (WHERE i < 0), sum(dec12) FILTER (WHERE i > 19990) FROM t").foreach { sql =>
      checkExact(sql, Seq(Agg))
      partialOurs(sql)
    }
    // The sum really leaves 64 bits: the exact value, and the row count that gets it there.
    val total = spark.sql("SELECT sum(big) FROM t").collect().head.getDecimal(0)
    assert(total.unscaledValue().abs().bitLength() > 63, total.toString)
  }

  test("a wide sum past its declared precision: Spark's error in ANSI mode, null otherwise") {
    // Each partition's partial fits; the merge of two rows does not. Spark's own Partial produces the
    // buffers (the 38-digit input has no lane); our Final merges them and applies the overflow rule.
    withPlugin(enabled = true) {
      // Spark's Partial adds with null-on-overflow and leaves the raising to CheckOverflowInSum, so
      // depending on how the scan is partitioned the merge sees a null buffer (ARITHMETIC_OVERFLOW,
      // "Overflow in sum of decimals") or a total past 38 digits (NUMERIC_VALUE_OUT_OF_RANGE).
      val e = intercept[Exception](spark.sql("SELECT sum(v) FROM wide").collect())
      assert(causes(e).exists(t => t.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE") || t.getMessage.contains("ARITHMETIC_OVERFLOW")), s"expected an overflow error, got $e")
      val sparkError = withPlugin(enabled = false)(intercept[Exception](spark.sql("SELECT sum(v) FROM wide").collect()))
      assert(causes(sparkError).exists(t => t.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE") || t.getMessage.contains("ARITHMETIC_OVERFLOW")))
      val df = spark.sql("SELECT sum(v) FROM wide WHERE i = 2")
      assert(df.collect().head.getDecimal(0).toPlainString === "99999999999999999999999999999999999992")
      assert(nodesOf[VectorHashAggregateExec](df).exists(_.isFinal), finalPlan(df).treeString)
    }
    withConf("spark.sql.ansi.enabled" -> "false") {
      val df = checkVectorized("SELECT sum(v) FROM wide", Seq(Agg))
      assert(df.collect().head.isNullAt(0))
      checkExact("SELECT i % 2 AS g, sum(v) FROM wide GROUP BY i % 2", Seq(Agg))
      checkExact("SELECT sum(v) FROM wide WHERE i = 2", Seq(Agg))
    }
  }

  test("both aggregate stages are ours for a decimal sum") {
    val df = checkVectorized("SELECT k, sum(dec7), avg(dec7) FROM t GROUP BY k", Seq(Agg))
    assert(nodesOf[VectorHashAggregateExec](df).exists(_.isFinal), finalPlan(df).treeString)
    assert(nodesOf[org.apache.spark.sql.execution.aggregate.HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
  }

  test("wide decimal averages: our Partial in 128 bits, our Final merging the (sum, count) buffer and dividing as Spark does (#26)") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    import org.apache.spark.sql.types.DecimalType
    def bothOurs(sql: String): Unit = {
      checkExact(sql, Seq(Agg))
      val df = checkVectorized(sql, Seq(Agg))
      val ours = nodesOf[VectorHashAggregateExec](df)
      assert(ours.exists(a => !a.isFinal) && ours.exists(_.isFinal), finalPlan(df).treeString)
      assert(nodesOf[HashAggregateExec](df).isEmpty, "both stages should be ours\n" + finalPlan(df).treeString)
      // The buffer the Partial emits is Spark's: a wide sum and a count.
      assert(ours.filter(!_.isFinal).forall(_.output.exists(a => a.dataType.isInstanceOf[DecimalType] && a.dataType.asInstanceOf[DecimalType].precision > 18)), finalPlan(df).treeString)
    }
    // Results within 18 digits (avg(dec12) is decimal(16,6)) and beyond (avg(dec18) is decimal(22,8)), grouped and not,
    // sums past 64 bits (`big`), nulls, groups empty in every partition, FILTER clauses, beside the other functions.
    Seq(
      "SELECT avg(dec12) FROM t",
      "SELECT avg(dec18) FROM t",
      "SELECT k, avg(dec12), avg(dec18) FROM t GROUP BY k",
      "SELECT avg(big), count(big), sum(big) FROM t",
      "SELECT k, avg(big) FROM t GROUP BY k",
      "SELECT s, avg(big), avg(dec12), sum(dec12) FROM t WHERE i > 100 GROUP BY s",
      "SELECT k, avg(big) FROM t WHERE i % 11 = 0 GROUP BY k",
      "SELECT avg(big) FROM t WHERE i % 11 = 0",
      "SELECT k, sum(dec7), avg(dec7), avg(dec12), max(big), count(*) FROM t GROUP BY k",
      "SELECT k, avg(big) FILTER (WHERE i < 0) AS none, avg(big) FILTER (WHERE i % 3 = 0) AS some FROM t GROUP BY k",
      "SELECT avg(big) FILTER (WHERE i < 0), avg(dec12) FILTER (WHERE i > 19990) FROM t",
      // try_avg is the same function in TRY mode.
      "SELECT k, try_avg(dec12), try_avg(big) FROM t GROUP BY k").foreach(bothOurs)
    // A declared-wide product under the average is speculative like the sum's (TPC-H Q1's shapes).
    val before = io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows()
    bothOurs("SELECT avg(dec12 * dec7) AS a FROM t")                               // decimal(20,4) product, decimal(24,8) result
    bothOurs("SELECT k, avg(dec12 * (1 - dec7)) AS a, sum(dec12 * (1 - dec7)) AS s FROM t GROUP BY k")
    assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() == before, "nothing should have escalated")
    bothOurs("SELECT k, avg(big * big) AS a FROM t WHERE i < 3000 GROUP BY k")      // every row escalates; the totals fit decimal(38,4)
    assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() > before)
    // A total past the buffer's decimal(38,4): an ungrouped Final divides the exact total (Spark's generated
    // code keeps it in a local nothing re-checks), a grouped one sees the buffer nulled; a partial past it is
    // null from either engine. In ANSI mode the null buffer is Spark's own ARITHMETIC_OVERFLOW from the division.
    withConf("spark.sql.ansi.enabled" -> "false") {
      bothOurs("SELECT avg(big * big) AS a FROM t")                                // partials fit, the total does not: a value
      bothOurs("SELECT k, avg(big * big * 10) AS a, count(*) AS n FROM t GROUP BY k") // grouped totals past the buffer: null
      bothOurs("SELECT avg(big * big * 10) AS a FROM t")                           // partials past the buffer: null
      assert(spark.sql("SELECT avg(big * big * 10) AS a FROM t").collect().head.isNullAt(0))
    }
    val ours = intercept[Exception] { withPlugin(enabled = true) { spark.sql("SELECT k, avg(big * big * 10) AS a FROM t GROUP BY k").collect() } }
    assert(ours.getMessage.contains("ARITHMETIC_OVERFLOW"), ours.getMessage)
    val sparks = withPlugin(enabled = false)(intercept[Exception](spark.sql("SELECT k, avg(big * big * 10) AS a FROM t GROUP BY k").collect()))
    assert(sparks.getMessage.contains("ARITHMETIC_OVERFLOW"), sparks.getMessage)
    // try_avg nulls where avg raises.
    bothOurs("SELECT k, try_avg(big * big * 10) AS a FROM t GROUP BY k")
    // The average's result itself past 18 digits rides above the aggregate as a pass-through column.
    checkExact("SELECT a FROM (SELECT k, avg(dec18) AS a FROM t GROUP BY k) WHERE a IS NOT NULL", Seq(Agg, Filter))
    // A wide result inside another expression is refused (the projection would have to divide wide decimals).
    assert(nodesOf[HashAggregateExec](checkFallback("SELECT avg(dec18) * 2 AS x FROM t", Seq.empty, "exceeds 18 digits")).nonEmpty)
  }

  test("a declared-wide sum or difference under a decimal sum or avg: rescaled and added in 64 bits, overflowing rows exact (#26)") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    def bothOurs(sql: String): Unit = {
      checkExact(sql, Seq(Agg))
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).isEmpty, "both stages should be ours\n" + finalPlan(df).treeString)
    }
    val before = io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows()
    // Sums that fit 64 bits: a lane plus a literal (decimal(19,4)), lanes of different scales, a product plus a lane, differences.
    bothOurs("SELECT sum(dec18 + 1) AS s FROM t")                                   // decimal(19,4)
    bothOurs("SELECT k, sum(dec18 - 1) AS s, avg(dec18 + 1) AS a FROM t GROUP BY k")
    bothOurs("SELECT sum(dec12 + dec18) AS s FROM t")                               // decimal(21,4): dec12 rescaled to 4
    bothOurs("SELECT k, sum(dec12 * dec7 + dec12) AS s FROM t GROUP BY k")         // decimal(21,4): a speculative product plus a lane
    bothOurs("SELECT sum(dec12 * dec7 - dec18) AS s FROM t")                        // decimal(21,4)
    bothOurs("SELECT sum(dec12 * (1 - dec7) + dec12 * dec7) AS s FROM t")           // two speculative products
    bothOurs("SELECT sum(big + big) AS s, sum(big - big) AS d FROM t")              // decimal(19,2): 17-digit values, fits
    assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() == before, "nothing should have escalated")
    // Escalation at the rescale: big * 100 is a 19-digit unscaled value that leaves 64 bits when shifted to scale 4.
    bothOurs("SELECT sum(big * 100 + dec18) AS s FROM t")                           // decimal(25,4)
    val nonNull = spark.sql("SELECT count(big) FROM t WHERE dec18 IS NOT NULL").collect().head.getLong(0)
    assert(io.sparkvector.spark.expr.SpeculativeDecimals.escalatedRows() == before + 2 * nonNull, "every row with both operands escalates (bothOurs runs the query twice)")
    // Escalation from an operand: every product of big * big is exact, the add combines exactly.
    bothOurs("SELECT k, sum(big * big + dec12) AS s FROM t GROUP BY k")             // decimal(38,4)
    bothOurs("SELECT k, avg(big * big - dec12 * dec7) AS a FROM t WHERE i < 3000 GROUP BY k")
    // Escalation at the add itself: two rescaled 18-digit values whose sum leaves 64 bits.
    bothOurs("SELECT sum(cast(if(i % 3 = 0, 900000000000000000, i) AS decimal(18,0)) + cast(if(i % 3 = 0, 90000000000000000.0, i) AS decimal(18,1))) AS s FROM t WHERE i < 100") // decimal(20,1): 9.9e18 unscaled
    // Still refused: a wide sum as a projected value, and the shape where Spark's precision cap lowers the scale (it rounds there).
    checkFallback("SELECT dec18 + 1 AS x FROM t WHERE i > 5", Seq(Project), "exceeds 18 digits")
    bothOurs("SELECT sum(dec18 * dec18 + dec18 * dec18) AS s FROM t")               // decimal(38,8): exactly at the cap, scale kept
    assert(nodesOf[HashAggregateExec](checkFallback("SELECT sum(big * big * 10 + dec18 * dec18) AS s FROM t WHERE i < 8", Seq.empty, "18 digits")).nonEmpty) // (38,4) + (37,8): the cap lowers the scale to 6
  }

  test("sum(DISTINCT) and avg(DISTINCT) over wide decimals: every stage of the distinct rewrite ours") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    def allOurs(sql: String): Unit = {
      checkExact(sql, Seq(Agg))
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[VectorHashAggregateExec](df).exists(_.isFinal), finalPlan(df).treeString)
      assert(nodesOf[HashAggregateExec](df).isEmpty, "every stage should be ours\n" + finalPlan(df).treeString)
    }
    allOurs("SELECT sum(DISTINCT dec12), avg(DISTINCT dec12), count(DISTINCT dec12) FROM t")
    allOurs("SELECT k, sum(DISTINCT dec12), avg(DISTINCT dec12) FROM t GROUP BY k")
    // Empty input: Spark's null, not the buffer's zero.
    allOurs("SELECT sum(DISTINCT dec12), avg(DISTINCT dec12) FROM t WHERE i < 0")
    allOurs("SELECT k, sum(DISTINCT big), avg(DISTINCT big) FROM t WHERE i % 11 = 0 GROUP BY k")
  }

  test("CheckOverflow: null or Spark's error past the precision, identity for the declared type") {
    import org.apache.spark.sql.catalyst.expressions.CheckOverflow
    import org.apache.spark.sql.functions.col
    import org.apache.spark.sql.types.DecimalType
    import org.apache.spark.sql.vector.TestExprs.column
    // Spark 4.1 leaves no CheckOverflow in a batch plan, so the node is built directly and viewed.
    def view(name: String, nullOnOverflow: Boolean, target: DecimalType, source: String = "dec12"): Unit =
      spark.table("t").select(col("i"), col(source), column(CheckOverflow(org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute(source), target, nullOnOverflow)).as("c"))
        .createOrReplaceTempView(name)
    // decimal(5,2) holds at most 999.99: some values of dec12 (about -3000 .. 7000) fit, others do not.
    view("co_null", nullOnOverflow = true, DecimalType(5, 2))
    checkExact("SELECT i, dec12, c FROM co_null", Seq(Project))
    checkExact("SELECT count(c) AS fit, count(*) AS n FROM co_null", Seq(Agg))
    // Rescaling up is exact and never overflows; the declared type itself is the identity.
    view("co_up", nullOnOverflow = false, DecimalType(16, 6))
    checkExact("SELECT i, c, c * 2 AS d FROM co_up", Seq(Project))
    view("co_same", nullOnOverflow = false, DecimalType(12, 2))
    checkExact("SELECT i, c FROM co_same WHERE c > 0", Seq(Project, Filter))
    // Rounding down half-up like Spark's toPrecision, from a decimal with a larger scale.
    view("co_round", nullOnOverflow = true, DecimalType(18, 1), source = "dec18")
    checkExact("SELECT i, dec18, c FROM co_round", Seq(Project))
    // nullOnOverflow = false raises Spark's NUMERIC_VALUE_OUT_OF_RANGE -- for active rows only.
    view("co_raise", nullOnOverflow = false, DecimalType(5, 2))
    val e = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT c FROM co_raise").collect()))
    assert(causes(e).exists(_.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE")), s"expected NUMERIC_VALUE_OUT_OF_RANGE, got $e")
    checkExact("SELECT i, c FROM co_raise WHERE dec12 BETWEEN -900 AND 900", Seq(Project, Filter))
  }

  test("the DecimalAggregates rewrite: MakeDecimal over sum(UnscaledValue), and no CheckOverflow anywhere") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    def allOurs(sql: String): Unit = {
      val plan = withPlugin(enabled = false)(spark.sql(sql).queryExecution.optimizedPlan.toString)
      assert(plan.contains("MakeDecimal") && plan.contains("UnscaledValue"), s"expected the DecimalAggregates shape in\n$plan")
      assert(!plan.contains("CheckOverflow("), s"Spark 4.1 should not insert CheckOverflow here\n$plan")
      checkExact(sql, Seq(Agg))
      withPlugin(enabled = true) { val df = spark.sql(sql); df.collect(); assert(nodesOf[HashAggregateExec](df).isEmpty, "every stage should be ours\n" + finalPlan(df).treeString) }
    }
    allOurs("SELECT sum(dec7) AS s FROM t")
    allOurs("SELECT i % 5 AS g, sum(dec7) AS s, count(*) AS n FROM t GROUP BY i % 5")
    allOurs("SELECT i % 3 AS g, sum(dec7) + sum(k) AS total, sum(k) AS sk FROM t GROUP BY i % 3")
    allOurs("SELECT i % 7 AS g, sum(dec7) AS s FROM t WHERE dec7 IS NOT NULL AND i > 500 GROUP BY i % 7")
    // The arithmetic operators check their own overflow in 4.1: no CheckOverflow wraps them either.
    for (sql <- Seq("SELECT dec7 * dec12 AS a, dec7 + dec12 AS b, dec18 / dec7 AS c, -dec7 AS d, dec7 + 1.5 AS e0 FROM t",
                    "SELECT avg(dec7) AS a, avg(dec12) AS b FROM t")) {
      val plan = withPlugin(enabled = false)(spark.sql(sql).queryExecution.optimizedPlan.toString)
      assert(!plan.contains("CheckOverflow("), s"Spark 4.1 should not insert CheckOverflow here\n$plan")
    }
  }

  private def causes(t: Throwable): Seq[Throwable] =
    Iterator.iterate(t)(_.getCause).takeWhile(_ != null).take(10).toSeq

  test("a union of aliased-key channel aggregates keeps the partitioning contract whatever the column type (TPC-DS q66)") {
    import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike
    // Each channel is a Final aggregate hash-partitioned by k whose sum(big) result is decimal(28,2): a
    // type our operators do not read, but the union only forwards batches. Left to Spark, the union
    // ran Spark 4.1.3's columnar UnionExec, which concatenates the co-partitioned children, and the
    // aggregate above (planned without a shuffle on the union's partitioning) emitted every key once per
    // channel -- q66 returned 10 rows for Spark's 5.
    val channel = (c: Int) => s"SELECT k, sum(big) AS total FROM t WHERE i % 3 = $c GROUP BY k"
    val sql = s"SELECT k, sum(total) AS total, count(*) AS parts FROM (${channel(0)} UNION ALL ${channel(1)}) u GROUP BY k"
    val expected = withPlugin(enabled = false)(spark.sql(sql).collect())
    val df = withPlugin(enabled = true) { val d = spark.sql(sql); d.collect(); d }
    assertRowsEqual(expected, df.collect(), 1e-9, sql)
    assert(nodesOf[org.apache.spark.sql.vector.VectorUnionExec](df).nonEmpty, finalPlan(df).treeString)
    assert(nodesOf[org.apache.spark.sql.execution.UnionExec](df).isEmpty, finalPlan(df).treeString)
    assert(nodesOf[ShuffleExchangeLike](df).size === 2, finalPlan(df).treeString)
    val rows = df.collect()
    assert(rows.length === 4 && rows.forall(_.getLong(2) === 2), rows.mkString("\n"))
    // q66 itself aliases the key (`d_year AS year`): the channel aggregate must report its partitioning
    // over the alias, as Spark's does, or the union cannot recognise the children as co-partitioned.
    val aliased = (c: Int) => s"SELECT k AS kk, sum(big) AS total FROM t WHERE i % 3 = $c GROUP BY k"
    val sql2 = s"SELECT kk, sum(total) AS total, count(*) AS parts FROM (${aliased(0)} UNION ALL ${aliased(1)}) u GROUP BY kk"
    val expected2 = withPlugin(enabled = false)(spark.sql(sql2).collect())
    val df2 = withPlugin(enabled = true) { val d = spark.sql(sql2); d.collect(); d }
    assertRowsEqual(expected2, df2.collect(), 1e-9, sql2)
    val union = nodesOf[org.apache.spark.sql.vector.VectorUnionExec](df2).head
    assert(union.outputPartitioning.isInstanceOf[org.apache.spark.sql.catalyst.plans.physical.HashPartitioningLike], union.outputPartitioning.toString)
    assert(nodesOf[ShuffleExchangeLike](df2).size === 2, finalPlan(df2).treeString)
    assert(df2.collect().forall(_.getLong(2) === 2), df2.collect().mkString("\n"))
  }
}
