package io.sparkvector.spark

import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorProjectExec, VectorSortExec}

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
        "cast(id % 4 as decimal(3,1)) as k",
        "if(id % 10 = 0, null, concat('s', id % 50)) as s")
      .repartition(3)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("t")
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
    checkFallback("SELECT sum(dec12) FROM t", Seq(Agg), "exceeds 18 digits")
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

  test("both aggregate stages are ours for a decimal sum") {
    val df = checkVectorized("SELECT k, sum(dec7), avg(dec7) FROM t GROUP BY k", Seq(Agg))
    assert(nodesOf[VectorHashAggregateExec](df).exists(_.isFinal), finalPlan(df).treeString)
    assert(nodesOf[org.apache.spark.sql.execution.aggregate.HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
  }

  private def causes(t: Throwable): Seq[Throwable] =
    Iterator.iterate(t)(_.getCause).takeWhile(_ != null).take(10).toSeq
}
