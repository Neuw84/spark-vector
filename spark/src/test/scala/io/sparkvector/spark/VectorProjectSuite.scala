package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}

import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorProjectExec}

class VectorProjectSuite extends VectorQuerySuite {

  private val Project = classOf[VectorProjectExec]
  private val Filter = classOf[VectorFilterExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("project/t"))
    // A timestamp table: hourly instants around the epoch with sub-second parts, some negative.
    val tsPath = newTempPath("project/ts")
    spark
      .range(0, 5000)
      .selectExpr(
        "cast(id as int) as i",
        "if(id % 9 = 0, null, timestamp_micros((id - 2500) * 3600000000 + id * 7919 % 1000000)) as ts",
        "date_add(date '1969-12-01', cast(id % 400 as int)) as d0")
      .repartition(2)
      .write
      .mode("overwrite")
      .parquet(tsPath)
    spark.read.parquet(tsPath).createOrReplaceTempView("ts")
  }

  test("date fields, truncation and arithmetic over date columns") {
    // dt spans 2020-01-01 .. 2021-12-30 (a leap year and a plain one); d0 straddles the epoch.
    checkVectorized("SELECT i, year(dt) AS y, month(dt) AS m, dayofmonth(dt) AS d, dayofyear(dt) AS doy, quarter(dt) AS q, dayofweek(dt) AS dow, weekday(dt) AS wd FROM t", Seq(Project))
    checkVectorized("SELECT extract(year FROM dt) AS y, extract(month FROM dt) AS m, day(dt) AS d FROM t WHERE i > 100", Seq(Filter, Project))
    checkVectorized("SELECT i, year(d0) AS y, month(d0) AS m, dayofmonth(d0) AS d, dayofyear(d0) AS doy, dayofweek(d0) AS dow FROM ts", Seq(Project))
    checkVectorized("SELECT trunc(dt, 'YEAR') AS ty, trunc(dt, 'quarter') AS tq, trunc(dt, 'MM') AS tm, trunc(dt, 'week') AS tw, i FROM t", Seq(Project))
    checkVectorized("SELECT date_add(dt, 90) AS a, date_sub(dt, i) AS b, date_add(dt, i) AS c, datediff(dt, DATE '2020-06-15') AS e, datediff(DATE '2021-01-01', dt) AS f FROM t", Seq(Project))
    checkVectorized("SELECT datediff(dt, d) AS g FROM (SELECT dt, date_add(dt, 3) AS d FROM t)", Seq(Project))
    // Nulls propagate: a date that is null where s is null.
    checkVectorized("SELECT year(IF(s IS NULL, NULL, dt)) AS y, trunc(IF(s IS NULL, NULL, dt), 'MM') AS tm, date_add(IF(s IS NULL, NULL, dt), 1) AS a FROM t", Seq(Project))
    // The TPC-H Q7/Q8/Q9 shape: extract(year) as a group key on our aggregate.
    checkVectorized("SELECT year(dt) AS y, count(*) AS c FROM t GROUP BY year(dt)", Seq(classOf[VectorHashAggregateExec]))
    checkFallback("SELECT trunc(dt, 'DAY') AS x FROM t", Seq(Project), "trunc unit 'DAY' not supported")
  }

  test("timestamp fields under UTC and fixed-offset session zones; zone rules fall back") {
    Seq("UTC", "+05:30", "-03:00", "Etc/GMT+7").foreach { zone =>
      withConf("spark.sql.session.timeZone" -> zone) {
        checkVectorized("SELECT i, cast(ts AS DATE) AS d, hour(ts) AS h, minute(ts) AS mi, second(ts) AS s FROM ts", Seq(Project))
        checkVectorized("SELECT year(ts) AS y, month(ts) AS m, dayofmonth(ts) AS d FROM ts WHERE i > 10", Seq(Filter, Project))
        checkVectorized("SELECT cast(ts AS DATE) AS d, count(*) AS c FROM ts GROUP BY cast(ts AS DATE)", Seq(classOf[VectorHashAggregateExec]))
      }
    }
    withConf("spark.sql.session.timeZone" -> "America/New_York") {
      checkFallback("SELECT hour(ts) AS h FROM ts", Seq(Project), "fixed-offset session zone")
      checkFallback("SELECT year(ts) AS y FROM ts", Seq(Project), "fixed-offset session zone")
    }
  }

  test("double arithmetic with columns and literals (ANSI mode, Spark 4 default)") {
    assert(spark.conf.get("spark.sql.ansi.enabled") === "true")
    checkVectorized("SELECT i, d * 2.0 AS x, d + d2 AS y, d - d2 AS z, 1.0 - d2 AS w, -d AS neg FROM t WHERE i > 100", Seq(Filter, Project))
    checkVectorized("SELECT d * (1.0 - d2) * (1.0 + d2) AS q1_style FROM t WHERE d IS NOT NULL", Seq(Filter, Project))
    checkVectorized("SELECT d2 / 4.0 AS a, 8.0 / (d2 + 1.0) AS b FROM t WHERE i < 5000", Seq(Filter, Project))
  }

  test("division by zero raises in ANSI mode, yields null in legacy mode") {
    // d2 is zero where id % 13 = 0.
    val sql = "SELECT i, d / d2 AS r FROM t WHERE i < 100"
    withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql(sql).collect())
      assert(causes(e).exists(c => c.isInstanceOf[ArithmeticException] && c.getMessage.contains("DIVIDE_BY_ZERO")), s"expected DIVIDE_BY_ZERO, got $e")
    }
    withConf("spark.sql.ansi.enabled" -> "false") {
      val df = checkVectorized(sql, Seq(Project))
      assert(df.filter("r IS NULL").count() === 17) // d null where i % 11 = 0 (10 rows) plus d2 = 0 where i % 13 = 0 (8 rows), overlapping at 0
      checkVectorized("SELECT i, 1.0 / d2 AS r, d / 0.0 AS all_null FROM t WHERE i < 100", Seq(Project))
    }
  }

  test("integer arithmetic is vectorized in legacy mode and, overflow-checked, in ANSI mode") {
    val queries = Seq(
      "SELECT i + 1 AS a, i * 3 AS b, l - 7 AS c, 100 - i AS d, -i AS e, l * l AS f FROM t WHERE i > 10",
      "SELECT i + i AS a, l + l AS b, CAST(i AS BIGINT) + l AS c FROM t WHERE d IS NULL")
    withConf("spark.sql.ansi.enabled" -> "false") {
      queries.foreach(q => checkVectorized(q, Seq(Project)))
    }
    // Spark 4's default. Nothing here overflows, so the results are the same and the plan is ours.
    queries.foreach(q => checkVectorized(q, Seq(Project)))
    checkVectorized("SELECT i * 2 + 1 AS k, -l AS nl FROM t WHERE i > 10", Seq(Project))
  }

  test("ANSI integer overflow raises Spark's ARITHMETIC_OVERFLOW, for active rows only") {
    def assertOverflow(sql: String, message: String, hint: String): Unit = withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql(sql).collect())
      val cause = causes(e).find(_.isInstanceOf[ArithmeticException]).getOrElse(fail(s"expected ARITHMETIC_OVERFLOW, got $e"))
      assert(cause.getMessage.contains("ARITHMETIC_OVERFLOW"), cause.getMessage)
      assert(cause.getMessage.contains(message), cause.getMessage)
      if (hint.nonEmpty) assert(cause.getMessage.contains(hint), cause.getMessage)
    }
    // i ranges 0..19999, so i * 2147483 overflows an int for i >= 1000 and l * l a long for large l.
    assertOverflow("SELECT i * 2147483 AS x FROM t", "integer overflow", "try_multiply")
    assertOverflow("SELECT i + 2147483000 AS x FROM t", "integer overflow", "try_add")
    assertOverflow("SELECT -2147483000 - i AS x FROM t", "integer overflow", "try_subtract")
    assertOverflow("SELECT l * l * l * l AS x FROM t WHERE l IS NOT NULL", "long overflow", "try_multiply")
    assertOverflow("SELECT -CAST(i - 2147483647 - 1 AS INT) AS x FROM t WHERE i = 0", "integer overflow", "")
    // Rows removed by the filter, or decided by an earlier conjunct, are never evaluated: no error.
    checkVectorized("SELECT i * 2147483 AS x FROM t WHERE i < 1000", Seq(Filter, Project))
    checkVectorized("SELECT i FROM t WHERE i < 1000 AND i * 2147483 > 0", Seq(Filter))
    // Legacy mode wraps, matching Spark.
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT i * 2147483 AS x, i + 2147483000 AS y FROM t", Seq(Project))
    }
  }

  test("widening casts and implicit casts inserted by the analyzer") {
    checkVectorized("SELECT CAST(i AS BIGINT) AS a, CAST(i AS DOUBLE) AS b, CAST(l AS DOUBLE) AS c FROM t WHERE i > 5", Seq(Project))
    checkVectorized("SELECT i + 1.5D AS a, l * 0.5D AS b, i / 2 AS c FROM t WHERE i > 5", Seq(Project))
    // A plain decimal literal makes Spark cast the int to a decimal; both run on long lanes.
    checkVectorized("SELECT i + 1.5 AS a FROM t WHERE i > 5", Seq(Project))
    checkFallback("SELECT CAST(d2 AS INT) AS a FROM t WHERE i > 5", Seq(Project), "unsupported cast")
  }

  // `l` is null on every seventh row (so `l > 3000` is a null condition there), `d` is null on
  // every eleventh (a null branch value), `s` on every tenth; `d` also carries NaN and infinities.
  test("CASE WHEN blends branches by mask, with nulls in conditions and branches") {
    checkVectorized("SELECT i, CASE WHEN i > 15000 THEN d WHEN l > 3000 THEN d2 ELSE 0.0 END AS x FROM t", Seq(Project))
    checkVectorized("SELECT i, CASE WHEN l > 3000 THEN d WHEN i > 100 THEN d2 END AS no_else FROM t WHERE i < 19000", Seq(Filter, Project))
    checkVectorized("SELECT CASE WHEN d > 50.0 THEN NULL WHEN d2 > 1.0 THEN d ELSE d2 END AS with_null_branch FROM t", Seq(Project))
    checkVectorized("SELECT CASE WHEN i > 10000 THEN l ELSE 7L END AS l_or_7, CASE WHEN b THEN d2 ELSE -d2 END AS signed FROM t WHERE l IS NOT NULL", Seq(Filter, Project))
    checkVectorized("SELECT CASE WHEN i > 10000 THEN s WHEN d IS NULL THEN 'no d' ELSE 'other' END AS label FROM t", Seq(Project))
    checkVectorized("SELECT CASE WHEN d > 5.0 THEN b ELSE d2 > 1.0 END AS flag, CASE WHEN i > 100 THEN dt END AS day FROM t WHERE d IS NOT NULL", Seq(Filter, Project))
    // Q14 / Q8 / Q12 shape: the conditional feeds an aggregate.
    checkVectorized("SELECT SUM(CASE WHEN i > 15000 THEN d * (1.0 - d2) ELSE 0.0 END) AS promo, SUM(d * (1.0 - d2)) AS total FROM t WHERE d IS NOT NULL",
      Seq(Filter, classOf[VectorHashAggregateExec]))
  }

  test("IF, COALESCE, NVL and NULLIF compile through the same blend") {
    checkVectorized("SELECT IF(i > 100, d, d2) AS pick, IF(l > 3000, 1L, 0L) AS flag_with_null_cond FROM t", Seq(Project))
    checkVectorized("SELECT COALESCE(l, CAST(i AS BIGINT)) AS a, COALESCE(d, d2, 0.0) AS b, COALESCE(s, 'none') AS c FROM t", Seq(Project))
    checkVectorized("SELECT NVL(l, -1L) AS a, NULLIF(i, 5) AS b, NVL2(l, d, d2) AS c FROM t WHERE i < 1000", Seq(Filter, Project))
    checkFallback("SELECT CASE WHEN i > 100 THEN CAST(d AS DECIMAL(30, 2)) ELSE NULL END AS wide FROM t", Seq(Project), "unsupported result type")
  }

  test("literal columns are materialised, dense or under a selection") {
    checkVectorized("SELECT 1 AS one, i, 2.5D AS x, 7L AS l7, DATE '2020-01-02' AS day FROM t WHERE i > 5", Seq(Filter, Project))
    checkVectorized("SELECT 1.25 AS dec, i FROM t WHERE d > 5 AND d2 > 0.5", Seq(Filter, Project))
    checkVectorized("SELECT 1 FROM range(10) HAVING MAX(id) > 0", Seq()) // Spark's own SQL tests tripped on this shape
    // String literals: a constant UTF8 column, dense and under a selection, incl. the empty string.
    checkVectorized("SELECT 'MAIL' AS mode, i, '' AS empty, s FROM t", Seq(Project))
    checkVectorized("SELECT 'BUILDING' AS seg, i FROM t WHERE i > 5 AND s = 's7'", Seq(Filter, Project))
  }

  test("monotonically_increasing_id() numbers exactly the rows Spark numbers, per partition") {
    // Same plan on both sides (same Parquet splits), so the ids must be identical, not merely
    // monotonic: the comparison is row for row against Spark.
    val plain = checkVectorized("SELECT monotonically_increasing_id() AS id, i FROM t", Seq(Project))
    val ids = plain.collect().map(_.getLong(0))
    assert(ids.distinct.length === ids.length, "ids are unique across partitions")
    assert(ids.map(_ >> 33).distinct.length === plain.rdd.getNumPartitions, "one prefix per scan partition")
    // A dense filter below forwards a selection to the project: only the surviving rows are numbered.
    checkVectorized("SELECT monotonically_increasing_id() AS id, i FROM t WHERE i > 100", Seq(Filter, Project))
    // A sparse filter below compacts first: the same rule through the other path.
    checkVectorized("SELECT monotonically_increasing_id() AS id, i, s FROM t WHERE s = 's7'", Seq(Filter, Project))
    // A filter above the project sees ids assigned before filtering.
    checkVectorized("SELECT id, i FROM (SELECT monotonically_increasing_id() AS id, i FROM t) WHERE id > 100 AND i < 15000", Seq(Filter, Project))
    // Ids feeding an aggregate and arithmetic.
    checkVectorized("SELECT count(*) AS c, max(id) AS mx, min(id) AS mn FROM (SELECT monotonically_increasing_id() AS id FROM t WHERE d IS NOT NULL)", Seq(Filter, Project, classOf[VectorHashAggregateExec]))
    checkVectorized("SELECT monotonically_increasing_id() + l AS x, i FROM t WHERE l IS NOT NULL", Seq(Filter, Project))
  }

  test("math basics: abs, sign, positive, %, pmod, div, greatest, least, nanvl") {
    // Values against Spark (ANSI mode, Spark 4's default). i is 0..19999, l nullable, d has NaN and infinities.
    checkVectorized("SELECT abs(i - 10000) AS a, abs(l - 30000) AS b, abs(d) AS c, sign(d) AS sd, sign(i) AS si, positive(i) AS p, negative(l) AS ng FROM t", Seq(Project))
    checkVectorized("SELECT i % 7 AS m1, l % 13 AS m2, pmod(i - 10000, 7) AS p1, pmod(l - 30000, -13) AS p2, d % 2.5 AS m3, pmod(d, 3.0) AS p3, (i - 10000) % -7 AS m4 FROM t", Seq(Project))
    checkVectorized("SELECT i div 3 AS d1, (i - 10000) div -7 AS d2, l div 3 AS d3, 100000 div (i + 1) AS d4, 100 % (i + 1) AS m5, pmod(100, i + 1) AS p5 FROM t", Seq(Project))
    checkVectorized("SELECT greatest(i, 100, 5000) AS g1, least(i, 100, 5000) AS l1, greatest(l, CAST(i AS BIGINT)) AS g2, least(l, CAST(i AS BIGINT)) AS l2, greatest(d, d2) AS g3, least(d, d2, 1.0) AS l3 FROM t", Seq(Project))
    checkVectorized("SELECT nanvl(d, 0.0) AS n1, nanvl(d, d2) AS n2, nanvl(d2, d) AS n3, nanvl(d, l) AS n4 FROM t", Seq(Project))
    checkVectorized("SELECT i FROM t WHERE i % 3 = 0 AND pmod(i, 5) = 1 AND abs(i - 500) < 200", Seq(Filter))
    // Zero divisors on rows a filter removed never raise; on active rows ANSI raises, legacy nulls.
    checkVectorized("SELECT i % (i - 5) AS m, i div (i - 5) AS q, pmod(i, i - 5) AS p FROM t WHERE i > 5", Seq(Filter, Project))
    checkVectorized("SELECT i FROM t WHERE i > 5 AND 100 % (i - 5) = 3", Seq(Filter))
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT i % (i - 5) AS m, i div (i - 5) AS q, pmod(i, i - 5) AS p, i % 0 AS z FROM t WHERE i < 20", Seq(Filter, Project))
      checkVectorized("SELECT abs(i - 2147483647 - 1) AS wrapped FROM t WHERE i < 3", Seq(Filter, Project))
    }
    def assertError(sql: String, marker: String): Unit = withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql(sql).collect())
      assert(causes(e).exists(c => c.isInstanceOf[ArithmeticException] && c.getMessage.contains(marker)), s"expected $marker, got $e")
    }
    assertError("SELECT i % (i - 5) AS m FROM t", "REMAINDER_BY_ZERO")
    assertError("SELECT pmod(i, 0) AS p FROM t WHERE i < 10", "REMAINDER_BY_ZERO")
    assertError("SELECT i div (i - 5) AS q FROM t", "DIVIDE_BY_ZERO")
    assertError("SELECT abs(i - 2147483647 - 1) AS a FROM t WHERE i < 3", "ARITHMETIC_OVERFLOW")
    assertError("SELECT abs(CAST(i AS BIGINT) - 9223372036854775807 - 1) AS a FROM t WHERE i < 3", "long overflow")
    assertError("SELECT (CAST(i AS BIGINT) - 9223372036854775807 - 1) div -1 AS q FROM t WHERE i < 3", "Overflow in integral divide")
    // Decimal operands stay a fallback with a reason.
    checkFallback("SELECT CAST(d AS DECIMAL(10, 2)) % CAST(i + 1 AS DECIMAL(10, 2)) AS m FROM t", Seq(Project), "% over decimal(10,2) not supported")
  }

  test("rounding: ceil, floor, rint, round, bround over doubles and integers") {
    // d2 = (id % 13) / 4 holds exact quarters and halves; d has NaN, infinities and nulls.
    checkVectorized("SELECT ceil(d) AS c, floor(d) AS f, rint(d) AS r, ceil(d2) AS c2, floor(d2) AS f2, rint(d2) AS r2, ceil(l) AS cl, floor(i) AS fi FROM t", Seq(Project))
    checkVectorized("SELECT round(d) AS r0, round(d, 2) AS r2, round(d, -1) AS rm, bround(d) AS b0, bround(d, 2) AS b2, bround(d, -1) AS bm FROM t", Seq(Project))
    checkVectorized("SELECT round(d2) AS r0, bround(d2) AS b0, round(-d2) AS rn, bround(-d2) AS bn, round(d2, 1) AS r1, bround(d2, 1) AS b1, round(d2 * 1.07, 3) AS r3 FROM t", Seq(Project))
    checkVectorized("SELECT round(i, -2) AS ri, bround(i, -2) AS bi, round(l - 30000, -3) AS rl, bround(l - 30000, -3) AS bl, round(i, 2) AS same, bround(l, 0) AS samel, round(i - 10000, -1) AS rneg FROM t", Seq(Project))
    checkVectorized("SELECT i FROM t WHERE round(d2) = 2.0 AND floor(d) < 100", Seq(Filter))
    // Overflow of an integer rounded up past its type: ANSI raises, legacy wraps like BigDecimal.intValue().
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT round(2147483600 + i, -3) AS wrapped FROM t WHERE i < 100", Seq(Filter, Project))
    }
    withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql("SELECT round(2147483600 + i, -3) AS r FROM t WHERE i < 100").collect())
      assert(causes(e).exists(c => c.isInstanceOf[ArithmeticException] && c.getMessage.contains("ARITHMETIC_OVERFLOW")), s"expected ARITHMETIC_OVERFLOW, got $e")
    }
    // Rows a filter removed never raise.
    checkVectorized("SELECT round(2147483000 + i, -3) AS r FROM t WHERE i < 400", Seq(Filter, Project))
    // The two-argument ceil over a double goes through decimal(30,15) in Spark and falls back (a non-literal scale is an analysis error in Spark itself).
    checkFallback("SELECT ceil(d, 1) AS c FROM t", Seq(Project), "not supported")
  }

  test("bitwise: & | ^ ~, the three shifts with literal and column amounts, bit_count") {
    checkVectorized("SELECT i & 255 AS a, i | 4096 AS o, i ^ 21845 AS x, ~i AS n, l & 65535 AS al, l | -1 AS ol, l ^ l AS xl, ~l AS nl, i & (i - 1) AS ii, l ^ CAST(i AS BIGINT) AS mixed FROM t", Seq(Project))
    // Literal and column amounts, incl. amounts at and past the width and negative ones (Java masks them, as Spark).
    checkVectorized("SELECT shiftleft(i, 3) AS sl, shiftright(i - 10000, 2) AS sr, shiftrightunsigned(i - 10000, 2) AS su, shiftleft(l, 40) AS sll, shiftright(l - 30000, 5) AS srl, shiftrightunsigned(l - 30000, 5) AS sul FROM t", Seq(Project))
    checkVectorized("SELECT shiftleft(i, i % 40) AS sl, shiftright(i - 10000, i % 70 - 3) AS sr, shiftrightunsigned(i - 10000, i % 33) AS su, shiftleft(l, i % 70) AS sll, shiftrightunsigned(l - 30000, i % 70) AS sul, shiftleft(1, i % 40) AS one, shiftleft(CAST(-1 AS BIGINT), i % 70) AS onel FROM t", Seq(Project))
    checkVectorized("SELECT shiftleft(i, 32) AS w32, shiftleft(i, 33) AS w33, shiftright(i - 10000, -1) AS neg, shiftrightunsigned(-1 - i, 1) AS unsigned32, shiftleft(l, 64) AS w64 FROM t", Seq(Project))
    // bit_count is Long.bitCount of the value widened: negative ints count 64 bits' worth.
    checkVectorized("SELECT bit_count(i) AS b, bit_count(i - 10000) AS bn, bit_count(l) AS bl, bit_count(~l) AS bnl FROM t", Seq(Project))
    checkVectorized("SELECT i FROM t WHERE i & 7 = 3 AND shiftright(i, 4) < 500 AND bit_count(i) > 3", Seq(Filter))
    // bit_get returns tinyint, which has no lane; byte/short/boolean operands likewise.
    checkFallback("SELECT bit_get(i, 3) AS g FROM t", Seq(Project), "bit_get returns tinyint")
    checkFallback("SELECT bit_count(b) AS c FROM t", Seq(Project), "bit_count over boolean not supported")
  }

  test("string predicates as projected booleans and in CASE conditions") {
    checkVectorized("SELECT s = 's1' AS eq, s <> 's1' AS ne, s < 's2' AS lt, s IN ('s1', 's17') AS inl, s LIKE 's1%' AS pre, s LIKE '%3' AS suf, contains(s, '2') AS has, i FROM t", Seq(Project))
    checkVectorized("SELECT CASE WHEN s = 's1' THEN 'one' WHEN s IN ('s2', 's3') THEN 'few' ELSE s END AS tag FROM t", Seq(Project))
    checkVectorized("SELECT IF(s > 's4', 1, 0) AS flag, i FROM t WHERE i > 100", Seq(Filter, Project))
  }

  test("forwarded and reordered columns, including strings and nulls") {
    checkVectorized("SELECT s, i, l, b, dt FROM t WHERE d > 1", Seq(Filter, Project))
    checkVectorized("SELECT dt AS when, s AS name, d2 * 2.0 AS twice FROM t WHERE l IS NOT NULL", Seq(Project))
  }

  test("chained filter and project both stay columnar") {
    val df = checkVectorized("SELECT i, d * d2 AS p FROM t WHERE d > 5 AND d2 > 0.5", Seq(Filter, Project))
    val project = nodesOf[VectorProjectExec](df).head
    assert(project.child.isInstanceOf[VectorFilterExec], project.treeString)
  }

  test("a filter below a projection forwards a selection bitmap instead of compacting") {
    val df = checkVectorized("SELECT s, i, d * d2 AS p, l FROM t WHERE d > 5 AND d2 > 0.5 AND s IS NOT NULL", Seq(Filter, Project))
    val filter = nodesOf[VectorFilterExec](df).head
    val project = nodesOf[VectorProjectExec](df).head
    assert(filter.emitSelection, "filter feeding our projection should emit a selection")
    assert(!project.emitSelection, "projection feeding Spark must compact")
    // Same query with selections disabled: identical rows, dense batches everywhere.
    withConf(VectorConf.SelectionEnabled -> "false") {
      val dense = checkVectorized("SELECT s, i, d * d2 AS p, l FROM t WHERE d > 5 AND d2 > 0.5 AND s IS NOT NULL", Seq(Filter, Project))
      assert(!nodesOf[VectorFilterExec](dense).head.emitSelection)
    }
  }

  test("ANSI errors are not raised for rows the filter removed") {
    // d2 is zero where id % 13 = 0; Spark never evaluates the projection for those rows.
    checkVectorized("SELECT i, 10.0 / d2 AS r FROM t WHERE d2 > 0.0", Seq(Filter, Project))
    // Same inside a conjunction: the right operand only matters where the left one holds.
    checkVectorized("SELECT i FROM t WHERE d2 > 0.0 AND 10.0 / d2 > 3.0", Seq(Filter))
  }

  test("selection survives an identity projection and reaches the aggregate") {
    val df = checkVectorized("SELECT count(*), sum(d), min(i), max(l), avg(d2) FROM t WHERE d > 5 AND d2 > 0.5", Seq(Filter, classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]))
    assert(nodesOf[VectorFilterExec](df).head.emitSelection)
    val grouped = checkVectorized("SELECT s, count(*), sum(d), count(l) FROM t WHERE d > 5 GROUP BY s", Seq(Filter, classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]))
    assert(nodesOf[VectorFilterExec](grouped).head.emitSelection)
    // Groups that only occur in filtered-out rows must not appear.
    checkVectorized("SELECT s, count(*) FROM t WHERE i < 30 GROUP BY s", Seq(Filter, classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]))
  }

  test("unsupported expressions in a projection fall back") {
    checkFallback("SELECT concat(s, 'x') AS c FROM t WHERE i > 5", Seq(Project), "unsupported expression")
    checkFallback("SELECT hash(i) AS m FROM t WHERE i > 5", Seq(Project), "unsupported expression")
  }

  test("project conversion can be disabled by configuration") {
    withConf(VectorConf.ProjectEnabled -> "false") {
      val df = withPlugin(enabled = true)(spark.sql("SELECT d * 2.0 AS x FROM t WHERE i > 5"))
      df.collect()
      assert(nodesOf[VectorProjectExec](df).isEmpty)
      assert(nodesOf[VectorFilterExec](df).nonEmpty)
    }
  }

  private def causes(t: Throwable): Seq[Throwable] =
    Iterator.iterate(t)(_.getCause).takeWhile(_ != null).take(10).toSeq
}
