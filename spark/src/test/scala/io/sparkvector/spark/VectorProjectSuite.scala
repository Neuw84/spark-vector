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
    checkFallback("SELECT CAST(s AS BINARY) AS a FROM t WHERE i > 5", Seq(Project), "unsupported cast")
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
    // sqrt: a negative argument is NaN, NaN and infinities pass through, integer arguments are cast by the analyzer.
    checkVectorized("SELECT sqrt(d2) AS s1, sqrt(i) AS s2, sqrt(l) AS s3, sqrt(d) AS s4, sqrt(i - 10000) AS s5 FROM t", Seq(Project))
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

  test("transcendental and trigonometric math: exp, log family, trig, hyperbolic, pow, atan2, hypot") {
    // Bit for bit against Spark (tolerance 0): the kernel makes exactly the Math / StrictMath call
    // Spark's generated code makes. d2 is 0..3 in quarter steps (0 where id % 13 = 0), i is 0..19999,
    // l is a nullable bigint, d carries NaN and infinities.
    def exact(sql: String, ops: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Seq(Project)): Unit = checkVectorized(sql, ops, tolerance = 0.0)
    // exp / expm1 and the log family: null at or below the asymptote (log(0), log2(0), log1p(-1)), NaN stays NaN.
    exact("SELECT exp(d2) AS a, expm1(d2 / 100) AS b, ln(i + 1) AS c, log(d2) AS d0, log2(i) AS e0, log10(l) AS f, log1p(d2 - 1.0) AS g, cbrt(i - 10000) AS h FROM t")
    exact("SELECT exp(d) AS a, expm1(d) AS b, ln(d) AS c, log10(d) AS d0, log1p(d) AS e0, log2(-d) AS f, cbrt(d) AS g FROM t")
    exact("SELECT exp(i) AS a, exp(-i) AS b, exp(l) AS c FROM t")
    // Trigonometric, hyperbolic and inverse hyperbolic: NaN outside the domain (asin past 1, acosh below 1, atanh past 1).
    exact("SELECT sin(d2) AS a, cos(d2) AS b, tan(d2) AS c, asin(d2 - 1.5) AS d0, acos(d2 / 3) AS e0, atan(i) AS f, atan(l - 30000) AS g FROM t")
    exact("SELECT sinh(d2) AS a, cosh(d2) AS b, tanh(l / 1000) AS c, asinh(d2 - 1.5) AS d0, acosh(d2) AS e0, atanh(d2 - 2) AS f, sinh(i) AS g FROM t")
    exact("SELECT sin(d) AS a, cos(d) AS b, tan(d) AS c, asin(d) AS d0, atan(d) AS e0, sinh(d) AS f, tanh(d) AS g, asinh(d) AS h, acosh(d) AS j, atanh(d) AS k FROM t")
    // The reciprocals (csc(0) and cot(0) are infinite) and the angle conversions.
    exact("SELECT cot(d2) AS a, sec(d2) AS b, csc(d2) AS c, degrees(d2) AS d0, radians(i) AS e0, degrees(l) AS f, radians(d) AS g FROM t")
    // Binary: pow / power with a literal on either side and its overflow to infinity, atan2 incl. signed zeros,
    // hypot, log(base, x) with a literal base or a literal argument and its null rules (either side non-positive).
    exact("SELECT pow(d2, i % 5) AS a, power(2.0, i % 60) AS b, pow(i, 0.5) AS c, pow(10.0, i) AS d0, pow(d2 - 1.5, 3) AS e0, pow(l, 2) AS f, pow(d, 2) AS g, pow(0.0, d2 - 1) AS h FROM t")
    exact("SELECT atan2(d2, i - 10000) AS a, atan2(d2 - 1.5, 0.0) AS b, atan2(0.0, d2 - 1.5) AS c, atan2(-d2 + d2, -1.0) AS d0, atan2(d, 1.0) AS e0, atan2(l, i) AS f FROM t")
    exact("SELECT hypot(d2, l) AS a, hypot(i, 3.0) AS b, hypot(d, 1.0) AS c, hypot(d2 - 1.5, i - 10000) AS d0 FROM t")
    exact("SELECT log(10, i) AS a, log(d2, 100) AS b, log(2, d2 - 1) AS c, log(0.5, d2) AS d0, log(i, d2) AS e0, log(-2.0, d2) AS f, log(d2, -1) AS g, log(d, d2) AS h, log(d2, d) AS j FROM t")
    // Folded constants and mixes in filters and aggregates.
    exact("SELECT pi() * d2 AS a, e() + i AS b, sin(pi() / 2 * d2) AS c, exp(1.0) - e() AS d0 FROM t")
    exact("SELECT i FROM t WHERE exp(d2 / 3) > 2.0 AND ln(i + 1) < 9.0 AND NOT isnan(tan(d2))", Seq(Filter))
    checkVectorized("SELECT i % 7 AS g, sum(exp(d2)), avg(log1p(i)), max(atan2(d2, i)), min(pow(d2, 2)) FROM t GROUP BY i % 7", Seq(Project, classOf[VectorHashAggregateExec]))
  }

  test("xxhash64 is Spark's, type by type") {
    // Bit for bit (tolerance is moot: longs compare exactly): ints, dates, longs, doubles incl. NaN and a
    // negative zero (normalised like Spark's), booleans, strings with nulls, several arguments, a literal argument.
    checkVectorized("SELECT xxhash64(i) AS a, xxhash64(l) AS b, xxhash64(d2) AS c, xxhash64(d) AS d0, xxhash64(-(d2 * 0.0)) AS e0, xxhash64(b) AS f, xxhash64(s) AS g, xxhash64(dt) AS h FROM t", Seq(Project))
    checkVectorized("SELECT xxhash64(i, l, s) AS a, xxhash64(s, 'salt', i) AS b, xxhash64(l, d2, b, dt) AS c, xxhash64(CAST(i AS DECIMAL(10,2))) AS d0 FROM t", Seq(Project))
    checkVectorized("SELECT i FROM t WHERE xxhash64(i) % 5 = 0 AND xxhash64(s, i) > 0", Seq(Filter))
  }

  test("datetime arithmetic: last_day, add_months, months_between, next_day, make_date, weekofyear, epoch scaling") {
    // Day-number relabels and epoch scaling, incl. negative instants.
    checkVectorized("SELECT unix_date(d0) AS a, date_from_unix_date(i - 2500) AS b, unix_micros(ts) AS c, unix_millis(ts) AS d1, unix_seconds(ts) AS e0, timestamp_micros(i * 1234567L) AS f, timestamp_millis(i * 100000L - 5000000000L) AS g, timestamp_seconds(i - 2500) AS h, timestamp_seconds(CAST(i AS BIGINT) * 100000) AS j FROM ts", Seq(Project))
    // Month and week arithmetic across month ends, leap days and pre-1970 dates (d0 spans 1969-12 .. 1971-01; dt spans the mixed table).
    checkVectorized("SELECT last_day(d0) AS a, add_months(d0, 1) AS b, add_months(d0, -13) AS c, add_months(d0, i % 30 - 15) AS d1, weekofyear(d0) AS e0, next_day(d0, 'Mon') AS f, next_day(d0, 'SUNDAY') AS g, next_day(d0, 'th') AS h, make_date(1970 + i % 60, i % 12 + 1, i % 28 + 1) AS j FROM ts", Seq(Project))
    checkVectorized("SELECT last_day(dt) AS a, add_months(dt, 6) AS b, weekofyear(dt) AS c, months_between(dt, date '2000-01-15') AS d1, months_between(date '2001-03-31', dt, false) AS e0 FROM t", Seq(Project))
    // months_between over timestamps and dates under UTC and fixed offsets; a zone with rules falls back.
    for (zone <- Seq("UTC", "+05:30", "-03:00")) {
      withConf("spark.sql.session.timeZone" -> zone) {
        checkVectorized("SELECT months_between(ts, timestamp '1970-01-31 10:30:00') AS a, months_between(timestamp '1969-10-30 00:00:00', ts) AS b, months_between(ts, d0) AS c, months_between(d0, ts, false) AS d1, months_between(ts, ts) AS e0 FROM ts", Seq(Project))
      }
    }
    withConf("spark.sql.session.timeZone" -> "America/New_York") {
      checkFallback("SELECT months_between(ts, d0) AS a FROM ts", Seq(Project), "fixed-offset session zone")
    }
    // make_date: invalid dates give null when ANSI is off, raise Spark's error under ANSI, never for a filtered row.
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT make_date(2023, i % 14, i % 32) AS a, make_date(i % 3 - 1, 2, 29) AS b FROM t", Seq(Project))
    }
    checkVectorized("SELECT make_date(2024, 2, i % 29 + 1) AS a FROM t WHERE i % 29 < 28", Seq(Project, Filter))
    val e = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT make_date(2023, 2, i % 30 + 1) AS a FROM t").collect()))
    assert(causes(e).exists(_.getMessage.contains("DATETIME_FIELD_OUT_OF_BOUNDS")), s"expected DATETIME_FIELD_OUT_OF_BOUNDS, got $e")
    // In a filter and as a grouping key.
    checkVectorized("SELECT i FROM ts WHERE last_day(d0) = d0 OR weekofyear(d0) = 53", Seq(Filter))
    checkVectorized("SELECT add_months(last_day(d0), 1) AS k, count(*) AS n, min(unix_seconds(ts)) AS m FROM ts GROUP BY add_months(last_day(d0), 1)", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: a non-literal day name, an unknown day name.
    checkFallback("SELECT next_day(d0, CASE WHEN i % 2 = 0 THEN 'MO' ELSE 'TU' END) AS a FROM ts", Seq(Project), "non-literal day name")
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkFallback("SELECT next_day(d0, 'Xy') AS a FROM ts", Seq(Project), "unknown day name")
    }
  }

  test("date_format, from_unixtime, unix_timestamp and date_trunc") {
    val units = Seq("MICROSECOND", "MILLISECOND", "SECOND", "MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR", "yy", "mon", "dd")
    for (zone <- Seq("UTC", "+05:30", "-03:00")) {
      withConf("spark.sql.session.timeZone" -> zone) {
        checkVectorized("SELECT date_format(ts, 'yyyy-MM-dd HH:mm:ss.SSSSSS') AS a, date_format(ts, 'EEEE, d MMM yyyy h:mm a') AS b, date_format(ts, 'Q/D/y XXX') AS c, date_format(d0, 'yyyy-MM-dd EEE') AS d1, from_unixtime(CAST(i AS BIGINT) * 100001 - 250000000) AS e0, from_unixtime(CAST(i AS BIGINT) * 3600, 'HH:mm dd/MM/yyyy') AS f, unix_timestamp(ts) AS g, unix_timestamp(d0) AS h, to_unix_timestamp(ts, 'yyyy') AS j, to_unix_timestamp(d0) AS k FROM ts", Seq(Project))
        checkVectorized("SELECT " + units.zipWithIndex.map { case (u, k) => s"date_trunc('$u', ts) AS t$k" }.mkString(", ") + ", date_trunc('MONTH', d0) AS dm, date_trunc('week', d0) AS dw FROM ts", Seq(Project))
      }
    }
    // A zone with rules: Spark's formatter handles a timestamp; a date's instant and date_trunc are ours and fall back.
    withConf("spark.sql.session.timeZone" -> "America/New_York") {
      checkVectorized("SELECT date_format(ts, 'yyyy-MM-dd HH:mm:ss zzz XXX') AS a, from_unixtime(CAST(i AS BIGINT) * 86400 * 30) AS b, unix_timestamp(ts) AS c FROM ts", Seq(Project))
      checkFallback("SELECT date_format(d0, 'yyyy-MM-dd') AS a FROM ts", Seq(Project), "fixed-offset session zone")
      checkFallback("SELECT unix_timestamp(d0) AS a FROM ts", Seq(Project), "fixed-offset session zone")
      checkFallback("SELECT date_trunc('HOUR', ts) AS a FROM ts", Seq(Project), "fixed-offset session zone")
    }
    // In a filter and as a grouping key.
    checkVectorized("SELECT count(*) AS n FROM ts WHERE date_format(ts, 'HH') = '07' AND unix_timestamp(ts) % 2 = 0", Seq(Filter, classOf[VectorHashAggregateExec]))
    checkVectorized("SELECT date_trunc('MONTH', ts) AS m, date_format(ts, 'yyyy-MM') AS k, count(*) AS n FROM ts GROUP BY date_trunc('MONTH', ts), date_format(ts, 'yyyy-MM')", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: parsing a string, a non-literal pattern, an unknown truncation unit (Spark gives null).
    checkFallback("SELECT unix_timestamp(CAST(d0 AS STRING), 'yyyy-MM-dd') AS a FROM ts", Seq(Project), "parsing a string")
    checkFallback("SELECT date_format(ts, CASE WHEN i % 2 = 0 THEN 'yyyy' ELSE 'MM' END) AS a FROM ts", Seq(Project), "non-literal pattern")
    checkFallback("SELECT date_trunc('DECADE', ts) AS a FROM ts", Seq(Project), "not supported")
  }

  test("casts: narrowing, booleans, date -> timestamp, to string") {
    // Legacy narrowing: Java's rule -- a long wraps, a double truncates toward zero and saturates, NaN is 0.
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT cast(l AS INT) AS a, cast(l * 1000000000 AS INT) AS b, cast(d AS INT) AS c, cast(d AS BIGINT) AS d1, cast(-d2 * 3 AS INT) AS e0, cast(d * 1e12 AS INT) AS f, cast(d * 1e300 AS BIGINT) AS g FROM t", Seq(Project))
    }
    // ANSI: in-range values agree; an out-of-range active row raises Spark's CAST_OVERFLOW; a filtered row never raises.
    checkVectorized("SELECT cast(l AS INT) AS a, cast(d2 * 1000 AS INT) AS b, cast(-d2 AS BIGINT) AS c, cast(l AS DOUBLE) AS d1 FROM t", Seq(Project))
    checkVectorized("SELECT cast(d AS INT) AS a, cast(d AS BIGINT) AS b FROM t WHERE d < 1000 AND d > -1000", Seq(Project, Filter))
    val overflow = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT cast(d AS INT) AS a FROM t").collect()))
    assert(causes(overflow).exists(_.getMessage.contains("CAST_OVERFLOW")), s"expected CAST_OVERFLOW, got $overflow")
    val wrap = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT cast(l * 1000000000 AS INT) AS a FROM t").collect()))
    assert(causes(wrap).exists(_.getMessage.contains("CAST_OVERFLOW")), s"expected CAST_OVERFLOW, got $wrap")
    // Booleans both ways, incl. NaN (true) and the string spellings Spark accepts.
    checkVectorized("SELECT cast(i AS BOOLEAN) AS a, cast(l AS BOOLEAN) AS b, cast(d AS BOOLEAN) AS c, cast(d2 AS BOOLEAN) AS d1, cast(b AS INT) AS e0, cast(b AS BIGINT) AS f, cast(b AS DOUBLE) AS g, cast(b AS STRING) AS h FROM t", Seq(Project))
    checkVectorized("SELECT cast(CASE i % 12 WHEN 0 THEN 'true' WHEN 1 THEN ' T ' WHEN 2 THEN 'Yes' WHEN 3 THEN '1' WHEN 4 THEN 'y' WHEN 5 THEN 'FALSE' WHEN 6 THEN 'f' WHEN 7 THEN 'no' WHEN 8 THEN '0' WHEN 9 THEN 'N' WHEN 10 THEN NULL ELSE 'tRuE' END AS BOOLEAN) AS a FROM t", Seq(Project))
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT cast(s AS BOOLEAN) AS a, cast(CASE WHEN i % 3 = 0 THEN 'yes' ELSE s END AS BOOLEAN) AS b FROM t", Seq(Project))
    }
    val badBool = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT cast(s AS BOOLEAN) AS a FROM t").collect()))
    assert(causes(badBool).exists(_.getMessage.contains("CAST_INVALID_INPUT")), s"expected CAST_INVALID_INPUT, got $badBool")
    checkVectorized("SELECT cast(s AS BOOLEAN) AS a FROM t WHERE s IS NULL", Seq(Project, Filter))
    // Date -> timestamp under UTC and fixed offsets; a zone with rules falls back.
    for (zone <- Seq("UTC", "+05:30", "-03:00")) {
      withConf("spark.sql.session.timeZone" -> zone) {
        checkVectorized("SELECT cast(dt AS TIMESTAMP) AS a FROM t", Seq(Project))
        checkVectorized("SELECT cast(d0 AS TIMESTAMP) AS a, cast(ts AS DATE) AS b FROM ts", Seq(Project))
      }
    }
    withConf("spark.sql.session.timeZone" -> "America/New_York") {
      checkFallback("SELECT cast(dt AS TIMESTAMP) AS a FROM t", Seq(Project), "fixed-offset session zone")
    }
    // To string: Java's toString for ints, longs and doubles (incl. NaN, the infinities, -0.0, exponents).
    checkVectorized("SELECT cast(i AS STRING) AS a, cast(l AS STRING) AS b, cast(d AS STRING) AS c, cast(d2 AS STRING) AS d1, cast(-d2 AS STRING) AS e0, cast(d * 1e20 AS STRING) AS f, cast(d / 1e10 AS STRING) AS g, cast(-i AS STRING) AS h FROM t", Seq(Project))
    // In a filter and as a grouping key.
    checkVectorized("SELECT count(*) AS n FROM t WHERE cast(d2 AS INT) = 1 AND cast(i AS BOOLEAN)", Seq(Filter, classOf[VectorHashAggregateExec]))
    checkVectorized("SELECT cast(d2 AS INT) AS k, cast(b AS STRING) AS s2, count(*) AS n FROM t GROUP BY cast(d2 AS INT), cast(b AS STRING)", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: try_cast into a decimal (the decimal path), and a cast outside the lane types.
    checkFallback("SELECT try_cast(d AS DECIMAL(10, 2)) AS a FROM t", Seq(Project), "try_cast")
    checkFallback("SELECT cast(s AS BINARY) AS a FROM t", Seq(Project), "unsupported cast")
  }

  test("casts: string -> number, date/timestamp <-> string") {
    val numbers = "CASE i % 16 WHEN 0 THEN ' 42 ' WHEN 1 THEN '-7' WHEN 2 THEN '12.9' WHEN 3 THEN '3e2' WHEN 4 THEN '2147483648' WHEN 5 THEN NULL WHEN 6 THEN 'abc' WHEN 7 THEN '' WHEN 8 THEN '+5' WHEN 9 THEN '1.5e300' WHEN 10 THEN 'NaN' WHEN 11 THEN ' -Infinity ' WHEN 12 THEN 'inf' WHEN 13 THEN '0x1p3' WHEN 14 THEN '1d' ELSE '9223372036854775807' END"
    // Legacy: Spark's own parsers per row -- trimmed, signed, fraction dropped for integers, null for the rest; specials for doubles.
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized(s"SELECT cast($numbers AS INT) AS a, cast($numbers AS BIGINT) AS b, cast($numbers AS DOUBLE) AS c, cast(s AS INT) AS d1, cast(s AS DOUBLE) AS e0 FROM t", Seq(Project))
    }
    // Round trips agree under ANSI (every value parses); a bad spelling raises Spark's CAST_INVALID_INPUT unless filtered away.
    checkVectorized("SELECT cast(cast(i AS STRING) AS INT) AS a, cast(cast(l AS STRING) AS BIGINT) AS b, cast(cast(d AS STRING) AS DOUBLE) AS c, cast(cast(-d2 AS STRING) AS DOUBLE) AS d1, cast(cast(i AS STRING) AS BIGINT) AS e0, cast(cast(l AS STRING) AS DOUBLE) AS f FROM t", Seq(Project))
    for (target <- Seq("INT", "BIGINT", "DOUBLE")) {
      val e = intercept[Exception](withPlugin(enabled = true)(spark.sql(s"SELECT cast(s AS $target) AS a FROM t").collect()))
      assert(causes(e).exists(_.getMessage.contains("CAST_INVALID_INPUT")), s"expected CAST_INVALID_INPUT for $target, got $e")
      checkVectorized(s"SELECT cast(s AS $target) AS a FROM t WHERE s IS NULL", Seq(Project, Filter))
    }
    // Dates and timestamps to strings with Spark's own formatters, under any zone; and back with Spark's own parser.
    val forms = "CASE i % 9 WHEN 0 THEN '2020-01-05' WHEN 1 THEN '2020-1-5' WHEN 2 THEN ' 2020-01-05T10:20:30 ' WHEN 3 THEN '2020-01-05 10:20:30.123456Z' WHEN 4 THEN '2020-01-05T10:20:30+05:30' WHEN 5 THEN '2020' WHEN 6 THEN NULL WHEN 7 THEN '1969-12-31 23:59:59.999' ELSE 'not a date' END"
    for (zone <- Seq("UTC", "+05:30", "America/New_York")) {
      withConf("spark.sql.session.timeZone" -> zone) {
        checkVectorized("SELECT cast(dt AS STRING) AS a, cast(ts AS STRING) AS b, cast(d0 AS STRING) AS c FROM t JOIN ts ON t.i = ts.i", Seq(Project))
        checkVectorized("SELECT cast(cast(dt AS STRING) AS DATE) AS a, cast(cast(ts AS STRING) AS TIMESTAMP) AS b, cast(cast(d0 AS STRING) AS TIMESTAMP) AS c, cast(cast(ts AS STRING) AS DATE) AS d1 FROM t JOIN ts ON t.i = ts.i", Seq(Project))
        withConf("spark.sql.ansi.enabled" -> "false") {
          checkVectorized(s"SELECT cast($forms AS DATE) AS a, cast($forms AS TIMESTAMP) AS b FROM t", Seq(Project))
        }
      }
    }
    val badDate = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT cast(s AS DATE) AS a FROM t").collect()))
    assert(causes(badDate).exists(_.getMessage.contains("CAST_INVALID_INPUT")), s"expected CAST_INVALID_INPUT, got $badDate")
    val badTs = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT cast(s AS TIMESTAMP) AS a FROM t").collect()))
    assert(causes(badTs).exists(_.getMessage.contains("CAST_INVALID_INPUT")), s"expected CAST_INVALID_INPUT, got $badTs")
    checkVectorized("SELECT cast(s AS DATE) AS a, cast(s AS TIMESTAMP) AS b FROM t WHERE s IS NULL", Seq(Project, Filter))
    // In a filter and as a grouping key.
    checkVectorized("SELECT count(*) AS n FROM t WHERE cast(cast(i AS STRING) AS INT) % 2 = 0 AND cast(dt AS STRING) < '2021'", Seq(Filter, classOf[VectorHashAggregateExec]))
    checkVectorized("SELECT cast(dt AS STRING) AS k, count(*) AS n FROM t GROUP BY cast(dt AS STRING)", Seq(Project, classOf[VectorHashAggregateExec]))
  }

  test("try_add, try_subtract, try_multiply, try_divide, try_mod and try_cast") {
    for (ansi <- Seq("true", "false")) {
      withConf("spark.sql.ansi.enabled" -> ansi) {
        // Some rows overflow, others do not: those rows are null, the rest are the exact result; the input's nulls stay null.
        checkVectorized("SELECT try_add(i, 2147483000) AS a, try_add(2147483000, i) AS b, try_subtract(-2147483000, i) AS c, try_multiply(i, 2147483) AS d1, try_multiply(l, l * l * l) AS e0, try_add(l, l) AS f, try_subtract(i, i) AS g, try_multiply(d, 1e300) AS h FROM t", Seq(Project))
        // try_divide is floating point and null on a zero divisor; try_mod nulls a zero divisor too.
        checkVectorized("SELECT try_divide(i, i % 5) AS a, try_divide(d, d2) AS b, try_divide(l, 0) AS c, try_divide(1, d2 - 1.5) AS d1, try_mod(i, i % 7) AS e0, try_mod(l, 4) AS f, try_mod(d, d2) AS g, try_mod(i, 0) AS h FROM t", Seq(Project))
        // try_cast: the #43 cast nodes with null instead of raise -- narrowing, strings, dates, booleans.
        checkVectorized("SELECT try_cast(d AS INT) AS a, try_cast(d AS BIGINT) AS b, try_cast(l * 1000000000 AS INT) AS c, try_cast(s AS INT) AS d1, try_cast(s AS DOUBLE) AS e0, try_cast(s AS BOOLEAN) AS f, try_cast(s AS DATE) AS g, try_cast(s AS TIMESTAMP) AS h, try_cast(cast(i AS STRING) AS INT) AS j, try_cast(d2 * 3 AS INT) AS k FROM t", Seq(Project))
      }
    }
    // Mixed with checked arithmetic on the same batch: the try_* result is null where the checked form would raise, nothing else changes.
    checkVectorized("SELECT try_multiply(i, 2147483) AS a, i * 2 AS b FROM t WHERE i < 100000", Seq(Project, Filter))
    // In a filter and as a grouping key.
    checkVectorized("SELECT count(*) AS n, count(try_add(i, 2147483000)) AS m FROM t WHERE try_divide(i, i % 3) IS NULL", Seq(Filter, classOf[VectorHashAggregateExec]))
    checkVectorized("SELECT try_cast(d AS INT) AS k, count(*) AS n FROM t GROUP BY try_cast(d AS INT)", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: decimals keep their own path (scalar and aggregate).
    checkFallback("SELECT try_add(CAST(i AS DECIMAL(10, 2)), CAST(l AS DECIMAL(12, 2))) AS a FROM t", Seq(Project), "try_*")
    checkFallback("SELECT try_sum(CAST(l AS DECIMAL(12, 2))) AS a FROM t", Seq(classOf[VectorHashAggregateExec]), "try_sum over a decimal")
  }

  test("structural: literals of every type, typed nulls, aliases and reason texts") {
    // Every literal type as a projected column, next to real columns and alone; typed nulls are all-invalid columns.
    checkVectorized("SELECT 1 AS a, 12345678901L AS b, 1.5D AS c, DATE '2020-02-29' AS d1, TIMESTAMP '2020-02-29 12:34:56.789' AS e0, 'lit' AS f, CAST(2.50 AS DECIMAL(10, 2)) AS g, -7 AS h, i FROM t", Seq(Project))
    checkVectorized("SELECT CAST(NULL AS INT) AS a, CAST(NULL AS BIGINT) AS b, CAST(NULL AS DOUBLE) AS c, CAST(NULL AS STRING) AS d1, CAST(NULL AS DATE) AS e0, CAST(NULL AS TIMESTAMP) AS f, CAST(NULL AS DECIMAL(10, 2)) AS g, i FROM t", Seq(Project))
    checkVectorized("SELECT 1 AS a, 'x' AS b, CAST(NULL AS INT) AS c FROM t WHERE i > 5", Seq(Project, Filter))
    // The nulls ordinary SQL leaves in expressions: nullif, a CASE branch, coalesce, if.
    checkVectorized("SELECT nullif(i, 3) AS a, CASE WHEN i % 2 = 0 THEN NULL ELSE i END AS b, coalesce(CAST(NULL AS INT), i) AS c, if(b, NULL, s) AS d1, nullif(s, 's3') AS e0 FROM t", Seq(Project))
    // Aliases: nested and over a subquery are transparent; an alias over an unsupported child reports the child's reason.
    checkVectorized("SELECT x AS y, (y2 + 1) AS z FROM (SELECT i AS x, l AS y2 FROM t) sub", Seq(Project))
    checkFallback("SELECT soundex(s) AS renamed FROM t", Seq(Project), "soundex")
    // Boolean literals are deliberately not constant columns (predicates over them take other paths).
    checkFallback("SELECT true AS b2 FROM t", Seq(Project), "unsupported literal type boolean")
    // Fallback reasons name what they refused -- the strings the suites, the UI and docs/expressions.md quote.
    val reasons = Seq(
      "SELECT soundex(s) AS a FROM t" -> "soundex",
      "SELECT cast(s AS BINARY) AS a FROM t" -> "binary",
      "SELECT date_format(dt, CASE WHEN i % 2 = 0 THEN 'yyyy' ELSE 'MM' END) AS a FROM t" -> "date_format",
      "SELECT try_cast(d AS DECIMAL(10, 2)) AS a FROM t" -> "try_cast",
      "SELECT collate(s, 'UNICODE_CI') AS a FROM t" -> "ollat")
    for ((sql, needle) <- reasons) checkFallback(sql, Seq(Project), needle)
  }

  test("hash, xxhash64 seeds, md5, sha1, sha2 and crc32") {
    // hash must be Spark's exact Murmur3 -- every supported type incl. nulls, NaN, -0.0, dates, timestamps, short decimals, several columns, a seed.
    checkVectorized("SELECT hash(i) AS a, hash(l) AS b, hash(d) AS c, hash(d2) AS d0, hash(b) AS e0, hash(s) AS f, hash(dt) AS g, hash(CAST(d2 AS DECIMAL(10, 2))) AS h, hash(i, l, s, b, d) AS j, hash(-0.0d * i) AS k FROM t", Seq(Project))
    checkVectorized("SELECT hash(ts) AS a, hash(ts, i) AS b FROM ts", Seq(Project))
    checkVectorized("SELECT hash(i, s) AS a, xxhash64(i, s) AS b FROM t", Seq(Project))
    // Digests over the string's bytes (Spark's string -> binary cast), hex out; sha2's bit lengths incl. 0 and an unknown one.
    checkVectorized("SELECT md5(s) AS a, sha1(s) AS b, sha(s) AS c, sha2(s, 256) AS d0, sha2(s, 0) AS e0, sha2(s, 224) AS f, sha2(s, 384) AS g, sha2(s, 512) AS h, sha2(s, 100) AS j, crc32(s) AS k, md5(concat(s, '日本')) AS m FROM t", Seq(Project))
    // In a filter and as a grouping key; hash of a literal folds in Spark.
    checkVectorized("SELECT i FROM t WHERE pmod(hash(s), 7) = 3 AND crc32(s) > 1000000", Seq(Filter))
    checkVectorized("SELECT pmod(hash(s, i % 3), 5) AS k, count(*) AS n, min(md5(s)) AS m FROM t GROUP BY pmod(hash(s, i % 3), 5)", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: a non-literal sha2 bit length.
    checkFallback("SELECT sha2(s, i) AS a FROM t", Seq(Project), "non-literal bit length")
  }

  test("instr, locate, replace, translate, substring_index, split_part and find_in_set") {
    // Haystacks with repeats, multi-byte text, commas and dots; needles as literals and lanes.
    val h = "CASE WHEN i % 6 = 0 THEN 'www.apache.org' WHEN i % 6 = 1 THEN '日本語テキスト日本' WHEN i % 6 = 2 THEN 'a,b,,c' WHEN i % 6 = 3 THEN '😀x😀y😀' WHEN i % 6 = 4 THEN '' ELSE concat(s, s) END"
    val nd = "CASE WHEN i % 5 = 0 THEN '.' WHEN i % 5 = 1 THEN '日本' WHEN i % 5 = 2 THEN '' WHEN i % 5 = 3 THEN 's' ELSE NULL END"
    // instr / locate / position: code-point positions, the empty needle, a start beyond the end, non-positive and null starts.
    checkVectorized(s"SELECT instr($h, 'a') AS a, instr($h, $nd) AS b, locate('s', $h) AS c, locate($nd, $h, 3) AS d0, position('本' IN $h) AS e0, locate('x', $h, i % 4 - 1) AS f, locate('a', $h, CASE WHEN i % 3 = 0 THEN NULL ELSE 1 END) AS g, instr(s, '1') AS h0 FROM t", Seq(Project))
    // replace: literal and lane search/replacement, an empty search, deletion, multi-byte.
    checkVectorized(s"SELECT replace($h, 'a', '<>') AS a, replace($h, $nd, '-') AS b, replace($h, 'x') AS c, replace(s, s, 'same') AS d0, replace($h, '日本', '🙂') AS e0 FROM t", Seq(Project))
    // translate: deletion, a repeated matching character (first mapping wins), multi-byte from/to.
    checkVectorized(s"SELECT translate($h, 'a.,', '1;') AS a, translate($h, 'aa日😀', '12本!') AS b, translate(s, 's0123456789', 'S') AS c FROM t", Seq(Project))
    // substring_index: positive and negative counts, count 0, empty delimiter, missing delimiter, lane counts.
    checkVectorized(s"SELECT substring_index($h, '.', 1) AS a, substring_index($h, '.', -1) AS b, substring_index($h, ',', 2) AS c, substring_index($h, '本', -2) AS d0, substring_index($h, '.', 0) AS e0, substring_index($h, '', 1) AS f, substring_index($h, 'zz', 1) AS g, substring_index($h, $nd, i % 5 - 2) AS h0 FROM t", Seq(Project))
    // split_part: from either end, past the ends, an empty delimiter, a lane part; find_in_set incl. a comma in the word.
    checkVectorized(s"SELECT split_part($h, '.', 2) AS a, split_part($h, ',', -1) AS b, split_part($h, ',', 9) AS c, split_part($h, '', 1) AS d0, split_part($h, '.', i % 3 + 1) AS e0, split_part($h, '本', -2) AS f FROM t", Seq(Project))
    checkVectorized(s"SELECT find_in_set('b', $h) AS a, find_in_set(s, concat('x,', s, ',y')) AS b, find_in_set('a,b', $h) AS c, find_in_set('', $h) AS d0, find_in_set($nd, 'www.apache.org,日本,,s') AS e0 FROM t", Seq(Project))
    // A zero part raises Spark's INVALID_INDEX_OF_ZERO in both engines; a filtered row never raises.
    checkVectorized("SELECT split_part(s, '1', i % 2 + 1) AS a FROM t WHERE i % 2 = 0", Seq(Project, Filter))
    val e = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT split_part(s, '1', i % 2) AS a FROM t").collect()))
    assert(causes(e).exists(_.getMessage.contains("INVALID_INDEX_OF_ZERO")), s"expected INVALID_INDEX_OF_ZERO, got $e")
    // In a filter and as a grouping key.
    checkVectorized("SELECT i FROM t WHERE instr(s, '1') = 2 OR substring_index(s, '2', 1) = 's'", Seq(Filter))
    checkVectorized("SELECT split_part(concat(s, '-', s), '-', 2) AS k, count(*) AS n, max(locate('s', s)) AS m FROM t GROUP BY split_part(concat(s, '-', s), '-', 2)", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: translate with a column from/to string.
    checkFallback("SELECT translate(s, s, 'x') AS a FROM t", Seq(Project), "non-literal from/to")
  }

  test("upper, lower, initcap and the trims") {
    // Rows the ASCII path decides, and rows it hands to Spark's own implementation: multi-byte letters,
    // the German sharp s, a Turkish dotted capital I, digits and punctuation at word starts.
    val m = "CASE WHEN i % 8 = 0 THEN 'héllo wörld' WHEN i % 8 = 1 THEN 'straße' WHEN i % 8 = 2 THEN 'İstanbul ışık' WHEN i % 8 = 3 THEN '3abc o''neil a-b' WHEN i % 8 = 4 THEN 'ǅemal  two  spaces' WHEN i % 8 = 5 THEN '' WHEN i % 8 = 6 THEN 'MiXeD Case 42' ELSE s END"
    checkVectorized(s"SELECT upper(s) AS a, lower(s) AS b, initcap(s) AS c, ucase($m) AS d0, lcase($m) AS e0, initcap($m) AS f, upper(concat(s, ' ', s)) AS g FROM t", Seq(Project))
    // The same under the pre-ICU JVM case mappings.
    withConf("spark.sql.icu.caseMappings.enabled" -> "false") {
      checkVectorized(s"SELECT upper($m) AS a, lower($m) AS b, initcap($m) AS c FROM t", Seq(Project))
    }
    // Trims: spaces by default (only ASCII 32), a literal trim set incl. multi-byte code points, every form.
    val p = "CASE WHEN i % 5 = 0 THEN concat('  ', s, '   ') WHEN i % 5 = 1 THEN concat(s, ' ') WHEN i % 5 = 2 THEN '   ' WHEN i % 5 = 3 THEN 'xyxhixyx' ELSE 'ééaéé' END"
    checkVectorized(s"SELECT trim($p) AS a, ltrim($p) AS b, rtrim($p) AS c, btrim($p) AS d0, trim(BOTH FROM $p) AS e0, trim(LEADING FROM $p) AS f, trim(TRAILING FROM $p) AS g FROM t", Seq(Project))
    checkVectorized(s"SELECT trim('xy' FROM $p) AS a, trim(BOTH 'éx' FROM $p) AS b, trim(LEADING 'xé' FROM $p) AS c, trim(TRAILING 'x' FROM $p) AS d0, ltrim('yx', $p) AS e0, rtrim('é', $p) AS f, btrim($p, ' x') AS g FROM t", Seq(Project))
    // In a filter and as a grouping key; nested into the other string functions.
    checkVectorized("SELECT i FROM t WHERE upper(s) = 'S1' OR trim(concat(' ', s)) = 's2'", Seq(Filter))
    checkVectorized("SELECT upper(substring(s, 1, 1)) AS k, count(*) AS n, min(length(trim(concat(s, '  ')))) AS l FROM t GROUP BY upper(substring(s, 1, 1))", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: a collated column follows ICU rules; a trim set from a column.
    checkFallback("SELECT upper(collate(s, 'UTF8_LCASE')) AS a FROM t", Seq(Project), "collated string follows ICU rules")
    checkFallback("SELECT trim(s FROM concat('s', s)) AS a FROM t", Seq(Project), "non-literal trim string")
  }

  test("concat, concat_ws and elt over several string inputs") {
    val m = "CASE WHEN i % 4 = 0 THEN 'héllo' WHEN i % 4 = 1 THEN '日本語' WHEN i % 4 = 2 THEN '😀' ELSE s END"
    val n = "CASE WHEN i % 3 = 0 THEN NULL ELSE s END"
    // concat: null-intolerant; literals between lanes; many inputs; multi-byte; nested in the other string functions.
    checkVectorized(s"SELECT concat(s, '-', $m) AS a, concat(s, $n) AS b, concat(s) AS c, concat(s, s, s, s, s, s, s, s) AS d0, concat('<', $m, '|', $n, '>') AS e0, upper_cased AS f FROM (SELECT *, concat(substring(s, 1, 1), lpad(s, 4, '0')) AS upper_cased FROM t)", Seq(Project))
    // concat_ws: literal and lane separators, null separator, nulls skipped, a single live input, no live input.
    checkVectorized(s"SELECT concat_ws(',', s, $n, $m) AS a, concat_ws($n, s, 'x') AS b, concat_ws('', s, s) AS c, concat_ws(' / ', $n, $n) AS d0, concat_ws(s, 'a', 'b', $n) AS e0, concat_ws('・', $m, $n, s) AS f FROM t", Seq(Project))
    // elt: lane and literal indices, out-of-range and null indices give null when ANSI is off, a null pick.
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized(s"SELECT elt(i % 4, s, $m, 'lit') AS a, elt(2, s, $n) AS b, elt(i % 3 - 1, s, s) AS c, elt(CASE WHEN i % 5 = 0 THEN NULL ELSE 1 END, s) AS d0 FROM t", Seq(Project))
    }
    // ANSI: an out-of-range index raises Spark's INVALID_ARRAY_INDEX; a filtered row never raises.
    checkVectorized(s"SELECT elt(i % 2 + 1, s, $m) AS a FROM t", Seq(Project))
    checkVectorized("SELECT elt(i, s, s, s) AS a FROM t WHERE i BETWEEN 1 AND 3", Seq(Project, Filter))
    val e = intercept[Exception](withPlugin(enabled = true)(spark.sql("SELECT elt(i % 4, s, s) AS a FROM t").collect()))
    assert(causes(e).exists(_.getMessage.contains("INVALID_ARRAY_INDEX")), s"expected INVALID_ARRAY_INDEX, got $e")
    // In a filter and as a grouping key.
    checkVectorized("SELECT i FROM t WHERE concat(s, '!') = 's1!' OR concat_ws('', s, s) = 's2s2'", Seq(Filter))
    checkVectorized("SELECT concat_ws('-', s, substring(s, 2)) AS k, count(*) AS n FROM t GROUP BY concat_ws('-', s, substring(s, 2))", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: binary and array forms.
    checkFallback("SELECT concat(CAST(s AS BINARY), CAST(s AS BINARY)) AS a FROM t", Seq(Project), "unsupported")
    checkFallback("SELECT concat_ws(',', array(s, s)) AS a FROM t", Seq(Project), "unsupported")
  }

  test("length, octet_length, bit_length, ascii and chr") {
    val m = "CASE WHEN i % 4 = 0 THEN 'héllo wörld' WHEN i % 4 = 1 THEN '日本語テキスト' WHEN i % 4 = 2 THEN '😀x😀y' ELSE s END"
    // Every alias, over ASCII with nulls and over multi-byte text: code points vs bytes vs bits.
    checkVectorized(s"SELECT length(s) AS a, len(s) AS b, char_length(s) AS c, character_length(s) AS d0, octet_length(s) AS e0, bit_length(s) AS f, length($m) AS g, octet_length($m) AS h, bit_length($m) AS j FROM t", Seq(Project))
    // ascii: the first code point, 0 on the empty string, multi-byte first characters, nulls.
    checkVectorized(s"SELECT ascii(s) AS a, ascii($m) AS b, ascii(CASE WHEN i % 5 = 0 THEN '' ELSE s END) AS c FROM t", Seq(Project))
    // chr over ints and longs: negatives, 0, 128..255 as two-byte characters, n % 256 wrap; char alias.
    checkVectorized("SELECT chr(i % 300) AS a, chr(i - 10) AS b, char(l % 512) AS c, chr(65 + i % 26) AS d0, chr(CAST(i AS BIGINT) * 100000) AS e0 FROM t", Seq(Project))
    // In a filter, as a grouping key, nested in arithmetic and in the slicing functions.
    checkVectorized("SELECT i FROM t WHERE length(s) = 3 AND ascii(s) = 115", Seq(Filter))
    checkVectorized(s"SELECT length(s) AS k, count(*) AS n, sum(octet_length($m)) AS b FROM t GROUP BY length(s)", Seq(Project, classOf[VectorHashAggregateExec]))
    checkVectorized(s"SELECT length($m) * 2 + bit_length(s) AS a, substring($m, length($m) - 1) AS b, lpad(s, length(s) + 2, chr(42)) AS c FROM t", Seq(Project))
    // Declined: a binary subject (its length is bytes, not a lane), chr of a double.
    checkFallback("SELECT length(CAST(s AS BINARY)) AS a FROM t", Seq(Project), "unsupported")
  }

  test("substring, left/right, lpad/rpad, repeat, space, overlay write new strings") {
    // s is 's0'..'s49' with nulls; m mixes multi-byte text in (2- and 3-byte code points and a 4-byte emoji).
    val m = "CASE WHEN i % 4 = 0 THEN 'héllo wörld' WHEN i % 4 = 1 THEN '日本語テキスト' WHEN i % 4 = 2 THEN '😀x😀y' ELSE s END"
    // substring: the TPC-H Q22 prefix, the tail, position 0, negative and out-of-range positions, zero and negative lengths, lanes as arguments.
    checkVectorized("SELECT substring(s, 1, 2) AS a, substr(s, 2) AS b, substring(s, -1) AS c, substring(s, 0, 1) AS d0, substring(s, 5, 10) AS e0, substring(s, 2, 0) AS f, substring(s, -50, 3) AS g, substring(s, 2, -1) AS h FROM t", Seq(Project))
    checkVectorized("SELECT substring(s, i % 4, i % 3) AS a, substring(s, i % 5 - 2) AS b, substring(s FROM 2 FOR 1) AS c FROM t", Seq(Project))
    checkVectorized(s"SELECT substring($m, 2, 3) AS a, substring($m, -3) AS b, substring($m, 1, 1) AS c, substring($m, 4, 100) AS d0, substring($m, -100, 5) AS e0 FROM t", Seq(Project))
    // left / right are Spark's rewrites onto substring (right through an If over the length).
    checkVectorized(s"SELECT left(s, 1) AS a, right(s, 2) AS b, left(s, i % 3) AS c, right(s, -1) AS d0, left($m, 4) AS e0, right($m, 2) AS f, right(s, i % 4) AS g FROM t", Seq(Project))
    // lpad / rpad: literal and lane lengths, a multi-byte pad, a lane pad, an empty pad, truncation, a zero length.
    checkVectorized(s"SELECT lpad(s, 6, '*') AS a, rpad(s, 6) AS b, lpad(s, 1, 'xy') AS c, rpad($m, 9, 'ñ') AS d0, lpad(s, i % 8, '-') AS e0, rpad(s, 5, s) AS f, lpad(s, 4, '') AS g, lpad($m, 0, 'x') AS h, rpad(s, 7, '日本') AS j FROM t", Seq(Project))
    // repeat and space, incl. non-positive counts and a lane count.
    checkVectorized(s"SELECT repeat(s, 2) AS a, repeat(s, i % 3) AS b, repeat(s, -1) AS c, repeat($m, 2) AS d0, space(i % 5) AS e0, space(i % 3 - 1) AS f FROM t", Seq(Project))
    // overlay: the PLACING form, a zero length, the default length, a lane position, a lane replacement, out-of-range positions.
    checkVectorized(s"SELECT overlay(s PLACING '_' FROM 2) AS a, overlay(s, 'XY', 1, 3) AS b, overlay($m, 'ab', 3, 0) AS c, overlay(s, 'q', i % 3 + 1, 1) AS d0, overlay(s, s, 2, -1) AS e0, overlay(s, 'z', 100, 1) AS f, overlay($m, 'Ü', -2, 2) AS g FROM t", Seq(Project))
    // In a filter, as a grouping key and inside a comparison.
    checkVectorized("SELECT i FROM t WHERE substring(s, 2, 1) = '1' AND left(s, 1) = 's'", Seq(Filter))
    checkVectorized("SELECT substring(s, 1, 2) AS k, count(*) AS n, min(rpad(s, 4, '.')) AS r FROM t GROUP BY substring(s, 1, 2)", Seq(Project, classOf[VectorHashAggregateExec]))
    // Declined: a binary subject, and a literal count past the batch output cap (Spark would try to allocate it).
    checkFallback("SELECT substring(CAST(s AS BINARY), 1, 2) AS a FROM t", Seq(Project), "unsupported")
    checkFallback("SELECT repeat(s, 2000000) AS a FROM t WHERE i < 2", Seq(Project), "exceeds the batch output cap")
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

  test("null-safe equality, isnan, boolean comparisons and InSet as projected booleans") {
    checkVectorized("SELECT l <=> CAST(i AS BIGINT) * 3 AS ns, s <=> 's7' AS str, d <=> d2 AS dd, isnan(d) AS n, NOT isnan(d) AS nn FROM t", Seq(Project))
    checkVectorized("SELECT b = true AS bt, b <> (i % 2 = 0) AS bne, b < (i % 2 = 0) AS blt, (i % 2 = 0) >= b AS bge, b <=> (i % 3 = 0) AS bns FROM t", Seq(Project))
    checkVectorized("SELECT i IN (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233) AS fib, l IN (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33) AS lset, d2 IN (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75) AS dset, s IN ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12') AS sset FROM t", Seq(Project))
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
    checkFallback("SELECT reverse(s) AS c FROM t WHERE i > 5", Seq(Project), "unsupported expression")
    checkFallback("SELECT soundex(s) AS m FROM t WHERE i > 5", Seq(Project), "unsupported expression")
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
