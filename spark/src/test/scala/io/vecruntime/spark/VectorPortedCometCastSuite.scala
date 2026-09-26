/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.vecruntime.spark

import io.vecruntime.spark.test.VectorQuerySuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.vector.VectorProjectExec

/**
 * SQL correctness coverage ported from DataFusion Comet's `CometCastSuite`, adapted to
 * spark-vector's validation model: each `CAST` is run twice on the same session with
 * `spark.vector.enabled` toggled and the rows compared (Comet's `checkSparkAnswerAndOperator`),
 * asserting the cast ran on our operator -- a `Cast` is a projection, so the assertion is a
 * `VectorProjectExec` in the final plan.
 *
 * This is the cast slice of the SQL-coverage survey. Comet's suite is a matrix of one test per
 * `(from, to)` pair over generated columns with edge values (min/max, zero, NaN, +/-Infinity,
 * nulls, and for strings valid and invalid numeric/date spellings). We keep exactly the pairs
 * spark-vector compiles, pinned in `docs/expressions.md`'s `Cast` rows: the lane source and target
 * types (INT32, INT64, FLOAT64, boolean, date, timestamp, string, decimal(<=18)). Types with no
 * lane in this project (tinyint / smallint / float / binary) are the fallback cases below -- Spark
 * accepts the cast, our planner declines it, and the query stays on Spark.
 *
 * Where a cast's semantics differ between ANSI and legacy mode (the narrowing and string-parse
 * casts, which null in legacy and raise on the same out-of-range input under ANSI) the case is run
 * in BOTH modes: legacy asserts the row nulls and stays on our operator; ANSI runs over an
 * in-range column so the accelerated result matches Spark's without a raise (the correctness of
 * the ANSI raise itself is a separate suite's concern). The session zone is fixed to UTC so the
 * date/timestamp <-> string and timestamp -> date casts, which need a fixed-offset zone, compile.
 *
 * A case that uncovered an engine bug would be marked `ignore` here with a one-line reason and a
 * minimal repro in the PR, per the task's rule not to fix engine code in a coverage change; none
 * did on this slice.
 */
class VectorPortedCometCastSuite extends VectorQuerySuite {

  private val Project = classOf[VectorProjectExec]

  // Edge values Comet's generators inject, per type, plus a null. Kept small and explicit so the
  // comparison is over the interesting rows, not a random cloud (the base suite already fuzzes
  // widely through the mixed table in the expression suite).
  private val ints = Seq(Int.MinValue, Int.MaxValue, -1, 0, 1, 42, -128, 127, -32768, 32767, 100000)
  private val longs = Seq(Long.MinValue, Long.MaxValue, -1L, 0L, 1L, 42L, Int.MaxValue.toLong + 1)
  private val doubles = Seq(0.0d, -0.0d, 1.5d, -1.5d, 42.0d, 1e18d, -1e18d, 123.456d, 0.999d)
  private val doublesSpecial = doubles ++ Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity)
  private val bools = Seq(true, false)

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    // A fixed-offset zone is required for the date/timestamp <-> string and timestamp -> date casts.
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    def view(name: String, tpe: DataType, values: Seq[Any]): Unit = {
      val rows = (values.map(v => Row(v)) :+ Row(null)).toList
      val rdd = spark.sparkContext.parallelize(rows, 2)
      val df = spark.createDataFrame(rdd, StructType(Seq(StructField("a", tpe, nullable = true))))
      // Write and re-read as Parquet: the plugin only accelerates over a columnar scan (a row-based
      // ExistingRDD scan falls back with "child Scan ExistingRDD is not columnar"), exactly as the
      // expression suite reads its mixed table back from Parquet.
      val path = newTempPath(s"cast/$name")
      df.write.mode("overwrite").parquet(path)
      spark.read.parquet(path).createOrReplaceTempView(name)
    }

    view("ci", IntegerType, ints)
    view("cl", LongType, longs)
    view("cd", DoubleType, doubles)
    view("cdsmall", DoubleType, Seq(0.0d, -0.0d, 1.5d, -1.5d, 42.0d, 123.456d, 0.999d, -1000.0d, 1000.0d))
    view("cdx", DoubleType, doublesSpecial) // includes NaN / +/-Inf for legacy narrowing
    view("cb", BooleanType, bools)
    view("cdec", DecimalType(10, 2), Seq("0.00", "1.23", "-1.23", "9999999.99", "-42.50").map(BigDecimal(_)))

    // Strings: valid and invalid spellings for the string -> {number,bool,date} parse casts.
    view(
      "cs_num",
      StringType,
      Seq("0", "1", "-1", "42", "2147483647", "-2147483648", " 12 ", "3.14", "1e3", "abc", "", "+")
    )
    view("cs_bool", StringType, Seq("t", "true", "Y", "yes", "1", "f", "false", "N", "no", "0", "maybe", ""))
    view("cs_date", StringType, Seq("2020-01-01", "1999-12-31", "2020-02-29", "not-a-date", "2020-13-40", ""))

    // Date / timestamp source columns.
    view("cdate", DateType, Seq("2020-01-01", "1969-12-31", "2000-02-29", "2038-01-19").map(java.sql.Date.valueOf))
    view(
      "cts",
      TimestampType,
      Seq("2020-01-01 12:34:56", "1969-12-31 23:59:59", "2000-02-29 00:00:00").map(java.sql.Timestamp.valueOf)
    )
  }

  /** CAST(a AS to) over `view`, identical rows to Spark and a VectorProjectExec in the plan. */
  private def castOk(view: String, to: String): Unit =
    checkVectorized(s"SELECT CAST(a AS $to) AS c FROM $view", Seq(Project))

  private def castOkAnsi(view: String, to: String): Unit =
    withConf("spark.sql.ansi.enabled" -> "true")(castOk(view, to))

  private def castLegacy(view: String, to: String): Unit =
    withConf("spark.sql.ansi.enabled" -> "false")(castOk(view, to))

  /** CAST(a AS to) must fall back to Spark (no VectorProjectExec) with a reason. */
  private def castFallback(view: String, to: String, reason: String): Unit =
    checkFallback(s"SELECT CAST(a AS $to) AS c FROM $view", Seq(Project), reason)

  // ---------------------------------------------------------------------------
  // Numeric widening: exact in either mode.
  // ---------------------------------------------------------------------------

  test("cast int -> bigint / double") {
    castOk("ci", "bigint")
    castOk("ci", "double")
  }

  test("cast bigint -> double") {
    castOk("cl", "double")
  }

  // ---------------------------------------------------------------------------
  // Narrowing (NarrowCastExpr): legacy wraps/truncates, ANSI raises on out-of-range.
  // Legacy runs over the full edge set (incl. NaN / +/-Inf for the double sources) and asserts
  // the null/truncation matches Spark; ANSI runs over an in-range column so no raise is expected.
  // ---------------------------------------------------------------------------

  test("cast bigint -> int (legacy wraps)") {
    castLegacy("cl", "int")
  }

  test("cast double -> int / bigint (legacy truncates toward zero, NaN 0, Inf saturates)") {
    castLegacy("cdx", "int")
    castLegacy("cdx", "bigint")
  }

  test("cast double -> int / bigint under ANSI over an in-range column (no raise)") {
    castOkAnsi("cdsmall", "int")
    castOkAnsi("cdsmall", "bigint")
  }

  // ---------------------------------------------------------------------------
  // Boolean casts.
  // ---------------------------------------------------------------------------

  test("cast int / bigint / double -> boolean (v != 0, NaN true)") {
    castOk("ci", "boolean")
    castOk("cl", "boolean")
    castOk("cdx", "boolean")
  }

  test("cast boolean -> int / bigint / double (1 / 0)") {
    castOk("cb", "int")
    castOk("cb", "bigint")
    castOk("cb", "double")
  }

  test("cast string -> boolean: Spark's spellings, else null (legacy)") {
    castLegacy("cs_bool", "boolean")
  }

  // ---------------------------------------------------------------------------
  // To string: Java's toString, which is what Spark's cast calls.
  // ---------------------------------------------------------------------------

  test("cast int / bigint / double / boolean -> string") {
    castOk("ci", "string")
    castOk("cl", "string")
    castOk("cdx", "string") // NaN / Infinity / -Infinity spellings
    castOk("cb", "string")
  }

  test("cast date / timestamp -> string (UTC session zone)") {
    castOk("cdate", "string")
    castOk("cts", "string")
  }

  // ---------------------------------------------------------------------------
  // String -> number / date / timestamp: Spark's own parsers per row (legacy null path).
  // ---------------------------------------------------------------------------

  test("cast string -> int / bigint / double (legacy: invalid -> null)") {
    castLegacy("cs_num", "int")
    castLegacy("cs_num", "bigint")
    castLegacy("cs_num", "double")
  }

  test("cast string -> date / timestamp (legacy: invalid -> null, UTC session zone)") {
    castLegacy("cs_date", "date")
    castLegacy("cs_date", "timestamp")
  }

  // ---------------------------------------------------------------------------
  // Date <-> timestamp under a fixed-offset zone.
  // ---------------------------------------------------------------------------

  test("cast timestamp -> date and date -> timestamp (UTC session zone)") {
    castOk("cts", "date")
    castOk("cdate", "timestamp")
  }

  // ---------------------------------------------------------------------------
  // Decimal(<=18) casts: rescale-and-check; legacy nulls an out-of-range value.
  // ---------------------------------------------------------------------------

  test("cast int / bigint / double -> decimal (legacy: out-of-range nulls, matches Spark)") {
    // decimal(18,2) holds every edge int; bigint edges (19-digit Long.MIN/MAX) overflow
    // decimal(18,0) and double 1e18 overflows decimal(10,2) -> null in legacy, as our engine does.
    castLegacy("ci", "decimal(18,2)")
    castLegacy("cl", "decimal(18,0)")
    castLegacy("cd", "decimal(10,2)")
  }

  test("cast decimal -> decimal / double / bigint / int") {
    castOk("cdec", "decimal(12,4)")
    castOk("cdec", "double")
    castOk("cdec", "bigint")
    castOk("cdec", "int")
  }

  // ---------------------------------------------------------------------------
  // Identity cast: compiled as the child (pinned so the shape does not regress).
  // ---------------------------------------------------------------------------

  test("cast to the operand's own type is the identity and stays on our operator") {
    castOk("ci", "int")
    castOk("cd", "double")
  }

  // ---------------------------------------------------------------------------
  // Documented-unsupported casts: Spark accepts them, our planner declines, the query stays on
  // Spark. Asserting the fallback (no VectorProjectExec + a matching reason) is the coverage --
  // it pins that these do NOT silently produce a wrong accelerated result.
  // ---------------------------------------------------------------------------

  test("cast to float has no lane and falls back") {
    // FLOAT32 is not a lane in this project; the cast target is unsupported.
    // (tinyint / smallint are deliberately NOT asserted here: #327 compiles int -> byte/short via
    // NarrowIntExpr onto the int lane, so they are not a clean cast-target fallback.)
    castFallback("cd", "float", "unsupported cast target float")
  }

  test("cast string -> decimal falls back (Spark's Decimal.fromString not reproduced yet)") {
    // Legacy mode: the parsed values that exceed decimal(10,2) null rather than raise, so the
    // fallback (this cast is not compiled) is observable without Spark's ANSI overflow error.
    withConf("spark.sql.ansi.enabled" -> "false")(
      castFallback("cs_num", "decimal(10,2)", "unsupported cast string -> decimal(10,2)")
    )
  }

  test("cast to binary falls back (no lane)") {
    castFallback("cs_num", "binary", "unsupported cast target binary")
  }
}
