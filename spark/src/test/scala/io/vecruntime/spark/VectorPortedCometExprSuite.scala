/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.vector.VectorProjectExec

/**
 * SQL correctness coverage ported from DataFusion Comet's expression suites
 * (`CometBitwiseExpressionSuite`, `CometMathExpressionSuite`, `CometExpressionSuite`), adapted to
 * spark-vector's validation model: every query is run twice on the same session with
 * `spark.vector.enabled` toggled and the rows compared (Comet's `checkSparkAnswerAndOperator`),
 * while asserting the projection ran on our operator (`VectorProjectExec`). This is the second
 * coverage lever of the SQL-coverage survey: the golden-file suite already runs the whole
 * enumerable universe of Spark's `sql-tests/inputs`, so the gain is data-driven expression cases
 * checked against Spark itself rather than more golden files.
 *
 * The cases are limited to expressions the kernels implement and whose Spark semantics are pinned
 * in `docs/expressions.md`: the bitwise family (`& | ^ ~`, `shiftleft`/`shiftright`/
 * `shiftrightunsigned`, `bit_count`) over the INT32 and INT64 lanes, and the math family
 * (`abs`/`signum`, the transcendental unary and binary functions, the log family, `%`/`pmod`,
 * `greatest`/`least`, `rint`) over the FLOAT64 lane. Doubles are compared with the base suite's
 * tolerance; the transcendental kernels are bit-identical to Spark, so the log/trig cases pass at
 * tolerance 0 too, but the shared comparison already covers that.
 *
 * These run in legacy (non-ANSI) arithmetic so the overflow / divide-by-zero rows null rather than
 * raise -- the correctness of the ANSI raises themselves is a separate suite's concern (the golden
 * files under `nonansi/` cover them). A case that DID uncover an engine bug would be marked
 * `ignore` here with a one-line reason and a minimal repro in the PR, per the task's rule not to
 * fix engine code in a coverage change; none did on this slice.
 */
class VectorPortedCometExprSuite extends VectorQuerySuite {

  private val Project = classOf[VectorProjectExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("ported/t"))
    // ANSI off: these cases intend for overflow / zero-divisor rows to null, matching Spark's
    // non-ANSI evaluation, so the answer comparison is about the arithmetic, not the raises.
    spark.conf.set("spark.sql.ansi.enabled", "false")
  }

  /** Runs each SELECT list over `t` and asserts identical rows plus a VectorProjectExec in the plan. */
  private def checkProjections(exprs: String*): Unit =
    exprs.foreach(e => checkVectorized(s"SELECT i, $e FROM t", Seq(Project)))

  // ---------------------------------------------------------------------------
  // Bitwise (CometBitwiseExpressionSuite): & | ^ ~ over INT32 (i) and INT64 (l).
  // ---------------------------------------------------------------------------

  test("bitwise and/or/xor: column-column, column-literal, both lanes") {
    checkProjections(
      "i & 1234 AS a, i | 1234 AS o, i ^ 1234 AS x",
      "i & (i + 7) AS a, i | (i + 7) AS o, i ^ (i + 7) AS x",
      // l is nullable BIGINT with nulls where id % 7 = 3; the null must pass through as Spark's.
      "l & 255 AS a, l | 255 AS o, l ^ 255 AS x",
      "l & (i * 3) AS a, l | (i * 3) AS o, l ^ (i * 3) AS x",
      "1234 & i AS a, 1234 | i AS o, 1234 ^ i AS x" // literal on the left
    )
  }

  test("bitwise not (~) over int and long, including negatives") {
    checkProjections(
      "~i AS n",
      "~(i - 10000) AS n", // spans negative values
      "~l AS n",
      "~(l - 500) AS n"
    )
  }

  test("shiftleft/shiftright/shiftrightunsigned: literal and column amounts") {
    checkProjections(
      "shiftleft(i, 2) AS sl, shiftright(i, 2) AS sr, shiftrightunsigned(i, 2) AS sru",
      "shiftleft(i, i % 5) AS sl, shiftright(i, i % 5) AS sr, shiftrightunsigned(i, i % 5) AS sru",
      // >>> on INT32 is computed in 32 bits; negatives (i - 10000) exercise the sign bit.
      "shiftrightunsigned(i - 10000, 3) AS sru",
      // INT64 value with an INT32 amount (Comet's "different left/right types" case).
      "shiftleft(l, 2) AS sl, shiftright(l, 2) AS sr, shiftrightunsigned(l, 2) AS sru",
      "shiftleft(l, i % 4) AS sl, shiftright(l, i % 4) AS sr",
      // The SQL operators <<, >>, >>>.
      "i << 3 AS sl, i >> 3 AS sr, (i - 10000) >>> 4 AS sru"
    )
  }

  test("shift amount is masked by the lane width, as Spark does") {
    checkProjections(
      "shiftleft(i, 33) AS sl", // == shiftleft(i, 1) on an int
      "shiftleft(l, 65) AS sl", // == shiftleft(l, 1) on a long
      "shiftright(i, 32) AS sr" // == shiftright(i, 0)
    )
  }

  test("bit_count: widened to a long, over int and long, negatives") {
    checkProjections(
      "bit_count(i) AS bc",
      "bit_count(i - 10000) AS bc", // negative ints: 32 low bits set contribute
      "bit_count(l) AS bc",
      "bit_count(CAST(-1 AS INT)) AS bc" // Spark's documented 64, not 32
    )
  }

  // ---------------------------------------------------------------------------
  // Math (CometMathExpressionSuite / CometExpressionSuite): FLOAT64 lane (d, d2).
  // d carries NaN, +/-Infinity and nulls; d2 is dense in [0, 3).
  // ---------------------------------------------------------------------------

  test("abs and signum over int, long and double (incl. NaN, +/-Inf, -0.0)") {
    checkProjections(
      "abs(i - 10000) AS a",
      "abs(l - 90000) AS a",
      "abs(d) AS a", // abs(NaN) = NaN, abs(-Inf) = +Inf
      "signum(d) AS s", // signum(NaN) = NaN, signum(-0.0) = -0.0
      "signum(d - d2) AS s",
      "abs(d2 - 1.5) AS a"
    )
  }

  test("unary transcendental functions are bit-identical to Spark") {
    // Argument in [0, 3): domain-safe for the whole set except the inverse-hyperbolic edges,
    // which Math handles (NaN) and the row comparison then matches.
    Seq(
      "sqrt",
      "cbrt",
      "exp",
      "expm1",
      "sin",
      "cos",
      "tan",
      "asin",
      "acos",
      "atan",
      "sinh",
      "cosh",
      "tanh",
      "asinh",
      "acosh",
      "atanh",
      "cot",
      "degrees",
      "radians"
    ).foreach { fn =>
      checkVectorized(s"SELECT i, $fn(d2) AS v FROM t", Seq(Project))
    }
    // Over d, which includes NaN / +/-Inf / null: the function must propagate them like Spark.
    checkProjections("sqrt(abs(d)) AS v", "exp(d) AS v", "sin(d) AS v", "atan(d) AS v")
  }

  test("log family is null (not NaN) at or below its asymptote") {
    checkProjections(
      "ln(d2) AS v", // d2 has zeros (id % 13 = 0) -> null, not -Inf/NaN
      "log10(d2) AS v",
      "log2(d2) AS v",
      "log1p(d2 - 1) AS v", // asymptote at -1: d2 - 1 reaches -1 where d2 = 0
      "log(2, d2) AS v", // two-arg log(base, x): null for a non-positive x
      "ln(abs(d) + 1) AS v"
    )
  }

  test("binary math functions: pow, atan2, hypot, log(base, x)") {
    checkProjections(
      "pow(d2, 2) AS v",
      "power(d2, d2) AS v",
      "atan2(d2, d2 + 1) AS v",
      "atan2(0.0, -0.0) AS v", // Spark's signed-zero folding
      "hypot(d2, d2 + 1) AS v",
      "pow(d, 0) AS v" // pow(NaN, 0) = 1, pow(Inf, 0) = 1 in Spark/StrictMath
    )
  }

  test("modulo, pmod, greatest, least and rint") {
    checkProjections(
      "i % 7 AS m, pmod(i - 10000, 7) AS pm", // pmod keeps a non-negative result for negatives
      "l % 13 AS m, pmod(l, 13) AS pm",
      "d2 % 0.7 AS m", // double remainder
      "greatest(i, 5000, i % 100) AS g, least(i, 5000, i % 100) AS le",
      "greatest(d, d2, 1.0) AS g, least(d, d2, 1.0) AS le", // NaN is greatest, nulls ignored
      "rint(d2 * 10) AS r"
    )
  }

  test("a zero divisor nulls the row in non-ANSI mode (matches Spark)") {
    // i % 100 = 0 for 200 rows; % and pmod by it must null, not raise, with ANSI off.
    checkProjections(
      "i % (i % 100) AS m",
      "pmod(i, i % 100) AS pm"
    )
  }

  test("expressions combine: a CASE over bitwise and math results stays on our operator") {
    checkVectorized(
      "SELECT i, CASE WHEN (i & 1) = 0 THEN abs(d2 - 1.0) ELSE sqrt(d2) END AS v, " +
        "bit_count(i) + CAST(round(d2) AS INT) AS w FROM t",
      Seq(Project)
    )
  }
}
