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

import io.vecruntime.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.vector.VectorHashAggregateExec

/**
 * SQL correctness coverage ported from DataFusion Comet's `CometAggregateSuite`, adapted to
 * spark-vector's plugin-on/off comparison model: every aggregate query is run twice on one session
 * with `spark.vector.enabled` toggled, the rows compared, and a `VectorHashAggregateExec` asserted
 * in the accelerated plan.
 *
 * This is the aggregate slice of the SQL-coverage survey. It complements the hand-written
 * `VectorAggregateSuite` (which already covers ungrouped count/sum/min/max/avg, NaN/Inf ordering,
 * all-null and empty inputs, and the string-key grouping paths) by adding Comet's data-driven
 * matrix that is not yet pinned here: `count(DISTINCT ...)` single and multi-column, grouped
 * sum/avg/min/max over one and several keys, `min`/`max` over strings and booleans, `first`/`last`,
 * `bit_and`/`bit_or`/`bit_xor`, and decimals within our 18-digit lanes -- each grouped AND ungrouped,
 * with nulls, empty inputs and negative-zero / NaN edges, exactly the shapes Comet's suite checks
 * ("multiple column distinct count", "avg decimal", "min/max floating point with negative zero",
 * "simple SUM/COUNT/MIN/MAX/AVG with non-distinct group keys", "group-by on variable length types").
 *
 * Only aggregates and input types `docs/expressions.md` pins as supported are used, so every case
 * is expected to stay on `VectorHashAggregateExec`. A case that uncovered an engine bug would be
 * marked `ignore` with a one-line reason and a minimal repro in the PR, per the task's rule not to
 * fix engine code in a coverage change; none did on this slice.
 */
class VectorPortedCometAggregateSuite extends VectorQuerySuite {

  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("pagg/t"))

    // A grouped table with a decimal MEASURE inside our INT64 lane (decimal(<=18)) and a small
    // decimal group key, nulls in the measure, several groups, some groups empty of non-nulls.
    // Written and re-read as Parquet so the scan is columnar.
    val rows = (0 until 4000).map { id =>
      val g = id % 13
      val amt: java.math.BigDecimal =
        if (id % 17 == 0) null else java.math.BigDecimal.valueOf((id % 900) - 450).movePointLeft(2)
      Row(g, amt)
    }.toList
    val schema = StructType(
      Seq(
        StructField("g", IntegerType, nullable = false),
        StructField("amt", DecimalType(15, 2), nullable = true)
      )
    )
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows, 4), schema)
    val path = newTempPath("pagg/dec")
    df.write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("decs")
  }

  private def agg(sql: String): Unit = checkVectorized(sql, Seq(Agg))

  // ---------------------------------------------------------------------------
  // COUNT DISTINCT (Comet's "multiple column distinct count"): Spark's distinct
  // rewrite (planAggregateWithOneDistinct / Expand) compiles onto our operator.
  // ---------------------------------------------------------------------------

  test("count distinct, single column, ungrouped and grouped") {
    agg("SELECT count(DISTINCT i) FROM t")
    agg("SELECT count(DISTINCT s) FROM t") // strings, with nulls
    agg("SELECT b, count(DISTINCT i) FROM t GROUP BY b")
    agg("SELECT i % 100 AS k, count(DISTINCT s) AS n FROM t GROUP BY i % 100")
  }

  test("multiple column distinct count, ungrouped and grouped") {
    agg("SELECT count(DISTINCT i, s) FROM t")
    agg("SELECT count(DISTINCT i), count(DISTINCT s), count(*) FROM t")
    agg("SELECT b, count(DISTINCT i, s), count(*) FROM t GROUP BY b")
  }

  // ---------------------------------------------------------------------------
  // Grouped SUM / COUNT / MIN / MAX / AVG over one and several keys.
  // ---------------------------------------------------------------------------

  test("grouped sum/count/min/max/avg over int, bigint and double") {
    agg("SELECT i % 50 AS k, count(*), sum(i), min(i), max(i), avg(i) FROM t GROUP BY i % 50")
    agg("SELECT b, count(l), sum(l), min(l), max(l), avg(l) FROM t GROUP BY b") // l has nulls
    agg("SELECT i % 7 AS k, sum(d2), min(d2), max(d2), avg(d2), count(d2) FROM t GROUP BY i % 7")
  }

  test("grouped by several keys, including a string key") {
    agg("SELECT b, i % 3 AS k, count(*), sum(d2), avg(i) FROM t GROUP BY b, i % 3")
    agg("SELECT s, b, count(*) AS n, sum(i) AS si FROM t GROUP BY s, b")
  }

  // ---------------------------------------------------------------------------
  // MIN / MAX over strings and booleans, and the negative-zero / NaN edges.
  // ---------------------------------------------------------------------------

  test("min/max over strings and booleans (ungrouped and grouped)") {
    agg("SELECT min(s), max(s) FROM t")
    agg("SELECT min(b), max(b), bool_and(b), bool_or(b) FROM t")
    agg("SELECT b, min(s), max(s) FROM t GROUP BY b")
  }

  test("min/max floating point with negative zero and NaN follow Spark") {
    // d holds NaN, +/-Inf and -0.0-adjacent values: grouped max/min must agree with Spark's ordering.
    agg("SELECT i % 11 AS k, min(d), max(d) FROM t GROUP BY i % 11")
  }

  // ---------------------------------------------------------------------------
  // FIRST / LAST and the bitwise aggregates.
  // ---------------------------------------------------------------------------

  test("first/last, with and without ignore nulls, grouped") {
    agg("SELECT b, first(l), last(l), first(l, true), last(l, true) FROM t GROUP BY b")
    agg("SELECT first(s, true), last(s, true) FROM t")
  }

  test("bit_and / bit_or / bit_xor over int and bigint, grouped and ungrouped") {
    agg("SELECT bit_and(i), bit_or(i), bit_xor(i) FROM t")
    agg("SELECT b, bit_and(i), bit_or(i), bit_xor(i) FROM t GROUP BY b")
    agg("SELECT i % 5 AS k, bit_and(l), bit_or(l), bit_xor(l) FROM t GROUP BY i % 5")
  }

  // ---------------------------------------------------------------------------
  // Decimals within our lanes (Comet's "avg decimal"): decimal(15,2) sum/avg,
  // grouped and ungrouped, with nulls and groups whose measure is all-null.
  // ---------------------------------------------------------------------------

  test("sum/avg/min/max over a decimal(15,2) measure, ungrouped") {
    agg("SELECT sum(amt), avg(amt), min(amt), max(amt), count(amt) FROM decs")
  }

  test("sum/avg/min/max over a decimal(15,2) measure, grouped") {
    agg("SELECT g, sum(amt), avg(amt), min(amt), max(amt), count(amt), count(*) FROM decs GROUP BY g")
  }

  test("decimal sum/avg over an all-null measure and an empty (ungrouped) input") {
    // All rows present but the measure all null: the buffer stays, sum is null, count 0.
    agg("SELECT sum(amt), avg(amt), count(amt) FROM decs WHERE amt IS NULL")
    // Ungrouped over zero rows: Spark still emits one output row (sum/avg null, count 0), so the
    // aggregate is not elided. (A GROUPED aggregate over zero rows is folded to EmptyRelation by
    // AQE and has no operator to assert -- deliberately not tested here.)
    agg("SELECT sum(amt), avg(amt), count(amt), count(*) FROM decs WHERE g < 0")
  }

  // ---------------------------------------------------------------------------
  // count_if, sum with FILTER, and grouped count over variable-length keys.
  // ---------------------------------------------------------------------------

  test("count_if and aggregate FILTER clauses") {
    agg("SELECT count_if(i % 2 = 0), count(*) FROM t")
    agg("SELECT b, sum(i) FILTER (WHERE i % 3 = 0), count(*) FILTER (WHERE d2 > 1.0) FROM t GROUP BY b")
  }

  test("group-by on variable length (string) keys with several aggregates") {
    agg("SELECT s, count(*), sum(i), avg(d2), min(l), max(l) FROM t GROUP BY s")
  }
}
