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
import org.apache.spark.sql.execution.FilterExec
import org.apache.spark.sql.vector.VectorFilterExec

class VectorFilterSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("filter/t"))
    TestTables.createLineitem(spark, newTempPath("filter/lineitem"))
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
    assert(
      notLike.count() === 20000 - 2000 - 5200
    ) // nulls drop; s3, s13, s23, s43 and s30..s39 minus the null s30 = 13 values
  }

  test("multi-wildcard LIKE as a multi-token matcher (#264): the Q13/Q16 shapes, dictionary columns, NOT LIKE") {
    // A comment-like column with the TPC-H phrases in several arrangements, and s ('s0'..'s49') is dictionary encoded.
    spark.range(0, 20000).selectExpr(
      "cast(id as int) as i",
      "case when id % 11 = 0 then null " +
        "     when id % 7 = 0 then concat('the special requests of ', id) " +
        "     when id % 7 = 1 then concat('requests special ', id) " +
        "     when id % 7 = 2 then concat('Customer Complaints ', id, ' Complaints') " +
        "     when id % 7 = 3 then concat('specialrequests', id) " +
        "     when id % 7 = 4 then concat('special ', id, ' request') " +
        "     when id % 7 = 5 then concat('日本語 special 本 requests ', id) " +
        "     else concat('nothing here ', id) end as c"
    )
      .repartition(2).write.mode("overwrite").parquet(newTempPath("filter/comments"))
    spark.read.parquet(newTempPath("filter/comments")).createOrReplaceTempView("comments")
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE '%special%requests%'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c NOT LIKE '%special%requests%'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE '%Customer%Complaints%'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE 'the%requests%'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE '%special%request'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE 'requests%special%9'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE '%日%本%'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE '%%special%%requests%%'", Seq(Filter))
    checkVectorized("SELECT i, c FROM comments WHERE c LIKE '%s%s%s%s%'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s LIKE 's%1%'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s LIKE '%1%2%'", Seq(Filter))
    checkVectorized("SELECT i, s FROM t WHERE s NOT LIKE 's%4%'", Seq(Filter))
    val q13 = checkVectorized("SELECT i FROM comments WHERE c NOT LIKE '%special%requests%'", Seq(Filter))
    // nulls drop (id % 11 = 0); the matches are id % 7 in {0, 3, 5} -- 'requests special' and 'special ... request' do not match.
    val expected = (0 until 20000).count(id => id % 11 != 0 && !Set(0, 3, 5).contains(id % 7))
    assert(q13.count() === expected)
  }

  test("unsupported expressions fall back with a reason") {
    checkFallback("SELECT * FROM t WHERE s LIKE 's_'", Seq(Filter), "`_` wildcard")
    checkFallback("SELECT * FROM t WHERE s LIKE 's%_%2'", Seq(Filter), "`_` wildcard")
    checkFallback("SELECT * FROM t WHERE s LIKE 's%\\%%2'", Seq(Filter), "escape character")
    checkFallback("SELECT * FROM t WHERE soundex(s) = 'S000'", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE s = reverse(s)", Seq(Filter), "unsupported expression")
    checkFallback("SELECT * FROM t WHERE startswith(s, s)", Seq(Filter), "string pattern is not a literal")
    // The optimizer turns a long IN list into InSet; one holding NULL still falls back (Spark's result is null for non-members).
    checkFallback("SELECT * FROM t WHERE i IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, NULL)", Seq(Filter), "NULL in IN set")
    checkFallback("SELECT * FROM t WHERE s IN ('s1', NULL)", Seq(Filter), "NULL in IN list")
  }

  test("null-safe equality, isnan, boolean comparisons, BETWEEN and InSet") {
    // <=> : both-null rows match, one-null rows do not, and the result is never null (so NOT flips every row).
    checkVectorized("SELECT i FROM t WHERE l <=> CAST(i AS BIGINT) * 3", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE NOT (l <=> CAST(i AS BIGINT) * 3)", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE s <=> 's7' OR d <=> d2", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE l <=> NULL", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE b <=> (i % 2 = 0)", Seq(Filter))
    // isnan: d has NaN, infinities and nulls; null is false, not null.
    checkVectorized("SELECT i FROM t WHERE isnan(d)", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE NOT isnan(d) AND d > 100", Seq(Filter))
    // Boolean columns against literals and each other (false < true).
    checkVectorized("SELECT i FROM t WHERE b = true", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE b <> false OR (i % 2 = 0) < b", Seq(Filter))
    checkVectorized("SELECT i FROM t WHERE b >= (i % 5 = 0) AND true > b", Seq(Filter))
    // BETWEEN is two comparisons after Spark's rewrite; pin it.
    checkVectorized("SELECT i FROM t WHERE i BETWEEN 100 AND 200 AND d NOT BETWEEN 1.5 AND 2.5", Seq(Filter))
    // InSet: the optimizer rewrites lists above spark.sql.optimizer.inSetConversionThreshold (10).
    checkVectorized(
      "SELECT i FROM t WHERE i IN (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946)",
      Seq(Filter)
    )
    checkVectorized(
      "SELECT i FROM t WHERE l IN (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45)",
      Seq(Filter)
    )
    checkVectorized(
      "SELECT i FROM t WHERE d2 IN (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75) AND s IN ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12')",
      Seq(Filter)
    )
    checkVectorized("SELECT i FROM t WHERE i NOT IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)", Seq(Filter))
    val hits = checkVectorized(
      "SELECT i FROM t WHERE i IN (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946)",
      Seq(Filter)
    ).count()
    assert(hits === 20, "every Fibonacci number below 20000 is a row")
  }

  test(
    "string IN lists go through the hash set (#371): dictionary and plain columns, duplicates, misses, the empty string"
  ) {
    // s is dictionary encoded ('s0'..'s49' with nulls); substr(s, 2) is a plain computed column, as q8's substr(ca_zip, 1, 5) is.
    val zips = (0 until 400).map(z => s"'$z'").mkString(", ")
    checkVectorized(
      "SELECT i FROM t WHERE s IN ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12', 's13')",
      Seq(Filter)
    )
    checkVectorized(
      "SELECT i FROM t WHERE s NOT IN ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12', 's13')",
      Seq(Filter)
    )
    checkVectorized(
      "SELECT i FROM t WHERE substr(s, 2) IN ('1', '2', '3', '4', '17', '18', '19', '20', '21', '22', '23', '49')",
      Seq(Filter)
    )
    checkVectorized(
      "SELECT i FROM t WHERE s IN ('s1', 's1', 's1', 's2', 's2')",
      Seq(Filter)
    ) // duplicates, below the InSet threshold
    checkVectorized(
      "SELECT i FROM t WHERE s IN ('', 's', 'S1', 's1 ', ' s1', 's100', 's001', 'nowhere', 'x', 'y', 'z')",
      Seq(Filter)
    ) // near misses only
    checkVectorized(s"SELECT i FROM t WHERE CAST(i AS STRING) IN ($zips)", Seq(Filter)) // q8's shape: 400 literals
    val four = checkVectorized("SELECT i FROM t WHERE s IN ('s1', 's17', 's30', 's49')", Seq(Filter)).count()
    assert(four === 1200) // 400 each for s1, s17 and s49; s30 never occurs
    val z = checkVectorized(s"SELECT i FROM t WHERE CAST(i AS STRING) IN ($zips)", Seq(Filter)).count()
    assert(z === 400)
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
    val plan =
      withPlugin(enabled = true)(spark.sql("SELECT * FROM t WHERE i > 5 AND l IS NOT NULL")).queryExecution.executedPlan
    val text = plan.treeString
    assert(text.contains("VectorFilter"), text)
  }

  test(
    "scalar subqueries are literals by execution time: filters, projections, aggregates, merged struct fields, nulls"
  ) {
    val Project = classOf[org.apache.spark.sql.vector.VectorProjectExec]
    val Agg = classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]
    // In a filter, alone and combined; a string-valued and a boolean-valued subquery.
    checkVectorized("SELECT i, d2 FROM t WHERE d2 > (SELECT avg(d2) FROM t)", Seq(Filter))
    checkVectorized(
      "SELECT count(*) FROM t WHERE l > (SELECT avg(l) FROM t) OR s = (SELECT max(s) FROM t)",
      Seq(Filter, Agg)
    )
    checkVectorized("SELECT i FROM t WHERE (SELECT bool_or(b) FROM t WHERE i < 3) AND i < 20", Seq(Filter))
    // In a projection, in arithmetic and in a CASE over the result.
    checkVectorized(
      "SELECT i, (SELECT max(l) FROM t) AS m, i + (SELECT min(i) FROM t WHERE b) AS j, CASE WHEN (SELECT count(*) FROM t) > 10 THEN i ELSE -i END AS c FROM t WHERE i < 100",
      Seq(Project, Filter)
    )
    // Two subqueries over the same table are merged by Spark into one struct-valued subquery read through GetStructField.
    val merged = checkVectorized(
      "SELECT i FROM t WHERE d2 > (SELECT avg(d2) FROM t) AND i > (SELECT count(*) FROM t) / 3",
      Seq(Filter)
    )
    assert(finalPlan(merged).toString.contains("Subquery subquery"), finalPlan(merged).treeString)
    // Under an aggregate and as an aggregate's input.
    checkVectorized(
      "SELECT i % 3 AS g, sum(d2 - (SELECT avg(d2) FROM t)) AS s, count(*) FROM t WHERE i < (SELECT max(i) FROM t) / 2 GROUP BY i % 3",
      Seq(Filter, Agg)
    )
    // A null result: no row passes the filter; the projected value is null.
    checkVectorized("SELECT i FROM t WHERE i < (SELECT max(i) FROM t WHERE i < 0)", Seq(Filter))
    checkVectorized("SELECT i, (SELECT max(d2) FROM t WHERE i < 0) AS nn FROM t WHERE i < 5", Seq(Project, Filter))
    // A string subquery in a projection and in a LIKE-free comparison.
    checkVectorized(
      "SELECT i, (SELECT min(s) FROM t WHERE s IS NOT NULL) AS ms FROM t WHERE s > (SELECT min(s) FROM t WHERE s IS NOT NULL)",
      Seq(Project, Filter)
    )
  }

  test("runtime bloom filter: the injected probe above the scan is ours") {
    import org.apache.spark.sql.catalyst.expressions.BloomFilterMightContain
    // A shuffle join whose small side has a selective filter makes Spark inject
    // Filter(BloomFilterMightContain(subquery, xxhash64(key))) above the large side's scan.
    withConf(
      "spark.sql.optimizer.runtime.bloomFilter.enabled" -> "true",
      "spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold" -> "0",
      "spark.sql.autoBroadcastJoinThreshold" -> "-1"
    ) {
      val sql =
        "SELECT count(*), sum(l_quantity) FROM lineitem JOIN t ON lineitem.l_partkey = t.i WHERE t.b AND t.i < 500"
      val df = checkVectorized(sql, Seq(Filter))
      val probes = nodesOf[VectorFilterExec](df).filter(_.condition.exists(_.isInstanceOf[BloomFilterMightContain]))
      assert(probes.nonEmpty, "Spark injected no bloom filter probe, or it is not ours:\n" + finalPlan(df).treeString)
      assert(
        !nodesOf[FilterExec](df).exists(_.condition.exists(_.isInstanceOf[BloomFilterMightContain])),
        finalPlan(df).treeString
      )
      // A probe over a string key (a self-join whose small side is the filtered one).
      val df2 =
        checkVectorized("SELECT count(*), sum(a.i) FROM t a JOIN t b ON a.s = b.s WHERE b.i < 40 AND b.b", Seq(Filter))
      assert(
        nodesOf[VectorFilterExec](df2).exists(_.condition.exists(_.isInstanceOf[BloomFilterMightContain])),
        finalPlan(df2).treeString
      )
    }
  }
}
