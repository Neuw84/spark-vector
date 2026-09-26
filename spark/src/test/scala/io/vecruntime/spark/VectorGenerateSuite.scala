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

import org.apache.spark.sql.execution.{GenerateExec, RowToColumnarExec}
import org.apache.spark.sql.vector.{VectorFilterExec, VectorGenerateExec, VectorHashAggregateExec, VectorProjectExec}

class VectorGenerateSuite extends VectorQuerySuite {

  private val Generate = classOf[VectorGenerateExec]
  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    // Arrays of every lane type with nulls inside, empty arrays, null arrays, and one array far longer than a batch.
    val path = newTempPath("generate/arrs")
    spark
      .range(0, 3000)
      .selectExpr(
        "cast(id AS int) AS i",
        "if(id % 7 = 3, null, concat('s', id % 5)) AS s",
        "case when id = 7 then sequence(0L, 9999L) when id % 13 = 0 then array() when id % 17 = 0 then null " +
          "else array(id, id + 1, if(id % 3 = 0, null, id * 2)) end AS arr",
        "case when id % 11 = 0 then array() else array(concat('a', id % 4), if(id % 5 = 1, null, 'b'), 'c') end AS sarr",
        "case when id % 19 = 2 then null else array(cast(id AS double) / 3, cast('NaN' AS double), -0.0d) end AS darr",
        "array(date_add(date '2020-01-01', cast(id % 300 AS int)), null) AS dtarr",
        "array(cast(id AS decimal(10, 2)) / 7, cast(null AS decimal(10, 2))) AS decarr",
        "array(id % 2 = 0, null, true) AS barr",
        "if(id % 23 = 5, null, named_struct('a', cast(id AS int), 'inner', if(id % 29 = 1, null, array(id * 10, id * 11)))) AS st",
        "map(concat('k', id % 3), id) AS mp"
      )
      .repartition(2)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("arrs")
  }

  private def noRowConversion(df: org.apache.spark.sql.DataFrame): Unit =
    assert(nodesOf[RowToColumnarExec](df).isEmpty, finalPlan(df).treeString)

  test("explode and posexplode over arrays of every lane type, with nulls, empty and null arrays and a long array") {
    noRowConversion(checkVectorized("SELECT i, s, e FROM arrs LATERAL VIEW explode(arr) t AS e", Seq(Generate)))
    noRowConversion(checkVectorized(
      "SELECT i, pos, e FROM arrs LATERAL VIEW posexplode(arr) t AS pos, e",
      Seq(Generate)
    ))
    checkVectorized("SELECT i, explode(sarr) AS e FROM arrs", Seq(Generate))
    checkVectorized("SELECT i, posexplode(darr) AS (p, e) FROM arrs", Seq(Generate))
    checkVectorized("SELECT i, explode(dtarr) AS d FROM arrs", Seq(Generate))
    checkVectorized("SELECT i, explode(decarr) AS d FROM arrs", Seq(Generate))
    checkVectorized("SELECT i, explode(barr) AS b FROM arrs", Seq(Generate))
    // The long array alone expands one input row past the batch size; the count of its rows is exact.
    val long = withPlugin(enabled = true) {
      spark.sql("SELECT count(*) AS n, sum(e) AS se FROM arrs LATERAL VIEW explode(arr) t AS e WHERE i = 7").collect()
    }
    assert(long.head.getLong(0) == 10000L && long.head.getLong(1) == 49995000L, long.mkString)
  }

  test("the outer forms emit one null row for an empty or null array; positions are null there too") {
    noRowConversion(checkVectorized("SELECT i, e FROM arrs LATERAL VIEW OUTER explode(arr) t AS e", Seq(Generate)))
    checkVectorized("SELECT i, explode_outer(arr) AS e FROM arrs WHERE i % 13 = 0 OR i % 17 = 0", Seq(Generate))
    checkVectorized("SELECT i, posexplode_outer(sarr) AS (p, e) FROM arrs", Seq(Generate))
    checkVectorized("SELECT i, posexplode_outer(darr) AS (p, e) FROM arrs WHERE i % 19 = 2", Seq(Generate))
  }

  test(
    "arrays inside a struct, columns without a lane beside the exploded one, selections below and aggregates above"
  ) {
    // A struct field's array: a null struct or a null inner array behaves as a null array.
    checkVectorized("SELECT i, e FROM arrs LATERAL VIEW OUTER explode(st.inner) t AS e", Seq(Generate))
    checkVectorized("SELECT st.a AS a, e FROM arrs LATERAL VIEW explode(st.inner) t AS e", Seq(Generate, Project))
    // The struct, the map and the source array ride beside the elements, gathered through the same repeat index.
    noRowConversion(checkVectorized(
      "SELECT i, st, mp, arr, e FROM arrs LATERAL VIEW explode(arr) t AS e WHERE i <> 7",
      Seq(Generate)
    ))
    // A filter below hands the generate a selection; a sparse one is compacted first.
    noRowConversion(checkVectorized(
      "SELECT i, e FROM arrs LATERAL VIEW explode(arr) t AS e WHERE i % 3 = 0 AND s IS NOT NULL",
      Seq(Filter, Generate)
    ))
    checkVectorized(
      "SELECT i, pos, e FROM arrs LATERAL VIEW posexplode(sarr) t AS pos, e WHERE i % 250 = 1",
      Seq(Filter, Generate)
    )
    // Computed over the elements, filtered on them, aggregated over them.
    checkVectorized(
      "SELECT i, e * 2 AS e2, upper(f) AS uf FROM arrs LATERAL VIEW explode(arr) t AS e LATERAL VIEW explode(sarr) u AS f WHERE e % 2 = 0",
      Seq(Generate, Filter, Project)
    )
    checkVectorized(
      "SELECT e % 5 AS k, count(*) AS n, sum(i) AS si FROM arrs LATERAL VIEW explode(arr) t AS e GROUP BY e % 5",
      Seq(Generate, Agg)
    )
  }

  test(
    "other generators, maps, computed arrays and nested element types fall back with a reason; the operator can be disabled"
  ) {
    checkFallback(
      "SELECT i, k, v FROM arrs LATERAL VIEW explode(mp) t AS k, v",
      Seq(Generate),
      "over a map not supported"
    )
    checkFallback(
      "SELECT i, e FROM arrs LATERAL VIEW explode(array(i, i + 1)) t AS e",
      Seq(Generate),
      "nested access over"
    )
    checkFallback("SELECT i, e FROM arrs LATERAL VIEW explode(array(st)) t AS e", Seq(Generate), "not supported")
    checkFallback("SELECT inline(array(st)) FROM arrs", Seq(Generate), "generator inline not supported")
    checkFallback("SELECT i, stack(2, i, i + 1) FROM arrs", Seq(Generate), "generator stack not supported")
    withConf("spark.vector.exec.generate.enabled" -> "false") {
      val df = withPlugin(enabled = true) {
        val d = spark.sql("SELECT i, e FROM arrs LATERAL VIEW explode(arr) t AS e"); d.collect(); d
      }
      assert(nodesOf[GenerateExec](df).nonEmpty && nodesOf[VectorGenerateExec](df).isEmpty, finalPlan(df).treeString)
    }
  }
}
