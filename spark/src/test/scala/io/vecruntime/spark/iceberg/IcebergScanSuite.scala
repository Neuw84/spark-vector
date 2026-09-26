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
package io.vecruntime.spark.iceberg

import io.vecruntime.spark.{VectorConf, VectorPlugin}
import io.vecruntime.spark.test.IcebergTest
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.vecruntime.{PlanUtils, VectorPlan, VectorPrefetchScanExec}
import org.scalatest.Tag

/**
 * Iceberg merge-on-read tables read through Iceberg's own JVM vectorized reader (Spark's
 * `BatchScanExec`), the path taken without Comet or when Comet falls back. On tables with deletes
 * the reader hands out row-id-mapped batches; the Iceberg adapter unwraps them into a selected
 * batch over the physical rows and reads the Arrow buffers in place. Needs only the Iceberg
 * runtime (`mvn -Piceberg`).
 */
class IcebergScanSuite extends IcebergMorSuiteBase {

  override protected def suiteTags: Seq[Tag] = Seq(IcebergTest)

  override protected def expectedScanClass: String = "BatchScanExec"

  override protected def extraSparkConf: Map[String, String] =
    icebergConf ++ Map(
      "spark.plugins" -> classOf[VectorPlugin].getName,
      "spark.sql.adaptive.enabled" -> "true",
      // Several batches per file so a delete-carrying batch is not the only shape seen.
      s"spark.sql.catalog.${IcebergTables.Catalog}.read.parquet.vectorization.batch-size" -> "1024"
    )

  icebergTest("row-id-mapped batches are normalized into selections and read in place") {
    useTable("t_pos")
    val batchesBefore = IcebergVectorAdapter.normalizedBatches()
    val columnsBefore = IcebergVectorAdapter.adaptedColumns()
    val dictBefore = IcebergVectorAdapter.adaptedDictionaryColumns()
    val df = checkVectorized("SELECT i, l, d, dt, b, s FROM t WHERE i > 100 AND d IS NOT NULL", Seq(Filter))
    assertScanUnder(df, Filter)
    assert(IcebergVectorAdapter.isRegistered, "Iceberg adapter should be registered when Iceberg is on the classpath")
    assert(IcebergVectorAdapter.normalizedBatches() > batchesBefore, "expected merge-on-read batches to be normalized")
    assert(IcebergVectorAdapter.adaptedColumns() > columnsBefore, "expected Iceberg vectors to be adapted zero-copy")
    // `s` has 50 distinct values, so Parquet dictionary-encodes it and Iceberg keeps the indices.
    assert(IcebergVectorAdapter.adaptedDictionaryColumns() > dictBefore, "expected a dictionary-encoded string column")
  }

  icebergTest("a table without deletes is read zero-copy with no normalization") {
    spark.sql(
      s"CREATE OR REPLACE TABLE ${IcebergTables.Db}.t_plain USING iceberg AS SELECT * FROM ${IcebergTables.Db}.t_pos"
    )
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.t_plain")
    val batchesBefore = IcebergVectorAdapter.normalizedBatches()
    val columnsBefore = IcebergVectorAdapter.adaptedColumns()
    checkVectorized("SELECT s, count(*), sum(d2), min(i), max(l) FROM t WHERE i > 5 GROUP BY s", Seq(Filter, Agg))
    assert(IcebergVectorAdapter.normalizedBatches() === batchesBefore)
    assert(IcebergVectorAdapter.adaptedColumns() > columnsBefore)
  }

  private val Sort = classOf[org.apache.spark.sql.vecruntime.VectorSortExec]

  icebergTest("wide decimals from Iceberg's reader become DECIMAL128 lanes: filtered, projected and sorted") {
    // Iceberg keeps decimal(p > 18) as a 16-byte FixedSizeBinaryVector of big-endian bytes; the adapter
    // converts each valid row into two limbs (#257), on a plain table and under row-id-mapped deletes.
    spark.sql(
      s"""CREATE OR REPLACE TABLE ${IcebergTables.Db}.t_wide USING iceberg AS
         |SELECT i, l, s,
         |  CASE WHEN i % 101 = 1 THEN CAST('9999999999999999999999999999.9999999999' AS DECIMAL(38,10))
         |       WHEN i % 101 = 2 THEN CAST('-9999999999999999999999999999.9999999999' AS DECIMAL(38,10))
         |       WHEN i % 13 = 0 THEN NULL
         |       ELSE CAST(CAST(i - 10000 AS DECIMAL(38,10)) * CAST('1234567890123.0000000001' AS DECIMAL(38,10)) AS DECIMAL(38,10)) END AS w38,
         |  CASE WHEN i % 17 = 0 THEN NULL ELSE CAST(CAST((i % 40) - 20 AS DECIMAL(27,2)) * CAST('1000000000000000.25' AS DECIMAL(27,2)) AS DECIMAL(27,2)) END AS w27
         |FROM ${IcebergTables.Db}.t_pos""".stripMargin
    )
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.t_wide")
    val columnsBefore = IcebergVectorAdapter.adaptedColumns()
    val copiedBefore = io.vecruntime.spark.adapter.ColumnVectorAdapters.copiedColumns()
    checkVectorized("SELECT i, w38, w27 FROM t WHERE i % 3 = 0 AND l IS NOT NULL", Seq(Filter))
    checkVectorized("SELECT w38, i + 1 AS j, w27 FROM t WHERE i % 5 = 0", Seq(Filter, Project))
    checkVectorized("SELECT w38, i FROM t SORT BY w38 DESC NULLS LAST", Seq(Sort))
    assert(
      IcebergVectorAdapter.adaptedColumns() > columnsBefore,
      "expected the wide columns to be adapted by the Iceberg adapter"
    )
    assert(
      io.vecruntime.spark.adapter.ColumnVectorAdapters.copiedColumns() === copiedBefore,
      "no column should have taken the generic copy path"
    )
    // Under positional deletes the batch is row-id mapped: the wide lane is built over the physical rows.
    spark.sql(s"DELETE FROM ${IcebergTables.Db}.t_wide WHERE i % 7 = 0")
    checkVectorized("SELECT i, w38 FROM t WHERE i % 2 = 0", Seq(Filter))
    checkVectorized("SELECT w27, w38 FROM t SORT BY w27", Seq(Sort))
  }

  icebergTest("decimals of up to 9 digits (an IntVector in Iceberg's reader) are widened into INT64 lanes") {
    // Iceberg reads a DECIMAL(p <= 9) as an IntVector of 4-byte unscaled values; the lane for every
    // decimal up to 18 digits is INT64. Wrapping the 4-byte buffer as 8-byte lanes read past its
    // end (TPC-DS store_sales' DECIMAL(7,2) prices crashed the compact and grouped-sum kernels).
    spark.sql(
      s"""CREATE OR REPLACE TABLE ${IcebergTables.Db}.t_small_dec USING iceberg AS
         |SELECT i, s,
         |  CASE WHEN i % 11 = 0 THEN NULL ELSE CAST((i % 20000) - 10000 AS DECIMAL(7,2)) END AS d7,
         |  CAST(i AS DECIMAL(15,2)) AS d15
         |FROM ${IcebergTables.Db}.t_pos""".stripMargin
    )
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.t_small_dec")
    val columnsBefore = IcebergVectorAdapter.adaptedColumns()
    val widenedBefore = IcebergVectorAdapter.widenedIntColumns()
    checkVectorized("SELECT i, d7, d15 FROM t WHERE i % 3 = 0", Seq(Filter))
    checkVectorized("SELECT s, sum(d7), sum(d15), count(d7) FROM t GROUP BY s", Seq(Agg))
    assert(IcebergVectorAdapter.adaptedColumns() > columnsBefore, "expected the decimal columns to be adapted")
    assert(IcebergVectorAdapter.widenedIntColumns() > widenedBefore, "expected the DECIMAL(7,2) lane to be widened")
    // Under positional deletes the widened lane is built over the physical rows.
    spark.sql(s"DELETE FROM ${IcebergTables.Db}.t_small_dec WHERE i % 7 = 0")
    checkVectorized("SELECT i, d7 FROM t WHERE i % 2 = 0", Seq(Filter))
    checkVectorized("SELECT s, sum(d7), max(d7) FROM t GROUP BY s", Seq(Agg))
  }

  icebergTest("dictionary-encoded decimals of up to 18 digits are decoded through one table per dictionary (#20)") {
    // Few distinct values, so Parquet dictionary-encodes both columns and Iceberg hands out a
    // DictionaryDecimalInt/LongAccessor over the ids; the copy path read those through getDecimal,
    // a Spark Decimal per row.
    spark.sql(
      s"""CREATE OR REPLACE TABLE ${IcebergTables.Db}.t_dict_dec USING iceberg AS
         |SELECT i, s,
         |  CASE WHEN i % 13 = 0 THEN NULL ELSE CAST(((i % 37) - 18) / 4.0 AS DECIMAL(7,2)) END AS d7,
         |  CAST((i % 23) * CAST(1000000007 AS BIGINT) / 8.0 AS DECIMAL(15,2)) AS d15
         |FROM ${IcebergTables.Db}.t_pos""".stripMargin
    )
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.t_dict_dec")
    val before = IcebergVectorAdapter.adaptedDictionaryDecimalColumns()
    checkVectorized("SELECT i, d7, d15 FROM t WHERE i % 3 = 0", Seq(Filter))
    checkVectorized("SELECT s, sum(d7), sum(d15), count(d7), min(d15) FROM t GROUP BY s", Seq(Agg))
    assert(
      IcebergVectorAdapter.adaptedDictionaryDecimalColumns() > before,
      "expected the dictionary-encoded decimal columns to go through the lookup table"
    )
    // Under positional deletes the lane is built over the physical rows.
    spark.sql(s"DELETE FROM ${IcebergTables.Db}.t_dict_dec WHERE i % 7 = 0")
    checkVectorized("SELECT i, d7, d15 FROM t WHERE i % 2 = 0", Seq(Filter))
    checkVectorized("SELECT s, sum(d7), max(d15) FROM t GROUP BY s", Seq(Agg))
  }

  icebergTest("the prefetching converter wraps Iceberg's scan and converts its batches on the helper (#403)") {
    val Prefetch = classOf[VectorPrefetchScanExec]
    def prefetched(sql: String, ops: Class[_ <: SparkPlan]*): Unit = {
      val off = withConf(VectorConf.ScanPrefetch -> "0")(spark.sql(sql).collect())
      withConf(VectorConf.ScanPrefetch -> "2") {
        val df = checkVectorized(sql, ops :+ Prefetch)
        assertRowsEqual(off, df.collect(), 1e-9, s"prefetch on vs off: $sql")
        val nodes = nodesOf[VectorPrefetchScanExec](df)
        assert(nodes.size === 1, s"expected one prefetch node:\n${finalPlan(df).treeString}")
        assert(nodes.head.child.getClass.getSimpleName === expectedScanClass, nodes.head.child.getClass.getName)
        val parents =
          PlanUtils.allNodes(finalPlan(df)).filter(_.children.exists(_.isInstanceOf[VectorPrefetchScanExec]))
        assert(parents.size === 1 && parents.head.isInstanceOf[VectorPlan])
      }
    }
    // Row-id-mapped batches (positional deletes): normalized on the helper, the live rows compacted.
    useTable("t_pos")
    val normalizedBefore = IcebergVectorAdapter.normalizedBatches()
    val adaptedBefore = IcebergVectorAdapter.adaptedColumns()
    prefetched("SELECT i, l, d, dt, b, s FROM t WHERE i > 100 AND d IS NOT NULL", Filter)
    prefetched("SELECT s, count(*), sum(d2), min(i), max(l) FROM t WHERE i > 5 GROUP BY s", Filter, Agg)
    assert(IcebergVectorAdapter.normalizedBatches() > normalizedBefore, "expected the helper to normalize the batches")
    assert(
      IcebergVectorAdapter.adaptedColumns() > adaptedBefore,
      "expected the helper to adapt Iceberg's vectors in place"
    )
    // Deletion vectors (v3) and a plain table (no normalization) through the same node.
    useTable("t_dv")
    prefetched("SELECT count(*), sum(d2), max(d), min(dt) FROM t WHERE i > 10", Filter, Agg)
    spark.sql(
      s"CREATE OR REPLACE TABLE ${IcebergTables.Db}.t_plain2 USING iceberg AS SELECT * FROM ${IcebergTables.Db}.t_pos"
    )
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.t_plain2")
    prefetched("SELECT i, s, d2 FROM t WHERE b OR i < 15000", Filter)
  }
}
