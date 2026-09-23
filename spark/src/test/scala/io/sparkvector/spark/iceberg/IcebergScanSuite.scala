package io.sparkvector.spark.iceberg

import io.sparkvector.spark.VectorPlugin
import io.sparkvector.spark.test.IcebergTest
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

  private val Sort = classOf[org.apache.spark.sql.vector.VectorSortExec]

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
    val copiedBefore = io.sparkvector.spark.adapter.ColumnVectorAdapters.copiedColumns()
    checkVectorized("SELECT i, w38, w27 FROM t WHERE i % 3 = 0 AND l IS NOT NULL", Seq(Filter))
    checkVectorized("SELECT w38, i + 1 AS j, w27 FROM t WHERE i % 5 = 0", Seq(Filter, Project))
    checkVectorized("SELECT w38, i FROM t SORT BY w38 DESC NULLS LAST", Seq(Sort))
    assert(
      IcebergVectorAdapter.adaptedColumns() > columnsBefore,
      "expected the wide columns to be adapted by the Iceberg adapter"
    )
    assert(
      io.sparkvector.spark.adapter.ColumnVectorAdapters.copiedColumns() === copiedBefore,
      "no column should have taken the generic copy path"
    )
    // Under positional deletes the batch is row-id mapped: the wide lane is built over the physical rows.
    spark.sql(s"DELETE FROM ${IcebergTables.Db}.t_wide WHERE i % 7 = 0")
    checkVectorized("SELECT i, w38 FROM t WHERE i % 2 = 0", Seq(Filter))
    checkVectorized("SELECT w27, w38 FROM t SORT BY w27", Seq(Sort))
  }
}
