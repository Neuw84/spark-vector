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
      s"spark.sql.catalog.${IcebergTables.Catalog}.read.parquet.vectorization.batch-size" -> "1024")

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
    spark.sql(s"CREATE OR REPLACE TABLE ${IcebergTables.Db}.t_plain USING iceberg AS SELECT * FROM ${IcebergTables.Db}.t_pos")
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.t_plain")
    val batchesBefore = IcebergVectorAdapter.normalizedBatches()
    val columnsBefore = IcebergVectorAdapter.adaptedColumns()
    checkVectorized("SELECT s, count(*), sum(d2), min(i), max(l) FROM t WHERE i > 5 GROUP BY s", Seq(Filter, Agg))
    assert(IcebergVectorAdapter.normalizedBatches() === batchesBefore)
    assert(IcebergVectorAdapter.adaptedColumns() > columnsBefore)
  }
}
