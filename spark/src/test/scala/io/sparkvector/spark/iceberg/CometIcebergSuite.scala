package io.sparkvector.spark.iceberg

import io.sparkvector.spark.comet.{CometTestConf, CometVectorAdapter}
import io.sparkvector.spark.test.{CometTest, IcebergTest}
import org.scalatest.Tag

/**
 * Iceberg merge-on-read tables read through Comet's native Iceberg scan (iceberg-rust), which
 * applies positional deletes, deletion vectors and equality deletes inside the reader. Our
 * operators therefore see plain Comet batches and adapt them zero-copy, exactly as over Comet's
 * Parquet scan. Needs both the Comet jar and the Iceberg runtime (`mvn -Pcomet,iceberg`).
 */
class CometIcebergSuite extends IcebergMorSuiteBase {

  override protected def suiteTags: Seq[Tag] = Seq(CometTest, IcebergTest)

  override protected def expectedScanClass: String = "CometIcebergNativeScanExec"

  /**
   * Comet 1.0 falls back to Iceberg's JVM reader for format version 3 (deletion vectors); newer
   * Comet reads v3 natively. The fallback batch carries Iceberg's row-id mapping, which the
   * Iceberg adapter turns into a selection (see IcebergScanSuite).
   */
  override protected def expectedScanClassFor(table: String): String =
    if (table == "t_dv") "BatchScanExec" else expectedScanClass

  override protected def extraSparkConf: Map[String, String] =
    CometTestConf.scanOnly ++ icebergConf ++ Map(
      "spark.comet.scan.icebergNative.enabled" -> "true",
      "spark.comet.scan.icebergNative.dataFileConcurrencyLimit" -> "2")

  icebergTest("Comet's vectors are adapted zero-copy") {
    useTable("t_pos")
    checkVectorized("SELECT i, s FROM t WHERE i > 100", Seq(Filter))
    assert(CometVectorAdapter.isRegistered, "Comet adapter should be registered when Comet is on the classpath")
  }
}
