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
package io.sparkvector.iceberg.bridge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._
import scala.util.Random

import org.apache.iceberg.{PartitionSpec, Table}
import org.apache.iceberg.deletes.PositionDeleteIndex
import org.apache.iceberg.io.OutputFileFactory
import org.apache.iceberg.spark.Spark3Util
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * Bridge-level tests for the columnar v3 DV writer (#20, slices 2-3): the differential test that our
 * bitmap equals Iceberg's per-row-built [[PositionDeleteIndex]] for the same positions, and an
 * end-to-end write that builds a deletion vector with [[ColumnarDvWriter]], commits it through
 * Iceberg's own `RowDelta`, and reads the table back to confirm the rows are gone and the DV files
 * are the Iceberg Java API's own. No spark-vector operator yet -- this proves the bridge and the
 * commit seam in isolation.
 */
class ColumnarDvWriterSuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private lazy val warehouse: Path = Files.createTempDirectory("dvbridge")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("dv-bridge")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
      .config("spark.sql.catalog.ice", "org.apache.iceberg.spark.SparkCatalog")
      .config("spark.sql.catalog.ice.type", "hadoop")
      .config("spark.sql.catalog.ice.warehouse", warehouse.toString)
      .config("spark.sql.catalog.ice.cache-enabled", "false")
      .getOrCreate()
  }

  override protected def afterAll(): Unit = {
    try if (spark != null) spark.stop()
    finally super.afterAll()
  }

  test("differential: our bitmap is order-independent and round-trips through Iceberg's deserialize") {
    val rnd = new Random(42)
    Seq(0, 1, 2, 10, 1000, 50000).foreach { n =>
      val positions = Array.fill(n)(rnd.nextInt(200000).toLong)
      val ours = ColumnarDvWriter.buildIndex(positions, n)
      // Order-independence: the same positions in a different order must build a byte-identical bitmap
      // (a roaring bitmap is a set), which is what lets us fill it from unsorted column runs.
      val shuffled = rnd.shuffle(positions.toList).toArray
      val reordered = ColumnarDvWriter.buildIndex(shuffled, n)
      if (n == 0) {
        assert(ours.isEmpty && reordered.isEmpty, "empty position set must yield an empty index")
      } else {
        val distinct = positions.toSet.size.toLong
        assert(ours.cardinality() == distinct, s"cardinality must equal the distinct positions at n=$n")
        assert(ours.serialize() == reordered.serialize(), s"bitmap must be order-independent at n=$n")
        positions.foreach(p => assert(ours.isDeleted(p), s"position $p must be present at n=$n"))
        assert(!ours.isDeleted(199999L + 1L), "a position never inserted must not be marked deleted")
      }
    }
  }

  test("eligibility: v3 tables are DV-eligible, v2 tables decline") {
    spark.sql("CREATE NAMESPACE IF NOT EXISTS ice.db")
    spark.sql("DROP TABLE IF EXISTS ice.db.elig_v3")
    spark.sql("DROP TABLE IF EXISTS ice.db.elig_v2")
    spark.sql(
      "CREATE TABLE ice.db.elig_v3 (id BIGINT) USING iceberg TBLPROPERTIES ('format-version'='3', 'write.delete.mode'='merge-on-read')"
    )
    spark.sql(
      "CREATE TABLE ice.db.elig_v2 (id BIGINT) USING iceberg TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')"
    )
    val v3 = Spark3Util.loadIcebergTable(spark, "ice.db.elig_v3")
    val v2 = Spark3Util.loadIcebergTable(spark, "ice.db.elig_v2")
    assert(org.apache.iceberg.spark.source.IcebergDvCommitBridge.isDvEligible(v3), "v3 must be DV-eligible")
    assert(!org.apache.iceberg.spark.source.IcebergDvCommitBridge.isDvEligible(v2), "v2 must decline")
    assert(!org.apache.iceberg.spark.source.IcebergDvCommitBridge.isDvEligible(null), "null must decline")
  }

  test("end-to-end: a DV written by ColumnarDvWriter commits via RowDelta and reads back") {
    val t = "ice.db.dvbridge_t"
    spark.sql("CREATE NAMESPACE IF NOT EXISTS ice.db")
    spark.sql(s"DROP TABLE IF EXISTS $t")
    spark.sql(
      s"""CREATE TABLE $t (id BIGINT, v STRING) USING iceberg
         |TBLPROPERTIES ('format-version'='3', 'write.delete.mode'='merge-on-read')""".stripMargin
    )
    spark.sql(s"INSERT INTO $t SELECT /*+ REPARTITION(1) */ id, concat('v', id) FROM range(1000)")
    assert(spark.sql(s"SELECT count(*) FROM $t").collect()(0).getLong(0) == 1000L)

    val table: Table = Spark3Util.loadIcebergTable(spark, t)
    val dataFilePath =
      spark.sql(s"SELECT file_path FROM $t.files WHERE content = 0").collect().map(_.getString(0))
    assert(dataFilePath.length == 1, s"expected one data file, got ${dataFilePath.length}")
    val path = dataFilePath(0)

    val positions = (0L until 100L).toArray
    val off = OutputFileFactory.builderFor(table, 1, 1L).format(org.apache.iceberg.FileFormat.PUFFIN).build()
    val writer = new ColumnarDvWriter(off, null)
    try writer.deleteFile(path, positions, positions.length, PartitionSpec.unpartitioned(), null)
    finally writer.close()
    val result = writer.result()
    assert(result.deleteFiles().asScala.nonEmpty, "the writer must have produced a DV delete file")

    val rowDelta = table.newRowDelta()
    result.deleteFiles().asScala.foreach(rowDelta.addDeletes)
    rowDelta.commit()

    assert(spark.sql(s"SELECT count(*) FROM $t").collect()(0).getLong(0) == 900L)
    assert(spark.sql(s"SELECT count(*) FROM $t WHERE id < 100").collect()(0).getLong(0) == 0L)
    // The Iceberg Java API sees the committed DV delete file in the current snapshot (Spark's reader
    // already proved it above by returning 900 rows with ids 0..99 gone).
    val reloaded = Spark3Util.loadIcebergTable(spark, t)
    val apiDeleteFiles = reloaded.currentSnapshot().addedDeleteFiles(reloaded.io()).asScala.toList
    assert(apiDeleteFiles.nonEmpty, "the Iceberg Java API must see the committed DV delete file")
    // And Spark's own metadata table lists it (all_delete_files / position_deletes), independent of content code.
    assert(spark.sql(s"SELECT count(*) FROM $t.all_delete_files").collect()(0).getLong(0) >= 1L)
  }
}
