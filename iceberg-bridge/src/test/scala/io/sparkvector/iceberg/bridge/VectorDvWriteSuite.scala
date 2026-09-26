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

import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * Integration tests for the columnar v3 deletion-vector writer (#20, slice 4-live), DELETE on v3.
 * They load the plugin's operator, planner strategy and session extensions (spark-vector-spark is a
 * test dependency of this module) and assert: the table is identical with the writer on and off, our
 * `VectorWriteDeltaExec` is in the plan when on, a v2 table falls back to Spark's writer, the DV files
 * are readable by Spark's metadata tables and the Iceberg Java API, and the snapshot summary counts
 * match Spark's. Runs in the iceberg-bridge module (which the gate's -pl list excludes), so it is run
 * manually -- see docs/iceberg-dv-writer.md.
 */
class VectorDvWriteSuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private lazy val warehouse: Path = Files.createTempDirectory("dvwrite")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("dv-write")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.extensions", "io.sparkvector.spark.VectorSparkSessionExtensions")
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

  private def withFlag[T](on: Boolean)(f: => T): T = {
    val key = "spark.vector.iceberg.dvWriter.enabled"
    val prev = spark.conf.getOption(key)
    spark.conf.set(key, on.toString)
    try f
    finally prev match { case Some(v) => spark.conf.set(key, v); case None => spark.conf.unset(key) }
  }

  private def rows(t: String): Array[Row] = spark.sql(s"SELECT * FROM $t ORDER BY id").collect()
  private def count(sql: String): Long = spark.sql(sql).collect()(0).getLong(0)

  private def createV3(name: String, fmtVersion: Int = 3): Unit = {
    spark.sql(s"DROP TABLE IF EXISTS $name")
    spark.sql(
      s"""CREATE TABLE $name (id BIGINT, p INT, v STRING) USING iceberg
         |TBLPROPERTIES ('format-version'='$fmtVersion', 'write.delete.mode'='merge-on-read',
         |  'write.target-file-size-bytes'='4096')""".stripMargin
    )
    spark.sql(
      s"""INSERT INTO $name
         |SELECT id, cast(id % 2 as int) as p, if(id % 10 = 0, null, concat('v', id)) as v
         |FROM range(2000)""".stripMargin
    )
  }

  private def planHasVectorWriteDelta(df: org.apache.spark.sql.DataFrame): Boolean =
    df.queryExecution.executedPlan.exists(_.getClass.getSimpleName == "VectorWriteDeltaExec")

  test("diagnostic: dump v3 DELETE plan") {
    createV3("ice.db.dv_diag")
    spark.conf.set("spark.vector.explainFallback.enabled", "true")
    info("bridge available: " + org.apache.spark.sql.vector.IcebergDvBridge.isAvailable)
    val tbl = org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark, "ice.db.dv_diag")
    info("isDvEligible(table): " + org.apache.spark.sql.vector.IcebergDvBridge.isDvEligible(tbl))
    withFlag(on = true) {
      val df = spark.sql("DELETE FROM ice.db.dv_diag WHERE id % 3 = 0")
      info("executedPlan:\n" + df.queryExecution.executedPlan.treeString)
      df.collect()
    }
  }

  test("DELETE on v3: identical on/off, our operator ran, DVs readable, snapshot counts match") {
    createV3("ice.db.dv_on")
    createV3("ice.db.dv_off")

    withFlag(on = false)(spark.sql("DELETE FROM ice.db.dv_off WHERE id % 3 = 0").collect())
    val onPlanHad = withFlag(on = true) {
      val df = spark.sql("DELETE FROM ice.db.dv_on WHERE id % 3 = 0")
      val had = planHasVectorWriteDelta(df)
      df.collect()
      had
    }
    assert(onPlanHad, "VectorWriteDeltaExec must be in the DELETE plan when the flag is on")

    val off = rows("ice.db.dv_off")
    val on = rows("ice.db.dv_on")
    assert(on.length == off.length, s"row count differs: on=${on.length} off=${off.length}")
    assert(on.sameElements(off), "table contents differ between writer on and off")
    assert(count("SELECT count(*) FROM ice.db.dv_on WHERE id % 3 = 0") == 0L, "deleted rows still present")

    assert(count("SELECT count(*) FROM ice.db.dv_on.all_delete_files") >= 1L, "no delete files visible to Spark")
    val table = org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark, "ice.db.dv_on")
    val summary = table.currentSnapshot().summary()
    val offTable = org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark, "ice.db.dv_off")
    val offSummary = offTable.currentSnapshot().summary()
    assert(
      summary.get("added-position-deletes") == offSummary.get("added-position-deletes"),
      s"added-position-deletes differ: on=${summary.get("added-position-deletes")} off=${offSummary.get("added-position-deletes")}"
    )
    assert(
      summary.get("added-delete-files") == offSummary.get("added-delete-files"),
      s"added-delete-files differ: on=${summary.get("added-delete-files")} off=${offSummary.get("added-delete-files")}"
    )
  }

  test("empty delete set on v3: no-op, identical on/off, our operator still planned") {
    createV3("ice.db.dv_empty_on")
    createV3("ice.db.dv_empty_off")
    withFlag(on = false)(spark.sql("DELETE FROM ice.db.dv_empty_off WHERE id > 100000").collect())
    withFlag(on = true) {
      val df = spark.sql("DELETE FROM ice.db.dv_empty_on WHERE id > 100000")
      assert(planHasVectorWriteDelta(df), "operator should still be planned for an empty delete")
      df.collect()
    }
    assert(rows("ice.db.dv_empty_on").sameElements(rows("ice.db.dv_empty_off")))
    assert(count("SELECT count(*) FROM ice.db.dv_empty_on") == 2000L)
  }

  test("v2 table falls back to Spark's WriteDeltaExec (our operator absent)") {
    createV3("ice.db.dv_v2", fmtVersion = 2)
    val had = withFlag(on = true) {
      val df = spark.sql("DELETE FROM ice.db.dv_v2 WHERE id % 3 = 0")
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    assert(!had, "v2 must fall back to Spark's writer; VectorWriteDeltaExec must be absent")
    assert(count("SELECT count(*) FROM ice.db.dv_v2 WHERE id % 3 = 0") == 0L)
  }

  test("repeated DELETEs on v3 merge previous DVs correctly") {
    createV3("ice.db.dv_rep_on")
    createV3("ice.db.dv_rep_off")
    for (m <- Seq(3, 4, 5)) {
      withFlag(on = false)(spark.sql(s"DELETE FROM ice.db.dv_rep_off WHERE id % $m = 0").collect())
      withFlag(on = true)(spark.sql(s"DELETE FROM ice.db.dv_rep_on WHERE id % $m = 0").collect())
    }
    assert(rows("ice.db.dv_rep_on").sameElements(rows("ice.db.dv_rep_off")), "repeated-merge contents differ")
  }
}
