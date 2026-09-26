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
package io.vecruntime.iceberg.bridge

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
      .config("spark.sql.extensions", "io.vecruntime.spark.VectorSparkSessionExtensions")
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

  private def planHasVectorWriteDelta(df: org.apache.spark.sql.DataFrame): Boolean = {
    // A row-level DELETE plans to a V2CommandExec, which Spark wraps in a CommandResultExec and runs
    // eagerly, caching the rows. CommandResultExec is a leaf in the executed-plan tree -- it holds the
    // real physical command in `commandPhysicalPlan`, not as a tree child -- and that command may in
    // turn be an AdaptiveSparkPlanExec that hides its body behind `executedPlan`. A plain tree walk of
    // `executedPlan` therefore never sees VectorWriteDeltaExec even when it ran. Unwrap both.
    import org.apache.spark.sql.execution.SparkPlan
    def unwrap(p: SparkPlan): Seq[SparkPlan] = {
      val nested = p.getClass.getSimpleName match {
        case "CommandResultExec" =>
          p.getClass.getMethods
            .find(m => m.getName == "commandPhysicalPlan" && m.getParameterCount == 0)
            .map(_.invoke(p).asInstanceOf[SparkPlan])
            .toSeq
        case "AdaptiveSparkPlanExec" =>
          p.getClass.getMethods
            .find(m => m.getName == "executedPlan" && m.getParameterCount == 0)
            .map(_.invoke(p).asInstanceOf[SparkPlan])
            .toSeq
        case _ => Seq.empty
      }
      p +: (p.children ++ nested).flatMap(unwrap)
    }
    unwrap(df.queryExecution.executedPlan).exists(_.getClass.getSimpleName == "VectorWriteDeltaExec")
  }

  test("bridge + eligibility preconditions hold on a v3 table") {
    createV3("ice.db.dv_pre")
    assert(org.apache.spark.sql.vector.IcebergDvBridge.isAvailable, "bridge must be on the classpath in this module")
    val tbl = org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark, "ice.db.dv_pre")
    assert(org.apache.spark.sql.vector.IcebergDvBridge.isDvEligible(tbl), "v3 table must be DV-eligible")
    val df = withFlag(on = true)(spark.sql("DELETE FROM ice.db.dv_pre WHERE id % 3 = 0"))
    assert(planHasVectorWriteDelta(df), "our operator must be planned when every gate passes")
    df.collect()
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

  test("repeated DELETEs on v3: first accelerates, later ones fall back, result correct") {
    createV3("ice.db.dv_rep_on")
    createV3("ice.db.dv_rep_off")
    val hadPlanned = scala.collection.mutable.ArrayBuffer[Boolean]()
    for (m <- Seq(3, 4, 5)) {
      withFlag(on = false)(spark.sql(s"DELETE FROM ice.db.dv_rep_off WHERE id % $m = 0").collect())
      val had = withFlag(on = true) {
        val df = spark.sql(s"DELETE FROM ice.db.dv_rep_on WHERE id % $m = 0")
        val h = planHasVectorWriteDelta(df)
        df.collect()
        h
      }
      hadPlanned += had
    }
    // First DELETE has no prior deletes -> operator runs; the table then carries a DV, so the next
    // DELETEs decline (repeated-DV merge is a later slice) and Spark's writer keeps them correct.
    assert(hadPlanned.head, "the first DELETE on a clean v3 table must use the columnar operator")
    assert(hadPlanned.tail.forall(!_), "DELETEs after the table has deletes must fall back")
    assert(rows("ice.db.dv_rep_on").sameElements(rows("ice.db.dv_rep_off")), "repeated-delete contents differ")
  }

  test("partitioned v3 DELETE falls back (partitioned not yet supported), result correct") {
    def createPart(name: String): Unit = {
      spark.sql(s"DROP TABLE IF EXISTS $name")
      spark.sql(
        s"""CREATE TABLE $name (id BIGINT, p INT, v STRING) USING iceberg PARTITIONED BY (p)
           |TBLPROPERTIES ('format-version'='3', 'write.delete.mode'='merge-on-read',
           |  'write.target-file-size-bytes'='4096')""".stripMargin
      )
      spark.sql(
        s"""INSERT INTO $name
           |SELECT id, cast(id % 4 as int) as p, if(id % 10 = 0, null, concat('v', id)) as v
           |FROM range(4000)""".stripMargin
      )
    }
    createPart("ice.db.dv_part_on")
    createPart("ice.db.dv_part_off")
    withFlag(on = false)(spark.sql("DELETE FROM ice.db.dv_part_off WHERE id % 3 = 0").collect())
    val onHad = withFlag(on = true) {
      val df = spark.sql("DELETE FROM ice.db.dv_part_on WHERE id % 3 = 0")
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    // A partitioned target is out of scope this landing: the strategy declines, Spark's writer runs.
    assert(!onHad, "partitioned v3 DELETE must fall back to Spark's writer (operator absent)")
    assert(rows("ice.db.dv_part_on").sameElements(rows("ice.db.dv_part_off")), "partitioned contents differ")
    assert(count("SELECT count(*) FROM ice.db.dv_part_on WHERE id % 3 = 0") == 0L, "deleted rows still present")
  }

  test("a failing task aborts: nothing is committed, table unchanged") {
    createV3("ice.db.dv_abort")
    val before = org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark, "ice.db.dv_abort")
    val snapBefore = before.currentSnapshot().snapshotId()
    val liveBefore = count("SELECT count(*) FROM ice.db.dv_abort")

    // A UDF that throws mid-scan forces the delete-write RDD job to fail, so VectorWriteDeltaExec's
    // catch path must call Iceberg's DeltaBatchWrite.abort and rethrow -- no snapshot may be created.
    spark.udf.register("dv_boom", (id: Long) => if (id == 123L) throw new RuntimeException("boom") else id % 3 == 0)
    val ex = intercept[Exception] {
      withFlag(on = true)(spark.sql("DELETE FROM ice.db.dv_abort WHERE dv_boom(id)").collect())
    }
    assert(ex != null, "the failing task must surface an exception")

    val after = org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark, "ice.db.dv_abort")
    assert(after.currentSnapshot().snapshotId() == snapBefore, "a failed delete must not create a new snapshot")
    assert(
      count("SELECT count(*) FROM ice.db.dv_abort") == liveBefore,
      "the table must be unchanged after an aborted delete"
    )
  }
}
