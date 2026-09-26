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
package io.vecruntime.spark.iceberg

import java.nio.file.{Files, Path}

import scala.collection.mutable

import io.vecruntime.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.scheduler.{SparkListener, SparkListenerTaskEnd}
import org.apache.spark.sql.SparkSession
import io.vecruntime.spark.test.IcebergTest
import org.scalatest.Tag

/**
 * Slice 1 of the option-B DV-writer plan (#20): a LOCAL timing split of the v3 merge-on-read write
 * stage into delete-writing (deletion vectors, Puffin) vs data-writing (Parquet data files), so the
 * design's gain estimate is a number rather than a range before any writer is built.
 *
 * It is a profiling probe, not a correctness assertion: it prints a table and asserts only that the
 * three phases ran and produced the delete/data files we expect. Attribution is by isolating the two
 * halves the CDC MERGE fuses, on the same v3 table shape, driven by Spark's own row writer (the
 * plugin is off): a pure `DELETE` writes only deletion vectors (no data files), a pure append of the
 * merge's inserted rows writes only Parquet data files (no deletes), and the real `MERGE` does both.
 * The delete-writer share of the write is then delete-task-time / (delete + data task time), and the
 * MERGE's own write-stage task time is reported beside it as the ground truth those two sum toward.
 */
class DvWriteProfileSuite extends VectorQuerySuite {

  override protected def extraSparkConf: Map[String, String] =
    super.extraSparkConf ++ IcebergTables.catalogConf(DvWriteProfileSuite.warehouse.toString) +
      ("spark.vector.enabled" -> "false") // Spark's own row writer, so the split is the baseline's

  /** Sums task run-time (ms) of the stages a body triggers, tagged by a marker we set per phase. */
  private class WriteStageTimer extends SparkListener {
    val taskMillis = new mutable.LongMap[Long]()
    @volatile var phase: String = "none"
    private val perPhase = mutable.Map[String, Long]().withDefaultValue(0L)
    override def onTaskEnd(e: SparkListenerTaskEnd): Unit = synchronized {
      perPhase(phase) += e.taskInfo.duration
    }
    def reset(p: String): Unit = synchronized { phase = p }
    def millis(p: String): Long = synchronized { perPhase(p) }
  }

  private def rows(spark: SparkSession, t: String): Long =
    spark.sql(s"SELECT count(*) FROM $t").collect()(0).getLong(0)

  private def deleteFiles(spark: SparkSession, t: String): Long = IcebergTables.deleteFileCount(spark, t)

  private def dataFiles(spark: SparkSession, t: String): Long =
    spark.sql(s"SELECT count(*) FROM $t.files WHERE content = 0").collect()(0).getLong(0)

  test("slice 1: v3 write-stage split, delete-writing vs data-writing", DvWriteProfileSuite.Tags: _*) {
    val Db = IcebergTables.Db
    val timer = new WriteStageTimer
    spark.sparkContext.addSparkListener(timer)
    try {
      spark.sql(s"CREATE NAMESPACE IF NOT EXISTS $Db")

      // A v3 merge-on-read table, heavily mutated so many data files carry DVs, like the cluster shape.
      def buildBase(name: String): Unit = {
        spark.sql(s"DROP TABLE IF EXISTS $name")
        val w = TestTables.mixedDataFrame(spark, DvWriteProfileSuite.Rows).repartition(8).writeTo(name).using("iceberg")
        Seq(
          "format-version" -> "3",
          "write.delete.mode" -> "merge-on-read",
          "write.update.mode" -> "merge-on-read",
          "write.merge.mode" -> "merge-on-read",
          "write.target-file-size-bytes" -> (512 * 1024).toString
        ).foldLeft(w) { case (b, (k, v)) => b.tableProperty(k, v) }.createOrReplace()
      }

      // Phase D: a pure DELETE on a v3 MoR table -> deletion vectors only, no data files written.
      val delT = s"$Db.p_delete"
      buildBase(delT)
      val dataFilesBefore = dataFiles(spark, delT)
      timer.reset("delete")
      spark.sql(s"DELETE FROM $delT WHERE i % 3 = 0").collect()
      val delFilesAfter = deleteFiles(spark, delT)
      val dataFilesAfterDelete = dataFiles(spark, delT)

      // Phase I: a pure append of a fresh row set the size of the DELETE's touched rows -> Parquet
      // data files only, no deletes. Sized to roughly match the number of positions the DELETE
      // removed (i % 3 == 0), so the two phases move comparable row volumes through the writer.
      val insT = s"$Db.p_insert"
      buildBase(insT)
      val insertRows = DvWriteProfileSuite.Rows / 3
      timer.reset("insert")
      spark
        .sql(
          s"""INSERT INTO $insT
             |SELECT cast(id as int) as i, cast(id * 5 as bigint) as l, cast(id as double) / 3 as d,
             |  10.0 + cast(id % 7 as double) as d2, date_add(date '2021-06-01', cast(id % 100 as int)) as dt,
             |  id % 2 = 0 as b, if(id % 15 = 0, null, concat('m', id % 5)) as s
             |FROM range(${DvWriteProfileSuite.Rows}, ${DvWriteProfileSuite.Rows + insertRows})""".stripMargin
        )
        .collect()

      // Phase M: the real CDC MERGE (delete + update + insert) on a fresh copy -> both halves.
      val merT = s"$Db.p_merge"
      buildBase(merT)
      IcebergTables.createMergeSource(spark)
      timer.reset("merge")
      spark.sql(IcebergTables.mergeSql(merT)).collect()

      val d = timer.millis("delete")
      val i = timer.millis("insert")
      val m = timer.millis("merge")
      val share = if (d + i > 0) d.toDouble / (d + i) else 0.0

      info("=== SLICE 1: v3 write-stage split (local, plugin OFF, Spark's row writer) ===")
      info(f"rows in base table:            ${DvWriteProfileSuite.Rows}%,d")
      info(
        f"phase D (DELETE -> DVs only):   task-time ${d}%,d ms, delete files after=$delFilesAfter, data files delta=${dataFilesAfterDelete - dataFilesBefore}"
      )
      info(f"phase I (INSERT -> data only):  task-time ${i}%,d ms")
      info(f"phase M (MERGE -> both halves): task-time ${m}%,d ms")
      info(f"delete-writer share D/(D+I):    ${share * 100}%.1f%%")

      assert(delFilesAfter > 0, "the pure DELETE must have written deletion vectors")
      assert(dataFilesAfterDelete == dataFilesBefore, "a v3 DELETE must not write data files")
    } finally {
      spark.sparkContext.removeSparkListener(timer)
    }
  }
}

object DvWriteProfileSuite {
  val Rows = 400000
  lazy val warehouse: Path = Files.createTempDirectory("spark-vector-dvprofile")
  val Tags: Seq[Tag] = Seq(IcebergTest)
}
