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
package io.sparkvector.spark.iceberg

import java.nio.file.{Files, Path}

import io.sparkvector.spark.VectorConf
import io.sparkvector.spark.test.{IcebergTest, VectorQuerySuite}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.vector.{IcebergDvBridge, VectorWriteDeltaStrategy}
import org.scalatest.Tag

/**
 * Gate-visible tests for the columnar v3 deletion-vector writer's DECLINE paths (#20, slice 4-live).
 *
 * The correctness suite for the operator itself (`VectorDvWriteSuite`) lives in the optional
 * `iceberg-bridge` module, which CI's gate (`-pl kernels,spark,shuffle,benchmarks`) does not build, so
 * it is run manually (see docs/iceberg-dv-writer.md). This suite runs INSIDE the gate's spark module,
 * where the bridge is by construction absent from the classpath, and asserts the other half of the
 * contract: whenever the strategy must decline -- the flag is off, the target is a v2 table, the write
 * has an insert half (UPDATE / MERGE, slice 5), or the bridge module is not present -- Spark's own
 * `WriteDeltaExec` plans instead and the DELETE / UPDATE / MERGE result is correct and unchanged.
 *
 * `VectorWriteDeltaExec` must therefore NEVER appear in a plan built in this module. The check is by
 * simple class name so it does not depend on the operator being loadable here.
 */
class DvWriteStrategyFallbackSuite extends VectorQuerySuite {

  override protected def extraSparkConf: Map[String, String] =
    super.extraSparkConf ++ IcebergTables.catalogConf(DvWriteStrategyFallbackSuite.warehouse.toString)

  private def planHasVectorWriteDelta(df: DataFrame): Boolean =
    df.queryExecution.executedPlan.exists(_.getClass.getSimpleName == "VectorWriteDeltaExec")

  private def liveRows(t: String): Long = spark.sql(s"SELECT count(*) FROM $t").collect()(0).getLong(0)

  private def createV3(name: String, formatVersion: Int = 3): Unit = {
    spark.sql(s"DROP TABLE IF EXISTS $name")
    spark.sql(
      s"""CREATE TABLE $name (id BIGINT, p INT, v STRING) USING iceberg
         |TBLPROPERTIES ('format-version'='$formatVersion', 'write.delete.mode'='merge-on-read',
         |  'write.update.mode'='merge-on-read', 'write.merge.mode'='merge-on-read',
         |  'write.target-file-size-bytes'='4096')""".stripMargin
    )
    spark.sql(
      s"""INSERT INTO $name
         |SELECT id, cast(id % 2 as int) as p, if(id % 10 = 0, null, concat('v', id)) as v
         |FROM range(2000)""".stripMargin
    )
  }

  private def withDvWriter[T](on: Boolean)(f: => T): T =
    withConf(VectorConf.IcebergDvWriterEnabled -> on.toString)(f)

  test(
    "the bridge module is absent in the gate build, so the strategy can never take a write",
    DvWriteStrategyFallbackSuite.Tags: _*
  ) {
    // The whole point of this suite: the operator's own module is not on the gate classpath.
    assert(!IcebergDvBridge.isAvailable, "iceberg-bridge must NOT be on the gate classpath")
    // With the bridge gone, the strategy declines every WriteDelta regardless of the other gates.
    createV3("ice.db.fb_bridge_absent")
    val had = withDvWriter(on = true) {
      val df = spark.sql("DELETE FROM ice.db.fb_bridge_absent WHERE id % 3 = 0")
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    assert(!had, "bridge absent: VectorWriteDeltaExec must not be planned")
    assert(liveRows("ice.db.fb_bridge_absent") == 2000L - 667L, "DELETE result must be correct via Spark's writer")
  }

  test("flag off: v3 DELETE falls back to Spark's writer, result correct", DvWriteStrategyFallbackSuite.Tags: _*) {
    createV3("ice.db.fb_flag_off")
    val had = withDvWriter(on = false) {
      val df = spark.sql("DELETE FROM ice.db.fb_flag_off WHERE id % 3 = 0")
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    assert(!had, "flag off: VectorWriteDeltaExec must be absent")
    assert(liveRows("ice.db.fb_flag_off") == 2000L - 667L)
  }

  test("v2 table: DELETE falls back to Spark's writer, result correct", DvWriteStrategyFallbackSuite.Tags: _*) {
    createV3("ice.db.fb_v2", formatVersion = 2)
    val had = withDvWriter(on = true) {
      val df = spark.sql("DELETE FROM ice.db.fb_v2 WHERE id % 3 = 0")
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    assert(!had, "v2: VectorWriteDeltaExec must be absent")
    assert(liveRows("ice.db.fb_v2") == 2000L - 667L)
  }

  test(
    "UPDATE and MERGE (insert half, slice 5) fall back to Spark's writer, result correct",
    DvWriteStrategyFallbackSuite.Tags: _*
  ) {
    createV3("ice.db.fb_update")
    val updHad = withDvWriter(on = true) {
      val df = spark.sql("UPDATE ice.db.fb_update SET v = concat('u', id) WHERE id % 7 = 0")
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    assert(!updHad, "UPDATE has an insert half: VectorWriteDeltaExec must be absent (slice 5)")
    assert(liveRows("ice.db.fb_update") == 2000L, "UPDATE keeps the row count")
    assert(
      spark.sql(
        "SELECT count(*) FROM ice.db.fb_update WHERE id % 7 = 0 AND v = concat('u', id)"
      ).collect()(0).getLong(0)
        == (2000L + 6) / 7,
      "UPDATE must have rewritten the matched rows"
    )

    createV3("ice.db.fb_merge")
    spark.sql(
      "CREATE OR REPLACE TEMP VIEW fb_src AS SELECT id, cast(id % 2 as int) p, concat('m', id) v FROM range(1500, 2500)"
    )
    val mergeHad = withDvWriter(on = true) {
      val df = spark.sql(
        """MERGE INTO ice.db.fb_merge t USING fb_src s ON t.id = s.id
          |WHEN MATCHED AND t.id % 5 = 0 THEN DELETE
          |WHEN MATCHED THEN UPDATE SET t.v = s.v
          |WHEN NOT MATCHED THEN INSERT *""".stripMargin
      )
      val h = planHasVectorWriteDelta(df)
      df.collect()
      h
    }
    assert(!mergeHad, "MERGE has an insert half: VectorWriteDeltaExec must be absent (slice 5)")
    // Sanity: no duplicated/lost rows -- 2000 original + 500 new (2000..2499) minus matched-and-deleted (id in [1500,2000), id%5==0).
    val deleted = (1500 until 2000).count(_ % 5 == 0)
    assert(liveRows("ice.db.fb_merge") == 2000L + 500L - deleted, "MERGE row count must match Spark's writer")
  }
}

object DvWriteStrategyFallbackSuite {
  lazy val warehouse: Path = Files.createTempDirectory("spark-vector-dvfallback")
  val Tags: Seq[Tag] = Seq(IcebergTest)
  // Force the strategy object to load so a stale classfile cannot mask a compile break in the suite.
  private val _strategyRef: Class[_] = classOf[VectorWriteDeltaStrategy]
}
