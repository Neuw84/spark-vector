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
package org.apache.spark.sql.vecruntime.shuffle

import java.nio.file.{Files, Path}

import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

/**
 * The shuffle manager was renamed in 0.0.2 from
 * `org.apache.spark.sql.vector.shuffle.VectorShuffleManager` to
 * `org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager`, with the old name kept as a
 * deprecated delegating alias. This verifies both class names configure a working columnar shuffle
 * end to end and produce identical results.
 */
class ShuffleManagerAliasSuite extends AnyFunSuite with BeforeAndAfterEach {

  private val NewName = "org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager"
  private val OldAlias = "org.apache.spark.sql.vector.shuffle.VectorShuffleManager"

  private var spark: SparkSession = _
  private var tempDir: Path = _

  override def afterEach(): Unit = {
    if (spark != null) {
      spark.stop()
      spark = null
    }
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()
  }

  private def sessionWith(manager: String): SparkSession = {
    tempDir = Files.createTempDirectory("vecruntime-alias")
    SparkSession.builder()
      .master("local[4]")
      .appName(s"ShuffleManagerAliasSuite-$manager")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "6")
      .config("spark.sql.warehouse.dir", tempDir.resolve("wh").toString)
      .config("spark.plugins", "io.vecruntime.spark.VectorPlugin")
      .config("spark.shuffle.manager", manager)
      .config("spark.vector.shuffle.enabled", "true")
      .config("spark.vector.exec.strictFloatingPoint", "true")
      .getOrCreate()
  }

  /** A small query that forces a shuffle (group-by aggregation), returned as sorted rows. */
  private def runShuffleQuery(session: SparkSession): Seq[Row] = {
    val df = session.range(0, 20000).selectExpr("id", "cast(id % 97 as int) as k")
    df.createOrReplaceTempView("t")
    session.sql("SELECT k, count(*) c, sum(id) s FROM t GROUP BY k ORDER BY k").collect().toSeq
  }

  test("the new shuffle manager name runs a shuffle query") {
    spark = sessionWith(NewName)
    assert(spark.conf.get("spark.shuffle.manager") == NewName)
    val rows = runShuffleQuery(spark)
    assert(rows.size == 97)
    assert(rows.map(_.getLong(1)).sum == 20000L)
  }

  test("the deprecated old alias name still runs the same shuffle query with equal results") {
    // Reference result from the new name.
    spark = sessionWith(NewName)
    val expected = runShuffleQuery(spark)
    spark.stop()
    spark = null
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()

    // Same query under the deprecated alias.
    spark = sessionWith(OldAlias)
    assert(spark.conf.get("spark.shuffle.manager") == OldAlias)
    val actual = runShuffleQuery(spark)

    assert(actual == expected)
    assert(actual.size == 97)
  }

  test("the deprecated alias class instantiates and is a ShuffleManager") {
    val conf = new org.apache.spark.SparkConf(false)
    val mgr: org.apache.spark.shuffle.ShuffleManager =
      new org.apache.spark.sql.vector.shuffle.VectorShuffleManager(conf): @annotation.nowarn(
        "cat=deprecation"
      )
    assert(mgr != null)
    mgr.stop()
  }
}
