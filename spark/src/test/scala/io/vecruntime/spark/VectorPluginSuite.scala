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
package io.sparkvector.spark

import io.sparkvector.spark.test.{SparkVectorFunSuite, TestTables}
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec}

/** The `spark.plugins` entry point must register the extension without any other configuration. */
class VectorPluginSuite extends SparkVectorFunSuite {

  override protected def extraSparkConf: Map[String, String] =
    Map("spark.plugins" -> classOf[VectorPlugin].getName)

  test("plugin registers the session extension") {
    assert(
      spark.sparkContext.getConf.get("spark.sql.extensions").contains(classOf[VectorSparkSessionExtensions].getName)
    )
    TestTables.createMixed(spark, newTempPath("plugin/t"), rows = 2000)
    val df = spark.sql("SELECT i FROM t WHERE i > 100")
    assert(df.count() === 1899)
    assert(PlanUtils.allNodes(df.queryExecution.executedPlan).exists(_.isInstanceOf[VectorFilterExec]))
  }
}
