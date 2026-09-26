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
package io.vecruntime.spark

import io.vecruntime.spark.test.SparkVectorFunSuite
import org.apache.spark.sql.execution.FileSourceScanExec

/**
 * Day-one viability check: Spark 4.1 must start and run a vectorized Parquet scan on the JDK we
 * target (25), and that scan must report itself as columnar so our operators can consume it.
 */
class SparkOnJdkSmokeSuite extends SparkVectorFunSuite {

  test("JVM is the targeted release and Vector API module is loaded") {
    assert(Runtime.version().feature() >= 25, s"expected JDK 25+, got ${Runtime.version()}")
    assert(
      ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent,
      "jdk.incubator.vector is not resolved; pass --add-modules=jdk.incubator.vector"
    )
  }

  test("vectorized Parquet scan is columnar") {
    val path = newTempPath("smoke/parquet")
    spark.range(0, 1000).selectExpr("id", "cast(id as double) as d", "id % 3 as m").write.parquet(path)

    val df = spark.read.parquet(path).filter("m = 1")
    assert(df.count() === 333L)

    val scans = df.queryExecution.executedPlan.collect { case s: FileSourceScanExec => s }
    assert(scans.nonEmpty, "expected a FileSourceScanExec")
    assert(scans.forall(_.supportsColumnar), "Parquet scan should be columnar (vectorized reader)")
  }
}
