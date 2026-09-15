package io.sparkvector.spark

import io.sparkvector.spark.test.SparkVectorFunSuite
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
      "jdk.incubator.vector is not resolved; pass --add-modules=jdk.incubator.vector")
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
