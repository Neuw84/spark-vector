package io.sparkvector.spark

import io.sparkvector.spark.test.{SparkVectorFunSuite, TestTables}
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec}

/** The `spark.plugins` entry point must register the extension without any other configuration. */
class VectorPluginSuite extends SparkVectorFunSuite {

  override protected def extraSparkConf: Map[String, String] =
    Map("spark.plugins" -> classOf[VectorPlugin].getName)

  test("plugin registers the session extension") {
    assert(spark.sparkContext.getConf.get("spark.sql.extensions").contains(classOf[VectorSparkSessionExtensions].getName))
    TestTables.createMixed(spark, newTempPath("plugin/t"), rows = 2000)
    val df = spark.sql("SELECT i FROM t WHERE i > 100")
    assert(df.count() === 1899)
    assert(PlanUtils.allNodes(df.queryExecution.executedPlan).exists(_.isInstanceOf[VectorFilterExec]))
  }
}
