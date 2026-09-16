package io.sparkvector.spark.ui

import java.net.{HttpURLConnection, URI}

import scala.io.Source

import io.sparkvector.spark.VectorPlugin
import io.sparkvector.spark.test.{SparkVectorFunSuite, TestTables}
import org.apache.spark.sql.vector.ui.{Engine, PlanAcceleration}
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec}
import org.scalatest.concurrent.Eventually
import org.scalatest.time.{Seconds, Span}

/**
 * End to end check of the acceleration tab: the plugin attaches it to a real (bound) Spark UI, a
 * query is classified, and both pages serve HTML that names the engines.
 */
class VectorAccelerationUiSuite extends SparkVectorFunSuite with Eventually {

  private val ExecutionIdPattern = """/vector/execution/\?id=(\d+)""".r

  override protected def extraSparkConf: Map[String, String] = Map(
    "spark.plugins" -> classOf[VectorPlugin].getName,
    "spark.ui.enabled" -> "true",
    "spark.ui.port" -> "0")

  private def uiUrl: String =
    spark.sparkContext.uiWebUrl.getOrElse(fail("the Spark UI is not running"))

  private def get(path: String): String = {
    val conn = URI.create(s"$uiUrl$path").toURL.openConnection().asInstanceOf[HttpURLConnection]
    conn.setRequestMethod("GET")
    try {
      assert(conn.getResponseCode === 200, s"GET $path returned ${conn.getResponseCode}")
      val src = Source.fromInputStream(conn.getInputStream, "UTF-8")
      try src.mkString
      finally src.close()
    } finally conn.disconnect()
  }

  test("the tab is attached and lists a classified execution") {
    TestTables.createMixed(spark, newTempPath("ui/t"), rows = 2000)
    val df = spark.sql("SELECT i FROM t WHERE i > 100")
    assert(df.count() === 1899)
    // The rule must have converted the filter, otherwise there is nothing to colour.
    assert(PlanUtils.allNodes(df.queryExecution.executedPlan).exists(_.isInstanceOf[VectorFilterExec]))

    val listPage = get("/vector/")
    assert(listPage.contains("Vector Acceleration"))
    assert(listPage.contains("Java Vector API"), "the legend should name the Vector API engine")

    // The listener runs on the bus asynchronously, so wait for the executions to show up. Several
    // are recorded (writing the Parquet table, reading it, the count); find the one holding the
    // filter rather than assuming an id.
    val planPage = eventually(timeout(Span(30, Seconds))) {
      val ids = ExecutionIdPattern.findAllMatchIn(get("/vector/")).map(_.group(1)).toSeq.distinct
      assert(ids.nonEmpty, "at least one execution should be listed")
      val pages = ids.map(id => get(s"/vector/execution/?id=$id"))
      assert(pages.forall(_.contains("digraph G")), "every plan page should embed a DOT file")
      pages.find(_.contains("VectorFilter"))
        .getOrElse(fail(s"no plan among ${ids.mkString(", ")} contains the converted filter"))
    }
    assert(planPage.contains("sv-engine-vector"), "a node should be coloured as Vector API")
    // The plan text is embedded, collapsed, behind Spark's own collapseTable toggle (Spark 4 ships
    // Bootstrap 4, so Bootstrap 5 data-bs-* attributes would silently do nothing).
    assert(planPage.contains("== Physical Plan =="), "the plan details should be on the page")
    assert(planPage.contains("collapseTable('collapse-sv-plan-details','sv-plan-details')"))
  }

  test("an unknown or malformed execution id renders a not-found page, not an error") {
    assert(get("/vector/execution/?id=999999999").contains("No acceleration information"))
    assert(get("/vector/execution/?id=not-a-number").contains("No acceleration information"))
    assert(get("/vector/execution/").contains("No acceleration information"))
  }

  test("static resources are served") {
    assert(get("/static/vector/vector-acceleration.css").contains("sv-engine-vector"))
    assert(get("/static/vector/vector-acceleration.js").contains("graphlibDot"))
  }

  test("a scan plus our operators is fully accelerated") {
    TestTables.createMixed(spark, newTempPath("ui/full"), rows = 1000)
    // Scan (columnar source) -> VectorFilter -> ColumnarToRow (transition). The scan and the
    // transition are plumbing, so the only operator is ours.
    val plan = PlanAcceleration.fromPlan(
      spark.sql("SELECT i FROM t WHERE i > 10").queryExecution.executedPlan)
    assert(plan.countBy(Engine.Vector) >= 1)
    assert(plan.countBy(Engine.ColumnarSource) === 1, "the vectorized scan is a columnar source")
    assert(plan.countBy(Engine.Spark) === 0)
    assert(plan.fullyAccelerated)
  }

  test("an operator we do not implement is not fully accelerated") {
    TestTables.createMixed(spark, newTempPath("ui/partial"), rows = 1000)
    // The shuffle behind the group-by is a Spark exchange: we have no columnar shuffle of our own
    // without Comet, so the plan must not claim to be fully accelerated.
    val plan = PlanAcceleration.fromPlan(
      spark.sql("SELECT s, count(*) FROM t GROUP BY s").queryExecution.executedPlan)
    assert(plan.countBy(Engine.Spark) >= 1)
    assert(!plan.fullyAccelerated)
  }

  test("classification separates accelerated operators from fallbacks") {
    TestTables.createMixed(spark, newTempPath("ui/classify"), rows = 1000)
    // A LIKE with inner wildcards does not compile to the kernels, so the filter stays with Spark and the
    // rule records why. That is a genuine miss, unlike a scan or an exchange.
    val fellBack = PlanAcceleration.fromPlan(
      spark.sql("SELECT i FROM t WHERE s LIKE 'a%b%c'").queryExecution.executedPlan)
    assert(fellBack.countBy(Engine.Spark) >= 1)
    assert(!fellBack.fullyAccelerated)
    assert(fellBack.fallbacks.nonEmpty, "the fallback reason should be recorded on the node")
    assert(fellBack.fallbacks.exists(_._2.contains("Like")))
  }
}
