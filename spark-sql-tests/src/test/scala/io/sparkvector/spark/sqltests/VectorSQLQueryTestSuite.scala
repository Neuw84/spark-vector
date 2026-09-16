package io.sparkvector.spark.sqltests

import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

import io.sparkvector.spark.VectorSparkSessionExtensions
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLQueryTestSuite
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.util.QueryExecutionListener
import org.apache.spark.sql.vector.{PlanUtils, VectorPlan}

/**
 * Apache Spark's SQL golden-file suite (`sql/core/src/test/resources/sql-tests`, taken from the
 * `spark-sql` tests jar) with the spark-vector extension injected, the way Comet validates itself
 * against Spark's own tests. Every query must still produce Spark's golden result, whether an
 * operator was converted or left to Spark.
 *
 * Runs only under the `spark-sql-tests` Maven profile; `-DsqlTests.filter=<regex>` (the system
 * property `spark.vector.sqlTests.filter`) restricts the test cases by name.
 */
class VectorSQLQueryTestSuite extends SQLQueryTestSuite {

  override protected def sparkConf: SparkConf = super.sparkConf
    .set("spark.sql.extensions", classOf[VectorSparkSessionExtensions].getName)
    .set("spark.sql.queryExecutionListeners", classOf[VectorCoverageListener].getName)
    // Spark's parquet-backed test tables are small; make sure the vectorized reader is in play.
    .set("spark.sql.parquet.enableVectorizedReader", "true")

  // Called from the parent constructor, before this class's fields exist: read the property here.
  private def selected(name: String): Boolean =
    System.getProperty("spark.vector.sqlTests.filter", ".*").r.findFirstIn(name).isDefined

  override protected def createScalaTestCase(testCase: TestCase): Unit =
    if (selected(testCase.name)) super.createScalaTestCase(testCase)

  /** Cases Spark itself keeps out of ordinary runs plus none of ours: every golden file must match. */
  override def ignoreList: Set[String] = super.ignoreList

  override def afterAll(): Unit = {
    try super.afterAll()
    finally {
      val total = VectorCoverageListener.executions.get()
      val accelerated = VectorCoverageListener.withVectorOperators.get()
      // scalastyle:off println
      println(s"[spark-vector] SQL test coverage: $accelerated of $total executions ran at least one spark-vector operator; " +
        s"${VectorCoverageListener.vectorOperators.get()} spark-vector operators in total")
      // scalastyle:on println
    }
  }
}

/** Counts, over the whole suite, how often a spark-vector operator ended up in an executed plan. */
class VectorCoverageListener extends QueryExecutionListener {
  override def onSuccess(funcName: String, qe: QueryExecution, durationNs: Long): Unit = record(qe)
  override def onFailure(funcName: String, qe: QueryExecution, exception: Exception): Unit = record(qe)

  private def record(qe: QueryExecution): Unit = {
    try {
      val ours = PlanUtils.allNodes(qe.executedPlan).count(_.isInstanceOf[VectorPlan])
      VectorCoverageListener.executions.incrementAndGet()
      if (ours > 0) {
        VectorCoverageListener.withVectorOperators.incrementAndGet()
        VectorCoverageListener.vectorOperators.addAndGet(ours)
      }
    } catch {
      case _: Throwable => // never let bookkeeping fail a query
    }
  }
}

object VectorCoverageListener {
  val executions = new AtomicLong()
  val withVectorOperators = new AtomicLong()
  val vectorOperators = new AtomicLong()

  private[sqltests] def lower(s: String): String = s.toLowerCase(Locale.ROOT)
}
