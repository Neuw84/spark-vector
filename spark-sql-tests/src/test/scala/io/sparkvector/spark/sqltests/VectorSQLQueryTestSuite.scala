package io.sparkvector.spark.sqltests

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

import scala.jdk.CollectionConverters._

import io.sparkvector.spark.VectorSparkSessionExtensions
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLQueryTestSuite
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.util.QueryExecutionListener
import org.apache.spark.sql.vector.{ListenerSync, PlanUtils, VectorPlan}

/**
 * Apache Spark's SQL golden-file suite (`sql/core/src/test/resources/sql-tests`, taken from the
 * `spark-sql` tests jar) with the spark-vector extension injected, the way Comet validates itself
 * against Spark's own tests. Every query must still produce Spark's golden result, whether an
 * operator was converted or left to Spark.
 *
 * Passing is the low bar -- a file passes just as well when every operator falls back -- so the run
 * also reports, per test case, how many executions ran at least one spark-vector operator, and a
 * full run compares those counts with the checked-in baseline (`vector-sql-coverage.tsv`): a case
 * whose accelerated count dropped fails the suite, because a fallback introduced by a planner
 * change is otherwise invisible (#17). `-DsqlTests.updateBaseline=true` rewrites the baseline from
 * the run instead.
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

  // Called from the parent constructor, before this class's fields exist: read the properties here.
  private def selected(name: String): Boolean =
    System.getProperty("spark.vector.sqlTests.filter", ".*").r.findFirstIn(name).isDefined &&
      Option(System.getProperty("spark.vector.sqlTests.exclude")).filter(_.nonEmpty)
        .getOrElse(VectorSQLQueryTestSuite.defaultExclude).r.findFirstIn(name).isEmpty

  override protected def createScalaTestCase(testCase: TestCase): Unit =
    if (selected(testCase.name)) super.createScalaTestCase(testCase)

  /** Cases Spark itself keeps out of ordinary runs plus none of ours: every golden file must match. */
  override def ignoreList: Set[String] = super.ignoreList

  /**
   * Attributes the listener's counts to the running test case. The listener bus delivers
   * asynchronously, so the bus is drained before the case is closed -- otherwise the last queries
   * of one case would land on the next.
   */
  override protected def runSqlTestCase(testCase: TestCase, listTestCases: Seq[TestCase]): Unit = {
    VectorCoverageListener.beginCase(testCase.name)
    try super.runSqlTestCase(testCase, listTestCases)
    finally {
      try ListenerSync.drain(spark)
      catch { case _: Throwable => }
      VectorCoverageListener.endCase()
    }
  }

  override def afterAll(): Unit = {
    try super.afterAll()
    finally VectorSQLQueryTestSuite.report(isFullRun =
        System.getProperty("spark.vector.sqlTests.filter", ".*") == ".*" &&
          Option(
            System.getProperty("spark.vector.sqlTests.exclude")
          ).filter(_.nonEmpty).forall(_ == VectorSQLQueryTestSuite.defaultExclude)
      )
  }
}

object VectorSQLQueryTestSuite {

  /**
   * Excluded by default (`-DsqlTests.exclude=<regex>` overrides, `-DsqlTests.exclude=^$` runs all):
   *  - `explain*.sql`: the golden files spell out Spark's physical plan, which is exactly what the
   *    extension replaces (Comet skips them for the same reason);
   *  - `hll`, `kllquantiles`, `thetasketch`: DataSketches' memory library refuses to initialise on
   *    any JDK newer than 21 (`Unsupported JDK Major Version`), which is Spark's dependency, not
   *    ours; the failure aborts the whole run;
   *  - `udtf/udtf.sql`: needs `pyspark` installed for `python3` (the Python UDF cases skip
   *    themselves when it is missing, this one fails instead).
   */
  val defaultExclude: String = "^(explain(-[a-z]+)?|hll|kllquantiles|thetasketch|udtf/udtf)\\.sql"

  /** The checked-in floor: `case<TAB>executions<TAB>accelerated` per line, sorted by case name. */
  val BaselineResource = "vector-sql-coverage.tsv"
  private val baselineSource = Paths.get("src", "test", "resources", BaselineResource)
  private val runOutput = Paths.get("target", BaselineResource)

  /** One test case's counts: executions, executions with at least one spark-vector operator, operators. */
  final case class Coverage(executions: Long, accelerated: Long, operators: Long)

  def readBaseline(): Map[String, Coverage] = {
    val in = getClass.getClassLoader.getResourceAsStream(BaselineResource)
    if (in == null) Map.empty
    else
      try {
        scala.io.Source.fromInputStream(in, "UTF-8").getLines().filter(l => l.nonEmpty && !l.startsWith("#")).map {
          line =>
            val f = line.split('\t')
            f(0) -> Coverage(f(1).toLong, f(2).toLong, if (f.length > 3) f(3).toLong else 0L)
        }.toMap
      } finally in.close()
  }

  private def write(path: Path, rows: Map[String, Coverage]): Unit = {
    val lines =
      Seq("# test case\texecutions\taccelerated executions\tspark-vector operators (VectorSQLQueryTestSuite, #17)") ++
        rows.toSeq.sortBy(_._1).map { case (name, c) => s"$name\t${c.executions}\t${c.accelerated}\t${c.operators}" }
    Files.createDirectories(path.getParent)
    Files.write(path, lines.mkString("", "\n", "\n").getBytes(StandardCharsets.UTF_8))
  }

  // scalastyle:off println
  private[sqltests] def report(isFullRun: Boolean): Unit = {
    val rows = VectorCoverageListener.perCase
    val total = VectorCoverageListener.executions.get()
    val accelerated = VectorCoverageListener.withVectorOperators.get()
    println(
      s"[spark-vector] SQL test coverage: $accelerated of $total executions ran at least one spark-vector operator; " +
        s"${VectorCoverageListener.vectorOperators.get()} spark-vector operators in total"
    )
    val (some, none) = rows.toSeq.sortBy(_._1).partition(_._2.accelerated > 0)
    println(
      s"[spark-vector] ${some.size} test cases ran a spark-vector operator, ${none.size} never did (no supported operator in any query, or analysis-only cases):"
    )
    some.foreach { case (name, c) => println(f"[spark-vector]   ${c.accelerated}%5d / ${c.executions}%-5d  $name") }
    if (none.nonEmpty) println("[spark-vector] never accelerated: " + none.map(_._1).mkString(", "))
    try write(runOutput, rows)
    catch { case e: Exception => println(s"[spark-vector] could not write $runOutput: $e") }

    if (isFullRun && rows.nonEmpty) {
      if (java.lang.Boolean.getBoolean("spark.vector.sqlTests.updateBaseline")) {
        write(baselineSource, rows)
        println(s"[spark-vector] baseline rewritten: $baselineSource")
      } else {
        val baseline = readBaseline()
        val regressions = baseline.toSeq.sortBy(_._1).flatMap { case (name, was) =>
          rows.get(name) match {
            case Some(now) if now.accelerated < was.accelerated =>
              Some(s"$name: ${was.accelerated} -> ${now.accelerated} accelerated executions")
            case None if was.accelerated > 0 => Some(s"$name: ${was.accelerated} -> (case missing from the run)")
            case _ => None
          }
        }
        val improvements = rows.toSeq.sortBy(_._1).filter { case (name, now) =>
          now.accelerated > baseline.get(name).map(_.accelerated).getOrElse(0L)
        }
        if (improvements.nonEmpty)
          println(s"[spark-vector] ${improvements.size} cases above the baseline (rerun with -DsqlTests.updateBaseline=true to record them): " +
            improvements.map { case (n, c) =>
              s"$n ${baseline.get(n).map(_.accelerated).getOrElse(0L)} -> ${c.accelerated}"
            }.mkString(", "))
        if (regressions.nonEmpty) {
          throw new AssertionError(
            s"[spark-vector] ${regressions.size} test cases run fewer spark-vector operators than the baseline $BaselineResource -- " +
              "a fallback introduced by a planner change, or a golden file that changed:\n  " + regressions.mkString(
                "\n  "
              )
          )
        }
        if (baseline.isEmpty) println(
          s"[spark-vector] no baseline $BaselineResource on the classpath; run with -DsqlTests.updateBaseline=true to create it"
        )
      }
    }
  }
  // scalastyle:on println
}

/** Counts, per test case and over the whole suite, how often a spark-vector operator ended up in an executed plan. */
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
      VectorCoverageListener.recordForCase(ours)
    } catch {
      case _: Throwable => // never let bookkeeping fail a query
    }
  }
}

object VectorCoverageListener {
  val executions = new AtomicLong()
  val withVectorOperators = new AtomicLong()
  val vectorOperators = new AtomicLong()

  /** The test case whose queries are running; queries outside any case (fixture setup) go to "(setup)". */
  @volatile private var current: String = "(setup)"
  private val cases = new ConcurrentHashMap[String, Array[AtomicLong]]()

  def beginCase(name: String): Unit = current = name
  def endCase(): Unit = current = "(setup)"

  private[sqltests] def recordForCase(ours: Int): Unit = {
    val slots = cases.computeIfAbsent(current, _ => Array(new AtomicLong(), new AtomicLong(), new AtomicLong()))
    slots(0).incrementAndGet()
    if (ours > 0) { slots(1).incrementAndGet(); slots(2).addAndGet(ours) }
  }

  def perCase: Map[String, VectorSQLQueryTestSuite.Coverage] =
    cases.asScala.filter(_._1 != "(setup)").map { case (name, s) =>
      name -> VectorSQLQueryTestSuite.Coverage(s(0).get(), s(1).get(), s(2).get())
    }.toMap

  private[sqltests] def lower(s: String): String = s.toLowerCase(Locale.ROOT)
}
