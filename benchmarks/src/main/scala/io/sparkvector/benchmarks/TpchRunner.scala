package io.sparkvector.benchmarks

import java.io.{File, PrintWriter}
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import scala.jdk.CollectionConverters._
import scala.io.Source

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}

/**
 * TPC-H Q1/Q6 runner. One configuration per JVM (plugins are loaded at SparkContext start), results
 * appended as JSON lines; `--report` aggregates every result file into a markdown table.
 *
 * {{{
 * TpchRunner --config vector --data benchmarks/data/sf1 --iterations 5 --warmup 2 --out benchmarks/results
 * TpchRunner --report benchmarks/results
 * }}}
 */
object TpchRunner {

  /** Comet with only its native Parquet scan active; every Comet operator and its shuffle are off. */
  val CometScanOnly: Map[String, String] = Map(
      "spark.comet.enabled" -> "true",
      "spark.comet.scan.enabled" -> "true",
      // Comet 1.0's only scan is the native DataFusion one, which requires exec to be enabled; keep
      // every Comet operator off so the scan is the only native piece and Spark's shuffle is used.
      "spark.comet.exec.enabled" -> "true",
      "spark.comet.exec.shuffle.enabled" -> "false",
      "spark.comet.exec.project.enabled" -> "false",
      "spark.comet.exec.filter.enabled" -> "false",
      "spark.comet.exec.aggregate.enabled" -> "false",
      "spark.comet.exec.sort.enabled" -> "false",
      "spark.comet.exec.localLimit.enabled" -> "false",
      "spark.comet.exec.globalLimit.enabled" -> "false",
      "spark.comet.exec.takeOrderedAndProject.enabled" -> "false",
      "spark.comet.exec.hashJoin.enabled" -> "false",
      "spark.comet.exec.sortMergeJoin.enabled" -> "false",
      "spark.comet.exec.broadcastHashJoin.enabled" -> "false",
      "spark.comet.exec.broadcastExchange.enabled" -> "false",
      "spark.comet.exec.expand.enabled" -> "false",
      "spark.comet.exec.union.enabled" -> "false",
      "spark.comet.exec.window.enabled" -> "false",
      "spark.comet.exec.coalesce.enabled" -> "false",
      "spark.comet.exec.collectLimit.enabled" -> "false",
      "spark.comet.exec.explode.enabled" -> "false",
      "spark.comet.exec.sample.enabled" -> "false",
      "spark.memory.offHeap.enabled" -> "true",
      "spark.memory.offHeap.size" -> "1g")

  /** Spark configurations under comparison. Comet configs need the Comet jar on the classpath. */
  val Configs: Map[String, Map[String, String]] = Map(
    "spark" -> Map.empty,
    "vector" -> Map(
      "spark.plugins" -> "io.sparkvector.spark.VectorPlugin"),
    "comet-scan" -> (Map("spark.plugins" -> "org.apache.spark.CometPlugin") ++ CometScanOnly),
    "comet-scan-vector" -> (Map("spark.plugins" -> "org.apache.spark.CometPlugin,io.sparkvector.spark.VectorPlugin") ++ CometScanOnly),
    "comet" -> Map(
      "spark.plugins" -> "org.apache.spark.CometPlugin",
      "spark.comet.enabled" -> "true",
      "spark.comet.scan.enabled" -> "true",
      "spark.comet.exec.enabled" -> "true",
      "spark.comet.exec.shuffle.enabled" -> "true",
      "spark.shuffle.manager" -> "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager",
      "spark.memory.offHeap.enabled" -> "true",
      "spark.memory.offHeap.size" -> "3g"))

  val ConfigOrder: Seq[String] = Seq("spark", "vector", "comet-scan", "comet-scan-vector", "comet")

  val Queries: Map[String, String] = Map("q1" -> TpchQueries.Q1, "q6" -> TpchQueries.Q6)

  final case class Args(
      config: String = "vector",
      data: String = "benchmarks/data/sf1",
      queries: Seq[String] = Seq("q1", "q6"),
      iterations: Int = 5,
      warmup: Int = 2,
      threads: Int = Runtime.getRuntime.availableProcessors(),
      shufflePartitions: Int = 8,
      out: String = "benchmarks/results",
      report: Option[String] = None,
      label: String = "")

  def main(argv: Array[String]): Unit = {
    val args = parse(argv.toList, Args())
    args.report match {
      case Some(dir) => report(Paths.get(dir))
      case None => run(args)
    }
  }

  private def parse(argv: List[String], a: Args): Args = argv match {
    case Nil => a
    case "--config" :: v :: rest => parse(rest, a.copy(config = v))
    case "--data" :: v :: rest => parse(rest, a.copy(data = v))
    case "--queries" :: v :: rest => parse(rest, a.copy(queries = v.split(",").map(_.trim.toLowerCase).toSeq))
    case "--iterations" :: v :: rest => parse(rest, a.copy(iterations = v.toInt))
    case "--warmup" :: v :: rest => parse(rest, a.copy(warmup = v.toInt))
    case "--threads" :: v :: rest => parse(rest, a.copy(threads = v.toInt))
    case "--shuffle-partitions" :: v :: rest => parse(rest, a.copy(shufflePartitions = v.toInt))
    case "--out" :: v :: rest => parse(rest, a.copy(out = v))
    case "--label" :: v :: rest => parse(rest, a.copy(label = v))
    case "--report" :: v :: rest => parse(rest, a.copy(report = Some(v)))
    case other :: _ => throw new IllegalArgumentException(s"unknown argument $other")
  }

  private def run(args: Args): Unit = {
    val conf = Configs.getOrElse(args.config, throw new IllegalArgumentException(s"unknown config ${args.config}; known: ${ConfigOrder.mkString(", ")}"))
    val builder = SparkSession.builder()
      .master(s"local[${args.threads}]")
      .appName(s"spark-vector-tpch-${args.config}")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", args.shufflePartitions.toString)
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.driver.host", "localhost")
    conf.foreach { case (k, v) => builder.config(k, v) }
    val spark = builder.getOrCreate()
    try {
      val lineitem = new File(args.data, "lineitem").getPath
      spark.read.parquet(lineitem).createOrReplaceTempView("lineitem")
      val rowCount = spark.table("lineitem").count()
      println(s"[tpch] config=${args.config} data=${args.data} lineitem rows=$rowCount threads=${args.threads}")

      Files.createDirectories(Paths.get(args.out))
      val outFile = Paths.get(args.out, s"${args.config}${if (args.label.isEmpty) "" else "-" + args.label}.jsonl")
      val writer = new PrintWriter(Files.newBufferedWriter(outFile, java.nio.charset.StandardCharsets.UTF_8,
        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND))
      try {
        args.queries.foreach { q =>
          val sql = Queries.getOrElse(q, throw new IllegalArgumentException(s"unknown query $q"))
          val result = measure(spark, q, sql, args)
          writer.println(result.toJson(args.config, args.data, rowCount))
          writer.flush()
          println(s"[tpch] ${args.config} $q median=${result.medianMs}ms p90=${result.p90Ms}ms min=${result.minMs}ms rows=${result.rows} operators=${result.operators}")
        }
      } finally writer.close()
    } finally spark.stop()
  }

  final case class Measurement(query: String, timesMs: Seq[Double], rows: Int, checksum: String, operators: String, plan: String) {
    private val sorted = timesMs.sorted
    def medianMs: Double = percentile(50)
    def p90Ms: Double = percentile(90)
    def minMs: Double = sorted.head
    private def percentile(p: Int): Double = {
      val idx = math.min(sorted.size - 1, math.max(0, math.ceil(p / 100.0 * sorted.size).toInt - 1))
      math.round(sorted(idx) * 10) / 10.0
    }
    def toJson(config: String, data: String, rowCount: Long): String = {
      def esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
      s"""{"timestamp":"${Instant.now()}","config":"$config","query":"$query","data":"${esc(data)}","lineitemRows":$rowCount,""" +
        s""""medianMs":$medianMs,"p90Ms":$p90Ms,"minMs":$minMs,"timesMs":[${timesMs.map(t => math.round(t * 10) / 10.0).mkString(",")}],""" +
        s""""rows":$rows,"checksum":"$checksum","operators":"${esc(operators)}","plan":"${esc(plan)}"}"""
    }
  }

  private def measure(spark: SparkSession, name: String, sql: String, args: Args): Measurement = {
    def once(): (Double, Array[org.apache.spark.sql.Row], SparkPlan) = {
      val df = spark.sql(sql)
      val start = System.nanoTime()
      val rows = df.collect()
      val ms = (System.nanoTime() - start) / 1e6
      (ms, rows, df.queryExecution.executedPlan)
    }
    (1 to args.warmup).foreach(_ => once())
    val runs = (1 to args.iterations).map(_ => once())
    val (_, rows, plan) = runs.last
    val checksum = rows.map(_.toSeq.map {
      case d: Double => f"$d%.4f"
      case v => String.valueOf(v)
    }.mkString("|")).sorted.mkString("\n").hashCode.toHexString
    val ops = allNodes(plan).map(_.getClass.getSimpleName).filter(n => n.startsWith("Vector") || n.startsWith("Comet"))
      .groupBy(identity).view.mapValues(_.size).toSeq.sortBy(_._1).map { case (n, c) => s"$n x$c" }.mkString(", ")
    Measurement(name, runs.map(_._1), rows.length, checksum, if (ops.isEmpty) "spark only" else ops, plan.treeString.take(4000))
  }

  private def allNodes(plan: SparkPlan): Seq[SparkPlan] = {
    val inner = plan match {
      case a: AdaptiveSparkPlanExec => Seq(a.executedPlan)
      case q: QueryStageExec => Seq(q.plan)
      case _ => Nil
    }
    plan +: (plan.children ++ inner).flatMap(allNodes)
  }

  // ------------------------------------------------------------------ report

  private final case class Row(config: String, query: String, medianMs: Double, p90Ms: Double, rows: Int, checksum: String, operators: String, lineitemRows: Long)

  private def report(dir: Path): Unit = {
    val files = Option(dir.toFile.listFiles()).getOrElse(Array.empty[File]).filter(_.getName.endsWith(".jsonl")).sorted
    val rows = files.flatMap { f =>
      Source.fromFile(f, "UTF-8").getLines().filter(_.nonEmpty).map(parseRow).toSeq
    }
    // Keep the latest measurement per (config, query).
    val latest = rows.groupBy(r => (r.config, r.query)).view.mapValues(_.last).toMap
    val configs = ConfigOrder.filter(c => latest.keys.exists(_._1 == c))
    val queries = latest.keys.map(_._2).toSeq.distinct.sorted
    val sb = new StringBuilder
    sb.append("# TPC-H results\n\n")
    latest.values.headOption.foreach(r => sb.append(s"lineitem rows: ${r.lineitemRows}\n\n"))
    sb.append("Median wall-clock time per query in milliseconds (speedup versus plain Spark in parentheses).\n\n")
    sb.append("| query | " + configs.mkString(" | ") + " |\n")
    sb.append("|---|" + configs.map(_ => "---:").mkString("|") + "|\n")
    queries.foreach { q =>
      val base = latest.get(("spark", q)).map(_.medianMs)
      val cells = configs.map { c =>
        latest.get((c, q)) match {
          case Some(r) =>
            val speedup = base.filter(_ > 0).map(b => f" (${b / r.medianMs}%.2fx)").getOrElse("")
            f"${r.medianMs}%.1f$speedup"
          case None => "-"
        }
      }
      sb.append(s"| $q | " + cells.mkString(" | ") + " |\n")
    }
    sb.append("\n## Operators in the final plan\n\n")
    queries.foreach { q =>
      configs.foreach { c =>
        latest.get((c, q)).foreach(r => sb.append(s"- $q / $c: ${r.operators} (rows=${r.rows}, checksum=${r.checksum})\n"))
      }
    }
    val mismatches = queries.flatMap { q =>
      val sums = configs.flatMap(c => latest.get((c, q)).map(_.checksum)).distinct
      if (sums.size > 1) Some(q) else None
    }
    sb.append("\n")
    sb.append(if (mismatches.isEmpty) "All configurations returned identical results (to 4 decimals).\n"
    else s"WARNING: result checksums differ for ${mismatches.mkString(", ")}\n")
    val md = sb.toString
    Files.writeString(dir.resolve("results.md"), md)
    println(md)
  }

  /** Minimal JSON field extraction for the flat records this runner writes. */
  private def parseRow(line: String): Row = {
    def str(k: String): String = {
      val m = ("\"" + k + "\":\"((?:[^\"\\\\]|\\\\.)*)\"").r.findFirstMatchIn(line)
      m.map(_.group(1).replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\")).getOrElse("")
    }
    def num(k: String): Double = ("\"" + k + "\":([-0-9.E]+)").r.findFirstMatchIn(line).map(_.group(1).toDouble).getOrElse(0.0)
    Row(str("config"), str("query"), num("medianMs"), num("p90Ms"), num("rows").toInt, str("checksum"), str("operators"), num("lineitemRows").toLong)
  }
}

object TpchQueries {
  val Q1: String =
    """SELECT l_returnflag, l_linestatus,
      |  sum(l_quantity) AS sum_qty,
      |  sum(l_extendedprice) AS sum_base_price,
      |  sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
      |  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
      |  avg(l_quantity) AS avg_qty,
      |  avg(l_extendedprice) AS avg_price,
      |  avg(l_discount) AS avg_disc,
      |  count(*) AS count_order
      |FROM lineitem
      |WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL 90 DAY
      |GROUP BY l_returnflag, l_linestatus
      |ORDER BY l_returnflag, l_linestatus""".stripMargin

  val Q6: String =
    """SELECT sum(l_extendedprice * l_discount) AS revenue
      |FROM lineitem
      |WHERE l_shipdate >= DATE '1994-01-01'
      |  AND l_shipdate < DATE '1995-01-01'
      |  AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
      |  AND l_quantity < 24""".stripMargin
}
