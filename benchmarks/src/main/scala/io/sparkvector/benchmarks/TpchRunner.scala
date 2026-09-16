package io.sparkvector.benchmarks

import java.io.{File, PrintWriter}
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import scala.jdk.CollectionConverters._
import scala.io.Source

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.vector.ui.{Engine, PlanAcceleration}

/**
 * TPC-H runner: the 22 queries over the tables `gen-tpch.sh` writes. One configuration per JVM
 * (plugins are loaded at SparkContext start), results appended as JSON lines; `--report` aggregates
 * every result file into markdown and HTML tables of medians plus a per-query acceleration column
 * (operators executed by our kernels or Comet versus operators left to Spark, the classification
 * the UI's Vector Acceleration tab uses).
 *
 * {{{
 * TpchRunner --config vector --data benchmarks/data/sf1 --iterations 5 --warmup 2 --out benchmarks/results
 * TpchRunner --config vector --data benchmarks/data/sf1 --queries q1,q6
 * TpchRunner --report benchmarks/results
 * }}}
 *
 * The measurement, JSON and report code is shared with [[TpcdsRunner]] through [[Suite]]: a runner
 * is a `main` that names its suite and calls [[mainWith]].
 */
object TpchRunner {

  /**
   * A benchmark: its tables (each a Parquet directory of that name under `--data`), the table whose
   * row count labels a dataset in the reports, and its queries in report order.
   */
  final case class Suite(name: String, title: String, tables: Seq[String], anchorTable: String, queries: Seq[(String, String)]) {
    lazy val queryMap: Map[String, String] = queries.toMap
    def queryOrder: Seq[String] = queries.map(_._1)
  }

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

  /**
   * Our plugin as benchmarked: fast floating point (lane-parallel and interleaved double sums), the
   * counterpart of Comet's default `spark.comet.exec.strictFloatingPoint=false`. The plugin's own
   * default is strict (Spark's rounding); Q15 compares a double sum for equality against a maximum
   * of the same sums computed by Spark and returns no rows in fast mode, so its checksum differs.
   */
  val VectorFast: Map[String, String] = Map(
    "spark.plugins" -> "io.sparkvector.spark.VectorPlugin",
    "spark.vector.exec.strictFloatingPoint" -> "false")

  /** Spark configurations under comparison. Comet configs need the Comet jar on the classpath. */
  val Configs: Map[String, Map[String, String]] = Map(
    "spark" -> Map.empty,
    "vector" -> VectorFast,
    "comet-scan" -> (Map("spark.plugins" -> "org.apache.spark.CometPlugin") ++ CometScanOnly),
    "comet-scan-vector" -> (VectorFast ++ Map("spark.plugins" -> "org.apache.spark.CometPlugin,io.sparkvector.spark.VectorPlugin") ++ CometScanOnly),
    // Comet scan and Comet native shuffle, everything in between (and the Final aggregate) ours.
    "comet-scan-vector-shuffle" -> (VectorFast ++ Map(
      "spark.plugins" -> "org.apache.spark.CometPlugin,io.sparkvector.spark.VectorPlugin",
      "spark.shuffle.manager" -> "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager") ++
      CometScanOnly ++ Map("spark.comet.exec.shuffle.enabled" -> "true")),
    "comet" -> Map(
      "spark.plugins" -> "org.apache.spark.CometPlugin",
      "spark.comet.enabled" -> "true",
      "spark.comet.scan.enabled" -> "true",
      "spark.comet.exec.enabled" -> "true",
      "spark.comet.exec.shuffle.enabled" -> "true",
      "spark.shuffle.manager" -> "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager",
      "spark.memory.offHeap.enabled" -> "true",
      "spark.memory.offHeap.size" -> "3g"))

  val ConfigOrder: Seq[String] = Seq("spark", "vector", "comet-scan", "comet-scan-vector", "comet-scan-vector-shuffle", "comet")

  /** The TPC-H tables, each a Parquet directory of that name under `--data` (see `gen-tpch.sh`). */
  val Tables: Seq[String] = Seq("customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier")

  val Queries: Map[String, String] = TpchQueries.All.toMap
  val QueryOrder: Seq[String] = TpchQueries.All.map(_._1)

  val Tpch: Suite = Suite("tpch", "TPC-H", Tables, "lineitem", TpchQueries.All)

  /** The tables a query reads, from its text: every table name is a distinct word (underscores included). */
  def tablesOf(suite: Suite, sql: String): Set[String] = {
    val words = sql.toLowerCase.split("[^a-z_]+").toSet
    suite.tables.filter(words.contains).toSet
  }

  final case class Args(
      suite: Suite = Tpch,
      config: String = "vector",
      data: String = "benchmarks/data/sf1",
      queries: Seq[String] = Nil,
      iterations: Int = 5,
      warmup: Int = 2,
      threads: Int = Runtime.getRuntime.availableProcessors(),
      shufflePartitions: Int = 8,
      out: String = "benchmarks/results",
      report: Option[String] = None,
      label: String = "",
      show: Boolean = false,
      keepAlive: Boolean = false,
      extraConf: Map[String, String] = Map.empty)

  def main(argv: Array[String]): Unit = mainWith(Tpch, argv)

  /** Entry point shared by the runners: `--report <dir>` rewrites the reports, anything else measures. */
  def mainWith(suite: Suite, argv: Array[String]): Unit = {
    val args = parse(argv.toList, Args(suite = suite))
    args.report match {
      case Some(dir) => report(suite, Paths.get(dir))
      case None => run(if (args.queries.isEmpty) args.copy(queries = suite.queryOrder) else args)
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
    case "--show" :: rest => parse(rest, a.copy(show = true))
    case "--keep-alive" :: rest => parse(rest, a.copy(keepAlive = true))
    case "--conf" :: kv :: rest =>
      val Array(k, v) = kv.split("=", 2)
      parse(rest, a.copy(extraConf = a.extraConf + (k -> v)))
    case other :: _ => throw new IllegalArgumentException(s"unknown argument $other")
  }

  private def run(args: Args): Unit = {
    val conf = Configs.getOrElse(args.config, throw new IllegalArgumentException(s"unknown config ${args.config}; known: ${ConfigOrder.mkString(", ")}"))
    val builder = SparkSession.builder()
      .master(s"local[${args.threads}]")
      .appName(s"spark-vector-${args.suite.name}-${args.config}")
      .config("spark.ui.enabled", args.keepAlive.toString)
      .config("spark.sql.shuffle.partitions", args.shufflePartitions.toString)
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.driver.host", "localhost")
    (conf ++ args.extraConf).foreach { case (k, v) => builder.config(k, v) }
    val spark = builder.getOrCreate()
    try {
      // Register every table the data directory holds; a directory generated by an older
      // gen-tpch.sh has lineitem alone and can still run Q1/Q6.
      val suite = args.suite
      val present = suite.tables.filter(t => new File(args.data, t).isDirectory)
      require(present.contains(suite.anchorTable), s"${args.data} holds no ${suite.anchorTable}/ directory (see gen-${suite.name}.sh)")
      present.foreach(t => spark.read.parquet(new File(args.data, t).getPath).createOrReplaceTempView(t))
      val rowCount = spark.table(suite.anchorTable).count()
      println(s"[${suite.name}] config=${args.config} data=${args.data} tables=${present.mkString(",")} ${suite.anchorTable} rows=$rowCount threads=${args.threads}")

      Files.createDirectories(Paths.get(args.out))
      val outFile = Paths.get(args.out, s"${args.config}${if (args.label.isEmpty) "" else "-" + args.label}.jsonl")
      val writer = new PrintWriter(Files.newBufferedWriter(outFile, java.nio.charset.StandardCharsets.UTF_8,
        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND))
      val failed = scala.collection.mutable.ArrayBuffer.empty[String]
      try {
        args.queries.foreach { q =>
          val sql = suite.queryMap.getOrElse(q, throw new IllegalArgumentException(s"unknown query $q"))
          val missing = tablesOf(suite, sql) -- present.toSet
          if (missing.nonEmpty) {
            println(s"[${suite.name}] skipping $q: ${missing.toSeq.sorted.mkString(", ")} not generated (rerun gen-${suite.name}.sh)")
          } else {
            // A query that fails (an analysis error against the generated schema, an engine bug) is
            // reported and leaves no record; the rest of the suite still runs.
            try {
              val result = measure(spark, q, sql, args)
              writer.println(result.toJson(args.config, args.data, rowCount))
              writer.flush()
              println(s"[${args.suite.name}] ${args.config} $q median=${result.medianMs}ms p90=${result.p90Ms}ms min=${result.minMs}ms rows=${result.rows} " +
                s"accelerated=${result.acceleratedOps}/${result.operatorCount} operators=${result.operators}")
              result.fallbacks.foreach(f => println(s"[${args.suite.name}]   fallback: $f"))
            } catch {
              case e: Exception =>
                failed += q
                println(s"[${suite.name}] ${args.config} $q FAILED: ${e.getClass.getSimpleName}: ${Option(e.getMessage).getOrElse("").linesIterator.take(3).mkString(" ")}")
            }
          }
        }
      } finally writer.close()
      if (failed.nonEmpty) println(s"[${suite.name}] ${failed.size} quer${if (failed.size == 1) "y" else "ies"} failed: ${failed.mkString(", ")}")
      if (args.keepAlive) {
        // For inspecting the Spark UI (the Vector Acceleration tab) after the queries ran.
        println(s"[${args.suite.name}] keeping the session open; Spark UI at ${spark.sparkContext.uiWebUrl.getOrElse("(disabled)")}. Ctrl-C to exit.")
        Thread.currentThread().join()
      }
    } finally spark.stop()
  }

  /**
   * One query's runs. `acceleratedOps` / `operatorCount` is the plan's share of operators executed
   * by our kernels or by Comet, counted the way the UI does (plumbing such as scans and
   * row/columnar transitions is neither): `PlanAcceleration.fromPlan`.
   */
  final case class Measurement(
      query: String, timesMs: Seq[Double], rows: Int, checksum: String, operators: String, plan: String,
      acceleratedOps: Int, operatorCount: Int,
      /** `Operator: reason` for every operator the planner rule tried to convert and could not. */
      fallbacks: Seq[String]) {
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
      s"""{"timestamp":"${Instant.now()}","config":"$config","query":"$query","data":"${esc(data)}","anchorRows":$rowCount,""" +
        s""""medianMs":$medianMs,"p90Ms":$p90Ms,"minMs":$minMs,"timesMs":[${timesMs.map(t => math.round(t * 10) / 10.0).mkString(",")}],""" +
        s""""rows":$rows,"checksum":"$checksum","acceleratedOps":$acceleratedOps,"operatorCount":$operatorCount,""" +
        s""""fallbacks":"${esc(fallbacks.mkString("; "))}","operators":"${esc(operators)}","plan":"${esc(plan)}"}"""
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
    if (args.show) rows.foreach(r => println(s"[${args.suite.name}]   row: ${r.mkString(" | ")}"))
    val checksum = rows.map(_.toSeq.map {
      // 10 significant digits: summation order differs between engines (and our interleaved
      // accumulators), which moves the last few bits of a 1e11 sum but nothing a user sees.
      case d: Double => f"$d%.10g"
      case v => String.valueOf(v)
    }.mkString("|")).sorted.mkString("\n").hashCode.toHexString
    val nodes = allNodes(plan)
    val ops = nodes.map(_.getClass.getSimpleName).filter(n => n.startsWith("Vector") || n.startsWith("Comet"))
      .groupBy(identity).view.mapValues(_.size).toSeq.sortBy(_._1).map { case (n, c) => s"$n x$c" }.mkString(", ")
    // Per-operator kernel time of the last run (summed over tasks, so it exceeds wall clock).
    nodes.filter(n => n.getClass.getSimpleName.startsWith("Vector") && n.metrics.contains("time")).foreach { n =>
      val t = n.metrics.get("time").map(m => f"${m.value / 1e6}%.1f ms").getOrElse("-")
      val r = n.metrics.get("numOutputRows").map(_.value).getOrElse(-1L)
      println(s"[${args.suite.name}]   ${n.getClass.getSimpleName}: kernel time $t, output rows $r")
    }
    val accelerated = PlanAcceleration.fromPlan(plan)
    val acceleratedOps = accelerated.nodes.count(n => !Engine.plumbing.contains(n.engine) && n.engine.isAccelerated)
    // One line per distinct (operator, reason): the same reason repeats across AQE stages.
    val fallbacks = accelerated.fallbacks.map { case (node, reason) => s"$node: $reason" }.distinct
    Measurement(name, runs.map(_._1), rows.length, checksum, if (ops.isEmpty) "spark only" else ops, plan.treeString.take(4000),
      acceleratedOps, accelerated.operatorCount, fallbacks)
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

  private final case class Row(
      timestamp: String,
      config: String,
      query: String,
      data: String,
      medianMs: Double,
      p90Ms: Double,
      minMs: Double,
      timesMs: Seq[Double],
      rows: Int,
      checksum: String,
      operators: String,
      /** Rows of the suite's anchor table (lineitem, store_sales) in the dataset. */
      anchorRows: Long,
      /** Accelerated / counted operators; None for records written before the column existed. */
      accelerated: Option[(Int, Int)],
      /** `Operator: reason` lines; empty for older records and for fully accelerated plans. */
      fallbacks: Seq[String]) {
    /** Dataset label: the last path element (`sf1`, `sf10`). */
    def dataset: String = data.stripSuffix("/").split('/').last
    /** `5/7` -- operators executed by our kernels or Comet over operators that count. */
    def acceleratedCell: String = accelerated.map { case (a, t) => s"$a/$t" }.getOrElse("-")
    def fullyAccelerated: Boolean = accelerated.exists { case (a, t) => t > 0 && a == t }
  }

  /** The suite's order (`q1`..`q22`) rather than lexical, with anything else after. */
  private def queryOrder(suite: Suite)(q: String): (Int, String) =
    (suite.queryOrder.indexOf(q) match { case -1 => Int.MaxValue; case i => i }, q)

  /** Everything the report needs about one dataset. */
  private final case class DatasetReport(name: String, anchorRows: Long, configs: Seq[String], queries: Seq[String], latest: Map[(String, String), Row]) {
    def speedup(c: String, q: String): Option[Double] =
      for (r <- latest.get((c, q)); base <- latest.get(("spark", q)) if r.medianMs > 0) yield base.medianMs / r.medianMs
    def mismatches: Seq[String] = queries.filter(q => configs.flatMap(c => latest.get((c, q)).map(_.checksum)).distinct.size > 1)
    /** Configurations other than plain Spark: the ones an acceleration column says something about. */
    def acceleratedConfigs: Seq[String] = configs.filter(_ != "spark")
    /** Queries every accelerated configuration runs entirely on our kernels or Comet. */
    def fullyAccelerated(c: String): Int = queries.count(q => latest.get((c, q)).exists(_.fullyAccelerated))
  }

  private def report(suite: Suite, dir: Path): Unit = {
    val files = Option(dir.toFile.listFiles()).getOrElse(Array.empty[File]).filter(_.getName.endsWith(".jsonl")).sorted
    val rows = files.flatMap { f =>
      Source.fromFile(f, "UTF-8").getLines().filter(_.nonEmpty).map(parseRow).toSeq
    }
    // Keep the latest measurement per (dataset, config, query); order datasets by size.
    val datasets = rows.groupBy(_.dataset).toSeq.sortBy(_._2.head.anchorRows).map { case (name, rs) =>
      val latest = rs.sortBy(_.timestamp).groupBy(r => (r.config, r.query)).view.mapValues(_.last).toMap
      val configs = ConfigOrder.filter(c => latest.keys.exists(_._1 == c))
      DatasetReport(name, rs.head.anchorRows, configs, latest.keys.map(_._2).toSeq.distinct.sortBy(queryOrder(suite)), latest)
    }
    val md = markdown(suite, datasets)
    Files.writeString(dir.resolve("results.md"), md)
    Files.writeString(dir.resolve("results.html"), html(suite, datasets))
    println(md)
    println(s"HTML report: ${dir.resolve("results.html")}")
  }

  private def markdown(suite: Suite, datasets: Seq[DatasetReport]): String = {
    val sb = new StringBuilder
    sb.append(s"# ${suite.title} results\n\n")
    sb.append("Median wall-clock time per query in milliseconds (speedup versus plain Spark in parentheses).\n")
    datasets.foreach { d =>
      sb.append(s"\n## ${d.name} (${suite.anchorTable} rows: ${d.anchorRows})\n\n")
      sb.append("| query | " + d.configs.mkString(" | ") + " |\n")
      sb.append("|---|" + d.configs.map(_ => "---:").mkString("|") + "|\n")
      d.queries.foreach { q =>
        val cells = d.configs.map { c =>
          d.latest.get((c, q)) match {
            case Some(r) => f"${r.medianMs}%.1f" + d.speedup(c, q).map(x => f" ($x%.2fx)").getOrElse("")
            case None => "-"
          }
        }
        sb.append(s"| $q | " + cells.mkString(" | ") + " |\n")
      }
      if (d.acceleratedConfigs.nonEmpty) {
        sb.append("\nAccelerated operators per query (operators run by our kernels or Comet / operators that count; scans and row/columnar transitions are neither):\n\n")
        sb.append("| query | " + d.acceleratedConfigs.mkString(" | ") + " |\n")
        sb.append("|---|" + d.acceleratedConfigs.map(_ => "---:").mkString("|") + "|\n")
        d.queries.foreach { q =>
          sb.append(s"| $q | " + d.acceleratedConfigs.map(c => d.latest.get((c, q)).map(_.acceleratedCell).getOrElse("-")).mkString(" | ") + " |\n")
        }
        sb.append("| fully accelerated | " + d.acceleratedConfigs.map(c => s"${d.fullyAccelerated(c)}/${d.queries.size}").mkString(" | ") + " |\n")
      }
      sb.append("\nOperators in the final plan:\n\n")
      d.queries.foreach { q =>
        d.configs.foreach { c =>
          d.latest.get((c, q)).foreach { r =>
            sb.append(s"- $q / $c: ${r.operators} (rows=${r.rows}, checksum=${r.checksum})\n")
            r.fallbacks.foreach(f => sb.append(s"  - not accelerated: $f\n"))
          }
        }
      }
      sb.append("\n")
      sb.append(if (d.mismatches.isEmpty) "All configurations returned identical results (to 10 significant digits).\n"
      else s"WARNING: result checksums differ for ${d.mismatches.mkString(", ")}\n")
    }
    sb.toString
  }

  private val ConfigColors: Map[String, String] = Map(
    "spark" -> "#8a8f98",
    "vector" -> "#2f6fdb",
    "comet-scan" -> "#c48a1a",
    "comet-scan-vector" -> "#3b9e5a",
    "comet-scan-vector-shuffle" -> "#1f7a5c",
    "comet" -> "#b3452e")

  private def esc(s: String): String = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

  /** Self-contained HTML: tables, inline SVG bar charts, per-run distributions, plans. */
  private def html(suite: Suite, datasets: Seq[DatasetReport]): String = {
    val sb = new StringBuilder
    sb.append(s"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>spark-vector ${suite.title} results</title>
<style>
 body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1d2430;line-height:1.45}
 h1{font-weight:600} h2{margin-top:2.5rem;border-bottom:1px solid #dde3ea;padding-bottom:.3rem} h3{margin-top:1.8rem}
 table{border-collapse:collapse;margin:1rem 0;font-variant-numeric:tabular-nums} th,td{padding:.35rem .7rem;border-bottom:1px solid #e6eaf0;text-align:right} th:first-child,td:first-child{text-align:left}
 thead th{background:#f4f6f9;font-weight:600} td.best{font-weight:600;color:#1f7a5c} td.worse{color:#b3452e}
 .speed{color:#5b6472;font-size:.85em}
 .legend span{display:inline-block;margin-right:1rem;font-size:.9em} .swatch{display:inline-block;width:.8em;height:.8em;border-radius:2px;margin-right:.35em;vertical-align:-1px}
 svg text{font-size:12px;fill:#1d2430} .muted{color:#5b6472}
 details{margin:.6rem 0} summary{cursor:pointer;color:#2f6fdb} pre{background:#f4f6f9;padding:.6rem .8rem;overflow:auto;font-size:.8em}
 .ok{color:#1f7a5c;font-weight:600} .warn{color:#b3452e;font-weight:600}
</style></head><body>
<h1>spark-vector: ${suite.title}</h1>
<p class="muted">Median wall-clock time of <code>collect()</code> per query, one JVM per configuration, after warm-up.
Bars are medians; the whisker marks p90. Speedups are relative to plain Spark on the same dataset.</p>
""")
    sb.append("<p class=\"legend\">")
    ConfigOrder.filter(c => datasets.exists(_.configs.contains(c))).foreach { c =>
      sb.append(s"""<span><i class="swatch" style="background:${ConfigColors.getOrElse(c, "#999")}"></i>${esc(c)}</span>""")
    }
    sb.append("</p>\n")
    sb.append(s"<p class=\"muted\">Generated ${Instant.now()} from ${datasets.map(_.latest.size).sum} measurements.</p>\n")

    datasets.foreach { d =>
      sb.append(s"<h2>${esc(d.name)} <span class=\"muted\">(${"%,d".format(d.anchorRows)} ${esc(suite.anchorTable)} rows)</span></h2>\n")
      // Summary table
      sb.append("<table><thead><tr><th>query</th>" + d.configs.map(c => s"<th>${esc(c)}</th>").mkString + "</tr></thead><tbody>\n")
      d.queries.foreach { q =>
        val best = d.configs.flatMap(c => d.latest.get((c, q))).map(_.medianMs).minOption
        sb.append(s"<tr><td>${esc(q)}</td>")
        d.configs.foreach { c =>
          d.latest.get((c, q)) match {
            case Some(r) =>
              val cls = if (best.contains(r.medianMs)) " class=\"best\"" else if (d.speedup(c, q).exists(_ < 0.98)) " class=\"worse\"" else ""
              val speed = d.speedup(c, q).map(x => f"<br><span class=\"speed\">$x%.2fx</span>").getOrElse("")
              sb.append(f"<td$cls>${r.medianMs}%.1f ms$speed</td>")
            case None => sb.append("<td>-</td>")
          }
        }
        sb.append("</tr>\n")
      }
      sb.append("</tbody></table>\n")
      sb.append(if (d.mismatches.isEmpty) "<p class=\"ok\">All configurations returned identical results (to 10 significant digits).</p>\n"
      else s"<p class=\"warn\">Result checksums differ for ${esc(d.mismatches.mkString(", "))}.</p>\n")
      if (d.acceleratedConfigs.nonEmpty) {
        sb.append("<h3>Accelerated operators</h3>\n<p class=\"muted\">Operators executed by our kernels or Comet over the operators that count (scans and row/columnar transitions are neither), from the final plan of the last run.</p>\n")
        sb.append("<table><thead><tr><th>query</th>" + d.acceleratedConfigs.map(c => s"<th>${esc(c)}</th>").mkString + "</tr></thead><tbody>\n")
        d.queries.foreach { q =>
          sb.append(s"<tr><td>${esc(q)}</td>")
          d.acceleratedConfigs.foreach { c =>
            d.latest.get((c, q)) match {
              case Some(r) => sb.append(s"<td${if (r.fullyAccelerated) " class=\"best\"" else ""}>${r.acceleratedCell}</td>")
              case None => sb.append("<td>-</td>")
            }
          }
          sb.append("</tr>\n")
        }
        sb.append("<tr><td>fully accelerated</td>" + d.acceleratedConfigs.map(c => s"<td>${d.fullyAccelerated(c)}/${d.queries.size}</td>").mkString + "</tr>\n")
        sb.append("</tbody></table>\n")
      }

      // Bar chart per query
      d.queries.foreach { q =>
        val entries = d.configs.flatMap(c => d.latest.get((c, q)).map(c -> _))
        val maxMs = entries.map(_._2.p90Ms).maxOption.getOrElse(1.0) * 1.05
        val rowH = 28
        val labelW = 190
        val chartW = 700
        val h = entries.size * rowH + 30
        sb.append(s"<h3>${esc(q)}</h3>\n")
        sb.append(s"""<svg width="${labelW + chartW + 120}" height="$h" role="img" aria-label="Median time per configuration for ${esc(q)}">""")
        entries.zipWithIndex.foreach { case ((c, r), i) =>
          val y = i * rowH + 6
          val w = math.max(2.0, r.medianMs / maxMs * chartW)
          val p90x = labelW + r.p90Ms / maxMs * chartW
          val color = ConfigColors.getOrElse(c, "#999")
          sb.append(s"""<text x="${labelW - 8}" y="${y + 15}" text-anchor="end">${esc(c)}</text>""")
          sb.append(f"""<rect x="$labelW" y="$y" width="$w%.1f" height="${rowH - 10}" fill="$color" rx="2"/>""")
          sb.append(f"""<line x1="$p90x%.1f" x2="$p90x%.1f" y1="${y + 2}" y2="${y + rowH - 12}" stroke="#1d2430" stroke-width="1.5"/>""")
          val speed = d.speedup(c, q).map(x => f" ($x%.2fx)").getOrElse("")
          val labelX = math.max(labelW + w, p90x) + 8
          sb.append(f"""<text x="$labelX%.1f" y="${y + 15}">${r.medianMs}%.1f ms$speed</text>""")
        }
        // axis
        val axisY = entries.size * rowH + 12
        sb.append(s"""<line x1="$labelW" x2="${labelW + chartW}" y1="$axisY" y2="$axisY" stroke="#c7cdd6"/>""")
        Seq(0.0, 0.25, 0.5, 0.75, 1.0).foreach { f =>
          val x = labelW + f * chartW
          sb.append(f"""<text x="$x%.1f" y="${axisY + 14}" text-anchor="middle" class="muted">${f * maxMs}%.0f</text>""")
        }
        sb.append("</svg>\n")
        // per-run details
        sb.append("<details><summary>runs, plans and checksums</summary>\n<table><thead><tr><th>config</th><th>min</th><th>median</th><th>p90</th><th>runs (ms)</th><th>rows</th><th>checksum</th></tr></thead><tbody>\n")
        entries.foreach { case (c, r) =>
          sb.append(f"<tr><td>${esc(c)}</td><td>${r.minMs}%.1f</td><td>${r.medianMs}%.1f</td><td>${r.p90Ms}%.1f</td><td style=\"text-align:left\">${r.timesMs.map(t => f"$t%.0f").mkString(" ")}</td><td>${r.rows}</td><td>${r.checksum}</td></tr>\n")
        }
        sb.append("</tbody></table>\n")
        entries.foreach { case (c, r) =>
          sb.append(s"<p><b>${esc(c)}</b>: ${esc(r.operators)}</p>\n")
          if (r.fallbacks.nonEmpty) sb.append("<ul>" + r.fallbacks.map(f => s"<li>not accelerated: ${esc(f)}</li>").mkString + "</ul>\n")
        }
        sb.append("</details>\n")
      }
    }
    sb.append("</body></html>\n")
    sb.toString
  }

  /** Minimal JSON field extraction for the flat records this runner writes. */
  private def parseRow(line: String): Row = {
    // A scan rather than a regex: Java's regex engine recurses per character on an alternation
    // repeated over a long value, and a TPC-DS plan field (4000 characters) overflows the stack.
    def str(k: String): String = {
      val start = line.indexOf("\"" + k + "\":\"")
      if (start < 0) "" else {
        val sb = new StringBuilder
        var i = start + k.length + 4
        var done = false
        while (!done && i < line.length) {
          line.charAt(i) match {
            case '\\' if i + 1 < line.length =>
              line.charAt(i + 1) match {
                case 'n' => sb.append('\n')
                case c => sb.append(c)
              }
              i += 2
            case '"' => done = true
            case c => sb.append(c); i += 1
          }
        }
        sb.toString
      }
    }
    def num(k: String): Double = ("\"" + k + "\":([-0-9.E]+)").r.findFirstMatchIn(line).map(_.group(1).toDouble).getOrElse(0.0)
    val times = ("\"timesMs\":\\[([^\\]]*)\\]").r.findFirstMatchIn(line).map(_.group(1)).getOrElse("")
      .split(",").map(_.trim).filter(_.nonEmpty).map(_.toDouble).toSeq
    def optNum(k: String): Option[Int] = ("\"" + k + "\":([0-9]+)").r.findFirstMatchIn(line).map(_.group(1).toInt)
    val accelerated = for (a <- optNum("acceleratedOps"); t <- optNum("operatorCount")) yield (a, t)
    val fallbacks = str("fallbacks").split("; ").map(_.trim).filter(_.nonEmpty).toSeq
    // `lineitemRows` is the name records written before the TPC-DS runner used for the anchor table.
    val anchorRows = "\"anchorRows\":([0-9]+)".r.findFirstMatchIn(line).map(_.group(1).toLong).getOrElse(num("lineitemRows").toLong)
    Row(str("timestamp"), str("config"), str("query"), str("data"), num("medianMs"), num("p90Ms"), num("minMs"), times,
      num("rows").toInt, str("checksum"), str("operators"), anchorRows, accelerated, fallbacks)
  }
}

