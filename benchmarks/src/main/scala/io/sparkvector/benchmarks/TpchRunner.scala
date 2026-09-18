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
import io.sparkvector.spark.iceberg.IcebergVectorAdapter

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
  final case class Suite(name: String, title: String, tables: Seq[String], anchorTable: String, queries: Seq[(String, String)],
      /** Runnable by name but not part of a default run (the MoR probes). */
      probes: Seq[(String, String)] = Nil) {
    lazy val queryMap: Map[String, String] = (queries ++ probes).toMap
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
   * The opt-in sort-merge join rewrite is on: Comet accelerates those joins too (natively, as
   * merge joins), and the rewrite is our only way to.
   */
  val VectorFast: Map[String, String] = Map(
    "spark.plugins" -> "io.sparkvector.spark.VectorPlugin",
    "spark.vector.exec.strictFloatingPoint" -> "false",
    "spark.vector.exec.sortMergeJoin.mode" -> "auto") // #287: the merge join where the order can show or statistics are missing, the hash rewrite otherwise

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

  val Tpch: Suite = Suite("tpch", "TPC-H", Tables, "lineitem", TpchQueries.All, TpchQueries.Probes)

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
      extraConf: Map[String, String] = Map.empty,
      /** Cluster mode (#246): take the session spark-submit built, tables from `--tables`, rows to any Hadoop file system. */
      cluster: Boolean = false,
      /** `s3://bucket/tpcds/sf1000/parquet` (a directory per table) or `catalog:<namespace>` (Iceberg tables). */
      tables: Option[String] = None,
      /** Dataset label in cluster mode (`sf1000-parquet`); defaults to the last path element of `--tables`. */
      dataset: Option[String] = None,
      /** Query texts from `<dir>/<name>.sql` instead of the classpath (a cluster image without the tests jar). */
      queriesDir: Option[String] = None,
      /** The data-on-EKS-style report over every `.jsonl` under a directory (local or any Hadoop file system). */
      clusterReport: Option[String] = None,
      /** Local Iceberg merge-on-read harness (#260): the Hadoop catalog warehouse `gen-iceberg-mor.sh` wrote ... */
      icebergWarehouse: Option[String] = None,
      /** ... and the `<namespace>.<variant>` table in it that stands in for `lineitem`; the dataset label is `iceberg:<namespace>.<variant>`. */
      icebergVariant: Option[String] = None)

  def main(argv: Array[String]): Unit = mainWith(Tpch, argv)

  /** Entry point shared by the runners: `--report <dir>` rewrites the reports, anything else measures. */
  def mainWith(suite: Suite, argv: Array[String]): Unit = {
    val args = parse(argv.toList, Args(suite = suite))
    (args.report, args.clusterReport) match {
      case (Some(dir), _) => report(suite, Paths.get(dir))
      case (None, Some(dir)) => clusterReport(suite, dir)
      case _ => run(if (args.queries.isEmpty) args.copy(queries = suite.queryOrder) else args)
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
    case "--cluster" :: rest => parse(rest, a.copy(cluster = true))
    case "--tables" :: v :: rest => parse(rest, a.copy(tables = Some(v), cluster = true))
    case "--dataset" :: v :: rest => parse(rest, a.copy(dataset = Some(v)))
    case "--queries-dir" :: v :: rest => parse(rest, a.copy(queriesDir = Some(v)))
    case "--cluster-report" :: v :: rest => parse(rest, a.copy(clusterReport = Some(v)))
    case "--iceberg" :: v :: rest => parse(rest, a.copy(icebergWarehouse = Some(v)))
    case "--variant" :: v :: rest => parse(rest, a.copy(icebergVariant = Some(v)))
    case "--conf" :: kv :: rest =>
      val Array(k, v) = kv.split("=", 2)
      parse(rest, a.copy(extraConf = a.extraConf + (k -> v)))
    case other :: _ => throw new IllegalArgumentException(s"unknown argument $other")
  }

  private def run(args: Args): Unit = {
    val conf = Configs.getOrElse(args.config, throw new IllegalArgumentException(s"unknown config ${args.config}; known: ${ConfigOrder.mkString(", ")}"))
    val spark =
      if (args.cluster) {
        // spark-submit built the context: master, plugins, shuffle manager and memory come from the
        // application's configuration (the SparkApplication manifest), not from Configs -- a plugin
        // cannot be added to a running context. --config only labels the rows; disagreements are printed.
        val s = SparkSession.builder().getOrCreate()
        (conf ++ args.extraConf).foreach { case (k, v) =>
          val actual = s.conf.getOption(k)
          if (!actual.contains(v)) println(s"[${args.suite.name}] WARNING: ${args.config} expects $k=$v, the session has ${actual.getOrElse("(unset)")}")
        }
        s
      } else {
        val builder = SparkSession.builder()
          .master(s"local[${args.threads}]")
          .appName(s"spark-vector-${args.suite.name}-${args.config}")
          .config("spark.ui.enabled", args.keepAlive.toString)
          .config("spark.sql.shuffle.partitions", args.shufflePartitions.toString)
          .config("spark.sql.adaptive.enabled", "true")
          .config("spark.driver.host", "localhost")
        (conf ++ args.extraConf).foreach { case (k, v) => builder.config(k, v) }
        // The Iceberg MoR harness: the generator's Hadoop catalog on this session (the plugin comes
        // through spark.plugins, so Iceberg's SQL extensions do not displace it).
        args.icebergWarehouse.foreach(w => IcebergMorGenerator.catalogConf(new File(w).getAbsolutePath).foreach { case (k, v) => builder.config(k, v) })
        builder.getOrCreate()
      }
    val listener = new ClusterRunner.StageMetricsListener
    spark.sparkContext.addSparkListener(listener)
    try {
      // Register every table the data directory holds; a directory generated by an older
      // gen-tpch.sh has lineitem alone and can still run Q1/Q6.
      val suite = args.suite
      val source = args.tables.getOrElse(args.data)
      val present =
        if (args.cluster) ClusterRunner.registerTables(spark, suite.tables, source)
        else {
          val p = suite.tables.filter(t => new File(args.data, t).isDirectory)
          p.foreach(t => spark.read.parquet(new File(args.data, t).getPath).createOrReplaceTempView(t))
          p
        }
      require(present.contains(suite.anchorTable), s"$source holds no ${suite.anchorTable} table (see gen-${suite.name}.sh)")
      // The Iceberg MoR harness: one generated variant stands in for lineitem; the other tables stay Parquet.
      val variant = args.icebergVariant.map { v =>
        require(args.icebergWarehouse.isDefined, "--variant needs --iceberg <warehouse>")
        require(suite.anchorTable == "lineitem", "--variant is a TPC-H lineitem table")
        val table = s"${IcebergMorGenerator.Catalog}.$v"
        require(spark.catalog.tableExists(table), s"$table does not exist in ${args.icebergWarehouse.get} (see gen-iceberg-mor.sh)")
        spark.table(table).createOrReplaceTempView("lineitem")
        val snapshot = spark.sql(s"SELECT snapshot_id FROM $table.snapshots ORDER BY committed_at DESC LIMIT 1").collect()(0).getLong(0)
        println(s"[${suite.name}] lineitem is the Iceberg table $table at snapshot $snapshot")
        v
      }
      val rowCount = spark.table(suite.anchorTable).count()
      // The dataset label the reports group by: the last path element (`sf1`, `sf10`), or --dataset.
      val data = args.dataset.getOrElse(variant.map(v => s"iceberg:$v").getOrElse(if (args.cluster) source.stripSuffix("/").split('/').last else args.data))
      // Query texts: the classpath (the tests jar) or, on a cluster image without it, `<dir>/<name>.sql`.
      val queryMap = args.queriesDir.map(d => ClusterRunner.queriesFrom(spark, d, args.queries).toMap).getOrElse(suite.queryMap)
      val env = ClusterRunner.Environment.of(spark)
      println(s"[${suite.name}] config=${args.config} data=$data tables=${present.mkString(",")} ${suite.anchorTable} rows=$rowCount " +
        (if (args.cluster) s"spark=${env.sparkVersion} executors=${env.executors}" else s"threads=${args.threads}"))

      val writer =
        if (args.cluster) ClusterRunner.openOutput(spark, args.out, args.config, args.label)
        else {
          Files.createDirectories(Paths.get(args.out))
          val outFile = Paths.get(args.out, s"${args.config}${if (args.label.isEmpty) "" else "-" + args.label}.jsonl")
          new PrintWriter(Files.newBufferedWriter(outFile, java.nio.charset.StandardCharsets.UTF_8,
            java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND))
        }
      val failed = scala.collection.mutable.ArrayBuffer.empty[String]
      try {
        args.queries.foreach { q =>
          val sql = queryMap.getOrElse(q, throw new IllegalArgumentException(s"unknown query $q"))
          val missing = tablesOf(suite, sql) -- present.toSet
          if (missing.nonEmpty) {
            println(s"[${suite.name}] skipping $q: ${missing.toSeq.sorted.mkString(", ")} not generated (rerun gen-${suite.name}.sh)")
          } else {
            // A query that fails (an analysis error against the generated schema, an engine bug) is
            // reported and leaves no record; the rest of the suite still runs.
            try {
              val result = measure(spark, listener, q, sql, args)
              writer.println(result.toJson(args.config, data, rowCount, env))
              writer.flush()
              println(s"[${args.suite.name}] ${args.config} $q median=${result.medianMs}ms p90=${result.p90Ms}ms min=${result.minMs}ms rows=${result.rows} " +
                s"accelerated=${result.acceleratedOps}/${result.operatorCount} operators=${result.operators}")
              if (result.scan.nonEmpty || result.morPhysicalRows > 0)
                println(s"[${args.suite.name}]   scan=${result.scan} " +
                  (if (result.morPhysicalRows > 0) f"merge-on-read live/physical=${result.morLiveRows}/${result.morPhysicalRows} (${100.0 * result.morLiveRows / result.morPhysicalRows}%.1f%% live)" else "merge-on-read batches=0"))
              result.metrics.foreach(m => println(s"[${args.suite.name}]   stages=${m.stages} executorRunTime=${m.executorRunTimeMs}ms gc=${m.jvmGcTimeMs}ms " +
                s"shuffleRead=${m.shuffleReadBytes} shuffleWrite=${m.shuffleWriteBytes} spill=${m.spillBytes} peakMemory=${m.peakExecutionMemory}"))
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
      fallbacks: Seq[String],
      /** Spark's stage-level metrics of the last measured run (cluster runs need them to explain a regression without a rerun). */
      metrics: Option[ClusterRunner.StageMetrics] = None,
      /** The scan operators of the plan (`BatchScanExec`, `CometIcebergNativeScanExec`, ...), distinct. */
      scan: String = "",
      /**
       * Iceberg merge-on-read batches the adapter normalized during the last measured run, as rows read
       * (physical) and rows the deletes left (live) -- local mode only, the counters live in the executor JVM.
       */
      morPhysicalRows: Long = 0, morLiveRows: Long = 0,
      /** Per-operator attribution of the last measured run (#279): ours and Comet's native operators, one entry per plan node. */
      operatorTimes: Seq[OperatorTime] = Nil) {
    private val sorted = timesMs.sorted
    def medianMs: Double = percentile(50)
    def p90Ms: Double = percentile(90)
    def minMs: Double = sorted.head
    private def percentile(p: Int): Double = {
      val idx = math.min(sorted.size - 1, math.max(0, math.ceil(p / 100.0 * sorted.size).toInt - 1))
      math.round(sorted(idx) * 10) / 10.0
    }
    def toJson(config: String, data: String, rowCount: Long, env: ClusterRunner.Environment = ClusterRunner.Environment.Unknown): String = {
      def esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
      s"""{"timestamp":"${Instant.now()}","config":"$config","query":"$query","data":"${esc(data)}","anchorRows":$rowCount,""" +
        s""""medianMs":$medianMs,"p90Ms":$p90Ms,"minMs":$minMs,"timesMs":[${timesMs.map(t => math.round(t * 10) / 10.0).mkString(",")}],""" +
        s""""rows":$rows,"checksum":"$checksum","acceleratedOps":$acceleratedOps,"operatorCount":$operatorCount,""" +
        metrics.map(m => m.json + ",").getOrElse("") +
        s""""sparkVersion":"${esc(env.sparkVersion)}","executors":"${esc(env.executors)}","engineConf":"${esc(env.engineConf)}",""" +
        s""""scan":"${esc(scan)}","morPhysicalRows":$morPhysicalRows,"morLiveRows":$morLiveRows,""" +
        s""""operatorTimes":"${esc(OperatorTime.encode(operatorTimes))}",""" +
        s""""fallbacks":"${esc(fallbacks.mkString("; "))}","operators":"${esc(operators)}","plan":"${esc(plan)}"}"""
    }
  }

  /**
   * One operator's share of a run (#279): the plan node's class, whose engine ran it (`ours`, `comet`),
   * its output rows and its time in milliseconds summed over tasks (so it exceeds wall clock). Ours
   * report the `time` metric in nanoseconds; Comet's native operators report DataFusion's
   * `elapsed_compute` (the operator's own compute, input waits excluded, in nanoseconds) and
   * `output_rows`. Spark's own operators carry no per-operator time and are not listed.
   */
  final case class OperatorTime(operator: String, engine: String, rows: Long, ms: Double) {
    /** `Filter`, `HashAggregate`, `ShuffledHashJoin`: the operator kind without the engine's prefix and the `Exec` suffix. */
    def kind: String = OperatorTime.kindOf(operator)
  }

  object OperatorTime {
    def kindOf(operator: String): String =
      operator.stripPrefix("Vector").stripPrefix("Comet").stripSuffix("Exec")
    /** `op|engine|rows|ms; op|engine|rows|ms` -- one string field, read back by [[decode]]. */
    def encode(ts: Seq[OperatorTime]): String = ts.map(t => f"${t.operator}|${t.engine}|${t.rows}|${t.ms}%.3f").mkString("; ")
    def decode(field: String): Seq[OperatorTime] = field.split("; ").map(_.trim).filter(_.nonEmpty).toSeq.flatMap { e =>
      e.split('|') match {
        case Array(op, engine, rows, ms) => scala.util.Try(OperatorTime(op, engine, rows.toLong, ms.toDouble)).toOption
        case _ => None
      }
    }
    /** Attribution from the executed plan: ours from `time`, Comet's native nodes from `elapsed_compute`. */
    def fromPlan(nodes: Seq[SparkPlan]): Seq[OperatorTime] = nodes.flatMap { n =>
      val name = n.getClass.getSimpleName
      if (name.startsWith("Vector") && n.metrics.contains("time")) {
        Some(OperatorTime(name, "ours", n.metrics.get("numOutputRows").map(_.value).getOrElse(-1L), n.metrics("time").value / 1e6))
      } else if (name.startsWith("Comet") && !name.contains("Scan") && !name.contains("ColumnarToRow") && !name.contains("Exchange")) {
        // Measured: the values are nanoseconds (a 21 ms filter reads 21200470), whatever the metric's
        // description says. An operator without `elapsed_compute` (a join) has its phases as `*_time`.
        val ns = n.metrics.get("elapsed_compute").map(_.value).orElse {
          val phases = n.metrics.collect { case (k, m) if k.endsWith("_time") => m.value }
          if (phases.isEmpty) None else Some(phases.sum)
        }
        ns match {
          case Some(v) => Some(OperatorTime(name, "comet", n.metrics.get("output_rows").map(_.value).getOrElse(-1L), v / 1e6))
          case None =>
            if (unattributed.add(name)) println(s"[attribution] $name has no time metric; its metrics: ${n.metrics.keys.toSeq.sorted.mkString(", ")}")
            None
        }
      } else None
    }
    private val unattributed = scala.collection.mutable.HashSet.empty[String]
  }

  private def measure(spark: SparkSession, listener: ClusterRunner.StageMetricsListener, name: String, sql: String, args: Args): Measurement = {
    // Each run under its own job group, so the listener attributes its stages to it.
    def once(i: Int): (Double, Array[org.apache.spark.sql.Row], SparkPlan, ClusterRunner.StageMetrics) = {
      val df = spark.sql(sql)
      val ((ms, rows), metrics) = ClusterRunner.measured(spark, listener, s"$name-$i") {
        val start = System.nanoTime()
        val rows = df.collect()
        ((System.nanoTime() - start) / 1e6, rows)
      }
      (ms, rows, df.queryExecution.executedPlan, metrics)
    }
    (1 to args.warmup).foreach(i => once(-i))
    // The adapter's merge-on-read counters around the last measured run (a delta: the JVM is shared by every query).
    val runs = (1 to args.iterations - 1).map(once)
    val (physicalBefore, liveBefore) = (IcebergVectorAdapter.normalizedPhysicalRows(), IcebergVectorAdapter.normalizedLiveRows())
    val last = once(args.iterations)
    val morPhysical = IcebergVectorAdapter.normalizedPhysicalRows() - physicalBefore
    val morLive = IcebergVectorAdapter.normalizedLiveRows() - liveBefore
    val (_, rows, plan, metrics) = last
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
    val operatorTimes = OperatorTime.fromPlan(nodes)
    operatorTimes.foreach(t => println(f"[${args.suite.name}]   ${t.operator} (${t.engine}): time ${t.ms}%.1f ms, output rows ${t.rows}"))
    val accelerated = PlanAcceleration.fromPlan(plan)
    val acceleratedOps = accelerated.nodes.count(n => !Engine.plumbing.contains(n.engine) && n.engine.isAccelerated)
    // One line per distinct (operator, reason): the same reason repeats across AQE stages.
    val fallbacks = (accelerated.fallbacks.map { case (node, reason) => s"$node: $reason" } ++ cometFallbacks(plan)).distinct
    val scan = nodes.map(_.getClass.getSimpleName).filter(_.endsWith("ScanExec")).distinct.sorted.mkString(", ")
    Measurement(name, (runs :+ last).map(_._1), rows.length, checksum, if (ops.isEmpty) "spark only" else ops, plan.treeString.take(4000),
      acceleratedOps, accelerated.operatorCount, fallbacks, Some(metrics), scan, morPhysical, morLive, operatorTimes)
  }

  /**
   * Comet's own fallback reasons for the plan (#279), `Comet: <reason>` each, read through its
   * `ExtendedExplainInfo` when the jar is on the classpath -- so a query where Comet is "slower" because
   * it left an operator to Spark reads as a fallback, not as an operator comparison. Empty without Comet.
   */
  private def cometFallbacks(plan: SparkPlan): Seq[String] =
    scala.util.Try {
      val info = Class.forName("org.apache.comet.ExtendedExplainInfo").getDeclaredConstructor().newInstance()
      info.getClass.getMethod("getFallbackReasons", classOf[SparkPlan]).invoke(info, plan).asInstanceOf[Seq[String]]
    }.toOption.getOrElse(Nil).map(r => s"Comet: ${r.replace('\n', ' ').trim}").filter(_.length > 7).distinct

  private def allNodes(plan: SparkPlan): Seq[SparkPlan] = {
    val inner = plan match {
      case a: AdaptiveSparkPlanExec => Seq(a.executedPlan)
      case q: QueryStageExec => Seq(q.plan)
      case _ => Nil
    }
    plan +: (plan.children ++ inner).flatMap(allNodes)
  }

  // ------------------------------------------------------------------ report

  /**
   * The data-on-EKS-style report over every `.jsonl` row under `dir` (local or `s3://...`): reads
   * through a session (spark-submit's, or a `local[1]` one when none is configured) so the object
   * store's file system is the one the cluster used, and writes `cluster-results.md` beside the rows.
   */
  private def clusterReport(suite: Suite, dir: String): Unit = {
    val builder = SparkSession.builder().appName(s"spark-vector-${suite.name}-report")
    if (sys.props.get("spark.master").isEmpty) builder.master("local[1]").config("spark.driver.host", "localhost").config("spark.ui.enabled", "false")
    val spark = builder.getOrCreate()
    try {
      val rows = ClusterRunner.readRows(spark, dir).map(parseRow).sortBy(_.timestamp).map { r =>
        ClusterRunner.ReportRow(r.config, r.query, r.dataset, r.medianMs, r.rows, r.checksum, r.accelerated, r.fallbacks,
          r.metrics, r.sparkVersion, r.executors, r.engineConf)
      }
      val md = ClusterRunner.report(suite.title, suite.queryOrder, rows)
      ClusterRunner.write(spark, s"${dir.stripSuffix("/")}/cluster-results.md", md)
      println(md)
      println(s"[${suite.name}] report: ${dir.stripSuffix("/")}/cluster-results.md (${rows.size} rows)")
    } finally spark.stop()
  }

  private[benchmarks] final case class Row(
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
      fallbacks: Seq[String],
      /** Stage-level metrics and the environment, written by runs since #246; absent in older records. */
      metrics: Option[ClusterRunner.StageMetrics] = None,
      sparkVersion: String = "",
      executors: String = "",
      engineConf: String = "",
      /** Scan operators of the plan and the Iceberg adapter's merge-on-read rows (#260); absent in older records. */
      scan: String = "",
      morPhysicalRows: Long = 0,
      morLiveRows: Long = 0,
      /** Per-operator attribution (#279); empty for older records. */
      operatorTimes: Seq[OperatorTime] = Nil) {
    /** Dataset label: the last path element (`sf1`, `sf10`). */
    def dataset: String = data.stripSuffix("/").split('/').last
    /** `5/7` -- operators executed by our kernels or Comet over operators that count. */
    def acceleratedCell: String = accelerated.map { case (a, t) => s"$a/$t" }.getOrElse("-")
    def fullyAccelerated: Boolean = accelerated.exists { case (a, t) => t > 0 && a == t }
    /** `scan=BatchScanExec, merge-on-read live/physical=...`, empty for records without the fields. */
    def scanCell: String =
      if (scan.isEmpty) "" else s"scan=$scan" + (if (morPhysicalRows > 0) f", merge-on-read live/physical=$morLiveRows/$morPhysicalRows (${100.0 * morLiveRows / morPhysicalRows}%.1f%% live)" else "")
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
    /** Configurations whose latest rows carry per-operator times (#279). */
    def attributedConfigs: Seq[String] = configs.filter(c => queries.exists(q => latest.get((c, q)).exists(_.operatorTimes.nonEmpty)))
    /** Milliseconds per operator kind in one query under one configuration, summed over the plan's nodes of that kind. */
    def kindMs(c: String, q: String): Map[String, Double] =
      latest.get((c, q)).map(_.operatorTimes.groupBy(_.kind).view.mapValues(_.map(_.ms).sum).toMap).getOrElse(Map.empty)
    /** Every operator kind any attributed configuration ran, ordered by its total time descending. */
    def kinds: Seq[String] = {
      val totals = for (c <- attributedConfigs; q <- queries; (k, ms) <- kindMs(c, q).toSeq) yield (k, ms)
      totals.groupBy(_._1).view.mapValues(_.map(_._2).sum).toSeq.sortBy(-_._2).map(_._1)
    }
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
      if (d.attributedConfigs.size >= 1 && d.kinds.nonEmpty) {
        sb.append("\nOperator matrix (#279): milliseconds per operator kind, summed over the plan's nodes of that kind and over tasks, " +
          "for the last measured run; a kind an engine did not run is `-` -- read it beside the fallbacks below, since a kind left to Spark has no time here.\n\n")
        sb.append("| operator kind | " + d.attributedConfigs.map(c => s"$c (all queries)").mkString(" | ") + " |\n")
        sb.append("|---|" + d.attributedConfigs.map(_ => "---:").mkString("|") + "|\n")
        d.kinds.foreach { k =>
          val cells = d.attributedConfigs.map { c =>
            val per = d.queries.flatMap(q => d.kindMs(c, q).get(k))
            if (per.isEmpty) "-" else f"${per.sum}%.0f (${per.size} queries)"
          }
          sb.append(s"| $k | " + cells.mkString(" | ") + " |\n")
        }
        if (d.attributedConfigs.size >= 2) {
          sb.append("\nPer query, the kinds both of the first two attributed configurations ran, `ours vs theirs (delta)` in milliseconds:\n\n")
          val Seq(a, b) = d.attributedConfigs.take(2)
          d.queries.foreach { q =>
            val (ma, mb) = (d.kindMs(a, q), d.kindMs(b, q))
            val shared = d.kinds.filter(k => ma.contains(k) && mb.contains(k))
            if (shared.nonEmpty) sb.append(s"- $q: " + shared.map(k => f"$k ${ma(k)}%.1f vs ${mb(k)}%.1f (${ma(k) - mb(k)}%+.1f)").mkString("; ") + "\n")
          }
        }
      }
      sb.append("\nOperators in the final plan:\n\n")
      d.queries.foreach { q =>
        d.configs.foreach { c =>
          d.latest.get((c, q)).foreach { r =>
            sb.append(s"- $q / $c: ${r.operators} (rows=${r.rows}, checksum=${r.checksum})\n")
            if (r.scanCell.nonEmpty) sb.append(s"  - ${r.scanCell}\n")
            r.fallbacks.foreach(f => sb.append(s"  - not accelerated: $f\n"))
          }
        }
      }
      sb.append("\n")
      sb.append(if (d.mismatches.isEmpty) "All configurations returned identical results (to 10 significant digits).\n"
      else s"WARNING: result checksums differ for ${d.mismatches.mkString(", ")}\n")
    }
    morSections(datasets).foreach { m =>
      sb.append(s"\n## Iceberg merge-on-read: `${m.namespace}` (${suite.anchorTable} variants of `gen-iceberg-mor.sh`)\n\n")
      sb.append("Median milliseconds per variant and configuration; in parentheses the speedup versus `spark` on the same variant, " +
        "then versus the same configuration on the `plain` table (what the deletes cost that engine: below 1x is slower than plain). " +
        "Live/physical is the share of the rows read that the deletes left, as the adapter saw it.\n")
      m.queries.foreach { q =>
        sb.append(s"\n### $q\n\n")
        sb.append("| variant | live rows | " + m.configs.mkString(" | ") + " | live/physical |\n")
        sb.append("|---|---:|" + m.configs.map(_ => "---:").mkString("|") + "|---:|\n")
        m.variants.foreach { v =>
          val cells = m.configs.map(c => m.cell(v, c, q).getOrElse("-"))
          sb.append(s"| `$v` | ${m.liveRows(v)} | " + cells.mkString(" | ") + s" | ${m.liveRatio(v, q).getOrElse("-")} |\n")
        }
      }
    }
    sb.toString
  }

  /** One Iceberg merge-on-read namespace of the report: its variants (datasets `iceberg:<ns>.<variant>`), configurations and queries. */
  private final case class MorSection(namespace: String, variants: Seq[String], configs: Seq[String], queries: Seq[String], byVariant: Map[String, DatasetReport]) {
    def liveRows(v: String): Long = byVariant(v).anchorRows
    def cell(v: String, c: String, q: String): Option[String] = byVariant(v).latest.get((c, q)).map { r =>
      val vsSpark = byVariant(v).speedup(c, q).map(x => f" ($x%.2fx").getOrElse(" (-")
      val vsPlain = byVariant.get("plain").flatMap(_.latest.get((c, q))).filter(_ => v != "plain" && r.medianMs > 0).map(p => f", ${p.medianMs / r.medianMs}%.2fx vs plain)").getOrElse(")")
      f"${r.medianMs}%.1f" + vsSpark + vsPlain
    }
    /** The merge ratio of the first configuration that reports one (the adapter counts only under our plugin). */
    def liveRatio(v: String, q: String): Option[String] =
      configs.flatMap(c => byVariant(v).latest.get((c, q))).find(_.morPhysicalRows > 0).map(r => f"${100.0 * r.morLiveRows / r.morPhysicalRows}%.1f%%")
  }

  private def morSections(datasets: Seq[DatasetReport]): Seq[MorSection] = {
    val Named = """iceberg:([^.]+)\.(.+)""".r
    datasets.collect { case d @ DatasetReport(Named(ns, v), _, _, _, _) => (ns, v, d) }.groupBy(_._1).toSeq.sortBy(_._1).map { case (ns, entries) =>
      val byVariant = entries.map { case (_, v, d) => v -> d }.toMap
      // plain first, then by name.
      val variants = byVariant.keys.toSeq.sortBy(v => (if (v == "plain") 0 else 1, v))
      val configs = ConfigOrder.filter(c => entries.exists(_._3.configs.contains(c)))
      val queries = entries.flatMap(_._3.queries).distinct.sortBy(queryOrder(Tpch))
      MorSection(ns, variants, configs, queries, byVariant)
    }
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
    morSections(datasets).foreach { m =>
      sb.append(s"<h2>Iceberg merge-on-read: ${esc(m.namespace)}</h2>\n")
      sb.append("<p>Median milliseconds per variant and configuration (speedup versus <code>spark</code> on the same variant, then versus the same configuration on the <code>plain</code> table); live/physical is the share of the rows read that the deletes left.</p>\n")
      m.queries.foreach { q =>
        sb.append(s"<h3>${esc(q)}</h3>\n<table><thead><tr><th>variant</th><th>live rows</th>" + m.configs.map(c => s"<th>${esc(c)}</th>").mkString + "<th>live/physical</th></tr></thead><tbody>\n")
        m.variants.foreach { v =>
          sb.append(s"<tr><td><code>${esc(v)}</code></td><td>${m.liveRows(v)}</td>" + m.configs.map(c => s"<td>${esc(m.cell(v, c, q).getOrElse("-"))}</td>").mkString +
            s"<td>${esc(m.liveRatio(v, q).getOrElse("-"))}</td></tr>\n")
        }
        sb.append("</tbody></table>\n")
      }
    }
    sb.append("</body></html>\n")
    sb.toString
  }

  /** Minimal JSON field extraction for the flat records this runner writes. */
  private[benchmarks] def parseRow(line: String): Row = {
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
    def optLong(k: String): Option[Long] = ("\"" + k + "\":([0-9]+)").r.findFirstMatchIn(line).map(_.group(1).toLong)
    val metrics = optLong("stages").map { stages =>
      ClusterRunner.StageMetrics(stages.toInt, optLong("executorRunTimeMs").getOrElse(0L), optLong("gcTimeMs").getOrElse(0L),
        optLong("shuffleReadBytes").getOrElse(0L), optLong("shuffleWriteBytes").getOrElse(0L), optLong("spillBytes").getOrElse(0L),
        optLong("peakExecutionMemory").getOrElse(0L))
    }
    Row(str("timestamp"), str("config"), str("query"), str("data"), num("medianMs"), num("p90Ms"), num("minMs"), times,
      num("rows").toInt, str("checksum"), str("operators"), anchorRows, accelerated, fallbacks,
      metrics, str("sparkVersion"), str("executors"), str("engineConf"),
      str("scan"), optLong("morPhysicalRows").getOrElse(0L), optLong("morLiveRows").getOrElse(0L),
      OperatorTime.decode(str("operatorTimes")))
  }
}

