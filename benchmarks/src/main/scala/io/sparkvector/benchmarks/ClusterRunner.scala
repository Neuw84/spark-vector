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
package io.sparkvector.benchmarks

import java.net.URI
import java.nio.charset.StandardCharsets
import java.time.Instant
import scala.collection.mutable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path => HPath}

import org.apache.spark.scheduler.{SparkListener, SparkListenerJobStart, SparkListenerStageCompleted}
import org.apache.spark.sql.SparkSession

/**
 * The cluster half of the runners (#246): a `spark-submit`-able mode of [[TpchRunner]] /
 * [[TpcdsRunner]] that takes the session it is given, registers the tables from a base URI or a
 * catalog, records Spark's stage-level metrics per query, writes the `.jsonl` rows to any Hadoop
 * file system (`s3://...`), and a report in the layout of the data-on-EKS Comet benchmark so our
 * numbers read next to Comet's.
 *
 * {{{
 * spark-submit --class io.sparkvector.benchmarks.TpcdsRunner benchmarks.jar \
 *   --cluster --config vector --tables s3://bucket/tpcds/sf1000/parquet --dataset sf1000-parquet \
 *   --queries-dir s3://bucket/tpcds/queries --iterations 1 --warmup 0 --out s3://bucket/results/sf1000-parquet
 * TpcdsRunner --cluster-report s3://bucket/results/sf1000-parquet
 * }}}
 */
object ClusterRunner {

  // ------------------------------------------------------------------ stage metrics

  /**
   * Spark's stage-level metrics of one query: the sums over its stages of what
   * `SparkListenerStageCompleted` reports, the evidence the data-on-EKS analysis is built on
   * (GC time, shuffle read bytes) so a regression can be explained without a rerun.
   */
  final case class StageMetrics(
      stages: Int,
      executorRunTimeMs: Long,
      jvmGcTimeMs: Long,
      shuffleReadBytes: Long,
      shuffleWriteBytes: Long,
      spillBytes: Long,
      peakExecutionMemory: Long
  ) {
    def json: String =
      s""""stages":$stages,"executorRunTimeMs":$executorRunTimeMs,"gcTimeMs":$jvmGcTimeMs,"shuffleReadBytes":$shuffleReadBytes,""" +
        s""""shuffleWriteBytes":$shuffleWriteBytes,"spillBytes":$spillBytes,"peakExecutionMemory":$peakExecutionMemory"""
  }

  object StageMetrics {
    val Empty: StageMetrics = StageMetrics(0, 0, 0, 0, 0, 0, 0)
  }

  /**
   * Attributes completed stages to the query that ran them through the job group the runner sets
   * before each run (`SparkContext.setJobGroup`), and sums their task metrics. Stages of a job that
   * carries no group (Spark's own housekeeping) are ignored.
   */
  final class StageMetricsListener extends SparkListener {
    private val stageGroup = mutable.HashMap.empty[Int, String]
    private val byGroup = mutable.HashMap.empty[String, StageMetrics]

    override def onJobStart(jobStart: SparkListenerJobStart): Unit = synchronized {
      Option(jobStart.properties).flatMap(p => Option(p.getProperty("spark.jobGroup.id"))).foreach { group =>
        jobStart.stageIds.foreach(id => stageGroup(id) = group)
      }
    }

    override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = synchronized {
      val info = stageCompleted.stageInfo
      stageGroup.get(info.stageId).foreach { group =>
        val m = info.taskMetrics
        val add = StageMetrics(
          1,
          m.executorRunTime,
          m.jvmGCTime,
          m.shuffleReadMetrics.totalBytesRead,
          m.shuffleWriteMetrics.bytesWritten,
          m.memoryBytesSpilled + m.diskBytesSpilled,
          m.peakExecutionMemory
        )
        val prev = byGroup.getOrElse(group, StageMetrics.Empty)
        byGroup(group) = StageMetrics(
          prev.stages + add.stages,
          prev.executorRunTimeMs + add.executorRunTimeMs,
          prev.jvmGcTimeMs + add.jvmGcTimeMs,
          prev.shuffleReadBytes + add.shuffleReadBytes,
          prev.shuffleWriteBytes + add.shuffleWriteBytes,
          prev.spillBytes + add.spillBytes,
          math.max(prev.peakExecutionMemory, add.peakExecutionMemory)
        )
      }
    }

    /** The metrics of one job group, after the listener bus has delivered every event of it. */
    def metrics(group: String): StageMetrics = synchronized(byGroup.getOrElse(group, StageMetrics.Empty))
  }

  /** Runs `body` under a fresh job group and returns its stage metrics, waiting for the listener bus to drain. */
  def measured[T](spark: SparkSession, listener: StageMetricsListener, group: String)(body: => T): (T, StageMetrics) = {
    spark.sparkContext.setJobGroup(group, group, interruptOnCancel = false)
    try {
      val result = body
      // Stage-completed events are delivered asynchronously; drain before reading them.
      org.apache.spark.sql.vector.BenchmarkListenerSync.drain(spark)
      (result, listener.metrics(group))
    } finally spark.sparkContext.clearJobGroup()
  }

  // ------------------------------------------------------------------ tables and queries

  /**
   * Where the tables come from: `s3://bucket/tpcds/sf1000/parquet` (each table a directory of that
   * name under it, any Hadoop file system), or `catalog:<namespace>` for catalog tables
   * (`<namespace>.<table>`, the Iceberg datasets). Returns the tables it found.
   */
  def registerTables(spark: SparkSession, tables: Seq[String], source: String): Seq[String] =
    if (source.startsWith("catalog:")) {
      val ns = source.stripPrefix("catalog:")
      tables.filter { t =>
        val exists = spark.catalog.tableExists(s"$ns.$t")
        if (exists) spark.table(s"$ns.$t").createOrReplaceTempView(t)
        exists
      }
    } else {
      val base = source.stripSuffix("/")
      val fs = fileSystem(spark, base)
      tables.filter { t =>
        val dir = new HPath(s"$base/$t")
        val exists = fs.exists(dir)
        if (exists) spark.read.parquet(dir.toString).createOrReplaceTempView(t)
        exists
      }
    }

  /** Query texts from `<dir>/<name>.sql` on any Hadoop file system, for a cluster without the spark-sql tests jar. */
  def queriesFrom(spark: SparkSession, dir: String, names: Seq[String]): Seq[(String, String)] = {
    val fs = fileSystem(spark, dir)
    names.flatMap { n =>
      val p = new HPath(s"${dir.stripSuffix("/")}/$n.sql")
      if (fs.exists(p)) {
        val in = fs.open(p)
        try Some(n -> new String(in.readAllBytes(), StandardCharsets.UTF_8))
        finally in.close()
      } else None
    }
  }

  private def fileSystem(spark: SparkSession, uriOrPath: String): FileSystem =
    FileSystem.get(
      new URI(if (uriOrPath.contains("://")) uriOrPath else new java.io.File(uriOrPath).toURI.toString),
      hadoopConf(spark)
    )

  private def hadoopConf(spark: SparkSession): Configuration = spark.sparkContext.hadoopConfiguration

  // ------------------------------------------------------------------ output

  /**
   * One result file per run: object stores have no append, so a cluster run writes
   * `<config>[-label]-<timestamp>.jsonl` and the report reads every file of the directory.
   */
  def openOutput(spark: SparkSession, out: String, config: String, label: String): java.io.PrintWriter = {
    val name = s"$config${if (label.isEmpty) "" else "-" + label}-${Instant.now().toString.replace(":", "")}.jsonl"
    val fs = fileSystem(spark, out)
    val dir = new HPath(out)
    fs.mkdirs(dir)
    new java.io.PrintWriter(new java.io.OutputStreamWriter(
      fs.create(new HPath(dir, name), true),
      StandardCharsets.UTF_8
    ))
  }

  /** Writes `text` to a path on any Hadoop file system (the report beside the rows). */
  def write(spark: SparkSession, path: String, text: String): Unit = {
    val fs = fileSystem(spark, path)
    val out = fs.create(new HPath(path), true)
    try out.write(text.getBytes(StandardCharsets.UTF_8))
    finally out.close()
  }

  /** Every `.jsonl` line under `dir` on any Hadoop file system. */
  def readRows(spark: SparkSession, dir: String): Seq[String] = {
    val fs = fileSystem(spark, dir)
    val files = fs.listStatus(
      new HPath(dir)
    ).filter(s => s.isFile && s.getPath.getName.endsWith(".jsonl")).sortBy(_.getPath.getName)
    files.toSeq.flatMap { s =>
      val in = fs.open(s.getPath)
      try new String(in.readAllBytes(), StandardCharsets.UTF_8).linesIterator.filter(_.nonEmpty).toSeq
      finally in.close()
    }
  }

  // ------------------------------------------------------------------ environment

  /** What the report's environment block needs from the session: versions, executor shape, and the engine's configuration. */
  final case class Environment(sparkVersion: String, executors: String, engineConf: String)

  object Environment {
    val Unknown: Environment = Environment("", "", "")

    private val EngineKeys = Seq(
      "spark.plugins",
      "spark.shuffle.manager",
      "spark.comet.",
      "spark.vector.",
      "spark.memory.offHeap",
      "spark.sql.adaptive.enabled"
    )

    def of(spark: SparkSession): Environment = {
      val sc = spark.sparkContext
      val conf = sc.getConf
      // The driver is one of the memory-status entries; local mode has only it.
      val live = math.max(0, sc.getExecutorMemoryStatus.size - 1)
      val shape = Seq(
        "spark.executor.instances",
        "spark.executor.cores",
        "spark.executor.memory",
        "spark.executor.memoryOverhead",
        "spark.driver.memory"
      )
        .flatMap(k => conf.getOption(k).map(v => s"$k=$v"))
      val executors =
        (if (conf.get("spark.master", "").startsWith("local")) Seq(s"master=${conf.get("spark.master")}")
         else Seq(s"live=$live")) ++ shape
      val engine = conf.getAll.filter { case (k, _) => EngineKeys.exists(k.startsWith) }.sortBy(_._1).map {
        case (k, v) => s"$k=$v"
      }
      Environment(
        s"${sc.version} / JDK ${System.getProperty("java.version")}",
        executors.mkString(" "),
        engine.mkString("; ")
      )
    }
  }

  // ------------------------------------------------------------------ the data-on-EKS-style report

  /** One row as the report reads it (a subset of [[TpchRunner]]'s record plus the cluster fields). */
  final case class ReportRow(
      config: String,
      query: String,
      dataset: String,
      medianMs: Double,
      rows: Int,
      checksum: String,
      accelerated: Option[(Int, Int)],
      fallbacks: Seq[String],
      metrics: Option[StageMetrics],
      sparkVersion: String,
      executors: String,
      engineConf: String
  ) {
    def seconds: Double = medianMs / 1000.0
  }

  private val Buckets: Seq[(String, Double => Boolean)] = Seq(
    (">= 20% improvement", s => s >= 1.25),
    ("10-20% improvement", s => s >= 1.111 && s < 1.25),
    ("within +/-10%", s => s > 0.909 && s < 1.111),
    ("10-20% degradation", s => s > 0.8 && s <= 0.909),
    (">= 20% degradation", s => s <= 0.8)
  )

  /**
   * The six sections of the reference report, per dataset, for every configuration against
   * `spark`. Queries whose checksums disagree between configurations are excluded from the totals
   * and listed, as correctness bugs rather than results.
   */
  def report(suiteTitle: String, queryOrder: Seq[String], rows: Seq[ReportRow]): String = {
    val sb = new StringBuilder
    sb.append(s"# $suiteTitle cluster results\n\n")
    sb.append(
      "Layout of the data-on-EKS Comet benchmark report: total completion time and speedup against plain Spark, the\n"
    )
    sb.append("distribution of per-query speedups, the largest improvements and every regression, the analysis table\n")
    sb.append(
      "(stage evidence pre-filled, cause to be written from the profile -- see the JFR-first protocol), the per-query\n"
    )
    sb.append("table and the environment. One pass per query; speedups are spark seconds over ours.\n")
    rows.groupBy(_.dataset).toSeq.sortBy(_._1).foreach { case (dataset, rs) =>
      val latest = rs.groupBy(r => (r.config, r.query)).view.mapValues(_.last).toMap
      val configs = TpchRunner.ConfigOrder.filter(c => latest.keys.exists(_._1 == c)) ++ latest.keys.map(
        _._1
      ).filterNot(TpchRunner.ConfigOrder.contains).toSeq.distinct.sorted
      val queries = latest.keys.map(_._2).toSeq.distinct.sortBy(q =>
        (queryOrder.indexOf(q) match { case -1 => Int.MaxValue; case i => i }, q)
      )
      val mismatched =
        queries.filter(q => configs.flatMap(c => latest.get((c, q)).map(_.checksum)).distinct.size > 1).toSet
      val comparable = queries.filterNot(mismatched.contains).filter(q => configs.forall(c => latest.contains((c, q))))
      def sec(c: String, q: String): Double = latest((c, q)).seconds
      def speedup(c: String, q: String): Double = sec("spark", q) / math.max(sec(c, q), 1e-9)
      sb.append(s"\n## $dataset\n\n")
      if (!configs.contains("spark")) {
        sb.append("No `spark` baseline in these rows: nothing to compare against.\n")
      } else {
        sb.append(
          s"${comparable.size} of ${queries.size} queries are comparable (every configuration ran them and agreed on the result)"
        )
        if (mismatched.nonEmpty)
          sb.append(s"; excluded as correctness bugs, checksums differ: ${mismatched.toSeq.sorted.mkString(", ")}")
        sb.append(
          ".\n\n### 1. Summary\n\n| configuration | total completion time (s) | speedup vs spark | % less runtime |\n|---|---:|---:|---:|\n"
        )
        val sparkTotal = comparable.map(sec("spark", _)).sum
        configs.foreach { c =>
          val total = comparable.map(sec(c, _)).sum
          sb.append(
            f"| $c | $total%.1f | ${sparkTotal / math.max(total, 1e-9)}%.2fx | ${(1 - total / math.max(sparkTotal, 1e-9)) * 100}%.1f%% |\n"
          )
        }
        configs.filter(_ != "spark").foreach { c =>
          val speedups = comparable.map(q => q -> speedup(c, q))
          sb.append(s"\n### 2. Performance distribution: $c\n\n| bucket | queries | share |\n|---|---:|---:|\n")
          Buckets.foreach { case (name, in) =>
            val n = speedups.count { case (_, s) => in(s) }
            sb.append(f"| $name | $n | ${if (speedups.isEmpty) 0.0 else 100.0 * n / speedups.size}%.1f%% |\n")
          }
          sb.append(
            s"\n### 3. Top improvements and every regression: $c\n\n| query | spark (s) | $c (s) | result |\n|---|---:|---:|---|\n"
          )
          val improvements = speedups.filter(_._2 > 1.0).sortBy(-_._2).take(10)
          improvements.foreach { case (q, s) =>
            sb.append(f"| $q | ${sec("spark", q)}%.2f | ${sec(c, q)}%.2f | $s%.2fx faster |\n")
          }
          if (improvements.isEmpty) sb.append("| - | | | no query is faster than spark |\n")
          val regressions = speedups.filter(_._2 < 1.0).sortBy(_._2)
          regressions.foreach { case (q, s) =>
            sb.append(f"| $q | ${sec("spark", q)}%.2f | ${sec(c, q)}%.2f | ${(1 / s - 1) * 100}%.0f%% slower |\n")
          }
          sb.append(s"\n### 4. Performance analysis: $c (cause and evidence to be completed from the profile)\n\n")
          sb.append(
            "| query | result | main cause | key evidence (stage metrics: ours vs spark) |\n|---|---|---|---|\n"
          )
          regressions.foreach { case (q, s) =>
            val ev = for (m <- latest((c, q)).metrics; b <- latest(("spark", q)).metrics)
              yield f"GC ${m.jvmGcTimeMs / 1000.0}%.1f s vs ${b.jvmGcTimeMs / 1000.0}%.1f s; shuffle read ${gb(m.shuffleReadBytes)} vs ${gb(b.shuffleReadBytes)}; executor time ${m.executorRunTimeMs / 1000.0}%.0f s vs ${b.executorRunTimeMs / 1000.0}%.0f s; spill ${gb(m.spillBytes)} vs ${gb(b.spillBytes)}"
            val fb = latest((c, q)).fallbacks
            val hint = if (fb.nonEmpty) s"operator fell back: ${fb.head}" else "(profile first: JFR-first protocol)"
            sb.append(
              f"| $q | ${(1 / s - 1) * 100}%.0f%% slower | $hint | ${ev.getOrElse("no stage metrics recorded")} |\n"
            )
          }
          if (regressions.isEmpty) sb.append("| - | no regressions | | |\n")
        }
        sb.append("\n### 5. Per-query results\n\n| query | " + configs.map(c => s"$c (s)").mkString(
          " | "
        ) + " | " + configs.filter(_ != "spark").map(c => s"$c accel.").mkString(" | ") + " |\n")
        sb.append("|---|" + configs.map(_ => "---:").mkString("|") + "|" + configs.filter(_ != "spark").map(_ =>
          "---:"
        ).mkString("|") + "|\n")
        queries.foreach { q =>
          val times = configs.map(c =>
            latest.get((c, q)).map(r =>
              f"${r.seconds}%.2f" + (if (c != "spark" && latest.contains(("spark", q))) f" (${speedup(c, q)}%.2fx)"
                                     else "")
            ).getOrElse("-")
          )
          val accel = configs.filter(_ != "spark").map(c =>
            latest.get((c, q)).flatMap(_.accelerated).map { case (a, t) => s"$a/$t" }.getOrElse("-")
          )
          sb.append(s"| $q${if (mismatched.contains(q)) " (checksum differs)" else ""} | " + times.mkString(
            " | "
          ) + " | " + accel.mkString(" | ") + " |\n")
        }
      }
      sb.append(
        "\n### 6. Environment\n\n| configuration | Spark | executors | engine configuration |\n|---|---|---|---|\n"
      )
      configs.foreach { c =>
        val r = latest.collectFirst { case ((cc, _), row) if cc == c => row }.get
        sb.append(s"| $c | ${r.sparkVersion} | ${r.executors} | ${r.engineConf} |\n")
      }
    }
    sb.toString
  }

  private def gb(bytes: Long): String =
    if (bytes >= (1L << 30)) f"${bytes / (1024.0 * 1024 * 1024)}%.1f GB" else f"${bytes / (1024.0 * 1024)}%.0f MB"
}
