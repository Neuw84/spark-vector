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

import java.io.{File, PrintWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.time.Instant

import scala.jdk.CollectionConverters._

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.vector.ui.{Engine, PlanAcceleration}
import io.sparkvector.spark.iceberg.IcebergVectorAdapter
import io.sparkvector.benchmarks.IcebergMorGenerator.TableProfile

/**
 * The CDC shape over an Iceberg v2 merge-on-read table (#260's read harness measures the reads;
 * this one measures the writes too): a `lineitem` variant carrying heavy positional deletes stands
 * in for a table a change-data-capture sink keeps mutating between compactions, and one
 * configuration at a time
 *
 *   1. runs the read queries a consumer would (`count`, the Q1 aggregate, the Q6 filter),
 *   2. runs a MERGE INTO of a pre-materialised change batch -- updates, deletes and inserts over
 *      about a fifth of the live rows -- several times, **rolling the table back to the pinned
 *      snapshot after every run** (`system.rollback_to_snapshot`), so every run and every
 *      configuration merges into the identical table state,
 *   3. applies the merge once more and re-runs the reads over the merged state (the table a
 *      consumer sees right after the CDC batch landed, one more delete layer thick),
 *
 * then rolls back and reports. Rollback is metadata-only, which is what makes the comparison fair;
 * the data and delete files the rolled-back merges wrote stay on disk until an
 * `expire_snapshots`/`remove_orphan_files` maintenance pass, exactly as they would in production.
 *
 * Results append to `<out>/cdc-<config>.jsonl`; `--report <dir>` rewrites
 * `<dir>/cdc-merge.{md,html}` from every `cdc-*.jsonl` there. One JVM per configuration, like
 * `TpchRunner`, whose `spark`/`vector` `Configs` this runner reuses.
 *
 * {{{
 * benchmarks/scripts/run-cdc-merge.sh benchmarks/data/iceberg sf25.pos_20 spark,vector
 * benchmarks/scripts/run-cdc-merge.sh --report            # only rewrite the reports
 * }}}
 */
object CdcMergeRunner {

  final case class Args(
      config: String = "vector",
      warehouse: String = "benchmarks/data/iceberg",
      table: String = "sf25.pos_20",
      baseTable: String = "lineitem",
      out: String = "benchmarks/results",
      /** Timed MERGE runs (each followed by a rollback). */
      iterations: Int = 3,
      /** Untimed MERGE runs before the timed ones (JIT, shuffle service warm-up). */
      warmup: Int = 1,
      readIterations: Int = 5,
      readWarmup: Int = 2,
      threads: Int = Runtime.getRuntime.availableProcessors(),
      shufflePartitions: Int = 8,
      extraConf: Map[String, String] = Map.empty,
      report: Option[String] = None,
      /** Skip the read phases (merge only). */
      mergeOnly: Boolean = false,
      /**
       * Size of the change batch as a % of the table's rows (before deletes), split like the default
       * batch -- ~73 % updates, ~19 % deletes, ~8 % inserts -- and placed on buckets no generator
       * delete touched (see [[IcebergMorGenerator.firstLiveBucket]]). None = the historical batch.
       */
      changePct: Option[Double] = None
  )

  def main(argv: Array[String]): Unit = {
    def parse(rest: List[String], a: Args): Args = rest match {
      case Nil => a
      case "--config" :: v :: t => parse(t, a.copy(config = v))
      case "--iceberg" :: v :: t => parse(t, a.copy(warehouse = v))
      case "--table" :: v :: t => parse(t, a.copy(table = v))
      case "--base-table" :: v :: t => parse(t, a.copy(baseTable = v))
      case "--out" :: v :: t => parse(t, a.copy(out = v))
      case "--iterations" :: v :: t => parse(t, a.copy(iterations = v.toInt))
      case "--warmup" :: v :: t => parse(t, a.copy(warmup = v.toInt))
      case "--read-iterations" :: v :: t => parse(t, a.copy(readIterations = v.toInt))
      case "--read-warmup" :: v :: t => parse(t, a.copy(readWarmup = v.toInt))
      case "--threads" :: v :: t => parse(t, a.copy(threads = v.toInt))
      case "--shuffle-partitions" :: v :: t => parse(t, a.copy(shufflePartitions = v.toInt))
      case "--merge-only" :: t => parse(t, a.copy(mergeOnly = true))
      case "--change-pct" :: v :: t =>
        parse(t, a.copy(changePct = if (v == "default") None else Some(v.toDouble)))
      case "--report" :: v :: t => parse(t, a.copy(report = Some(v)))
      case "--conf" :: kv :: t =>
        val Array(k, v) = kv.split("=", 2)
        parse(t, a.copy(extraConf = a.extraConf + (k -> v)))
      case other :: _ => throw new IllegalArgumentException(s"unknown argument $other")
    }
    val args = parse(argv.toList, Args())
    args.report match {
      case Some(dir) => report(Paths.get(dir))
      case None => run(args)
    }
  }

  // ------------------------------------------------------------------ the change batch

  /**
   * The change batch, as buckets of `pmod(xxhash64(<hashKey>), 1000)`. The `pos_<pct>` generator
   * variants deleted buckets `[0, 10 * pct)`, so over a `pos_20`-style table these ranges touch live
   * rows only: updates ~18.75 % of the live rows, deletes ~5 % -- about a quarter of the live data --
   * and inserts re-key another ~2 % above every existing key.
   */
  private val UpdateBuckets = (200, 350)
  private val DeleteBuckets = (350, 390)
  private val InsertBuckets = (390, 406)

  /**
   * (updates, deletes, inserts) bucket ranges. Without `changePct` the historical batch above; with
   * it, `changePct * 10` buckets starting at the table's first untouched bucket, split 73/19/8 like
   * the historical 150/40/16. The buckets hash the ticket/order key, which is spread over every data
   * file, so the batch touches all of them.
   */
  private[benchmarks] def changeRanges(table: String, changePct: Option[Double]): ((Int, Int), (Int, Int), (Int, Int)) =
    changePct match {
      case None => (UpdateBuckets, DeleteBuckets, InsertBuckets)
      case Some(pct) =>
        val from = IcebergMorGenerator.firstLiveBucket(table.substring(table.lastIndexOf('.') + 1))
        val total = math.round(pct * 10).toInt
        require(total > 0 && from + total <= 1000, s"--change-pct $pct does not fit buckets [$from, 1000)")
        val u = math.round(total * 0.73).toInt
        val d = math.round(total * 0.19).toInt
        ((from, from + u), (from + u, from + u + d), (from + u + d, from + total))
    }

  private def buckets(p: TableProfile, range: (Int, Int)): String =
    s"pmod(xxhash64(${p.hashKey}), 1000) >= ${range._1} AND pmod(xxhash64(${p.hashKey}), 1000) < ${range._2}"

  private def mergeSql(p: TableProfile): String = {
    val on = p.grain.map(g => s"t.$g = s.$g").mkString(" AND ")
    s"""MERGE INTO %s t USING cdc_changes s
       |ON $on
       |WHEN MATCHED AND s.op = 'D' THEN DELETE
       |WHEN MATCHED THEN UPDATE SET ${p.cdcMergeSet}
       |WHEN NOT MATCHED THEN INSERT *""".stripMargin
  }

  /** Reads a CDC consumer runs; `checksum` compares configurations, `agg-after-merge` also pins the merged state. */
  private def readQueries(p: TableProfile): Seq[(String, String)] = Seq(
    "count" -> "SELECT count(*) FROM %s",
    "scan-agg" -> p.scanAggSql,
    "filter-agg" -> p.filterAggSql
  )

  // ------------------------------------------------------------------ run

  private def run(args: Args): Unit = {
    val p = IcebergMorGenerator.Profiles.getOrElse(
      args.baseTable,
      throw new IllegalArgumentException(
        s"unknown base table ${args.baseTable}; known: ${IcebergMorGenerator.Profiles.keys.mkString(", ")}"
      )
    )
    val conf = TpchRunner.Configs.getOrElse(
      args.config,
      throw new IllegalArgumentException(
        s"unknown config ${args.config}; known: ${TpchRunner.ConfigOrder.mkString(", ")}"
      )
    )
    val warehouse = IcebergMorGenerator.resolveWarehouse(args.warehouse)
    // On the cluster spark-submit sets --master; locally default to local[threads].
    val builder = SparkSession.builder()
      .appName(s"spark-vector-cdc-${args.config}")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", args.shufflePartitions.toString)
      .config("spark.sql.adaptive.enabled", "true")
    if (Option(System.getProperty("spark.master")).isEmpty && sys.env.get("SPARK_MASTER").isEmpty)
      builder.master(s"local[${args.threads}]").config("spark.driver.host", "localhost")
    (conf ++ IcebergMorGenerator.catalogConf(args.warehouse) ++ args.extraConf).foreach { case (k, v) =>
      builder.config(k, v)
    }
    val spark = builder.getOrCreate()
    try {
      val table = s"${IcebergMorGenerator.Catalog}.${args.table}"
      require(spark.catalog.tableExists(table), s"$table does not exist under $warehouse (see gen-iceberg-mor.sh)")
      // The *current* snapshot (the main ref), not the last committed one: after an earlier run's
      // merge + rollback, the newest row of `.snapshots` is a rolled-back merge, not the state read.
      val baseline = spark.sql(s"SELECT snapshot_id FROM $table.refs WHERE name = 'main'").collect()(0).getLong(0)
      val live = spark.table(table).count()
      val stats = tableStats(spark, table)
      println(s"[cdc] config=${args.config} table=$table snapshot=$baseline live=$live " +
        s"dataFiles=${stats.dataFiles} (${stats.dataBytes / (1 << 20)} MiB) deleteFiles=${stats.deleteFiles} deleteRows=${stats.deleteRows}")

      // The change batch, materialised once (outside every timing) so each merge reads identical
      // files. On the cluster it must live where the executors can read it: an object-store prefix
      // when `out` has a scheme, a driver-local dir otherwise.
      val outScheme = args.out.contains("://")
      val changesDir =
        if (outScheme) s"${args.out.stripSuffix("/")}/cdc-changes-${args.table.replace('.', '_')}"
        else new File(new File(args.out), s"cdc-changes-${args.table.replace('.', '_')}").getAbsolutePath
      val changeExists =
        if (outScheme) {
          try { spark.read.parquet(changesDir).limit(1).count() > 0 }
          catch { case _: Exception => false }
        } else new File(changesDir, "_SUCCESS").exists()
      if (!changeExists) {
        println(s"[cdc] materialising the change batch under $changesDir")
        val maxKey = spark.table(table).selectExpr(s"max(${p.reKey})").collect()(0).getAs[Number](0).longValue()
        spark.table(table).createOrReplaceTempView("cdc_base")
        // Every branch projects the identical column list in table order: UNION ALL matches by position.
        val columns = spark.table(table).schema.fieldNames.toSeq
        def branch(op: String, tweaks: Map[String, String]) =
          s"SELECT '$op' AS op, ${columns.map(c => tweaks.getOrElse(c, c) + s" AS $c").mkString(", ")} FROM cdc_base"
        val grainCols = p.grain.mkString(", ")
        val (updR, delR, insR) = changeRanges(args.table, args.changePct)
        println(s"[cdc] change buckets: U=$updR D=$delR I=$insR (of 1000)")
        val touched = spark.sql(s"SELECT count(DISTINCT _file) FROM $table WHERE " +
          s"pmod(xxhash64(${p.hashKey}), 1000) >= ${updR._1} AND pmod(xxhash64(${p.hashKey}), 1000) < ${delR._2}")
          .collect()(0).getLong(0)
        println(s"[cdc] change batch updates/deletes rows in $touched of ${stats.dataFiles} data files")
        val batch = spark.sql(
          branch(
            "U",
            p.cdcUpdateTweaks
          ) + s" WHERE ${buckets(p, updR)}" +
            " UNION ALL " + branch("D", Map.empty) + s" WHERE ${buckets(p, delR)}" +
            " UNION ALL " + branch(
              "I",
              Map(p.reKey -> s"${p.reKey} + ${maxKey}L")
            ) + s" WHERE ${buckets(p, insR)}"
        )
        // At most one change row per grain: the timed MERGE rejects a target matched more than once
        // (SQLSTATE 23K01). lineitem's grain is a true key (no-op); store_sales is not, so dedupe.
        batch.createOrReplaceTempView("cdc_raw")
        spark.sql(s"SELECT ${batch.columns.mkString(", ")} FROM (SELECT *, " +
          s"row_number() OVER (PARTITION BY $grainCols ORDER BY $grainCols) AS _rn FROM cdc_raw) WHERE _rn = 1")
          .repartition(args.threads)
          .write.mode("overwrite").parquet(changesDir)
      }
      val changes = spark.read.parquet(changesDir)
      changes.createOrReplaceTempView("cdc_changes")
      val changeCounts = changes.groupBy("op").count().collect().map(r => r.getString(0) -> r.getLong(1)).toMap
      val changeBytes =
        if (outScheme) {
          // The parquet files' total size, from Hadoop's FileSystem (executors wrote them to S3).
          val path = new org.apache.hadoop.fs.Path(changesDir)
          val fs = path.getFileSystem(spark.sparkContext.hadoopConfiguration)
          fs.listStatus(path).filter(_.getPath.getName.endsWith(".parquet")).map(_.getLen).sum
        } else new File(changesDir).listFiles().filter(_.getName.endsWith(".parquet")).map(_.length()).sum
      println(s"[cdc] change batch: U=${changeCounts.getOrElse("U", 0L)} D=${changeCounts.getOrElse("D", 0L)} " +
        s"I=${changeCounts.getOrElse("I", 0L)} (${changeBytes / (1 << 20)} MiB parquet)")

      // The JSONL result always lands on the driver's local disk (the cluster script uploads it);
      // `out` may itself be an object-store prefix, which is not a local Path.
      val localOut = if (outScheme) "cdc-results" else args.out
      Files.createDirectories(Paths.get(localOut))
      val outFile = Paths.get(localOut, s"cdc-${args.config}.jsonl")
      val writer = new PrintWriter(Files.newBufferedWriter(
        outFile,
        StandardCharsets.UTF_8,
        java.nio.file.StandardOpenOption.CREATE,
        java.nio.file.StandardOpenOption.APPEND
      ))
      def emit(m: Measurement): Unit = {
        writer.println(m.toJson(args.config, args.table, live, changeCounts, changeBytes)); writer.flush()
        println(
          f"[cdc] ${args.config} ${m.phase}/${m.name} median=${m.medianMs}%.1fms p90=${m.p90Ms}%.1fms rows=${m.rows} " +
            s"accelerated=${m.acceleratedOps}/${m.operatorCount}" +
            (if (m.morPhysicalRows > 0) f" mor live/physical=${m.morLiveRows}/${m.morPhysicalRows}" else "")
        )
        m.fallbacks.foreach(f => println(s"[cdc]   fallback: $f"))
      }
      // Whatever happens mid-phase (an exception, Ctrl-C reaching the shutdown hook), put the table
      // back; only a hard kill can leave a merge applied, and the printed baseline is the id to
      // rollback_to_snapshot by hand then.
      def restore(): Unit =
        try {
          val current = spark.sql(s"SELECT snapshot_id FROM $table.refs WHERE name = 'main'").collect()(0).getLong(0)
          if (current != baseline) {
            spark.sql(s"CALL ${IcebergMorGenerator.Catalog}.system.rollback_to_snapshot('${args.table}', ${baseline}L)")
            println(s"[cdc] restored $table to snapshot $baseline")
          }
          spark.sql(s"CALL ${IcebergMorGenerator.Catalog}.system.expire_snapshots(table => '${args.table}', " +
            s"older_than => TIMESTAMP '${java.sql.Timestamp.from(java.time.Instant.now().plusSeconds(60))}', retain_last => 1)").collect()
        } catch {
          case e: Exception => println(s"[cdc] WARNING: could not restore $table to $baseline: ${e.getMessage}")
        }
      try {
        // A rolled-back merge leaves its data and delete files orphaned; at this scale that is
        // gigabytes per run, and together with the merge's shuffle spill it can fill the disk.
        // Expiring everything but the current snapshot deletes those files (the baseline's own
        // files stay: the current snapshot references them). Untimed, like the rollback.
        // The free space right after start-up: the level each merge's scratch must return to.
        val freeAtStart = new File("/").getUsableSpace
        def expireOrphans(): Unit = {
          try spark.sql(s"CALL ${IcebergMorGenerator.Catalog}.system.expire_snapshots(table => '${args.table}', " +
              s"older_than => TIMESTAMP '${java.sql.Timestamp.from(java.time.Instant.now().plusSeconds(60))}', retain_last => 1)").collect()
          catch { case e: Exception => println(s"[cdc] WARNING: expire_snapshots failed: ${e.getMessage}") }
          // Shuffle files live until the ContextCleaner sees their dependencies collected, which is
          // asynchronous: wait for the disk to actually recover (a merge writes 15+ GiB of scratch,
          // and unreclaimed runs stack up to a full disk -- measured). Untimed, like the rollback.
          // Only meaningful for a local run, where the scratch is on this same disk; on the cluster
          // the executors hold it, so the driver's free space is not the signal.
          if (!outScheme && Option(System.getProperty("spark.master")).isEmpty && sys.env.get("SPARK_MASTER").isEmpty) {
            val target = freeAtStart - (4L << 30)
            val deadline = System.nanoTime() + 180L * 1000 * 1000 * 1000
            System.gc()
            while (new File("/").getUsableSpace < target && System.nanoTime() < deadline) {
              Thread.sleep(3000)
              System.gc()
            }
            println(s"[cdc]   disk free: ${new File("/").getUsableSpace / (1L << 30)} GiB")
          }
        }
        def rollback(): Unit = {
          spark.sql(s"CALL ${IcebergMorGenerator.Catalog}.system.rollback_to_snapshot('${args.table}', ${baseline}L)")
          val back = spark.table(table).count()
          require(back == live, s"rollback left $back rows, expected $live")
          expireOrphans()
        }
        expireOrphans() // files a previous aborted run may have left behind
        // MERGE executes eagerly inside spark.sql; the returned frame's plan is a CommandResultExec
        // wrapping the executed physical plan of the command (its read side is what the plugin can
        // accelerate). The stats are extracted immediately and the plan dropped: a retained plan
        // pins the merge's shuffle files through its RDD lineage -- 15+ GiB per merge at this scale.
        final case class MergeAccel(acceleratedOps: Int, operatorCount: Int, fallbacks: Seq[String], plan: String)
        @volatile var mergeAccel: Option[MergeAccel] = None
        def mergeOnce(): Double = {
          val start = System.nanoTime()
          val df = spark.sql(mergeSql(p).format(table))
          val ms = (System.nanoTime() - start) / 1e6
          val plan = df.queryExecution.executedPlan match {
            case c: org.apache.spark.sql.execution.CommandResultExec => c.commandPhysicalPlan
            case p => p
          }
          val a = PlanAcceleration.fromPlan(plan)
          mergeAccel = Some(MergeAccel(
            a.nodes.count(n => !Engine.plumbing.contains(n.engine) && n.engine.isAccelerated),
            a.operatorCount,
            a.fallbacks.map { case (n, r) => s"$n: $r" }.distinct,
            plan.treeString.take(4000)
          ))
          ms
        }

        // 1. Reads over the baseline state.
        if (!args.mergeOnly) readQueries(p).foreach { case (name, sql) =>
          emit(measureRead(spark, "read", name, sql.format(table), args.readWarmup, args.readIterations))
        }

        // 2. The timed merges, each into the identical baseline state.
        (1 to args.warmup).foreach { i => mergeOnce(); rollback(); println(s"[cdc] merge warm-up $i done") }
        val times = (1 to args.iterations).map { i =>
          val ms = mergeOnce()
          val merged = spark.table(table).count()
          val snapshot = latestSnapshot(spark, table)
          if (i < args.iterations) rollback() // the last merged state stays for phase 3
          println(f"[cdc] merge $i: $ms%.0f ms, $merged rows after")
          (ms, merged, snapshot)
        }
        val mergedRows = times.head._2
        require(times.map(_._2).distinct.size == 1, s"merge row counts diverged: ${times.map(_._2)}")
        val accel = mergeAccel.getOrElse(MergeAccel(0, 0, Seq("merge plan not captured"), ""))
        emit(Measurement(
          "merge",
          "merge",
          times.map(_._1),
          mergedRows.toInt,
          checksumOf(spark, table, p),
          plan = accel.plan,
          acceleratedOps = accel.acceleratedOps,
          operatorCount = accel.operatorCount,
          fallbacks = accel.fallbacks,
          mergeSnapshotSummary = times.last._3
        ))

        // 3. Reads over the merged state (the last timed merge's), then leave the table as we found it.
        if (!args.mergeOnly) readQueries(p).foreach { case (name, sql) =>
          emit(measureRead(spark, "read-after-merge", name, sql.format(table), args.readWarmup, args.readIterations))
        }
      } finally { restore(); writer.close() }
    } finally spark.stop()
  }

  /** `content, records, bytes` of the table's current files, split data vs deletes. */
  private final case class TableStats(dataFiles: Long, dataBytes: Long, deleteFiles: Long, deleteRows: Long)
  private def tableStats(spark: SparkSession, table: String): TableStats = {
    val rows = spark.sql(
      s"SELECT content, count(*), sum(record_count), sum(file_size_in_bytes) FROM $table.files GROUP BY content"
    ).collect()
    def of(pred: Int => Boolean, col: Int) = rows.filter(r => pred(r.getInt(0))).map(_.getLong(col)).sum
    TableStats(of(_ == 0, 1), of(_ == 0, 3), of(_ > 0, 1), of(_ > 0, 2))
  }

  private def latestSnapshot(spark: SparkSession, table: String): Map[String, String] = {
    val row = spark.sql(s"SELECT summary FROM $table.snapshots ORDER BY committed_at DESC LIMIT 1").collect()(0)
    row.getMap[String, String](0).toMap
  }

  /** A cheap whole-table checksum: global aggregates to 10 significant digits, like TpchRunner's row checksums. */
  private def checksumOf(spark: SparkSession, table: String, p: TableProfile): String = {
    // replace, not format: the checksum SQL uses the `%` modulo operator, which String.format rejects.
    val r = spark.sql(p.checksumSql.replace("%s", table)).collect()(0)
    (0 until r.length).map { i =>
      r.get(i) match { case d: java.lang.Double => f"${d.doubleValue()}%.10g"; case v => String.valueOf(v) }
    }.mkString("|").hashCode.toHexString
  }

  private def measureRead(
      spark: SparkSession,
      phase: String,
      name: String,
      sql: String,
      warmup: Int,
      iterations: Int
  ): Measurement = {
    def once(): (Double, Array[org.apache.spark.sql.Row], SparkPlan) = {
      val df = spark.sql(sql)
      val start = System.nanoTime()
      val rows = df.collect()
      ((System.nanoTime() - start) / 1e6, rows, df.queryExecution.executedPlan)
    }
    (1 to warmup).foreach(_ => once())
    val runs = (1 until iterations).map(_ => once())
    val (physicalBefore, liveBefore) =
      (IcebergVectorAdapter.normalizedPhysicalRows(), IcebergVectorAdapter.normalizedLiveRows())
    val last = once()
    val morPhysical = IcebergVectorAdapter.normalizedPhysicalRows() - physicalBefore
    val morLive = IcebergVectorAdapter.normalizedLiveRows() - liveBefore
    val (_, rows, plan) = last
    val checksum = rows.map(_.toSeq.map {
      case d: java.lang.Double => f"${d.doubleValue()}%.10g"
      case v => String.valueOf(v)
    }.mkString("|")).sorted.mkString("\n").hashCode.toHexString
    val accel = PlanAcceleration.fromPlan(plan)
    Measurement(
      phase,
      name,
      (runs :+ last).map(_._1),
      rows.length,
      checksum,
      plan.treeString.take(4000),
      accel.nodes.count(n => !Engine.plumbing.contains(n.engine) && n.engine.isAccelerated),
      accel.operatorCount,
      accel.fallbacks.map { case (n, r) => s"$n: $r" }.distinct,
      morPhysical,
      morLive
    )
  }

  final case class Measurement(
      phase: String,
      name: String,
      timesMs: Seq[Double],
      rows: Int,
      checksum: String,
      plan: String,
      acceleratedOps: Int,
      operatorCount: Int,
      fallbacks: Seq[String],
      morPhysicalRows: Long = 0,
      morLiveRows: Long = 0,
      mergeSnapshotSummary: Map[String, String] = Map.empty
  ) {
    private val sorted = timesMs.sorted
    def medianMs: Double = percentile(50)
    def p90Ms: Double = percentile(90)
    private def percentile(p: Int): Double = {
      val idx = math.min(sorted.size - 1, math.max(0, math.ceil(p / 100.0 * sorted.size).toInt - 1))
      math.round(sorted(idx) * 10) / 10.0
    }
    def toJson(config: String, table: String, liveRows: Long, changes: Map[String, Long], changeBytes: Long): String = {
      def esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
      val summary = mergeSnapshotSummary.map { case (k, v) => s""""${esc(k)}":"${esc(v)}"""" }.mkString("{", ",", "}")
      s"""{"timestamp":"${Instant.now()}","config":"$config","table":"$table","phase":"$phase","name":"$name",""" +
        s""""liveRows":$liveRows,"changeU":${changes.getOrElse("U", 0L)},"changeD":${changes.getOrElse(
            "D",
            0L
          )},"changeI":${changes.getOrElse("I", 0L)},"changeBytes":$changeBytes,""" +
        s""""medianMs":$medianMs,"p90Ms":$p90Ms,"timesMs":[${timesMs.map(t => math.round(t * 10) / 10.0).mkString(
            ","
          )}],""" +
        s""""rows":$rows,"checksum":"$checksum","acceleratedOps":$acceleratedOps,"operatorCount":$operatorCount,""" +
        s""""morPhysicalRows":$morPhysicalRows,"morLiveRows":$morLiveRows,"mergeSummary":$summary,""" +
        s""""fallbacks":"${esc(fallbacks.mkString("; "))}","plan":"${esc(plan)}"}"""
    }
  }

  // ------------------------------------------------------------------ report

  private final case class Record(
      config: String,
      table: String,
      phase: String,
      name: String,
      medianMs: Double,
      p90Ms: Double,
      timesMs: Seq[Double],
      rows: Long,
      checksum: String,
      acceleratedOps: Int,
      operatorCount: Int,
      liveRows: Long,
      changeU: Long,
      changeD: Long,
      changeI: Long,
      changeBytes: Long,
      morPhysicalRows: Long,
      morLiveRows: Long,
      mergeSummary: Map[String, String],
      fallbacks: String
  )

  private def parseRecord(line: String): Record = {
    def str(k: String) = s""""$k":"(.*?)(?<!\\\\)"""".r.findFirstMatchIn(line).map(_.group(1)).getOrElse("")
    def num(k: String) = s""""$k":(-?[0-9.]+)""".r.findFirstMatchIn(line).map(_.group(1).toDouble).getOrElse(0.0)
    val times = s""""timesMs":\\[(.*?)\\]""".r.findFirstMatchIn(line).map(_.group(1)).getOrElse("")
      .split(",").filter(_.nonEmpty).map(_.toDouble).toSeq
    val summary = s""""mergeSummary":\\{(.*?)\\}""".r.findFirstMatchIn(line).map(_.group(1)).getOrElse("")
    val summaryMap = """"([^"]+)":"([^"]*)"""".r.findAllMatchIn(summary).map(m => m.group(1) -> m.group(2)).toMap
    Record(
      str("config"),
      str("table"),
      str("phase"),
      str("name"),
      num("medianMs"),
      num("p90Ms"),
      times,
      num("rows").toLong,
      str("checksum"),
      num("acceleratedOps").toInt,
      num("operatorCount").toInt,
      num("liveRows").toLong,
      num("changeU").toLong,
      num("changeD").toLong,
      num("changeI").toLong,
      num("changeBytes").toLong,
      num("morPhysicalRows").toLong,
      num("morLiveRows").toLong,
      summaryMap,
      str("fallbacks")
    )
  }

  private def report(dir: java.nio.file.Path): Unit = {
    val files = Files.list(
      dir
    ).iterator().asScala.filter(_.getFileName.toString.matches("cdc-.*\\.jsonl")).toSeq.sortBy(_.toString)
    require(files.nonEmpty, s"no cdc-*.jsonl under $dir")
    val records = files.flatMap(f => Files.readAllLines(f).asScala.filter(_.nonEmpty).map(parseRecord))
    // Latest record per (table, config, phase, name).
    val latest = records.zipWithIndex.groupBy(r => (r._1.table, r._1.config, r._1.phase, r._1.name))
      .view.mapValues(_.maxBy(_._2)._1).toMap
    val tables = latest.keys.map(_._1).toSeq.distinct.sorted
    val md = new StringBuilder
    val html = new StringBuilder
    html.append(
      "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n<title>Iceberg CDC merge benchmark</title>\n<style>\n"
    )
    html.append(
      " body{font:15px/1.5 -apple-system,'Segoe UI',sans-serif;color:#1c2733;max-width:1100px;margin:2rem auto;padding:0 1rem}\n"
    )
    html.append(
      " table{border-collapse:collapse;margin:1rem 0;font-variant-numeric:tabular-nums} th,td{padding:.35rem .7rem;border-bottom:1px solid #e6eaf0;text-align:right} th:first-child,td:first-child{text-align:left}\n"
    )
    html.append(
      " th{background:#f4f6f9} code{background:#f4f6f9;padding:.1rem .3rem;border-radius:4px} .muted{color:#6b7684} .good{color:#0a7f42;font-weight:600} .bad{color:#b3261e}\n"
    )
    html.append("</style></head><body>\n<h1>Iceberg v2 merge-on-read: the CDC shape</h1>\n")
    md.append("# Iceberg v2 merge-on-read: the CDC shape\n")
    tables.foreach { table =>
      val ofTable = latest.filter(_._1._1 == table)
      val configs = TpchRunner.ConfigOrder.filter(c => ofTable.keys.exists(_._2 == c)) ++
        ofTable.keys.map(_._2).filterNot(TpchRunner.ConfigOrder.contains).toSeq.distinct.sorted
      val any = ofTable.values.head
      val merged = configs.flatMap(c => ofTable.get((table, c, "merge", "merge")))
      val header = f"Table <code>${esc(table)}</code>: ${any.liveRows}%,d live rows; change batch " +
        f"${any.changeU}%,d updates + ${any.changeD}%,d deletes + ${any.changeI}%,d inserts (${any.changeBytes / (1 << 20)} MiB Parquet). " +
        "Every merge runs against the identical pinned snapshot (rollback between runs)."
      html.append(s"<h2>${esc(table)}</h2>\n<p>$header</p>\n")
      md.append(s"\n## ${table}\n\n${header.replaceAll("<[^>]+>", "")}\n")
      def section(title: String, phase: String, names: Seq[String]): Unit = {
        html.append(s"<h3>${esc(title)}</h3>\n<table><thead><tr><th>query</th>")
        configs.foreach(c => html.append(s"<th>${esc(c)} (ms)</th>"))
        html.append("<th>speedup</th><th>accelerated</th><th>checksums</th></tr></thead><tbody>\n")
        md.append(s"\n### $title\n\n| query | ${configs.mkString(
            " | "
          )} | speedup | accelerated | checksums |\n|---|${configs.map(_ => "---:").mkString("|")}|---:|---:|---|\n")
        names.foreach { n =>
          val rs = configs.map(c => ofTable.get((table, c, phase, n)))
          val spark = ofTable.get((table, "spark", phase, n)).map(_.medianMs)
          val best = rs.flatten.filter(_.config != "spark").map(_.medianMs).minOption
          val speedup = for (s <- spark; b <- best) yield s / b
          val checksums = rs.flatten.map(_.checksum).distinct
          val check = if (checksums.size <= 1) "identical"
          else "DIFFER: " + rs.flatten.map(r => s"${r.config}=${r.checksum}").mkString(", ")
          val accel = rs.flatten.filterNot(_.config == "spark").map(r =>
            s"${r.acceleratedOps}/${r.operatorCount}"
          ).distinct.mkString(" ")
          html.append(s"<tr><td>${esc(n)}</td>")
          rs.foreach(r => html.append(s"<td>${r.map(x => f"${x.medianMs}%.0f").getOrElse("-")}</td>"))
          html.append(
            f"<td>${speedup.map(s => f"<span class=\"${if (s >= 1) "good" else "bad"}\">$s%.2fx</span>").getOrElse("-")}</td>"
          )
          html.append(
            s"<td>$accel</td><td>${if (checksums.size <= 1) "identical" else s"<span class=\"bad\">${esc(check)}</span>"}</td></tr>\n"
          )
          md.append(s"| $n | ${rs.map(r =>
              r.map(x => f"${x.medianMs}%.0f").getOrElse("-")
            ).mkString(" | ")} | ${speedup.map(s => f"$s%.2fx").getOrElse("-")} | $accel | $check |\n")
        }
        html.append("</tbody></table>\n")
      }
      val readNames = Seq("count", "scan-agg", "filter-agg")
      if (ofTable.keys.exists(_._3 == "read")) section("Reads over the baseline table", "read", readNames)
      if (merged.nonEmpty) {
        section("The CDC merge (median of the timed runs, table rolled back in between)", "merge", Seq("merge"))
        val s = merged.head.mergeSummary
        val detail =
          Seq("added-data-files", "added-delete-files", "added-position-deletes", "added-records", "deleted-records")
            .flatMap(k => s.get(k).map(v => s"$k=$v")).mkString(", ")
        if (detail.nonEmpty) {
          html.append(s"<p class=\"muted\">Last merge snapshot: $detail.</p>\n");
          md.append(s"\nLast merge snapshot: $detail.\n")
        }
        merged.foreach { m =>
          val line = f"${m.config}: runs ${m.timesMs.map(t => f"$t%.0f").mkString(" ")} ms, p90 ${m.p90Ms}%.0f ms"
          html.append(s"<p class=\"muted\">${esc(line)}</p>\n"); md.append(s"$line\n")
        }
      }
      if (ofTable.keys.exists(_._3 == "read-after-merge"))
        section("Reads over the merged state (one more delete layer)", "read-after-merge", readNames)
      val fallbacks = ofTable.values.filter(r => r.config != "spark" && r.fallbacks.nonEmpty)
        .flatMap(r => r.fallbacks.split("; ").map(f => s"${r.phase}/${r.name}: $f")).toSeq.distinct.sorted
      if (fallbacks.nonEmpty) {
        html.append("<details><summary>What stayed on Spark (fallback reasons)</summary><ul>\n")
        md.append("\n### What stayed on Spark\n\n")
        fallbacks.foreach { f => html.append(s"<li><code>${esc(f)}</code></li>\n"); md.append(s"- `$f`\n") }
        html.append("</ul></details>\n")
      }
    }
    html.append(s"<p class=\"muted\">Generated ${Instant.now()} by CdcMergeRunner.</p>\n</body></html>\n")
    Files.writeString(dir.resolve("cdc-merge.html"), html.toString, StandardCharsets.UTF_8)
    Files.writeString(dir.resolve("cdc-merge.md"), md.toString, StandardCharsets.UTF_8)
    println(s"[cdc] wrote ${dir.resolve("cdc-merge.html")} and ${dir.resolve("cdc-merge.md")}")
  }

  private def esc(s: String): String = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
}
