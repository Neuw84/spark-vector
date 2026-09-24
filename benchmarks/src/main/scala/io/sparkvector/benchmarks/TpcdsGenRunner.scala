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

import org.apache.spark.sql.{DataFrame, Dataset, SaveMode, SparkSession, TPCDSSchema}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType

import scala.sys.process._

/**
 * Generates the TPC-DS tables as Parquet on any Hadoop file system by running `dsdgen` on the
 * executors (#247): each of `--parallel` children generates its slice of every table
 * (`dsdgen -TABLE t -SCALE sf -PARALLEL n -CHILD i -FILTER Y`, the rows on stdout), the lines are
 * parsed with Spark's own `TPCDSSchema` (money as DECIMAL(7,2), dates as DATE, keys as INT -- the
 * schema the runners and the per-query issues assume) and written with ZSTD, the seven fact tables
 * partitioned by their date key as the reference generator does, dimensions as one file each.
 *
 * {{{
 *   TpcdsGenRunner --scale 100 --out s3a://bucket/tpcds/sf100/parquet [--parallel 64] [--children-per-round 64]
 *   TpcdsGenRunner --count-only 1 --out s3a://bucket/tpcds/sf100/parquet     # rows and files per table
 *                  [--dsdgen /opt/tpcds-kit/tools] [--tables store_sales,date_dim]
 * }}}
 *
 * `dsdgen` and `tpcds.idx` must be at `--dsdgen` on every executor (the image builds tpcds-kit
 * there). At 1 TB one child of store_sales is ~2.9 GB of text; `--parallel` around the executor
 * core count times four keeps a child under a few minutes.
 */
object TpcdsGenRunner {

  private object Schema extends TPCDSSchema {
    def columns: Map[String, String] = tableColumns
    def partitions: Map[String, Seq[String]] = tablePartitionColumns
  }

  /** Tables dsdgen only emits from child 1 (it does not split them): generated once, not per child. */
  private val Unsplit = Set(
    "call_center",
    "catalog_page",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "time_dim",
    "warehouse",
    "web_page",
    "web_site"
  )

  def main(args: Array[String]): Unit = {
    val opts = args.sliding(2, 2).collect { case Array(k, v) if k.startsWith("--") => k.stripPrefix("--") -> v }.toMap
    val scale = opts.getOrElse("scale", "1").toInt
    val out = opts.getOrElse("out", sys.error("--out <base URI> is required")).stripSuffix("/")
    val parallel = opts.get("parallel").map(_.toInt).getOrElse(math.max(4, scale / 2))
    val dsdgenDir = opts.getOrElse("dsdgen", "/opt/tpcds-kit/tools")
    val childrenPerRound = opts.get("children-per-round").map(_.toInt).getOrElse(parallel)
    val only = opts.get("tables").map(_.split(",").map(_.trim).toSet)

    val spark = SparkSession.builder().appName(s"tpcds-gen-sf$scale").getOrCreate()
    val tables = Schema.columns.keys.toSeq.sorted.filter(t => only.forall(_.contains(t)))

    if (opts.contains("count-only")) {
      // Verification of an existing dataset: rows and files per table (a long run's driver log rotates
      // away before it ends).
      tables.foreach { table =>
        val df = spark.read.parquet(s"$out/$table")
        val files = df.inputFiles.length
        println(f"[tpcds-gen] $table: ${df.count()}%,d rows, $files%,d files")
      }
      spark.stop(); return
    }
    println(s"[tpcds-gen] scale $scale, ${tables.length} tables, $parallel children, dsdgen at $dsdgenDir, out $out")

    tables.foreach { table =>
      val start = System.nanoTime()
      val children = if (Unsplit(table)) 1 else parallel
      val ddl = Schema.columns(table)
      val schema = StructType.fromDDL(ddl)
      val partitionColumns =
        Schema.partitions.getOrElse(table, Nil).map(_.stripPrefix("`").stripSuffix("`")) // TPCDSSchema quotes them
      def rowsOf(childRange: Range): DataFrame = {
        val lines: Dataset[String] = spark.createDataset(
          spark.sparkContext.parallelize(childRange, childRange.length).flatMap { child =>
            // An unsplit table is generated whole (no -PARALLEL): dsdgen splits some of them, customer_demographics
            // among them, across the children, and child 1 alone is one slice.
            generate(dsdgenDir, table, scale, if (children == 1) 1 else parallel, child)
          }
        )(org.apache.spark.sql.Encoders.STRING)
        // dsdgen ends every row with a '|': one trailing empty field the schema does not have.
        spark.read.schema(schema).option(
          "sep",
          "|"
        ).option("nullValue", "").csv(lines.map(l => l.stripSuffix("|"))(org.apache.spark.sql.Encoders.STRING))
      }
      if (partitionColumns.isEmpty) {
        rowsOf(1 to children).coalesce(1).write.mode(SaveMode.Overwrite).option(
          "compression",
          "zstd"
        ).parquet(s"$out/$table")
      } else {
        // The repartition by the date key shuffles the whole table through the executors' local disks; in
        // rounds of `--children-per-round` children (append after the first) one round's shuffle is what
        // has to fit. Each round writes one file per date partition.
        val rounds = (1 to children).grouped(childrenPerRound).toSeq
        rounds.zipWithIndex.foreach { case (range, i) =>
          val mode = if (i == 0) SaveMode.Overwrite else SaveMode.Append
          rowsOf(range.head to range.last).repartition(partitionColumns.map(col): _*)
            .write.mode(mode).option("compression", "zstd").partitionBy(partitionColumns: _*).parquet(s"$out/$table")
          if (rounds.length > 1) println(
            s"[tpcds-gen]   $table round ${i + 1}/${rounds.length} (children ${range.head}-${range.last}) written"
          )
        }
      }
      val count = spark.read.parquet(s"$out/$table").count()
      println(f"[tpcds-gen] $table: $count%,d rows, ${(System.nanoTime() - start) / 1e9}%.0f s" +
        (if (partitionColumns.nonEmpty) s", partitioned by ${partitionColumns.mkString(",")}" else ""))
    }
    spark.stop()
  }

  /** One dsdgen child's rows of one table, streamed from its stdout (a child's slice can be gigabytes of text). */
  private def generate(dsdgenDir: String, table: String, scale: Int, parallel: Int, child: Int): Iterator[String] = {
    val cmd = Seq(
      s"$dsdgenDir/dsdgen",
      "-TABLE",
      table,
      "-SCALE",
      scale.toString,
      "-FILTER",
      "Y",
      "-QUIET",
      "Y",
      "-RNGSEED",
      "100",
      "-DISTRIBUTIONS",
      s"$dsdgenDir/tpcds.idx"
    ) ++
      (if (parallel > 1) Seq("-PARALLEL", parallel.toString, "-CHILD", child.toString) else Nil)
    val errors = new StringBuilder
    // lazyLines throws at the end of the stream when dsdgen exits non-zero, with stderr collected here.
    Process(
      cmd,
      new java.io.File(dsdgenDir)
    ).lazyLines(ProcessLogger(_ => (), err => errors.synchronized { errors.append(err).append('\n') })).iterator
  }
}
