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

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import scala.collection.mutable.ArrayBuffer

import org.apache.iceberg.{FileFormat, Table}
import org.apache.iceberg.data.{GenericAppenderFactory, GenericRecord}
import org.apache.iceberg.io.OutputFileFactory
import org.apache.iceberg.spark.Spark3Util
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Builds the Iceberg merge-on-read variants of TPC-H `lineitem` the MoR harness measures (#260):
 * the same rows as the Parquet table `gen-tpch.sh` produced, in a local Hadoop catalog, mutated in
 * the shapes a lakehouse table has between compactions. One table per variant, all under one
 * namespace, plus a `README.md` next to the warehouse with what each holds -- live rows, data and
 * delete files, delete rows per data file, the snapshot id -- and the SQL that produced it.
 *
 * Variants (`<pct>` is the share of the rows the deletes remove):
 *   - `plain`                  v2, no deletes: the reader's own cost.
 *   - `pos_<pct>`              v2 positional deletes over every data file, scattered: rows picked by
 *                              a hash of `l_orderkey`, so each 64-row block loses a few rows.
 *   - `pos_<pct>_clustered`    v2 positional deletes of whole `l_shipdate` ranges: entire blocks go,
 *                              which is what the active-block skipping in `EvalContext` is for.
 *   - `pos_upd_<pct>`          `pos_10` plus an UPDATE of a further `<pct>` % of the rows (deletes of
 *                              the old images plus new small data files) and one MERGE INTO that
 *                              deletes, updates and inserts: the heavily mutated shape.
 *   - `eq_<pct>`               v2 with equality delete files on `l_orderkey`, written through the
 *                              Iceberg Java API the way a CDC sink does (Spark never writes them).
 *   - `dv_<pct>`, `dv_<pct>_clustered`, `dv_upd_<pct>`
 *                              v3 (`format-version=3`): the `pos_*` mutations encoded as deletion
 *                              vectors in Puffin files.
 *
 * Run through `gen-iceberg-mor.sh`; Spark alone (no plugin) writes the tables, so the counts in the
 * README come from Spark's row path -- the oracle the harness compares every configuration to.
 */
object IcebergMorGenerator {

  /** The catalog name the runner and this generator agree on. */
  val Catalog = "local"

  /**
   * The table-shaped facts the MoR variants need, so the same delete/update/merge machinery drives
   * `lineitem` and other base tables (e.g. TPC-DS `store_sales`). `hashKey` picks scattered rows;
   * `clusteredExpr(pct)` selects whole leading blocks (a date range or a surrogate-key range);
   * `grain` is the MERGE match key; `updateSet` / `mergeUpdateSet` are the SET clauses; `reKey` is the
   * key column an INSERT branch offsets so new rows sort above every existing one; `eqKey` is the
   * equality-delete column. Bare column names, valid under both engines.
   */
  final case class TableProfile(
      name: String,
      relation: String, // the child under --data holding the Parquet
      hashKey: String,
      clusteredExpr: Int => String,
      grain: Seq[String],
      reKey: String,
      updateSet: String, // UPDATE ... SET <this> (the generator's plain-update variant)
      mergeUpdateSet: String, // MERGE ... WHEN MATCHED THEN UPDATE SET <this>
      updateComment: String, // the column concat(' u')'d in the plain update (readback marker)
      eqKey: String,
      probeGroupKeys: Seq[String], // the group-by of the read harness's probe-group
      sumCol: String, // a numeric column the probes sum
      // --- the CdcMergeRunner's needs ---
      cdcUpdateTweaks: Map[String, String], // column -> expr for the change batch's UPDATE rows
      cdcMergeSet: String, // MERGE ... WHEN MATCHED THEN UPDATE SET <this> (t./s. qualified)
      scanAggSql: String, // the probe-group read (%s = table)
      filterAggSql: String, // the probe-filter read (%s = table)
      checksumSql: String // whole-table checksum aggregates (%s = table)
  ) {
    def scattered(pct: Int, offset: Int = 0): String =
      s"pmod(xxhash64($hashKey), 1000) >= ${offset * 10} AND pmod(xxhash64($hashKey), 1000) < ${(offset + pct) * 10}"
  }

  val LineitemProfile = TableProfile(
    "lineitem",
    "lineitem",
    "l_orderkey",
    pct => s"l_shipdate < date_add(DATE '1992-01-01', ${2557 * pct / 100})",
    Seq("l_orderkey", "l_linenumber"),
    "l_orderkey",
    "l_quantity = l_quantity + 1, l_comment = concat(l_comment, ' u')",
    "t.l_discount = s.l_discount, t.l_comment = concat(s.l_comment, ' m')",
    "l_comment",
    "l_orderkey",
    Seq("l_returnflag", "l_linestatus"),
    "l_quantity",
    Map(
      "l_quantity" -> "l_quantity + 1",
      "l_extendedprice" -> "round(l_extendedprice * 1.01, 2)",
      "l_comment" -> "concat(l_comment, ' u')"
    ),
    "t.l_quantity = s.l_quantity, t.l_extendedprice = s.l_extendedprice, t.l_discount = s.l_discount, t.l_comment = s.l_comment",
    """SELECT l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice),
      |  sum(l_extendedprice * (1 - l_discount)), avg(l_discount), count(*)
      |FROM %s GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus""".stripMargin,
    """SELECT sum(l_extendedprice * l_discount) FROM %s
      |WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
      |  AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24""".stripMargin,
    "SELECT count(*), sum(l_quantity), sum(l_extendedprice), sum(l_orderkey % 1000003) FROM %s"
  )

  val StoreSalesProfile = TableProfile(
    "store_sales",
    "store_sales",
    "ss_ticket_number",
    pct => s"ss_sold_date_sk < 2450816 + ${1823 * pct / 100}",
    Seq("ss_ticket_number", "ss_item_sk"),
    "ss_ticket_number",
    "ss_quantity = ss_quantity + 1",
    "t.ss_sales_price = s.ss_sales_price, t.ss_ext_sales_price = s.ss_ext_sales_price",
    "",
    "ss_ticket_number",
    Seq("ss_store_sk"),
    "ss_quantity",
    Map(
      "ss_quantity" -> "ss_quantity + 1",
      "ss_sales_price" -> "round(ss_sales_price * 1.01, 2)",
      "ss_ext_sales_price" -> "round(ss_ext_sales_price * 1.01, 2)"
    ),
    "t.ss_quantity = s.ss_quantity, t.ss_sales_price = s.ss_sales_price, t.ss_ext_sales_price = s.ss_ext_sales_price",
    """SELECT ss_store_sk, sum(ss_quantity), sum(ss_sales_price), sum(ss_ext_sales_price),
      |  avg(ss_sales_price), count(*)
      |FROM %s GROUP BY ss_store_sk ORDER BY ss_store_sk""".stripMargin,
    """SELECT sum(ss_ext_sales_price) FROM %s
      |WHERE ss_sold_date_sk BETWEEN 2451545 AND 2451910 AND ss_quantity < 24""".stripMargin,
    "SELECT count(*), sum(ss_quantity), sum(ss_sales_price), sum(ss_ticket_number % 1000003) FROM %s"
  )

  val Profiles: Map[String, TableProfile] =
    Seq(LineitemProfile, StoreSalesProfile).map(p => p.name -> p).toMap

  /**
   * Resolve a warehouse argument to what the catalog should be given. A path with a URI scheme
   * (`s3a://…`, `file:…`) is passed through unchanged so it can be an object store; a bare path is
   * made absolute so a `local[N]` run finds it whatever the working directory. Add the scheme test
   * here, not at the call sites, so the generator and the CDC runner agree.
   */
  def resolveWarehouse(warehouse: String): String =
    if (warehouse.contains("://") || warehouse.startsWith("file:")) warehouse
    else new java.io.File(warehouse).getAbsolutePath

  /**
   * Session configuration for a catalog under `warehouse` plus Iceberg's SQL extensions. A local
   * (schemeless) warehouse uses the default Hadoop `FileIO`; an `s3a://` warehouse uses Iceberg's
   * `S3FileIO` with the Analytics Accelerator stream (#249), so the cluster legs read S3 through the
   * same path the TPC-DS Iceberg campaign will.
   */
  def catalogConf(warehouse: String): Map[String, String] = {
    val resolved = resolveWarehouse(warehouse)
    val base = Map(
      "spark.sql.extensions" -> "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
      s"spark.sql.catalog.$Catalog" -> "org.apache.iceberg.spark.SparkCatalog",
      s"spark.sql.catalog.$Catalog.type" -> "hadoop",
      s"spark.sql.catalog.$Catalog.warehouse" -> resolved,
      // Equality deletes are committed outside Spark: never serve a stale Table.
      s"spark.sql.catalog.$Catalog.cache-enabled" -> "false"
    )
    if (resolved.startsWith("s3")) base ++ Map(
      s"spark.sql.catalog.$Catalog.io-impl" -> "org.apache.iceberg.aws.s3.S3FileIO",
      s"spark.sql.catalog.$Catalog.s3.analytics-accelerator.enabled" -> "true"
    )
    else base
  }

  val DefaultVariants: Seq[String] = Seq(
    "plain",
    "pos_2",
    "pos_10",
    "pos_30",
    "pos_10_clustered",
    "pos_30_clustered",
    "pos_upd_1",
    "pos_upd_5",
    "eq_2",
    "eq_10",
    "dv_2",
    "dv_10",
    "dv_30",
    "dv_10_clustered",
    "dv_30_clustered",
    "dv_upd_1",
    "dv_upd_5"
  )

  final case class Args(
      data: String = "benchmarks/data/sf1",
      warehouse: String = "benchmarks/data/iceberg",
      namespace: String = "sf1",
      baseTable: String = "lineitem",
      variants: Seq[String] = DefaultVariants,
      threads: Int = Runtime.getRuntime.availableProcessors(),
      /** Data files the base table is written as; deletes then span all of them. */
      files: Int = 16
  )

  private val Pattern = """^(plain|(pos|dv)_(\d+)(_clustered)?|(pos|dv)_upd_(\d+)|eq_(\d+))$""".r

  def main(argv: Array[String]): Unit = {
    def parse(rest: List[String], a: Args): Args = rest match {
      case Nil => a
      case "--data" :: v :: t => parse(t, a.copy(data = v))
      case "--warehouse" :: v :: t => parse(t, a.copy(warehouse = v))
      case "--namespace" :: v :: t => parse(t, a.copy(namespace = v))
      case "--base-table" :: v :: t => parse(t, a.copy(baseTable = v))
      case "--variants" :: v :: t => parse(t, a.copy(variants = v.split(",").map(_.trim).filter(_.nonEmpty).toSeq))
      case "--threads" :: v :: t => parse(t, a.copy(threads = v.toInt))
      case "--files" :: v :: t => parse(t, a.copy(files = v.toInt))
      case other :: _ => throw new IllegalArgumentException(s"unknown argument $other")
    }
    val args = parse(argv.toList, Args())
    args.variants.foreach(v => require(Pattern.matches(v), s"unknown variant $v (see IcebergMorGenerator)"))
    val profile = Profiles.getOrElse(
      args.baseTable,
      throw new IllegalArgumentException(
        s"unknown base table ${args.baseTable}; known: ${Profiles.keys.mkString(", ")}"
      )
    )
    // store_sales has no free-text column to append a readback marker to; its plain-update variant
    // relies on the numeric change alone. eq/upd variants that need one are guarded in build().
    // `data` may be a local dir or an object-store prefix (s3a://…); the relation is the child under it.
    val scheme = args.data.contains("://")
    val sourcePath =
      if (scheme) s"${args.data.stripSuffix("/")}/${profile.relation}"
      else new File(args.data, profile.relation).getPath
    if (!scheme) {
      require(new File(sourcePath).isDirectory, s"$sourcePath is not a directory (generate the base table first)")
      Files.createDirectories(Paths.get(args.warehouse))
    }
    val warehouse = catalogConf(args.warehouse)(s"spark.sql.catalog.$Catalog.warehouse")
    // On the cluster spark-submit sets --master; locally there is none, so default to local[threads].
    val builder = SparkSession.builder()
      .appName("spark-vector-iceberg-mor-generator")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", args.threads.toString)
    if (Option(System.getProperty("spark.master")).isEmpty && sys.env.get("SPARK_MASTER").isEmpty)
      builder.master(s"local[${args.threads}]").config("spark.driver.host", "localhost")
    catalogConf(args.warehouse).foreach { case (k, v) => builder.config(k, v) }
    val spark = builder.getOrCreate()
    try {
      val ns = s"$Catalog.${args.namespace}"
      spark.sql(s"CREATE NAMESPACE IF NOT EXISTS $ns")
      val source = spark.read.parquet(sourcePath)
      val sourceRows = source.count()
      println(s"[mor] source $sourcePath (${profile.name}): $sourceRows rows, ${source.schema.fields.length} columns")
      val summaries = args.variants.map { v =>
        val start = System.nanoTime()
        val s = build(spark, source, s"$ns.$v", v, args.files, profile)
        println(
          f"[mor] $v: live=${s.liveRows} data files=${s.dataFiles} delete files=${s.deleteFiles} (${s.deleteFormats}) " +
            f"delete rows=${s.deleteRows} (${s.deleteRowsPerDataFile}%.0f per data file) snapshot=${s.snapshotId} in ${(System.nanoTime() - start) / 1e9}%.0fs"
        )
        s
      }
      val readmeText = readmeOf(args, sourceRows, summaries)
      if (scheme) {
        // No local warehouse dir to write next to; print the summary so the run log carries it.
        println(s"[mor] warehouse summary (${args.namespace}):\n$readmeText")
      } else {
        val readme = Paths.get(args.warehouse, s"README-${args.namespace}.md")
        Files.writeString(readme, readmeText, StandardCharsets.UTF_8)
        println(s"[mor] wrote $readme")
      }
    } finally spark.stop()
  }

  final case class Summary(
      variant: String,
      formatVersion: Int,
      liveRows: Long,
      dataFiles: Long,
      deleteFiles: Long,
      deleteFormats: String,
      deleteRows: Long,
      snapshotId: Long,
      statements: Seq[String]
  ) {
    def deleteRowsPerDataFile: Double = if (dataFiles == 0) 0 else deleteRows.toDouble / dataFiles
  }

  private def build(
      spark: SparkSession,
      source: DataFrame,
      name: String,
      variant: String,
      files: Int,
      p: TableProfile
  ): Summary = {
    val statements = ArrayBuffer.empty[String]
    def sql(s: String): Unit = { statements += s; spark.sql(s) }
    val Pattern(_, kind, pct, clusteredFlag, updKind, updPct, eqPct) = variant
    val formatVersion = if (kind == "dv" || updKind == "dv") 3 else 2
    // The base table: the Parquet rows as `files` data files, merge-on-read for every mutation kind.
    val props = Seq(
      "format-version" -> formatVersion.toString,
      "write.delete.mode" -> "merge-on-read",
      "write.update.mode" -> "merge-on-read",
      "write.merge.mode" -> "merge-on-read",
      "write.target-file-size-bytes" -> (512L * 1024 * 1024).toString
    )
    val writer = source.repartition(files).writeTo(name).using("iceberg")
    props.foldLeft(writer) { case (w, (k, v)) => w.tableProperty(k, v) }.createOrReplace()
    statements += s"CREATE OR REPLACE TABLE $name USING iceberg TBLPROPERTIES (${props.map { case (k, v) =>
        s"'$k'='$v'"
      }.mkString(", ")}) AS SELECT * FROM ${p.name}  -- repartition($files)"
    variant match {
      case "plain" =>
      case _ if kind != null =>
        val cond = if (clusteredFlag != null) p.clusteredExpr(pct.toInt) else p.scattered(pct.toInt)
        sql(s"DELETE FROM $name WHERE $cond")
      case _ if updKind != null =>
        // pos_10 / dv_10, then the update of a further pct % of the live rows, then the merge.
        sql(s"DELETE FROM $name WHERE ${p.scattered(10)}")
        sql(s"UPDATE $name SET ${p.updateSet} WHERE ${p.scattered(updPct.toInt, offset = 10)}")
        // The merge source: 1 % of the live rows come back as updates, and another 1 % as brand-new
        // rows re-keyed above every existing row of the re-key column.
        val maxKey = spark.table(name).selectExpr(s"max(${p.reKey})").collect()(0).getAs[Number](0).longValue()
        spark.table(name).createOrReplaceTempView("mor_base")
        spark.sql(s"SELECT * FROM mor_base WHERE ${p.scattered(1, offset = 90)} " +
          s"UNION ALL SELECT ${p.reKey} + ${maxKey}L AS ${p.reKey}, * EXCEPT (${p.reKey}) FROM mor_base WHERE ${p.scattered(1, offset = 80)}")
          .createOrReplaceTempView("mor_src")
        val on = p.grain.map(g => s"t.$g = s.$g").mkString(" AND ")
        // Delete the rows the merge names (a stable slice of the matched rows, by the last grain col's parity).
        val delPred = s"pmod(s.${p.grain.last}, 10) = 1"
        sql(s"""MERGE INTO $name t USING mor_src s ON $on
               |WHEN MATCHED AND $delPred THEN DELETE
               |WHEN MATCHED THEN UPDATE SET ${p.mergeUpdateSet}
               |WHEN NOT MATCHED THEN INSERT *""".stripMargin)
      case _ =>
        // Equality deletes on the eq-key through the Java API: every key value whose hash falls in
        // the first pct %, in files of 20000 keys.
        val keys = spark.table(name).where(p.scattered(eqPct.toInt)).select(p.eqKey).distinct().orderBy(p.eqKey)
          .collect().map(_.getAs[Number](0).longValue())
        addEqualityDeletes(spark, name, p.eqKey, keys.toSeq, 20000)
        statements += s"-- equality delete files on ${p.eqKey} for the ${keys.length} keys WHERE ${p.scattered(eqPct.toInt)} (Java API, ${(keys.length + 19999) / 20000} files of <= 20000 keys)"
    }
    summarize(spark, name, variant, formatVersion, statements.toSeq)
  }

  /**
   * Writes Parquet equality delete files on `eqKey` with the Iceberg Java API and commits them as
   * one row delta, the way a streaming CDC writer does. The deletes get a sequence number above
   * every existing data file, so they apply to all of them.
   */
  def addEqualityDeletes(spark: SparkSession, name: String, eqKey: String, keys: Seq[Long], perFile: Int): Unit = {
    val table: Table = Spark3Util.loadIcebergTable(spark, name)
    val schema = table.schema()
    val keyField = schema.findField(eqKey)
    val deleteSchema = schema.select(eqKey)
    val isLong = keyField.`type`().typeId() == org.apache.iceberg.types.Type.TypeID.LONG
    val factory = new GenericAppenderFactory(schema, table.spec(), Array(keyField.fieldId()), deleteSchema, null)
    val delta = table.newRowDelta()
    keys.grouped(perFile).zipWithIndex.foreach { case (chunk, i) =>
      val outputFile = OutputFileFactory.builderFor(table, 1, i + 1).format(FileFormat.PARQUET).build().newOutputFile()
      val writer = factory.newEqDeleteWriter(outputFile, FileFormat.PARQUET, null)
      try {
        chunk.foreach { k =>
          val record = GenericRecord.create(deleteSchema)
          record.setField(eqKey, if (isLong) Long.box(k) else Int.box(k.toInt))
          writer.write(record)
        }
      } finally writer.close()
      delta.addDeletes(writer.toDeleteFile())
    }
    delta.commit()
  }

  private def summarize(
      spark: SparkSession,
      name: String,
      variant: String,
      formatVersion: Int,
      statements: Seq[String]
  ): Summary = {
    val live = spark.table(name).count()
    val files = spark.sql(
      s"SELECT content, file_format, count(*) AS n, sum(record_count) AS rows FROM $name.files GROUP BY content, file_format"
    ).collect()
    val dataFiles = files.filter(_.getInt(0) == 0).map(_.getLong(2)).sum
    val deletes = files.filter(_.getInt(0) > 0)
    val deleteFiles = deletes.map(_.getLong(2)).sum
    val deleteRows = deletes.map(_.getLong(3)).sum
    val formats = deletes.map(r =>
      s"${if (r.getInt(0) == 1) "position" else "equality"}/${r.getString(1).toLowerCase}"
    ).distinct.sorted.mkString(", ")
    val snapshot =
      spark.sql(s"SELECT snapshot_id FROM $name.snapshots ORDER BY committed_at DESC LIMIT 1").collect()(0).getLong(0)
    Summary(
      variant,
      formatVersion,
      live,
      dataFiles,
      deleteFiles,
      if (formats.isEmpty) "none" else formats,
      deleteRows,
      snapshot,
      statements
    )
  }

  private def readmeOf(args: Args, sourceRows: Long, summaries: Seq[Summary]): String = {
    val sb = new StringBuilder
    sb.append(s"# Iceberg merge-on-read variants of `lineitem` (`${args.namespace}`)\n\n")
    sb.append(
      s"Source: `${args.data}/${args.baseTable}` ($sourceRows rows), written as ${args.files} data files per table into the Hadoop catalog\n"
    )
    sb.append(
      s"`${Catalog}` at `${args.warehouse}` (tables `$Catalog.${args.namespace}.<variant>`). Live rows are `count(*)` through Spark's row\n"
    )
    sb.append(
      "path with no plugin loaded -- the oracle every configuration of the harness is compared to. Delete rows are\n"
    )
    sb.append(
      "the `record_count` of the delete files (an equality delete row removes every line of its order; a positional\n"
    )
    sb.append(
      "delete or deletion-vector row removes one). The harness reads the current snapshot; the id is the one to pin\n"
    )
    sb.append("with `VERSION AS OF` if the table is mutated again.\n\n")
    sb.append(
      "| variant | format | live rows | rows gone vs source (%) | data files | delete files | delete rows | per data file | snapshot |\n"
    )
    sb.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    summaries.foreach { s =>
      val deletedPct = 100.0 * (sourceRows - s.liveRows) / sourceRows
      sb.append(
        f"| `${s.variant}` | v${s.formatVersion} | ${s.liveRows} | $deletedPct%.1f | ${s.dataFiles} | ${s.deleteFiles} (${s.deleteFormats}) | ${s.deleteRows} | ${s.deleteRowsPerDataFile}%.0f | ${s.snapshotId} |\n"
      )
    }
    sb.append("\n## How each variant was produced\n")
    summaries.foreach { s =>
      sb.append(s"\n### `${s.variant}`\n\n```sql\n${s.statements.mkString(";\n")}\n```\n")
    }
    sb.toString
  }
}
