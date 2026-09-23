package io.sparkvector.spark.iceberg

import io.sparkvector.spark.test.TestTables
import org.apache.iceberg.{FileFormat, Table}
import org.apache.iceberg.data.{GenericAppenderFactory, GenericRecord}
import org.apache.iceberg.io.OutputFileFactory
import org.apache.iceberg.spark.Spark3Util
import org.apache.spark.sql.SparkSession

/**
 * Iceberg merge-on-read tables for the Iceberg suites. Every table is written through Spark's
 * Iceberg catalog and then mutated so the snapshot read has to merge deletes:
 *
 *  - `t_pos` (v2): positional delete files from `DELETE`, plus an `UPDATE` (delete + insert);
 *  - `t_dv` (v3): the same mutations, encoded as deletion vectors (Puffin);
 *  - `t_eq` (v2): an equality delete file written with the Iceberg Java API, on top of the
 *    positional deletes, so both delete kinds apply to the same data files;
 *  - `lineitem` (v2): the TPC-H shaped table with ~2% of the rows deleted positionally.
 *
 * The temp views `t` and `lineitem` are pointed at whichever table a test asks for.
 */
object IcebergTables {

  val Catalog = "ice"
  val Db = s"$Catalog.db"

  /** Session configuration for a Hadoop catalog under `warehouse` plus Iceberg's SQL extensions. */
  def catalogConf(warehouse: String): Map[String, String] = Map(
    "spark.sql.extensions" -> "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    s"spark.sql.catalog.$Catalog" -> "org.apache.iceberg.spark.SparkCatalog",
    s"spark.sql.catalog.$Catalog.type" -> "hadoop",
    s"spark.sql.catalog.$Catalog.warehouse" -> warehouse,
    // Tables are also committed to outside Spark (equality deletes); never serve a stale Table.
    s"spark.sql.catalog.$Catalog.cache-enabled" -> "false"
  )

  val MixedRows = 20000
  val LineitemRows = 60000

  /** Keys removed by the equality delete file: every 23rd row plus a few explicit ones. */
  val EqDeletedKeys: Set[Int] = (0 until MixedRows).filter(_ % 23 == 4).toSet ++ Set(1, 2, 3, 19999)

  /** Rows of `t_pos`/`t_dv`/`t_eq` that survive the mutations applied by [[createMixedMor]]. */
  val MixedLiveRows: Int = (0 until MixedRows).count(i => i % 5 != 0)

  /** Rows of `t_eq` that survive the positional and the equality deletes. */
  val EqLiveRows: Int = (0 until MixedRows).count(i => i % 5 != 0 && !EqDeletedKeys.contains(i))

  private def morProps(formatVersion: Int): Seq[(String, String)] = Seq(
    "format-version" -> formatVersion.toString,
    "write.delete.mode" -> "merge-on-read",
    "write.update.mode" -> "merge-on-read",
    "write.merge.mode" -> "merge-on-read",
    // Several data files so deletes span files and the reader has to match them per file.
    "write.target-file-size-bytes" -> (256 * 1024).toString
  )

  def createAll(spark: SparkSession): Unit = {
    spark.sql(s"CREATE NAMESPACE IF NOT EXISTS $Db")
    createMixedMor(spark, s"$Db.t_pos", formatVersion = 2)
    createMixedMor(spark, s"$Db.t_dv", formatVersion = 3)
    createMixedMor(spark, s"$Db.t_eq", formatVersion = 2)
    addEqualityDeletes(spark, s"$Db.t_eq", EqDeletedKeys.toSeq.sorted)
    createLineitemMor(spark, s"$Db.lineitem")
  }

  /**
   * A v2 table with many merge-on-read rounds on top of [[createMixedMor]]: `rounds` further
   * DELETE/UPDATE pairs, each touching rows spread over every data file, so most files carry
   * several positional delete files and the update-inserted rows are scattered across many small
   * new files. Two calls with the same arguments produce identical table contents, which lets a
   * test mutate one copy with the plugin on and one with it off and compare them.
   */
  def createHeavilyMutated(spark: SparkSession, name: String, rounds: Int = 6): Unit = {
    createMixedMor(spark, name, formatVersion = 2)
    (1 to rounds).foreach { r =>
      spark.sql(s"DELETE FROM $name WHERE pmod(i, 97) = $r")
      spark.sql(s"UPDATE $name SET d2 = d2 + 0.5, l = l + 1 WHERE pmod(i, 89) = $r")
    }
  }

  /** Keys of `t` rows the merge source updates (`d2 >= 0`) or deletes (`d2 < 0`). */
  val MergeUpdatedKeys: Seq[Int] = (0 until MixedRows).filter(_ % 13 == 2)
  val MergeDeletedKeys: Set[Int] = MergeUpdatedKeys.filter(_ % 101 == 0).toSet
  val MergeInsertedKeys: Range = MixedRows until MixedRows + 2000

  /**
   * Registers the temp view `src`: an incoming batch with the target's schema, holding new
   * versions of existing rows (some flagged for deletion with a negative `d2`) and brand-new rows.
   */
  def createMergeSource(spark: SparkSession): Unit = {
    import spark.implicits._
    val keys = (MergeUpdatedKeys ++ MergeInsertedKeys).toDF("k")
    keys
      .selectExpr(
        "cast(k as int) as i",
        "cast(k * 5 as bigint) as l",
        "cast(k as double) / 3 as d",
        s"case when k < $MixedRows and k % 13 = 2 and k % 101 = 0 then -1.0 else 10.0 + cast(k % 7 as double) end as d2",
        "date_add(date '2021-06-01', cast(k % 100 as int)) as dt",
        "k % 2 = 0 as b",
        "if(k % 15 = 0, null, concat('m', k % 5)) as s"
      )
      .createOrReplaceTempView("src")
  }

  /** The MERGE INTO statement the suites run against `target`, with `src` as the source. */
  def mergeSql(target: String): String =
    s"""MERGE INTO $target t USING src s ON t.i = s.i
       |WHEN MATCHED AND s.d2 < 0 THEN DELETE
       |WHEN MATCHED THEN UPDATE SET t.l = s.l, t.d2 = s.d2, t.s = s.s
       |WHEN NOT MATCHED THEN INSERT *""".stripMargin

  /** Points the temp view `t` at `table`. */
  def useAsT(spark: SparkSession, table: String): Unit =
    spark.table(table).createOrReplaceTempView("t")

  def useAsLineitem(spark: SparkSession, table: String): Unit =
    spark.table(table).createOrReplaceTempView("lineitem")

  def createMixedMor(spark: SparkSession, name: String, formatVersion: Int): Unit = {
    val writer = TestTables.mixedDataFrame(spark, MixedRows).repartition(3).writeTo(name).using("iceberg")
    morProps(formatVersion).foldLeft(writer) { case (w, (k, v)) => w.tableProperty(k, v) }.createOrReplace()
    // Positional deletes (v2) or deletion vectors (v3) against the initial data files.
    spark.sql(s"DELETE FROM $name WHERE i % 5 = 0")
    // An update is a delete of the old row image plus an insert into a new data file.
    spark.sql(s"UPDATE $name SET d2 = d2 + 1.0, s = concat(s, '_u') WHERE i % 17 = 1")
  }

  def createLineitemMor(spark: SparkSession, name: String): Unit = {
    val writer = TestTables.lineitemDataFrame(spark, LineitemRows).repartition(4).writeTo(name).using("iceberg")
    morProps(2).foldLeft(writer) { case (w, (k, v)) => w.tableProperty(k, v) }.createOrReplace()
    spark.sql(s"DELETE FROM $name WHERE pmod(l_orderkey, 50) = 7")
  }

  /**
   * Writes one Parquet equality delete file on column `i` with the Iceberg Java API and commits
   * it as a row delta, the way a streaming CDC writer would. The delete gets a sequence number
   * above every existing data file, so it applies to all of them.
   */
  def addEqualityDeletes(spark: SparkSession, name: String, keys: Seq[Int]): Unit = {
    val table: Table = Spark3Util.loadIcebergTable(spark, name)
    val schema = table.schema()
    val keyField = schema.findField("i")
    val deleteSchema = schema.select("i")
    val factory = new GenericAppenderFactory(
      schema,
      table.spec(),
      Array(keyField.fieldId()),
      deleteSchema,
      null
    )
    val outputFile = OutputFileFactory.builderFor(table, 1, 1).format(FileFormat.PARQUET).build().newOutputFile()
    val writer = factory.newEqDeleteWriter(outputFile, FileFormat.PARQUET, null)
    try {
      keys.foreach { k =>
        val record = GenericRecord.create(deleteSchema)
        record.setField("i", Int.box(k))
        writer.write(record)
      }
    } finally {
      writer.close()
    }
    table.newRowDelta().addDeletes(writer.toDeleteFile()).commit()
  }

  /** Number of delete files (positional, equality or deletion vector) in the current snapshot. */
  def deleteFileCount(spark: SparkSession, name: String): Long =
    spark.sql(s"SELECT count(*) FROM $name.files WHERE content > 0").collect()(0).getLong(0)
}
