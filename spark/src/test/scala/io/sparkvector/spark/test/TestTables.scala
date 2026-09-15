package io.sparkvector.spark.test

import org.apache.spark.sql.SparkSession

/** Shared synthetic tables written as Parquet and registered as temp views. */
object TestTables {

  /**
   * `t`: 20k rows, several Parquet batches, every supported type, nulls in `l`, `d`, `s`, NaN and
   * infinities in `d`, and a second double column `d2` for column-vs-column comparisons.
   */
  def createMixed(spark: SparkSession, path: String, rows: Int = 20000): Unit = {
    spark
      .range(0, rows)
      .selectExpr(
        "cast(id as int) as i",
        "if(id % 7 = 3, null, id * 3) as l",
        "case when id % 97 = 5 then cast('NaN' as double) " +
          "     when id % 97 = 6 then cast('Infinity' as double) " +
          "     when id % 97 = 7 then cast('-Infinity' as double) " +
          "     when id % 11 = 0 then null " +
          "     else (cast(id as double) % 1000) / 7 end as d",
        "cast(id % 13 as double) / 4 as d2",
        "date_add(date '2020-01-01', cast(id % 730 as int)) as dt",
        "id % 3 = 0 as b",
        "if(id % 10 = 0, null, concat('s', id % 50)) as s")
      .repartition(3)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("t")
  }
}
