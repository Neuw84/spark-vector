/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package io.vecruntime.spark.test

import org.apache.spark.sql.{DataFrame, SparkSession}

/** Shared synthetic tables written as Parquet and registered as temp views. */
object TestTables {

  /**
   * `t`: 20k rows, several Parquet batches, every supported type, nulls in `l`, `d`, `s`, NaN and
   * infinities in `d`, and a second double column `d2` for column-vs-column comparisons.
   */
  def createMixed(spark: SparkSession, path: String, rows: Int = 20000): Unit = {
    mixedDataFrame(spark, rows)
      .repartition(3)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("t")
  }

  /** The rows of `t` as a DataFrame, for writers other than Parquet-on-path (Iceberg). */
  def mixedDataFrame(spark: SparkSession, rows: Int = 20000): DataFrame =
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
        "if(id % 10 = 0, null, concat('s', id % 50)) as s"
      )

  /**
   * `lineitem`: a TPC-H shaped table with decimals replaced by doubles (the benchmark variant we
   * target). Distributions loosely follow dbgen so Q1/Q6 predicates have realistic selectivity.
   */
  def createLineitem(spark: SparkSession, path: String, rows: Int = 60000): Unit = {
    lineitemDataFrame(spark, rows)
      .repartition(4)
      .write
      .mode("overwrite")
      .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView("lineitem")
  }

  /** The rows of `lineitem` as a DataFrame. */
  def lineitemDataFrame(spark: SparkSession, rows: Int = 60000): DataFrame =
    spark
      .range(0, rows)
      .selectExpr(
        "cast(id / 4 as bigint) as l_orderkey",
        "cast(id % 200000 as bigint) as l_partkey",
        "cast(id % 10000 as bigint) as l_suppkey",
        "cast(id % 7 + 1 as int) as l_linenumber",
        "cast(pmod(id * 7919, 50) + 1 as double) as l_quantity",
        "cast(round(900.0 + pmod(id * 104729, 95000) + pmod(id, 100) / 100.0, 2) as double) as l_extendedprice",
        "cast(pmod(id * 31, 11) as double) / 100.0 as l_discount",
        "cast(pmod(id * 17, 9) as double) / 100.0 as l_tax",
        "case when pmod(id, 4) = 0 then 'A' when pmod(id, 4) = 1 then 'R' else 'N' end as l_returnflag",
        "case when pmod(id, 4) <= 1 then 'F' else 'O' end as l_linestatus",
        "date_add(date '1992-01-01', cast(pmod(id * 2654435761, 2557) as int)) as l_shipdate",
        "date_add(date '1992-01-01', cast(pmod(id * 40503, 2557) as int)) as l_commitdate",
        "date_add(date '1992-01-01', cast(pmod(id * 69069, 2557) as int)) as l_receiptdate",
        "concat('cmt', id % 97) as l_comment"
      )

  val TpchQ6: String =
    """SELECT sum(l_extendedprice * l_discount) AS revenue
      |FROM lineitem
      |WHERE l_shipdate >= DATE '1994-01-01'
      |  AND l_shipdate < DATE '1995-01-01'
      |  AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
      |  AND l_quantity < 24""".stripMargin

  val TpchQ1: String =
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
}
