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
package io.vecruntime.benchmarks

import java.nio.charset.StandardCharsets

import io.vecruntime.benchmarks.TpchRunner.Suite

/**
 * TPC-DS runner: the 99 queries (103 with the a/b variants) over the 24 tables `gen-tpcds.sh`
 * writes, sharing every measurement and report facility of [[TpchRunner]]. The query text is
 * Spark's own -- `tpcds/q*.sql` from the spark-sql tests jar, the files its plan-stability suite
 * runs -- so the plans measured here are the ones Spark's optimizer is tested against, and the
 * per-query issues can quote the approved plans.
 *
 * {{{
 * TpcdsRunner --config vector --data benchmarks/data/tpcds-sf1 --out benchmarks/results/tpcds
 * TpcdsRunner --config vector --data benchmarks/data/tpcds-sf1 --queries q10,q35,q45
 * TpcdsRunner --report benchmarks/results/tpcds
 * }}}
 */
object TpcdsRunner {

  /** The 24 TPC-DS tables, each a Parquet directory of that name under `--data` (see `gen-tpcds.sh`). */
  val Tables: Seq[String] = Seq(
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site"
  )

  /** `q1`..`q99`, with the a/b variants of 14, 23, 24 and 39: the 103 files Spark ships. */
  val QueryNames: Seq[String] = (1 to 99).flatMap {
    case n @ (14 | 23 | 24 | 39) => Seq(s"q${n}a", s"q${n}b")
    case n => Seq(s"q$n")
  }

  /** One query's text from the spark-sql tests jar (`tpcds/<name>.sql`). */
  def query(name: String): String = {
    val path = s"tpcds/$name.sql"
    val in = Thread.currentThread.getContextClassLoader.getResourceAsStream(path)
    require(
      in != null,
      s"$path not on the classpath: the spark-sql tests jar (classifier `tests`) is missing (see benchmarks/pom.xml)"
    )
    try new String(in.readAllBytes(), StandardCharsets.UTF_8)
    finally in.close()
  }

  lazy val Queries: Seq[(String, String)] = QueryNames.map(n => n -> query(n))

  lazy val Tpcds: Suite = Suite("tpcds", "TPC-DS", Tables, "store_sales", Queries)

  def main(argv: Array[String]): Unit = TpchRunner.mainWith(Tpcds, argv)
}
