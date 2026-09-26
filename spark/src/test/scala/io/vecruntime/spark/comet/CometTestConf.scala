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
package io.vecruntime.spark.comet

import io.vecruntime.spark.VectorPlugin

/** Session configuration shared by the Comet-backed suites. */
object CometTestConf {

  /**
   * Comet as the scan (and only the scan): both plugins registered, Comet first so its scan rule
   * runs first; every Comet operator switched off so Spark's shuffle and our operators are used.
   * Comet 1.0's scans are native DataFusion / iceberg-rust readers, which require exec enabled.
   */
  val scanOnly: Map[String, String] = Map(
    "spark.plugins" -> s"org.apache.spark.CometPlugin,${classOf[VectorPlugin].getName}",
    "spark.comet.enabled" -> "true",
    "spark.comet.scan.enabled" -> "true",
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
    "spark.memory.offHeap.size" -> "1g",
    "spark.comet.explainFallback.enabled" -> "false",
    "spark.sql.parquet.enableVectorizedReader" -> "true",
    "spark.sql.adaptive.enabled" -> "true"
  )
}
