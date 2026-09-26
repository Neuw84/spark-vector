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
package io.sparkvector.spark

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.vector.{VectorColumnarRule, VectorWriteDeltaStrategy}

/**
 * Registers the planner rule. Enable with
 * `--conf spark.sql.extensions=io.sparkvector.spark.VectorSparkSessionExtensions`, or let
 * [[VectorPlugin]] do it through `--conf spark.plugins=io.sparkvector.spark.VectorPlugin`.
 */
class VectorSparkSessionExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectColumnar(session => VectorColumnarRule(session))
    // The columnar v3 deletion-vector writer (#20): a planner strategy over the logical WriteDelta,
    // gated on spark.vector.iceberg.dvWriter.enabled and a v3 target; declines to Spark otherwise.
    extensions.injectPlannerStrategy(session => VectorWriteDeltaStrategy(session))
  }
}
