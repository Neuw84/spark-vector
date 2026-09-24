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
package org.apache.spark.sql.vector

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.{BinaryExecNode, ColumnarToRowExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}

/**
 * Common plumbing for spark-vector's columnar-only physical operators, whatever their arity. The
 * planner rule and the UI identify our operators by this trait.
 */
trait VectorPlan extends SparkPlan {

  override def supportsColumnar: Boolean = true

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "time" -> SQLMetrics.createNanoTimingMetric(sparkContext, "time in spark-vector kernels")
  )

  protected def vectorMetrics: VectorMetrics = new VectorMetrics(
    longMetric("numInputBatches"),
    longMetric("numOutputBatches"),
    longMetric("numOutputRows"),
    longMetric("time")
  )

  /**
   * Row-based execution, for consumers that call `execute()` on a columnar child without going
   * through Spark's transition insertion (Comet's columnar shuffle does). Converts our batches
   * row by row exactly like an inserted `ColumnarToRowExec` would.
   */
  override protected def doExecute(): RDD[InternalRow] = ColumnarToRowExec(this).doExecute()
}

/** A spark-vector operator with one child (filter, project, aggregate, sort). */
trait VectorExec extends UnaryExecNode with VectorPlan

/** A spark-vector operator with two children (joins). */
trait VectorBinaryExec extends BinaryExecNode with VectorPlan

/**
 * A spark-vector operator that forwards its children's batches unchanged (union, coalesce). It
 * evaluates nothing, so a selection a child filter forwarded would reach whatever sits above --
 * possibly a Spark operator that cannot read one; the rule therefore does not mark the children of
 * a pass-through operator as selection producers and they compact instead.
 */
trait VectorPassThrough extends VectorPlan
