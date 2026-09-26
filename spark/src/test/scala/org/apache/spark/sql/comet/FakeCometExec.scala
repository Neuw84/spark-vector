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
package org.apache.spark.sql.comet

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan, UnaryExecNode}

/**
 * Stand-ins for Comet operators, used to test that the UI colours them as Comet.
 *
 * They live in `org.apache.spark.sql.comet` on purpose: that package is exactly what
 * `PlanAcceleration` matches on, the same convention `CometShuffle` uses reflectively. Comet itself
 * cannot be exercised on every platform (the published jar only bundles Linux natives), so this
 * pins the classification rule without needing Comet installed.
 */
case class FakeCometScanExec(override val output: Seq[Attribute]) extends LeafExecNode {
  override def supportsColumnar: Boolean = true
  override protected def doExecute(): RDD[InternalRow] = throw new UnsupportedOperationException
}

case class FakeCometShuffleExchangeExec(child: SparkPlan) extends UnaryExecNode {
  override def output: Seq[Attribute] = child.output
  override def supportsColumnar: Boolean = true
  override protected def doExecute(): RDD[InternalRow] = throw new UnsupportedOperationException
  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/** Stand-in for Comet's JVM shuffle, which reads its child as rows; the UI flags that in the tooltip. */
case class FakeCometColumnarExchangeExec(child: SparkPlan) extends UnaryExecNode {
  override def nodeName: String = "CometColumnarExchange"
  override def output: Seq[Attribute] = child.output
  override def supportsColumnar: Boolean = true
  override protected def doExecute(): RDD[InternalRow] = throw new UnsupportedOperationException
  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}
