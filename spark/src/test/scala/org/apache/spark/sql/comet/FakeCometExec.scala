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
