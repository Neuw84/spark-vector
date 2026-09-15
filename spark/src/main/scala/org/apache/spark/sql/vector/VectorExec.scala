package org.apache.spark.sql.vector

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}

/** Common plumbing for spark-vector's columnar-only physical operators. */
trait VectorExec extends UnaryExecNode {

  override def supportsColumnar: Boolean = true

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "time" -> SQLMetrics.createNanoTimingMetric(sparkContext, "time in spark-vector kernels"))

  protected def vectorMetrics: VectorMetrics = new VectorMetrics(
    longMetric("numInputBatches"),
    longMetric("numOutputBatches"),
    longMetric("numOutputRows"),
    longMetric("time"))

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException(
      s"${nodeName} only supports columnar execution; Spark should have inserted ColumnarToRowExec")

  override def outputPartitioning: org.apache.spark.sql.catalyst.plans.physical.Partitioning =
    child.outputPartitioning
}
