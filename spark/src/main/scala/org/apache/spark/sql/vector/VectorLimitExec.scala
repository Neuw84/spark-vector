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

import java.lang.foreign.MemorySegment

import io.sparkvector.kernels.Bitmap
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.ArrowOutput
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{
  AllTuples,
  Distribution,
  Partitioning,
  SinglePartition,
  UnspecifiedDistribution
}
import org.apache.spark.sql.catalyst.types.DataTypeUtils
import org.apache.spark.sql.execution.{CollectLimitExec, GlobalLimitExec, LocalLimitExec, SparkPlan}
import org.apache.spark.sql.execution.metric.{
  SQLMetric,
  SQLMetrics,
  SQLShuffleReadMetricsReporter,
  SQLShuffleWriteMetricsReporter
}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Cuts a partition after `limit` live rows. Whole batches pass through untouched (owned by the
 * child); the batch that crosses the boundary is compacted to its first surviving rows, honouring a
 * forwarded selection; once the limit is reached no further batch is pulled from the child, whose
 * arena / allocator release runs from its task-completion listener as for every abandoned columnar
 * iterator in this project.
 */
private[vector] class VectorLimitIterator(
    input: Iterator[ColumnarBatch],
    limit: Int,
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics
) extends VectorBatchIterator(input, "VectorLimitExec") {

  private var remaining: Int = limit

  override protected def exhausted: Boolean = remaining <= 0

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val live = if (ctx.hasSelection) ctx.selectedCount else ctx.numRows
      if (live == 0) {
        null
      } else if (live <= remaining && !ctx.hasSelection) {
        remaining -= live
        metrics.numOutputBatches += 1
        metrics.numOutputRows += live
        batch
      } else {
        // Either a boundary batch (keep the first `remaining` live rows) or a selected batch that
        // has to be compacted anyway (the consumer above may not understand selections).
        val keep = math.min(live, remaining)
        val selection = VectorLimitIterator.head(ctx.selection, ctx.numRows, keep, ctx)
        val columns = new Array[ColumnVector](outputAttrs.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), selection, keep, allocator)
          c += 1
        }
        remaining -= keep
        metrics.numOutputBatches += 1
        metrics.numOutputRows += keep
        new ColumnarBatch(columns, keep)
      }
    }
  }
}

private[vector] object VectorLimitIterator {

  /** A selection with only the first `keep` live rows of `selection` (every row when it is null). */
  def head(
      selection: MemorySegment,
      numRows: Int,
      keep: Int,
      ctx: io.sparkvector.spark.expr.EvalContext
  ): MemorySegment = {
    val out = ctx.bitmap()
    Bitmap.fill(out, numRows, false)
    var kept = 0
    var i = 0
    while (i < numRows && kept < keep) {
      if (selection == null || Bitmap.isSet(selection, i)) {
        Bitmap.set(out, i)
        kept += 1
      }
      i += 1
    }
    out
  }
}

/** Columnar replacement for LocalLimitExec: the first `limit` rows of every partition. */
case class VectorLocalLimitExec(limit: Int, child: SparkPlan) extends VectorExec {
  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val n = limit
    val attrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal(iter => new VectorLimitIterator(iter, n, attrs, m))
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/**
 * Columnar replacement for GlobalLimitExec (no offset): the first `limit` rows of the single
 * partition Spark's distribution requirement guarantees.
 */
case class VectorGlobalLimitExec(limit: Int, child: SparkPlan) extends VectorExec {
  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def requiredChildDistribution: Seq[Distribution] = AllTuples :: Nil

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val n = limit
    val attrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal(iter => new VectorLimitIterator(iter, n, attrs, m))
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/**
 * Columnar replacement for CollectLimitExec (no offset): the first `limit` rows of every partition
 * stay columnar; those at most `limit` rows per partition go as rows through Spark's own
 * single-partition shuffle, the first `limit` of them are taken there and re-materialised as one
 * columnar batch -- the same two stages as Spark's operator and as [[VectorTakeOrderedAndProjectExec]].
 */
case class VectorCollectLimitExec(limit: Int, child: SparkPlan) extends VectorExec {
  override def output: Seq[Attribute] = child.output
  override def outputPartitioning: Partitioning = SinglePartition
  override def requiredChildDistribution: Seq[Distribution] = UnspecifiedDistribution :: Nil

  private lazy val writeMetrics = SQLShuffleWriteMetricsReporter.createShuffleWriteMetrics(sparkContext)
  private lazy val readMetrics = SQLShuffleReadMetricsReporter.createShuffleReadMetrics(sparkContext)
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "time" -> SQLMetrics.createNanoTimingMetric(sparkContext, "time in spark-vector kernels")
  ) ++ readMetrics ++ writeMetrics

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val n = limit
    val childOutput = child.output
    val attrs = childOutput.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    val schema = DataTypeUtils.fromAttributes(output)
    val numOutputBatches = longMetric("numOutputBatches")
    val numOutputRows = longMetric("numOutputRows")
    if (n == 0) return sparkContext.emptyRDD[ColumnarBatch]
    val local = child.executeColumnar().mapPartitionsInternal { iter =>
      VectorRowStages.toUnsafeRows(new VectorLimitIterator(iter, n, attrs, m), childOutput)
    }
    VectorRowStages.singlePartition(local, childOutput, writeMetrics, readMetrics).mapPartitionsInternal { iter =>
      val rows = iter.take(n).map(_.copy()).toArray
      if (rows.isEmpty) Iterator.empty
      else {
        numOutputBatches += 1
        numOutputRows += rows.length
        Iterator.single(VectorRowStages.toBatch(schema, rows))
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/** Planning-time checks for the three limits, shared by the rule. */
object VectorLimitPlanner {
  def planLocal(l: LocalLimitExec): Either[String, VectorLocalLimitExec] =
    if (l.limit < 0) Left(s"negative limit ${l.limit}") else Right(VectorLocalLimitExec(l.limit, l.child))

  def planGlobal(g: GlobalLimitExec): Either[String, VectorGlobalLimitExec] =
    if (g.offset != 0) Left(s"offset ${g.offset} not supported")
    else if (g.limit < 0) Left(s"negative limit ${g.limit}")
    else Right(VectorGlobalLimitExec(g.limit, g.child))

  def planCollect(c: CollectLimitExec): Either[String, VectorCollectLimitExec] = {
    val outputFailures = c.child.output.filterNot(a => TypeMapping.hasLane(a.dataType)).map(a =>
      s"unsupported output type ${a.dataType.simpleString} for ${a.name}"
    )
    if (c.offset != 0) Left(s"offset ${c.offset} not supported")
    else if (c.limit < 0) Left(s"negative limit ${c.limit}")
    else if (outputFailures.nonEmpty) Left(outputFailures.mkString("; "))
    else Right(VectorCollectLimitExec(c.limit, c.child))
  }
}
