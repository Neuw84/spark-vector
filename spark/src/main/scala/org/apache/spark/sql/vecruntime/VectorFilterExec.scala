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
package org.apache.spark.sql.vecruntime

import io.vecruntime.spark.adapter.TypeMapping
import io.vecruntime.spark.arrow.{ArrowOutput, RemappedColumnVector, SelectedColumnarBatch}
import io.vecruntime.spark.expr.{ExpressionCompiler, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{FilterExec, SparkPlan}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for FilterExec. Evaluates the condition with SIMD kernels to a selection
 * bitmap. When the parent is another spark-vector operator (`emitSelection`) the child's columns
 * are forwarded untouched together with the bitmap as a [[SelectedColumnarBatch]]; otherwise every
 * column is compacted into new Arrow vectors for Spark. Batches with no surviving rows are dropped;
 * batches where every row survives are passed through without copying.
 */
case class VectorFilterExec(condition: Expression, child: SparkPlan, emitSelection: Boolean = false)
    extends VectorExec {

  /**
   * The output Spark's own filter reports: the attributes its `IsNotNull` predicates guard become
   * non-nullable. Nullability is part of the plan AQE validates (#416): a broadcast join requires a
   * hash-relation mode over the build side's bound keys, nullability included, and the broadcast
   * exchange fixed its mode at the first planning, when the filter under it was still Spark's. A
   * filter reporting its child's nullability made the re-planned join's requirement differ from the
   * exchange's mode, `ValidateRequirements` rejected the stage plan, and AQE silently dropped the
   * shuffle-partition coalescing of every join in it -- at 1 TB / 1000 partitions the two shuffled
   * joins of q18 ran 1000 tasks each over 1000-partition inputs, 330 s of the 556 s the partition
   * count cost.
   */
  override lazy val output: Seq[Attribute] = FilterExec(condition, child).output
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  /** Compiled on the driver; failure here is a bug because the rule already checked it. */
  @transient private lazy val compiled: VectorExpr =
    ExpressionCompiler.compilePredicate(condition, child.output) match {
      case Right(e) => e
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize filter: $reason")
    }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val predicate = compiled
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    val selectionOut = emitSelection
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorFilterIterator(iter, predicate, outputAttrs, selectionOut, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

private[vecruntime] class VectorFilterIterator(
    input: Iterator[ColumnarBatch],
    predicate: VectorExpr,
    outputAttrs: Array[(String, org.apache.spark.sql.types.DataType)],
    emitSelection: Boolean,
    metrics: VectorMetrics
) extends VectorBatchIterator(input, "VectorFilterExec") {

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val pred = predicate.eval(ctx)
      val (selection, count) = VectorExpr.selection(pred, ctx)
      if (count == 0) {
        null
      } else if (count == ctx.numRows && !ctx.hasSelection) {
        // Every physical row survives: pass through.
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        batch
      } else if (emitSelection && SelectionPolicy.keep(count, ctx.numRows)) {
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        SelectedColumnarBatch.of(SelectedColumnarBatch.columnsOf(batch), ctx.numRows, selection, count, false)
      } else {
        val columns = new Array[ColumnVector](outputAttrs.length)
        var foreignRows: Array[Int] = null // the selection as row ids, for columns with no lane (passed through)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) =
            if (TypeMapping.hasLane(dt)) ArrowOutput.compact(name, dt, ctx.input(c), selection, count, allocator)
            else {
              if (foreignRows == null) foreignRows = RemappedColumnVector.rowsOf(selection, ctx.numRows, count)
              RemappedColumnVector.of(ctx.column(c), foreignRows)
            }
          c += 1
        }
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        new ColumnarBatch(columns, count)
      }
    }
  }
}

/**
 * When to forward a selection instead of compacting. Downstream kernels run over every physical
 * row of a selected batch, so a sparse selection (Q6 keeps 2%) is cheaper to compact once than to
 * carry; a dense one (Q1 keeps 98%) is cheaper to carry.
 */
object SelectionPolicy {

  /** Minimum surviving fraction for forwarding a selection; `sparkvector.selection.minFraction`. */
  val MinFraction: Double = java.lang.Double.parseDouble(System.getProperty("sparkvector.selection.minFraction", "0.5"))

  def keep(selected: Int, numRows: Int): Boolean = selected >= numRows * MinFraction
}
