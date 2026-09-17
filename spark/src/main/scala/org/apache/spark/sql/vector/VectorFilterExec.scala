package org.apache.spark.sql.vector

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, RemappedColumnVector, SelectedColumnarBatch}
import io.sparkvector.spark.expr.{ExpressionCompiler, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
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

  override def output: Seq[Attribute] = child.output
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

private[vector] class VectorFilterIterator(
    input: Iterator[ColumnarBatch],
    predicate: VectorExpr,
    outputAttrs: Array[(String, org.apache.spark.sql.types.DataType)],
    emitSelection: Boolean,
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorFilterExec") {

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
            if (TypeMapping.isSupported(dt)) ArrowOutput.compact(name, dt, ctx.input(c), selection, count, allocator)
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
