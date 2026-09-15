package org.apache.spark.sql.vector

import io.sparkvector.spark.arrow.ArrowOutput
import io.sparkvector.spark.expr.{ExpressionCompiler, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, SortOrder}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for FilterExec. Evaluates the condition with SIMD kernels to a selection
 * bitmap, then compacts every column into new Arrow vectors. Batches with no surviving rows are
 * dropped; batches where every row survives are passed through without copying.
 */
case class VectorFilterExec(condition: Expression, child: SparkPlan) extends VectorExec {

  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

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
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorFilterIterator(iter, predicate, outputAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

private[vector] class VectorFilterIterator(
    input: Iterator[ColumnarBatch],
    predicate: VectorExpr,
    outputAttrs: Array[(String, org.apache.spark.sql.types.DataType)],
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorFilterExec") {

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val pred = predicate.eval(ctx)
      val (selection, count) = VectorExpr.selection(pred, ctx)
      if (count == 0) {
        null
      } else if (count == ctx.numRows) {
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        batch
      } else {
        val columns = new Array[ColumnVector](outputAttrs.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), selection, count, allocator)
          c += 1
        }
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        new ColumnarBatch(columns, count)
      }
    }
  }
}
