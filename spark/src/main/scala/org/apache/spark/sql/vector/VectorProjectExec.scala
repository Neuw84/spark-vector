package org.apache.spark.sql.vector

import io.sparkvector.spark.arrow.{ArrowOutput, BorrowedColumnVector, SelectedColumnarBatch}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, NamedExpression, SortOrder}
import org.apache.spark.sql.execution.{OrderPreservingUnaryExecNode, PartitioningPreservingUnaryExecNode, SparkPlan}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for ProjectExec. Computed expressions are evaluated with the kernels into a
 * scratch arena and copied into Arrow vectors; forwarded input columns are borrowed when the child
 * already produced Arrow memory and copied otherwise. A projection that is exactly the child's
 * output passes batches through.
 *
 * A [[SelectedColumnarBatch]] from the child is forwarded as such when the parent is a spark-vector
 * operator (`emitSelection`); otherwise the selection is applied here and only surviving rows are
 * materialised for Spark.
 */
case class VectorProjectExec(projectList: Seq[NamedExpression], child: SparkPlan, emitSelection: Boolean = false)
    extends VectorExec
    with PartitioningPreservingUnaryExecNode
    with OrderPreservingUnaryExecNode {

  override def output: Seq[Attribute] = projectList.map(_.toAttribute)
  override protected def outputExpressions: Seq[NamedExpression] = projectList
  override protected def orderingExpressions: Seq[SortOrder] = child.outputOrdering

  @transient private lazy val compiled: Array[VectorExpr] = projectList.map { e =>
    ExpressionCompiler.compile(e, child.output) match {
      case Right(v) => v
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize projection ${e.sql}: $reason")
    }
  }.toArray

  private def isIdentity: Boolean =
    projectList.length == child.output.length &&
      projectList.zip(child.output).forall { case (e, a) => e.toAttribute.exprId == a.exprId && e.isInstanceOf[Attribute] }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val exprs = compiled
    val identity = isIdentity
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    val selectionOut = emitSelection
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorProjectIterator(iter, exprs, identity, outputAttrs, selectionOut, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

private[vector] class VectorProjectIterator(
    input: Iterator[ColumnarBatch],
    exprs: Array[VectorExpr],
    identity: Boolean,
    outputAttrs: Array[(String, DataType)],
    emitSelection: Boolean,
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorProjectExec") {

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    metrics.numOutputBatches += 1
    val selected = batch.isInstanceOf[SelectedColumnarBatch]
    if (identity && (emitSelection || !selected)) {
      metrics.numOutputRows += (batch match {
        case s: SelectedColumnarBatch => s.selectedCount()
        case _ => batch.numRows()
      })
      batch
    } else {
      withEvalContext(batch) { ctx =>
        // With a selection we either keep it (parent is ours, and enough rows survive to make
        // computing on the whole batch cheaper than compacting) or apply it now.
        val compactTo = if (selected && (!emitSelection || !SelectionPolicy.keep(ctx.selectedCount, ctx.numRows))) ctx.selection else null
        val outRows = if (compactTo != null) ctx.selectedCount else ctx.numRows
        val columns = new Array[ColumnVector](exprs.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) = exprs(c) match {
            case ColumnRef(ordinal, _) if compactTo != null =>
              ArrowOutput.compact(name, dt, ctx.input(ordinal), compactTo, outRows, allocator)
            case ColumnRef(ordinal, _) =>
              // Forwarded columns are never copied: the child keeps them alive until its next batch.
              BorrowedColumnVector.of(batch.column(ordinal))
            case lit: LiteralExpr => ArrowOutput.constant(name, dt, lit.value, outRows, allocator)
            case e if compactTo != null => ArrowOutput.compact(name, dt, e.eval(ctx), compactTo, outRows, allocator)
            case e => ArrowOutput.copy(name, dt, e.eval(ctx), allocator)
          }
          c += 1
        }
        metrics.numOutputRows += outRows
        if (selected && compactTo == null) {
          SelectedColumnarBatch.of(columns, ctx.numRows, ctx.selection, ctx.selectedCount, true)
        } else {
          new ColumnarBatch(columns, outRows)
        }
      }
    }
  }
}
