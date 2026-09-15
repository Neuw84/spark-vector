package org.apache.spark.sql.vector

import io.sparkvector.spark.arrow.{ArrowOutput, VectorArrowColumnVector, VectorDictionaryColumnVector}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, VectorExpr}
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
 */
case class VectorProjectExec(projectList: Seq[NamedExpression], child: SparkPlan)
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
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorProjectIterator(iter, exprs, identity, outputAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

private[vector] class VectorProjectIterator(
    input: Iterator[ColumnarBatch],
    exprs: Array[VectorExpr],
    identity: Boolean,
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorProjectExec") {

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    metrics.numOutputBatches += 1
    metrics.numOutputRows += batch.numRows()
    if (identity) {
      batch
    } else {
      withEvalContext(batch) { ctx =>
        val columns = new Array[ColumnVector](exprs.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) = exprs(c) match {
            case ColumnRef(ordinal, _) =>
              batch.column(ordinal) match {
                case v: VectorArrowColumnVector => v.borrow()
                case v: VectorDictionaryColumnVector => v.borrow()
                case _ => ArrowOutput.copy(name, dt, ctx.input(ordinal), allocator)
              }
            case e => ArrowOutput.copy(name, dt, e.eval(ctx), allocator)
          }
          c += 1
        }
        new ColumnarBatch(columns, ctx.numRows)
      }
    }
  }
}
