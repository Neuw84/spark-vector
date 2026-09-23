package org.apache.spark.sql.vector

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{
  ArrowOutput,
  BorrowedColumnVector,
  NestedFieldColumnVector,
  RemappedColumnVector,
  SelectedColumnarBatch
}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, NestedColumnRef, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{
  Alias,
  Attribute,
  AttributeReference,
  Expression,
  GetStructField,
  Literal,
  NamedExpression,
  SortOrder
}
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
 *
 * A column whose type has no lane (struct, array, map, a decimal wider than 18 digits) does not
 * disqualify the operator when the projection only passes it through: Spark's own vector for it is
 * borrowed into the output batch, or viewed through a [[RemappedColumnVector]] when a selection is
 * applied. The batch then mixes our Arrow vectors with the child's; Spark's `ColumnarToRowExec` reads
 * both through the `ColumnVector` interface, our operators above refuse the column by type as before,
 * and the Comet shuffle bridge declines such a child (a foreign vector cannot cross the C Data
 * interface). Only computed expressions must compile.
 */
case class VectorProjectExec(projectList: Seq[NamedExpression], child: SparkPlan, emitSelection: Boolean = false)
    extends VectorExec
    with PartitioningPreservingUnaryExecNode
    with OrderPreservingUnaryExecNode {

  override def output: Seq[Attribute] = projectList.map(_.toAttribute)
  override protected def outputExpressions: Seq[NamedExpression] = projectList
  override protected def orderingExpressions: Seq[SortOrder] = child.outputOrdering

  @transient private lazy val compiled: Array[VectorExpr] = projectList.map { e =>
    if (VectorProjectExec.isPassThrough(e)) {
      val passed = VectorProjectExec.passedThrough(e)
      ExpressionCompiler.nestedColumnPath(passed, child.output) match {
        case Right((ordinal, Nil)) => ColumnRef(ordinal, passed.dataType)
        case Right((ordinal, path)) => NestedColumnRef(ordinal, path, passed.dataType)
        case Left(reason) => throw new IllegalStateException(s"cannot pass through ${e.sql}: $reason")
      }
    } else VectorProjectExec.constantSlot(e).getOrElse(ExpressionCompiler.compile(e, child.output) match {
      case Right(v) => v
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize projection ${e.sql}: $reason")
    })
  }.toArray

  private def isIdentity: Boolean =
    projectList.length == child.output.length &&
      projectList.zip(child.output).forall { case (e, a) =>
        e.toAttribute.exprId == a.exprId && e.isInstanceOf[Attribute]
      }

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

object VectorProjectExec {

  /**
   * A projection that forwards a child column as is (possibly renamed), whatever its type, or a struct
   * field of one whose type has no lane (Spark's `NestedColumnAliasing` projects `st.inner` for a
   * generate over it): never compiled, Spark's vector is passed through.
   */
  /**
   * A bare literal slot (`true AS flag`, `CAST(NULL AS int) AS n`): a whole constant or null column of any
   * lane type, materialised without the compiler, which refuses boolean and null literals as operands (#273).
   */
  def constantSlot(e: NamedExpression): Option[VectorExpr] = e match {
    case Alias(Literal(v, dt), _) if TypeMapping.hasLane(dt) => Some(LiteralExpr(v, dt))
    case _ => None
  }

  def isPassThrough(e: NamedExpression): Boolean = e match {
    case _: AttributeReference => true
    case Alias(_: AttributeReference, _) => true
    case g: GetStructField => !TypeMapping.isSupported(g.dataType) && isNestedColumn(g)
    case Alias(g: GetStructField, _) => !TypeMapping.isSupported(g.dataType) && isNestedColumn(g)
    case _ => false
  }

  private def isNestedColumn(e: Expression): Boolean = e match {
    case _: AttributeReference => true
    case GetStructField(child, _, _) => isNestedColumn(child)
    case _ => false
  }

  def passedThrough(e: NamedExpression): Expression = e match {
    case a: AttributeReference => a
    case Alias(a: AttributeReference, _) => a
    case g: GetStructField => g
    case Alias(g: GetStructField, _) => g
    case other => throw new IllegalArgumentException(s"not a pass-through: ${other.sql}")
  }
}

private[vector] class VectorProjectIterator(
    input: Iterator[ColumnarBatch],
    exprs: Array[VectorExpr],
    identity: Boolean,
    outputAttrs: Array[(String, DataType)],
    emitSelection: Boolean,
    metrics: VectorMetrics
) extends VectorBatchIterator(input, "VectorProjectExec") {

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
        val compactTo = if (selected && (!emitSelection || !SelectionPolicy.keep(ctx.selectedCount, ctx.numRows)))
          ctx.selection
        else null
        val outRows = if (compactTo != null) ctx.selectedCount else ctx.numRows
        val columns = new Array[ColumnVector](exprs.length)
        var foreignRows: Array[Int] = null // the selection as row ids, built once for the columns with no lane
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) = exprs(c) match {
            case ColumnRef(ordinal, _) if compactTo != null && !TypeMapping.hasLane(dt) =>
              if (foreignRows == null) foreignRows = RemappedColumnVector.rowsOf(compactTo, ctx.numRows, outRows)
              RemappedColumnVector.of(ctx.column(ordinal), foreignRows)
            case NestedColumnRef(ordinal, path, _) =>
              val field = NestedFieldColumnVector.of(ctx.column(ordinal), path.toArray, ctx.numRows)
              if (compactTo == null) field
              else {
                if (foreignRows == null) foreignRows = RemappedColumnVector.rowsOf(compactTo, ctx.numRows, outRows)
                RemappedColumnVector.of(field, foreignRows)
              }
            case ColumnRef(ordinal, _) if compactTo != null =>
              ArrowOutput.compact(name, dt, ctx.input(ordinal), compactTo, outRows, allocator)
            case ColumnRef(ordinal, _) =>
              // Forwarded columns are never copied: the child keeps them alive until its next batch.
              BorrowedColumnVector.of(batch.column(ordinal))
            case lit: LiteralExpr if lit.value == null => ArrowOutput.nulls(name, dt, outRows, allocator)
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
