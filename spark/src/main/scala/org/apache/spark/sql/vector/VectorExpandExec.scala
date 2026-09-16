package org.apache.spark.sql.vector

import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, BorrowedColumnVector, SelectedColumnarBatch, VectorAllocators}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, Literal}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{ExpandExec, SparkPlan}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for ExpandExec (`ROLLUP`, `CUBE`, `GROUPING SETS`, and the distinct-aggregate
 * rewrite). Every projection of every input batch becomes one output batch with no data copy: a
 * retained column is borrowed from the input, a nulled key is an all-invalid constant column and the
 * grouping id (or any other literal) a constant column. Projection slots therefore have to be column
 * references or literals, which is what Spark's grouping-set analysis emits after it has aliased the
 * key expressions into the project below.
 *
 * Ownership, made explicit because this operator emits several batches per input: the input batch is
 * held while its projections are emitted and the child is only asked for the next one after the last
 * projection has been handed out and released by the consumer -- so borrowed columns stay valid
 * exactly as long as Spark's producer contract keeps them. Constant columns belong to the output
 * batch and are released when the consumer asks for the next batch. A pass-through operator for the
 * selection rule: a filter below compacts rather than forwarding a selection.
 */
case class VectorExpandExec(projections: Seq[Seq[Expression]], output: Seq[Attribute], child: SparkPlan)
    extends VectorExec
    with VectorPassThrough {

  override def outputPartitioning: Partitioning = child.outputPartitioning

  /** Compiled on the driver; the rule already checked every slot. */
  @transient private lazy val compiled: Array[Array[VectorExpr]] = projections.map { p =>
    p.map {
      // Literals go straight to LiteralExpr: the compiler refuses NULL and boolean literals as
      // operands, but here they are whole constant columns.
      case Literal(v, dt) => LiteralExpr(v, dt)
      case e =>
        ExpressionCompiler.compile(e, child.output) match {
          case Right(c) => c
          case Left(reason) => throw new IllegalStateException(s"cannot vectorize expand slot ${e.sql}: $reason")
        }
    }.toArray
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val slots = compiled
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal(iter => new VectorExpandIterator(iter, slots, outputAttrs, m))
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def simpleString(maxFields: Int): String = s"VectorExpand(projections=${projections.length}, output=${output.map(_.name).mkString(", ")})"
}

/** One input batch in, `projections.length` output batches out, then the next input. */
private[vector] class VectorExpandIterator(
    input: Iterator[ColumnarBatch],
    projections: Array[Array[VectorExpr]],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorExpandExec")
  private var current: ColumnarBatch = _ // the input batch whose projections are being emitted
  private var currentWrapped = false // `current` is a normalization wrapper we own (selection only)
  private var nextProjection = 0
  private var emitted: ColumnarBatch = _ // our last output, released when the next is requested
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = {
    while ((current == null || nextProjection >= projections.length) && input.hasNext) {
      // Asking the child for its next batch is what releases the previous one; every projection of
      // it has been emitted and the consumer has moved past the last of them by now.
      releaseCurrent()
      val raw = input.next()
      val batch = InputBatches.normalize(raw)
      if (batch.numRows() > 0) {
        current = batch
        currentWrapped = batch ne raw
        nextProjection = 0
        metrics.numInputBatches += 1
      } else if (batch ne raw) {
        batch.close()
      }
    }
    current != null && nextProjection < projections.length
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseEmitted()
    val slots = projections(nextProjection)
    nextProjection += 1
    val out = metrics.timed {
      current match {
        case s: SelectedColumnarBatch =>
          // A row-id-mapped foreign batch (Iceberg merge-on-read) arrives as physical rows plus a
          // selection; our own children compact, so this is the only selected shape seen here.
          EvalContexts.withBatch(s) { ctx =>
            val n = ctx.selectedCount
            val columns = new Array[ColumnVector](slots.length)
            var c = 0
            while (c < columns.length) {
              val (name, dt) = outputAttrs(c)
              columns(c) = slots(c) match {
                case ColumnRef(ordinal, _) => ArrowOutput.compact(name, dt, ctx.input(ordinal), ctx.selection, n, allocator)
                case lit: LiteralExpr if lit.value == null => ArrowOutput.nulls(name, dt, n, allocator)
                case lit: LiteralExpr => ArrowOutput.constant(name, dt, lit.value, n, allocator)
                case other => throw new IllegalStateException(s"expand slot is not a column or literal: $other")
              }
              c += 1
            }
            new ColumnarBatch(columns, n)
          }
        case batch =>
          val n = batch.numRows()
          val columns = new Array[ColumnVector](slots.length)
          var c = 0
          while (c < columns.length) {
            val (name, dt) = outputAttrs(c)
            columns(c) = slots(c) match {
              case ColumnRef(ordinal, _) => BorrowedColumnVector.of(batch.column(ordinal))
              case lit: LiteralExpr if lit.value == null => ArrowOutput.nulls(name, dt, n, allocator)
              case lit: LiteralExpr => ArrowOutput.constant(name, dt, lit.value, n, allocator)
              case other => throw new IllegalStateException(s"expand slot is not a column or literal: $other")
            }
            c += 1
          }
          new ColumnarBatch(columns, n)
      }
    }
    metrics.numOutputBatches += 1
    metrics.numOutputRows += out.numRows()
    emitted = out
    out
  }

  private def releaseEmitted(): Unit = {
    if (emitted != null) { emitted.close(); emitted = null } // borrowed columns ignore close
  }

  /** Drops our reference to the input; a normalization wrapper (selection bitmap only) is ours to close. */
  private def releaseCurrent(): Unit = {
    if (current != null) {
      if (currentWrapped) current.close()
      current = null
      currentWrapped = false
    }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseEmitted()
      releaseCurrent()
      allocator.close()
    }
  }
}

/** Planning-time checks for the expand, shared by the rule. */
object VectorExpandPlanner {

  /** Every slot a column reference or a literal (incl. `NULL`) of a supported type. */
  def plan(e: ExpandExec): Either[String, VectorExpandExec] = {
    val outputFailures = e.output.filterNot(a => TypeMapping.isSupported(a.dataType)).map(a => s"unsupported output type ${a.dataType.simpleString} for ${a.name}")
    val slotFailures = e.projections.zipWithIndex.flatMap { case (p, i) =>
      p.flatMap {
        case Literal(_, dt) if !TypeMapping.isSupported(dt) => Some(s"projection $i: literal of unsupported type ${dt.simpleString}")
        case _: Literal => None
        case expr =>
          ExpressionCompiler.compile(expr, e.child.output) match {
            case Right(_: ColumnRef) => None
            case Right(_) => Some(s"projection $i: ${expr.sql} is not a column or literal")
            case Left(reason) => Some(s"projection $i: ${expr.sql}: $reason")
          }
      }
    }
    if (e.projections.isEmpty) Left("expand without projections")
    else if (outputFailures.nonEmpty) Left(outputFailures.mkString("; "))
    else if (slotFailures.nonEmpty) Left(slotFailures.distinct.mkString("; "))
    else Right(VectorExpandExec(e.projections, e.output, e.child))
  }
}
