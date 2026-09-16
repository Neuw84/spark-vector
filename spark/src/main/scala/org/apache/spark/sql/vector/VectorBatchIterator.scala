package org.apache.spark.sql.vector

import java.lang.foreign.Arena

import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.expr.EvalContext
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Base iterator for spark-vector operators.
 *
 * Lifecycle rules:
 *  - an output batch we built is closed when the consumer asks for the next one (Spark's
 *    columnar iterators never close batches; the producer reuses or releases them);
 *  - an input batch may be passed through untouched (e.g. a filter that keeps every row); it is
 *    then owned by the child and never closed here;
 *  - the per-task Arrow child allocator is closed from a task-completion listener after the last
 *    emitted batch has been released.
 */
abstract class VectorBatchIterator(input: Iterator[ColumnarBatch], name: String)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  protected val allocator: BufferAllocator = io.sparkvector.spark.arrow.VectorAllocators.newChild(name)

  private var pending: ColumnarBatch = _
  private var pendingOwned = false
  private var emitted: ColumnarBatch = _
  private var emittedOwned = false
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  /**
   * Processes one input batch. Return `null` to emit nothing for it, the input itself to pass it
   * through, or a freshly built batch (owned by this iterator).
   */
  protected def process(batch: ColumnarBatch): ColumnarBatch

  override def hasNext: Boolean = {
    while (pending == null && input.hasNext) {
      val raw = input.next()
      if (raw.numRows() > 0) {
        // A foreign reader's row-id-mapped batch becomes a selected batch over its physical rows.
        // The wrapper owns only its selection bitmap; the columns stay with the child.
        val in = InputBatches.normalize(raw)
        val wrapped = in ne raw
        val out = process(in)
        if (out == null) {
          if (wrapped) in.close()
        } else {
          pending = out
          pendingOwned = (out ne in) || wrapped
          if (wrapped && (out ne in)) in.close()
        }
      }
    }
    pending != null
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseEmitted()
    emitted = pending
    emittedOwned = pendingOwned
    pending = null
    emitted
  }

  private def releaseEmitted(): Unit = {
    if (emitted != null) {
      if (emittedOwned) emitted.close()
      emitted = null
    }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseEmitted()
      if (pending != null) {
        if (pendingOwned) pending.close()
        pending = null
      }
      allocator.close()
    }
  }

  /** Runs `f` with an evaluation context over `batch` and a confined scratch arena. */
  protected def withEvalContext[T](batch: ColumnarBatch)(f: EvalContext => T): T =
    EvalContexts.withBatch(batch)(f)
}

object EvalContexts {
  /**
   * Runs `f` with an evaluation context over `batch`; the scratch arena is closed afterwards. A
   * row-id-mapped batch from a foreign reader is normalized first (see [[InputBatches]]), so
   * consumers that read batches without going through [[VectorBatchIterator]] (aggregate, sort)
   * see the physical rows plus a selection too.
   */
  def withBatch[T](raw: ColumnarBatch)(f: EvalContext => T): T = {
    val batch = InputBatches.normalize(raw)
    val arena = Arena.ofConfined()
    try {
      val n = batch.numRows()
      val ctx = batch match {
        case s: io.sparkvector.spark.arrow.SelectedColumnarBatch =>
          new EvalContext(arena, n, c => ColumnVectorAdapters.adapt(batch.column(c), n, arena), s.selection(), s.selectedCount())
        case _ =>
          new EvalContext(arena, n, c => ColumnVectorAdapters.adapt(batch.column(c), n, arena))
      }
      f(ctx)
    } finally {
      arena.close()
      if (batch ne raw) batch.close()
    }
  }
}

/**
 * Normalization of input batches from readers whose batch shape differs from Spark's plain
 * `ColumnarBatch`: today Iceberg's JVM reader on merge-on-read tables, whose columns remap row
 * ids through a shared mapping. The result is either the input itself or a
 * [[io.sparkvector.spark.arrow.SelectedColumnarBatch]] over the unwrapped columns that the caller
 * must close (it owns only the selection bitmap).
 */
object InputBatches {
  def normalize(batch: ColumnarBatch): ColumnarBatch =
    io.sparkvector.spark.iceberg.IcebergVectorAdapter.normalize(batch)
}

/** Metric bookkeeping shared by the operators. */
final class VectorMetrics(
    val numInputBatches: SQLMetric,
    val numOutputBatches: SQLMetric,
    val numOutputRows: SQLMetric,
    val time: SQLMetric)
    extends Serializable {

  def timed[T](f: => T): T = {
    val start = System.nanoTime()
    try f
    finally time += System.nanoTime() - start
  }
}
