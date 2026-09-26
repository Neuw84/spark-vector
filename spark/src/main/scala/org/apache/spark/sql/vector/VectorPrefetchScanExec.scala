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
package org.apache.spark.sql.vector

import java.lang.foreign.Arena
import java.util.concurrent.{ArrayBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean

import io.vecruntime.spark.adapter.ColumnVectorAdapters
import io.vecruntime.spark.arrow.{ArrowOutput, SelectedColumnarBatch, VectorAllocators, VectorArrowColumnVector}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * The prefetching scan converter (#403, lever 2). Inserted by the rule between a Spark vectorized
 * file scan and the first operator of ours above it when `spark.vector.scan.prefetch` is positive.
 * Per task a helper thread pulls the reader's batches, converts every column into our own Arrow
 * vectors (which the adapters read zero-copy) and hands the converted batch through a bounded queue
 * of `depth` batches to the task thread, so the reader's waits (S3, decode) overlap our kernels.
 *
 * Not a [[VectorPlan]]: it computes nothing and must not count as an accelerated operator (the UI
 * classifies it as a conversion, like `RowToColumnar`), nor be mistaken by the rule for an operator
 * that reads a forwarded selection. It is columnar, so the operator above accepts it as a child.
 */
case class VectorPrefetchScanExec(child: SparkPlan, depth: Int) extends UnaryExecNode {

  override def supportsColumnar: Boolean = true
  override def output: Seq[Attribute] = child.output
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "batches" -> SQLMetrics.createMetric(sparkContext, "number of prefetched batches"),
    "prefetchWaitMs" -> SQLMetrics.createTimingMetric(sparkContext, "task thread waiting for a converted batch"),
    "readWaitMs" -> SQLMetrics.createTimingMetric(sparkContext, "helper thread waiting on the reader"),
    "convertMs" -> SQLMetrics.createTimingMetric(sparkContext, "helper thread converting batches")
  )

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val attrs = output.map(a => (a.name, a.dataType)).toArray
    val d = depth
    val m = PrefetchMetrics(
      longMetric("batches"),
      longMetric("prefetchWaitMs"),
      longMetric("readWaitMs"),
      longMetric("convertMs")
    )
    child.executeColumnar().mapPartitionsInternal { iter => new PrefetchingBatchConverter(iter, attrs, d, m) }
  }

  /** Rows, for a consumer that calls `execute()` on a columnar child (see [[VectorPlan]]). */
  override protected def doExecute(): RDD[InternalRow] = ColumnarToRowExec(this).doExecute()

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/** The converter's SQL metrics; the helper's two timers are flushed once per task. */
final case class PrefetchMetrics(
    batches: SQLMetric,
    prefetchWaitMs: SQLMetric,
    readWaitMs: SQLMetric,
    convertMs: SQLMetric
) extends Serializable

/**
 * Per-task converter behind [[VectorPrefetchScanExec]]: a daemon helper thread, started lazily on
 * the first `hasNext`, consumes `input` and converts each batch into one of our own; the task thread
 * takes converted batches from a queue of `depth`.
 *
 * Contracts kept:
 *  - Spark's vectorized reader recycles its column vectors when the next batch is requested, so the
 *    helper converts batch N completely (every column materialised into Arrow memory) before it asks
 *    `input` for N+1 -- the loop is sequential on the helper, so the reader never advances under a
 *    conversion in progress.
 *  - `TaskContext` is installed on the helper, so the reader's metrics and memory accounting land on
 *    the task; the input iterator is used by the helper thread only.
 *  - A handed-out batch is closed when the consumer takes the following one or the converter is
 *    closed (Spark's columnar contract, as [[VectorBatchIterator]] and the spill reader keep it).
 *  - Termination: an end-of-stream sentinel; a `Throwable` on the helper is captured and rethrown on
 *    the task thread from `hasNext`; a task kill (`TaskContext.isInterrupted`) or `close()` stops the
 *    helper between batches, and the helper never blocks forever on the queue (`offer` with a timeout).
 *  - `close()` -- from the task-completion listener, registered after the reader's own so it runs
 *    first -- stops and joins the helper, closes every batch still queued or handed out, and closes
 *    the allocator. The reader is closed by its own listener only afterwards.
 *
 * A row-id-mapped batch from a foreign reader (Iceberg merge-on-read) is normalized on the helper
 * and its live rows compacted, so the consumer always sees a dense batch of our vectors; a batch
 * without live rows is dropped. Columns are built with the existing builders
 * (`ColumnVectorAdapters.adapt` into a confined scratch arena, then `ArrowOutput.copy` /
 * `ArrowOutput.compact`), never with a copy loop of this class's own.
 */
final class PrefetchingBatchConverter(
    input: Iterator[ColumnarBatch],
    attrs: Array[(String, DataType)],
    depth: Int,
    metrics: PrefetchMetrics
) extends Iterator[ColumnarBatch]
    with AutoCloseable
    with Logging {

  require(depth > 0, s"prefetch depth must be positive: $depth")

  /** The per-task Arrow allocator every converted batch is allocated from (test-visible). */
  val allocator: BufferAllocator = VectorAllocators.newChild("VectorPrefetchScanExec")

  private val taskContext: TaskContext = TaskContext.get()
  private val queue = new ArrayBlockingQueue[AnyRef](depth)
  private val alive = new AtomicBoolean(true)
  @volatile private var failure: Throwable = _
  @volatile private var readNanos = 0L
  @volatile private var convertNanos = 0L
  private var waitNanos = 0L
  private var helper: Thread = _
  private var pending: ColumnarBatch = _
  private var emitted: ColumnarBatch = _
  private var finished = false
  private var closed = false
  private val transferableClasses = new java.util.HashMap[DataType, Class[_]]() // helper thread only

  if (taskContext != null) taskContext.addTaskCompletionListener[Unit](_ => close())

  /** Whether the helper thread is still running (test-visible). */
  def isHelperAlive: Boolean = helper != null && helper.isAlive

  override def hasNext: Boolean = {
    if (closed || finished) pending != null
    else if (pending != null) true
    else {
      startHelper()
      val start = System.nanoTime()
      var item: AnyRef = null
      while (item == null) {
        item = queue.poll(100, TimeUnit.MILLISECONDS)
        if (item == null && !helper.isAlive) {
          // The helper always posts the sentinel last; an empty queue after its death is a bug, not an end.
          item = queue.poll()
          if (item == null) {
            item = PrefetchingBatchConverter.End
            if (failure == null) failure = new IllegalStateException("prefetch helper ended without a result")
          }
        }
      }
      waitNanos += System.nanoTime() - start
      item match {
        case b: ColumnarBatch =>
          pending = b
          true
        case _ =>
          finished = true
          if (failure != null) throw failure
          false
      }
    }
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseEmitted()
    emitted = pending
    pending = null
    metrics.batches += 1
    emitted
  }

  private def startHelper(): Unit = if (helper == null) {
    val name =
      if (taskContext == null) "vecruntime-prefetch" else s"vecruntime-prefetch-${taskContext.taskAttemptId()}"
    helper = new Thread(() => runHelper(), name)
    helper.setDaemon(true)
    helper.start()
  }

  private def killed: Boolean = taskContext != null && taskContext.isInterrupted()

  private def running: Boolean = alive.get() && !killed

  /** Offers `item` until it is taken or the converter stopped; false when it was not taken. */
  private def offer(item: AnyRef): Boolean = {
    var taken = false
    while (!taken && running) taken = queue.offer(item, 100, TimeUnit.MILLISECONDS)
    taken
  }

  private def runHelper(): Unit = {
    if (taskContext != null) TaskContext.setTaskContext(taskContext)
    try {
      var more = true
      while (more && running) {
        val start = System.nanoTime()
        more = input.hasNext
        if (more) {
          val raw = input.next()
          val read = System.nanoTime()
          readNanos += read - start
          val out = convert(raw)
          convertNanos += System.nanoTime() - read
          // The batch is fully materialised before the loop asks the reader for the next one (above).
          if (out != null && !offer(out)) {
            out.close()
            more = false
          }
        } else readNanos += System.nanoTime() - start
      }
    } catch {
      case t: Throwable => failure = t
    } finally {
      offer(PrefetchingBatchConverter.End)
    }
  }

  /** Every column of `raw` as one of our own Arrow vectors; `null` for a batch with no live rows. */
  private def convert(raw: ColumnarBatch): ColumnarBatch = {
    val batch = InputBatches.normalize(raw)
    val arena = Arena.ofConfined()
    try {
      val n = batch.numRows()
      val (selection, count) = batch match {
        case s: SelectedColumnarBatch => (s.selection(), s.selectedCount())
        case _ => (null, n)
      }
      if (count == 0) null
      else {
        val columns = new Array[ColumnVector](attrs.length)
        try {
          var c = 0
          while (c < columns.length) {
            val (name, dt) = attrs(c)
            val cv = batch.column(c)
            columns(c) = cv match {
              case v: VectorArrowColumnVector if selection == null && transferable(v, dt) =>
                // Already one of our own unshaded Arrow vectors (no reader hands those out today): its
                // buffers move to a vector of ours without a copy, as the spill reader takes a root's
                // (AggregateSpill.toBatch); the source is left empty for its producer to refill or close.
                // Iceberg's vectors are shaded Arrow and the reader's own -- they take the copy below.
                val target = ArrowOutput.newVector(name, dt, allocator)
                v.getValueVector.makeTransferPair(target).transfer()
                ArrowOutput.wrap(target, dt)
              case _ =>
                val in = ColumnVectorAdapters.adapt(cv, n, arena)
                if (selection == null) ArrowOutput.copy(name, dt, in, allocator)
                else ArrowOutput.compact(name, dt, in, selection, count, allocator)
            }
            c += 1
          }
        } catch {
          case t: Throwable =>
            columns.foreach(v => if (v != null) v.close())
            throw t
        }
        new ColumnarBatch(columns, count)
      }
    } finally {
      arena.close()
      if (batch ne raw) batch.close()
    }
  }

  private def releaseEmitted(): Unit = if (emitted != null) {
    emitted.close()
    emitted = null
  }

  /**
   * Whether the vector under `v` is the class our builders make for `dt`, so a transfer pair applies
   * (an empty probe vector allocates no buffer). Decided once per column, on the first batch.
   */
  private def transferable(v: VectorArrowColumnVector, dt: DataType): Boolean = {
    val cls = v.getValueVector.getClass
    val target = transferableClasses.computeIfAbsent(
      dt,
      _ => {
        val probe = ArrowOutput.newVector("probe", dt, allocator)
        try probe.getClass
        finally probe.close()
      }
    )
    target == cls
  }

  override def close(): Unit = if (!closed) {
    closed = true
    alive.set(false)
    if (helper != null) {
      helper.interrupt()
      helper.join(PrefetchingBatchConverter.JoinMillis)
      if (helper.isAlive)
        logWarning(s"prefetch helper ${helper.getName} did not stop within ${PrefetchingBatchConverter.JoinMillis} ms")
    }
    releaseEmitted()
    if (pending != null) {
      pending.close()
      pending = null
    }
    var item = queue.poll()
    while (item != null) {
      item match {
        case b: ColumnarBatch => b.close()
        case _ =>
      }
      item = queue.poll()
    }
    metrics.prefetchWaitMs += waitNanos / 1000000L
    metrics.readWaitMs += readNanos / 1000000L
    metrics.convertMs += convertNanos / 1000000L
    try allocator.close()
    catch {
      case e: IllegalStateException => logWarning(s"prefetch allocator not closed cleanly: ${e.getMessage}")
    }
  }
}

object PrefetchingBatchConverter {

  /** End-of-stream sentinel the helper posts last, after a failure too. */
  private object End

  /** How long `close()` waits for the helper to stop before giving up on it. */
  val JoinMillis: Long = 10000L
}
