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

import io.sparkvector.spark.arrow.{ArrowOutput, BorrowedColumnVector, VectorAllocators}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, Literal}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}

/**
 * A `ROLLUP` as a chain of aggregates in one pass (#383), replacing `Partial over Expand`.
 *
 * `levels(0)` is the partial aggregate on the finest grouping set over `child`; `levels(k)` is the
 * partial-merge on the next coarser set, planned over `levels(k - 1)` (for compilation and display --
 * at run time every level runs inside this operator's task). Each batch a level emits goes out under
 * that level's projection (its keys, nulls for the rolled-up ones, the grouping id, the buffers under
 * the original partial's attributes) AND is fed into the next level's table, so the input is hashed
 * once on the full key tuple and each coarser set hashes only the groups of the set before it -- not
 * the input once per set, which is what Expand's copies cost.
 *
 * Ownership follows the aggregate's contract: a level closes the batch it emitted when it produces
 * the next, and the chain only asks a level for its next batch once the consumer has taken the
 * previous one; the batches a level emits early under its memory budget are owned by the chain and
 * closed when the consumer takes the next.
 */
case class VectorRollupExec(
    levels: Seq[VectorHashAggregateExec],
    projections: Seq[Seq[Expression]],
    output: Seq[Attribute],
    child: SparkPlan
) extends VectorExec {

  require(levels.nonEmpty && levels.length == projections.length)

  override def outputPartitioning: Partitioning = child.outputPartitioning

  /** Each level's projection over that level's output: column references or literals. */
  @transient private lazy val compiled: Array[Array[VectorExpr]] = levels.zip(projections).map { case (level, p) =>
    p.map {
      case Literal(v, dt) => LiteralExpr(v, dt)
      case e =>
        ExpressionCompiler.compileLaneColumn(e, level.output) match {
          case Right(c) => c
          case Left(reason) => throw new IllegalStateException(s"cannot vectorize rollup slot ${e.sql}: $reason")
        }
    }.toArray
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val makers = levels.map(_.levelIterator).toArray
    val slots = compiled
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      val its = makers.zipWithIndex.map { case (make, k) => make(if (k == 0) iter else Iterator.empty) }
      new RollupChainIterator(its, slots, outputAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
    copy(
      levels = levels.updated(0, levels.head.withNewChildren(Seq(newChild)).asInstanceOf[VectorHashAggregateExec]),
      child = newChild
    )

  override def simpleString(maxFields: Int): String =
    s"VectorRollup(levels=${levels.length}, keys=${levels.head.groupingExpressions.map(_.sql).mkString(", ")})"

  override def verboseStringWithOperatorId(): String =
    s"""$formattedNodeName
       |Levels: ${levels.map(_.groupingExpressions.map(_.sql).mkString("(", ", ", ")")).mkString(" -> ")}
       |Functions: ${levels.head.aggregateExpressions.map(_.sql).mkString(", ")}
       |""".stripMargin
}

/** Drives the levels: level 0 over the input, each next level fed with the batches of the one before. */
private[vector] class RollupChainIterator(
    levels: Array[RollupLevel],
    slots: Array[Array[VectorExpr]],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics
) extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorRollupExec")
  private var level = 0
  // Batches ready for the consumer: (projected batch, the level's own batch to close when owned).
  private val pending = new java.util.ArrayDeque[(ColumnarBatch, ColumnarBatch)]()
  private var emitted: (ColumnarBatch, ColumnarBatch) = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = {
    fill()
    !pending.isEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseEmitted()
    emitted = pending.poll()
    metrics.numOutputBatches += 1
    metrics.numOutputRows += emitted._1.numRows()
    emitted._1
  }

  /** Pulls from the current level until something is pending or every level is exhausted. */
  private def fill(): Unit = {
    while (pending.isEmpty && level < levels.length) {
      val it = levels(level)
      if (it.hasNext) {
        val batch = it.next() // the level closes it when asked for the next one: not ours
        emit(level, batch, owned = null)
      } else {
        level += 1
        if (level < levels.length) levels(level).finishFeed()
      }
    }
  }

  /** Projects one level's batch for the consumer and feeds it to the next level (whose early emits recurse). */
  private def emit(k: Int, batch: ColumnarBatch, owned: ColumnarBatch): Unit = {
    pending.add((project(k, batch), owned))
    if (k + 1 < levels.length) {
      levels(k + 1).feed(batch).foreach(early => emit(k + 1, early, owned = early))
    }
  }

  private def project(k: Int, batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    val n = batch.numRows()
    val s = slots(k)
    val columns = new Array[ColumnVector](s.length)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = outputAttrs(c)
      columns(c) = s(c) match {
        case ColumnRef(ordinal, _) => BorrowedColumnVector.of(batch.column(ordinal))
        case lit: LiteralExpr if lit.value == null => ArrowOutput.nulls(name, dt, n, allocator)
        case lit: LiteralExpr => ArrowOutput.constant(name, dt, lit.value, n, allocator)
        case other => throw new IllegalStateException(s"rollup slot is not a column or literal: $other")
      }
      c += 1
    }
    new ColumnarBatch(columns, n)
  }

  private def releaseEmitted(): Unit = {
    if (emitted != null) {
      emitted._1.close() // borrowed columns ignore close; the constant columns are ours
      if (emitted._2 != null) emitted._2.close()
      emitted = null
    }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseEmitted()
      while (!pending.isEmpty) { val p = pending.poll(); p._1.close(); if (p._2 != null) p._2.close() }
      levels.foreach(l =>
        try l.close()
        catch { case _: Exception => }
      )
      allocator.close()
    }
  }
}
