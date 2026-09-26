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

import java.lang.foreign.{Arena, MemorySegment}
import java.util.ArrayDeque

import io.vecruntime.kernels.{Bitmap, ColumnBuilder, RunKernels, RunMerge, SegmentVectorBuffers, VecType, VectorBuffers}
import io.vecruntime.spark.arrow.{ArrowOutput, VectorAllocators}
import io.vecruntime.spark.expr.VectorExpr
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, Expression, SortOrder}
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.physical.{
  ClusteredDistribution,
  Distribution,
  Partitioning,
  PartitioningCollection,
  UnknownPartitioning
}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.joins.SortMergeJoinExec
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for SortMergeJoinExec: a real, order-preserving merge join (#286).
 *
 * Same contract as Spark's: both children clustered on the keys and sorted by them ascending, so
 * the sorts Spark placed below stay (ours over a columnar child, Spark's over its row shuffle with
 * a RowToColumnarExec inserted above); the output keeps the left side's order for inner, left
 * outer, semi, anti and existence joins, the right side's for a right outer join, none for a full
 * outer join -- as Spark's operator reports.
 *
 * Execution streams both sides: the right side is read run by run (a run being the rows sharing
 * one key, found by [[RunKernels.boundaries]] over the sorted key columns, lane-parallel) and the
 * current run is the only buffered state -- copied into its own arena, so a run spanning batches
 * is one run and memory is bounded by the largest run, not the side. The left side is walked run
 * by run inside each batch; runs on the two sides are compared with the sort's own total order
 * ([[RunMerge.compareKeys]], so nulls, doubles and strings compare as Spark does) and equal runs
 * emit their cross product as index pairs in left order then right order -- Spark's order -- in
 * chunks of 8192 pairs gathered through the same kernels as the hash join, `-1` padding an outer
 * row's missing side. Null keys never match: a null run is emitted unmatched for outer types and
 * skipped otherwise. A condition is evaluated over the gathered pairs and the survivors set the
 * per-row matched flags that semi, anti, existence and the outer types read.
 */
case class VectorSortMergeJoinExec(
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    joinType: JoinType,
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan
) extends VectorBinaryExec {

  override def output: Seq[Attribute] = joinType match {
    case _: InnerLike => left.output ++ right.output
    case LeftOuter => left.output ++ right.output.map(_.withNullability(true))
    case RightOuter => left.output.map(_.withNullability(true)) ++ right.output
    case FullOuter => left.output.map(_.withNullability(true)) ++ right.output.map(_.withNullability(true))
    case LeftSemi | LeftAnti => left.output
    case ExistenceJoin(exists) => left.output :+ exists
    case _ => left.output ++ right.output
  }

  private def joinedOutput: Seq[Attribute] = left.output ++ right.output

  override def requiredChildDistribution: Seq[Distribution] =
    ClusteredDistribution(leftKeys) :: ClusteredDistribution(rightKeys) :: Nil

  override def requiredChildOrdering: Seq[Seq[SortOrder]] =
    Seq(leftKeys.map(SortOrder(_, Ascending)), rightKeys.map(SortOrder(_, Ascending)))

  /** As Spark's SortMergeJoinExec: the preserved side's order, none for a full outer join. */
  override def outputOrdering: Seq[SortOrder] = joinType match {
    case _: InnerLike | LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin => left.outputOrdering
    case RightOuter => right.outputOrdering
    case _ => Nil
  }

  override def outputPartitioning: Partitioning = joinType match {
    case _: InnerLike => PartitioningCollection(Seq(left.outputPartitioning, right.outputPartitioning))
    case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin => left.outputPartitioning
    case RightOuter => right.outputPartitioning
    case FullOuter => UnknownPartitioning(left.outputPartitioning.numPartitions)
    case _ => left.outputPartitioning
  }

  @transient private lazy val compiledLeftKeys: Array[VectorExpr] = VectorJoinPlanner.compileKeys(leftKeys, left.output)
  @transient private lazy val compiledRightKeys: Array[VectorExpr] =
    VectorJoinPlanner.compileKeys(rightKeys, right.output)
  @transient private lazy val compiledCondition: Option[VectorExpr] =
    condition.map(c =>
      VectorJoinPlanner.compileCondition(
        c,
        joinedOutput
      ).fold(r => throw new IllegalStateException(s"cannot vectorize join condition: $r"), identity)
    )

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    // Spark streams the preserved side: a right outer join walks the right rows and buffers the left
    // key group, so its output is in right order. The iterator streams its first input and buffers
    // the second; a right outer join runs it with the sides swapped, as a left outer join whose
    // columns are laid out back in Spark's left ++ right order.
    val swapped = joinType == RightOuter
    val spec = MergeJoinSpec(
      if (swapped) LeftOuter else joinType,
      if (swapped) compiledRightKeys else compiledLeftKeys,
      if (swapped) compiledLeftKeys else compiledRightKeys,
      compiledCondition,
      output.map(a => (a.name, a.dataType)).toArray,
      joinedOutput.map(a => (a.name, a.dataType)).toArray,
      left.output.length,
      right.output.length,
      swapped
    )
    val m = vectorMetrics
    if (swapped) {
      right.executeColumnar().zipPartitions(left.executeColumnar(), preservesPartitioning = true) { (r, l) =>
        new VectorSortMergeJoinIterator(r, l, spec, m)
      }
    } else {
      left.executeColumnar().zipPartitions(right.executeColumnar(), preservesPartitioning = true) { (l, r) =>
        new VectorSortMergeJoinIterator(l, r, spec, m)
      }
    }
  }

  override protected def withNewChildrenInternal(newLeft: SparkPlan, newRight: SparkPlan): SparkPlan =
    copy(left = newLeft, right = newRight)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Left keys: ${leftKeys.map(_.sql).mkString(", ")}
       |Right keys: ${rightKeys.map(_.sql).mkString(", ")}
       |Join type: $joinType
       |Condition: ${condition.map(_.sql).getOrElse("None")}
       |Output: ${output.map(_.name).mkString(", ")}
       |${getTagValue(VectorExecRule.SortMergeWhy).map(w => s"Sort-merge join $w").getOrElse("")}
       |""".stripMargin
  }
}

final case class MergeJoinSpec(
    joinType: JoinType,
    leftKeys: Array[VectorExpr],
    rightKeys: Array[VectorExpr],
    condition: Option[VectorExpr],
    /** The operator's output: left ++ right, the left side alone for semi/anti, left plus the exists flag. */
    outputAttrs: Array[(String, DataType)],
    /** Always left ++ right: the row the condition is evaluated on. */
    joinedAttrs: Array[(String, DataType)],
    leftWidth: Int,
    rightWidth: Int,
    /** The iterator streams Spark's right side and buffers the left (a right outer join). */
    swapped: Boolean = false
)

/**
 * The merge over two sorted partitions. Both sides arrive as batches; a side's batch becomes a
 * [[SortedRows]] -- its output columns, its evaluated keys and its run starts -- borrowed from the
 * batch when it carries no selection, compacted into the iterator's arena otherwise.
 */
private[vecruntime] class VectorSortMergeJoinIterator(
    leftInput: Iterator[ColumnarBatch],
    rightInput: Iterator[ColumnarBatch],
    spec: MergeJoinSpec,
    metrics: VectorMetrics
) extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val OutputBatchSize = 8192

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorSortMergeJoinExec")
  private val pending = new ArrayDeque[ColumnarBatch]()
  private var emitted: ColumnarBatch = _
  private var closed = false
  private var leftDone = false
  private var drained = false

  private val joinType = spec.joinType
  private val emitsRight =
    joinType match { case _: InnerLike | LeftOuter | RightOuter | FullOuter => true; case _ => false }
  private val leftOuter = joinType == LeftOuter || joinType == FullOuter
  private val rightOuter = joinType == RightOuter || joinType == FullOuter
  private val leftOnly = joinType match { case LeftSemi | LeftAnti | _: ExistenceJoin => true; case _ => false }
  private val numKeys = spec.leftKeys.length
  private val ascending = Array.fill(numKeys)(true)
  private val nullsFirst = Array.fill(numKeys)(true)

  /** A batch of one side as sorted rows: columns, keys, run starts (with the end sentinel). */
  private final class SortedRows(
      val columns: Array[VectorBuffers],
      val keys: Array[VectorBuffers],
      val rows: Int,
      val arena: Arena
  ) {
    val runStarts: Array[Int] = {
      val bits = Bitmap.allocate(arena, math.max(rows, 1))
      RunKernels.boundaries(keys, rows, bits)
      RunKernels.runStarts(bits, rows)
    }
    def numRuns: Int = runStarts.length - 1
    def keyIsNull(row: Int): Boolean = {
      var k = 0
      var nul = false
      while (k < numKeys && !nul) { nul = keys(k).isNull(row); k += 1 }
      nul
    }

    /** Which rows matched (right and full outer joins over a right batch): one bitmap per batch, allocated on the first match. */
    private var matchedBits: MemorySegment = _
    def setMatched(row: Int): Unit = {
      if (matchedBits == null) matchedBits = Bitmap.allocate(arena, math.max(rows, 1))
      Bitmap.set(matchedBits, row)
    }
    def isMatched(row: Int): Boolean = matchedBits != null && Bitmap.isSet(matchedBits, row)
  }

  /**
   * The right side's current run as cursor state, one object for the whole task: rows
   * `[offset, offset + rows)` of `columns` / `keys`. A run inside one batch is a view of the batch's
   * copy (`arena` null; `batch` is the SortedRows it lives in, whose bitmap records its matches) --
   * nothing is allocated per run, which is what #310 is about: with ~4-row runs the per-run object,
   * bitmap and call machinery were the join. A run reaching a batch's edge is copied into its own
   * arena with its own bitmap.
   */
  private final class RunCursor {
    var columns: Array[VectorBuffers] = _
    var keys: Array[VectorBuffers] = _
    var offset: Int = 0
    var rows: Int = 0
    var arena: Arena = _
    var batch: SortedRows = _
    var keyIsNull: Boolean = false
    private var ownMatched: MemorySegment = _

    def setView(b: SortedRows, from: Int, to: Int): Unit = {
      release()
      columns = b.columns; keys = b.keys; offset = from; rows = to - from; arena = null; batch = b
      keyIsNull = b.keyIsNull(from)
    }
    def setCopy(cols: Array[VectorBuffers], ks: Array[VectorBuffers], n: Int, a: Arena): Unit = {
      release()
      columns = cols; keys = ks; offset = 0; rows = n; arena = a; batch = null
      keyIsNull = { var k = 0; var nul = false; while (k < numKeys && !nul) { nul = ks(k).isNull(0); k += 1 }; nul }
    }
    def setMatched(row: Int): Unit =
      if (batch != null) batch.setMatched(offset + row)
      else {
        if (ownMatched == null) ownMatched = Bitmap.allocate(arena, math.max(rows, 1)); Bitmap.set(ownMatched, row)
      }
    def isMatched(row: Int): Boolean =
      if (batch != null) batch.isMatched(offset + row) else ownMatched != null && Bitmap.isSet(ownMatched, row)

    /** Releases a copied run's arena; a view owns nothing. */
    def release(): Unit = {
      if (arena != null) { arena.close(); arena = null }; ownMatched = null; batch = null; columns = null
    }
  }

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  // ---------------------------------------------------------------- the right side, run by run

  /** The right batch being cut into runs, and the position of its next run. */
  private var rightRows: SortedRows = _
  private var rightRunIdx = 0
  private var rightDone = false
  private val current = new RunCursor

  /** Whether `current` holds a run (false once the right side is exhausted). */
  private var haveRun = false

  /** Advances `current` to the right side's next run; false when exhausted. A run reaching a batch's edge continues into the next batch while the key is the same. */
  private def nextRightRun(): Boolean = {
    if (candCount > 0 && (candRightCursor eq current)) flushCandidates()
    current.release()
    haveRun = false
    if (!ensureRightBatch()) return false
    var from = rightRows.runStarts(rightRunIdx)
    var to = rightRows.runStarts(rightRunIdx + 1)
    if (to < rightRows.rows) {
      // The common run, inside the batch: cursor state over the batch's copy, nothing allocated.
      rightRunIdx += 1
      current.setView(rightRows, from, to)
      haveRun = true
      return true
    }
    val arena =
      Arena.ofConfined() // consumed on the task thread; a shared arena's close is a handshake with every thread, ruinous per run
    val builders = rightRows.columns.map(c => new ColumnBuilder(arena, c.`type`(), to - from))
    val keyBuilders = rightRows.keys.map(k => new ColumnBuilder(arena, k.`type`(), to - from))
    var rows = 0
    var more = true
    while (more) {
      val sel = rangeSelection(rightRows.arena, rightRows.rows, from, to)
      var c = 0
      while (c < builders.length) { builders(c).append(rightRows.columns(c), sel, to - from); c += 1 }
      var k = 0
      while (k < keyBuilders.length) { keyBuilders(k).append(rightRows.keys(k), sel, to - from); k += 1 }
      rows += to - from
      rightRunIdx += 1
      more = to == rightRows.rows && ensureRightBatch() && sameKey(keyBuilders, rightRows)
      if (more) { from = 0; to = rightRows.runStarts(1) }
    }
    current.setCopy(builders.map(_.view()), keyBuilders.map(_.view()), rows, arena)
    haveRun = true
    true
  }

  /** A batch with runs left to read, loading the next one when needed. */
  private def ensureRightBatch(): Boolean = {
    if (rightRows != null && rightRunIdx < rightRows.numRuns) return true
    if (rightRows != null) { releasingRight(rightRows.columns); rightRows.arena.close(); rightRows = null }
    if (rightDone) return false
    val loaded = loadRightBatch()
    if (!loaded) rightDone = true
    loaded
  }

  /** Whether the first row of `rows` shares every key with the run being buffered (nulls included). */
  private def sameKey(keyBuilders: Array[ColumnBuilder], rows: SortedRows): Boolean = {
    var k = 0
    var same = true
    while (k < numKeys && same) {
      same = RunMerge.compareKeys(keyBuilders(k).view(), 0, rows.keys(k), 0, false, true) == 0
      k += 1
    }
    same
  }

  private def rangeSelection(arena: Arena, n: Int, from: Int, to: Int): MemorySegment = {
    val sel = Bitmap.allocate(arena, math.max(n, 1))
    var i = from
    while (i < to) { Bitmap.set(sel, i); i += 1 }
    sel
  }

  private def loadRightBatch(): Boolean = {
    while (rightInput.hasNext) {
      val batch = rightInput.next()
      if (batch.numRows() > 0) {
        metrics.numInputBatches += 1
        val rows = sortedRows(batch, spec.rightKeys)
        if (rows.rows > 0) {
          rightRows = rows
          rightRunIdx = 0
          return true
        }
        rows.arena.close()
      }
    }
    false
  }

  /**
   * A batch as sorted rows in a fresh shared arena: the columns adapted and the keys evaluated; a
   * batch with a selection is compacted first so rows are dense.
   */
  private def sortedRows(batch: ColumnarBatch, keyExprs: Array[VectorExpr]): SortedRows = {
    val arena =
      Arena.ofConfined() // consumed on the task thread; a shared arena's close is a handshake with every thread, ruinous per run
    EvalContexts.withBatch(batch) { ctx =>
      val count = ctx.selectedCount
      val width = batch.numCols()
      val columns = new Array[ColumnBuilder](width)
      var c = 0
      while (c < width) {
        val b = new ColumnBuilder(arena, ctx.input(c).`type`(), math.max(count, 1))
        b.append(ctx.input(c), ctx.selection, count)
        columns(c) = b
        c += 1
      }
      val keys = keyExprs.map { k =>
        val b = new ColumnBuilder(arena, k.vecType, math.max(count, 1))
        b.append(k.eval(ctx), ctx.selection, count)
        b
      }
      new SortedRows(columns.map(_.view()), keys.map(_.view()), count, arena)
    }
  }

  // ---------------------------------------------------------------- the left side, batch by batch

  override def hasNext: Boolean = {
    while (pending.isEmpty && !drained) {
      if (!leftDone) {
        if (leftInput.hasNext) {
          val batch = leftInput.next()
          if (batch.numRows() > 0) {
            metrics.numInputBatches += 1
            metrics.timed(joinLeftBatch(batch))
          }
        } else {
          leftDone = true
        }
      } else {
        metrics.timed(drainRight())
        drained = true
      }
    }
    !pending.isEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more joined rows")
    if (emitted != null) { emitted.close(); emitted = null }
    emitted = pending.poll()
    metrics.numOutputBatches += 1
    metrics.numOutputRows += emitted.numRows()
    emitted
  }

  /** Every right run left once the left side is done: unmatched for right/full outer joins. */
  private def drainRight(): Unit = {
    if (!haveRun && !rightDone) nextRightRun()
    while (haveRun) {
      if (rightOuter) emitRightUnmatched(current)
      if (current.arena != null) releasingRight(current.columns)
      nextRightRun()
    }
    flushRows()
  }

  private def joinLeftBatch(batch: ColumnarBatch): Unit = {
    val rows = sortedRows(batch, spec.leftKeys)
    try {
      if (!haveRun && !rightDone) nextRightRun()
      val leftMatched = if (leftOuter || leftOnly) Bitmap.allocate(rows.arena, math.max(rows.rows, 1)) else null
      var r = 0
      while (r < rows.numRuns) {
        val from = rows.runStarts(r)
        val to = rows.runStarts(r + 1)
        var matchedRun = false
        if (!rows.keyIsNull(from)) {
          // Pass the right runs that sort before this key (unmatched for right/full outer).
          var cmp = if (!haveRun) 1 else compareRuns(rows, from, current)
          while (haveRun && cmp < 0) {
            if (rightOuter) emitRightUnmatched(current)
            if (current.arena != null) releasingRight(current.columns)
            nextRightRun()
            cmp = if (!haveRun) 1 else compareRuns(rows, from, current)
          }
          if (haveRun && cmp == 0) {
            emitRun(rows, from, to, current, leftMatched)
            matchedRun = true
          }
        }
        // A left run with no equal right run: its rows leave padded, in place, for the outer types.
        if (!matchedRun && leftOuter) {
          flushCandidates(); var i = from; while (i < to) { appendRow(rows, null, i, -1); i += 1 }
        }
        r += 1
      }
      flushCandidates() // the left batch's last pairs, before its matched bits are read or it is released
      if (leftOnly) emitLeftOnly(rows, leftMatched)
      flushRows() // the left batch is released below
    } finally {
      rows.arena.close()
    }
  }

  /** The left run's key against the right run's: negative when the right run sorts first. */
  private def compareRuns(left: SortedRows, leftRow: Int, right: RunCursor): Int = {
    if (right.keyIsNull) return -1 // a null run on the right never matches and sorts first (nulls first)
    if (numKeys == 1) return RunMerge.compareKeys(right.keys(0), right.offset, left.keys(0), leftRow, false, true)
    var k = 0
    var c = 0
    while (k < numKeys && c == 0) {
      c = RunMerge.compareKeys(right.keys(k), right.offset, left.keys(k), leftRow, false, true)
      k += 1
    }
    c
  }

  // ---------------------------------------------------------------- emission

  /**
   * The cross product of the left run `[from, to)` and the right run, in Spark's order: for each
   * left row its pairs in right order, or -- under a condition that rejected them all -- the row
   * padded with nulls for the outer types. Chunks hold whole left rows when the right run fits a
   * batch; a wider right run spreads one left row over several chunks.
   */
  private def emitRun(left: SortedRows, from: Int, to: Int, right: RunCursor, leftMatched: MemorySegment): Unit = {
    val perRow = right.rows
    if (spec.condition.isDefined && (to - from).toLong * perRow <= OutputBatchSize) {
      // A short run under a condition: its pairs join the candidate buffer and are gathered and
      // tested with ~2000 other runs' pairs at once, instead of a 16-row Arrow batch per run.
      queueCandidates(left, from, to, right, leftMatched)
      return
    }
    flushCandidates()
    if (perRow <= OutputBatchSize) {
      val rowsPerChunk = math.max(1, OutputBatchSize / perRow)
      var i = from
      while (i < to) {
        val end = math.min(to, i + rowsPerChunk)
        emitChunk(left, right, i, end, 0, perRow, leftMatched, padHere = true)
        i = end
      }
    } else {
      var i = from
      while (i < to) {
        var any = false
        var j = 0
        while (j < perRow) {
          val jEnd = math.min(perRow, j + OutputBatchSize)
          any |= emitChunk(left, right, i, i + 1, j, jEnd, leftMatched, padHere = false)
          j = jEnd
        }
        if (!any && leftOuter) emitLeftPadded(left, Array(i))
        i += 1
      }
    }
  }

  /**
   * The pairs of left rows `[i0, i1)` with right rows `[j0, j1)`: gathered, filtered by the
   * condition, the matches recorded, then queued in order with the pads of the left rows nothing
   * survived for (when `padHere`). Returns whether any pair survived.
   */
  private def emitChunk(
      left: SortedRows,
      right: RunCursor,
      i0: Int,
      i1: Int,
      j0: Int,
      j1: Int,
      leftMatched: MemorySegment,
      padHere: Boolean
  ): Boolean = {
    val width = j1 - j0
    val count = (i1 - i0) * width
    spec.condition match {
      case None =>
        // Every pair survives: matches recorded and the pairs appended to the ordered row buffer,
        // which gathers 8192 at a time -- with 4-row runs one gather then carries ~2000 runs' worth
        // of pairs, where a gather per run built an Arrow batch of 4 rows.
        var i = i0
        while (i < i1) {
          if (leftMatched != null) Bitmap.set(leftMatched, i)
          var j = j0
          while (j < j1) {
            if (rightOuter) right.setMatched(j)
            if (emitsRight) appendRow(left, right, i, j)
            j += 1
          }
          i += 1
        }
        true
      case Some(cond) =>
        val leftIdx = new Array[Int](count)
        val rightIdx = new Array[Int](count)
        var p = 0
        var i = i0
        while (i < i1) {
          var j = j0
          while (j < j1) { leftIdx(p) = i; rightIdx(p) = j; p += 1; j += 1 }
          i += 1
        }
        val joined = gatherColumns(
          left,
          right.columns,
          spec.joinedAttrs,
          leftIdx,
          shifted(rightIdx, right.offset, count),
          count,
          conditionRefs
        )
        val (sel, survivors) = EvalContexts.withBatch(joined) { ctx =>
          val pred = cond.eval(ctx)
          val (s, n) = VectorExpr.selection(pred, ctx)
          // The selection lives in the context's arena: copy the bits out before it closes.
          val copy = new Array[Long](Bitmap.wordsFor(count))
          var w = 0
          while (w < copy.length) { copy(w) = Bitmap.wordAt(s, w, count); w += 1 }
          (copy, n)
        }
        joined.close()
        val selected = (p: Int) => (sel(p >>> 6) & (1L << p)) != 0L
        if (leftMatched != null || rightOuter) {
          var q = 0
          while (q < count) {
            if (selected(q)) {
              if (leftMatched != null) Bitmap.set(leftMatched, leftIdx(q))
              if (rightOuter) right.setMatched(rightIdx(q))
            }
            q += 1
          }
        }
        if (emitsRight) {
          // Survivors in order, a pad for a left row that lost every pair.
          var q = 0
          i = i0
          while (i < i1) {
            var any = false
            var j = 0
            while (j < width) {
              if (selected(q)) { appendRow(left, right, i, rightIdx(q)); any = true }
              q += 1
              j += 1
            }
            if (!any && leftOuter && padHere) appendRow(left, null, i, -1)
            i += 1
          }
        }
        survivors > 0
    }
  }

  /** Run-relative right indices as batch rows: shifted by the run's offset, `-1` pads kept. */
  private def shifted(idx: Array[Int], offset: Int, count: Int): Array[Int] =
    if (offset == 0) idx
    else {
      val out = new Array[Int](count)
      var i = 0
      while (i < count) { val j = idx(i); out(i) = if (j < 0) -1 else j + offset; i += 1 }
      out
    }

  /** Left rows padded with a null right side, in the given order (left and full outer joins). */
  private def emitLeftPadded(left: SortedRows, idx: Array[Int]): Unit = {
    var i = 0
    while (i < idx.length) { appendRow(left, null, idx(i), -1); i += 1 }
  }

  // ------------------------------------------------------------ the ordered row buffer
  // Single rows -- a padded left row, a right-only row, a condition's survivors -- are buffered in
  // output order and gathered together, up to 8192 at a time: a full outer join over unique keys
  // would otherwise emit one Arrow batch per row. The buffer flushes when it is full, when a
  // source it references changes or is about to be released, and before a direct chunk gather.

  private val bufLeft = new Array[Int](OutputBatchSize)
  private val bufRight = new Array[Int](OutputBatchSize)
  private var bufCount = 0
  private var bufLeftRows: SortedRows = _
  private var bufRightCols: Array[VectorBuffers] = _

  /** Buffers one output row: left row `l` (or -1) of `left`, right row `r` (run-relative, or -1) of `right`. */
  private def appendRow(left: SortedRows, right: RunCursor, l: Int, r: Int): Unit =
    appendRowAbs(left, if (right == null || r < 0) null else right.columns, l, if (r < 0) -1 else r + right.offset)

  /** As [[appendRow]] with the right row absolute over `rightCols`. */
  private def appendRowAbs(left: SortedRows, rightCols: Array[VectorBuffers], l: Int, r: Int): Unit = {
    val rc = if (r < 0) null else rightCols
    val ls = if (l < 0) null else left
    if (
      bufCount == OutputBatchSize
      || (ls != null && bufLeftRows != null && (ls ne bufLeftRows))
      || (rc != null && bufRightCols != null && (rc ne bufRightCols))
    ) flushRows()
    if (ls != null) bufLeftRows = ls
    if (rc != null) bufRightCols = rc
    bufLeft(bufCount) = l
    bufRight(bufCount) = r
    bufCount += 1
  }

  // ------------------------------------------------------------ the candidate buffer (#310)
  // Under a condition, short runs' pairs are collected here in output order and tested together:
  // one gather of the joined columns and one predicate evaluation per 8192 pairs. A run's pairs
  // never straddle a flush, so a left row's pad (outer types) is decided inside the flush that
  // holds its pairs. Flushed before anything that must follow the pairs in Spark's order -- a pad
  // for an unmatched left run, a passed right run's unmatched rows, the semi/anti emission, the
  // end of the left batch -- and before a right source is released.

  private val candLeft = new Array[Int](OutputBatchSize)
  private val candRight = new Array[Int](OutputBatchSize) // absolute over candRightCols
  private var candCount = 0
  private var candLeftRows: SortedRows = _
  private var candLeftMatched: MemorySegment = _
  private var candRightCols: Array[VectorBuffers] = _

  /** Where a right match is recorded: the batch the runs are views of, or the cursor holding a copied run. */
  private var candRightBatch: SortedRows = _
  private var candRightCursor: RunCursor = _

  private def queueCandidates(
      left: SortedRows,
      from: Int,
      to: Int,
      right: RunCursor,
      leftMatched: MemorySegment
  ): Unit = {
    val n = (to - from) * right.rows
    if (
      candCount + n > OutputBatchSize
      || (candCount > 0 && ((left ne candLeftRows) || (right.columns ne candRightCols)))
    ) flushCandidates()
    candLeftRows = left
    candLeftMatched = leftMatched
    candRightCols = right.columns
    candRightBatch = right.batch
    candRightCursor = if (right.batch == null) right else null
    var i = from
    while (i < to) {
      var j = 0
      while (j < right.rows) {
        candLeft(candCount) = i; candRight(candCount) = right.offset + j; candCount += 1; j += 1
      }
      i += 1
    }
  }

  private def flushCandidates(): Unit = if (candCount > 0) {
    val count = candCount
    val left = candLeftRows
    val rightCols = candRightCols
    val cond = spec.condition.get
    val joined = gatherColumns(left, rightCols, spec.joinedAttrs, candLeft, candRight, count, conditionRefs)
    val sel = EvalContexts.withBatch(joined) { ctx =>
      val (s, _) = VectorExpr.selection(cond.eval(ctx), ctx)
      val copy = new Array[Long](Bitmap.wordsFor(count))
      var w = 0
      while (w < copy.length) { copy(w) = Bitmap.wordAt(s, w, count); w += 1 }
      copy
    }
    joined.close()
    var q = 0
    while (q < count) {
      // One left row's pairs are contiguous: survivors in order, or its pad when none survived.
      val l = candLeft(q)
      var any = false
      while (q < count && candLeft(q) == l) {
        if ((sel(q >>> 6) & (1L << q)) != 0L) {
          any = true
          if (candLeftMatched != null) Bitmap.set(candLeftMatched, l)
          if (rightOuter) {
            if (candRightBatch != null) candRightBatch.setMatched(candRight(q))
            else candRightCursor.setMatched(candRight(q))
          }
          if (emitsRight) appendRowAbs(left, rightCols, l, candRight(q))
        }
        q += 1
      }
      if (!any && leftOuter) appendRowAbs(left, null, l, -1)
    }
    candCount = 0
    candLeftRows = null; candLeftMatched = null; candRightCols = null; candRightBatch = null; candRightCursor = null
  }

  private def flushRows(): Unit = if (bufCount > 0) {
    pending.add(gatherColumns(bufLeftRows, bufRightCols, spec.outputAttrs, bufLeft, bufRight, bufCount))
    bufCount = 0
    bufLeftRows = null
    bufRightCols = null
  }

  /** Flushes the buffer if it references the given right columns (they are about to be released). */
  private def releasingRight(cols: Array[VectorBuffers]): Unit = {
    if (candCount > 0 && (candRightCols eq cols)) flushCandidates()
    if (bufCount > 0 && (bufRightCols eq cols)) flushRows()
  }

  /** Left ++ right columns for the given pairs (right indices run-relative); a `-1` index pads the side with nulls. */
  private def gather(
      left: SortedRows,
      right: RunCursor,
      attrs: Array[(String, DataType)],
      leftIdx: Array[Int],
      rightIdx: Array[Int],
      count: Int
  ): ColumnarBatch =
    gatherColumns(
      left,
      if (right == null) null else right.columns,
      attrs,
      leftIdx,
      if (right == null) rightIdx else shifted(rightIdx, right.offset, count),
      count
    )

  /** As [[gather]] over the right columns directly, right indices absolute. */
  private def gatherColumns(
      left: SortedRows,
      rightCols: Array[VectorBuffers],
      attrs: Array[(String, DataType)],
      leftIdx: Array[Int],
      rightIdx: Array[Int],
      count: Int
  ): ColumnarBatch =
    gatherColumns(left, rightCols, attrs, leftIdx, rightIdx, count, null)

  /**
   * As above; with `only` set, the ordinals not in it are placeholders the evaluation context never
   * adapts (it adapts lanes lazily per ordinal), so a condition over two of thirty columns gathers two.
   */
  private def gatherColumns(
      left: SortedRows,
      rightCols: Array[VectorBuffers],
      attrs: Array[(String, DataType)],
      leftIdx: Array[Int],
      rightIdx: Array[Int],
      count: Int,
      only: Array[Boolean]
  ): ColumnarBatch = {
    val columns = new Array[ColumnVector](attrs.length)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = attrs(c)
      columns(c) =
        if (only != null && !only(c)) PlaceholderColumn
        else if ((c < spec.leftWidth) != spec.swapped) {
          // Spark's left side: the streamed rows, or the buffered run when the sides are swapped.
          val ordinal = if (spec.swapped) c - spec.leftWidth else c
          if (left == null) ArrowOutput.nulls(name, dt, count, allocator)
          else ArrowOutput.gather(name, dt, left.columns(ordinal), leftIdx, 0, count, allocator)
        } else {
          val ordinal = if (spec.swapped) c else c - spec.leftWidth
          if (rightCols == null) ArrowOutput.nulls(name, dt, count, allocator)
          else ArrowOutput.gather(name, dt, rightCols(ordinal), rightIdx, 0, count, allocator)
        }
      c += 1
    }
    new ColumnarBatch(columns, count)
  }

  /** The joined ordinals the condition reads. */
  private val conditionRefs: Array[Boolean] = spec.condition.map { cond =>
    val refs = new Array[Boolean](spec.joinedAttrs.length)
    def walk(e: VectorExpr): Unit = e match {
      case io.vecruntime.spark.expr.ColumnRef(o, _) => refs(o) = true
      case other => other.children.foreach(walk)
    }
    walk(cond)
    refs
  }.orNull

  /** A right run's rows without a match, the left side null (right and full outer joins). */
  private def emitRightUnmatched(right: RunCursor): Unit = {
    flushCandidates()
    var i = 0
    while (i < right.rows) {
      if (!right.isMatched(i)) appendRow(null, right, -1, i)
      i += 1
    }
  }

  /** Semi: the matched left rows; anti: the unmatched ones; existence: every left row plus the flag. */
  private def emitLeftOnly(left: SortedRows, leftMatched: MemorySegment): Unit = {
    val n = left.rows
    val idx: Array[Int] = joinType match {
      case LeftSemi => matchedRows(leftMatched, n)
      case LeftAnti => unmatchedRows(leftMatched, n)
      case _ => Array.tabulate(n)(identity)
    }
    if (idx.isEmpty) return
    emitInChunks(idx) { (chunkIdx, count) =>
      val columns = new Array[ColumnVector](spec.outputAttrs.length)
      var c = 0
      while (c < spec.leftWidth) {
        val (name, dt) = spec.outputAttrs(c)
        columns(c) = ArrowOutput.gather(name, dt, left.columns(c), chunkIdx, 0, count, allocator)
        c += 1
      }
      if (joinType.isInstanceOf[ExistenceJoin]) {
        val (name, dt) = spec.outputAttrs(spec.leftWidth)
        val exists = SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, leftMatched)
        columns(spec.leftWidth) = ArrowOutput.gather(name, dt, exists, chunkIdx, 0, count, allocator)
      }
      pending.add(new ColumnarBatch(columns, count))
    }
  }

  private def emitInChunks(idx: Array[Int])(emit: (Array[Int], Int) => Unit): Unit = {
    var from = 0
    while (from < idx.length) {
      val to = math.min(idx.length, from + OutputBatchSize)
      val chunk = if (from == 0 && to == idx.length) idx else java.util.Arrays.copyOfRange(idx, from, to)
      emit(chunk, to - from)
      from = to
    }
  }

  private def matchedRows(bits: MemorySegment, n: Int): Array[Int] = rowsWhere(bits, n, set = true)
  private def unmatchedRows(bits: MemorySegment, n: Int): Array[Int] = rowsWhere(bits, n, set = false)

  private def rowsWhere(bits: MemorySegment, n: Int, set: Boolean): Array[Int] = {
    val out = new Array[Int](n)
    var o = 0
    var i = 0
    while (i < n) {
      if (Bitmap.isSet(bits, i) == set) { out(o) = i; o += 1 }
      i += 1
    }
    if (o == n) out else java.util.Arrays.copyOf(out, o)
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      if (emitted != null) { emitted.close(); emitted = null }
      while (!pending.isEmpty) pending.poll().close()
      current.release()
      if (rightRows != null) { rightRows.arena.close(); rightRows = null }
      allocator.close()
    }
  }
}
