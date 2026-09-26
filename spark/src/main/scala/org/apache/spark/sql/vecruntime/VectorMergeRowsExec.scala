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

import java.lang.foreign.MemorySegment
import java.util.ArrayDeque

import org.roaringbitmap.longlong.Roaring64Bitmap

import io.vecruntime.kernels.{Bitmap, BitmapKernels, VecType}
import io.vecruntime.spark.adapter.TypeMapping
import io.vecruntime.spark.arrow.{ArrowOutput, RemappedColumnVector, SelectedColumnarBatch}
import io.vecruntime.spark.expr.{ColumnRef, EvalContext, ExpressionCompiler, LiteralExpr, NullLiteralExpr, VectorExpr}
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, Expression, Literal, SortOrder}
import org.apache.spark.sql.catalyst.plans.logical.MergeRows
import org.apache.spark.sql.catalyst.plans.logical.MergeRows.{Discard, Instruction, Keep, Split}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.errors.QueryExecutionErrors
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.v2.MergeRowsExec
import org.apache.spark.sql.types.{BooleanType, DataType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for Spark's `MergeRowsExec`, the operator of a row-level `MERGE INTO` that
 * turns the joined target/source rows into the delta records the writer receives (#21). Spark's
 * operator decides each row alone: which of the three groups it belongs to (matched, not matched,
 * not matched by source -- from the two presence predicates), the first instruction of that group
 * whose condition holds, and what that instruction emits: `Keep` projects the row, `Discard` drops
 * it, `Split` projects it twice (an update as a delete plus an insert). Over a batch the same
 * decisions are masks: the group masks from the presence predicates, then, clause by clause, the
 * condition evaluated over the rows still undecided; each `Keep`/`Split` output is that clause's
 * projection compacted by its mask, so a batch becomes at most one output batch per projection.
 * Rows a group leaves undecided are dropped, as Spark drops them.
 *
 * The cardinality check is Spark's: when the merge carries a `__row_id` (a target row matched by
 * more than one source row is an error), the row ids of the matched rows go into one roaring bitmap
 * per partition and a repeat raises `MERGE_CARDINALITY_VIOLATION`, exactly the error Spark raises,
 * before any output of the batch is emitted.
 *
 * Output order differs from Spark's -- rows are grouped by clause within a batch instead of
 * interleaved in join order -- which the delta write does not depend on: the join already delivers
 * an arbitrary order and the writer sorts or buffers what it needs.
 */
case class VectorMergeRowsExec(
    isSourceRowPresent: Expression,
    isTargetRowPresent: Expression,
    matchedInstructions: Seq[Instruction],
    notMatchedInstructions: Seq[Instruction],
    notMatchedBySourceInstructions: Seq[Instruction],
    checkCardinality: Boolean,
    output: Seq[Attribute],
    child: SparkPlan
) extends VectorExec {

  override def outputOrdering: Seq[SortOrder] = Nil
  override def outputPartitioning: Partitioning = child.outputPartitioning

  /** Compiled on the driver; failure here is a bug because the rule already checked it. */
  @transient private lazy val program: MergeProgram =
    VectorMergeRowsPlanner.compile(this).fold(
      r => throw new IllegalStateException(s"cannot vectorize merge: $r"),
      identity
    )

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val p = program
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter => new VectorMergeRowsIterator(iter, p, m) }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/** One clause of a group: its compiled condition and what it emits (none for Discard, one for Keep, two for Split). */
private[vector] final case class CompiledInstruction(condition: Option[VectorExpr], outputs: Seq[Array[CompiledOutput]])

/** One output column of a projection: a compiled expression, or a foreign (lane-less) column passed through by ordinal. */
private[vector] final case class CompiledOutput(name: String, dataType: DataType, expr: VectorExpr, foreignOrdinal: Int)

/** A boolean column that is `true` on every row of the batch: the presence predicate of a side every row has. */
private[vector] object AllRowsExpr extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Nil
  override def eval(ctx: EvalContext): io.vecruntime.kernels.VectorBuffers = {
    val bits = io.vecruntime.kernels.ArrowLayout.allocateBitmap(ctx.arena, ctx.numRows)
    io.vecruntime.kernels.Bitmap.fill(bits, ctx.numRows, true)
    io.vecruntime.kernels.SegmentVectorBuffers.fixedWidth(io.vecruntime.kernels.VecType.BOOL, ctx.numRows, null, bits)
  }
}

private[vector] final case class MergeProgram(
    sourcePresent: VectorExpr,
    targetPresent: VectorExpr,
    matched: Seq[CompiledInstruction],
    notMatched: Seq[CompiledInstruction],
    notMatchedBySource: Seq[CompiledInstruction],
    /** Ordinal of the `__row_id` column in the child's output, -1 when the cardinality check is off. */
    rowIdOrdinal: Int
)

object VectorMergeRowsPlanner {

  /** Why the merge would not compile over its child's output, input aside. */
  def reason(m: MergeRowsExec): Option[String] = compile(m).left.toOption

  private[vector] def compile(m: MergeRowsExec): Either[String, MergeProgram] =
    compile(
      m.isSourceRowPresent,
      m.isTargetRowPresent,
      m.matchedInstructions,
      m.notMatchedInstructions,
      m.notMatchedBySourceInstructions,
      m.checkCardinality,
      m.output,
      m.child
    )

  /** Why a hand-built operator would not compile (tests). */
  def reason(v: VectorMergeRowsExec): Option[String] = compile(v).left.toOption

  private[vector] def compile(v: VectorMergeRowsExec): Either[String, MergeProgram] =
    compile(
      v.isSourceRowPresent,
      v.isTargetRowPresent,
      v.matchedInstructions,
      v.notMatchedInstructions,
      v.notMatchedBySourceInstructions,
      v.checkCardinality,
      v.output,
      v.child
    )

  private def compile(
      isSourceRowPresent: Expression,
      isTargetRowPresent: Expression,
      matched: Seq[Instruction],
      notMatched: Seq[Instruction],
      notMatchedBySource: Seq[Instruction],
      checkCardinality: Boolean,
      output: Seq[Attribute],
      child: SparkPlan
  ): Either[String, MergeProgram] = {
    val input = child.output
    val rowIdOrdinal =
      if (!checkCardinality) Right(-1)
      else {
        val i = input.indexWhere(a => a.name.equalsIgnoreCase(MergeRows.ROW_ID))
        if (i < 0) Left(s"merge cardinality check without a ${MergeRows.ROW_ID} column")
        else if (input(i).dataType != org.apache.spark.sql.types.LongType)
          Left(s"${MergeRows.ROW_ID} of type ${input(i).dataType.simpleString}")
        else Right(i)
      }
    for {
      rowId <- rowIdOrdinal
      s <- presence(isSourceRowPresent, input).left.map(r => s"source presence: $r")
      t <- presence(isTargetRowPresent, input).left.map(r => s"target presence: $r")
      mi <- all(matched.map(instruction(_, input, output)))
      ni <- all(notMatched.map(instruction(_, input, output)))
      bi <- all(notMatchedBySource.map(instruction(_, input, output)))
    } yield MergeProgram(s, t, mi, ni, bi, rowId)
  }

  /**
   * A presence predicate. Spark passes a `true` literal for a side every row has (the source of a
   * right outer merge join, #273): every live row is present; a `false` or null literal none.
   */
  private def presence(e: Expression, input: Seq[Attribute]): Either[String, VectorExpr] = e match {
    case Literal(true, BooleanType) => Right(AllRowsExpr)
    case Literal(false, BooleanType) | Literal(null, BooleanType) => Right(NullLiteralExpr(BooleanType))
    case other => ExpressionCompiler.compilePredicate(other, input)
  }

  private def all[T](rs: Seq[Either[String, T]]): Either[String, Seq[T]] =
    rs.collectFirst { case Left(r) => r }.map(Left(_)).getOrElse(Right(rs.collect { case Right(v) => v }))

  private def instruction(
      i: Instruction,
      input: Seq[Attribute],
      output: Seq[Attribute]
  ): Either[String, CompiledInstruction] = {
    val outs: Either[String, Seq[Array[CompiledOutput]]] = i match {
      case Keep(_, _, out) => projection(out, input, output).map(Seq(_))
      case Discard(_) => Right(Nil)
      case Split(_, out, other) =>
        for (a <- projection(out, input, output); b <- projection(other, input, output)) yield Seq(a, b)
      case other => Left(s"merge instruction ${other.getClass.getSimpleName} not supported")
    }
    // An unconditional clause (`WHEN MATCHED THEN ...`) carries a `true` literal; a folded `false` never fires.
    val cond: Either[String, Option[VectorExpr]] = i.condition match {
      case Literal(true, BooleanType) => Right(None)
      case Literal(false, BooleanType) | Literal(null, BooleanType) => Right(Some(NullLiteralExpr(BooleanType)))
      case c => ExpressionCompiler.compilePredicate(c, input).left.map(r => s"merge clause condition: $r").map(Some(_))
    }
    for {
      c <- cond
      o <- outs
    } yield CompiledInstruction(c, o)
  }

  /**
   * One instruction's projection, column by column: a compiled expression of the output's lane type,
   * or -- for an output type with no lane (Iceberg's struct `_partition` metadata) -- the bare input
   * column it forwards, remapped by row id at emission like the project operator does (#19).
   */
  private def projection(
      exprs: Seq[Expression],
      input: Seq[Attribute],
      output: Seq[Attribute]
  ): Either[String, Array[CompiledOutput]] = {
    if (exprs.length != output.length) Left(s"merge projection of ${exprs.length} columns for ${output.length} outputs")
    else all(exprs.zip(output).map { case (e, out) =>
      if (TypeMapping.hasLane(out.dataType)) {
        val compiled = if (TypeMapping.isSupported(out.dataType)) ExpressionCompiler.compile(e, input)
        else ExpressionCompiler.compileLaneColumn(e, input)
        compiled.left.map(r => s"${e.sql}: $r").map(c => CompiledOutput(out.name, out.dataType, c, -1))
      } else e match {
        case a: AttributeReference =>
          val ordinal = input.indexWhere(_.exprId == a.exprId)
          if (ordinal < 0) Left(s"unbound attribute ${a.name}")
          else Right(CompiledOutput(out.name, out.dataType, null, ordinal))
        // An inserted row has no partition struct yet: a `NULL` of the lane-less type is a null column (#273).
        case Literal(null, _) => Right(CompiledOutput(out.name, out.dataType, NullLiteralExpr(out.dataType), -1))
        case other => Left(s"${other.sql}: unsupported output type ${out.dataType.simpleString} for ${out.name}")
      }
    }).map(_.toArray)
  }
}

private[vector] class VectorMergeRowsIterator(
    input: Iterator[ColumnarBatch],
    program: MergeProgram,
    metrics: VectorMetrics
) extends Iterator[ColumnarBatch] with AutoCloseable {

  private val allocator = io.vecruntime.spark.arrow.VectorAllocators.newChild("VectorMergeRowsExec")
  private val pending = new ArrayDeque[ColumnarBatch]()
  private var emitted: ColumnarBatch = _
  private var closed = false

  /** Row ids of the target rows matched so far in this partition (Spark's `BitmapCardinalityValidator`). */
  private val matchedRowIds: Roaring64Bitmap = if (program.rowIdOrdinal >= 0) new Roaring64Bitmap() else null

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = {
    while (pending.isEmpty && input.hasNext) {
      val raw = input.next()
      if (raw.numRows() > 0) {
        val in = InputBatches.normalize(raw)
        try metrics.timed { process(in) }
        finally if (in ne raw) in.close()
      }
    }
    !pending.isEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more merged batches")
    if (emitted != null) emitted.close()
    emitted = pending.poll()
    metrics.numOutputBatches += 1
    metrics.numOutputRows += emitted.numRows()
    emitted
  }

  private def process(batch: ColumnarBatch): Unit = {
    metrics.numInputBatches += 1
    EvalContexts.withBatch(batch) { ctx =>
      val n = ctx.numRows
      // The presence predicates never yield null (they test a non-nullable marker), but a selection
      // on the batch narrows both, so every group mask is already within the live rows.
      val (source, _) = VectorExpr.selection(program.sourcePresent.eval(ctx), ctx)
      val (target, _) = VectorExpr.selection(program.targetPresent.eval(ctx), ctx)
      val matched = ctx.bitmap()
      BitmapKernels.and(source, target, matched, n)
      val notMatched = ctx.bitmap()
      BitmapKernels.andNot(source, target, notMatched, n)
      val notMatchedBySource = ctx.bitmap()
      BitmapKernels.andNot(target, source, notMatchedBySource, n)
      if (matchedRowIds != null) checkCardinality(ctx, matched, n)
      applyGroup(ctx, program.matched, matched, n)
      applyGroup(ctx, program.notMatched, notMatched, n)
      applyGroup(ctx, program.notMatchedBySource, notMatchedBySource, n)
    }
  }

  /** A target row matched twice is an error, raised before anything of the batch is emitted. */
  private def checkCardinality(ctx: EvalContext, matched: MemorySegment, n: Int): Unit = {
    val ids = ctx.input(program.rowIdOrdinal)
    var i = 0
    while (i < n) {
      if (Bitmap.isSet(matched, i)) {
        val id = ids.getLong(i)
        if (matchedRowIds.contains(id)) throw QueryExecutionErrors.mergeCardinalityViolationError()
        matchedRowIds.add(id)
      }
      i += 1
    }
  }

  /**
   * Spark's clause order over a group: each instruction's condition is evaluated over the rows the
   * earlier ones left undecided, and the rows it takes are removed from that set. A condition that
   * is null counts as false, as in Spark's `BasePredicate`.
   */
  private def applyGroup(
      ctx: EvalContext,
      instructions: Seq[CompiledInstruction],
      group: MemorySegment,
      n: Int
  ): Unit = {
    if (instructions.isEmpty || Bitmap.popcount(group, n) == 0) return
    val remaining = ctx.bitmap()
    BitmapKernels.copy(group, remaining, n)
    var left = Bitmap.popcount(remaining, n)
    val it = instructions.iterator
    while (left > 0 && it.hasNext) {
      val instr = it.next()
      val hit = ctx.bitmap()
      instr.condition match {
        case None => BitmapKernels.copy(remaining, hit, n) // unconditional: every undecided row
        case Some(cond) =>
          // Only the undecided rows matter: evaluate the condition with them as the active set.
          val pred = ctx.withActive(remaining)(cond.eval(ctx))
          BitmapKernels.selection(pred.data(), pred.validity(), hit, n)
          BitmapKernels.and(hit, remaining, hit, n)
      }
      val count = Bitmap.popcount(hit, n)
      if (count > 0) {
        instr.outputs.foreach(out => pending.add(emit(ctx, out, hit, count, n)))
        BitmapKernels.andNot(remaining, hit, remaining, n)
        left -= count
      }
    }
  }

  /** One projection compacted by its mask: every column is a fresh, owned vector (several outputs share one input batch). */
  private def emit(
      ctx: EvalContext,
      outputs: Array[CompiledOutput],
      mask: MemorySegment,
      count: Int,
      n: Int
  ): ColumnarBatch = {
    val columns = new Array[ColumnVector](outputs.length)
    var foreignRows: Array[Int] = null
    var c = 0
    while (c < outputs.length) {
      val o = outputs(c)
      columns(c) =
        if (o.foreignOrdinal >= 0) {
          if (foreignRows == null) foreignRows = RemappedColumnVector.rowsOf(mask, n, count)
          RemappedColumnVector.of(ctx.column(o.foreignOrdinal), foreignRows)
        } else o.expr match {
          case _: NullLiteralExpr if !TypeMapping.hasLane(o.dataType) =>
            val v = new org.apache.spark.sql.execution.vectorized.ConstantColumnVector(count, o.dataType); v.setNull();
            v
          case lit: LiteralExpr if lit.value == null => ArrowOutput.nulls(o.name, o.dataType, count, allocator)
          case lit: LiteralExpr => ArrowOutput.constant(o.name, o.dataType, lit.value, count, allocator)
          case ColumnRef(ordinal, _) =>
            ArrowOutput.compact(o.name, o.dataType, ctx.input(ordinal), mask, count, allocator)
          case e => ArrowOutput.compact(o.name, o.dataType, ctx.withActive(mask)(e.eval(ctx)), mask, count, allocator)
        }
      c += 1
    }
    new ColumnarBatch(columns, count)
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      if (emitted != null) { emitted.close(); emitted = null }
      while (!pending.isEmpty) pending.poll().close()
      allocator.close()
    }
  }
}
