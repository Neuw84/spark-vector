package org.apache.spark.sql.vector

import io.sparkvector.kernels.{Bitmap, GroupAssignment, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.agg.{GroupedAggState, Rows, VectorAggFunction, VectorAggregates}
import io.sparkvector.spark.arrow.{ArrowOutput, BorrowedColumnVector, SelectedColumnarBatch, VectorAllocators}
import io.sparkvector.spark.expr.{ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, DenseRank, Expression, NamedExpression, Rank, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedFollowing, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete}
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.window.{Final, Partial, WindowExec, WindowGroupLimitExec, WindowGroupLimitMode}
import org.apache.spark.sql.types.{DataType, DecimalType, DoubleType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

import scala.collection.mutable

/**
 * Columnar replacement for WindowExec in two shapes (#58). Both walk the child's rows in the order
 * Spark already established -- the child arrives sorted by the partition keys and then the order keys,
 * because this operator requires exactly what `WindowExec` does -- and detect a partition boundary
 * where a partition key differs from the previous row's (null-safe equality, as Spark's window
 * functions compare with `<=>`).
 *
 * Ranking (layer 1): `row_number`, `rank` and `dense_rank`. A row starts a new peer group when an
 * order key differs from the previous row's; `row_number` counts rows since the partition began,
 * `rank` is the row number at which the current peer group began, `dense_rank` counts peer groups.
 * The previous row's keys and the counters carry across batches, so a partition longer than a batch
 * is one partition. Streaming: every input batch is one output batch, the input columns borrowed,
 * the ranks new INT32 columns.
 *
 * Whole-partition aggregates (layer 2): `sum`, `avg`, `count`, `min`, `max` and the rest of the
 * grouped aggregate family over the frame `UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` -- Spark's
 * default frame for a spec without `ORDER BY`, the shape of TPC-DS q12, q20, q47, q53, q57, q63, q89,
 * q98. Each partition is a group: the aggregate functions of the grouped hash aggregate run over the
 * batch with the partition ordinal as the group id, and the partition's value is the function's
 * `evaluateExpression` over its buffers, exactly as the Final aggregate computes it. Rows are held
 * (copied, in memory, no spill -- like the sort) until their partition ends, since the value is
 * known only then, and are emitted batch by batch with the values gathered per row.
 *
 * The child may be a row operator: Spark plans `Window` above `Sort` above an exchange, and without
 * Comet's shuffle that sort stays Spark's, so `RowToColumnarExec` is inserted below us; from here up
 * the chain (a filter on the rank, a projection, a limit) is columnar again.
 *
 * Running and sliding frames, offset functions (`lag`, `lead`, `first_value`), `percent_rank`,
 * `cume_dist`, `ntile` and `WindowGroupLimitExec` are later layers; each is refused with a reason
 * naming the function and frame.
 */
case class VectorWindowExec(
    windowExpression: Seq[NamedExpression],
    partitionSpec: Seq[Expression],
    orderSpec: Seq[SortOrder],
    child: SparkPlan)
    extends VectorExec {

  override def output: Seq[Attribute] = child.output ++ windowExpression.map(_.toAttribute)

  /** Same contracts as WindowExec, so the exchange and sort Spark planned below are the right ones. */
  override def requiredChildDistribution: Seq[Distribution] =
    if (partitionSpec.isEmpty) AllTuples :: Nil else ClusteredDistribution(partitionSpec) :: Nil
  override def requiredChildOrdering: Seq[Seq[SortOrder]] = Seq(partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec)
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  private def compileKey(e: Expression): VectorExpr = ExpressionCompiler.compile(e, child.output) match {
    case Right(v) => v
    case Left(reason) => throw new IllegalStateException(s"cannot vectorize window key ${e.sql}: $reason")
  }

  // A constant key never separates two rows (`ORDER BY length('abc')` folds to a literal: every row is a
  // peer), so literals are dropped rather than evaluated as columns.
  @transient private lazy val partitionKeys: Array[VectorExpr] = partitionSpec.map(compileKey).filterNot(_.isInstanceOf[LiteralExpr]).toArray
  @transient private lazy val orderKeys: Array[VectorExpr] = orderSpec.map(o => compileKey(o.child)).filterNot(_.isInstanceOf[LiteralExpr]).toArray

  private def aggregateMode: Boolean = VectorWindowPlanner.wholePartitionAggregates(windowExpression).isDefined

  @transient private lazy val kinds: Array[Int] = windowExpression.map(e => VectorWindowPlanner.rankKind(e).getOrElse(
    throw new IllegalStateException(s"cannot vectorize window function ${e.sql}"))).toArray

  @transient private lazy val aggregates: Seq[AggregateExpression] = VectorWindowPlanner.wholePartitionAggregates(windowExpression).getOrElse(
    throw new IllegalStateException("window expressions are not whole-partition aggregates"))
  @transient private lazy val aggFunctions: Array[VectorAggFunction] = aggregates.map { a =>
    VectorAggregates.compile(a, child.output) match {
      case Right(f) => f
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize window aggregate ${a.sql}: $reason")
    }
  }.toArray
  @transient private lazy val aggResults: Array[VectorExpr] = {
    val attrs = aggregates.map(_.resultAttribute)
    VectorAggregatePlanner.compileFinalResults(Nil, aggregates, attrs, attrs) match {
      case Right(exprs) => exprs.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize window aggregate results: $reason")
    }
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val pk = partitionKeys
    val childAttrs = child.output.map(a => (a.name, a.dataType)).toArray
    val windowAttrs = windowExpression.map(e => (e.name, e.dataType)).toArray
    val m = vectorMetrics
    if (aggregateMode) {
      val aggs = aggFunctions
      val results = aggResults
      val layout = VectorAggregatePlanner.bufferLayout(Nil, aggregates).toArray
      val bufferAttrs = VectorAggregatePlanner.bufferAttributes(Nil, aggregates).map(a => (a.name, a.dataType)).toArray
      child.executeColumnar().mapPartitionsInternal { iter =>
        new VectorWindowAggregateIterator(iter, pk, aggs, layout, bufferAttrs, results, childAttrs, windowAttrs, m)
      }
    } else {
      val ok = orderKeys
      val ks = kinds
      child.executeColumnar().mapPartitionsInternal { iter =>
        new VectorWindowIterator(iter, pk, ok, ks, childAttrs, windowAttrs, m)
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Window: ${windowExpression.map(_.sql).mkString(", ")}
       |Partition: ${partitionSpec.map(_.sql).mkString(", ")}
       |Order: ${orderSpec.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

object VectorWindowPlanner {
  val RowNumberKind = 0
  val RankKind = 1
  val DenseRankKind = 2

  /** Which ranking function a window expression is, or why it is not one this layer computes. */
  def rankKind(e: NamedExpression): Either[String, Int] = e match {
    case Alias(WindowExpression(f, _), _) => f match {
      case _: RowNumber => Right(RowNumberKind)
      case _: Rank => Right(RankKind)
      case _: DenseRank => Right(DenseRankKind)
      case a: AggregateExpression => Left(s"window aggregate ${a.aggregateFunction.prettyName} over a frame not supported")
      case other => Left(s"window function ${other.prettyName} not supported (row_number, rank and dense_rank are)")
    }
    case other => Left(s"window expression ${other.sql} not supported")
  }

  /** The ranking kind of a bare ranking function (WindowGroupLimitExec's `rankLikeFunction`). */
  def rankLikeKind(f: Expression): Either[String, Int] = f match {
    case _: RowNumber => Right(RowNumberKind)
    case _: Rank => Right(RankKind)
    case _: DenseRank => Right(DenseRankKind)
    case other => Left(s"window group limit function ${other.prettyName} not supported")
  }

  /** The aggregate of a whole-partition frame (`UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`), if that is what `e` is. */
  private def wholePartitionAggregate(e: NamedExpression): Option[AggregateExpression] = e match {
    case Alias(WindowExpression(a: AggregateExpression, WindowSpecDefinition(_, _, SpecifiedWindowFrame(_, UnboundedPreceding, UnboundedFollowing))), _)
        if a.mode == Complete && a.filter.isEmpty => Some(a)
    case _ => None
  }

  /** All of the operator's expressions as whole-partition aggregates, or None when any is something else. */
  def wholePartitionAggregates(es: Seq[NamedExpression]): Option[Seq[AggregateExpression]] = {
    val aggs = es.map(wholePartitionAggregate)
    if (es.nonEmpty && aggs.forall(_.isDefined)) Some(aggs.flatten) else None
  }

  /** Why a window aggregate is not computed: the frame, or the function itself. */
  private def aggregateReason(e: NamedExpression, input: Seq[Attribute]): Option[String] = e match {
    case Alias(WindowExpression(a: AggregateExpression, WindowSpecDefinition(_, _, frame)), _) =>
      frame match {
        case SpecifiedWindowFrame(_, UnboundedPreceding, UnboundedFollowing) =>
          // A decimal sum's buffer is Spark's Decimal(p + 10): the 128-bit lane (#28) before this can gather it.
          if (a.dataType.isInstanceOf[DecimalType] || a.aggregateFunction.children.exists(_.dataType.isInstanceOf[DecimalType]))
            Some(s"window aggregate ${a.aggregateFunction.prettyName} over decimals not supported (its buffer is a wide decimal)")
          else VectorAggregates.compile(a, input).left.toOption.map(r => s"window aggregate ${a.aggregateFunction.prettyName}: $r")
        case f => Some(s"window aggregate ${a.aggregateFunction.prettyName} over frame ${f.sql} not supported (whole-partition frames are)")
      }
    case _ => None
  }

  def plan(w: WindowExec): Either[String, VectorWindowExec] = {
    val keys = w.partitionSpec ++ w.orderSpec.map(_.child)
    val keyFailures = keys.flatMap { k =>
      if (k.dataType == DoubleType) Some(s"window key ${k.sql}: double keys not supported (Spark compares them after NaN and zero normalisation)")
      else if (!TypeMapping.isSupported(k.dataType)) Some(s"window key type ${k.dataType.simpleString} not supported")
      else ExpressionCompiler.compile(k, w.child.output).left.toOption.map(r => s"window key ${k.sql}: $r")
    }
    wholePartitionAggregates(w.windowExpression) match {
      case Some(aggs) =>
        val reasons = w.windowExpression.flatMap(aggregateReason(_, w.child.output))
        val attrs = aggs.map(_.resultAttribute)
        val resultReason = if (reasons.nonEmpty) None else VectorAggregatePlanner.compileFinalResults(Nil, aggs, attrs, attrs).left.toOption.map(r => s"window aggregate result: $r")
        (reasons ++ resultReason ++ keyFailures).headOption.toLeft(VectorWindowExec(w.windowExpression, w.partitionSpec, w.orderSpec, w.child))
      case None =>
        val kinds = w.windowExpression.map(rankKind)
        kinds.collectFirst { case Left(reason) => reason } match {
          case Some(reason) =>
            // A ranking function beside an aggregate in one spec, or an aggregate over another frame: say which.
            Left(w.windowExpression.flatMap(aggregateReason(_, w.child.output)).headOption.getOrElse(reason))
          case None if w.orderSpec.isEmpty => Left("ranking window without an ORDER BY")
          case None => keyFailures.headOption.toLeft(VectorWindowExec(w.windowExpression, w.partitionSpec, w.orderSpec, w.child))
        }
    }
  }
}

/** Detects partition (or peer) boundaries: null-safe equality of a row's keys with the previous row's. */
private[vector] final class KeyTracker(keys: Array[VectorExpr]) {
  private val previous = new Array[Any](keys.length)
  private var lanes: Array[VectorBuffers] = _
  var hasPrevious = false

  def startBatch(ctx: io.sparkvector.spark.expr.EvalContext): Unit = lanes = keys.map(_.eval(ctx))

  /** True when row `i` differs from the remembered row (or none is remembered). */
  def changed(i: Int): Boolean = {
    if (!hasPrevious) return true
    var k = 0
    while (k < lanes.length) {
      val current = if (Rows.valid(lanes(k), i)) Rows.box(lanes(k), i) else null
      val before = previous(k)
      if (!(if (before == null) current == null else before.equals(current))) return true
      k += 1
    }
    false
  }

  def remember(i: Int): Unit = {
    var k = 0
    while (k < lanes.length) {
      previous(k) = if (Rows.valid(lanes(k), i)) Rows.box(lanes(k), i) else null
      k += 1
    }
    hasPrevious = true
  }
}

/** The ranking walk over sorted batches; the previous row's keys and the counters carry across batches. */
private[vector] class VectorWindowIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    orderKeys: Array[VectorExpr],
    kinds: Array[Int],
    childAttrs: Array[(String, DataType)],
    windowAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorWindowExec") {

  private val partition = new KeyTracker(partitionKeys)
  private val order = new KeyTracker(orderKeys)
  private var rowNumber = 0L
  private var rank = 0L
  private var denseRank = 0L

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      val outRows = if (live == null) n else ctx.selectedCount
      partition.startBatch(ctx)
      order.startBatch(ctx)
      val results = new Array[Array[Int]](kinds.length)
      var f = 0
      while (f < kinds.length) { results(f) = new Array[Int](outRows); f += 1 }
      var out = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = partition.changed(i)
          if (newPartition) {
            rowNumber = 0L
            rank = 0L
            denseRank = 0L
          }
          rowNumber += 1
          val peer = !newPartition && !order.changed(i)
          if (!peer) {
            rank = rowNumber
            denseRank += 1
          }
          partition.remember(i)
          order.remember(i)
          f = 0
          while (f < kinds.length) {
            results(f)(out) = kinds(f) match {
              case VectorWindowPlanner.RowNumberKind => rowNumber.toInt
              case VectorWindowPlanner.RankKind => rank.toInt
              case _ => denseRank.toInt
            }
            f += 1
          }
          out += 1
        }
        i += 1
      }
      val columns = new Array[ColumnVector](childAttrs.length + kinds.length)
      var c = 0
      while (c < childAttrs.length) {
        val (name, dt) = childAttrs(c)
        // Forwarded columns are never copied: the child keeps them alive until its next batch.
        columns(c) = if (live == null) BorrowedColumnVector.of(batch.column(c)) else ArrowOutput.compact(name, dt, ctx.input(c), live, outRows, allocator)
        c += 1
      }
      f = 0
      while (f < kinds.length) {
        val (name, dt) = windowAttrs(f)
        val buffers = ArrowOutput.allocateFixed(name, dt, outRows, allocator)
        val data = buffers.data()
        var r = 0
        while (r < outRows) { data.setAtIndex(VectorBuffers.LE_INT, r, results(f)(r)); r += 1 }
        columns(childAttrs.length + f) = ArrowOutput.finish(buffers, outRows, true)
        f += 1
      }
      metrics.numOutputBatches += 1
      metrics.numOutputRows += outRows
      new ColumnarBatch(columns, outRows)
    }
  }
}

/**
 * Whole-partition aggregates over sorted batches. Partitions are groups numbered in arrival order;
 * the grouped aggregate states accumulate every batch with the partition ordinal as the group id.
 * A batch is held (its live rows copied) until the partition its last row belongs to has ended --
 * the next batch starts a new one, or the input ends -- and is then emitted with the values of its
 * partitions gathered per row.
 */
private[vector] class VectorWindowAggregateIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    aggs: Array[VectorAggFunction],
    layout: Array[OutputSlot],
    bufferAttrs: Array[(String, DataType)],
    results: Array[VectorExpr],
    childAttrs: Array[(String, DataType)],
    windowAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch] with AutoCloseable {

  /** A held batch: owned copies of the live rows and each row's partition ordinal. */
  private final class Held(val columns: Array[ColumnVector], val numRows: Int, val groups: Array[Int])

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorWindowExec")
  private val states: Array[GroupedAggState] = aggs.map(_.newGroupedState())
  private val partition = new KeyTracker(partitionKeys)
  private var numGroups = 0
  private val held = mutable.Queue.empty[Held]
  private val ready = mutable.Queue.empty[ColumnarBatch]
  private var idScratch = new Array[Int](0)
  private var inputDone = false
  private var current: ColumnarBatch = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  private def consume(batch: ColumnarBatch): Unit = metrics.timed {
    metrics.numInputBatches += 1
    EvalContexts.withBatch(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      val outRows = if (live == null) n else ctx.selectedCount
      if (idScratch.length < n) idScratch = new Array[Int](n)
      partition.startBatch(ctx)
      val groups = new Array[Int](outRows)
      var out = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          if (partition.changed(i)) { numGroups += 1; partition.remember(i) }
          idScratch(i) = numGroups - 1
          groups(out) = numGroups - 1
          out += 1
        } else idScratch(i) = -1
        i += 1
      }
      val assignment = GroupAssignment.of(idScratch, n, numGroups, ctx.arena, live)
      var s = 0
      while (s < states.length) { states(s).update(ctx, assignment); s += 1 }
      val columns = new Array[ColumnVector](childAttrs.length)
      var c = 0
      while (c < childAttrs.length) {
        val (name, dt) = childAttrs(c)
        // The child reuses or releases its batch once we pull the next one: the held rows are copies.
        columns(c) = if (live == null) ArrowOutput.copy(name, dt, ctx.input(c), allocator) else ArrowOutput.compact(name, dt, ctx.input(c), live, outRows, allocator)
        c += 1
      }
      held.enqueue(new Held(columns, outRows, groups))
    }
  }

  /** Emits every held batch whose partitions have all ended (all of them once the input is done). */
  private def release(): Unit = metrics.timed {
    while (held.nonEmpty && (inputDone || held.head.groups(held.head.numRows - 1) < numGroups - 1)) {
      val h = held.dequeue()
      val from = h.groups(0)
      val to = h.groups(h.numRows - 1) + 1
      val idx = new Array[Int](h.numRows)
      var r = 0
      while (r < h.numRows) { idx(r) = h.groups(r) - from; r += 1 }
      // The partitions' buffers as a batch of `to - from` rows, the results evaluated over it and gathered per row.
      val bufferColumns = new Array[ColumnVector](layout.length)
      var c = 0
      while (c < layout.length) {
        val (name, dt) = bufferAttrs(c)
        bufferColumns(c) = layout(c) match {
          case BufferSlot(aggIdx, slot) => AggBufferColumns.column(name, dt, states(aggIdx), slot, from, to, allocator)
          case KeySlot(_) => throw new IllegalStateException("key slot in a window aggregate")
        }
        c += 1
      }
      val buffers = new ColumnarBatch(bufferColumns, to - from)
      val columns = new Array[ColumnVector](childAttrs.length + results.length)
      try {
        EvalContexts.withBatch(buffers) { ctx =>
          var f = 0
          while (f < results.length) {
            val (name, dt) = windowAttrs(f)
            columns(childAttrs.length + f) = results(f) match {
              case lit: LiteralExpr => ArrowOutput.constant(name, dt, lit.value, h.numRows, allocator)
              case e => ArrowOutput.gather(name, dt, e.eval(ctx), idx, 0, h.numRows, allocator)
            }
            f += 1
          }
        }
      } finally buffers.close()
      System.arraycopy(h.columns, 0, columns, 0, childAttrs.length)
      metrics.numOutputBatches += 1
      metrics.numOutputRows += h.numRows
      ready.enqueue(new ColumnarBatch(columns, h.numRows))
    }
  }

  override def hasNext: Boolean = {
    while (ready.isEmpty && !inputDone) {
      if (input.hasNext) {
        val batch = input.next()
        if (batch.numRows() > 0) consume(batch)
      } else inputDone = true
      release()
    }
    ready.nonEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more batches")
    releaseCurrent()
    current = ready.dequeue()
    current
  }

  private def releaseCurrent(): Unit = if (current != null) { current.close(); current = null }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseCurrent()
      ready.foreach(_.close())
      ready.clear()
      held.foreach(_.columns.foreach(_.close()))
      held.clear()
      allocator.close()
    }
  }
}

/**
 * Columnar replacement for WindowGroupLimitExec (#58, layer 4): Spark's per-partition top-k below a
 * ranking window, inserted for a filter `rank <= k` (k up to `spark.sql.optimizer.windowGroupLimitThreshold`)
 * -- in Partial mode below the exchange, over whatever produced the rows, and in Final mode above the
 * sort. Both modes require the same ordering as the window (partition keys, then order keys) and the
 * Final mode the window's distribution, so the walk is the ranking walk of [[VectorWindowIterator]]
 * with a filter: a row is kept while the ranking function's value at that row is at most `limit`.
 * Spark's own operator also lets the first row of the peer group past the limit through (it checks the
 * previous row's rank); both are pre-filters under the window that computes the real ranks, so the
 * tighter set is exact for the query and smaller for the shuffle.
 *
 * Output rows are compacted (a top-k keeps a few rows per partition); a batch with every row kept is
 * forwarded, one with none is dropped.
 */
case class VectorWindowGroupLimitExec(
    partitionSpec: Seq[Expression],
    orderSpec: Seq[SortOrder],
    rankLikeFunction: Expression,
    limit: Int,
    mode: WindowGroupLimitMode,
    child: SparkPlan)
    extends VectorExec {

  override def output: Seq[Attribute] = child.output

  override def requiredChildDistribution: Seq[Distribution] = mode match {
    case Partial => super.requiredChildDistribution
    case Final => if (partitionSpec.isEmpty) AllTuples :: Nil else ClusteredDistribution(partitionSpec) :: Nil
  }
  override def requiredChildOrdering: Seq[Seq[SortOrder]] = Seq(partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec)
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  private def compileKey(e: Expression): VectorExpr = ExpressionCompiler.compile(e, child.output) match {
    case Right(v) => v
    case Left(reason) => throw new IllegalStateException(s"cannot vectorize window group limit key ${e.sql}: $reason")
  }
  @transient private lazy val partitionKeys: Array[VectorExpr] = partitionSpec.map(compileKey).filterNot(_.isInstanceOf[LiteralExpr]).toArray
  @transient private lazy val orderKeys: Array[VectorExpr] = orderSpec.map(o => compileKey(o.child)).filterNot(_.isInstanceOf[LiteralExpr]).toArray
  @transient private lazy val kind: Int = VectorWindowPlanner.rankLikeKind(rankLikeFunction).getOrElse(
    throw new IllegalStateException(s"cannot vectorize window group limit function ${rankLikeFunction.sql}"))

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val pk = partitionKeys
    val ok = orderKeys
    val k = kind
    val lim = limit
    val childAttrs = child.output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorWindowGroupLimitIterator(iter, pk, ok, k, lim, childAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Function: ${rankLikeFunction.sql} <= $limit ($mode)
       |Partition: ${partitionSpec.map(_.sql).mkString(", ")}
       |Order: ${orderSpec.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

object VectorWindowGroupLimitPlanner {
  def plan(g: WindowGroupLimitExec): Either[String, VectorWindowGroupLimitExec] = {
    VectorWindowPlanner.rankLikeKind(g.rankLikeFunction) match {
      case Left(reason) => Left(reason)
      case Right(_) =>
        val keys = g.partitionSpec ++ g.orderSpec.map(_.child)
        val keyFailures = keys.flatMap { k =>
          if (k.dataType == DoubleType) Some(s"window key ${k.sql}: double keys not supported (Spark compares them after NaN and zero normalisation)")
          else if (!TypeMapping.isSupported(k.dataType)) Some(s"window key type ${k.dataType.simpleString} not supported")
          else ExpressionCompiler.compile(k, g.child.output).left.toOption.map(r => s"window key ${k.sql}: $r")
        }
        keyFailures.headOption.toLeft(VectorWindowGroupLimitExec(g.partitionSpec, g.orderSpec, g.rankLikeFunction, g.limit, g.mode, g.child))
    }
  }
}

/** The ranking walk with a filter: rows whose ranking value is at most `limit` survive. */
private[vector] class VectorWindowGroupLimitIterator(
    input: Iterator[ColumnarBatch],
    partitionKeys: Array[VectorExpr],
    orderKeys: Array[VectorExpr],
    kind: Int,
    limit: Int,
    childAttrs: Array[(String, DataType)],
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorWindowGroupLimitExec") {

  private val partition = new KeyTracker(partitionKeys)
  private val order = new KeyTracker(orderKeys)
  private var rowNumber = 0L
  private var rank = 0L
  private var denseRank = 0L

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      partition.startBatch(ctx)
      order.startBatch(ctx)
      val kept = Bitmap.allocate(ctx.arena, n)
      var keptCount = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = partition.changed(i)
          if (newPartition) {
            rowNumber = 0L
            rank = 0L
            denseRank = 0L
          }
          rowNumber += 1
          val peer = !newPartition && !order.changed(i)
          if (!peer) {
            rank = rowNumber
            denseRank += 1
          }
          partition.remember(i)
          order.remember(i)
          val value = kind match {
            case VectorWindowPlanner.RowNumberKind => rowNumber
            case VectorWindowPlanner.RankKind => rank
            case _ => denseRank
          }
          if (value <= limit) { Bitmap.set(kept, i); keptCount += 1 }
        }
        i += 1
      }
      if (keptCount == 0) {
        null
      } else if (live == null && keptCount == n) {
        metrics.numOutputBatches += 1
        metrics.numOutputRows += n
        batch
      } else {
        val columns = new Array[ColumnVector](childAttrs.length)
        var c = 0
        while (c < childAttrs.length) {
          val (name, dt) = childAttrs(c)
          columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), kept, keptCount, allocator)
          c += 1
        }
        metrics.numOutputBatches += 1
        metrics.numOutputRows += keptCount
        new ColumnarBatch(columns, keptCount)
      }
    }
  }
}
