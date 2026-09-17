package org.apache.spark.sql.vector

import io.sparkvector.kernels.{Bitmap, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.agg.Rows
import io.sparkvector.spark.arrow.{ArrowOutput, BorrowedColumnVector}
import io.sparkvector.spark.expr.{ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, DenseRank, Expression, NamedExpression, Rank, RowNumber, SortOrder, WindowExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.types.{DataType, DoubleType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for WindowExec, first layer (#58): the ranking functions `row_number`, `rank`
 * and `dense_rank`. The child arrives sorted by the partition keys and then the order keys -- Spark
 * planned the sort and the shuffle for its own operator, and this one requires exactly what
 * `WindowExec` does -- so the whole computation is a walk over the rows in order: a row starts a new
 * partition when a partition key differs from the previous row's, and a new peer group when an order
 * key does (null-safe equality on both, as Spark's `RankLike` compares with `<=>`); `row_number`
 * counts rows since the partition began, `rank` is the row number at which the current peer group
 * began, `dense_rank` counts peer groups. The walk carries the previous row's keys and the counters
 * across batches, so a partition longer than a batch is one partition. Every input column is
 * forwarded (borrowed, never copied), the results are new INT32 columns.
 *
 * The child may be a row operator: Spark plans `Window` above `Sort` above an exchange, and without
 * Comet's shuffle that sort stays Spark's, so `RowToColumnarExec` is inserted below us; from here up
 * the chain (a filter on the rank, a projection, a limit) is columnar again.
 *
 * Aggregate windows (`sum` over a frame), offset functions (`lag`, `lead`, `first_value`), the other
 * ranking functions (`percent_rank`, `cume_dist`, `ntile`) and `WindowGroupLimitExec` are later
 * layers; each is refused with a reason naming the function and frame.
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
  @transient private lazy val kinds: Array[Int] = windowExpression.map(e => VectorWindowPlanner.rankKind(e).getOrElse(
    throw new IllegalStateException(s"cannot vectorize window function ${e.sql}"))).toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val pk = partitionKeys
    val ok = orderKeys
    val ks = kinds
    val childAttrs = child.output.map(a => (a.name, a.dataType)).toArray
    val windowAttrs = windowExpression.map(e => (e.name, e.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorWindowIterator(iter, pk, ok, ks, childAttrs, windowAttrs, m)
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

  def plan(w: WindowExec): Either[String, VectorWindowExec] = {
    val kinds = w.windowExpression.map(rankKind)
    kinds.collectFirst { case Left(reason) => reason } match {
      case Some(reason) => Left(reason)
      case None if w.orderSpec.isEmpty => Left("ranking window without an ORDER BY")
      case None =>
        val keys = w.partitionSpec ++ w.orderSpec.map(_.child)
        val keyFailures = keys.flatMap { k =>
          if (k.dataType == DoubleType) Some(s"window key ${k.sql}: double keys not supported (Spark compares them after NaN and zero normalisation)")
          else if (!TypeMapping.isSupported(k.dataType)) Some(s"window key type ${k.dataType.simpleString} not supported")
          else ExpressionCompiler.compile(k, w.child.output).left.toOption.map(r => s"window key ${k.sql}: $r")
        }
        if (keyFailures.nonEmpty) Left(keyFailures.mkString("; "))
        else Right(VectorWindowExec(w.windowExpression, w.partitionSpec, w.orderSpec, w.child))
    }
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

  private var hasPrevious = false
  private val previousPartition = new Array[Any](partitionKeys.length)
  private val previousOrder = new Array[Any](orderKeys.length)
  private var rowNumber = 0L
  private var rank = 0L
  private var denseRank = 0L

  /** Null-safe equality of the previous row's keys with row `i` of the current lanes. */
  private def sameKeys(previous: Array[Any], lanes: Array[VectorBuffers], i: Int): Boolean = {
    var k = 0
    while (k < lanes.length) {
      val current = if (Rows.valid(lanes(k), i)) Rows.box(lanes(k), i) else null
      val before = previous(k)
      if (!(if (before == null) current == null else before.equals(current))) return false
      k += 1
    }
    true
  }

  private def remember(previous: Array[Any], lanes: Array[VectorBuffers], i: Int): Unit = {
    var k = 0
    while (k < lanes.length) {
      previous(k) = if (Rows.valid(lanes(k), i)) Rows.box(lanes(k), i) else null
      k += 1
    }
  }

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val live = ctx.selection
      val outRows = if (live == null) n else ctx.selectedCount
      val partitionLanes = partitionKeys.map(_.eval(ctx))
      val orderLanes = orderKeys.map(_.eval(ctx))
      val results = new Array[Array[Int]](kinds.length)
      var f = 0
      while (f < kinds.length) { results(f) = new Array[Int](outRows); f += 1 }
      var out = 0
      var i = 0
      while (i < n) {
        if (live == null || Bitmap.isSet(live, i)) {
          val newPartition = !hasPrevious || !sameKeys(previousPartition, partitionLanes, i)
          if (newPartition) {
            rowNumber = 0L
            rank = 0L
            denseRank = 0L
          }
          rowNumber += 1
          val peer = !newPartition && sameKeys(previousOrder, orderLanes, i)
          if (!peer) {
            rank = rowNumber
            denseRank += 1
          }
          remember(previousPartition, partitionLanes, i)
          remember(previousOrder, orderLanes, i)
          hasPrevious = true
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
