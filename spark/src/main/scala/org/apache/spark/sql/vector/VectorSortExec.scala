package org.apache.spark.sql.vector

import java.lang.foreign.Arena

import io.sparkvector.kernels.{ColumnBuilder, SortKernels, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, VectorAllocators}
import io.sparkvector.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, NullsFirst, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{Distribution, OrderedDistribution, Partitioning, UnspecifiedDistribution}
import org.apache.spark.sql.execution.{SortExec, SparkPlan}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for SortExec.
 *
 * A blocking operator: every batch of the partition is appended to one [[ColumnBuilder]] per
 * output attribute and per computed key (Spark's columnar contract lets the producer reuse a batch
 * as soon as the next one is requested), a permutation is computed by [[SortKernels.sortIndices]]
 * and the output is gathered through it in batches of 4096 rows into Arrow vectors. Sort keys that are plain column references reuse the output
 * column rather than being copied twice.
 *
 * Same distribution contract as SortExec: a global sort requires the range partitioning the
 * exchange below already provides, a local sort accepts anything. The operator is only planned
 * over a columnar child (Comet's columnar shuffle, or a spark-vector operator for a local sort);
 * over Spark's row shuffle the rule leaves SortExec in place, since converting rows to columns just
 * to sort them buys nothing.
 *
 * Everything is held in memory: there is no spill. Partitions large enough to need one should keep
 * Spark's sort (`spark.vector.exec.sort.enabled=false`).
 */
case class VectorSortExec(sortOrder: Seq[SortOrder], global: Boolean, child: SparkPlan) extends VectorExec {

  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = sortOrder
  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def requiredChildDistribution: Seq[Distribution] =
    if (global) OrderedDistribution(sortOrder) :: Nil else UnspecifiedDistribution :: Nil

  @transient private lazy val compiledKeys: Array[VectorExpr] = sortOrder.map { o =>
    VectorSortPlanner.compileKey(o, child.output) match {
      case Right(k) => k
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize sort key ${o.sql}: $reason")
    }
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val keys = compiledKeys
    val ascending = sortOrder.map(_.direction == Ascending).toArray
    val nullsFirst = sortOrder.map(_.nullOrdering == NullsFirst).toArray
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorSortIterator(iter, keys, ascending, nullsFirst, outputAttrs, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Sort: ${sortOrder.map(_.sql).mkString(", ")}
       |Global: $global
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

/** Drains the partition, sorts it, then emits the rows in order -- at most `limit` of them. */
private[vector] class VectorSortIterator(
    input: Iterator[ColumnarBatch],
    keyExprs: Array[VectorExpr],
    ascending: Array[Boolean],
    nullsFirst: Array[Boolean],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics,
    limit: Int = Int.MaxValue)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val OutputBatchSize = 4096

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorSortExec")
  /** Owns every copied chunk and the joined columns; shared because Spark may hand the iterator across threads. */
  private val arena: Arena = Arena.ofShared()

  private val numColumns = outputAttrs.length
  /** Key c is output column `keyColumn(c)`, or -1 when it is a computed expression with its own chunks. */
  private val keyColumn: Array[Int] = keyExprs.map {
    case ColumnRef(ordinal, _) => ordinal
    case _ => -1
  }
  private val computedKeys: Array[Int] = keyColumn.indices.filter(keyColumn(_) < 0).toArray

  private var sorted = false
  private var columns: Array[VectorBuffers] = _
  private var permutation: Array[Int] = _
  private var total = 0
  private var emitted = 0
  private var current: ColumnarBatch = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  private def drainAndSort(): Unit = {
    if (sorted) return
    sorted = true
    var columnBuilders: Array[ColumnBuilder] = null
    var keyBuilders: Array[ColumnBuilder] = null
    while (input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) {
        metrics.timed {
          metrics.numInputBatches += 1
          EvalContexts.withBatch(batch) { ctx =>
            val count = ctx.selectedCount
            if (count > 0) {
              if (columnBuilders == null) {
                columnBuilders = Array.tabulate(numColumns)(c => new ColumnBuilder(arena, ctx.input(c).`type`(), count))
                keyBuilders = Array.tabulate(computedKeys.length)(k => new ColumnBuilder(arena, keyExprs(computedKeys(k)).vecType, count))
              }
              var c = 0
              while (c < numColumns) {
                columnBuilders(c).append(ctx.input(c), ctx.selection, count)
                c += 1
              }
              var k = 0
              while (k < computedKeys.length) {
                keyBuilders(k).append(keyExprs(computedKeys(k)).eval(ctx), ctx.selection, count)
                k += 1
              }
              total += count
            }
          }
        }
      }
    }
    metrics.timed {
      if (total > 0) {
        columns = columnBuilders.map(_.view())
        val computed = keyBuilders.map(_.view())
        val keys = new Array[VectorBuffers](keyExprs.length)
        var computedIdx = 0
        var k = 0
        while (k < keys.length) {
          if (keyColumn(k) >= 0) keys(k) = columns(keyColumn(k))
          else { keys(k) = computed(computedIdx); computedIdx += 1 }
          k += 1
        }
        permutation = SortKernels.sortIndices(keys, ascending, nullsFirst, total)
        // A top-N emits only the head of the permutation (the whole partition was still sorted).
        total = math.min(total, limit)
      }
    }
  }

  override def hasNext: Boolean = {
    drainAndSort()
    emitted < total
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more sorted rows")
    releaseCurrent()
    val from = emitted
    val to = math.min(total, from + OutputBatchSize)
    val count = to - from
    val out = new Array[ColumnVector](numColumns)
    metrics.timed {
      var c = 0
      while (c < numColumns) {
        val (name, dt) = outputAttrs(c)
        out(c) = ArrowOutput.gather(name, dt, columns(c), permutation, from, to, allocator)
        c += 1
      }
    }
    emitted = to
    metrics.numOutputBatches += 1
    metrics.numOutputRows += count
    current = new ColumnarBatch(out, count)
    current
  }

  private def releaseCurrent(): Unit = {
    if (current != null) { current.close(); current = null }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseCurrent()
      columns = null
      permutation = null
      arena.close()
      allocator.close()
    }
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorSortPlanner {

  /** Any compiled non-literal expression of a supported type can be a sort key. */
  def compileKey(order: SortOrder, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(order.child, input).flatMap {
      case _: LiteralExpr => Left("literal sort key")
      case k if !TypeMapping.isSupported(order.child.dataType) => Left(s"sort key type ${order.child.dataType.simpleString} not supported")
      case k => Right(k)
    }

  /** Attempts to convert a Spark SortExec; Left explains the fallback. */
  def plan(s: SortExec): Either[String, VectorSortExec] = {
    val failures = s.sortOrder.flatMap(o => compileKey(o, s.child.output).left.toOption.map(r => s"${o.sql}: $r"))
    if (s.sortOrder.isEmpty) Left("sort without keys")
    else if (failures.nonEmpty) Left(failures.mkString("; "))
    else Right(VectorSortExec(s.sortOrder, s.global, s.child))
  }
}
