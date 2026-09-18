package org.apache.spark.sql.vector

import java.lang.foreign.Arena

import scala.collection.mutable.ArrayBuffer

import io.sparkvector.kernels.{ColumnBuilder, RunMerge, SortKernels, VecType, VectorBuffers}
import io.sparkvector.spark.VectorConf
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
 * as soon as the next one is requested). Every `spark.vector.sort.runRows` rows the builders are
 * sealed into a run -- its columns and the permutation [[SortKernels.sortIndices]] computed over
 * them -- so the sort's scratch is bounded by the run, not the partition. One run emits its rows
 * through the permutation; several are k-way merged by [[RunMerge]] in the same total order, ties by
 * run then position (stable), and gathered run by run. Output batches are 4096 rows of Arrow
 * vectors; a dictionary-encoded string column is decoded once when its run is sealed (#285). Sort
 * keys that are plain column references reuse the output column rather than being copied twice.
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
    val runRows = VectorConf.sortRunRows(conf)
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorSortIterator(iter, keys, ascending, nullsFirst, outputAttrs, m, runRows = runRows)
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

/**
 * Drains the partition into sorted runs of `runRows` rows, then emits the rows in order -- at most
 * `limit` of them -- merging the runs when there are several.
 */
private[vector] class VectorSortIterator(
    input: Iterator[ColumnarBatch],
    keyExprs: Array[VectorExpr],
    ascending: Array[Boolean],
    nullsFirst: Array[Boolean],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics,
    limit: Int = Int.MaxValue,
    runRows: Int = 1 << 20)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  /** A sealed run: its output columns, its key columns and the permutation that orders it. */
  private final class Run(val columns: Array[VectorBuffers], val keys: Array[VectorBuffers], val rows: Int) {
    val permutation: Array[Int] = SortKernels.sortIndices(keys, ascending, nullsFirst, rows)
  }

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
  private val runs = ArrayBuffer.empty[Run]
  /** One run: its columns and permutation. Several: the merge, and every output column's run columns. */
  private var columns: Array[VectorBuffers] = _
  private var permutation: Array[Int] = _
  private var merge: RunMerge = _
  private var runColumns: Array[Array[VectorBuffers]] = _
  private var runOf: Array[Int] = _
  private var rowOf: Array[Int] = _
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
    var runTotal = 0

    /** Seals the builders into a run: plain string columns (a dictionary decoded once), the keys, the permutation. */
    def seal(): Unit = {
      if (runTotal > 0) {
        val cols = columnBuilders.map(_.view()).map { v =>
          if (v.`type`() == VecType.UTF8 && v.isDictionaryEncoded) ArrowOutput.decodeDictionary(v, arena) else v
        }
        val computed = keyBuilders.map(_.view())
        val keys = new Array[VectorBuffers](keyExprs.length)
        var computedIdx = 0
        var k = 0
        while (k < keys.length) {
          if (keyColumn(k) >= 0) keys(k) = cols(keyColumn(k))
          else { keys(k) = computed(computedIdx); computedIdx += 1 }
          k += 1
        }
        runs += new Run(cols, keys, runTotal)
        total += runTotal
      }
      columnBuilders = null
      keyBuilders = null
      runTotal = 0
    }

    while (input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) {
        metrics.timed {
          metrics.numInputBatches += 1
          EvalContexts.withBatch(batch) { ctx =>
            val count = ctx.selectedCount
            if (count > 0) {
              if (columnBuilders == null) {
                val expected = math.max(math.min(runRows, count), 1)
                columnBuilders = Array.tabulate(numColumns)(c => new ColumnBuilder(arena, ctx.input(c).`type`(), expected))
                keyBuilders = Array.tabulate(computedKeys.length)(k => new ColumnBuilder(arena, keyExprs(computedKeys(k)).vecType, expected))
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
              runTotal += count
              if (runTotal >= runRows) seal()
            }
          }
        }
      }
    }
    metrics.timed {
      seal()
      if (runs.length == 1) {
        columns = runs.head.columns
        permutation = runs.head.permutation
      } else if (runs.length > 1) {
        merge = new RunMerge(runs.map(_.keys).toArray, runs.map(_.permutation).toArray, runs.map(_.rows).toArray, ascending, nullsFirst)
        runColumns = Array.tabulate(numColumns)(c => runs.map(_.columns(c)).toArray)
        runOf = new Array[Int](OutputBatchSize)
        rowOf = new Array[Int](OutputBatchSize)
      }
      // A top-N emits only the head of the order (every run was still sorted whole).
      total = math.min(total, limit)
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
      if (merge == null) {
        var c = 0
        while (c < numColumns) {
          val (name, dt) = outputAttrs(c)
          out(c) = ArrowOutput.gather(name, dt, columns(c), permutation, from, to, allocator)
          c += 1
        }
      } else {
        val n = merge.next(runOf, rowOf, count)
        assert(n == count, s"merge emitted $n rows, expected $count")
        var c = 0
        while (c < numColumns) {
          val (name, dt) = outputAttrs(c)
          out(c) = ArrowOutput.gatherRuns(name, dt, runColumns(c), runOf, rowOf, count, allocator)
          c += 1
        }
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
      merge = null
      runColumns = null
      runs.clear()
      arena.close()
      allocator.close()
    }
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorSortPlanner {

  /**
   * Any compiled non-literal expression of a supported type can be a sort key, and so can a bare
   * wide decimal column (a DECIMAL128 lane the sort kernel orders limb by limb, #257).
   */
  def compileKey(order: SortOrder, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compileLaneColumn(order.child, input).flatMap {
      case _: LiteralExpr => Left("literal sort key")
      case k if !TypeMapping.hasLane(order.child.dataType) => Left(s"sort key type ${order.child.dataType.simpleString} not supported")
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
