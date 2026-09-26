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

import io.vecruntime.kernels.{Bitmap, GroupAssignment, GroupKeyTable, VecType, VectorBuffers}
import io.vecruntime.spark.adapter.TypeMapping
import io.vecruntime.spark.agg.{AggState, GroupedAggState, VectorAggFunction, VectorAggregates}
import io.vecruntime.spark.VectorConf
import io.vecruntime.spark.arrow.{ArrowOutput, ArrowVectorBuffers, VectorAllocators, VectorDictionaryColumnVector}
import java.lang.foreign.MemorySegment
import org.apache.arrow.vector.{IntVector, VarCharVector}
import io.vecruntime.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{
  Alias,
  Attribute,
  AttributeReference,
  AttributeSet,
  Cast,
  CheckOverflowInSum,
  DecimalDivideWithOverflowCheck,
  EqualTo,
  Expression,
  If,
  Literal,
  NamedExpression
}
import org.apache.spark.sql.catalyst.expressions.aggregate.{
  AggregateExpression,
  AggregateMode,
  Average,
  Complete,
  DeclarativeAggregate,
  Final,
  Partial,
  PartialMerge,
  Sum
}
import org.apache.spark.sql.catalyst.plans.physical.{
  AllTuples,
  ClusteredDistribution,
  Distribution,
  Partitioning,
  UnspecifiedDistribution
}
import org.apache.spark.sql.execution.{PartitioningPreservingUnaryExecNode, SparkPlan}
import org.apache.spark.sql.execution.aggregate.{BaseAggregateExec, HashAggregateExec}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.types.{DataType, DecimalType, IntegerType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/** Where an output column of the partial aggregate comes from. */
sealed trait OutputSlot extends Serializable
final case class KeySlot(key: Int) extends OutputSlot
final case class BufferSlot(agg: Int, slot: Int) extends OutputSlot

/**
 * Columnar replacement for a HashAggregateExec in Partial or Final mode.
 *
 * Partial mode emits the same aggregation buffer schema as Spark (`resultExpressions` are the
 * grouping attributes followed by the functions' `inputAggBufferAttributes`), so any exchange and
 * Final aggregate consume it unchanged. Final mode merges those buffers per group (sum of sums,
 * sum of counts, min of mins...) and then evaluates the `resultExpressions`, with each aggregate's
 * result attribute replaced by the function's `evaluateExpression` over the merged buffer (for
 * `avg`, `sum / count`), through the same expression kernels as a projection.
 *
 * Without grouping keys one row is emitted per partition, even for empty input, exactly like
 * Spark. With grouping keys, rows are assigned dense group ids through a [[GroupKeyTable]] and
 * reduced per group (masked SIMD reductions while there are very few groups, scalar scatter
 * beyond); one row per group is emitted at the end.
 */
case class VectorHashAggregateExec(
    requiredChildDistributionExpressions: Option[Seq[Expression]],
    groupingExpressions: Seq[NamedExpression],
    aggregateExpressions: Seq[AggregateExpression],
    aggregateAttributes: Seq[Attribute],
    resultExpressions: Seq[NamedExpression],
    child: SparkPlan,
    strictFloatingPoint: Boolean = true
) extends VectorExec with PartitioningPreservingUnaryExecNode {

  override def output: Seq[Attribute] = resultExpressions.map(_.toAttribute)

  /**
   * The child's partitioning expressed over this operator's output, through the result aliases, as
   * Spark's HashAggregateExec reports it: `GROUP BY d_year` with `d_year AS year` in the result is
   * partitioned by `year`, not by the input attribute `d_year` this operator no longer outputs. A
   * union above compares its children's partitionings over their outputs -- an expression naming an
   * attribute the child does not output makes the union's partitioning unknown at execution, and its
   * plain concatenation then emits every key once per child (TPC-DS q66: 10 rows for Spark's 5, #162).
   */
  override protected def outputExpressions: Seq[NamedExpression] = resultExpressions

  /** Same contract as HashAggregateExec, so adaptive execution treats the exchange below alike. */
  override def requiredChildDistribution: List[Distribution] = requiredChildDistributionExpressions match {
    case Some(exprs) if exprs.isEmpty => AllTuples :: Nil
    case Some(exprs) => ClusteredDistribution(exprs) :: Nil
    case None => UnspecifiedDistribution :: Nil
  }

  def isFinal: Boolean = aggregateExpressions.nonEmpty && aggregateExpressions.forall(_.mode == Final)

  /** The modes present, in plan order. */
  def modes: Seq[AggregateMode] = aggregateExpressions.map(_.mode).distinct

  /**
   * `Final` and `Complete` evaluate the result expressions over the aggregated state; `Partial` and
   * `PartialMerge` emit the aggregation buffers for a later stage. The planner refuses any other mix.
   */
  def emitsResults: Boolean = VectorAggregatePlanner.emitsResults(modes)

  /** Buffer and result attributes originate here (mirrors HashAggregateExec.producedAttributes). */
  override def producedAttributes: AttributeSet =
    AttributeSet(aggregateAttributes) ++
      AttributeSet(resultExpressions.diff(groupingExpressions).map(_.toAttribute)) ++
      AttributeSet(aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)) ++
      AttributeSet(aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes))

  /** Buffer modes: the emitted columns. Result modes: the aggregated-state batch the result projection reads. */
  @transient private lazy val layout: Array[OutputSlot] =
    if (emitsResults) VectorAggregatePlanner.bufferLayout(groupingExpressions, aggregateExpressions).toArray
    else VectorAggregatePlanner.outputLayout(groupingExpressions, aggregateExpressions, resultExpressions) match {
      case Right(l) => l.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate: $reason")
    }

  @transient private lazy val bufferAttributes: Seq[Attribute] =
    VectorAggregatePlanner.bufferAttributes(groupingExpressions, aggregateExpressions)

  @transient private lazy val resultProjection: Array[VectorExpr] =
    VectorAggregatePlanner.compileFinalResults(
      groupingExpressions,
      aggregateExpressions,
      aggregateAttributes,
      resultExpressions
    ) match {
      case Right(exprs) => exprs.toArray
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate results: $reason")
    }

  @transient private lazy val compiledKeys: Array[VectorExpr] = groupingExpressions.map { e =>
    VectorAggregatePlanner.compileKey(e, child.output) match {
      case Right(k) => k
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize grouping key ${e.sql}: $reason")
    }
  }.toArray

  // A merging stage's input is its grouping columns then the buffers (Spark's initialInputBufferOffset).
  @transient private lazy val compiled: Array[VectorAggFunction] =
    aggregateExpressions.zip(VectorAggregates.bufferOffsets(groupingExpressions.length, aggregateExpressions)).map {
      case (agg, offset) =>
        VectorAggregates.compile(agg, child.output, offset, strictFloatingPoint) match {
          case Right(f) => f
          case Left(reason) => throw new IllegalStateException(s"cannot vectorize aggregate ${agg.sql}: $reason")
        }
    }.toArray

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "time" -> SQLMetrics.createNanoTimingMetric(sparkContext, "time in spark-vector kernels"),
    "spills" -> SQLMetrics.createMetric(sparkContext, "number of spills (#363)"),
    "spilledGroups" -> SQLMetrics.createMetric(sparkContext, "groups spilled or emitted early")
  )

  /**
   * Past the budget (`spark.vector.agg.spillThreshold`, 0 = never): buffer-emitting modes emit the
   * table and start over; a Final whose buffers are re-mergeable as emitted spills hash-partitioned
   * (#363). A Final that emits a result in a buffer slot (the wide decimal average) or a Complete
   * aggregate, whose input is not buffers, keeps everything in memory.
   */
  private def spillPolicy(aggs: Array[VectorAggFunction]): AggSpillPolicy = {
    val conf = org.apache.spark.sql.internal.SQLConf.get
    val threshold = org.apache.spark.network.util.JavaUtils.byteStringAsBytes(conf.getConfString(
      AggSpillPolicy.ThresholdKey,
      AggSpillPolicy.DefaultThreshold
    ))
    val passThrough =
      conf.getConfString(AggSpillPolicy.PassThroughKey, AggSpillPolicy.DefaultPassThroughRatio.toString).toDouble
    if (threshold <= 0 || groupingExpressions.isEmpty) AggSpillPolicy.InMemory
    else if (aggregateExpressions.isEmpty) {
      // Keys only (a distinct): before the exchange it emits keys, after it it merges them -- either way re-readable.
      if (
        requiredChildDistributionExpressions.isDefined && AggregateSpill.supportsKeys(
          groupingExpressions.map(_.dataType)
        )
      ) {
        val buckets = conf.getConfString(AggSpillPolicy.BucketsKey, AggSpillPolicy.DefaultBuckets.toString).toInt
        AggSpillPolicy.GraceHash(threshold, math.max(2, buckets))
      } else AggSpillPolicy.EmitAndReset(threshold, passThrough)
    } else if (VectorAggregatePlanner.emitsBuffers(modes)) AggSpillPolicy.EmitAndReset(threshold, passThrough)
    else if (
      VectorAggregatePlanner.mergesBuffers(modes) && AggregateSpill.supportsKeys(groupingExpressions.map(_.dataType)) &&
      aggregateExpressions.zip(aggs).forall { case (a, f) =>
        val d = a.aggregateFunction.aggBufferAttributes.map(_.dataType); f.emittedTypes(d) == d
      }
    ) {
      val buckets = conf.getConfString(AggSpillPolicy.BucketsKey, AggSpillPolicy.DefaultBuckets.toString).toInt
      AggSpillPolicy.GraceHash(threshold, math.max(2, buckets))
    } else AggSpillPolicy.InMemory
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val make = partitionIterator
    child.executeColumnar().mapPartitionsInternal(iter => make(iter))
  }

  /**
   * The per-partition iterator as a function built on the driver (everything it captures is
   * compiled here, once), so the rollup chain (#383) can run one of these levels over an input of its
   * own choosing.
   */
  private[vector] def partitionIterator: Iterator[ColumnarBatch] => Iterator[ColumnarBatch] = {
    val aggs = compiled
    val keys = compiledKeys
    val l = layout
    val finalMode = emitsResults
    val policy = spillPolicy(aggs)
    val spillMetrics = (longMetric("spills"), longMetric("spilledGroups"))
    // In result modes a function may emit its result in a buffer slot under the result's type
    // (the wide decimal average), so the batch's column types are the functions' emitted types.
    val bufferAttrs =
      if (finalMode) {
        val declared = aggregateExpressions.map(_.aggregateFunction.aggBufferAttributes.map(_.dataType))
        val types = groupingExpressions.map(_.dataType) ++ aggs.zip(declared).flatMap { case (f, d) =>
          f.emittedTypes(d)
        }
        bufferAttributes.map(_.name).zip(types).toArray
      } else output.map(a => (a.name, a.dataType)).toArray
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val results = if (finalMode) resultProjection else Array.empty[VectorExpr]
    val m = vectorMetrics
    (iter: Iterator[ColumnarBatch]) => {
      val buffers: Iterator[ColumnarBatch] =
        if (keys.isEmpty) new VectorUngroupedAggregateIterator(iter, aggs, l, bufferAttrs, m)
        // Every mode keeps UTF8 keys by id (#377): buffer-emitting modes feed our shuffle writer, which
        // stages the ids per block, and the result modes' byte-record layout (#437) read slower at 1 TB
        // than ids -- q67's final stage 59 -> 67 s, GC 22 -> 38 s -- so it is kept for tests only.
        else new VectorGroupedAggregateIterator(
          iter,
          keys,
          aggs,
          l,
          bufferAttrs,
          m,
          policy,
          Some(spillMetrics),
          dictionaryStrings = true
        )
      if (finalMode) {
        // The projection's own bookkeeping goes to unregistered metrics so rows are not counted twice.
        val scratch =
          new VectorMetrics(new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("timing"))
        new VectorProjectIterator(buffers, results, identity = false, outputAttrs, emitSelection = false, scratch)
      } else buffers
    }
  }

  /** A buffer-emitting level of the rollup chain (#383): the iterator itself, for push-mode feeding (the empty set is ungrouped). */
  private[vector] def levelIterator: Iterator[ColumnarBatch] => RollupLevel = {
    require(!emitsResults, "a rollup level is a buffer-emitting aggregate")
    val aggs = compiled
    val keys = compiledKeys
    val l = layout
    val policy = spillPolicy(aggs)
    val spillMetrics = (longMetric("spills"), longMetric("spilledGroups"))
    val bufferAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    (iter: Iterator[ColumnarBatch]) =>
      if (keys.isEmpty) new VectorUngroupedAggregateIterator(iter, aggs, l, bufferAttrs, m)
      else new VectorGroupedAggregateIterator(iter, keys, aggs, l, bufferAttrs, m, policy, Some(spillMetrics))
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Mode: ${modes.mkString(", ")}
       |Floating point: ${if (strictFloatingPoint) "strict (Spark's rounding)" else "fast"}
       |Keys: ${groupingExpressions.map(_.sql).mkString(", ")}
       |Functions: ${aggregateExpressions.map(_.sql).mkString(", ")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

/**
 * A level of the rollup chain (#383): an aggregate iterator that can also take its input pushed --
 * `feed` one batch at a time (early emissions under a memory budget come back, owned by the caller),
 * then `finishFeed`, then the remaining table through `next`.
 */
private[vector] trait RollupLevel extends Iterator[ColumnarBatch] with AutoCloseable {
  def feed(batch: ColumnarBatch): Seq[ColumnarBatch]
  def finishFeed(): Unit
}

/** Drains the partition, then emits exactly one buffer row. */
private[vector] class VectorUngroupedAggregateIterator(
    input: Iterator[ColumnarBatch],
    aggs: Array[VectorAggFunction],
    layout: Array[OutputSlot],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics
) extends RollupLevel {

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorHashAggregateExec")
  private var emitted = false
  private var result: ColumnarBatch = _
  private var closed = false
  private val states: Array[AggState] = aggs.map(_.newState())

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = !emitted

  private def update(batch: ColumnarBatch): Unit = if (batch.numRows() > 0) {
    metrics.timed {
      metrics.numInputBatches += 1
      EvalContexts.withBatch(batch) { ctx =>
        var i = 0
        while (i < states.length) { states(i).update(ctx); i += 1 }
      }
    }
  }

  override def feed(batch: ColumnarBatch): Seq[ColumnarBatch] = { update(batch); Nil }
  override def finishFeed(): Unit = ()

  override def next(): ColumnarBatch = {
    if (emitted) throw new NoSuchElementException("aggregate already emitted")
    emitted = true
    while (input.hasNext) update(input.next())
    val buffers = states.map(_.bufferValues)
    val columns = new Array[ColumnVector](layout.length)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = outputAttrs(c)
      columns(c) = layout(c) match {
        case BufferSlot(aggIdx, slot) => ArrowOutput.scalarColumn(name, dt, buffers(aggIdx)(slot), allocator)
        case KeySlot(_) => throw new IllegalStateException("key slot in ungrouped aggregate")
      }
      c += 1
    }
    metrics.numOutputBatches += 1
    metrics.numOutputRows += 1
    result = new ColumnarBatch(columns, 1)
    result
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      if (result != null) { result.close(); result = null }
      allocator.close()
    }
  }
}

/**
 * Drains the partition into a group table, then emits the groups in batches. Past its memory budget
 * (#363) it either emits the table and starts over (buffer-emitting modes: the next stage merges)
 * or spills the table hash-partitioned and merges one bucket at a time (Final: [[AggregateSpill]]).
 */
private[vector] class VectorGroupedAggregateIterator(
    input: Iterator[ColumnarBatch],
    keyExprs: Array[VectorExpr],
    aggs: Array[VectorAggFunction],
    layout: Array[OutputSlot],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics,
    policy: AggSpillPolicy = AggSpillPolicy.InMemory,
    spillMetrics: Option[(SQLMetric, SQLMetric)] = None,
    dictionaryStrings: Boolean = true
) extends RollupLevel {

  private val OutputBatchSize = 4096

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorHashAggregateExec")
  private var table = new GroupKeyTable(keyExprs.map(_.vecType), dictionaryStrings)
  private val dictionaryKeys = VectorConf.aggDictionaryKeys(org.apache.spark.sql.internal.SQLConf.get)
  private val keyDicts = new Array[SharedDictionary](keyExprs.length)
  private var states: Array[GroupedAggState] = aggs.map(_.newGroupedState())
  private var idScratch = new Array[Int](0)
  private var inputDone = false
  private var emittedGroups = 0
  private var current: ColumnarBatch = _
  private var closed = false

  // Grace hash (Final past the budget): the spill, the bucket being merged and its batches.
  private var spill: AggregateSpill = _
  private var bucket = -1
  private var bucketInput: Iterator[ColumnarBatch] with AutoCloseable = _

  /**
   * The accumulators' heap per group at capacity (#367): each buffer slot is up to two arrays of
   * longs (a sum and a count), interleaved `INTERLEAVE` ways for the vector kernels, at the capacity
   * they have grown to (64, doubling); the table reports its own arrays as allocated. The copy a
   * growth step holds -- the new arrays at twice the capacity next to the old -- is counted only once
   * a doubling is imminent (#376: counted always, the estimate ran two to three times the live
   * footprint and the pool's answer emptied tables that fit).
   */
  private val accumulatorBytesPerGroup: Long =
    aggs.map(a =>
      math.max(1, a.bufferTypes.length).toLong * 2L * 8L * io.vecruntime.kernels.GroupedAccumulators.INTERLEAVE
    ).sum
  private def estimatedBytes: Long = {
    val groups = table.size()
    var capacity = 64L
    while (capacity < groups) capacity <<= 1
    val accumulators = capacity * accumulatorBytesPerGroup
    val growth = if (groups * 8L >= capacity * 7L) 2L * accumulators else 0L
    table.memoryBytes() + accumulators + growth
  }

  // Spark's execution memory as the arbiter (#367): the estimate is acquired from the task's share of the
  // pool as the table grows, and a refusal -- the executor's tasks share it fairly -- is what triggers the
  // emit or the spill, with the configured threshold as a hard cap. Nothing is spilled on another
  // consumer's request; the memory is returned on every reset and at close.
  private val consumer: org.apache.spark.memory.MemoryConsumer = Option(TaskContext.get()).map { tc =>
    val tmm = tc.taskMemoryManager()
    new org.apache.spark.memory.MemoryConsumer(tmm, tmm.pageSizeBytes(), org.apache.spark.memory.MemoryMode.ON_HEAP) {
      override def spill(size: Long, trigger: org.apache.spark.memory.MemoryConsumer): Long = 0L
    }
  }.orNull
  private var reserved = 0L

  /** Whether the table has outgrown its budget: the cap, or what Spark's pool grants this task. */
  private def overBudget(budget: Long): Boolean = {
    val estimate = estimatedBytes
    if (estimate > budget) return true
    if (consumer != null && estimate > reserved) {
      reserved += consumer.acquireMemory(estimate - reserved)
      if (reserved < estimate) return true
    }
    false
  }

  private def releaseMemory(): Unit =
    if (consumer != null && reserved > 0) { consumer.freeMemory(reserved); reserved = 0L }

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  private def consume(batch: ColumnarBatch): Unit = if (batch.numRows() > 0) {
    metrics.timed {
      metrics.numInputBatches += 1
      EvalContexts.withBatch(batch) { ctx =>
        // Physical rows: a normalized foreign batch reports its live count as numRows.
        val n = ctx.numRows
        if (idScratch.length < n) idScratch = new Array[Int](n)
        val keys = new Array[VectorBuffers](keyExprs.length)
        var k = 0
        while (k < keys.length) { keys(k) = keyExprs(k).eval(ctx); k += 1 }
        val numGroups = table.assign(keys, n, idScratch, ctx.selection)
        val assignment = GroupAssignment.of(idScratch, n, numGroups, ctx.arena, ctx.selection)
        var i = 0
        while (i < states.length) { states(i).update(ctx, assignment); i += 1 }
      }
    }
  }

  // Rows consumed into the current table, for the reduction ratio a partial aggregate is judged by (#376).
  private var fillRows = 0L
  // Pass-through: the input has shown it does not reduce, so each batch goes out as its own table.
  private var passThrough = false

  /** Consumes `source` into the table until it is exhausted (true) or the table passes `budget` (false; 0 = no budget). */
  private def fill(source: Iterator[ColumnarBatch], budget: Long): Boolean = {
    while (source.hasNext) {
      val batch = source.next()
      fillRows += batch.numRows()
      consume(batch)
      if (passThrough) return !source.hasNext
      if (budget > 0 && overBudget(budget)) return false
    }
    true
  }

  private def reset(): Unit = {
    releaseMemory()
    releaseDictionaries()
    table = new GroupKeyTable(keyExprs.map(_.vecType), dictionaryStrings)
    states = aggs.map(_.newGroupedState())
    emittedGroups = 0
    fillRows = 0L
  }

  /** The whole table into the spill, as the batches the operator would emit. */
  private def spillTable(buckets: Int): Unit = {
    if (spill == null) {
      val keyOrdinals = layout.zipWithIndex.collect { case (KeySlot(_), i) => i }
      spill = new AggregateSpill(buckets, outputAttrs, keyOrdinals, allocator)
    }
    var from = 0
    while (from < table.size()) {
      val to = math.min(table.size(), from + OutputBatchSize)
      val b = buildBatch(from, to)
      try spill.write(b)
      finally b.close()
      from = to
    }
    spillMetrics.foreach { case (spills, rows) => spills += 1; rows += table.size() }
  }

  /** Makes groups available to emit; false when the iterator is exhausted. */
  private def advance(): Boolean = {
    if (fed) return inputDone && emittedGroups < table.size()
    while (emittedGroups >= table.size()) {
      policy match {
        case AggSpillPolicy.InMemory =>
          if (inputDone) return false
          fill(input, 0L)
          inputDone = true
        case AggSpillPolicy.EmitAndReset(budget, ratio) =>
          if (inputDone) return false
          reset()
          inputDone = fill(input, budget)
          // Past the budget: this table goes out now and the next stage merges it with the rest. A table
          // that reduced its rows by less than `ratio` says the input does not aggregate here: from now on
          // every batch is its own table (#376) -- no growth, no copies, the exchange gets what it would
          // have got anyway.
          if (!inputDone && !passThrough) {
            spillMetrics.foreach { case (spills, groups) => spills += 1; groups += table.size() }
            if (ratio > 0 && table.size() > 0 && fillRows < ratio * table.size()) passThrough = true
          }
          if (table.size() == 0 && inputDone) return false
        case AggSpillPolicy.GraceHash(budget, buckets) =>
          if (!inputDone) {
            // Phase 1: the input, the whole table spilled whenever it passes the budget.
            while (!fill(input, budget)) { spillTable(buckets); reset() }
            inputDone = true
            if (spill == null) return table.size() > 0 // never spilled: the table is the answer
            spillTable(buckets); reset() // the remainder joins its buckets
          } else {
            // Phase 2: one bucket at a time, merged through the same path as the exchange's batches.
            if (spill == null) return false
            if (bucketInput != null) { bucketInput.close(); bucketInput = null }
            bucket += 1
            if (bucket >= buckets) return false
            reset()
            bucketInput = spill.read(bucket)
            fill(bucketInput, 0L)
          }
      }
    }
    true
  }

  override def hasNext: Boolean = advance()

  // Push mode (#383): a rollup level receives the previous level's partial rows through feed() instead
  // of pulling an input; finishFeed() closes the input and the table is emitted through next().
  private var fed = false

  /**
   * Consumes one batch pushed by the caller. Under an emit-and-reset policy a table past its budget is
   * returned as finished batches (the caller owns them and closes them) and the table starts over.
   */
  override def feed(batch: ColumnarBatch): Seq[ColumnarBatch] = {
    fed = true
    fillRows += batch.numRows()
    consume(batch)
    policy match {
      case AggSpillPolicy.EmitAndReset(budget, _) if budget > 0 && overBudget(budget) =>
        spillMetrics.foreach { case (spills, groups) => spills += 1; groups += table.size() }
        val out = Seq.newBuilder[ColumnarBatch]
        var from = 0
        while (from < table.size()) {
          val to = math.min(table.size(), from + OutputBatchSize)
          out += buildBatch(from, to)
          metrics.numOutputBatches += 1
          metrics.numOutputRows += (to - from)
          from = to
        }
        reset()
        out.result()
      case _ => Nil
    }
  }

  /** No more batches will be fed: what the table holds is emitted through next(). */
  override def finishFeed(): Unit = { fed = true; inputDone = true }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more groups")
    releaseCurrent()
    val from = emittedGroups
    val to = math.min(table.size(), from + OutputBatchSize)
    current = buildBatch(from, to)
    emittedGroups = to
    metrics.numOutputBatches += 1
    metrics.numOutputRows += (to - from)
    current
  }

  private def buildBatch(from: Int, to: Int): ColumnarBatch = {
    val count = to - from
    val columns = new Array[ColumnVector](layout.length)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = outputAttrs(c)
      columns(c) = layout(c) match {
        case KeySlot(k) => keyColumn(name, dt, k, from, to)
        case BufferSlot(aggIdx, slot) => bufferColumn(name, dt, states(aggIdx), slot, from, to)
      }
      c += 1
    }
    new ColumnarBatch(columns, count)
  }

  private def keyColumn(name: String, dt: DataType, k: Int, from: Int, to: Int): ColumnVector = {
    val count = to - from
    if (table.`type`(k) == VecType.UTF8) {
      val dict = sharedDictionary(name, k)
      if (dict != null) {
        // The keys as ids over the table's dictionary (#377): 4 bytes a row, and the consumers --
        // the shuffle writer, the final aggregate -- work on the ids and read the dictionary once.
        val ids = ArrowOutput.allocateFixed(name, IntegerType, count, allocator)
        table.writeKeyIds(k, from, to, ids.validity(), ids.data())
        ArrowOutput.finish(ids, count, false)
        dict.retain()
        new VectorDictionaryColumnVector(ids.vector().asInstanceOf[IntVector], dict.vector, () => dict.release())
      } else {
        val out = ArrowOutput.allocateUtf8(name, count, table.utf8Bytes(k, from, to), allocator)
        table.writeKeys(k, from, to, out.validity(), out.data(), out.offsets())
        ArrowOutput.finish(out, count, false)
      }
    } else {
      val out = ArrowOutput.allocateFixed(name, dt, count, allocator)
      table.writeKeys(k, from, to, out.validity(), out.data(), null)
      ArrowOutput.finish(out, count, false)
    }
  }

  /**
   * One Arrow vector of UTF8 key column {@code k}'s dictionary per emission, shared by the key
   * batches through a reference count (the batches of an emit-and-reset burst outlive the reset),
   * or null when the column goes out plain (dictionary encoding is off, or the column has no
   * value). The dictionary's bytes never exceed the plain column's -- it holds each distinct value
   * once -- and the consumers work on the ids: the shuffle writer stages them and remaps per block
   * (going plain itself when a block's dictionary does not pay, #356), the final aggregate maps the
   * dictionary's entries rather than the rows.
   */
  private def sharedDictionary(name: String, k: Int): SharedDictionary = {
    if (!dictionaryKeys || !table.isDictionaryColumn(k)) return null
    val distinct = table.dictionarySize(k)
    if (distinct == 0) return null
    var d = keyDicts(k)
    if (d != null && d.size != distinct) { d.release(); d = null; keyDicts(k) = null }
    if (d == null) {
      val view = table.dictionary(k)
      val bytes = view.offsets().get(VectorBuffers.LE_INT, distinct.toLong << 2)
      val out = ArrowOutput.allocateUtf8(name, distinct, bytes, allocator)
      MemorySegment.copy(view.offsets(), 0L, out.offsets(), 0L, (distinct + 1).toLong << 2)
      MemorySegment.copy(view.data(), 0L, out.data(), 0L, bytes.toLong)
      ArrowOutput.finish(out, distinct, true)
      d = new SharedDictionary(out.vector().asInstanceOf[VarCharVector], distinct)
      keyDicts(k) = d
    }
    d
  }

  /** The dictionary vector of one key column for one emission; closed when the iterator and every batch let go. */
  private final class SharedDictionary(val vector: VarCharVector, val size: Int) {
    private var refs = 1
    def retain(): Unit = synchronized { refs += 1 }
    def release(): Unit = synchronized {
      if (refs <= 0) throw new IllegalStateException(s"dictionary released $refs"); refs -= 1;
      if (refs == 0) vector.close()
    }
  }

  private def releaseDictionaries(): Unit = {
    var k = 0
    while (k < keyDicts.length) { if (keyDicts(k) != null) { keyDicts(k).release(); keyDicts(k) = null }; k += 1 }
  }

  private def bufferColumn(
      name: String,
      dt: DataType,
      state: GroupedAggState,
      slot: Int,
      from: Int,
      to: Int
  ): ColumnVector =
    AggBufferColumns.column(name, dt, state, slot, from, to, allocator)

  private def releaseCurrent(): Unit = {
    if (current != null) { current.close(); current = null }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseCurrent()
      if (bucketInput != null) {
        try bucketInput.close()
        catch { case _: Exception => }; bucketInput = null
      }
      if (spill != null) {
        try spill.close()
        catch { case _: Exception => }; spill = null
      }
      releaseMemory()
      releaseDictionaries()
      allocator.close()
    }
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorAggregatePlanner {

  private val keyTypes: Set[VecType] =
    Set(VecType.INT32, VecType.INT64, VecType.BOOL, VecType.UTF8, VecType.FLOAT64, VecType.DECIMAL128)

  /**
   * Grouping keys: ints, longs, booleans, strings, doubles and wide decimals (two limbs hashed and
   * compared, #259). The key table compares doubles by
   * bits, which matches Spark because the optimizer wraps double grouping keys in
   * NormalizeNaNAndZero (compiled to a real normalisation pass) at the updating stage, and the
   * merging stage, like Spark's own, groups the already normalised values.
   */
  def compileKey(e: NamedExpression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compileLaneColumn(e, input).flatMap {
      case _: LiteralExpr => Left("literal grouping key")
      case k if !keyTypes.contains(k.vecType) => Left(s"grouping key type ${e.dataType.simpleString} not supported")
      case k => Right(k)
    }

  /** The merged-buffer batch of a Final aggregate: keys, then every function's buffer slots. */
  def bufferLayout(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression]
  ): Seq[OutputSlot] =
    groupingExpressions.indices.map(KeySlot(_): OutputSlot) ++
      aggregateExpressions.zipWithIndex.flatMap { case (agg, i) =>
        agg.aggregateFunction.aggBufferAttributes.indices.map(slot => BufferSlot(i, slot): OutputSlot)
      }

  /** Attributes of [[bufferLayout]]'s columns, the input of a Final aggregate's result projection. */
  def bufferAttributes(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression]
  ): Seq[Attribute] =
    groupingExpressions.map(_.toAttribute) ++ aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)

  /**
   * Final-mode result expressions with each aggregate's result attribute replaced by the
   * function's `evaluateExpression`, compiled against the merged-buffer batch.
   */
  def compileFinalResults(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression],
      aggregateAttributes: Seq[Attribute],
      resultExpressions: Seq[NamedExpression]
  ): Either[String, Seq[VectorExpr]] = {
    val input = bufferAttributes(groupingExpressions, aggregateExpressions)
    // The result expressions reference the operator's `aggregateAttributes`, positionally aligned with
    // its aggregate expressions -- not necessarily each expression's own `resultAttribute`: Spark's
    // distinct rewrite builds the Final distinct expression afresh but keeps the original attribute.
    val evaluate: Map[org.apache.spark.sql.catalyst.expressions.ExprId, Expression] =
      aggregateExpressions.zip(aggregateAttributes).flatMap { case (agg, attr) =>
        agg.aggregateFunction match {
          case d: DeclarativeAggregate =>
            Seq(attr.exprId -> d.evaluateExpression, agg.resultAttribute.exprId -> d.evaluateExpression)
          case _ => Nil
        }
      }.toMap
    val compiled = resultExpressions.map { e =>
      val substituted = e.transform { case a: AttributeReference if evaluate.contains(a.exprId) => evaluate(a.exprId) }
      (wideResult(substituted, input) match {
        // The merge already applied isEmpty / the count test and the overflow rule at emission: forward the column.
        case Some(ordinal) => Right(ColumnRef(ordinal, e.dataType))
        // A literal result (`'store' AS channel` beside the aggregates) is a constant column the
        // result projection materialises like any other.
        case None =>
          // A wide sum or average inside a larger expression (`sum(x) / 7.0`, `0.5 * sum(x)`,
          // `sum(a) - sum(b)`, #259): each emitted result stands in for its sub-expression as a column
          // typed as the result (the slot's attribute carries the buffer type, the emitted column the
          // result's), and the arithmetic around it compiles through the wide kernels (#258).
          val forwarded = substituted.transform {
            case sub if sub.isInstanceOf[If] && wideResult(sub, input).isDefined =>
              val a = input(wideResult(sub, input).get)
              AttributeReference(a.name, sub.dataType, sub.nullable, a.metadata)(a.exprId, a.qualifier)
          }
          val body = forwarded match { case Alias(c, _) => c; case other => other }
          ExpressionCompiler.compileLaneColumn(body, input).flatMap {
            case v if !TypeMapping.isSupported(e.dataType) && !TypeMapping.hasLane(e.dataType) =>
              Left(s"unsupported result type ${e.dataType.simpleString} for ${e.name}")
            case v => Right(v)
          }
      }).left.map(r => s"${e.sql}: $r")
    }
    compiled.collectFirst { case Left(r) => r } match {
      case Some(reason) => Left(reason)
      case None => Right(compiled.collect { case Right(v) => v })
    }
  }

  /**
   * A wide decimal aggregate's result over its own buffer attributes, in the shape and only the
   * shape Spark builds: a decimal `Sum`'s `If(isEmpty, null, CheckOverflowInSum(sum, ...))`, or a
   * decimal `Average`'s `If(count = 0, null, DecimalDivideWithOverflowCheck(sum, count, ...))`. The
   * function emits the result in the sum slot already, so the projection is the ordinal of that
   * slot. Returns None for anything else.
   */
  private def wideResult(e: Expression, input: Seq[Attribute]): Option[Int] = {
    val body = e match { case Alias(child, _) => child; case other => other }
    def slot(sum: AttributeReference, other: AttributeReference): Option[Int] = {
      val ordinal = input.indexWhere(_.exprId == sum.exprId)
      if (ordinal >= 0 && input.exists(_.exprId == other.exprId)) Some(ordinal) else None
    }
    body match {
      case If(
            isEmpty: AttributeReference,
            Literal(null, _),
            CheckOverflowInSum(sum: AttributeReference, dt: DecimalType, _, _)
          )
          if dt.precision > TypeMapping.MAX_DECIMAL_PRECISION && sum.dataType == dt =>
        slot(sum, isEmpty)
      case If(
            EqualTo(count: AttributeReference, Literal(0L, _)),
            Literal(null, _),
            DecimalDivideWithOverflowCheck(sum: AttributeReference, Cast(count2: AttributeReference, _, _, _), _, _, _)
          )
          if count.exprId == count2.exprId && sum.dataType.isInstanceOf[
            DecimalType
          ] && sum.dataType.asInstanceOf[DecimalType].precision > TypeMapping.MAX_DECIMAL_PRECISION =>
        slot(sum, count)
      case _ => None
    }
  }

  /** Wide decimal sum and average buffers a merging aggregate reads: the one kind of wide input the operator accepts. */
  def wideBuffers(a: BaseAggregateExec): Set[org.apache.spark.sql.catalyst.expressions.ExprId] =
    a.aggregateExpressions.zip(VectorAggregates.bufferOffsets(a)).collect {
      case (agg, offset) if VectorAggregates.merges(agg.mode) && wideDecimalBuffer(agg.aggregateFunction) =>
        // By id, and by Spark's position when the Final's function instance carries fresh ids.
        Seq(agg.aggregateFunction.inputAggBufferAttributes.head.exprId) ++ a.child.output.lift(offset).map(_.exprId)
    }.flatten.toSet

  private def wideDecimalBuffer(f: org.apache.spark.sql.catalyst.expressions.aggregate.AggregateFunction): Boolean =
    f match {
      case s: Sum => s.dataType.isInstanceOf[DecimalType]
      case a: Average => a.child.dataType.isInstanceOf[DecimalType]
      case _ => false
    }

  /** Maps each result attribute to the grouping key or the (aggregate, buffer slot) producing it. */
  def outputLayout(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression]
  ): Either[String, Seq[OutputSlot]] = {
    val keySlots: Map[org.apache.spark.sql.catalyst.expressions.ExprId, OutputSlot] =
      groupingExpressions.zipWithIndex.map { case (g, i) => g.toAttribute.exprId -> (KeySlot(i): OutputSlot) }.toMap
    val bufferSlots: Map[org.apache.spark.sql.catalyst.expressions.ExprId, OutputSlot] =
      aggregateExpressions.zipWithIndex.flatMap { case (agg, i) =>
        agg.aggregateFunction.inputAggBufferAttributes.zipWithIndex.map { case (a, slot) =>
          a.exprId -> (BufferSlot(i, slot): OutputSlot)
        }
      }.toMap
    // A buffer-emitting stage's result attributes are its keys then its buffers in order, so a result
    // attribute whose id matches nothing (a function instance rewritten between planning and this
    // check) is still bound by that position, as Spark binds it.
    val positional: Seq[OutputSlot] =
      groupingExpressions.indices.map(KeySlot(_): OutputSlot) ++
        aggregateExpressions.zipWithIndex.flatMap { case (agg, i) =>
          agg.aggregateFunction.inputAggBufferAttributes.indices.map(BufferSlot(i, _): OutputSlot)
        }
    val mapped = resultExpressions.zipWithIndex.map {
      case (a: Attribute, pos) =>
        keySlots.get(a.exprId).orElse(bufferSlots.get(a.exprId)).orElse(if (
          resultExpressions.length == positional.length
        ) positional.lift(pos)
        else None)
          .toRight(s"result attribute ${a.name} is neither a grouping key nor an aggregation buffer")
      // A grouping key under another name (`ss_customer_sk AS customer_sk ... GROUP BY ss_customer_sk`,
      // TPC-DS q97): the same lane, emitted under the alias's attribute (#328).
      case (Alias(a: Attribute, name), _) =>
        keySlots.get(a.exprId).orElse(bufferSlots.get(a.exprId))
          .toRight(
            s"result expression $name aliases ${a.name}, which is neither a grouping key nor an aggregation buffer"
          )
      case (other, _) => Left(s"result expression ${other.sql} is not a plain attribute")
    }
    mapped.collectFirst { case Left(r) => r } match {
      case Some(reason) => Left(reason)
      case None => Right(mapped.collect { case Right(x) => x })
    }
  }

  /** Every mode evaluates result expressions (`Final`, `Complete`). */
  def emitsResults(modes: Seq[AggregateMode]): Boolean =
    modes.nonEmpty && modes.forall(m => m == Final || m == Complete)

  /** Every mode emits aggregation buffers for a later stage (`Partial`, `PartialMerge`). */
  def emitsBuffers(modes: Seq[AggregateMode]): Boolean =
    modes.nonEmpty && modes.forall(m => m == Partial || m == PartialMerge)

  /** Every mode merges buffers (`PartialMerge`, `Final`): the child is an exchange whose types alone matter. */
  def mergesBuffers(modes: Seq[AggregateMode]): Boolean = modes.nonEmpty && modes.forall(VectorAggregates.merges)

  /**
   * Whether the operator reads an exchange (so only its input types matter): the merge modes, or a
   * keys-only aggregate (`SELECT DISTINCT`, the first aggregate of Spark's distinct rewrite) whose
   * required child distribution is set -- Spark plans those as a Partial with no distribution and a
   * Final over the exchange with the keys as distribution, and both merely emit the keys.
   */
  def readsExchange(a: BaseAggregateExec): Boolean =
    if (a.aggregateExpressions.isEmpty) a.requiredChildDistributionExpressions.isDefined
    else mergesBuffers(a.aggregateExpressions.map(_.mode).distinct)

  /**
   * Attempts to convert a Spark HashAggregateExec; Left explains the fallback. Two independent
   * decisions: each aggregate expression's mode says whether its state is updated from the input
   * (`Partial`, `Complete`) or merged from buffers (`PartialMerge`, `Final`); the operator's mode set
   * says whether it emits buffers (`Partial` / `PartialMerge`, as Spark's distinct rewrite mixes
   * them) or results (`Final` / `Complete`). `finalEnabled` gates the modes that read an exchange.
   */
  def plan(
      a: BaseAggregateExec,
      finalEnabled: Boolean,
      strictFloatingPoint: Boolean = true
  ): Either[String, VectorHashAggregateExec] = {
    val modes = a.aggregateExpressions.map(_.mode).distinct
    val keysOnly = a.aggregateExpressions.isEmpty
    // A keys-only aggregate emits its keys in both of Spark's stages: the buffer layout fits both.
    val results = !keysOnly && emitsResults(modes)
    if (keysOnly && a.groupingExpressions.isEmpty) Left("aggregate without keys or functions")
    else if (!keysOnly && !results && !emitsBuffers(modes))
      Left(s"aggregation modes ${modes.mkString(", ")} mix buffer and result output")
    else if (readsExchange(a) && !finalEnabled) Left("merging aggregation stages disabled by configuration")
    else {
      val keyFailures =
        a.groupingExpressions.flatMap(g => compileKey(g, a.child.output).left.toOption.map(r => s"${g.sql}: $r"))
      val aggFailures = a.aggregateExpressions.zip(VectorAggregates.bufferOffsets(a)).flatMap { case (agg, offset) =>
        VectorAggregates.compile(agg, a.child.output, offset).left.toOption.map(r => s"${agg.sql}: $r")
      }
      val failures = keyFailures ++ aggFailures
      val layoutCheck: Either[String, Seq[Any]] =
        if (results)
          compileFinalResults(a.groupingExpressions, a.aggregateExpressions, a.aggregateAttributes, a.resultExpressions)
        else outputLayout(a.groupingExpressions, a.aggregateExpressions, a.resultExpressions)
      if (failures.nonEmpty) Left(failures.mkString("; "))
      else layoutCheck.flatMap { compiled =>
        // A wide decimal (p > 18) output is a DECIMAL128 lane in three places: a buffer-emitting
        // operator's `sum` buffer of a decimal sum (Spark's Decimal(p + 10, s)), a Final's result of
        // that sum, which the merge emits ready-made and the projection forwards, and a Final's
        // arithmetic over such results, which the wide kernels compute (#259).
        val wideOutputs: Set[org.apache.spark.sql.catalyst.expressions.ExprId] =
          if (results) a.resultExpressions.zip(compiled).collect {
            case (e, v: VectorExpr)
                if e.dataType.isInstanceOf[DecimalType] && e.dataType.asInstanceOf[
                  DecimalType
                ].precision > TypeMapping.MAX_DECIMAL_PRECISION && v.vecType == VecType.DECIMAL128 =>
              e.toAttribute.exprId
          }.toSet
          else (a.groupingExpressions.map(_.toAttribute) ++ a.aggregateExpressions.flatMap(
            _.aggregateFunction.inputAggBufferAttributes
          )).collect {
            case attr if attr.dataType.isInstanceOf[DecimalType] => attr.exprId
          }.toSet
        a.resultExpressions.map(_.toAttribute).find(attr =>
          !TypeMapping.isSupported(attr.dataType) && !wideOutputs.contains(attr.exprId)
        ) match {
          case Some(attr) => Left(s"unsupported output type ${attr.dataType.simpleString} for ${attr.name}")
          case None =>
            Right(VectorHashAggregateExec(
              a.requiredChildDistributionExpressions,
              a.groupingExpressions,
              a.aggregateExpressions,
              a.aggregateAttributes,
              a.resultExpressions,
              a.child,
              strictFloatingPoint
            ))
        }
      }
    }
  }
}

/** Boxed buffer values of a run of groups `[from, to)` as one Arrow column; shared with the window aggregate. */
object AggBufferColumns {
  def column(
      name: String,
      dt: DataType,
      state: GroupedAggState,
      slot: Int,
      from: Int,
      to: Int,
      allocator: BufferAllocator
  ): ColumnVector = {
    // A state that can write its lane directly does (#416); the rest go value by value, boxed.
    if (dt != org.apache.spark.sql.types.StringType) {
      val out = ArrowOutput.allocateFixed(name, dt, to - from, allocator)
      if (state.writeBuffer(slot, from, to, out)) return ArrowOutput.finish(out, to - from, false)
      out.vector().close()
    }
    values(name, dt, to - from, o => state.bufferValue(from + o, slot), allocator)
  }

  /** Boxed values `get(0 until count)` (Spark's internal representation, or null) as one Arrow column. */
  def values(name: String, dt: DataType, count: Int, get: Int => Any, allocator: BufferAllocator): ColumnVector = {
    dt match {
      case d: DecimalType if d.precision > TypeMapping.MAX_DECIMAL_PRECISION =>
        // The sum buffer of a wide decimal sum: the exact totals written as two limbs each into a
        // DECIMAL128 lane (Arrow's 128-bit vector), which the merge side reads back in place (#257).
        val out = ArrowOutput.allocateFixed(name, d, count, allocator)
        val data = out.data()
        val validity = out.validity()
        var anyNull = false
        var o = 0
        while (o < count) {
          get(o) match {
            case null =>
              anyNull = true
              Bitmap.setTo(validity, o, false)
              io.vecruntime.kernels.Decimal128.set(data, o, 0L, 0L)
            case v: java.math.BigDecimal =>
              val unscaled = v.unscaledValue()
              Bitmap.setTo(validity, o, true)
              io.vecruntime.kernels.Decimal128.set(
                data,
                o,
                io.vecruntime.kernels.Decimal128.hiOf(unscaled),
                io.vecruntime.kernels.Decimal128.loOf(unscaled)
              )
          }
          o += 1
        }
        return ArrowOutput.finish(out, count, !anyNull)
      case org.apache.spark.sql.types.StringType =>
        // min/max, first/last, min_by/max_by over strings: boxed UTF8Strings into a varchar vector.
        val values = new Array[Array[Byte]](count)
        var o = 0
        while (o < count) {
          val v = get(o)
          values(o) = if (v == null) null else v.asInstanceOf[org.apache.spark.unsafe.types.UTF8String].getBytes
          o += 1
        }
        return ArrowOutput.utf8Column(name, values, allocator)
      case _ =>
    }
    val out: ArrowVectorBuffers = ArrowOutput.allocateFixed(name, dt, count, allocator)
    val data = out.data()
    val validity = out.validity()
    var o = 0
    while (o < count) {
      get(o) match {
        case null => io.vecruntime.kernels.Bitmap.clear(validity, o)
        case v: java.lang.Double =>
          io.vecruntime.kernels.Bitmap.set(validity, o);
          data.set(VectorBuffers.LE_DOUBLE, o.toLong << 3, v.doubleValue())
        case v: java.lang.Long =>
          io.vecruntime.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_LONG, o.toLong << 3, v.longValue())
        case v: java.lang.Integer =>
          io.vecruntime.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_INT, o.toLong << 2, v.intValue())
        case v: java.lang.Byte => // tinyint / smallint values ride the int lane (#327)
          io.vecruntime.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_INT, o.toLong << 2, v.intValue())
        case v: java.lang.Short =>
          io.vecruntime.kernels.Bitmap.set(validity, o); data.set(VectorBuffers.LE_INT, o.toLong << 2, v.intValue())
        case v: java.lang.Boolean =>
          io.vecruntime.kernels.Bitmap.set(validity, o); io.vecruntime.kernels.Bitmap.setTo(data, o, v.booleanValue())
        case other => throw new IllegalStateException(s"unexpected buffer value $other")
      }
      o += 1
    }
    ArrowOutput.finish(out, count, false)
  }
}
