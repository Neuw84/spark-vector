package org.apache.spark.sql.vector

import java.lang.foreign.{Arena, MemorySegment}
import java.util.ArrayDeque

import io.sparkvector.kernels._
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, VectorAllocators}
import io.sparkvector.spark.expr.{EvalContext, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, BindReferences, Expression}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, HashedRelationBroadcastMode, HashJoin, ShuffledHashJoinExec}
import org.apache.spark.sql.execution.vector.HashedRelationAccess
import org.apache.spark.sql.types.{DataType, DecimalType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * What the two hash joins share: an equi-join whose build side is turned into columns plus a
 * [[GroupKeyTable]] keyed on the build keys, probed batch by batch with the streamed keys, the
 * matches gathered into output batches by [[GatherKernels]].
 *
 * Inner, left outer, right outer (build side left), full outer, left semi and left anti joins are
 * supported, each with an optional non-equi `condition`. Inner joins evaluate it on the joined
 * batch and compact the rows failing it away. Semi, anti and outer joins evaluate it on the
 * candidate pairs of every streamed row first and only then decide what that row becomes: kept or
 * dropped (semi/anti), its passing pairs or one null-padded row (outer). A full outer join also
 * remembers which build rows were ever paired and emits the rest, null-padded, after the last
 * streamed batch. Null keys never match, as in Spark. Double keys are refused: Spark compares them
 * after NaN/zero normalisation, the key table by bits.
 */
trait VectorHashJoinLike extends VectorBinaryExec {
  def leftKeys: Seq[Expression]
  def rightKeys: Seq[Expression]
  def joinType: JoinType
  def buildSide: BuildSide
  def condition: Option[Expression]

  def buildPlan: SparkPlan = buildSide match { case BuildLeft => left; case BuildRight => right }
  def streamedPlan: SparkPlan = buildSide match { case BuildLeft => right; case BuildRight => left }
  def buildKeys: Seq[Expression] = buildSide match { case BuildLeft => leftKeys; case BuildRight => rightKeys }
  def streamedKeys: Seq[Expression] = buildSide match { case BuildLeft => rightKeys; case BuildRight => leftKeys }

  override def output: Seq[Attribute] = joinType match {
    case _: InnerLike => left.output ++ right.output
    case LeftOuter => left.output ++ right.output.map(_.withNullability(true))
    case RightOuter => left.output.map(_.withNullability(true)) ++ right.output
    case FullOuter => left.output.map(_.withNullability(true)) ++ right.output.map(_.withNullability(true))
    case LeftSemi | LeftAnti => left.output
    // EXISTS used as a value: every left row plus the boolean the subquery planner named.
    case ExistenceJoin(exists) => left.output :+ exists
    case _ => left.output ++ right.output // never planned; see VectorJoinPlanner.supportedType
  }

  /** The joined row a condition sees: both sides, whatever the join type outputs. */
  private def joinedOutput: Seq[Attribute] = left.output ++ right.output

  @transient protected lazy val compiledBuildKeys: Array[VectorExpr] = VectorJoinPlanner.compileKeys(buildKeys, buildPlan.output)
  @transient protected lazy val compiledStreamedKeys: Array[VectorExpr] = VectorJoinPlanner.compileKeys(streamedKeys, streamedPlan.output)
  @transient protected lazy val compiledCondition: Option[VectorExpr] =
    condition.map(c => VectorJoinPlanner.compileCondition(c, joinedOutput).fold(r => throw new IllegalStateException(s"cannot vectorize join condition: $r"), identity))

  protected def joinSpec: JoinSpec = JoinSpec(
    joinType,
    buildIsLeft = buildSide == BuildLeft,
    compiledBuildKeys,
    compiledStreamedKeys,
    compiledCondition,
    output.map(a => (a.name, a.dataType)).toArray,
    joinedOutput.map(a => (a.name, a.dataType)).toArray,
    buildPlan.output.map(_.dataType).toArray,
    streamedPlan.output.length)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Left keys: ${leftKeys.map(_.sql).mkString(", ")}
       |Right keys: ${rightKeys.map(_.sql).mkString(", ")}
       |Join type: $joinType, build $buildSide
       |Condition: ${condition.map(_.sql).getOrElse("none")}
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

/** Everything a task needs to run the join, serialised with the operator. */
final case class JoinSpec(
    joinType: JoinType,
    buildIsLeft: Boolean,
    buildKeys: Array[VectorExpr],
    streamedKeys: Array[VectorExpr],
    condition: Option[VectorExpr],
    /** The operator's output: left ++ right, or the left side alone for semi/anti joins. */
    outputAttrs: Array[(String, DataType)],
    /** Always left ++ right: the row the condition is evaluated on. */
    joinedAttrs: Array[(String, DataType)],
    buildTypes: Array[DataType],
    streamedWidth: Int)

/**
 * Columnar replacement for BroadcastHashJoinExec. The build side is Spark's own broadcast
 * `HashedRelation` (the exchange stays Spark's), read once per task into columns; the streamed side
 * must be columnar.
 */
case class VectorBroadcastHashJoinExec(
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    joinType: JoinType,
    buildSide: BuildSide,
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan)
    extends VectorHashJoinLike {

  override def requiredChildDistribution: Seq[Distribution] = {
    val boundKeys = BindReferences.bindReferences(HashJoin.rewriteKeyExpr(buildKeys), buildPlan.output)
    val mode = HashedRelationBroadcastMode(boundKeys, isNullAware = false)
    buildSide match {
      case BuildLeft => BroadcastDistribution(mode) :: UnspecifiedDistribution :: Nil
      case BuildRight => UnspecifiedDistribution :: BroadcastDistribution(mode) :: Nil
    }
  }

  override def outputPartitioning: Partitioning = joinType match {
    case _: InnerLike => streamedPlan.outputPartitioning
    case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin => left.outputPartitioning
    case RightOuter => right.outputPartitioning
    case _ => UnknownPartitioning(0)
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val spec = joinSpec
    val m = vectorMetrics
    val relation = buildPlan.executeBroadcast[Any]()
    streamedPlan.executeColumnar().mapPartitionsInternal { iter =>
      val build = BuildTable.fromRelation(HashedRelationAccess.rows(relation.value), spec)
      new VectorHashJoinIterator(iter, build, spec, m)
    }
  }

  override protected def withNewChildrenInternal(newLeft: SparkPlan, newRight: SparkPlan): SparkPlan =
    copy(left = newLeft, right = newRight)
}

/**
 * Columnar replacement for BroadcastNestedLoopJoinExec: a join with no equi-keys (an inequality
 * join, a cross join with a filter, a non-equi `EXISTS`). The build side is Spark's identity
 * broadcast of the rows, read once per task into columns; every build row is a candidate of every
 * streamed row, and the streamed rows are processed in chunks sized to a fixed pair budget so the
 * product is never materialised. Inner and cross joins with either build side; left semi, left
 * anti, existence and left outer joins with the right side broadcast; right outer with the left
 * side broadcast. An outer join whose preserved side is the broadcast one (and a full outer join)
 * needs a matched bitmap over the broadcast side and is refused, as Comet refuses it.
 */
case class VectorBroadcastNestedLoopJoinExec(
    joinType: JoinType,
    buildSide: BuildSide,
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan)
    extends VectorHashJoinLike {

  override def leftKeys: Seq[Expression] = Nil
  override def rightKeys: Seq[Expression] = Nil

  override def requiredChildDistribution: Seq[Distribution] = buildSide match {
    case BuildLeft => BroadcastDistribution(IdentityBroadcastMode) :: UnspecifiedDistribution :: Nil
    case BuildRight => UnspecifiedDistribution :: BroadcastDistribution(IdentityBroadcastMode) :: Nil
  }

  override def outputPartitioning: Partitioning = joinType match {
    case _: InnerLike => streamedPlan.outputPartitioning
    case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin => left.outputPartitioning
    case RightOuter => right.outputPartitioning
    case _ => UnknownPartitioning(0)
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val spec = joinSpec
    val m = vectorMetrics
    val relation = buildPlan.executeBroadcast[Array[InternalRow]]()
    streamedPlan.executeColumnar().mapPartitionsInternal { iter =>
      val build = BuildTable.fromRelation(relation.value.iterator, spec)
      new VectorHashJoinIterator(iter, build, spec, m)
    }
  }

  override protected def withNewChildrenInternal(newLeft: SparkPlan, newRight: SparkPlan): SparkPlan =
    copy(left = newLeft, right = newRight)
}

/**
 * Columnar replacement for ShuffledHashJoinExec: both sides arrive partitioned on the keys, the
 * build side of each partition is drained into columns first.
 */
case class VectorShuffledHashJoinExec(
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    joinType: JoinType,
    buildSide: BuildSide,
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan)
    extends VectorHashJoinLike {

  override def requiredChildDistribution: Seq[Distribution] =
    ClusteredDistribution(leftKeys) :: ClusteredDistribution(rightKeys) :: Nil

  override def outputPartitioning: Partitioning = joinType match {
    case _: InnerLike => PartitioningCollection(Seq(left.outputPartitioning, right.outputPartitioning))
    case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin => left.outputPartitioning
    case RightOuter => right.outputPartitioning
    case _ => UnknownPartitioning(0)
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val spec = joinSpec
    val m = vectorMetrics
    buildPlan.executeColumnar().zipPartitions(streamedPlan.executeColumnar()) { (buildIter, streamIter) =>
      val build = BuildTable.fromBatches(buildIter, spec)
      new VectorHashJoinIterator(streamIter, build, spec, m)
    }
  }

  override protected def withNewChildrenInternal(newLeft: SparkPlan, newRight: SparkPlan): SparkPlan =
    copy(left = newLeft, right = newRight)
}

/**
 * The build side of one task: its columns, a key table over the non-null-key rows and, per key
 * (group id), the chain of build rows holding it.
 */
final class BuildTable(val arena: Arena, val columns: Array[VectorBuffers], val numRows: Int, spec: JoinSpec) extends AutoCloseable {
  /** The key table of an equi-join; a nested loop join (no keys) never builds one. */
  lazy val table = new GroupKeyTable(spec.buildKeys.map(_.vecType))
  /** First build row of each key, -1 for none. */
  var head: Array[Int] = new Array[Int](0)
  /** Next build row with the same key, -1 at the end. */
  val next: Array[Int] = new Array[Int](numRows)

  def build(): BuildTable = {
    if (numRows > 0 && spec.buildKeys.nonEmpty) {
      val ctx = new EvalContext(arena, numRows, c => columns(c))
      val keys = spec.buildKeys.map(_.eval(ctx))
      val ids = new Array[Int](numRows)
      val groups = table.assign(keys, numRows, ids, BuildTable.nonNullKeys(ctx, keys))
      head = Array.fill(groups)(-1)
      // Walk backwards so chains list rows in build order, like Spark's relation does.
      var i = numRows - 1
      while (i >= 0) {
        val g = ids(i)
        if (g >= 0) { next(i) = head(g); head(g) = i } else next(i) = -1
        i -= 1
      }
    }
    this
  }

  override def close(): Unit = arena.close()
}

object BuildTable {

  /** Rows whose keys are all non-null (and selected), or null when every row qualifies. */
  def nonNullKeys(ctx: EvalContext, keys: Seq[VectorBuffers]): MemorySegment = {
    var mask: MemorySegment = ctx.selection
    keys.foreach { k =>
      if (k.validity() != null) {
        val m = ctx.bitmap()
        if (mask == null) BitmapKernels.copy(k.validity(), m, ctx.numRows) else BitmapKernels.and(mask, k.validity(), m, ctx.numRows)
        mask = m
      }
    }
    mask
  }

  /** The build side from columnar batches (shuffled hash join). */
  def fromBatches(batches: Iterator[ColumnarBatch], spec: JoinSpec): BuildTable = {
    val arena = Arena.ofShared()
    val builders = spec.buildTypes.map(dt => new ColumnBuilder(arena, TypeMapping.vecTypeOf(dt), 4096))
    var total = 0
    while (batches.hasNext) {
      val batch = batches.next()
      if (batch.numRows() > 0) {
        EvalContexts.withBatch(batch) { ctx =>
          var c = 0
          while (c < builders.length) { builders(c).append(ctx.input(c), ctx.selection, ctx.selectedCount); c += 1 }
          total += ctx.selectedCount
        }
      }
    }
    new BuildTable(arena, builders.map(_.view()), total, spec).build()
  }

  /**
   * The build side from Spark's broadcast relation: every row of every key, read through the
   * `InternalRow` getters into typed arrays, then laid out as columns.
   */
  def fromRelation(rows: Iterator[InternalRow], spec: JoinSpec): BuildTable = {
    val arena = Arena.ofShared()
    val types = spec.buildTypes
    val builders = types.map(dt => new RowColumnBuilder(dt))
    while (rows.hasNext) {
      val row = rows.next()
      var c = 0
      while (c < builders.length) { builders(c).add(row, c); c += 1 }
    }
    val columns = builders.map(_.build(arena))
    new BuildTable(arena, columns, if (columns.isEmpty) 0 else columns.head.length(), spec).build()
  }
}

/** Accumulates one column of `InternalRow`s, then lays it out in Arrow memory. */
private[vector] final class RowColumnBuilder(dt: DataType) {
  private val vecType = TypeMapping.vecTypeOf(dt)
  private var n = 0
  private var nulls = new Array[Boolean](1024)
  private var anyNull = false
  private var ints = if (vecType == VecType.INT32) new Array[Int](1024) else null
  private var longs = if (vecType == VecType.INT64) new Array[Long](1024) else null
  private var doubles = if (vecType == VecType.FLOAT64) new Array[Double](1024) else null
  private var bools = if (vecType == VecType.BOOL) new Array[Boolean](1024) else null
  private var strings = if (vecType == VecType.UTF8) new Array[String](1024) else null

  def add(row: InternalRow, ordinal: Int): Unit = {
    if (n == nulls.length) grow()
    if (row.isNullAt(ordinal)) { nulls(n) = true; anyNull = true }
    else vecType match {
      case VecType.INT32 => ints(n) = row.getInt(ordinal)
      case VecType.INT64 => longs(n) = dt match {
        case d: DecimalType => row.getDecimal(ordinal, d.precision, d.scale).toUnscaledLong
        case _ => row.getLong(ordinal)
      }
      case VecType.FLOAT64 => doubles(n) = row.getDouble(ordinal)
      case VecType.BOOL => bools(n) = row.getBoolean(ordinal)
      case VecType.UTF8 => strings(n) = row.getUTF8String(ordinal).toString
    }
    n += 1
  }

  private def grow(): Unit = {
    val cap = n * 2
    nulls = java.util.Arrays.copyOf(nulls, cap)
    if (ints != null) ints = java.util.Arrays.copyOf(ints, cap)
    if (longs != null) longs = java.util.Arrays.copyOf(longs, cap)
    if (doubles != null) doubles = java.util.Arrays.copyOf(doubles, cap)
    if (bools != null) bools = java.util.Arrays.copyOf(bools, cap)
    if (strings != null) strings = java.util.Arrays.copyOf(strings, cap)
  }

  def build(arena: Arena): VectorBuffers = {
    val nn = if (anyNull) java.util.Arrays.copyOf(nulls, n) else null
    vecType match {
      case VecType.INT32 => ArrowLayout.ofInts(arena, java.util.Arrays.copyOf(ints, n), nn)
      case VecType.INT64 => ArrowLayout.ofLongs(arena, java.util.Arrays.copyOf(longs, n), nn)
      case VecType.FLOAT64 => ArrowLayout.ofDoubles(arena, java.util.Arrays.copyOf(doubles, n), nn)
      case VecType.BOOL => ArrowLayout.ofBooleans(arena, java.util.Arrays.copyOf(bools, n), nn)
      case VecType.UTF8 => ArrowLayout.ofStrings(arena, java.util.Arrays.copyOf(strings, n)) // nulls stay null entries
    }
  }
}

/**
 * Probes the build table with every streamed batch and gathers the matches.
 *
 * Without a condition, or for an inner join, the matches of a batch are gathered straight into
 * output batches (an inner condition is applied to the gathered batch and the failing rows
 * compacted away). Semi, anti and outer joins with a condition go through [[emitConditional]]:
 * the candidate pairs are gathered as the joined row and the condition evaluated over them, and
 * only then does each streamed row become kept/dropped (semi/anti) or its passing pairs / one
 * null-padded row (outer). A full outer join records which build rows were ever paired and emits
 * the others, null-padded on the streamed side, once the input is exhausted.
 */
private[vector] class VectorHashJoinIterator(
    input: Iterator[ColumnarBatch],
    build: BuildTable,
    spec: JoinSpec,
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val OutputBatchSize = 8192
  /** A nested loop join pairs every streamed row with every build row: this bounds the pairs in flight. */
  private val PairBudget = OutputBatchSize * 4

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorHashJoinExec")
  private val pending = new ArrayDeque[ColumnarBatch]()
  private var emitted: ColumnarBatch = _
  private var closed = false
  /** The candidate pairs of the current batch: streamed row and build row (-1 pads an outer join). */
  private var probeIdx = new Array[Int](OutputBatchSize)
  private var buildIdx = new Array[Int](OutputBatchSize)
  /** The pairs an outer join with a condition emits, rewritten from the candidates. */
  private var outProbeIdx = new Array[Int](0)
  private var outBuildIdx = new Array[Int](0)
  /** Per candidate pair, whether the condition held. */
  private var passed = new Array[Boolean](0)
  /** Per streamed row, whether any of its candidates passed the condition. */
  private var rowMatched = new Array[Boolean](0)
  private var idScratch = new Array[Int](0)

  private val numBuildCols = spec.buildTypes.length
  private val streamedWidth = spec.streamedWidth
  /** No keys: a nested loop join, every build row is a candidate of every streamed row. */
  private val nestedLoop = spec.streamedKeys.isEmpty
  private val isSemiOrAnti = spec.joinType == LeftSemi || spec.joinType == LeftAnti
  /** `ExistenceJoin`: the semi join's probe, emitting every streamed row plus a match boolean. */
  private val isExistence = spec.joinType.isInstanceOf[ExistenceJoin]
  private val isFullOuter = spec.joinType == FullOuter
  private val keepUnmatched = spec.joinType == LeftOuter || spec.joinType == RightOuter || isFullOuter
  /** Build rows paired with a streamed row so far; only a full outer join needs to know. */
  private val buildMatched: Array[Boolean] = if (isFullOuter) new Array[Boolean](build.numRows) else null
  private var buildDrained = !isFullOuter

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = {
    while (pending.isEmpty && input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) metrics.timed { probe(batch) }
    }
    if (pending.isEmpty && !buildDrained) {
      buildDrained = true
      metrics.timed { emitUnmatchedBuild() }
    }
    !pending.isEmpty
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more joined batches")
    if (emitted != null) emitted.close()
    emitted = pending.poll()
    metrics.numOutputBatches += 1
    metrics.numOutputRows += emitted.numRows()
    emitted
  }

  private def probe(batch: ColumnarBatch): Unit = {
    metrics.numInputBatches += 1
    EvalContexts.withBatch(batch) { ctx =>
      val n = ctx.numRows
      if (!nestedLoop) {
        val keys = spec.streamedKeys.map(_.eval(ctx))
        val candidates = BuildTable.nonNullKeys(ctx, keys)
        if (idScratch.length < n) idScratch = new Array[Int](n)
        if (build.numRows > 0) build.table.lookup(keys, n, idScratch, candidates)
        else java.util.Arrays.fill(idScratch, 0, n, -1)
      }
      // A hash join's candidates are few per row: one chunk. A nested loop's are the whole build
      // side, so the streamed rows go in chunks that keep the pairs in flight under the budget.
      val rowsPerChunk = if (nestedLoop) math.max(1, PairBudget / math.max(1, build.numRows)) else n
      spec.condition match {
        case Some(cond) if isSemiOrAnti || keepUnmatched || isExistence =>
          if (rowMatched.length < n) rowMatched = new Array[Boolean](n) else java.util.Arrays.fill(rowMatched, 0, n, false)
          var from = 0
          while (from < n) { val until = math.min(n, from + rowsPerChunk); emitConditional(ctx, cond, from, until); from = until }
          if (isExistence) emitExistence(ctx, i => rowMatched(i))
          else if (isSemiOrAnti) emitSemiAnti(ctx, i => rowMatched(i))
        case _ if isExistence => emitExistence(ctx, i => firstCandidate(i) >= 0)
        case _ if isSemiOrAnti => emitSemiAnti(ctx, i => firstCandidate(i) >= 0)
        case _ =>
          var from = 0
          while (from < n) { val until = math.min(n, from + rowsPerChunk); emitMatches(ctx, from, until); from = until }
      }
    }
  }

  private def selected(ctx: EvalContext, i: Int): Boolean = ctx.selection == null || Bitmap.isSet(ctx.selection, i)

  /** First candidate build row of streamed row `i`, -1 for none. */
  private def firstCandidate(i: Int): Int =
    if (nestedLoop) (if (build.numRows > 0) 0 else -1)
    else if (idScratch(i) >= 0) build.head(idScratch(i)) else -1

  /** The candidate after build row `r` for the same streamed row, -1 at the end. */
  private def nextCandidate(r: Int): Int =
    if (nestedLoop) (if (r + 1 < build.numRows) r + 1 else -1) else build.next(r)

  /** Semi/anti joins without a condition keep or drop streamed rows on the key lookup alone. */
  private def emitSemiAnti(ctx: EvalContext, matchedAt: Int => Boolean): Unit = {
    val n = ctx.numRows
    val sel = ctx.bitmap()
    val wantMatch = spec.joinType == LeftSemi
    var i = 0
    while (i < n) {
      if (selected(ctx, i) && matchedAt(i) == wantMatch) Bitmap.set(sel, i)
      i += 1
    }
    compactStreamed(ctx, sel)
  }

  /** Emits the streamed rows selected by `sel` as one batch of the operator's output. */
  private def compactStreamed(ctx: EvalContext, sel: MemorySegment): Unit = {
    val count = Bitmap.popcount(sel, ctx.numRows)
    if (count > 0) {
      val columns = new Array[ColumnVector](spec.outputAttrs.length)
      var c = 0
      while (c < columns.length) {
        val (name, dt) = spec.outputAttrs(c)
        columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), sel, count, allocator)
        c += 1
      }
      pending.add(new ColumnarBatch(columns, count))
    }
  }

  /** Inner and outer joins whose condition, if any, an inner join applies to the gathered batch. */
  private def emitMatches(ctx: EvalContext, from: Int, until: Int): Unit = {
    var count = 0
    var i = from
    while (i < until) {
      if (selected(ctx, i)) {
        var r = firstCandidate(i)
        if (r < 0) {
          if (keepUnmatched) { count = append(count, i, -1); }
        } else {
          while (r >= 0) {
            count = append(count, i, r)
            if (buildMatched != null) buildMatched(r) = true
            r = nextCandidate(r)
          }
        }
      }
      i += 1
    }
    if (count > 0) flush(ctx, probeIdx, buildIdx, count, filter = spec.condition.isDefined)
  }

  /**
   * Semi, anti and outer joins with a condition. Every candidate pair of the batch is gathered as
   * the joined row and the condition evaluated over those; a streamed row then counts as matched
   * when at least one of its pairs passed, exactly as if the condition had been checked per
   * candidate. Rows with no candidate at all have no pair here and stay unmatched.
   */
  private def emitConditional(ctx: EvalContext, cond: VectorExpr, from: Int, until: Int): Unit = {
    var count = 0
    var i = from
    while (i < until) {
      if (selected(ctx, i)) {
        var r = firstCandidate(i)
        while (r >= 0) { count = append(count, i, r); r = nextCandidate(r) }
      }
      i += 1
    }
    if (passed.length < count) passed = new Array[Boolean](probeIdx.length)
    evaluateCondition(ctx, cond, count)
    var p = 0
    while (p < count) { if (passed(p)) rowMatched(probeIdx(p)) = true; p += 1 }
    // Semi, anti and existence joins decide per streamed row once every chunk has run (the caller
    // reads `rowMatched`); an outer join emits this chunk's rows now.
    if (keepUnmatched) emitOuterConditional(ctx, count, from, until)
  }

  /**
   * Existence join: every selected streamed row comes out, borrowed through the same compaction the
   * semi join uses, followed by the `exists` column -- a BOOL bitmap of the rows that matched, never
   * null (a null key or an empty build side is simply `false`).
   */
  private def emitExistence(ctx: EvalContext, matchedAt: Int => Boolean): Unit = {
    val n = ctx.numRows
    val sel = if (ctx.selection != null) ctx.selection else { val all = ctx.bitmap(); Bitmap.fill(all, n, true); all }
    val count = Bitmap.popcount(sel, n)
    if (count > 0) {
      val bits = ctx.bitmap()
      var i = 0
      while (i < n) { if (Bitmap.isSet(sel, i) && matchedAt(i)) Bitmap.set(bits, i); i += 1 }
      val columns = new Array[ColumnVector](spec.outputAttrs.length)
      var c = 0
      while (c < streamedWidth) {
        val (name, dt) = spec.outputAttrs(c)
        columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), sel, count, allocator)
        c += 1
      }
      val (existsName, existsType) = spec.outputAttrs(streamedWidth)
      val exists = SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
      columns(streamedWidth) = ArrowOutput.compact(existsName, existsType, exists, sel, count, allocator)
      pending.add(new ColumnarBatch(columns, count))
    }
  }

  /** Fills `passed(0 until count)`: the condition over the gathered candidate pairs, chunk by chunk. */
  private def evaluateCondition(ctx: EvalContext, cond: VectorExpr, count: Int): Unit = {
    var from = 0
    while (from < count) {
      val to = math.min(count, from + OutputBatchSize)
      val joined = gather(ctx, spec.joinedAttrs, probeIdx, buildIdx, from, to)
      try {
        EvalContexts.withBatch(joined) { jctx =>
          val (sel, _) = VectorExpr.selection(cond.eval(jctx), jctx)
          var j = 0
          while (j < to - from) { passed(from + j) = Bitmap.isSet(sel, j); j += 1 }
        }
      } finally joined.close()
      from = to
    }
  }

  /**
   * Outer joins with a condition: per streamed row, its passing pairs, or one null-padded pair
   * when it matched no candidate (whether it had none or they all failed the condition).
   */
  private def emitOuterConditional(ctx: EvalContext, count: Int, from: Int, until: Int): Unit = {
    val n = until - from
    if (outProbeIdx.length < count + n) {
      outProbeIdx = new Array[Int](count + n)
      outBuildIdx = new Array[Int](count + n)
    }
    var w = 0
    var p = 0
    var i = from
    while (i < until) {
      if (selected(ctx, i)) {
        if (rowMatched(i)) {
          while (p < count && probeIdx(p) == i) {
            if (passed(p)) {
              outProbeIdx(w) = i
              outBuildIdx(w) = buildIdx(p)
              if (buildMatched != null) buildMatched(buildIdx(p)) = true
              w += 1
            }
            p += 1
          }
        } else {
          while (p < count && probeIdx(p) == i) p += 1
          outProbeIdx(w) = i
          outBuildIdx(w) = -1
          w += 1
        }
      }
      i += 1
    }
    if (w > 0) flush(ctx, outProbeIdx, outBuildIdx, w, filter = false)
  }

  /** Full outer join: the build rows no streamed row was ever paired with, streamed side null. */
  private def emitUnmatchedBuild(): Unit = {
    var count = 0
    var r = 0
    while (r < build.numRows) {
      if (!buildMatched(r)) count = append(count, -1, r)
      r += 1
    }
    var from = 0
    while (from < count) {
      val to = math.min(count, from + OutputBatchSize)
      val columns = new Array[ColumnVector](spec.outputAttrs.length)
      var c = 0
      while (c < columns.length) {
        val (name, dt) = spec.outputAttrs(c)
        columns(c) =
          if (isBuildColumn(c)) ArrowOutput.gather(name, dt, build.columns(buildOrdinal(c)), buildIdx, from, to, allocator)
          else ArrowOutput.nulls(name, dt, to - from, allocator)
        c += 1
      }
      pending.add(new ColumnarBatch(columns, to - from))
      from = to
    }
  }

  private def append(count: Int, probe: Int, buildRow: Int): Int = {
    if (count == probeIdx.length) {
      probeIdx = java.util.Arrays.copyOf(probeIdx, count * 2)
      buildIdx = java.util.Arrays.copyOf(buildIdx, count * 2)
    }
    probeIdx(count) = probe
    buildIdx(count) = buildRow
    count + 1
  }

  // Joined rows are left ++ right; which of the two is the build side depends on buildSide.
  private def isBuildColumn(c: Int): Boolean = if (spec.buildIsLeft) c < numBuildCols else c >= streamedWidth
  private def buildOrdinal(c: Int): Int = if (spec.buildIsLeft) c else c - streamedWidth
  private def streamedOrdinal(c: Int): Int = if (spec.buildIsLeft) c - numBuildCols else c

  /** Gathers the pairs `[from, to)` into a batch laid out as `attrs` (left ++ right). */
  private def gather(ctx: EvalContext, attrs: Array[(String, DataType)], probe: Array[Int], bld: Array[Int], from: Int, to: Int): ColumnarBatch = {
    val columns = new Array[ColumnVector](attrs.length)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = attrs(c)
      columns(c) =
        if (isBuildColumn(c)) ArrowOutput.gather(name, dt, build.columns(buildOrdinal(c)), bld, from, to, allocator)
        else ArrowOutput.gather(name, dt, ctx.input(streamedOrdinal(c)), probe, from, to, allocator)
      c += 1
    }
    new ColumnarBatch(columns, to - from)
  }

  /** Gathers `count` pairs into output batches of at most [[OutputBatchSize]] rows. */
  private def flush(ctx: EvalContext, probe: Array[Int], bld: Array[Int], count: Int, filter: Boolean): Unit = {
    var from = 0
    while (from < count) {
      val to = math.min(count, from + OutputBatchSize)
      val joined = gather(ctx, spec.outputAttrs, probe, bld, from, to)
      if (filter) filtered(joined, spec.condition.get).foreach(pending.add)
      else pending.add(joined)
      from = to
    }
  }

  /** Applies the non-equi condition to a joined batch, which is consumed unless every row passes. */
  private def filtered(joined: ColumnarBatch, cond: VectorExpr): Option[ColumnarBatch] = {
    val result: Either[Unit, Option[ColumnarBatch]] = EvalContexts.withBatch(joined) { ctx =>
      val pred = cond.eval(ctx)
      val (sel, count) = VectorExpr.selection(pred, ctx)
      if (count == ctx.numRows) Left(())
      else if (count == 0) Right(None)
      else {
        val columns = new Array[ColumnVector](spec.outputAttrs.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = spec.outputAttrs(c)
          columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), sel, count, allocator)
          c += 1
        }
        Right(Some(new ColumnarBatch(columns, count)))
      }
    }
    result match {
      case Left(_) => Some(joined)
      case Right(out) => joined.close(); out
    }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      if (emitted != null) { emitted.close(); emitted = null }
      while (!pending.isEmpty) pending.poll().close()
      build.close()
      allocator.close()
    }
  }
}

/** Planning-time checks shared by the rule and the operators. */
object VectorJoinPlanner {

  private val keyTypes: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.BOOL, VecType.UTF8)

  def compileKeys(keys: Seq[Expression], input: Seq[Attribute]): Array[VectorExpr] =
    keys.map(k => compileKey(k, input).fold(r => throw new IllegalStateException(s"cannot vectorize join key ${k.sql}: $r"), identity)).toArray

  def compileKey(key: Expression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(key, input).flatMap {
      case _: LiteralExpr => Left("literal join key")
      case k if !keyTypes.contains(k.vecType) => Left(s"join key type ${key.dataType.simpleString} not supported")
      case k => Right(k)
    }

  def compileCondition(cond: Expression, output: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compilePredicate(cond, output)

  /**
   * The build side's estimated size in bytes: a materialised AQE stage's runtime statistics when the
   * build side is one, the logical plan's estimate otherwise. None when unknown -- Spark's own
   * "unknown" is `Long.MaxValue`, and a plan with no logical link has no estimate at all.
   */
  def estimatedBuildSize(plan: SparkPlan): Option[Long] = plan match {
    case r: org.apache.spark.sql.execution.adaptive.AQEShuffleReadExec => estimatedBuildSize(r.child)
    case q: org.apache.spark.sql.execution.adaptive.QueryStageExec => q.computeStats().map(_.sizeInBytes).flatMap(known)
    case other => other.logicalLink.map(_.stats.sizeInBytes).flatMap(known)
  }

  private def known(size: BigInt): Option[Long] = if (size >= 0 && size < BigInt(Long.MaxValue)) Some(size.toLong) else None

  /**
   * The joins hold the build side in memory per task with no limit but the JVM's (#86): a build side
   * estimated above `spark.vector.join.maxBuildSize` stays with Spark. An unknown estimate converts --
   * Spark planned this join after its own size checks, so "unknown" means the statistic is absent,
   * not that the side is large.
   */
  def buildSizeReason(buildPlan: SparkPlan, maxBuildSize: Long): Option[String] =
    estimatedBuildSize(buildPlan).filter(_ > maxBuildSize).map { size =>
      s"build side estimated at $size bytes exceeds ${io.sparkvector.spark.VectorConf.JoinMaxBuildSize}=$maxBuildSize"
    }

  private def supportedType(joinType: JoinType, buildSide: BuildSide): Either[String, Unit] = joinType match {
    case _: InnerLike | FullOuter => Right(())
    case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin if buildSide == BuildRight => Right(())
    case RightOuter if buildSide == BuildLeft => Right(())
    case other => Left(s"join type $other with build side $buildSide not supported")
  }

  private def check(
      leftKeys: Seq[Expression], rightKeys: Seq[Expression], joinType: JoinType, buildSide: BuildSide,
      condition: Option[Expression], left: SparkPlan, right: SparkPlan): Either[String, Unit] = {
    if (leftKeys.isEmpty) Left("join without equi-join keys")
    else supportedType(joinType, buildSide).flatMap { _ =>
      val keyFailures = leftKeys.flatMap(k => compileKey(k, left.output).left.toOption) ++ rightKeys.flatMap(k => compileKey(k, right.output).left.toOption)
      if (keyFailures.nonEmpty) Left(keyFailures.mkString("; "))
      // The condition sees both sides, whatever the join type outputs.
      else condition.map(c => compileCondition(c, left.output ++ right.output).map(_ => ())).getOrElse(Right(()))
    }
  }

  def plan(j: BroadcastHashJoinExec): Either[String, VectorBroadcastHashJoinExec] = {
    if (j.isNullAwareAntiJoin) Left("null-aware anti join not supported")
    // Spark never broadcasts a full outer join (JoinSelection.canBuildBroadcastLeft / Right exclude
    // it): the build side is shared by every task, so the trailing pass over unmatched build rows
    // would emit them once per task. Refused with a reason rather than assumed away.
    else if (j.joinType == FullOuter) Left("full outer join over a broadcast not supported (Spark plans it as a shuffled join)")
    else {
      val v = VectorBroadcastHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right)
      check(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right).map(_ => v)
    }
  }

  def plan(j: BroadcastNestedLoopJoinExec): Either[String, VectorBroadcastNestedLoopJoinExec] = {
    val typeOk: Either[String, Unit] = j.joinType match {
      case _: InnerLike => Right(())
      case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin if j.buildSide == BuildRight => Right(())
      case RightOuter if j.buildSide == BuildLeft => Right(())
      case FullOuter => Left("full outer nested loop join not supported (needs a matched bitmap over the broadcast side)")
      case LeftOuter | RightOuter => Left(s"${j.joinType} nested loop join with the preserved side broadcast not supported (needs a matched bitmap over the broadcast side)")
      case other => Left(s"join type $other with build side ${j.buildSide} not supported")
    }
    typeOk.flatMap { _ =>
      j.condition.map(c => compileCondition(c, j.left.output ++ j.right.output).map(_ => ())).getOrElse(Right(()))
    }.map(_ => VectorBroadcastNestedLoopJoinExec(j.joinType, j.buildSide, j.condition, j.left, j.right))
  }

  def plan(j: ShuffledHashJoinExec): Either[String, VectorShuffledHashJoinExec] = {
    if (j.isSkewJoin) Left("skew join not supported")
    else {
      val v = VectorShuffledHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right)
      check(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right).map(_ => v)
    }
  }
}
