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
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, HashedRelationBroadcastMode, HashJoin, ShuffledHashJoinExec}
import org.apache.spark.sql.execution.vector.HashedRelationAccess
import org.apache.spark.sql.types.{DataType, DecimalType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * What the two hash joins share: an equi-join whose build side is turned into columns plus a
 * [[GroupKeyTable]] keyed on the build keys, probed batch by batch with the streamed keys, the
 * matches gathered into output batches by [[GatherKernels]].
 *
 * Inner, left outer, right outer (build side left), left semi and left anti joins are supported;
 * a non-equi `condition` only for inner joins, where it is evaluated on the joined batch and the
 * rows failing it compacted away. Null keys never match, as in Spark. Double keys are refused:
 * Spark compares them after NaN/zero normalisation, the key table by bits.
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
    case LeftSemi | LeftAnti => left.output
    case _ => left.output ++ right.output // never planned; see VectorJoinPlanner.supportedType
  }

  @transient protected lazy val compiledBuildKeys: Array[VectorExpr] = VectorJoinPlanner.compileKeys(buildKeys, buildPlan.output)
  @transient protected lazy val compiledStreamedKeys: Array[VectorExpr] = VectorJoinPlanner.compileKeys(streamedKeys, streamedPlan.output)
  @transient protected lazy val compiledCondition: Option[VectorExpr] =
    condition.map(c => VectorJoinPlanner.compileCondition(c, output).fold(r => throw new IllegalStateException(s"cannot vectorize join condition: $r"), identity))

  protected def joinSpec: JoinSpec = JoinSpec(
    joinType,
    buildIsLeft = buildSide == BuildLeft,
    compiledBuildKeys,
    compiledStreamedKeys,
    compiledCondition,
    output.map(a => (a.name, a.dataType)).toArray,
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
    outputAttrs: Array[(String, DataType)],
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
    case LeftOuter | LeftSemi | LeftAnti => left.outputPartitioning
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
    case LeftOuter | LeftSemi | LeftAnti => left.outputPartitioning
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
  val table = new GroupKeyTable(spec.buildKeys.map(_.vecType))
  /** First build row of each key, -1 for none. */
  var head: Array[Int] = new Array[Int](0)
  /** Next build row with the same key, -1 at the end. */
  val next: Array[Int] = new Array[Int](numRows)

  def build(): BuildTable = {
    if (numRows > 0) {
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

/** Probes the build table with every streamed batch and gathers the matches. */
private[vector] class VectorHashJoinIterator(
    input: Iterator[ColumnarBatch],
    build: BuildTable,
    spec: JoinSpec,
    metrics: VectorMetrics)
    extends Iterator[ColumnarBatch]
    with AutoCloseable {

  private val OutputBatchSize = 8192

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorHashJoinExec")
  private val pending = new ArrayDeque[ColumnarBatch]()
  private var emitted: ColumnarBatch = _
  private var closed = false
  private var probeIdx = new Array[Int](OutputBatchSize)
  private var buildIdx = new Array[Int](OutputBatchSize)
  private var idScratch = new Array[Int](0)

  private val numBuildCols = spec.buildTypes.length
  private val streamedWidth = spec.streamedWidth
  private val isSemiOrAnti = spec.joinType == LeftSemi || spec.joinType == LeftAnti
  private val keepUnmatched = spec.joinType == LeftOuter || spec.joinType == RightOuter

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  override def hasNext: Boolean = {
    while (pending.isEmpty && input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) metrics.timed { probe(batch) }
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
      val keys = spec.streamedKeys.map(_.eval(ctx))
      val candidates = BuildTable.nonNullKeys(ctx, keys)
      if (idScratch.length < n) idScratch = new Array[Int](n)
      if (build.numRows > 0) build.table.lookup(keys, n, idScratch, candidates)
      else java.util.Arrays.fill(idScratch, 0, n, -1)
      if (isSemiOrAnti) emitSemiAnti(ctx, batch, candidates)
      else emitMatches(ctx, candidates)
    }
  }

  /** Semi/anti joins keep or drop streamed rows: a selection over the batch, compacted once. */
  private def emitSemiAnti(ctx: EvalContext, batch: ColumnarBatch, candidates: MemorySegment): Unit = {
    val n = ctx.numRows
    val sel = ctx.bitmap()
    val wantMatch = spec.joinType == LeftSemi
    var i = 0
    while (i < n) {
      val selected = ctx.selection == null || Bitmap.isSet(ctx.selection, i)
      if (selected) {
        val matched = idScratch(i) >= 0
        if (matched == wantMatch) Bitmap.set(sel, i)
      }
      i += 1
    }
    val count = Bitmap.popcount(sel, n)
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

  private def emitMatches(ctx: EvalContext, candidates: MemorySegment): Unit = {
    val n = ctx.numRows
    var count = 0
    var i = 0
    while (i < n) {
      val selected = ctx.selection == null || Bitmap.isSet(ctx.selection, i)
      if (selected) {
        var r = if (idScratch(i) >= 0) build.head(idScratch(i)) else -1
        if (r < 0) {
          if (keepUnmatched) { count = append(count, i, -1); }
        } else {
          while (r >= 0) { count = append(count, i, r); r = build.next(r) }
        }
      }
      i += 1
    }
    if (count > 0) flush(ctx, count)
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

  /** Gathers the matches into output batches of at most [[OutputBatchSize]] rows. */
  private def flush(ctx: EvalContext, count: Int): Unit = {
    var from = 0
    while (from < count) {
      val to = math.min(count, from + OutputBatchSize)
      val columns = new Array[ColumnVector](spec.outputAttrs.length)
      var c = 0
      while (c < columns.length) {
        val (name, dt) = spec.outputAttrs(c)
        // Output order is left ++ right; which of the two is the build side depends on buildSide.
        val fromBuild = if (spec.buildIsLeft) c < numBuildCols else c >= streamedWidth
        columns(c) =
          if (fromBuild) {
            val bc = if (spec.buildIsLeft) c else c - streamedWidth
            ArrowOutput.gather(name, dt, build.columns(bc), buildIdx, from, to, allocator)
          } else {
            val sc = if (spec.buildIsLeft) c - numBuildCols else c
            ArrowOutput.gather(name, dt, ctx.input(sc), probeIdx, from, to, allocator)
          }
        c += 1
      }
      val joined = new ColumnarBatch(columns, to - from)
      spec.condition match {
        case Some(cond) => filtered(joined, cond).foreach(pending.add)
        case None => pending.add(joined)
      }
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

  private def supportedType(joinType: JoinType, condition: Option[Expression], buildSide: BuildSide): Either[String, Unit] = joinType match {
    case _: InnerLike => Right(())
    case LeftOuter | LeftSemi | LeftAnti if buildSide == BuildRight =>
      if (condition.isDefined) Left(s"$joinType with a non-equi condition not supported") else Right(())
    case RightOuter if buildSide == BuildLeft =>
      if (condition.isDefined) Left("right outer join with a non-equi condition not supported") else Right(())
    case other => Left(s"join type $other with build side $buildSide not supported")
  }

  private def check(
      leftKeys: Seq[Expression], rightKeys: Seq[Expression], joinType: JoinType, buildSide: BuildSide,
      condition: Option[Expression], left: SparkPlan, right: SparkPlan, output: Seq[Attribute]): Either[String, Unit] = {
    if (leftKeys.isEmpty) Left("join without equi-join keys")
    else supportedType(joinType, condition, buildSide).flatMap { _ =>
      val keyFailures = leftKeys.flatMap(k => compileKey(k, left.output).left.toOption) ++ rightKeys.flatMap(k => compileKey(k, right.output).left.toOption)
      if (keyFailures.nonEmpty) Left(keyFailures.mkString("; "))
      else condition.map(c => compileCondition(c, output).map(_ => ())).getOrElse(Right(()))
    }
  }

  def plan(j: BroadcastHashJoinExec): Either[String, VectorBroadcastHashJoinExec] = {
    if (j.isNullAwareAntiJoin) Left("null-aware anti join not supported")
    else {
      val v = VectorBroadcastHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right)
      check(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right, v.output).map(_ => v)
    }
  }

  def plan(j: ShuffledHashJoinExec): Either[String, VectorShuffledHashJoinExec] = {
    if (j.isSkewJoin) Left("skew join not supported")
    else {
      val v = VectorShuffledHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right)
      check(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right, v.output).map(_ => v)
    }
  }
}
