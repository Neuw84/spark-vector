package org.apache.spark.sql.vector

import java.lang.foreign.{Arena, MemorySegment}
import java.util.ArrayDeque

import io.sparkvector.kernels._
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, RemappedColumnVector, VectorAllocators}
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
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, HashedRelationBroadcastMode, HashJoin, ShuffledHashJoinExec, SortMergeJoinExec}
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
 * streamed batch. Null keys never match, as in Spark. Double keys compare by bits, which agrees with
 * Spark because its optimizer normalises NaN and -0.0 on every double join key (compiled as a pass).
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

  /** Spark's plan for `NOT IN (subquery)` over nullable keys; only the broadcast join carries it. */
  def isNullAwareAntiJoin: Boolean = false

  protected def joinSpec: JoinSpec = JoinSpec(
    joinType,
    buildIsLeft = buildSide == BuildLeft,
    compiledBuildKeys,
    compiledStreamedKeys,
    compiledCondition,
    output.map(a => (a.name, a.dataType)).toArray,
    joinedOutput.map(a => (a.name, a.dataType)).toArray,
    buildPlan.output.map(_.dataType).toArray,
    streamedPlan.output.length,
    dropNullStreamedKeys = isNullAwareAntiJoin)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Left keys: ${leftKeys.map(_.sql).mkString(", ")}
       |Right keys: ${rightKeys.map(_.sql).mkString(", ")}
       |Join type: $joinType, build $buildSide
       |Condition: ${condition.map(_.sql).getOrElse("none")}
       |Output: ${output.map(_.name).mkString(", ")}
       |${getTagValue(VectorExecRule.SortMergeWhy).map(w => s"Sort-merge join $w").getOrElse("")}
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
    streamedWidth: Int,
    /**
     * A null-aware anti join with a non-empty, null-free build side: a streamed row whose key is
     * null is dropped like a matched one (`x NOT IN (...)` is unknown, never true, for a null `x`).
     * The two other regimes of the null-aware join never reach the probe (see the broadcast exec).
     */
    dropNullStreamedKeys: Boolean = false)

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
    right: SparkPlan,
    override val isNullAwareAntiJoin: Boolean = false)
    extends VectorHashJoinLike {

  override def requiredChildDistribution: Seq[Distribution] = {
    val boundKeys = BindReferences.bindReferences(HashJoin.rewriteKeyExpr(buildKeys), buildPlan.output)
    // The same mode as Spark's operator, so the exchange already in the plan is the one required: a
    // null-aware relation is one of two singletons when the build side is empty or holds a null key.
    val mode = HashedRelationBroadcastMode(boundKeys, isNullAware = isNullAwareAntiJoin)
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
      // A null-aware anti join whose build side held a null key keeps nothing: `x NOT IN (..., NULL)`
      // is never true. Spark marks that relation with a singleton that has no rows to read.
      if (isNullAwareAntiJoin && HashedRelationAccess.allNullKeys(relation.value)) Iterator.empty
      else {
        // The relation object stays referenced by this closure for the task's lifetime, which keeps
        // the shared table (keyed on it) alive; see BuildTable.sharedFromRelation. An empty
        // null-aware relation is the other singleton: the anti join over an empty table keeps every
        // streamed row, null keys included, which is that regime's answer.
        val build = BuildTable.sharedFromRelation(relation.value.asInstanceOf[AnyRef], spec)
        new VectorHashJoinIterator(iter, build, spec, m)
      }
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
final class BuildTable(val arena: Arena, val columns: Array[VectorBuffers], val numRows: Int, spec: JoinSpec, val shared: Boolean = false) extends AutoCloseable {
  /**
   * The key table of an equi-join; a nested loop join (no keys) never builds one. Built without
   * on-the-fly string dictionaries: build keys are mostly distinct, and the table stays immutable
   * after `build()`, which is what makes concurrent probing of a shared table safe.
   */
  lazy val table = new GroupKeyTable(spec.buildKeys.map(_.vecType), false)
  /** First build row of each key, -1 for none. */
  var head: Array[Int] = new Array[Int](0)
  /** Next build row with the same key, -1 at the end. */
  val next: Array[Int] = new Array[Int](numRows)
  /**
   * The build rows clustered by key: the rows of key `g`, in build order, are
   * `rangeRows(rangeStart(g) until rangeStart(g + 1))`. The residual of an inner join is tested over
   * such a range as one array scan instead of a walk of the chain (q72's 10^10 candidate pairs).
   */
  var rangeStart: Array[Int] = new Array[Int](0)
  var rangeRows: Array[Int] = new Array[Int](0)
  /** Heap mirrors of the fixed-width columns a join condition reads, made on first use (#332). */
  private val mirrors = new Array[io.sparkvector.kernels.HeapMirror](columns.length)
  /** The same lanes in key-clustered order (`rangeRows`), for the range scan; made on first use. */
  private val clustered = new Array[io.sparkvector.kernels.HeapMirror](columns.length)

  /** The heap mirror of column `c`, or null when the column is not one a mirror covers. A shared table may race to make it; the result is the same. */
  def mirror(c: Int): io.sparkvector.kernels.HeapMirror = {
    var m = mirrors(c)
    if (m == null && io.sparkvector.kernels.HeapMirror.mirrors(columns(c))) {
      m = io.sparkvector.kernels.HeapMirror.of(columns(c))
      mirrors(c) = m
    }
    m
  }

  /** The mirror of column `c` in key-clustered order, or null when the column has no mirror. Same race note as `mirror`. */
  def clusteredMirror(c: Int): io.sparkvector.kernels.HeapMirror = {
    var m = clustered(c)
    if (m == null) {
      val plain = mirror(c)
      if (plain != null) {
        m = io.sparkvector.kernels.RangeResidual.cluster(plain, rangeRows)
        clustered(c) = m
      }
    }
    m
  }

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
      // The same rows clustered by key (a counting sort by group id keeps build order within a key).
      rangeStart = new Array[Int](groups + 1)
      i = 0
      while (i < numRows) { val g = ids(i); if (g >= 0) rangeStart(g + 1) += 1; i += 1 }
      var g = 0
      while (g < groups) { rangeStart(g + 1) += rangeStart(g); g += 1 }
      val fill = java.util.Arrays.copyOf(rangeStart, groups)
      rangeRows = new Array[Int](rangeStart(groups))
      i = 0
      while (i < numRows) { val g = ids(i); if (g >= 0) { rangeRows(fill(g)) = i; fill(g) += 1 }; i += 1 }
    }
    this
  }

  /** A shared table (a broadcast relation's, used by every task of the executor) outlives its tasks. */
  override def close(): Unit = if (!shared) arena.close()
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
  def fromRelation(rows: Iterator[InternalRow], spec: JoinSpec, shared: Boolean = false): BuildTable = {
    val arena = Arena.ofShared()
    val types = spec.buildTypes
    val builders = types.map(dt => new RowColumnBuilder(dt))
    var numRows = 0
    while (rows.hasNext) {
      val row = rows.next()
      var c = 0
      while (c < builders.length) { builders(c).add(row, c); c += 1 }
      numRows += 1
    }
    // The build side may have been pruned to no columns at all (a cross join that projects only
    // the streamed side): the row count must not come from a column.
    val columns = builders.map(_.build(arena))
    new BuildTable(arena, columns, numRows, spec, shared).build()
  }

  /**
   * One build table per broadcast relation and join per executor, like Spark's `HashedRelation`
   * itself, which every task of the executor reads and none rebuilds. Tasks probe the same table
   * concurrently (it is read-only once built and its arena is shared); building is serialised on
   * the relation. The relation object is the key, held weakly: Spark keeps it alive while the
   * broadcast block is in memory, and a task holds it for its whole run through the operator's
   * closure. When it is collected the table's native memory goes with it, through a `Cleaner`.
   */
  private val sharedTables = new java.util.WeakHashMap[AnyRef, java.util.HashMap[String, BuildTable]]()
  private val cleaner = java.lang.ref.Cleaner.create()

  def sharedFromRelation(relation: AnyRef, spec: JoinSpec): BuildTable = {
    val perRelation = sharedTables.synchronized {
      var m = sharedTables.get(relation)
      if (m == null) { m = new java.util.HashMap[String, BuildTable](); sharedTables.put(relation, m) }
      m
    }
    // The spec of one join over the relation; another join on other keys gets its own table.
    val key = spec.buildKeys.mkString("|") + "#" + spec.buildTypes.mkString("|")
    perRelation.synchronized {
      var t = perRelation.get(key)
      if (t == null) {
        t = fromRelation(HashedRelationAccess.rows(relation), spec, shared = true)
        val arena = t.arena
        cleaner.register(relation, () => arena.close())
        perRelation.put(key, t)
      }
      t
    }
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
  private var bytes = if (vecType == VecType.UTF8) new Array[Byte](8192) else null
  private var wides = if (vecType == VecType.DECIMAL128) new Array[java.math.BigInteger](1024) else null
  private var offsets = if (vecType == VecType.UTF8) new Array[Int](1025) else null
  private var used = 0

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
      case VecType.DECIMAL128 =>
        // A wide decimal from the broadcast rows: the unscaled value, laid out as two limbs below (#259).
        val d = dt.asInstanceOf[DecimalType]
        wides(n) = row.getDecimal(ordinal, d.precision, d.scale).toJavaBigDecimal.unscaledValue()
      case VecType.BOOL => bools(n) = row.getBoolean(ordinal)
      case VecType.UTF8 =>
        val str = row.getUTF8String(ordinal)
        val len = str.numBytes()
        if (used + len > bytes.length) bytes = java.util.Arrays.copyOf(bytes, math.max(bytes.length * 2, used + len))
        str.writeToMemory(bytes, org.apache.spark.unsafe.Platform.BYTE_ARRAY_OFFSET + used)
        used += len
    }
    if (offsets != null) offsets(n + 1) = used
    n += 1
  }

  private def grow(): Unit = {
    val cap = n * 2
    nulls = java.util.Arrays.copyOf(nulls, cap)
    if (ints != null) ints = java.util.Arrays.copyOf(ints, cap)
    if (longs != null) longs = java.util.Arrays.copyOf(longs, cap)
    if (doubles != null) doubles = java.util.Arrays.copyOf(doubles, cap)
    if (wides != null) wides = java.util.Arrays.copyOf(wides, cap)
    if (bools != null) bools = java.util.Arrays.copyOf(bools, cap)
    if (offsets != null) offsets = java.util.Arrays.copyOf(offsets, cap + 1)
  }

  def build(arena: Arena): VectorBuffers = {
    val nn = if (anyNull) java.util.Arrays.copyOf(nulls, n) else null
    vecType match {
      case VecType.INT32 => ArrowLayout.ofInts(arena, java.util.Arrays.copyOf(ints, n), nn)
      case VecType.INT64 => ArrowLayout.ofLongs(arena, java.util.Arrays.copyOf(longs, n), nn)
      case VecType.FLOAT64 => ArrowLayout.ofDoubles(arena, java.util.Arrays.copyOf(doubles, n), nn)
      case VecType.DECIMAL128 => ArrowLayout.ofDecimal128(arena, java.util.Arrays.copyOf(wides, n), nn)
      case VecType.BOOL => ArrowLayout.ofBooleans(arena, java.util.Arrays.copyOf(bools, n), nn)
      case VecType.UTF8 => ArrowLayout.ofUtf8(arena, bytes, offsets, n, nn)
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
  /** The pairs of one chunk that passed an inner join's condition (#332). */
  private var survProbeIdx = new Array[Int](0)
  private var survBuildIdx = new Array[Int](0)
  private val gatherScratch = new io.sparkvector.kernels.HeapMirror.GatherScratch
  /** Heap mirrors of the streamed columns the condition reads, for the current batch (#332). */
  private val streamedMirrors = new Array[io.sparkvector.kernels.HeapMirror](spec.streamedWidth)
  private val streamedMirrorTried = new Array[Boolean](spec.streamedWidth)

  private def streamedMirror(ctx: EvalContext, ordinal: Int): io.sparkvector.kernels.HeapMirror = {
    if (!streamedMirrorTried(ordinal)) {
      streamedMirrorTried(ordinal) = true
      val in = ctx.input(ordinal)
      if (io.sparkvector.kernels.HeapMirror.mirrors(in)) streamedMirrors(ordinal) = io.sparkvector.kernels.HeapMirror.of(in)
    }
    streamedMirrors(ordinal)
  }

  /** The joined ordinals the condition reads; the others are never gathered for it. */
  private val conditionRefs: Array[Boolean] = spec.condition.map { cond =>
    val refs = new Array[Boolean](spec.joinedAttrs.length)
    def walk(e: VectorExpr): Unit = e match {
      case io.sparkvector.spark.expr.ColumnRef(o, _) => refs(o) = true
      case other => other.children.foreach(walk)
    }
    walk(cond)
    refs
  }.orNull
  /** The pairs an outer join with a condition emits, rewritten from the candidates. */
  private var outProbeIdx = new Array[Int](0)
  private var outBuildIdx = new Array[Int](0)
  /** Per candidate pair, whether the condition held. */
  private var passed = new Array[Boolean](0)
  /** Per streamed row, whether any of its candidates passed the condition. */
  private var rowMatched = new Array[Boolean](0)
  private var idScratch = new Array[Int](0)
  private var hashScratch = new Array[Int](0)
  /** The streamed rows of the current batch whose keys are all non-null (null when every key is). */
  private var nonNullKeys: MemorySegment = _

  private val numBuildCols = spec.buildTypes.length
  private val streamedWidth = spec.streamedWidth
  /** No keys: a nested loop join, every build row is a candidate of every streamed row. */
  private val nestedLoop = spec.streamedKeys.isEmpty
  private val isSemiOrAnti = spec.joinType == LeftSemi || spec.joinType == LeftAnti
  /**
   * The null-aware anti join's middle regime: the build side has rows and none has a null key (a
   * null key there is a singleton relation that never reaches this iterator), so a streamed row with
   * a null key is dropped like a matched one. Over an empty build side every streamed row is kept.
   */
  private val dropNullKeys = spec.dropNullStreamedKeys && build.numRows > 0
  /** `ExistenceJoin`: the semi join's probe, emitting every streamed row plus a match boolean. */
  private val isExistence = spec.joinType.isInstanceOf[ExistenceJoin]
  private val isFullOuter = spec.joinType == FullOuter
  /** Whether unmatched streamed rows come out null-padded: the outer join preserves the streamed side. */
  private val preservesStreamed = isFullOuter || (spec.joinType == LeftOuter && !spec.buildIsLeft) || (spec.joinType == RightOuter && spec.buildIsLeft)
  /** Whether unmatched build rows come out at the end: the outer join preserves the build side (#273). */
  private val preservesBuild = isFullOuter || (spec.joinType == LeftOuter && spec.buildIsLeft) || (spec.joinType == RightOuter && !spec.buildIsLeft)
  /** An outer join of either kind runs the conditional outer path (pairs per streamed row, then the padding rules). */
  private val keepUnmatched = preservesStreamed || preservesBuild
  /** Build rows paired with a streamed row so far; a join preserving the build side needs to know. */
  private val buildMatched: Array[Boolean] = if (preservesBuild) new Array[Boolean](build.numRows) else null
  private var buildDrained = !preservesBuild

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
    java.util.Arrays.fill(streamedMirrorTried, false)
    java.util.Arrays.fill(streamedMirrors.asInstanceOf[Array[AnyRef]], null)
    EvalContexts.withBatch(batch) { ctx =>
      val n = ctx.numRows
      if (!nestedLoop) {
        val keys = spec.streamedKeys.map(_.eval(ctx))
        val candidates = BuildTable.nonNullKeys(ctx, keys)
        nonNullKeys = candidates
        if (idScratch.length < n) { idScratch = new Array[Int](n); hashScratch = new Array[Int](n) }
        if (build.numRows > 0) build.table.lookup(keys, n, idScratch, candidates, hashScratch)
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
        case _ if isSemiOrAnti && dropNullKeys => emitSemiAnti(ctx, i => firstCandidate(i) >= 0 || nullKeyAt(i))
        case _ if isSemiOrAnti => emitSemiAnti(ctx, i => firstCandidate(i) >= 0)
        case _ =>
          var from = 0
          while (from < n) { val until = math.min(n, from + rowsPerChunk); emitMatches(ctx, from, until); from = until }
      }
    }
  }

  private def selected(ctx: EvalContext, i: Int): Boolean = ctx.selection == null || Bitmap.isSet(ctx.selection, i)

  /** Whether a key of streamed row `i` is null (`nonNullKeys` folds the selection in; `null` means none is). */
  private def nullKeyAt(i: Int): Boolean = nonNullKeys != null && !Bitmap.isSet(nonNullKeys, i)

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
      var foreignRows: Array[Int] = null // the selection as row ids, for columns with no lane (passed through, #273)
      var c = 0
      while (c < columns.length) {
        val (name, dt) = spec.outputAttrs(c)
        columns(c) =
          if (TypeMapping.hasLane(dt)) ArrowOutput.compact(name, dt, ctx.input(c), sel, count, allocator)
          else {
            if (foreignRows == null) foreignRows = RemappedColumnVector.rowsOf(sel, ctx.numRows, count)
            RemappedColumnVector.of(ctx.column(c), foreignRows)
          }
        c += 1
      }
      pending.add(new ColumnarBatch(columns, count))
    }
  }

  /** Inner and outer joins whose condition, if any, an inner join applies to the gathered batch. */
  private def emitMatches(ctx: EvalContext, from: Int, until: Int): Unit = {
    // A simple residual over integer lanes is tested per candidate pair on the heap mirrors, so a
    // failing pair is never appended, gathered or compacted (#332: q72's 10^9 pairs, a few percent kept).
    val fused = fusedPredicate(ctx)
    // When exactly one side of it is a build lane, the test runs as one array scan over the key's
    // clustered build rows instead of a walk of the chain with a call per pair (q72).
    val ranged = if (fused != null && !nestedLoop && fused.rangeable) fused else null
    var count = 0
    var i = from
    while (i < until) {
      if (selected(ctx, i)) {
        if (ranged != null) {
          val g = idScratch(i)
          if (g >= 0) count = scanRange(ranged, i, build.rangeStart(g), build.rangeStart(g + 1), count)
        } else {
          var r = firstCandidate(i)
          if (r < 0) {
            if (preservesStreamed) { count = append(count, i, -1); }
          } else {
            while (r >= 0) {
              if (fused == null || fused.test(i, r)) {
                count = append(count, i, r)
                if (buildMatched != null) buildMatched(r) = true
              }
              r = nextCandidate(r)
            }
          }
        }
      }
      i += 1
    }
    if (count > 0) flush(ctx, probeIdx, buildIdx, count, filter = spec.condition.isDefined && fused == null)
  }

  /** Positions of the clustered build rows that passed the residual for one streamed row (scratch). */
  private var rangeHits = new Array[Int](0)

  /**
   * The residual over the clustered build rows `[s, e)` of streamed row `i`: the streamed operand is
   * one value for the row (null: no pairs), the build operand a clustered lane, and the survivors are
   * appended as (streamed row, build row) pairs in build order, as the chain walk would have.
   */
  private def scanRange(p: PairPredicate, i: Int, s: Int, e: Int, count0: Int): Int = {
    val n = e - s
    if (n <= 0 || !p.streamedValid(i)) return count0
    if (rangeHits.length < n) rangeHits = new Array[Int](math.max(n, rangeHits.length * 2))
    val hits = p.scan(i, s, e, rangeHits)
    var count = count0
    if (hits > 0) {
      if (count + hits > probeIdx.length) {
        val cap = math.max(count + hits, probeIdx.length * 2)
        probeIdx = java.util.Arrays.copyOf(probeIdx, cap)
        buildIdx = java.util.Arrays.copyOf(buildIdx, cap)
      }
      val rows = build.rangeRows
      var h = 0
      while (h < hits) {
        val r = rows(rangeHits(h))
        probeIdx(count) = i
        buildIdx(count) = r
        if (buildMatched != null) buildMatched(r) = true
        count += 1
        h += 1
      }
    }
    count
  }

  /**
   * The condition as a per-pair test over heap mirrors, when it is `lane OP lane` -- either side
   * optionally `± literal` (#332: q72's `d3.d_date > d1.d_date + 5`, a date lane plus an integer
   * literal) -- on INT32 or INT64 lanes both mirrored for this batch; null otherwise (the
   * gather-then-compact path applies). A null on either side fails the pair, as Spark's condition
   * does; the offset is added in the lane's width, wrapping as Spark's non-ANSI Add and DateAdd do.
   */
  private def fusedPredicate(ctx: EvalContext): PairPredicate = spec.condition match {
    case Some(io.sparkvector.spark.expr.CompareExpr(op, LaneWithOffset(a, ao), LaneWithOffset(b, bo)))
        if isSemiOrAnti == false && !keepUnmatched && !isExistence =>
      val ma = mirrorOf(ctx, a); val mb = mirrorOf(ctx, b)
      if (ma == null || mb == null || ma.`type` != mb.`type` || (ma.`type` != io.sparkvector.kernels.VecType.INT32 && ma.`type` != io.sparkvector.kernels.VecType.INT64)) null
      else new PairPredicate(op, ma, isBuildColumn(a), a, ao, mb, isBuildColumn(b), b, bo)
    case _ => null
  }

  /** A joined lane, possibly plus or minus an integer literal (date_add / date_sub / +, - on integers). */
  private object LaneWithOffset {
    import io.sparkvector.spark.expr.{ArithExpr, ColumnRef, LiteralExpr}
    def unapply(e: io.sparkvector.spark.expr.VectorExpr): Option[(Int, Long)] = e match {
      case ColumnRef(c, _) => Some((c, 0L))
      // Only the plain wrapping form: an ANSI / try_add arithmetic checks overflow, which the fused
      // test does not reproduce, so it keeps the gather-then-compact path.
      case ArithExpr(op, ColumnRef(c, _), LiteralExpr(v, _), _, false, _, false) if v != null && integral(v) && (op == io.sparkvector.kernels.ArithOp.ADD || op == io.sparkvector.kernels.ArithOp.SUB) =>
        Some((c, if (op == io.sparkvector.kernels.ArithOp.ADD) toLong(v) else -toLong(v)))
      case ArithExpr(io.sparkvector.kernels.ArithOp.ADD, LiteralExpr(v, _), ColumnRef(c, _), _, false, _, false) if v != null && integral(v) =>
        Some((c, toLong(v)))
      case _ => None
    }
    private def integral(v: Any): Boolean = v.isInstanceOf[Int] || v.isInstanceOf[Long] || v.isInstanceOf[Short] || v.isInstanceOf[Byte]
    private def toLong(v: Any): Long = v.asInstanceOf[Number].longValue()
  }

  /** The heap mirror behind joined ordinal `c` for this batch, or null. */
  private def mirrorOf(ctx: EvalContext, c: Int): io.sparkvector.kernels.HeapMirror =
    if (isBuildColumn(c)) build.mirror(buildOrdinal(c))
    else if (TypeMapping.hasLane(spec.joinedAttrs(c)._2)) streamedMirror(ctx, streamedOrdinal(c))
    else null

  /** `left [+ lo] OP right [+ ro]` per pair; each side reads the build row or the streamed row of the pair. */
  private final class PairPredicate(
      op: io.sparkvector.kernels.CompareOp, left: io.sparkvector.kernels.HeapMirror, leftIsBuild: Boolean, leftOrdinal: Int, leftOffset: Long,
      right: io.sparkvector.kernels.HeapMirror, rightIsBuild: Boolean, rightOrdinal: Int, rightOffset: Long) {
    private val ints = left.`type` == io.sparkvector.kernels.VecType.INT32
    def test(streamed: Int, buildRow: Int): Boolean = {
      val li = if (leftIsBuild) buildRow else streamed
      val ri = if (rightIsBuild) buildRow else streamed
      if (!left.isValid(li) || !right.isValid(ri)) false
      else {
        // The offset is added in the lane's own width, wrapping as Spark's non-ANSI Add / DateAdd do.
        val cmp =
          if (ints) Integer.compare(left.ints(li) + leftOffset.toInt, right.ints(ri) + rightOffset.toInt)
          else java.lang.Long.compare(left.longs(li) + leftOffset, right.longs(ri) + rightOffset)
        op match {
          case io.sparkvector.kernels.CompareOp.EQ => cmp == 0
          case io.sparkvector.kernels.CompareOp.NE => cmp != 0
          case io.sparkvector.kernels.CompareOp.LT => cmp < 0
          case io.sparkvector.kernels.CompareOp.LE => cmp <= 0
          case io.sparkvector.kernels.CompareOp.GT => cmp > 0
          case io.sparkvector.kernels.CompareOp.GE => cmp >= 0
        }
      }
    }

    /** One build lane against one streamed value: the shape the range scan handles. */
    val rangeable: Boolean = leftIsBuild != rightIsBuild
    /** The streamed operand's mirror and offset, and the build lane's ordinal and offset. */
    private val streamedMirror = if (leftIsBuild) right else left
    private val streamedOffset = if (leftIsBuild) rightOffset else leftOffset
    private val buildColumn = buildOrdinal(if (leftIsBuild) leftOrdinal else rightOrdinal)
    private val laneOffset = if (leftIsBuild) leftOffset else rightOffset
    /** `build OP streamed` when the build lane is the left operand; the mirrored operator otherwise. */
    private val mask = io.sparkvector.kernels.RangeResidual.mask(if (leftIsBuild) op else op match {
      case io.sparkvector.kernels.CompareOp.LT => io.sparkvector.kernels.CompareOp.GT
      case io.sparkvector.kernels.CompareOp.LE => io.sparkvector.kernels.CompareOp.GE
      case io.sparkvector.kernels.CompareOp.GT => io.sparkvector.kernels.CompareOp.LT
      case io.sparkvector.kernels.CompareOp.GE => io.sparkvector.kernels.CompareOp.LE
      case other => other
    })
    private var lane: io.sparkvector.kernels.HeapMirror = null

    def streamedValid(streamed: Int): Boolean = streamedMirror.isValid(streamed)

    /** The positions in `[s, e)` of the clustered build lane passing the residual against streamed row `i`, into `hits`; their count. */
    def scan(streamed: Int, s: Int, e: Int, hits: Array[Int]): Int = {
      if (lane == null) lane = build.clusteredMirror(buildColumn)
      if (ints)
        io.sparkvector.kernels.RangeResidual.scanInts(lane.ints, lane.validity, s, e, laneOffset.toInt, streamedMirror.ints(streamed) + streamedOffset.toInt, mask, hits, 0)
      else
        io.sparkvector.kernels.RangeResidual.scanLongs(lane.longs, lane.validity, s, e, laneOffset, streamedMirror.longs(streamed) + streamedOffset, mask, hits, 0)
    }
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
        columns(c) = compactOrPass(ctx, c, name, dt, sel, count)
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
      val joined = gather(ctx, spec.joinedAttrs, probeIdx, buildIdx, from, to, conditionRefs)
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
          if (preservesStreamed) {
            outProbeIdx(w) = i
            outBuildIdx(w) = -1
            w += 1
          }
        }
      }
      i += 1
    }
    if (w > 0) flush(ctx, outProbeIdx, outBuildIdx, w, filter = false)
  }

  /** An outer join preserving the build side (full outer, or the build side is the preserved one): the build rows no streamed row was ever paired with, streamed side null. */
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
          else if (TypeMapping.hasLane(dt)) ArrowOutput.nulls(name, dt, to - from, allocator)
          else nullsWithoutLane(dt, to - from)
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

  /** Compacts an output column by `sel`, or passes a column without a lane through as a remapped view (#273). */
  private def compactOrPass(ctx: EvalContext, c: Int, name: String, dt: DataType, sel: MemorySegment, count: Int): ColumnVector =
    if (TypeMapping.hasLane(dt)) ArrowOutput.compact(name, dt, ctx.input(c), sel, count, allocator)
    else RemappedColumnVector.of(ctx.column(c), RemappedColumnVector.rowsOf(sel, ctx.numRows, count))

  /** A column of nulls for a type without a lane (the streamed side of an unmatched build row): Spark's constant vector, set null. */
  private def nullsWithoutLane(dt: DataType, length: Int): ColumnVector = {
    val v = new org.apache.spark.sql.execution.vectorized.ConstantColumnVector(length, dt)
    v.setNull()
    v
  }

  // Joined rows are left ++ right; which of the two is the build side depends on buildSide.
  private def isBuildColumn(c: Int): Boolean = if (spec.buildIsLeft) c < numBuildCols else c >= streamedWidth
  private def buildOrdinal(c: Int): Int = if (spec.buildIsLeft) c else c - streamedWidth
  private def streamedOrdinal(c: Int): Int = if (spec.buildIsLeft) c - numBuildCols else c

  /**
   * Gathers the pairs `[from, to)` into a batch laid out as `attrs` (left ++ right). With `only`,
   * the columns it does not mark are [[PlaceholderColumn]]: the batch is for the condition alone.
   */
  private def gather(ctx: EvalContext, attrs: Array[(String, DataType)], probe: Array[Int], bld: Array[Int], from: Int, to: Int, only: Array[Boolean] = null): ColumnarBatch = {
    val columns = new Array[ColumnVector](attrs.length)
    var probeRows: Array[Int] = null // the probe ids of this batch, for streamed columns with no lane (passed through, #273)
    var c = 0
    while (c < columns.length) {
      val (name, dt) = attrs(c)
      columns(c) =
        if (only != null && !only(c)) PlaceholderColumn
        else if (isBuildColumn(c) && build.mirror(buildOrdinal(c)) != null)
          ArrowOutput.gatherHeap(name, dt, build.mirror(buildOrdinal(c)), bld, from, to, allocator, gatherScratch)
        else if (!isBuildColumn(c) && TypeMapping.hasLane(dt) && streamedMirror(ctx, streamedOrdinal(c)) != null)
          ArrowOutput.gatherHeap(name, dt, streamedMirror(ctx, streamedOrdinal(c)), probe, from, to, allocator, gatherScratch)
        else if (isBuildColumn(c)) ArrowOutput.gather(name, dt, build.columns(buildOrdinal(c)), bld, from, to, allocator)
        else if (TypeMapping.hasLane(dt)) ArrowOutput.gather(name, dt, ctx.input(streamedOrdinal(c)), probe, from, to, allocator)
        else {
          if (probeRows == null) probeRows = java.util.Arrays.copyOfRange(probe, from, to)
          RemappedColumnVector.of(ctx.column(streamedOrdinal(c)), probeRows)
        }
      c += 1
    }
    new ColumnarBatch(columns, to - from)
  }

  /**
   * Gathers `count` pairs into output batches of at most [[OutputBatchSize]] rows. With `filter`
   * (an inner join with a condition) the condition is evaluated first, over a gather of only the
   * columns it reads, and the output columns are gathered for the surviving pairs alone: a
   * many-to-many key with a selective residual (TPC-DS q72, 10^9 candidate pairs of which a few
   * survive) otherwise materialises every pair's full row before dropping it (#332).
   */
  private def flush(ctx: EvalContext, probe: Array[Int], bld: Array[Int], count: Int, filter: Boolean): Unit = {
    var from = 0
    while (from < count) {
      val to = math.min(count, from + OutputBatchSize)
      if (filter) {
        val k = survivors(ctx, spec.condition.get, probe, bld, from, to)
        if (k > 0) pending.add(gather(ctx, spec.outputAttrs, survProbeIdx, survBuildIdx, 0, k))
      } else pending.add(gather(ctx, spec.outputAttrs, probe, bld, from, to))
      from = to
    }
  }

  /** The pairs of `[from, to)` that pass the condition, compacted into `survProbeIdx` / `survBuildIdx`; their count. */
  private def survivors(ctx: EvalContext, cond: VectorExpr, probe: Array[Int], bld: Array[Int], from: Int, to: Int): Int = {
    val n = to - from
    if (survProbeIdx.length < n) { survProbeIdx = new Array[Int](n); survBuildIdx = new Array[Int](n) }
    val narrow = gather(ctx, spec.joinedAttrs, probe, bld, from, to, conditionRefs)
    try {
      EvalContexts.withBatch(narrow) { jctx =>
        val (sel, passing) = VectorExpr.selection(cond.eval(jctx), jctx)
        var k = 0
        if (passing == n) {
          System.arraycopy(probe, from, survProbeIdx, 0, n); System.arraycopy(bld, from, survBuildIdx, 0, n); k = n
        } else if (passing > 0) {
          val words = Bitmap.wordsFor(n)
          var w = 0
          while (w < words) {
            var word = Bitmap.wordAt(sel, w, n)
            val base = w << 6
            while (word != 0L) {
              val j = base + java.lang.Long.numberOfTrailingZeros(word)
              survProbeIdx(k) = probe(from + j); survBuildIdx(k) = bld(from + j); k += 1
              word &= word - 1
            }
            w += 1
          }
        }
        k
      }
    } finally narrow.close()
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

  /**
   * Doubles included: Spark's optimizer wraps double join keys in NormalizeNaNAndZero (all NaNs and
   * both zeros canonical), which is compiled to a real pass, so the tables' bit comparison agrees
   * with Spark's equality.
   */
  /** Key lanes the table hashes and compares; a wide decimal key is two limbs (#259). */
  private val keyTypes: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.BOOL, VecType.UTF8, VecType.FLOAT64, VecType.DECIMAL128)

  def compileKeys(keys: Seq[Expression], input: Seq[Attribute]): Array[VectorExpr] =
    keys.map(k => compileKey(k, input).fold(r => throw new IllegalStateException(s"cannot vectorize join key ${k.sql}: $r"), identity)).toArray

  def compileKey(key: Expression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compileLaneColumn(key, input).flatMap {
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
    // A sort keeps its input's bytes: read the stage below it. The sort's own logical estimate is the
    // estimate of whatever it sorts -- for a sort over a join, the join's product estimate (#329:
    // TPC-DS q1/q30/q81 read 10^16 bytes at SF1 and were left to Spark).
    case s: org.apache.spark.sql.execution.SortExec => estimatedBuildSize(s.child)
    case other =>
      val logical = other.logicalLink.map(_.stats.sizeInBytes).flatMap(known)
      // A materialised stage below, through unary nodes, has run: its bytes bound the estimate above
      // it (a filter or a project does not multiply rows the way a join's estimate does).
      (logical, runtimeSizeBelow(other)) match {
        case (Some(l), Some(r)) => Some(math.min(l, r))
        case (l, r) => l.orElse(r)
      }
  }

  /** The runtime bytes of a query stage reached from `plan` through unary operators only, if it has run. */
  private def runtimeSizeBelow(plan: SparkPlan): Option[Long] = plan match {
    case q: org.apache.spark.sql.execution.adaptive.QueryStageExec => q.computeStats().map(_.sizeInBytes).flatMap(known)
    case r: org.apache.spark.sql.execution.adaptive.AQEShuffleReadExec => runtimeSizeBelow(r.child)
    case u if u.children.size == 1 => runtimeSizeBelow(u.children.head)
    case _ => None
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

  /**
   * Join type against build side. An outer join may preserve the build side (`RightOuter` built from the
   * right, `LeftOuter` from the left) in the shuffled join -- its per-task matched bitmap emits the
   * unmatched build rows once, as the full outer join does (#273) -- but not over a broadcast, whose
   * relation is shared by every task.
   */
  private def supportedType(joinType: JoinType, buildSide: BuildSide, preservedBuild: Boolean): Either[String, Unit] = joinType match {
    case _: InnerLike | FullOuter => Right(())
    case LeftOuter | LeftSemi | LeftAnti | _: ExistenceJoin if buildSide == BuildRight => Right(())
    case LeftOuter | RightOuter if preservedBuild => Right(())
    case RightOuter if buildSide == BuildLeft => Right(())
    case other => Left(s"join type $other with build side $buildSide not supported")
  }

  private def check(
      leftKeys: Seq[Expression], rightKeys: Seq[Expression], joinType: JoinType, buildSide: BuildSide,
      condition: Option[Expression], left: SparkPlan, right: SparkPlan, preservedBuild: Boolean): Either[String, Unit] = {
    if (leftKeys.isEmpty) Left("join without equi-join keys")
    else supportedType(joinType, buildSide, preservedBuild).flatMap { _ =>
      val keyFailures = leftKeys.flatMap(k => compileKey(k, left.output).left.toOption) ++ rightKeys.flatMap(k => compileKey(k, right.output).left.toOption)
      if (keyFailures.nonEmpty) Left(keyFailures.mkString("; "))
      // The condition sees both sides, whatever the join type outputs.
      else condition.map(c => compileCondition(c, left.output ++ right.output).map(_ => ())).getOrElse(Right(()))
    }
  }

  def plan(j: BroadcastHashJoinExec): Either[String, VectorBroadcastHashJoinExec] = {
    // Spark plans the null-aware anti join (`NOT IN (subquery)` over nullable keys) as a single-key
    // left anti join with the right side broadcast and no condition (ExtractSingleColumnNullAwareAntiJoin);
    // anything else under the flag is not a shape whose semantics we know.
    if (j.isNullAwareAntiJoin && (j.joinType != LeftAnti || j.buildSide != BuildRight || j.leftKeys.length != 1 || j.condition.isDefined))
      Left("null-aware anti join that is not a single-key, condition-free left anti join with the right side broadcast")
    // Spark never broadcasts a full outer join (JoinSelection.canBuildBroadcastLeft / Right exclude
    // it): the build side is shared by every task, so the trailing pass over unmatched build rows
    // would emit them once per task. Refused with a reason rather than assumed away.
    else if (j.joinType == FullOuter) Left("full outer join over a broadcast not supported (Spark plans it as a shuffled join)")
    else {
      val v = VectorBroadcastHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right, j.isNullAwareAntiJoin)
      check(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right, preservedBuild = false).map(_ => v)
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
      check(j.leftKeys, j.rightKeys, j.joinType, j.buildSide, j.condition, j.left, j.right, preservedBuild = true).map(_ => v)
    }
  }

  /**
   * A sort-merge join as our merge join (#286): Spark's contract kept (clustered on the keys, both
   * sides sorted by them), every join type Spark's operator supports, skew joins included (the
   * partitions are already split). Keys and the condition must compile; every output column must
   * have a lane (the pair gather has no pass-through for a struct yet).
   */
  def planMergeJoin(j: SortMergeJoinExec): Either[String, VectorSortMergeJoinExec] = {
    if (j.leftKeys.isEmpty) Left("join without equi-join keys")
    else {
      val typeOk: Either[String, Unit] = j.joinType match {
        case _: InnerLike | LeftOuter | RightOuter | FullOuter | LeftSemi | LeftAnti | _: ExistenceJoin => Right(())
        case other => Left(s"join type $other not supported")
      }
      typeOk.flatMap { _ =>
        val keyFailures = j.leftKeys.flatMap(k => compileKey(k, j.left.output).left.toOption) ++ j.rightKeys.flatMap(k => compileKey(k, j.right.output).left.toOption)
        val laneless = (j.left.output ++ j.right.output).filterNot(a => TypeMapping.hasLane(a.dataType))
        if (keyFailures.nonEmpty) Left(keyFailures.mkString("; "))
        else if (laneless.nonEmpty) Left(s"unsupported column type ${laneless.head.dataType.simpleString} for ${laneless.head.name}")
        else j.condition.map(c => compileCondition(c, j.left.output ++ j.right.output).map(_ => ())).getOrElse(Right(()))
      }.map(_ => VectorSortMergeJoinExec(j.leftKeys, j.rightKeys, j.joinType, j.condition, j.left, j.right))
    }
  }

  /**
   * A sort-merge join re-expressed as our shuffled hash join (#10, opt-in). Both operators require the
   * same distribution of the same keys and produce the same rows for the same join types, so the
   * hash join is a drop-in per partition; what changes is the memory profile -- the smaller side is
   * held in a per-task hash table -- hence the build side is chosen by the runtime statistics of the
   * two sides (`estimatedBuildSize`: an AQE stage's real size) and must fit `spark.vector.join.maxBuildSize`.
   * A side without statistics is not assumed small: the rule of the issue is "when statistics say
   * the build side fits". `left` / `right` are the join's inputs with the sorts Spark placed for the
   * merge already removed (the caller strips them: a hash join does not need them).
   */
  def sortMergeBuildSide(
      leftKeys: Seq[Expression], rightKeys: Seq[Expression], joinType: JoinType, condition: Option[Expression],
      isSkewJoin: Boolean, left: SparkPlan, right: SparkPlan, maxBuildSize: Long): Either[String, BuildSide] = {
    if (isSkewJoin) Left("skew join not supported")
    else {
      val sides: Seq[BuildSide] = joinType match {
        // An outer join may build either side too: the shuffled join preserves a build side by its matched bitmap (#273).
        case _: InnerLike | FullOuter | LeftOuter | RightOuter => Seq(BuildRight, BuildLeft)
        case LeftSemi | LeftAnti | _: ExistenceJoin => Seq(BuildRight)
        case _ => Nil
      }
      if (sides.isEmpty) Left(s"join type $joinType not supported")
      else {
        val sized = sides.flatMap { s =>
          estimatedBuildSize(if (s == BuildLeft) left else right).map(size => (s, size))
        }
        if (sized.isEmpty) Left("no size statistics for a build side (a sort-merge join is re-expressed only when statistics say the build side fits)")
        else {
          val (buildSide, size) = sized.minBy(_._2)
          val streamed = estimatedBuildSize(if (buildSide == BuildLeft) right else left)
          if (size > maxBuildSize) Left(s"smallest side estimated at $size bytes exceeds ${io.sparkvector.spark.VectorConf.JoinMaxBuildSize}=$maxBuildSize")
          // A semi or anti join may only build its right side; when that is the larger one (TPC-H q4:
          // orders semi-joined with lineitem), hashing it costs more than Spark's merge over the
          // sorted inputs -- measured 14% slower at SF10 -- so the rewrite declines (#311).
          else if (streamed.exists(_ < size)) Left(s"build side estimated at $size bytes is larger than the streamed side (${streamed.get} bytes)")
          else check(leftKeys, rightKeys, joinType, buildSide, condition, left, right, preservedBuild = true).map(_ => buildSide)
        }
      }
    }
  }
}
