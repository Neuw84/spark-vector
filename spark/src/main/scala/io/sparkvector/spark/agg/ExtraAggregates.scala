package io.sparkvector.spark.agg

import io.sparkvector.kernels.{Bitmap, GroupAssignment, VecType, VectorBuffers}
import io.sparkvector.spark.expr.{EvalContext, VectorExpr}
import org.apache.spark.sql.types.{BooleanType, DataType, StringType}
import org.apache.spark.unsafe.types.UTF8String

/**
 * Scalar per-row accumulators: the aggregates whose state is a value or two per group and whose
 * step is a comparison or a bit operation -- min/max over booleans and strings, first/last, the bit
 * aggregates, min_by/max_by. Each runs the same loop over the rows a group assignment (or the
 * ungrouped context's selection) admits and boxes Spark's internal value for the buffer; the merge
 * modes reuse the update classes over the buffer columns wherever the merge is the same operation.
 */
/** Row-at-a-time helpers over lanes, shared with the window operator (boundary detection boxes keys). */
object Rows {
  @inline def valid(v: VectorBuffers, i: Int): Boolean = v.validity() == null || Bitmap.isSet(v.validity(), i)

  /** Every row of the ungrouped context (selection honoured), as group 0. */
  def ungrouped(ctx: EvalContext)(f: Int => Unit): Unit = {
    val n = ctx.numRows
    val sel = ctx.selection
    var i = 0
    while (i < n) { if (sel == null || Bitmap.isSet(sel, i)) f(i); i += 1 }
  }

  /** Every row with a group (unselected rows carry -1). */
  def grouped(ctx: EvalContext, groups: GroupAssignment)(f: (Int, Int) => Unit): Unit = {
    val n = ctx.numRows
    val ids = groups.ids()
    var i = 0
    while (i < n) { val g = ids(i); if (g >= 0) f(g, i); i += 1 }
  }

  /** Row `i` boxed as Spark's internal value, strings as UTF8String. */
  def box(v: VectorBuffers, i: Int): Any = v.`type`() match {
    case VecType.UTF8 => UTF8String.fromBytes(v.getUtf8Bytes(i))
    case _ => FirstAgg.box(v, i)
  }

  /** Spark's ordering of two rows of the same lane: signed for integers, total for doubles, binary for strings. */
  def compare(a: VectorBuffers, i: Int, b: VectorBuffers, j: Int): Int = a.`type`() match {
    case VecType.INT32 => Integer.compare(a.data().getAtIndex(VectorBuffers.LE_INT, i), b.data().getAtIndex(VectorBuffers.LE_INT, j))
    case VecType.INT64 => java.lang.Long.compare(a.data().getAtIndex(VectorBuffers.LE_LONG, i), b.data().getAtIndex(VectorBuffers.LE_LONG, j))
    case VecType.FLOAT64 => java.lang.Double.compare(a.data().getAtIndex(VectorBuffers.LE_DOUBLE, i), b.data().getAtIndex(VectorBuffers.LE_DOUBLE, j))
    case VecType.BOOL => java.lang.Boolean.compare(Bitmap.isSet(a.data(), i), Bitmap.isSet(b.data(), j))
    case VecType.UTF8 => java.util.Arrays.compareUnsigned(a.getUtf8Bytes(i), b.getUtf8Bytes(j))
    case t => throw new IllegalStateException(s"ordering over $t")
  }

  /** Compares a boxed value with row `j`: the boxed side is an earlier row of the same lane. */
  def compareBoxed(boxed: Any, b: VectorBuffers, j: Int): Int = b.`type`() match {
    case VecType.INT32 => Integer.compare(boxed.asInstanceOf[java.lang.Integer], b.data().getAtIndex(VectorBuffers.LE_INT, j))
    case VecType.INT64 => java.lang.Long.compare(boxed.asInstanceOf[java.lang.Long], b.data().getAtIndex(VectorBuffers.LE_LONG, j))
    case VecType.FLOAT64 => java.lang.Double.compare(boxed.asInstanceOf[java.lang.Double], b.data().getAtIndex(VectorBuffers.LE_DOUBLE, j))
    case VecType.BOOL => java.lang.Boolean.compare(boxed.asInstanceOf[java.lang.Boolean], Bitmap.isSet(b.data(), j))
    case VecType.UTF8 => java.util.Arrays.compareUnsigned(boxed.asInstanceOf[UTF8String].getBytes, b.getUtf8Bytes(j))
    case t => throw new IllegalStateException(s"ordering over $t")
  }

  def supportsOrdering(t: VecType): Boolean = t != null
}

/** Growable per-group boxed state: one value and one flag per group. */
private[agg] final class GroupValues {
  var values = new Array[Any](16)
  var set = new Array[Boolean](16)
  def ensure(groups: Int): Unit = if (groups > values.length) {
    val cap = math.max(groups, values.length * 2)
    values = java.util.Arrays.copyOf(values.asInstanceOf[Array[AnyRef]], cap).asInstanceOf[Array[Any]]
    set = java.util.Arrays.copyOf(set, cap)
  }
  def value(g: Int): Any = if (g < set.length && set(g)) values(g) else null
  def isSet(g: Int): Boolean = g < set.length && set(g)
}

/**
 * MIN / MAX over a boolean or string lane (Spark's `bool_and` / `bool_or` are `min` / `max` over a
 * boolean after their rewrite): a comparison per row, Spark's binary order for strings. The merge is
 * the same operation over the buffer column.
 */
final case class OrderedMinMaxAgg(input: VectorExpr, isMin: Boolean, dataType: DataType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType)
  private def better(c: Int): Boolean = if (isMin) c < 0 else c > 0

  override def newState(): AggState = new AggState {
    private var best: Any = null
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      Rows.ungrouped(ctx) { i =>
        if (Rows.valid(v, i) && (best == null || better(-Rows.compareBoxed(best, v, i)))) best = Rows.box(v, i)
      }
    }
    override def bufferValues: Array[Any] = Array(best)
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val state = new GroupValues
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      state.ensure(groups.numGroups())
      val v = input.eval(ctx)
      Rows.grouped(ctx, groups) { (g, i) =>
        if (Rows.valid(v, i) && (!state.set(g) || better(-Rows.compareBoxed(state.values(g), v, i)))) { state.values(g) = Rows.box(v, i); state.set(g) = true }
      }
    }
    override def bufferValue(g: Int, slot: Int): Any = state.value(g)
  }
}

/**
 * LAST with or without `ignoreNulls`: the value of the last row (the last non-null row when nulls are
 * ignored) in partition order, Spark's `(last, valueSet)` buffer. In a merge mode the buffers arrive in
 * partition order too, so the same class serves with `valueSet` as the guard.
 */
final case class LastAgg(input: VectorExpr, dataType: DataType, ignoreNulls: Boolean, valueSet: Option[VectorExpr]) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType, BooleanType)
  /** Whether row `i` counts: a non-null value (or any row without ignoreNulls); in a merge, a set buffer. */
  private def counts(v: VectorBuffers, set: VectorBuffers, i: Int): Boolean =
    if (set != null) Rows.valid(set, i) && Bitmap.isSet(set.data(), i) else !ignoreNulls || Rows.valid(v, i)

  override def newState(): AggState = new AggState {
    private var value: Any = null
    private var seen = false
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      val set = valueSet.map(_.eval(ctx)).orNull
      Rows.ungrouped(ctx) { i => if (counts(v, set, i)) { value = if (Rows.valid(v, i)) FirstAgg.box(v, i, dataType) else null; seen = true } }
    }
    override def bufferValues: Array[Any] = Array(value, java.lang.Boolean.valueOf(seen))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val state = new GroupValues
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      state.ensure(groups.numGroups())
      val v = input.eval(ctx)
      val set = valueSet.map(_.eval(ctx)).orNull
      Rows.grouped(ctx, groups) { (g, i) => if (counts(v, set, i)) { state.values(g) = if (Rows.valid(v, i)) FirstAgg.box(v, i, dataType) else null; state.set(g) = true } }
    }
    override def bufferValue(g: Int, slot: Int): Any = if (slot == 0) state.value(g) else java.lang.Boolean.valueOf(state.isSet(g))
  }
}

/** `bit_and` / `bit_or` / `bit_xor` over int and bigint lanes; null over no rows. The merge is the same operation. */
final case class BitAgg(input: VectorExpr, op: BitAgg.Op, dataType: DataType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType)
  private def apply(acc: Long, x: Long): Long = op match {
    case BitAgg.And => acc & x
    case BitAgg.Or => acc | x
    case BitAgg.Xor => acc ^ x
  }
  private def read(v: VectorBuffers, i: Int): Long =
    if (v.`type`() == VecType.INT32) v.data().getAtIndex(VectorBuffers.LE_INT, i).toLong else v.data().getAtIndex(VectorBuffers.LE_LONG, i)
  private def box(acc: Long): Any = if (input.vecType == VecType.INT32) java.lang.Integer.valueOf(acc.toInt) else java.lang.Long.valueOf(acc)

  override def newState(): AggState = new AggState {
    private var acc = 0L
    private var seen = false
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      Rows.ungrouped(ctx) { i => if (Rows.valid(v, i)) { acc = if (seen) apply(acc, read(v, i)) else read(v, i); seen = true } }
    }
    override def bufferValues: Array[Any] = Array(if (seen) box(acc) else null)
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private var acc = new Array[Long](16)
    private var seen = new Array[Boolean](16)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      if (groups.numGroups() > acc.length) {
        val cap = math.max(groups.numGroups(), acc.length * 2)
        acc = java.util.Arrays.copyOf(acc, cap); seen = java.util.Arrays.copyOf(seen, cap)
      }
      val v = input.eval(ctx)
      Rows.grouped(ctx, groups) { (g, i) => if (Rows.valid(v, i)) { acc(g) = if (seen(g)) apply(acc(g), read(v, i)) else read(v, i); seen(g) = true } }
    }
    override def bufferValue(g: Int, slot: Int): Any = if (g < seen.length && seen(g)) box(acc(g)) else null
  }
}

object BitAgg {
  sealed trait Op
  case object And extends Op
  case object Or extends Op
  case object Xor extends Op
}

/**
 * `max_by` / `min_by`: the value of the row with the greatest / least ordering, a two-field state
 * (value, ordering) per group as Spark's buffer. Rows with a null ordering are skipped; on a tie the
 * later row wins, as in Spark, whose predicate keeps the old value only when it is strictly better.
 * The merge is the same operation over the two buffers.
 */
final case class MaxMinByAgg(value: VectorExpr, ordering: VectorExpr, isMax: Boolean, valueType: DataType, orderingType: DataType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(valueType, orderingType)
  /** Whether the held ordering (compared to the new one as `c`) is strictly better and stays. */
  private def keepOld(c: Int): Boolean = if (isMax) c > 0 else c < 0

  override def newState(): AggState = new AggState {
    private var bestValue: Any = null
    private var bestKey: Any = null
    private var seen = false
    override def update(ctx: EvalContext): Unit = {
      val v = value.eval(ctx)
      val k = ordering.eval(ctx)
      Rows.ungrouped(ctx) { i =>
        if (Rows.valid(k, i) && (!seen || !keepOld(Rows.compareBoxed(bestKey, k, i)))) {
          bestKey = Rows.box(k, i); bestValue = if (Rows.valid(v, i)) Rows.box(v, i) else null; seen = true
        }
      }
    }
    override def bufferValues: Array[Any] = Array(bestValue, bestKey)
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val values = new GroupValues
    private val keys = new GroupValues
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      values.ensure(groups.numGroups()); keys.ensure(groups.numGroups())
      val v = value.eval(ctx)
      val k = ordering.eval(ctx)
      Rows.grouped(ctx, groups) { (g, i) =>
        if (Rows.valid(k, i) && (!keys.set(g) || !keepOld(Rows.compareBoxed(keys.values(g), k, i)))) {
          keys.values(g) = Rows.box(k, i); keys.set(g) = true
          values.values(g) = if (Rows.valid(v, i)) Rows.box(v, i) else null; values.set(g) = true
        }
      }
    }
    override def bufferValue(g: Int, slot: Int): Any = if (slot == 0) values.value(g) else keys.value(g)
  }
}
