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
package io.vecruntime.spark.agg

import java.lang.foreign.MemorySegment

import io.vecruntime.kernels.{Bitmap, BitmapKernels, Decimal128, GroupAssignment, VecType, VectorBuffers}
import io.vecruntime.spark.expr.{EvalContext, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression}
import org.apache.spark.sql.types.{BooleanType, DataType, DateType, DoubleType, IntegerType, LongType}

/**
 * An aggregate with a `FILTER (WHERE p)` clause in an update mode: the inner function sees only the
 * rows where `p` is true (null counts as false), on top of the batch's selection. Ungrouped, that is
 * a derived context whose selection is narrowed; grouped, a copy of the assignment with the other
 * rows' ids cleared. Spark drops the clause in the merge modes, so this never wraps a merge.
 */
final case class FilteredAgg(inner: VectorAggFunction, predicate: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = inner.bufferTypes
  override def emittedTypes(declared: Seq[DataType]): Seq[DataType] = inner.emittedTypes(declared)

  override def newState(): AggState = new AggState {
    private val s = inner.newState()
    override def update(ctx: EvalContext): Unit = {
      val bits = FilteredAgg.passing(ctx, predicate)
      val count = Bitmap.popcount(bits, ctx.numRows)
      if (count > 0) s.update(new EvalContext(ctx.arena, ctx.numRows, ctx.input, bits, count))
    }
    override def bufferValues: Array[Any] = s.bufferValues
  }

  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private val s = inner.newGroupedState()
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      val n = ctx.numRows
      val bits = FilteredAgg.passing(ctx, predicate)
      if (Bitmap.popcount(bits, n) > 0) {
        val ids = groups.ids().clone()
        var i = 0
        while (i < n) { if (!Bitmap.isSet(bits, i)) ids(i) = -1; i += 1 }
        s.update(ctx, GroupAssignment.of(ids, n, groups.numGroups(), ctx.arena, groups.useMasks(), bits))
      }
    }
    override def bufferValue(g: Int, slot: Int): Any = s.bufferValue(g, slot)
  }
}

object FilteredAgg {

  /** Compiles a `FILTER` predicate against the operator's input. */
  def predicate(p: Expression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(p, input).flatMap {
      case _: LiteralExpr => Left("FILTER on a literal")
      case e if e.dataType != BooleanType => Left(s"FILTER of type ${e.dataType.simpleString}")
      case e => Right(e)
    }.left.map(r => s"FILTER ${p.sql}: $r")

  /** Rows where the predicate is true and the batch selects them. */
  private[agg] def passing(ctx: EvalContext, predicate: VectorExpr): MemorySegment = {
    val n = ctx.numRows
    val p = predicate.eval(ctx)
    val bits = ctx.bitmap()
    if (p.validity() == null) BitmapKernels.copy(p.data(), bits, n)
    else BitmapKernels.and(p.data(), p.validity(), bits, n)
    if (ctx.selection != null) BitmapKernels.and(bits, ctx.selection, bits, n)
    bits
  }
}

/**
 * `first(x, ignoreNulls = true)` in an update mode: the first non-null value in row order per group,
 * with Spark's `(first, valueSet)` buffer. Ints, longs, doubles, dates and booleans; the buffer is
 * boxed per group as the output path expects.
 */
final case class FirstAgg(input: VectorExpr, dataType: DataType, ignoreNulls: Boolean = true)
    extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType, BooleanType)

  override def newState(): AggState = new AggState {
    private var value: Any = null
    private var set = false
    override def update(ctx: EvalContext): Unit = if (!set) {
      val v = input.eval(ctx)
      // With ignoreNulls the first selected non-null row; without, the first selected row, null or not.
      val i = if (ignoreNulls) FirstAgg.firstValid(ctx.masked(v), ctx.numRows, 0)
      else if (ctx.selection == null) (if (ctx.numRows > 0) 0 else -1)
      else FirstAgg.firstValid(
        new io.vecruntime.kernels.SegmentVectorBuffers(
          v.`type`(),
          ctx.numRows,
          ctx.selection,
          v.data(),
          v.offsets(),
          v.dictionary()
        ),
        ctx.numRows,
        0
      )
      if (i >= 0) { value = if (Rows.valid(v, i)) FirstAgg.box(v, i, dataType) else null; set = true }
    }
    override def bufferValues: Array[Any] = Array(value, java.lang.Boolean.valueOf(set))
  }

  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private var values = new Array[Any](16)
    private var set = new Array[Boolean](16)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      val n = ctx.numRows
      val v = input.eval(ctx)
      val ids = groups.ids()
      if (groups.numGroups() > values.length) {
        val cap = math.max(groups.numGroups(), values.length * 2)
        val nv = new Array[Any](cap); System.arraycopy(values, 0, nv, 0, values.length); values = nv
        val ns = new Array[Boolean](cap); System.arraycopy(set, 0, ns, 0, set.length); set = ns
      }
      var i = 0
      while (i < n) {
        val g = ids(i)
        if (g >= 0 && !set(g) && (!ignoreNulls || Rows.valid(v, i))) {
          values(g) = if (Rows.valid(v, i)) FirstAgg.box(v, i, dataType) else null; set(g) = true
        }
        i += 1
      }
    }
    override def bufferValue(g: Int, slot: Int): Any =
      if (slot == 0) (if (g < set.length && set(g)) values(g) else null)
      else java.lang.Boolean.valueOf(g < set.length && set(g))
  }
}

object FirstAgg {
  def supports(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | DoubleType | DateType | BooleanType | org.apache.spark.sql.types.StringType => true
    case _: org.apache.spark.sql.types.DecimalType => io.vecruntime.spark.adapter.TypeMapping.isSupported(dt)
    case _ => false
  }

  /** First row at or after `from` that is valid (validity null = all valid), or -1. */
  private[agg] def firstValid(v: VectorBuffers, n: Int, from: Int): Int = {
    val validity = v.validity()
    if (validity == null) return if (n > from) from else -1
    var i = from
    while (i < n) { if (Bitmap.isSet(validity, i)) return i; i += 1 }
    -1
  }

  /** Row `i` of `v` boxed as Spark's internal value. */
  /** Row `i` of a DECIMAL128 lane as well, boxed with the scale of `dt` (first / last over a wide decimal, #259). */
  private[agg] def box(v: VectorBuffers, i: Int, dt: DataType): Any = v.`type`() match {
    case VecType.DECIMAL128 =>
      new java.math.BigDecimal(
        Decimal128.toBigInteger(Decimal128.hi(v.data(), i), Decimal128.lo(v.data(), i)),
        dt.asInstanceOf[org.apache.spark.sql.types.DecimalType].scale
      )
    case _ => box(v, i)
  }

  /** `supports` plus the wide decimals, for the functions that box a row ([[FirstAgg]], [[LastAgg]]). */
  def supportsWide(dt: DataType): Boolean = supports(dt) || (dt match {
    case _: org.apache.spark.sql.types.DecimalType => io.vecruntime.spark.adapter.TypeMapping.hasLane(dt)
    case _ => false
  })

  private[agg] def box(v: VectorBuffers, i: Int): Any = v.`type`() match {
    case VecType.INT32 => java.lang.Integer.valueOf(v.data().getAtIndex(VectorBuffers.LE_INT, i))
    case VecType.INT64 => java.lang.Long.valueOf(v.data().getAtIndex(VectorBuffers.LE_LONG, i))
    case VecType.FLOAT64 => java.lang.Double.valueOf(v.data().getAtIndex(VectorBuffers.LE_DOUBLE, i))
    case VecType.BOOL => java.lang.Boolean.valueOf(Bitmap.isSet(v.data(), i))
    case VecType.UTF8 => org.apache.spark.unsafe.types.UTF8String.fromBytes(v.getUtf8Bytes(i))
    // A wide decimal as its unscaled BigInteger: equality and order only (the window's key changes, #259); the scaled form is box(v, i, dt).
    case VecType.DECIMAL128 => Decimal128.toBigInteger(Decimal128.hi(v.data(), i), Decimal128.lo(v.data(), i))
    case t => throw new IllegalStateException(s"first over $t")
  }
}

/** `first` in a merge mode: the first buffer with `valueSet` in row order per group. */
final case class FirstMergeAgg(first: VectorExpr, valueSet: VectorExpr, dataType: DataType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType, BooleanType)

  override def newState(): AggState = new AggState {
    private var value: Any = null
    private var set = false
    override def update(ctx: EvalContext): Unit = if (!set) {
      val f = first.eval(ctx)
      val s = ctx.masked(valueSet.eval(ctx))
      val n = ctx.numRows
      var i = FirstMergeAgg.firstSet(s, n, 0)
      if (i >= 0) {
        // A set buffer whose value is null is a legitimate (null) first value only when ignoreNulls is
        // false; with ignoreNulls the value is non-null whenever valueSet is.
        value = if (f.validity() == null || Bitmap.isSet(f.validity(), i)) FirstAgg.box(f, i, dataType) else null
        set = true
      }
    }
    override def bufferValues: Array[Any] = Array(value, java.lang.Boolean.valueOf(set))
  }

  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private var values = new Array[Any](16)
    private var set = new Array[Boolean](16)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      val n = ctx.numRows
      val f = first.eval(ctx)
      val s = valueSet.eval(ctx)
      val sValidity = groups.effectiveValidity(s)
      val ids = groups.ids()
      if (groups.numGroups() > values.length) {
        val cap = math.max(groups.numGroups(), values.length * 2)
        val nv = new Array[Any](cap); System.arraycopy(values, 0, nv, 0, values.length); values = nv
        val ns = new Array[Boolean](cap); System.arraycopy(set, 0, ns, 0, set.length); set = ns
      }
      var i = 0
      while (i < n) {
        val g = ids(i)
        if (g >= 0 && !set(g) && (sValidity == null || Bitmap.isSet(sValidity, i)) && Bitmap.isSet(s.data(), i)) {
          values(g) = if (f.validity() == null || Bitmap.isSet(f.validity(), i)) FirstAgg.box(f, i, dataType) else null
          set(g) = true
        }
        i += 1
      }
    }
    override def bufferValue(g: Int, slot: Int): Any =
      if (slot == 0) (if (g < set.length && set(g)) values(g) else null)
      else java.lang.Boolean.valueOf(g < set.length && set(g))
  }
}

object FirstMergeAgg {

  /** First row at or after `from` whose `valueSet` is valid and true, or -1. */
  private[agg] def firstSet(s: VectorBuffers, n: Int, from: Int): Int = {
    val validity = s.validity()
    var i = from
    while (i < n) {
      if ((validity == null || Bitmap.isSet(validity, i)) && Bitmap.isSet(s.data(), i)) return i
      i += 1
    }
    -1
  }
}
