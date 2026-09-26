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
package io.vecruntime.spark.expr

import java.lang.foreign.MemorySegment

import io.vecruntime.kernels.{
  ArrowLayout,
  Bitmap,
  BitmapKernels,
  CompareKernels,
  CompareOp,
  MathKernels,
  OverflowKernels,
  PredicateKernels,
  SegmentVectorBuffers,
  VecType,
  VectorBuffers
}
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{DataType, DoubleType, LongType}
import org.apache.spark.sql.vecruntime.VectorErrors

/** `abs`; in ANSI mode an integer `MIN_VALUE` raises Spark's overflow error (the `negateExact` message, no hint). */
final case class AbsExpr(child: VectorExpr, ansi: Boolean, queryContext: QueryContext) extends VectorExpr {
  override def dataType: DataType = child.dataType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    MathKernels.abs(a, data)
    if (ansi && (vecType == VecType.INT32 || vecType == VecType.INT64)) {
      val overflow = ctx.bitmap()
      OverflowKernels.negateOverflow(a, overflow)
      if (ArithExpr.anyActive(overflow, a.validity(), ctx)) {
        throw VectorErrors.arithmeticOverflow(ArithExpr.overflowMessage(vecType), "", queryContext)
      }
    }
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), data)
  }
}

/**
 * Spark's `NormalizeNaNAndZero` over a double: every NaN becomes the canonical one and `-0.0`
 * becomes `0.0`, so that the bit comparison of the group key table and the join tables agrees with
 * Spark's equality on the grouping and join keys the optimizer wraps in it.
 */
final case class NormalizeDoubleExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, ctx.numRows)
    MathKernels.normalizeNaNAndZero(a, data)
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, ctx.numRows, a.validity(), data)
  }
}

/** `sign` / `signum`: double in (Spark casts), double out. */
/** `sqrt` over a double lane: `Math.sqrt` per lane, NaN for a negative argument as in Spark. */
final case class SqrtExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, ctx.numRows)
    MathKernels.sqrt(a, data)
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, ctx.numRows, a.validity(), data)
  }
}

final case class SignumExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, ctx.numRows)
    MathKernels.signum(a, data)
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, ctx.numRows, a.validity(), data)
  }
}

/**
 * `%` / `pmod` and `div` share the divisor-zero rule of `/`: in ANSI mode a zero divisor on an active
 * row raises, otherwise the lane is null. `div` produces a long and, in ANSI mode, raises Spark's
 * `Overflow in integral divide` for `Long.MIN_VALUE div -1`.
 */
final case class DivideLikeExpr(
    kind: DivideLikeExpr.Kind,
    left: VectorExpr,
    right: VectorExpr,
    dataType: DataType,
    ansi: Boolean,
    queryContext: QueryContext
) extends VectorExpr {
  import DivideLikeExpr._

  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, vecType, n)
    var validity: MemorySegment = null
    var divisorZero: MemorySegment = null
    var overflow: MemorySegment = null
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        validity = a.validity()
        kind match {
          case Rem => MathKernels.remainderScalar(MathKernels.RemOp.REM, a, lit.number, false, data)
          case Pmod => MathKernels.remainderScalar(MathKernels.RemOp.PMOD, a, lit.number, false, data)
          case Div =>
            MathKernels.integralDivideScalar(a, lit.number, false, data)
            if (ansi) {
              overflow = ctx.bitmap(); MathKernels.integralDivideOverflowScalar(a, lit.number, false, overflow)
            }
        }
        if (lit.number.doubleValue() == 0.0) { divisorZero = ctx.bitmap(); Bitmap.fill(divisorZero, n, true) }
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        validity = b.validity()
        kind match {
          case Rem => MathKernels.remainderScalar(MathKernels.RemOp.REM, b, lit.number, true, data)
          case Pmod => MathKernels.remainderScalar(MathKernels.RemOp.PMOD, b, lit.number, true, data)
          case Div =>
            MathKernels.integralDivideScalar(b, lit.number, true, data)
            if (ansi) {
              overflow = ctx.bitmap(); MathKernels.integralDivideOverflowScalar(b, lit.number, true, overflow)
            }
        }
        divisorZero = zeroMask(b, ctx)
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        if (a.validity() != null || b.validity() != null) {
          validity = ctx.bitmap()
          BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n)
        }
        kind match {
          case Rem => MathKernels.remainder(MathKernels.RemOp.REM, a, b, data)
          case Pmod => MathKernels.remainder(MathKernels.RemOp.PMOD, a, b, data)
          case Div =>
            MathKernels.integralDivide(a, b, data)
            if (ansi) { overflow = ctx.bitmap(); MathKernels.integralDivideOverflow(a, b, overflow) }
        }
        divisorZero = zeroMask(b, ctx)
    }
    if (overflow != null && ArithExpr.anyActive(overflow, validity, ctx)) {
      throw VectorErrors.arithmeticOverflow("Overflow in integral divide", "try_divide", queryContext)
    }
    // Same tail as ArithExpr's division: affected = zero & valid & active -> raise or null.
    if (divisorZero != null) {
      val affected = ctx.bitmap()
      if (validity == null) BitmapKernels.copy(divisorZero, affected, n)
      else BitmapKernels.and(divisorZero, validity, affected, n)
      if (ctx.active != null) BitmapKernels.and(affected, ctx.active, affected, n)
      if (Bitmap.popcount(affected, n) > 0) {
        if (ansi) throw (if (kind == Div) VectorErrors.divideByZero(queryContext)
                         else VectorErrors.remainderByZero(queryContext))
        val newValidity = ctx.bitmap()
        if (validity == null) BitmapKernels.not(divisorZero, newValidity, n)
        else BitmapKernels.andNot(validity, divisorZero, newValidity, n)
        validity = newValidity
      }
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, validity, data)
  }

  private def zeroMask(b: VectorBuffers, ctx: EvalContext): MemorySegment = {
    val zero = ctx.bitmap()
    CompareKernels.compareScalar(b, java.lang.Double.valueOf(0.0), CompareOp.EQ, zero)
    zero
  }
}

object DivideLikeExpr {
  sealed trait Kind
  case object Rem extends Kind
  case object Pmod extends Kind
  case object Div extends Kind
}

/** `greatest` / `least` over same-typed children; nulls ignored, all-null is null. */
final case class PickExpr(pick: MathKernels.Pick, children: Seq[VectorExpr], dataType: DataType) extends VectorExpr {
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val cols = children.map(CaseWhenExpr.materialise(_, ctx)).toArray
    val data = ArrowLayout.allocateData(ctx.arena, vecType, n)
    val validity = ctx.bitmap()
    MathKernels.pick(pick, cols, data, validity)
    SegmentVectorBuffers.fixedWidth(vecType, n, validity, data)
  }
}

/** `nanvl(a, b)` over doubles with Spark's eval semantics (see `MathKernels.nanvl`). */
final case class NanvlExpr(left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = DoubleType
  override def children: Seq[VectorExpr] = Seq(left, right)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = CaseWhenExpr.materialise(left, ctx)
    // Spark's nanvl evaluates the second argument only where the first is NaN (a null first
    // argument is null): `nanvl(c, 1/c)` must not raise DIVIDE_BY_ZERO on the rows that keep `c`.
    val nan = ctx.bitmap()
    PredicateKernels.isNaN(a, nan)
    if (a.validity() != null) BitmapKernels.and(nan, a.validity(), nan, n)
    if (ctx.active != null) BitmapKernels.and(nan, ctx.active, nan, n)
    val b = ctx.withActive(nan)(CaseWhenExpr.materialise(right, ctx))
    val data = ArrowLayout.allocateData(ctx.arena, VecType.FLOAT64, n)
    val validity = ctx.bitmap()
    MathKernels.nanvl(a, b, data, validity)
    SegmentVectorBuffers.fixedWidth(VecType.FLOAT64, n, validity, data)
  }
}
