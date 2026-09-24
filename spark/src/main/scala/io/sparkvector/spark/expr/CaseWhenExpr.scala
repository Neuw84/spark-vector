/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.spark.expr

import java.lang.foreign.MemorySegment

import io.sparkvector.kernels._
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.types.{BooleanType, DataType, StringType}
import org.apache.spark.unsafe.types.UTF8String

/**
 * `CASE WHEN c1 THEN v1 WHEN c2 THEN v2 ... [ELSE e] END`, and through it `IF` (one branch with an
 * ELSE) and `COALESCE` (`IS NOT NULL` conditions over the same columns).
 *
 * Each condition is evaluated only over the rows still undecided -- `EvalContext.active` narrowed
 * by every earlier branch's wins, the way `AND` narrows its right operand -- and its winning rows
 * are `condition is true` (a null condition counts as false). Each branch value is evaluated only
 * over its own winning rows, the ELSE only over what is left. [[SelectKernels.select]] then blends:
 * the winning branch's value per row, else the ELSE, else null; a null-valued winner is null.
 *
 * A branch value of `None` is a `NULL` literal. A [[LiteralExpr]] branch is materialised as a
 * constant column of the batch's length (string literals included, which is why they are accepted
 * here and nowhere else yet).
 */
final case class CaseWhenExpr(
    branches: Seq[(VectorExpr, Option[VectorExpr])],
    otherwise: Option[Option[VectorExpr]],
    dataType: DataType
) extends VectorExpr {

  override def children: Seq[VectorExpr] =
    branches.flatMap { case (c, v) => c +: v.toSeq } ++ otherwise.flatten.toSeq

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val wins = new Array[MemorySegment](branches.length)
    val values = new Array[VectorBuffers](branches.length)
    // Rows not yet taken by an earlier branch, within the active set; null means "all rows".
    var remaining: MemorySegment = ctx.active
    var k = 0
    while (k < branches.length) {
      val (cond, value) = branches(k)
      val c = ctx.withActive(remaining)(cond.eval(ctx))
      val win = ctx.bitmap()
      if (c.validity() == null) BitmapKernels.copy(c.data(), win, n)
      else BitmapKernels.and(c.data(), c.validity(), win, n)
      if (remaining != null) BitmapKernels.and(win, remaining, win, n)
      wins(k) = win
      values(k) = value.map(v => ctx.withActive(win)(CaseWhenExpr.materialise(v, ctx))).orNull
      val next = ctx.bitmap()
      if (remaining == null) BitmapKernels.not(win, next, n) else BitmapKernels.andNot(remaining, win, next, n)
      remaining = next
      k += 1
    }
    val rest = otherwise.flatten.map(e => ctx.withActive(remaining)(CaseWhenExpr.materialise(e, ctx))).orNull
    SelectKernels.select(TypeMapping.vecTypeOf(dataType), n, wins, values, rest, ctx.active, ctx.arena)
  }
}

object CaseWhenExpr {

  /** A branch value as a column: literals become a constant column of the batch's length. */
  private[expr] def materialise(e: VectorExpr, ctx: EvalContext): VectorBuffers = e match {
    case lit: LiteralExpr => constant(lit, ctx)
    case other => other.eval(ctx)
  }

  private def constant(lit: LiteralExpr, ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val arena = ctx.arena
    val vecType = TypeMapping.vecTypeOf(lit.dataType)
    vecType match {
      case VecType.UTF8 =>
        val bytes = lit.value match {
          case s: UTF8String => s.getBytes
          case s: String => s.getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
        val offsets = ArrowLayout.allocateOffsets(arena, n)
        val data = ArrowLayout.allocateBytes(arena, bytes.length.toLong * n)
        var i = 0
        while (i < n) {
          offsets.setAtIndex(VectorBuffers.LE_INT, i, i * bytes.length)
          MemorySegment.copy(MemorySegment.ofArray(bytes), 0L, data, i.toLong * bytes.length, bytes.length.toLong)
          i += 1
        }
        offsets.setAtIndex(VectorBuffers.LE_INT, n, n * bytes.length)
        SegmentVectorBuffers.utf8(n, null, offsets, data)
      case VecType.BOOL =>
        val bits = ArrowLayout.allocateBitmap(arena, n)
        Bitmap.fill(bits, n, lit.value.asInstanceOf[Boolean])
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
      case fixed =>
        val data = ArrowLayout.allocateData(arena, fixed, n)
        val v = lit.number
        var i = 0
        fixed match {
          case VecType.INT32 => while (i < n) { data.setAtIndex(VectorBuffers.LE_INT, i, v.intValue()); i += 1 }
          case VecType.INT64 => while (i < n) { data.setAtIndex(VectorBuffers.LE_LONG, i, v.longValue()); i += 1 }
          case VecType.FLOAT64 => while (i < n) { data.setAtIndex(VectorBuffers.LE_DOUBLE, i, v.doubleValue()); i += 1 }
          case VecType.DECIMAL128 => // the unscaled value as two limbs per row (#326)
            val big = v.asInstanceOf[java.math.BigInteger]
            val hi = Decimal128.hiOf(big); val lo = Decimal128.loOf(big)
            while (i < n) { Decimal128.set(data, i, hi, lo); i += 1 }
          case other => throw new IllegalStateException(s"no constant column for $other")
        }
        SegmentVectorBuffers.fixedWidth(fixed, n, null, data)
    }
  }

  /** Literal types a CASE branch may carry, beyond the operand literals the compiler accepts elsewhere. */
  def isBranchLiteralType(dt: DataType): Boolean =
    dt == StringType || dt == BooleanType || TypeMapping.isSupported(dt) || TypeMapping.isWideDecimal(
      dt
    ) // wide: a two-limb constant (#326)
}
