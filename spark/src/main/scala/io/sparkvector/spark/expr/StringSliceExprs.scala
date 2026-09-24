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

import io.sparkvector.kernels.{BitmapKernels, StringSliceKernels, VectorBuffers}
import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.unsafe.types.UTF8String

/**
 * The string functions that write new UTF8 data -- `substring`, `lpad`/`rpad`, `repeat`, `space`,
 * `overlay` -- over `StringSliceKernels`. An integer argument is a literal (passed as the scalar) or
 * an INT32 lane; a string argument other than the subject is a lane or a literal's bytes. A row is
 * null when any lane input is null, as Spark's null-intolerant expressions; the kernel writes no
 * bytes for those rows.
 */
private[expr] object StringSlices {

  /** The int argument as (lane or null, scalar). */
  def intArg(e: VectorExpr, ctx: EvalContext): (VectorBuffers, Int) = e match {
    case lit: LiteralExpr => (null, lit.number.intValue())
    case other => (other.eval(ctx), 0)
  }

  /** The string argument as (lane or null, literal bytes or null). */
  def strArg(e: VectorExpr, ctx: EvalContext): (VectorBuffers, Array[Byte]) = e match {
    case lit: LiteralExpr => (null, lit.utf8Bytes)
    case other => (other.eval(ctx), null)
  }

  /** The AND of the lanes' validities, or null when none has nulls. */
  def validity(ctx: EvalContext, lanes: VectorBuffers*): MemorySegment = {
    val withNulls = lanes.filter(v => v != null && v.validity() != null)
    if (withNulls.isEmpty) null
    else if (withNulls.length == 1) withNulls.head.validity()
    else {
      val out = ctx.bitmap()
      BitmapKernels.combineValidity(withNulls(0).validity(), withNulls(1).validity(), out, ctx.numRows)
      withNulls.drop(2).foreach(v => BitmapKernels.and(out, v.validity(), out, ctx.numRows))
      out
    }
  }

  /** A literal too large to size a batch for: refused at planning rather than capped at run time. */
  val MaxLiteralCount: Int = 1 << 20
}

final case class SubstringExpr(str: VectorExpr, pos: VectorExpr, len: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(str, pos, len)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = str.eval(ctx)
    val (pc, p) = StringSlices.intArg(pos, ctx)
    val (lc, l) = StringSlices.intArg(len, ctx)
    StringSliceKernels.substring(s, pc, p, lc, l, StringSlices.validity(ctx, s, pc, lc), ctx.arena)
  }
}

final case class PadExpr(str: VectorExpr, len: VectorExpr, pad: VectorExpr, left: Boolean) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(str, len, pad)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = str.eval(ctx)
    val (lc, l) = StringSlices.intArg(len, ctx)
    val (pc, pb) = StringSlices.strArg(pad, ctx)
    StringSliceKernels.pad(s, lc, l, pc, pb, left, StringSlices.validity(ctx, s, lc, pc), ctx.arena)
  }
}

final case class RepeatExpr(str: VectorExpr, times: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(str, times)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = str.eval(ctx)
    val (tc, t) = StringSlices.intArg(times, ctx)
    StringSliceKernels.repeat(s, tc, t, StringSlices.validity(ctx, s, tc), ctx.arena)
  }
}

final case class SpaceExpr(count: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(count)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (cc, c) = StringSlices.intArg(count, ctx)
    StringSliceKernels.space(ctx.numRows, cc, c, StringSlices.validity(ctx, cc), ctx.arena)
  }
}

final case class OverlayExpr(input: VectorExpr, replace: VectorExpr, pos: VectorExpr, len: VectorExpr)
    extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(input, replace, pos, len)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = input.eval(ctx)
    val (rc, rb) = StringSlices.strArg(replace, ctx)
    val (pc, p) = StringSlices.intArg(pos, ctx)
    val (lc, l) = StringSlices.intArg(len, ctx)
    StringSliceKernels.overlay(s, rc, rb, pc, p, lc, l, StringSlices.validity(ctx, s, rc, pc, lc), ctx.arena)
  }
}
