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

import io.sparkvector.kernels.{StringConcatKernels, VectorBuffers}
import io.sparkvector.kernels.StringConcatKernels.Part
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.sql.vector.VectorErrors

/**
 * `concat`, `concat_ws` and `elt` over `StringConcatKernels`: each input is a UTF8 lane or a string
 * literal; the kernel sums the per-row lengths across the inputs and copies once.
 */
private[expr] object StringConcats {

  /** Evaluates the inputs to kernel parts; the lanes come back separately for the validity. */
  def parts(inputs: Seq[VectorExpr], ctx: EvalContext): (Array[Part], Seq[VectorBuffers]) = {
    val lanes = Seq.newBuilder[VectorBuffers]
    val ps = inputs.map {
      case lit: LiteralExpr => Part.literal(lit.utf8Bytes)
      case e =>
        val v = e.eval(ctx)
        lanes += v
        Part.of(v)
    }
    (ps.toArray, lanes.result())
  }
}

/** `concat(s1, s2, ...)`: null when any input is null. */
final case class ConcatExpr(inputs: Seq[VectorExpr]) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = inputs
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (ps, lanes) = StringConcats.parts(inputs, ctx)
    StringConcatKernels.concat(ps, ctx.numRows, StringSlices.validity(ctx, lanes: _*), ctx.arena)
  }
}

/** `concat_ws(sep, s1, s2, ...)`: null for a null separator; null inputs are skipped. */
final case class ConcatWsExpr(sep: VectorExpr, inputs: Seq[VectorExpr]) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = sep +: inputs
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (sp, _) = StringConcats.parts(Seq(sep), ctx)
    val (ps, _) = StringConcats.parts(inputs, ctx)
    StringConcatKernels.concatWs(sp(0), ps, ctx.numRows, ctx.arena)
  }
}

/**
 * `elt(index, s1, s2, ...)`: the input at the 1-based index; null for a null or out-of-range index
 * (under ANSI an out-of-range index on an active row raises Spark's INVALID_ARRAY_INDEX) or a null
 * pick.
 */
final case class EltExpr(index: VectorExpr, inputs: Seq[VectorExpr], ansi: Boolean, queryContext: QueryContext)
    extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = index +: inputs
  override def eval(ctx: EvalContext): VectorBuffers = {
    val idx = CaseWhenExpr.materialise(index, ctx)
    val (ps, _) = StringConcats.parts(inputs, ctx)
    if (ansi) {
      val bad = StringConcatKernels.firstInvalidIndex(idx, inputs.length, ctx.active, ctx.numRows)
      if (bad >= 0) throw VectorErrors.invalidArrayIndex(idx.getInt(bad), inputs.length, queryContext)
    }
    StringConcatKernels.elt(idx, ps, ctx.numRows, ctx.arena)
  }
}
