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

import io.sparkvector.kernels.{Bitmap, StringCaseKernels, VectorBuffers}
import io.sparkvector.kernels.StringCaseKernels.{Kind, Side}
import org.apache.spark.sql.catalyst.util.CollationSupport
import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.unsafe.types.UTF8String

/**
 * `upper` / `lower` / `initcap` over a UTF8_BINARY lane. The kernel maps the ASCII rows and flags
 * the rows it cannot decide; those go through Spark's own `CollationSupport` implementation (the
 * same `useICU` choice Spark made for this session), so every row is Spark's result by construction.
 */
final case class CaseMapExpr(kind: Kind, child: VectorExpr, useICU: Boolean) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = child.eval(ctx)
    val flags = ctx.bitmap()
    val slow = StringCaseKernels.slowRows(s, kind, useICU, s.validity(), flags)
    val overrides = if (slow == 0) null
    else {
      val o = new Array[Array[Byte]](ctx.numRows)
      var i = 0
      while (i < ctx.numRows) {
        if (Bitmap.isSet(flags, i))
          o(i) = CaseMapExpr.spark(kind, UTF8String.fromBytes(s.getUtf8Bytes(i)), useICU).getBytes
        i += 1
      }
      o
    }
    StringCaseKernels.caseMap(s, kind, overrides, s.validity(), ctx.arena)
  }
}

object CaseMapExpr {
  private val Utf8Binary = 0

  /** Spark's own mapping for the UTF8_BINARY collation. */
  def spark(kind: Kind, v: UTF8String, useICU: Boolean): UTF8String = kind match {
    case Kind.UPPER => CollationSupport.Upper.exec(v, Utf8Binary, useICU)
    case Kind.LOWER => CollationSupport.Lower.exec(v, Utf8Binary, useICU)
    case Kind.INITCAP => CollationSupport.InitCap.exec(v, Utf8Binary, useICU)
  }
}

/** `trim` / `ltrim` / `rtrim` / `btrim` with a literal trim set (none = the space). */
final case class TrimExpr(side: Side, child: VectorExpr, trimSet: Option[Array[Int]]) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = child.eval(ctx)
    StringCaseKernels.trim(s, side, trimSet.orNull, s.validity(), ctx.arena)
  }
}
