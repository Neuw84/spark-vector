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

import io.sparkvector.kernels.{StringLengthKernels, VectorBuffers}
import org.apache.spark.sql.types.{DataType, IntegerType, StringType}

/**
 * `length` (code points), `octet_length`, `bit_length` and `ascii` over a UTF8 lane -- an INT32 out,
 * computed once per dictionary entry when the input is dictionary-encoded -- and `chr` over an
 * integer lane, the string writer of the family. Null in, null out.
 */
final case class StringMeasureExpr(measure: StringLengthKernels.Measure, child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = IntegerType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val s = child.eval(ctx)
    StringLengthKernels.measure(measure, s, s.validity(), ctx.arena)
  }
}

final case class ChrExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = child.eval(ctx)
    StringLengthKernels.chr(n, n.validity(), ctx.arena)
  }
}
