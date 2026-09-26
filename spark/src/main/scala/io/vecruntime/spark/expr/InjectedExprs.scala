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

import java.io.ByteArrayInputStream

import io.vecruntime.kernels.{ArrowLayout, Bitmap, SegmentVectorBuffers, VecType, VectorBuffers}
import io.vecruntime.spark.adapter.TypeMapping
import org.apache.spark.sql.catalyst.expressions.{EmptyRow, Expression, XXH64}
import org.apache.spark.sql.execution.ExecSubqueryExpression
import org.apache.spark.sql.types.{
  BooleanType,
  DataType,
  DateType,
  DecimalType,
  DoubleType,
  IntegerType,
  LongType,
  StringType,
  TimestampType
}
import org.apache.spark.unsafe.Platform
import org.apache.spark.util.sketch.BloomFilter

/**
 * A reference-free expression over a scalar subquery -- the `ScalarSubquery` itself, the
 * `GetStructField` Spark's `MergeScalarSubqueries` leaves over a struct-valued one, a `CASE` over such
 * fields -- read as a literal at execution. Spark runs the subquery before the operator executes
 * (`waitForSubqueries`) and stores the result in the `ScalarSubquery` node, which travels to the
 * executors with the expression exactly as in Spark's own interpreted path; the value is read on
 * first use and materialised as a constant lane (a null result is a null lane).
 */
final case class SubqueryLiteralExpr(expr: Expression) extends VectorExpr {
  override def dataType: DataType = expr.dataType
  override def children: Seq[VectorExpr] = Nil

  @transient lazy val value: Any = expr.eval(EmptyRow)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val v = value
    if (v == null) SubqueryLiteralExpr.nulls(dataType, ctx) else CaseWhenExpr.materialise(LiteralExpr(v, dataType), ctx)
  }
}

object SubqueryLiteralExpr {

  /** Whether `e` is a subquery result in disguise: no column references, deterministic, a subquery somewhere below. */
  def isDeferred(e: Expression): Boolean =
    e.references.isEmpty && e.deterministic && e.exists(_.isInstanceOf[ExecSubqueryExpression])

  /** Literal types a subquery result may take (those the constant-lane materialisation handles). */
  def supports(dt: DataType): Boolean = CaseWhenExpr.isBranchLiteralType(dt)

  /** A lane of `n` nulls of the given type. */
  private[expr] def nulls(dt: DataType, ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val validity = ArrowLayout.allocateBitmap(ctx.arena, n) // a fresh bitmap is all clear: every lane null
    TypeMapping.vecTypeOf(dt) match {
      case VecType.UTF8 =>
        SegmentVectorBuffers.utf8(
          n,
          validity,
          ArrowLayout.allocateOffsets(ctx.arena, n),
          ArrowLayout.allocateBytes(ctx.arena, 0L)
        )
      case VecType.BOOL =>
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, ArrowLayout.allocateBitmap(ctx.arena, n))
      case fixed =>
        SegmentVectorBuffers.fixedWidth(fixed, n, validity, ArrowLayout.allocateData(ctx.arena, fixed, n))
    }
  }
}

/**
 * `xxhash64(c1, c2, ..., seed)`: Spark's `XxHash64` step by step over the lanes -- the seed, then each
 * non-null child folded in with the same `XXH64` call Spark's `HashExpression` makes for its type
 * (`hashInt` for ints and dates, `hashLong` for longs, timestamps and decimals up to 18 digits,
 * `hashLong(doubleToLongBits)` with `-0.0` normalised for doubles, `hashInt(1 | 0)` for booleans,
 * `hashUnsafeBytes` for strings); a null child leaves the running hash unchanged, so the result is
 * never null. Bit-identical to Spark, which is what a bloom filter probe needs.
 */
final case class XxHash64Expr(children: Seq[VectorExpr], types: Seq[DataType], seed: Long) extends VectorExpr {
  override def dataType: DataType = LongType
  private val kinds: Array[Int] = types.map(XxHash64Expr.kindOf).toArray

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val cols = children.map(CaseWhenExpr.materialise(_, ctx)).toArray
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    var i = 0
    while (i < n) {
      var h = seed
      var k = 0
      while (k < cols.length) {
        val v = cols(k)
        if (v.validity() == null || Bitmap.isSet(v.validity(), i)) h = XxHash64Expr.step(kinds(k), v, i, h)
        k += 1
      }
      out.setAtIndex(VectorBuffers.LE_LONG, i, h)
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, null, out)
  }
}

object XxHash64Expr {
  private[expr] val INT = 0; private[expr] val LONG = 1; private[expr] val DOUBLE = 2; private[expr] val BOOL = 3;
  private[expr] val UTF8 = 4

  def supports(dt: DataType): Boolean = dt match {
    case IntegerType | DateType | LongType | TimestampType | DoubleType | BooleanType | StringType => true
    case d: DecimalType => d.precision <= 18
    case _ => false
  }

  private[expr] def kindOf(dt: DataType): Int = dt match {
    case IntegerType | DateType => INT
    case LongType | TimestampType | _: DecimalType => LONG
    case DoubleType => DOUBLE
    case BooleanType => BOOL
    case StringType => UTF8
    case other => throw new IllegalArgumentException(s"xxhash64 over ${other.simpleString}")
  }

  private def step(kind: Int, v: VectorBuffers, i: Int, h: Long): Long = kind match {
    case INT => XXH64.hashInt(v.data().getAtIndex(VectorBuffers.LE_INT, i), h)
    case LONG => XXH64.hashLong(v.data().getAtIndex(VectorBuffers.LE_LONG, i), h)
    case DOUBLE =>
      val d = v.data().getAtIndex(VectorBuffers.LE_DOUBLE, i)
      XXH64.hashLong(java.lang.Double.doubleToLongBits(if (d == -0.0d) 0.0d else d), h)
    case BOOL => XXH64.hashInt(if (Bitmap.isSet(v.data(), i)) 1 else 0, h)
    case UTF8 =>
      val bytes = v.getUtf8Bytes(i)
      XXH64.hashUnsafeBytes(bytes, Platform.BYTE_ARRAY_OFFSET, bytes.length, h)
  }
}

/**
 * `BloomFilterMightContain(filter, xxhash64(key))`: Spark's runtime join filter, probed per lane
 * through Spark's own `BloomFilter` deserialised once from the subquery's bytes. A null filter (the
 * build side produced none) gives a null lane for every row, as Spark's does.
 */
final case class BloomProbeExpr(filter: SubqueryLiteralExpr, hash: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(hash)

  @transient private lazy val bloom: BloomFilter = filter.value match {
    case null => null
    case bytes: Array[Byte] => BloomFilter.readFrom(new ByteArrayInputStream(bytes))
  }

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val b = bloom
    if (b == null) return SubqueryLiteralExpr.nulls(BooleanType, ctx)
    val h = hash.eval(ctx)
    val bits = ArrowLayout.allocateBitmap(ctx.arena, n)
    var i = 0
    while (i < n) {
      if (h.validity() == null || Bitmap.isSet(h.validity(), i)) {
        Bitmap.setTo(bits, i, b.mightContainLong(h.data().getAtIndex(VectorBuffers.LE_LONG, i)))
      }
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, h.validity(), bits)
  }
}
