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

import java.security.MessageDigest
import java.util.zip.CRC32

import io.sparkvector.kernels.{ArrowLayout, Bitmap, SegmentVectorBuffers, StringConcatKernels, VecType, VectorBuffers}
import org.apache.spark.sql.types.{DataType, IntegerType, LongType, StringType}
import org.apache.spark.unsafe.Platform
import org.apache.spark.unsafe.hash.Murmur3_x86_32

/**
 * `hash(...)`: Spark's Murmur3 chain over the children in order, calling Spark's own
 * `Murmur3_x86_32` steps -- a null child leaves the running value alone, `-0.0` is normalised, ints
 * and dates hash as ints, longs, timestamps and short decimals as longs, booleans as 1/0, strings as
 * their bytes. The same per-type rules `XxHash64Expr` uses, so the two agree on what they accept.
 */
final case class Murmur3HashExpr(children: Seq[VectorExpr], types: Seq[DataType], seed: Int) extends VectorExpr {
  override def dataType: DataType = IntegerType
  private val kinds: Array[Int] = types.map(XxHash64Expr.kindOf).toArray

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val cols = children.map(CaseWhenExpr.materialise(_, ctx)).toArray
    val out = ArrowLayout.allocateData(ctx.arena, VecType.INT32, n)
    var i = 0
    while (i < n) {
      var h = seed
      var k = 0
      while (k < cols.length) {
        val v = cols(k)
        if (v.validity() == null || Bitmap.isSet(v.validity(), i)) h = Murmur3HashExpr.step(kinds(k), v, i, h)
        k += 1
      }
      out.setAtIndex(VectorBuffers.LE_INT, i, h)
      i += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT32, n, null, out)
  }
}

object Murmur3HashExpr {
  private def step(kind: Int, v: VectorBuffers, i: Int, h: Int): Int = kind match {
    case XxHash64Expr.INT => Murmur3_x86_32.hashInt(v.data().getAtIndex(VectorBuffers.LE_INT, i), h)
    case XxHash64Expr.LONG => Murmur3_x86_32.hashLong(v.data().getAtIndex(VectorBuffers.LE_LONG, i), h)
    case XxHash64Expr.DOUBLE =>
      val d = v.data().getAtIndex(VectorBuffers.LE_DOUBLE, i)
      Murmur3_x86_32.hashLong(java.lang.Double.doubleToLongBits(if (d == -0.0d) 0.0d else d), h)
    case XxHash64Expr.BOOL => Murmur3_x86_32.hashInt(if (Bitmap.isSet(v.data(), i)) 1 else 0, h)
    case _ =>
      val bytes = v.getUtf8Bytes(i)
      Murmur3_x86_32.hashUnsafeBytes(bytes, Platform.BYTE_ARRAY_OFFSET, bytes.length, h)
  }
}

/**
 * `md5`, `sha1`, `sha2` and `crc32` over the bytes of a UTF8 lane (Spark's `string -> binary` cast is
 * a reinterpretation): a per-row digest loop -- not vectorisable, present so a query containing one
 * keeps the operator. The hex strings go out through the row writer.
 */
final case class DigestExpr(kind: DigestExpr.Kind, child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = if (kind == DigestExpr.Crc32) LongType else StringType
  override def children: Seq[VectorExpr] = Seq(child)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val v = CaseWhenExpr.materialise(child, ctx)
    val n = ctx.numRows
    kind match {
      case DigestExpr.Crc32 =>
        val out = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
        val crc = new CRC32
        var i = 0
        while (i < n) {
          if (v.validity() == null || Bitmap.isSet(v.validity(), i)) {
            crc.reset()
            crc.update(v.getUtf8Bytes(i))
            out.setAtIndex(VectorBuffers.LE_LONG, i, crc.getValue)
          }
          i += 1
        }
        SegmentVectorBuffers.fixedWidth(VecType.INT64, n, v.validity(), out)
      case DigestExpr.Sha2(bits) if !DigestExpr.sha2Algorithm(bits).isDefined =>
        // Spark returns null for a bit length it does not know.
        SegmentVectorBuffers.utf8(
          n,
          ArrowLayout.allocateBitmap(ctx.arena, n),
          ArrowLayout.allocateOffsets(ctx.arena, n),
          ArrowLayout.allocateBytes(ctx.arena, 0)
        )
      case _ =>
        val md = MessageDigest.getInstance(kind match {
          case DigestExpr.Md5 => "MD5"
          case DigestExpr.Sha1 => "SHA-1"
          case DigestExpr.Sha2(bits) => DigestExpr.sha2Algorithm(bits).get
          case DigestExpr.Crc32 => throw new IllegalStateException("unreachable")
        })
        val rows = new Array[Array[Byte]](n)
        var i = 0
        while (i < n) {
          if (v.validity() == null || Bitmap.isSet(v.validity(), i)) {
            md.reset()
            rows(i) = DigestExpr.hex(md.digest(v.getUtf8Bytes(i)))
          }
          i += 1
        }
        StringConcatKernels.fromRows(rows, null, ctx.arena)
    }
  }
}

object DigestExpr {
  sealed trait Kind
  case object Md5 extends Kind
  case object Sha1 extends Kind
  final case class Sha2(bits: Int) extends Kind
  case object Crc32 extends Kind

  /** Spark's `sha2` bit lengths: 0 means 256; anything else is a null result. */
  def sha2Algorithm(bits: Int): Option[String] = bits match {
    case 224 => Some("SHA-224")
    case 0 | 256 => Some("SHA-256")
    case 384 => Some("SHA-384")
    case 512 => Some("SHA-512")
    case _ => None
  }

  private val Digits = "0123456789abcdef".getBytes

  /** Lowercase hex, as Spark's `DigestUtils.*Hex`. */
  def hex(bytes: Array[Byte]): Array[Byte] = {
    val out = new Array[Byte](bytes.length * 2)
    var i = 0
    while (i < bytes.length) {
      out(2 * i) = Digits((bytes(i) >> 4) & 0xf)
      out(2 * i + 1) = Digits(bytes(i) & 0xf)
      i += 1
    }
    out
  }
}
