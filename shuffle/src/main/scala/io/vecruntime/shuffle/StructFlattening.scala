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
package io.sparkvector.shuffle

import java.lang.foreign.{Arena, MemorySegment}

import io.sparkvector.kernels.{SegmentVectorBuffers, VecType, VectorBuffers}
import io.sparkvector.spark.adapter.{ColumnVectorAdapters, TypeMapping}
import org.apache.spark.sql.types.{BooleanType, DataType, Decimal, StructField, StructType}
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarArray, ColumnarBatch, ColumnarMap}
import org.apache.spark.unsafe.types.UTF8String

/**
 * Struct columns through the columnar shuffle. The IPC writer and reader move lanes only, so a struct
 * whose leaves all have lanes (Iceberg's `_partition` metadata column on a MERGE, for one) is
 * flattened at the exchange boundary: the struct's own position carries a BOOL lane that is true
 * where the struct is valid, and its fields follow every top-level column, depth first, as ordinary
 * lanes. The reader puts each struct back as a [[StructColumnView]] over those lanes, so the shuffle's
 * output is the child's output exactly.
 *
 * As a hash key a struct hashes its leaves in field order, which is Spark's `Murmur3Hash` over a
 * struct: a null struct leaves the hash unchanged, and so does a null field. A field under a null
 * struct is written as null, so the leaf-by-leaf hash skips it as Spark skips the whole struct.
 */
object StructFlattening {

  sealed trait Node extends Serializable
  final case class Leaf(flat: Int) extends Node
  final case class Struct(dataType: StructType, validity: Int, children: IndexedSeq[Node]) extends Node

  /** Whether a column can cross the shuffle: a lane, or a struct of such columns. */
  def supported(dt: DataType): Boolean = dt match {
    case s: StructType => s.fields.forall(f => supported(f.dataType))
    case other => TypeMapping.hasLane(other)
  }

  /**
   * The layout of a written schema, or None when it holds no struct (the shuffle then runs exactly as
   * before). Every struct must be [[supported]].
   */
  def plan(schema: StructType): Option[Layout] =
    if (!schema.fields.exists(_.dataType.isInstanceOf[StructType])) None
    else {
      val extra = scala.collection.mutable.ArrayBuffer.empty[StructField]
      val width = schema.fields.length
      def child(f: StructField): Node = f.dataType match {
        case s: StructType =>
          val validity = width + extra.length
          extra += StructField(f.name, BooleanType, nullable = false)
          Struct(s, validity, s.fields.map(child).toIndexedSeq)
        case dt =>
          require(TypeMapping.hasLane(dt), s"struct field ${f.name} of type ${dt.simpleString} has no lane")
          val flat = width + extra.length
          extra += StructField(f.name, dt, nullable = true)
          Leaf(flat)
      }
      val top = schema.fields.zipWithIndex.map { case (f, i) =>
        f.dataType match {
          case s: StructType => Struct(s, i, s.fields.map(child).toIndexedSeq)
          case _ => Leaf(i): Node
        }
      }.toIndexedSeq
      val topFields = schema.fields.map { f =>
        f.dataType match {
          case _: StructType => StructField(f.name, BooleanType, nullable = false)
          case _ => f
        }
      }
      Some(Layout(top, StructType(topFields ++ extra), width))
    }

  final case class Layout(top: IndexedSeq[Node], flatSchema: StructType, writtenWidth: Int) {

    /** The flat ordinals a written column hashes as: itself, or a struct's leaves in field order. */
    def hashOrdinals(ordinal: Int): Seq[Int] = {
      def leaves(n: Node): Seq[Int] = n match {
        case Leaf(f) => Seq(f)
        case Struct(_, _, children) => children.flatMap(leaves)
      }
      leaves(top(ordinal))
    }

    /**
     * The written columns as the flat lanes: `columns` are the batch's written columns and `buffers`
     * their adapted lanes (null where a struct has none).
     */
    def flatten(
        columns: Array[ColumnVector],
        buffers: Array[VectorBuffers],
        n: Int,
        arena: Arena
    ): Array[VectorBuffers] = {
      val out = new Array[VectorBuffers](flatSchema.fields.length)
      def fill(node: Node, cv: ColumnVector): Unit = node match {
        case Leaf(f) => out(f) = ColumnVectorAdapters.adapt(cv, n, arena)
        case Struct(_, validity, children) =>
          out(validity) = validityLane(cv, n, arena)
          // A field under a null struct row must read as null (see the object's note on hashing).
          val masked = cv.hasNull
          children.indices.foreach { j =>
            val c = cv.getChild(j)
            fill(children(j), if (masked) new NullMaskedColumnVector(c, cv) else c)
          }
      }
      top.indices.foreach { i =>
        top(i) match {
          case Leaf(f) => out(f) = buffers(i)
          case s: Struct => fill(s, columns(i))
        }
      }
      out
    }

    /** A read batch of flat lanes back to the written columns, structs as views over their lanes. */
    def unflatten(batch: ColumnarBatch): ColumnarBatch = {
      def build(node: Node): ColumnVector = node match {
        case Leaf(f) => batch.column(f)
        case Struct(dt, validity, children) =>
          new StructColumnView(dt, batch.column(validity), children.map(build).toArray, batch.numRows())
      }
      new ColumnarBatch(top.map(build).toArray, batch.numRows())
    }
  }

  /** A BOOL lane, true where `cv` (a struct) is not null. */
  private def validityLane(cv: ColumnVector, n: Int, arena: Arena): VectorBuffers = {
    val bytes = ((n + 63) >>> 6).toLong << 3
    val bits = arena.allocate(math.max(bytes, 8L), 8)
    if (!cv.hasNull) bits.fill(0xff.toByte)
    else {
      bits.fill(0.toByte)
      var i = 0
      while (i < n) {
        if (!cv.isNullAt(i)) io.sparkvector.kernels.Bitmap.set(bits, i)
        i += 1
      }
    }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
  }
}

/** A struct column rebuilt from the shuffle's lanes: null where the validity lane is false. */
final class StructColumnView(dt: StructType, validity: ColumnVector, children: Array[ColumnVector], numRows: Int)
    extends ColumnVector(dt) {
  private var nulls = -1

  override def close(): Unit = { validity.close(); children.foreach(_.close()) }
  override def hasNull(): Boolean = numNulls() > 0
  override def numNulls(): Int = {
    if (nulls < 0) {
      var k = 0
      var i = 0
      while (i < numRows) { if (!validity.getBoolean(i)) k += 1; i += 1 }
      nulls = k
    }
    nulls
  }
  override def isNullAt(rowId: Int): Boolean = !validity.getBoolean(rowId)
  override def getChild(ordinal: Int): ColumnVector = children(ordinal)

  private def unsupported = throw new UnsupportedOperationException("a struct column has only children")
  override def getBoolean(rowId: Int): Boolean = unsupported
  override def getByte(rowId: Int): Byte = unsupported
  override def getShort(rowId: Int): Short = unsupported
  override def getInt(rowId: Int): Int = unsupported
  override def getLong(rowId: Int): Long = unsupported
  override def getFloat(rowId: Int): Float = unsupported
  override def getDouble(rowId: Int): Double = unsupported
  override def getArray(rowId: Int): ColumnarArray = unsupported
  override def getMap(ordinal: Int): ColumnarMap = unsupported
  override def getDecimal(rowId: Int, precision: Int, scale: Int): Decimal = unsupported
  override def getUTF8String(rowId: Int): UTF8String = unsupported
  override def getBinary(rowId: Int): Array[Byte] = unsupported
}

/**
 * A struct's field read under the struct's own nulls: null where either is. The copy path reads it
 * through these getters, so the flattened lane holds a null wherever the struct row was null.
 */
final class NullMaskedColumnVector(inner: ColumnVector, parent: ColumnVector) extends ColumnVector(inner.dataType()) {
  override def close(): Unit = ()
  override def hasNull(): Boolean = true
  override def numNulls(): Int = throw new UnsupportedOperationException("numNulls of a masked view")
  override def isNullAt(rowId: Int): Boolean = parent.isNullAt(rowId) || inner.isNullAt(rowId)
  override def getChild(ordinal: Int): ColumnVector = new NullMaskedColumnVector(inner.getChild(ordinal), this)
  override def getBoolean(rowId: Int): Boolean = inner.getBoolean(rowId)
  override def getByte(rowId: Int): Byte = inner.getByte(rowId)
  override def getShort(rowId: Int): Short = inner.getShort(rowId)
  override def getInt(rowId: Int): Int = inner.getInt(rowId)
  override def getLong(rowId: Int): Long = inner.getLong(rowId)
  override def getFloat(rowId: Int): Float = inner.getFloat(rowId)
  override def getDouble(rowId: Int): Double = inner.getDouble(rowId)
  override def getArray(rowId: Int): ColumnarArray = inner.getArray(rowId)
  override def getMap(ordinal: Int): ColumnarMap = inner.getMap(ordinal)
  override def getDecimal(rowId: Int, precision: Int, scale: Int): Decimal = inner.getDecimal(rowId, precision, scale)
  override def getUTF8String(rowId: Int): UTF8String = inner.getUTF8String(rowId)
  override def getBinary(rowId: Int): Array[Byte] = inner.getBinary(rowId)
}
