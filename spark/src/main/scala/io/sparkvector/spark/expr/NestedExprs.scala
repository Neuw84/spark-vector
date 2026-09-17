package io.sparkvector.spark.expr

import io.sparkvector.kernels.{Bitmap, BitmapKernels, SegmentVectorBuffers, VecType, VectorBuffers}
import io.sparkvector.spark.adapter.ColumnVectorAdapters
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnVector

/**
 * `col.field` (and a chain, `col.a.b`) over a struct column that reaches the operator as Spark's own
 * vector (#19 passes such columns through). Spark's `ColumnarBatch` exposes a struct's fields as
 * child vectors, so the field is read by walking `getChild` down `path` and adapting the leaf like any
 * input column (borrowed when it is ours or off-heap, copied otherwise); nothing is assembled row by
 * row. Spark's `GetStructField` is null when the struct itself is null, so every ancestor's nulls are
 * folded into the lane's validity. The struct column is never adapted as a whole -- it has no lane.
 */
final case class StructFieldExpr(inputOrdinal: Int, path: Seq[Int], dataType: DataType) extends VectorExpr {
  private val steps = path.toArray

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    var cv: ColumnVector = ctx.column(inputOrdinal)
    var parentNulls: java.lang.foreign.MemorySegment = null // bitmap of rows whose every ancestor is non-null
    var s = 0
    while (s < steps.length) {
      if (cv.hasNull) {
        if (parentNulls == null) {
          parentNulls = ctx.bitmap()
          var r = 0
          while (r < n) { if (!cv.isNullAt(r)) Bitmap.set(parentNulls, r); r += 1 }
        } else {
          var r = 0
          while (r < n) { if (cv.isNullAt(r)) Bitmap.clear(parentNulls, r); r += 1 }
        }
      }
      cv = cv.getChild(steps(s))
      s += 1
    }
    val leaf = ColumnVectorAdapters.adapt(cv, n, ctx.arena)
    if (parentNulls == null) leaf
    else {
      val validity = if (leaf.validity() == null) parentNulls else { BitmapKernels.and(parentNulls, leaf.validity(), parentNulls, n); parentNulls }
      StructFieldExpr.withValidity(leaf, validity)
    }
  }

  override def children: Seq[VectorExpr] = Nil
}

/**
 * `size(arr)` / `size(map)` / `cardinality` over a column (or a struct field of one) that has no lane:
 * the element count per row read from Spark's vector; a null array is null (or -1 under
 * `spark.sql.legacy.sizeOfNull`, as Spark's `Size` yields).
 */
final case class SizeExpr(inputOrdinal: Int, path: Seq[Int], isMap: Boolean, legacySizeOfNull: Boolean) extends VectorExpr {
  private val steps = path.toArray
  override def dataType: DataType = org.apache.spark.sql.types.IntegerType

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    var cv: ColumnVector = ctx.column(inputOrdinal)
    val ancestors = new scala.collection.mutable.ArrayBuffer[ColumnVector]
    var s = 0
    while (s < steps.length) { if (cv.hasNull) ancestors += cv; cv = cv.getChild(steps(s)); s += 1 }
    val parents = ancestors.toArray
    val data = io.sparkvector.kernels.ArrowLayout.allocateData(ctx.arena, VecType.INT32, n)
    val validity = if (legacySizeOfNull) null else ctx.bitmap()
    var r = 0
    while (r < n) {
      val isNull = cv.isNullAt(r) || parents.exists(_.isNullAt(r))
      if (isNull) { if (legacySizeOfNull) data.set(VectorBuffers.LE_INT, r.toLong << 2, -1) }
      else {
        val len = if (isMap) cv.getMap(r).numElements() else cv.getArray(r).numElements()
        data.set(VectorBuffers.LE_INT, r.toLong << 2, len)
        if (validity != null) Bitmap.set(validity, r)
      }
      r += 1
    }
    SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, data)
  }

  override def children: Seq[VectorExpr] = Nil
}

/**
 * A struct field of a type without a lane (struct, array, map, wide decimal) projected as a value: never
 * evaluated as a lane -- [[org.apache.spark.sql.vector.VectorProjectExec]] turns it into a
 * `NestedFieldColumnVector` view of Spark's child vector, exactly as it passes a whole such column through.
 */
final case class NestedColumnRef(inputOrdinal: Int, path: Seq[Int], dataType: DataType) extends VectorExpr {
  override def eval(ctx: EvalContext): VectorBuffers =
    throw new IllegalStateException(s"a nested column of type ${dataType.simpleString} has no lane; it is passed through, not evaluated")
  override def children: Seq[VectorExpr] = Nil
}

/**
 * The nulls of a column without a lane (or a struct field of one), for `IS NULL` / `IS NOT NULL` over it
 * (Spark adds `isnotnull(arr)` below a non-outer explode): a BOOL lane whose validity is the non-null
 * rows, ancestors' nulls included; the null tests read only the validity.
 */
final case class NestedValidityExpr(inputOrdinal: Int, path: Seq[Int]) extends VectorExpr {
  private val steps = path.toArray
  override def dataType: DataType = org.apache.spark.sql.types.BooleanType

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    var cv: ColumnVector = ctx.column(inputOrdinal)
    val ancestors = new scala.collection.mutable.ArrayBuffer[ColumnVector]
    var s = 0
    while (s < steps.length) { if (cv.hasNull) ancestors += cv; cv = cv.getChild(steps(s)); s += 1 }
    val parents = ancestors.toArray
    val validity = ctx.bitmap()
    var r = 0
    while (r < n) { if (!(cv.isNullAt(r) || parents.exists(_.isNullAt(r)))) Bitmap.set(validity, r); r += 1 }
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, validity)
  }

  override def children: Seq[VectorExpr] = Nil
}

object StructFieldExpr {
  /** The same buffers under a narrower validity bitmap, whatever the lane's shape. */
  def withValidity(v: VectorBuffers, validity: java.lang.foreign.MemorySegment): VectorBuffers = v match {
    case s: SegmentVectorBuffers => s.withValidity(validity)
    case _ if v.dictionary() != null => SegmentVectorBuffers.dictionaryUtf8(v.length(), validity, v.data(), v.dictionary())
    case _ if v.`type`() == VecType.UTF8 => SegmentVectorBuffers.utf8(v.length(), validity, v.offsets(), v.data())
    case _ => SegmentVectorBuffers.fixedWidth(v.`type`(), v.length(), validity, v.data())
  }
}
