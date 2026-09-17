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

object StructFieldExpr {
  /** The same buffers under a narrower validity bitmap, whatever the lane's shape. */
  def withValidity(v: VectorBuffers, validity: java.lang.foreign.MemorySegment): VectorBuffers = v match {
    case s: SegmentVectorBuffers => s.withValidity(validity)
    case _ if v.dictionary() != null => SegmentVectorBuffers.dictionaryUtf8(v.length(), validity, v.data(), v.dictionary())
    case _ if v.`type`() == VecType.UTF8 => SegmentVectorBuffers.utf8(v.length(), validity, v.offsets(), v.data())
    case _ => SegmentVectorBuffers.fixedWidth(v.`type`(), v.length(), validity, v.data())
  }
}
