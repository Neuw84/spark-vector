package io.sparkvector.spark.arrow

import java.nio.charset.StandardCharsets
import io.sparkvector.kernels.{Bitmap, VecType, VectorBuffers}
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class ArrowVectorBuffersSuite extends AnyFunSuite with BeforeAndAfterAll {

  private val allocator = new RootAllocator(Long.MaxValue)

  override protected def afterAll(): Unit = {
    allocator.close()
    super.afterAll()
  }

  private val n = 77
  private def nullAt(i: Int): Boolean = i % 5 == 0

  test("IntVector / DateDayVector map to INT32 and read zero copy") {
    val iv = new IntVector("i", allocator)
    val dv = new DateDayVector("d", allocator)
    try {
      iv.allocateNew(n); dv.allocateNew(n)
      (0 until n).foreach { i =>
        if (nullAt(i)) { iv.setNull(i); dv.setNull(i) } else { iv.set(i, i * 7 - 100); dv.set(i, 19000 + i) }
      }
      iv.setValueCount(n); dv.setValueCount(n)

      val vi = ArrowVectorBuffers.forRead(iv)
      val vd = ArrowVectorBuffers.forRead(dv)
      assert(vi.`type`() === VecType.INT32)
      assert(vd.`type`() === VecType.INT32)
      assert(vi.length() === n)
      assert(vi.hasNulls)
      (0 until n).foreach { i =>
        assert(vi.isNull(i) === nullAt(i), s"row $i")
        assert(vd.isNull(i) === nullAt(i), s"row $i")
        if (!nullAt(i)) {
          assert(vi.getInt(i) === iv.get(i))
          assert(vd.getInt(i) === dv.get(i))
        }
      }
      assert(vi.nullCount() === iv.getNullCount)
    } finally { iv.close(); dv.close() }
  }

  test("BigIntVector and Float8Vector read zero copy; no nulls drops validity") {
    val lv = new BigIntVector("l", allocator)
    val fv = new Float8Vector("f", allocator)
    try {
      lv.allocateNew(n); fv.allocateNew(n)
      (0 until n).foreach { i => lv.set(i, i.toLong * 1000000007L); fv.set(i, i * 0.25) }
      lv.setValueCount(n); fv.setValueCount(n)
      val vl = ArrowVectorBuffers.forRead(lv)
      val vf = ArrowVectorBuffers.forRead(fv)
      assert(vl.`type`() === VecType.INT64)
      assert(vf.`type`() === VecType.FLOAT64)
      assert(vl.validity() === null, "no nulls should mean no validity segment")
      assert(vf.validity() === null)
      (0 until n).foreach { i =>
        assert(vl.getLong(i) === lv.get(i))
        assert(vf.getDouble(i) === fv.get(i))
      }
    } finally { lv.close(); fv.close() }
  }

  test("BitVector maps to BOOL bitmap") {
    val bv = new BitVector("b", allocator)
    try {
      bv.allocateNew(n)
      (0 until n).foreach(i => if (nullAt(i)) bv.setNull(i) else bv.set(i, if (i % 3 == 0) 1 else 0))
      bv.setValueCount(n)
      val vb = ArrowVectorBuffers.forRead(bv)
      assert(vb.`type`() === VecType.BOOL)
      (0 until n).foreach { i =>
        assert(vb.isNull(i) === nullAt(i))
        if (!nullAt(i)) assert(vb.getBoolean(i) === (bv.get(i) == 1))
      }
    } finally bv.close()
  }

  test("VarCharVector maps to UTF8 with offsets") {
    val sv = new VarCharVector("s", allocator)
    try {
      sv.allocateNew(n * 8, n)
      (0 until n).foreach { i =>
        if (nullAt(i)) sv.setNull(i) else sv.setSafe(i, s"row-$i-ü".getBytes(StandardCharsets.UTF_8))
      }
      sv.setValueCount(n)
      val vs = ArrowVectorBuffers.forRead(sv)
      assert(vs.`type`() === VecType.UTF8)
      assert(vs.offsets() != null)
      (0 until n).foreach { i =>
        assert(vs.isNull(i) === nullAt(i))
        if (!nullAt(i)) assert(vs.getString(i) === s"row-$i-ü")
      }
    } finally sv.close()
  }

  test("forWrite lets kernels populate a freshly allocated vector through segments") {
    val fv = new Float8Vector("out", allocator)
    try {
      fv.allocateNew(n)
      val w = ArrowVectorBuffers.forWrite(fv, n)
      assert(w.validity() != null)
      (0 until n).foreach { i =>
        if (nullAt(i)) Bitmap.clear(w.validity(), i)
        else {
          Bitmap.set(w.validity(), i)
          w.data().set(VectorBuffers.LE_DOUBLE, i.toLong * 8, i * 1.5)
        }
      }
      fv.setValueCount(n)
      assert(fv.getNullCount === (0 until n).count(nullAt))
      (0 until n).foreach { i =>
        assert(fv.isNull(i) === nullAt(i))
        if (!nullAt(i)) assert(fv.get(i) === i * 1.5)
      }
      // Reading it back goes through the same zero-copy path our operators use.
      val cv = new VectorArrowColumnVector(fv)
      assert(cv.getValueVector eq fv)
      assert(cv.numNulls() === fv.getNullCount)
      assert(cv.getDouble(2) === 3.0)
    } finally fv.close()
  }

  test("unsupported vector types are rejected") {
    val v = new Float4Vector("f4", allocator)
    try {
      assert(!ArrowVectorBuffers.isSupported(v))
      intercept[IllegalArgumentException](ArrowVectorBuffers.forRead(v))
    } finally v.close()
  }
}
