package io.sparkvector.spark.arrow;

import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.BitmapKernels;
import io.sparkvector.kernels.CompactKernels;
import io.sparkvector.kernels.GatherKernels;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import java.lang.foreign.MemorySegment;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BaseVariableWidthVector;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.sql.vectorized.ColumnVector;

/**
 * Builds output columns as unshaded Arrow vectors that Spark reads through {@link
 * VectorArrowColumnVector}. Kernels write straight into the vectors' buffers via {@link
 * ArrowVectorBuffers#forWrite}.
 */
public final class ArrowOutput {

  private ArrowOutput() {}

  /** Allocates an empty Arrow vector for a supported Spark type. */
  public static FieldVector newVector(String name, DataType dt, BufferAllocator allocator) {
    if (dt instanceof IntegerType) {
      return new IntVector(name, allocator);
    }
    if (dt instanceof DateType) {
      return new DateDayVector(name, allocator);
    }
    if (dt instanceof LongType) {
      return new BigIntVector(name, allocator);
    }
    if (dt instanceof TimestampType) {
      return new TimeStampMicroTZVector(name, allocator, "UTC");
    }
    if (dt instanceof DoubleType) {
      return new Float8Vector(name, allocator);
    }
    if (dt instanceof BooleanType) {
      return new BitVector(name, allocator);
    }
    if (dt instanceof StringType) {
      return new VarCharVector(name, allocator);
    }
    throw new UnsupportedOperationException("unsupported output type " + dt);
  }

  /**
   * Allocates a fixed-width (or BOOL) vector for {@code length} elements and returns writable
   * buffers. Call {@link #finish} once the kernels have written values and validity bits.
   */
  public static ArrowVectorBuffers allocateFixed(
      String name, DataType dt, int length, BufferAllocator allocator) {
    FieldVector v = newVector(name, dt, allocator);
    v.setInitialCapacity(length);
    v.allocateNew();
    return ArrowVectorBuffers.forWrite(v, length);
  }

  /** Allocates a UTF8 vector with room for {@code length} elements and {@code bytes} of data. */
  public static ArrowVectorBuffers allocateUtf8(
      String name, int length, long bytes, BufferAllocator allocator) {
    VarCharVector v = new VarCharVector(name, allocator);
    v.allocateNew(Math.max(bytes, 1L), length);
    return ArrowVectorBuffers.forWrite(v, length);
  }

  /**
   * Marks the vector as fully written. When {@code allValid} is true the validity buffer is set to
   * all ones so Arrow reports zero nulls.
   */
  public static ColumnVector finish(ArrowVectorBuffers out, int length, boolean allValid) {
    if (allValid) {
      Bitmap.fill(out.validity(), length, true);
    }
    FieldVector v = (FieldVector) out.vector();
    if (v instanceof BaseVariableWidthVector vw) {
      // setValueCount "fills holes" in the offsets from lastSet+1; we wrote them all directly.
      vw.setLastSet(length - 1);
    }
    v.setValueCount(length);
    return new VectorArrowColumnVector(v);
  }

  /**
   * Materialises {@code in} filtered by {@code selection} into a new Arrow vector of the given
   * Spark type. {@code outCount} must equal the selection's popcount.
   */
  public static ColumnVector compact(
      String name,
      DataType dt,
      VectorBuffers in,
      MemorySegment selection,
      int outCount,
      BufferAllocator allocator) {
    if (in.type() == VecType.UTF8) {
      if (in.isDictionaryEncoded()) {
        return compactDictionary(name, in, selection, outCount, allocator);
      }
      long bytes = CompactKernels.selectedUtf8Bytes(in, selection);
      ArrowVectorBuffers out = allocateUtf8(name, outCount, bytes, allocator);
      CompactKernels.compactUtf8(in, selection, outCount, out.offsets(), out.data(), out.validity());
      return finish(out, outCount, !in.hasNulls());
    }
    ArrowVectorBuffers out = allocateFixed(name, dt, outCount, allocator);
    CompactKernels.compactFixed(in, selection, outCount, out.data(), out.validity());
    return finish(out, outCount, !in.hasNulls());
  }

  /**
   * Dictionary-encoded input stays dictionary encoded: the int32 indices are compacted with the
   * fixed-width kernel and the (small) dictionary is copied into its own vector.
   */
  private static ColumnVector compactDictionary(
      String name, VectorBuffers in, MemorySegment selection, int outCount, BufferAllocator allocator) {
    IntVector indices = new IntVector(name, allocator);
    indices.setInitialCapacity(outCount);
    indices.allocateNew();
    ArrowVectorBuffers out = ArrowVectorBuffers.forWrite(indices, outCount);
    if (selection == null) {
      MemorySegment.copy(in.data(), 0, out.data(), 0, (long) outCount << 2);
      if (in.hasNulls()) {
        BitmapKernels.copy(in.validity(), out.validity(), outCount);
      }
    } else {
      CompactKernels.compactFixed(in, selection, outCount, out.data(), out.validity());
    }
    if (!in.hasNulls()) {
      Bitmap.fill(out.validity(), outCount, true);
    }
    indices.setValueCount(outCount);
    VarCharVector dictionary = copyDictionary(name + ".dictionary", in.dictionary(), allocator);
    return new VectorDictionaryColumnVector(indices, dictionary);
  }

  private static VarCharVector copyDictionary(String name, VectorBuffers dict, BufferAllocator allocator) {
    int n = dict.length();
    int end = dict.offsets().get(VectorBuffers.LE_INT, (long) n << 2);
    ArrowVectorBuffers out = allocateUtf8(name, n, end, allocator);
    MemorySegment.copy(dict.offsets(), 0, out.offsets(), 0, ((long) n + 1) << 2);
    MemorySegment.copy(dict.data(), 0, out.data(), 0, end);
    if (dict.hasNulls()) {
      BitmapKernels.copy(dict.validity(), out.validity(), n);
    } else {
      Bitmap.fill(out.validity(), n, true);
    }
    VarCharVector v = (VarCharVector) out.vector();
    v.setLastSet(n - 1);
    v.setValueCount(n);
    return v;
  }

  /**
   * A one-row column holding {@code value} (boxed Int/Long/Double/Boolean in Spark's internal
   * representation, or {@code null}). Used to emit ungrouped aggregation buffers.
   */
  public static ColumnVector scalarColumn(
      String name, DataType dt, Object value, BufferAllocator allocator) {
    FieldVector v = newVector(name, dt, allocator);
    v.setInitialCapacity(1);
    v.allocateNew();
    if (value == null) {
      ((org.apache.arrow.vector.BaseFixedWidthVector) v).setNull(0);
    } else if (v instanceof IntVector iv) {
      iv.setSafe(0, ((Number) value).intValue());
    } else if (v instanceof DateDayVector dv) {
      dv.setSafe(0, ((Number) value).intValue());
    } else if (v instanceof BigIntVector lv) {
      lv.setSafe(0, ((Number) value).longValue());
    } else if (v instanceof TimeStampMicroTZVector tv) {
      tv.setSafe(0, ((Number) value).longValue());
    } else if (v instanceof Float8Vector fv) {
      fv.setSafe(0, ((Number) value).doubleValue());
    } else if (v instanceof BitVector bv) {
      bv.setSafe(0, ((Boolean) value) ? 1 : 0);
    } else {
      v.close();
      throw new UnsupportedOperationException("scalar output not supported for " + dt);
    }
    v.setValueCount(1);
    return new VectorArrowColumnVector(v);
  }

  /**
   * Gathers rows {@code idx[from..to)} of {@code in} into a new Arrow vector; an index of {@code -1}
   * becomes a null (outer-join padding). Dictionary-encoded strings are decoded: the output is a
   * plain vector Spark can read, and the gathered rows come from a whole partition whose chunks
   * never shared a dictionary anyway.
   */
  public static ColumnVector gather(
      String name, DataType dt, VectorBuffers in, int[] idx, int from, int to, BufferAllocator allocator) {
    int count = to - from;
    boolean padded = false;
    for (int o = from; o < to && !padded; o++) {
      padded = idx[o] < 0;
    }
    boolean nulls = in.hasNulls() || padded;
    if (in.type() == VecType.UTF8) {
      VectorBuffers plain = in.isDictionaryEncoded() ? decodeDictionary(in) : in;
      long bytes = GatherKernels.gatherUtf8Bytes(plain, idx, from, to);
      ArrowVectorBuffers out = allocateUtf8(name, count, bytes, allocator);
      GatherKernels.gatherUtf8(plain, idx, from, to, out.offsets(), out.data(), nulls ? out.validity() : null);
      return finish(out, count, !nulls);
    }
    ArrowVectorBuffers out = allocateFixed(name, dt, count, allocator);
    GatherKernels.gatherFixed(in, idx, from, to, out.data(), nulls ? out.validity() : null);
    return finish(out, count, !nulls);
  }

  /** A plain UTF8 view of a dictionary-encoded column, built in a temporary arena-free copy. */
  private static VectorBuffers decodeDictionary(VectorBuffers in) {
    // The gather reads a handful of rows out of a whole column; decoding through a scratch
    // ColumnBuilder keeps GatherKernels ignorant of dictionaries.
    java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
    io.sparkvector.kernels.ColumnBuilder b = new io.sparkvector.kernels.ColumnBuilder(arena, VecType.UTF8, in.length());
    b.append(in);
    return b.view();
  }

  /** Copies a whole column (no selection) into a new Arrow vector. */
  public static ColumnVector copy(String name, DataType dt, VectorBuffers in, BufferAllocator allocator) {
    int n = in.length();
    if (in.type() == VecType.UTF8) {
      if (in.isDictionaryEncoded()) {
        return compactDictionary(name, in, null, n, allocator);
      }
      int end = in.offsets().get(VectorBuffers.LE_INT, (long) n << 2);
      ArrowVectorBuffers out = allocateUtf8(name, n, end, allocator);
      MemorySegment.copy(in.offsets(), 0, out.offsets(), 0, ((long) n + 1) << 2);
      MemorySegment.copy(in.data(), 0, out.data(), 0, end);
      if (in.hasNulls()) {
        BitmapKernels.copy(in.validity(), out.validity(), n);
      }
      return finish(out, n, !in.hasNulls());
    }
    ArrowVectorBuffers out = allocateFixed(name, dt, n, allocator);
    if (in.type() == VecType.BOOL) {
      BitmapKernels.copy(in.data(), out.data(), n);
    } else {
      MemorySegment.copy(in.data(), 0, out.data(), 0, (long) n * in.type().byteWidth());
    }
    if (in.hasNulls()) {
      BitmapKernels.copy(in.validity(), out.validity(), n);
    }
    return finish(out, n, !in.hasNulls());
  }

}
