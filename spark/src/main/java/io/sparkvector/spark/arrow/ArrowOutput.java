package io.sparkvector.spark.arrow;

import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.BitmapKernels;
import io.sparkvector.kernels.CompactKernels;
import io.sparkvector.kernels.GatherKernels;
import io.sparkvector.kernels.HeapMirror;
import io.sparkvector.kernels.RunMerge;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.spark.adapter.TypeMapping;
import java.lang.foreign.MemorySegment;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BaseVariableWidthVector;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
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
    if (dt instanceof DecimalType d && d.precision() <= TypeMapping.MAX_DECIMAL_PRECISION) {
      return new BigIntVector(name, allocator); // unscaled values; see VectorDecimalColumnVector
    }
    if (dt instanceof DecimalType d) {
      // A wide decimal (p > 18): Arrow's 128-bit vector over the DECIMAL128 lane's own layout, which
      // Spark's ArrowColumnVector reads through getDecimal (a BigDecimal per row; accepted, #257).
      return new DecimalVector(name, allocator, d.precision(), d.scale());
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
    return ArrowVectorBuffers.forWrite(v, length, dt);
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
    return wrap(v, out.sparkType());
  }

  /** The Spark-facing column over a finished vector: decimals need their own wrapper. */
  public static ColumnVector wrap(FieldVector v, DataType sparkType) {
    if (sparkType instanceof DecimalType d && v instanceof BigIntVector lv) {
      return new VectorDecimalColumnVector(lv, d);
    }
    return new VectorArrowColumnVector(v);
  }

  /** A string column from boxed values ({@code null} entries are nulls): an aggregate buffer over strings. */
  public static ColumnVector utf8Column(String name, byte[][] values, BufferAllocator allocator) {
    long bytes = 0;
    for (byte[] b : values) {
      bytes += b == null ? 0 : b.length;
    }
    ArrowVectorBuffers out = allocateUtf8(name, values.length, bytes, allocator);
    MemorySegment offsets = out.offsets();
    MemorySegment data = out.data();
    MemorySegment validity = out.validity();
    long pos = 0;
    for (int i = 0; i < values.length; i++) {
      offsets.setAtIndex(VectorBuffers.LE_INT, i, (int) pos);
      if (values[i] == null) {
        Bitmap.clear(validity, i);
      } else {
        Bitmap.set(validity, i);
        MemorySegment.copy(MemorySegment.ofArray(values[i]), 0, data, pos, values[i].length);
        pos += values[i].length;
      }
    }
    offsets.setAtIndex(VectorBuffers.LE_INT, values.length, (int) pos);
    return finish(out, values.length, false);
  }

  /**
   * A wide decimal column ({@code p > 18}) from boxed values: {@code null} entries are nulls. Used
   * for the {@code sum} buffer of a decimal aggregate; the values are the exact 128-bit totals.
   */
  public static ColumnVector decimalColumn(
      String name, DecimalType dt, java.math.BigDecimal[] values, BufferAllocator allocator) {
    DecimalVector v = new DecimalVector(name, allocator, dt.precision(), dt.scale());
    v.setInitialCapacity(values.length);
    v.allocateNew();
    for (int i = 0; i < values.length; i++) {
      if (values[i] == null) {
        v.setNull(i);
      } else {
        v.setSafe(i, values[i]);
      }
    }
    v.setValueCount(values.length);
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
    VectorBuffers dict = in.dictionary();
    VarCharVector dictionary = outCount < dict.length()
        ? usedDictionary(name + ".dictionary", dict, out, outCount, allocator)
        : copyDictionary(name + ".dictionary", dict, allocator);
    return new VectorDictionaryColumnVector(indices, dictionary);
  }

  /** Scratch of the used-entries remap, per thread: old id to new id (-1 unused), and the used ids in order. */
  private static final ThreadLocal<int[][]> REMAP = ThreadLocal.withInitial(() -> new int[][] {new int[0], new int[0]});

  /**
   * The dictionary of the entries the compacted ids actually use, the ids rewritten to it (#345). A
   * slice of a few dozen rows out of a dictionary of thousands otherwise carried the whole dictionary
   * -- at 200 shuffle partitions that was ~800 bytes per row on the wire (31x Spark's bytes for
   * customer names) and one dictionary copy per slice in the writer's memory. Null rows get id 0.
   */
  private static VarCharVector usedDictionary(
      String name, VectorBuffers dict, ArrowVectorBuffers ids, int outCount, BufferAllocator allocator) {
    int n = dict.length();
    int[][] scratch = REMAP.get();
    if (scratch[0].length < n) {
      scratch[0] = new int[Math.max(n, 2 * scratch[0].length)];
      java.util.Arrays.fill(scratch[0], -1);
    }
    if (scratch[1].length < outCount) {
      scratch[1] = new int[Math.max(outCount, 2 * scratch[1].length)];
    }
    int[] remap = scratch[0];
    int[] used = scratch[1];
    MemorySegment idData = ids.data();
    MemorySegment idValidity = ids.validity();
    MemorySegment offsets = dict.offsets();
    int next = 0;
    long bytes = 0;
    for (int i = 0; i < outCount; i++) {
      if (!Bitmap.isSet(idValidity, i)) {
        idData.set(VectorBuffers.LE_INT, (long) i << 2, 0);
        continue;
      }
      int id = idData.get(VectorBuffers.LE_INT, (long) i << 2);
      int mapped = remap[id];
      if (mapped < 0) {
        mapped = next++;
        remap[id] = mapped;
        used[mapped] = id;
        bytes += offsets.get(VectorBuffers.LE_INT, (long) (id + 1) << 2) - offsets.get(VectorBuffers.LE_INT, (long) id << 2);
      }
      idData.set(VectorBuffers.LE_INT, (long) i << 2, mapped);
    }
    ArrowVectorBuffers out = allocateUtf8(name, next, bytes, allocator);
    MemorySegment outOffsets = out.offsets();
    MemorySegment outData = out.data();
    boolean dictNulls = dict.hasNulls();
    int pos = 0;
    outOffsets.set(VectorBuffers.LE_INT, 0, 0);
    for (int k = 0; k < next; k++) {
      int id = used[k];
      remap[id] = -1; // the scratch back to unused for the next slice
      int start = offsets.get(VectorBuffers.LE_INT, (long) id << 2);
      int len = offsets.get(VectorBuffers.LE_INT, (long) (id + 1) << 2) - start;
      MemorySegment.copy(dict.data(), start, outData, pos, len);
      pos += len;
      outOffsets.set(VectorBuffers.LE_INT, (long) (k + 1) << 2, pos);
      if (dictNulls) {
        Bitmap.setTo(out.validity(), k, Bitmap.isSet(dict.validity(), id));
      }
    }
    if (!dictNulls) {
      Bitmap.fill(out.validity(), next, true);
    }
    VarCharVector v = (VarCharVector) out.vector();
    v.setLastSet(next - 1);
    v.setValueCount(next);
    return v;
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
      if (v instanceof BaseVariableWidthVector vv) {
        vv.setNull(0);
      } else {
        ((org.apache.arrow.vector.BaseFixedWidthVector) v).setNull(0);
      }
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
    } else if (v instanceof DecimalVector dv) {
      dv.setSafe(0, (java.math.BigDecimal) value);
    } else if (v instanceof VarCharVector sv) {
      sv.setSafe(0, ((org.apache.spark.unsafe.types.UTF8String) value).getBytes());
    } else {
      v.close();
      throw new UnsupportedOperationException("scalar output not supported for " + dt);
    }
    v.setValueCount(1);
    return wrap(v, dt);
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
      if (in.isDictionaryEncoded()) {
        // Gather through the codes: output row o is dictionary entry code(idx[o]). Decoding the whole
        // column first cost a string append per input row -- 20x the join on a shuffled dictionary column.
        int[] codes = new int[count];
        boolean anyNull = padded;
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          if (i < 0 || in.isNull(i)) {
            codes[o] = -1;
            anyNull = true;
          } else {
            codes[o] = in.getInt(i);
          }
        }
        VectorBuffers dict = in.dictionary();
        long bytes = GatherKernels.gatherUtf8Bytes(dict, codes, 0, count);
        ArrowVectorBuffers out = allocateUtf8(name, count, bytes, allocator);
        GatherKernels.gatherUtf8(dict, codes, 0, count, out.offsets(), out.data(), anyNull ? out.validity() : null);
        return finish(out, count, !anyNull);
      }
      long bytes = GatherKernels.gatherUtf8Bytes(in, idx, from, to);
      ArrowVectorBuffers out = allocateUtf8(name, count, bytes, allocator);
      GatherKernels.gatherUtf8(in, idx, from, to, out.offsets(), out.data(), nulls ? out.validity() : null);
      return finish(out, count, !nulls);
    }
    ArrowVectorBuffers out = allocateFixed(name, dt, count, allocator);
    GatherKernels.gatherFixed(in, idx, from, to, out.data(), nulls ? out.validity() : null);
    return finish(out, count, !nulls);
  }


  /**
   * Gathers rows {@code idx[from..to)} of a heap-mirrored fixed-width column into a new Arrow vector
   * (#332): the pair-wise reads are array reads, the result one bulk copy. A negative index pads a
   * null row.
   */
  public static ColumnVector gatherHeap(
      String name, DataType dt, HeapMirror in, int[] idx, int from, int to, BufferAllocator allocator,
      HeapMirror.GatherScratch scratch) {
    int count = to - from;
    boolean padded = false;
    for (int o = from; o < to && !padded; o++) {
      padded = idx[o] < 0;
    }
    boolean nulls = in.validity != null || padded;
    ArrowVectorBuffers out = allocateFixed(name, dt, count, allocator);
    in.gather(idx, from, to, out.data(), nulls ? out.validity() : null, scratch);
    return finish(out, count, !nulls);
  }
  /**
   * Gathers {@code count} rows from several sorted runs into a new Arrow vector: output row {@code o}
   * is row {@code rowOf[o]} of {@code runs[runOf[o]]} (the sort's k-way merge, #285). UTF8 runs must
   * be plain: the sort decodes a dictionary once when it seals the run.
   */
  public static ColumnVector gatherRuns(
      String name, DataType dt, VectorBuffers[] runs, int[] runOf, int[] rowOf, int count, BufferAllocator allocator) {
    boolean nulls = false;
    for (VectorBuffers run : runs) {
      nulls |= run.hasNulls();
    }
    if (runs[0].type() == VecType.UTF8) {
      long bytes = RunMerge.gatherUtf8Bytes(runs, runOf, rowOf, count);
      ArrowVectorBuffers out = allocateUtf8(name, count, bytes, allocator);
      RunMerge.gatherUtf8(runs, runOf, rowOf, count, out.offsets(), out.data(), nulls ? out.validity() : null);
      return finish(out, count, !nulls);
    }
    ArrowVectorBuffers out = allocateFixed(name, dt, count, allocator);
    RunMerge.gatherFixed(runs, runOf, rowOf, count, out.data(), nulls ? out.validity() : null);
    return finish(out, count, !nulls);
  }

  /** A plain UTF8 copy of a dictionary-encoded column in the given arena (the sort decodes a run once, #285). */
  public static VectorBuffers decodeDictionary(VectorBuffers in, java.lang.foreign.Arena arena) {
    io.sparkvector.kernels.ColumnBuilder b = new io.sparkvector.kernels.ColumnBuilder(arena, VecType.UTF8, in.length());
    b.append(in);
    return b.view();
  }

  /** A column of {@code length} copies of a non-null literal (Spark's internal representation). */
  public static ColumnVector constant(String name, DataType dt, Object value, int length, BufferAllocator allocator) {
    if (dt instanceof StringType) {
      byte[] bytes = ((org.apache.spark.unsafe.types.UTF8String) value).getBytes();
      ArrowVectorBuffers out = allocateUtf8(name, length, (long) bytes.length * length, allocator);
      MemorySegment offsets = out.offsets();
      MemorySegment data = out.data();
      MemorySegment src = MemorySegment.ofArray(bytes);
      long pos = 0;
      for (int i = 0; i < length; i++) {
        offsets.setAtIndex(VectorBuffers.LE_INT, i, (int) pos);
        MemorySegment.copy(src, 0, data, pos, bytes.length);
        pos += bytes.length;
      }
      offsets.setAtIndex(VectorBuffers.LE_INT, length, (int) pos);
      return finish(out, length, true);
    }
    ArrowVectorBuffers out = allocateFixed(name, dt, length, allocator);
    MemorySegment data = out.data();
    switch (out.type()) {
      case INT32 -> {
        int v = ((Number) value).intValue();
        for (int i = 0; i < length; i++) {
          data.setAtIndex(VectorBuffers.LE_INT, i, v);
        }
      }
      case INT64 -> {
        long v = value instanceof org.apache.spark.sql.types.Decimal d ? d.toUnscaledLong() : ((Number) value).longValue();
        for (int i = 0; i < length; i++) {
          data.setAtIndex(VectorBuffers.LE_LONG, i, v);
        }
      }
      case FLOAT64 -> {
        double v = ((Number) value).doubleValue();
        for (int i = 0; i < length; i++) {
          data.setAtIndex(VectorBuffers.LE_DOUBLE, i, v);
        }
      }
      case BOOL -> Bitmap.fill(data, length, (Boolean) value);
      case DECIMAL128 -> {
        // A wide decimal literal (an expand's constant slot, #259): the unscaled value as two limbs per row.
        java.math.BigInteger u = ((org.apache.spark.sql.types.Decimal) value).toJavaBigDecimal().unscaledValue();
        long hi = io.sparkvector.kernels.Decimal128.hiOf(u);
        long lo = io.sparkvector.kernels.Decimal128.loOf(u);
        for (int i = 0; i < length; i++) {
          io.sparkvector.kernels.Decimal128.set(data, i, hi, lo);
        }
      }
      default -> throw new UnsupportedOperationException("constant column of " + dt);
    }
    return finish(out, length, true);
  }

  /**
   * A column of {@code length} nulls. Used for the streamed side of the build rows a full outer
   * join emits after the last streamed batch, where there is no input column to gather from.
   */
  public static ColumnVector nulls(String name, DataType dt, int length, BufferAllocator allocator) {
    ArrowVectorBuffers out =
        TypeMapping.vecTypeOf(dt) == VecType.UTF8
            ? allocateUtf8(name, length, 0, allocator)
            : allocateFixed(name, dt, length, allocator);
    Bitmap.fill(out.validity(), length, false);
    if (out.type() == VecType.UTF8) {
      out.offsets().asSlice(0, ((long) length + 1) << 2).fill((byte) 0);
    }
    return finish(out, length, false);
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
