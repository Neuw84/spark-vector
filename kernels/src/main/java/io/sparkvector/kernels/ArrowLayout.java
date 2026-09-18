package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * Allocates and populates Arrow-layout buffers in an {@link Arena}. Used by tests, by the copy
 * adapter for Spark's on-heap vectors, and by kernels that need scratch space.
 *
 * <p>All data buffers are padded to a multiple of 64 bytes so that whole-vector loads at the tail
 * never run past the end of the segment.
 */
public final class ArrowLayout {

  /** Padding granularity for data buffers (matches Arrow's recommended 64-byte alignment). */
  public static final int PAD = 64;

  private ArrowLayout() {}

  public static long padded(long bytes) {
    return Math.max(PAD, ((bytes + PAD - 1) / PAD) * PAD);
  }

  /** Zeroed data buffer for {@code length} elements of a fixed-width type. */
  public static MemorySegment allocateData(Arena arena, VecType type, int length) {
    if (!type.isFixedWidth()) {
      throw new IllegalArgumentException("not fixed width: " + type);
    }
    return arena.allocate(padded((long) length * type.byteWidth()), PAD);
  }

  /** Zeroed validity (or BOOL data) bitmap for {@code length} elements. */
  public static MemorySegment allocateBitmap(Arena arena, int length) {
    return arena.allocate(padded(Bitmap.bytesFor(length)), PAD);
  }

  /** Zeroed offsets buffer for a UTF8 column of {@code length} elements. */
  public static MemorySegment allocateOffsets(Arena arena, int length) {
    return arena.allocate(padded(((long) length + 1) << 2), PAD);
  }

  public static MemorySegment allocateBytes(Arena arena, long bytes) {
    return arena.allocate(padded(bytes), PAD);
  }

  /**
   * Builds a validity bitmap from a nullability array; returns {@code null} when there are no
   * nulls (or when {@code nulls} itself is {@code null}).
   */
  public static MemorySegment validityFrom(Arena arena, boolean[] nulls, int length) {
    if (nulls == null) {
      return null;
    }
    boolean any = false;
    for (int i = 0; i < length; i++) {
      any |= nulls[i];
    }
    if (!any) {
      return null;
    }
    MemorySegment v = allocateBitmap(arena, length);
    for (int i = 0; i < length; i++) {
      if (!nulls[i]) {
        Bitmap.set(v, i);
      }
    }
    return v;
  }

  /**
   * Builds a validity bitmap from a byte-per-row nullability array where a non-zero byte marks a
   * null (the layout Iceberg's {@code NullabilityHolder} keeps instead of an Arrow validity
   * buffer). Bit {@code i} is set when row {@code i} is valid. The caller decides whether the
   * column has nulls at all; this always allocates.
   */
  public static MemorySegment validityFromNullBytes(Arena arena, byte[] isNull, int length) {
    MemorySegment v = allocateBitmap(arena, length);
    int words = Bitmap.wordsFor(length);
    int row = 0;
    for (int w = 0; w < words; w++) {
      int end = Math.min(row + 64, length);
      long word = 0L;
      for (int i = row; i < end; i++) {
        if (isNull[i] == 0) {
          word |= 1L << (i - row);
        }
      }
      Bitmap.setWord(v, w, length, word);
      row = end;
    }
    return v;
  }

  /**
   * Builds a selection bitmap of {@code length} bits with the bits at {@code indices[0..count)}
   * set (the indices are row positions, in any order, each below {@code length}).
   */
  public static MemorySegment selectionFromIndices(Arena arena, int[] indices, int count, int length) {
    MemorySegment s = allocateBitmap(arena, length);
    for (int i = 0; i < count; i++) {
      int idx = indices[i];
      if (idx < 0 || idx >= length) {
        throw new IndexOutOfBoundsException("selected row " + idx + " outside batch of " + length);
      }
      Bitmap.set(s, idx);
    }
    return s;
  }

  public static SegmentVectorBuffers ofInts(Arena arena, int[] values, boolean[] nulls) {
    MemorySegment data = allocateData(arena, VecType.INT32, values.length);
    MemorySegment.copy(values, 0, data, VectorBuffers.LE_INT, 0, values.length);
    return SegmentVectorBuffers.fixedWidth(
        VecType.INT32, values.length, validityFrom(arena, nulls, values.length), data);
  }

  public static SegmentVectorBuffers ofLongs(Arena arena, long[] values, boolean[] nulls) {
    MemorySegment data = allocateData(arena, VecType.INT64, values.length);
    MemorySegment.copy(values, 0, data, VectorBuffers.LE_LONG, 0, values.length);
    return SegmentVectorBuffers.fixedWidth(
        VecType.INT64, values.length, validityFrom(arena, nulls, values.length), data);
  }

  public static SegmentVectorBuffers ofDoubles(Arena arena, double[] values, boolean[] nulls) {
    MemorySegment data = allocateData(arena, VecType.FLOAT64, values.length);
    MemorySegment.copy(values, 0, data, VectorBuffers.LE_DOUBLE, 0, values.length);
    return SegmentVectorBuffers.fixedWidth(
        VecType.FLOAT64, values.length, validityFrom(arena, nulls, values.length), data);
  }

  /** DECIMAL128 column from unscaled values; each must fit in 128 bits. */
  public static SegmentVectorBuffers ofDecimal128(Arena arena, java.math.BigInteger[] values, boolean[] nulls) {
    MemorySegment data = allocateData(arena, VecType.DECIMAL128, values.length);
    for (int i = 0; i < values.length; i++) {
      java.math.BigInteger v = values[i] == null ? java.math.BigInteger.ZERO : values[i];
      if (v.bitLength() > 127) {
        throw new IllegalArgumentException("does not fit in 128 bits: " + v);
      }
      Decimal128.set(data, i, Decimal128.hiOf(v), Decimal128.loOf(v));
    }
    return SegmentVectorBuffers.fixedWidth(
        VecType.DECIMAL128, values.length, validityFrom(arena, nulls, values.length), data);
  }

  public static SegmentVectorBuffers ofBooleans(Arena arena, boolean[] values, boolean[] nulls) {
    MemorySegment data = allocateBitmap(arena, values.length);
    for (int i = 0; i < values.length; i++) {
      if (values[i]) {
        Bitmap.set(data, i);
      }
    }
    return SegmentVectorBuffers.fixedWidth(
        VecType.BOOL, values.length, validityFrom(arena, nulls, values.length), data);
  }

  /**
   * UTF8 column from already encoded bytes: value {@code i} is {@code bytes[offsets[i], offsets[i +
   * 1])}, {@code offsets} has {@code n + 1} entries. Nothing is decoded or re-encoded on the way.
   */
  public static SegmentVectorBuffers ofUtf8(Arena arena, byte[] bytes, int[] offsets, int n, boolean[] nulls) {
    MemorySegment offsetSegment = allocateOffsets(arena, n);
    MemorySegment.copy(offsets, 0, offsetSegment, VectorBuffers.LE_INT, 0, n + 1);
    int total = offsets[n];
    MemorySegment data = allocateBytes(arena, total);
    MemorySegment.copy(bytes, 0, data, ValueLayout.JAVA_BYTE, 0, total);
    return SegmentVectorBuffers.utf8(n, validityFrom(arena, nulls, n), offsetSegment, data);
  }

  /** UTF8 column; {@code null} entries become nulls. */
  public static SegmentVectorBuffers ofStrings(Arena arena, String[] values) {
    int n = values.length;
    byte[][] encoded = new byte[n][];
    boolean[] nulls = new boolean[n];
    long total = 0;
    for (int i = 0; i < n; i++) {
      if (values[i] == null) {
        nulls[i] = true;
        encoded[i] = new byte[0];
      } else {
        encoded[i] = values[i].getBytes(StandardCharsets.UTF_8);
        total += encoded[i].length;
      }
    }
    MemorySegment offsets = allocateOffsets(arena, n);
    MemorySegment data = allocateBytes(arena, total);
    int pos = 0;
    for (int i = 0; i < n; i++) {
      offsets.set(VectorBuffers.LE_INT, (long) i << 2, pos);
      MemorySegment.copy(encoded[i], 0, data, ValueLayout.JAVA_BYTE, pos, encoded[i].length);
      pos += encoded[i].length;
    }
    offsets.set(VectorBuffers.LE_INT, (long) n << 2, pos);
    return SegmentVectorBuffers.utf8(n, validityFrom(arena, nulls, n), offsets, data);
  }
}
