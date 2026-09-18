package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Gathers rows of a column by index: {@code out[o] = in[idx[from + o]]}. An index of {@code -1}
 * produces a null (the way an outer join pads unmatched rows), so {@code outValidity} must be
 * given whenever the input has nulls or indices may be negative; it is written completely.
 *
 * <p>A permutation of arbitrary indices has no SIMD form worth the name on any current ISA (the
 * Vector API's gather is array-only and not faster than a scalar loop on NEON), so these are
 * straight loops over the segments. They are what a sort and a hash join spend their time on
 * after the ordering or the matches have been computed, so keep them tight: no per-element
 * allocation, dictionary indices moved instead of strings.
 */
public final class GatherKernels {

  private GatherKernels() {}

  /**
   * Gathers a fixed-width or BOOL column (or the int32 indices of a dictionary-encoded one).
   */
  public static void gatherFixed(
      VectorBuffers in, int[] idx, int from, int to, MemorySegment outData, MemorySegment outValidity) {
    VecType type = in.isDictionaryEncoded() ? VecType.INT32 : in.type();
    MemorySegment data = in.data();
    int count = to - from;
    switch (type) {
      case INT32 -> {
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          outData.set(VectorBuffers.LE_INT, (long) o << 2, i < 0 ? 0 : data.get(VectorBuffers.LE_INT, (long) i << 2));
        }
      }
      case INT64, FLOAT64 -> {
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          outData.set(VectorBuffers.LE_LONG, (long) o << 3, i < 0 ? 0L : data.get(VectorBuffers.LE_LONG, (long) i << 3));
        }
      }
      case BOOL -> gatherBits(data, idx, from, to, outData, false);
      case DECIMAL128 -> {
        // Two limbs per value; a padded (-1) index leaves the slot zero like the other lanes.
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          if (i < 0) {
            Decimal128.set(outData, o, 0L, 0L);
          } else {
            Decimal128.copy(data, i, outData, o);
          }
        }
      }
      default -> throw new IllegalArgumentException("not fixed width: " + type);
    }
    if (outValidity != null) {
      gatherValidity(in.validity(), idx, from, to, outValidity);
    }
  }

  /** Validity bits of the gathered rows: set unless the row is null or the index is -1. */
  public static void gatherValidity(MemorySegment validity, int[] idx, int from, int to, MemorySegment outValidity) {
    int count = to - from;
    if (validity == null) {
      for (int base = 0; base < count; base += 64) {
        int limit = Math.min(64, count - base);
        long word = 0L;
        for (int j = 0; j < limit; j++) {
          if (idx[from + base + j] >= 0) {
            word |= 1L << j;
          }
        }
        Bitmap.setWord(outValidity, base >>> 6, count, word);
      }
    } else {
      gatherBits(validity, idx, from, to, outValidity, false);
    }
  }

  /** Gathers bits; a negative index yields {@code padValue}. */
  static void gatherBits(MemorySegment bits, int[] idx, int from, int to, MemorySegment out, boolean padValue) {
    int count = to - from;
    for (int base = 0; base < count; base += 64) {
      int limit = Math.min(64, count - base);
      long word = 0L;
      for (int j = 0; j < limit; j++) {
        int i = idx[from + base + j];
        boolean bit = i < 0 ? padValue : Bitmap.isSet(bits, i);
        if (bit) {
          word |= 1L << j;
        }
      }
      Bitmap.setWord(out, base >>> 6, count, word);
    }
  }

  /** Bytes needed by {@link #gatherUtf8} for the given rows of a plain UTF8 column. */
  public static long gatherUtf8Bytes(VectorBuffers in, int[] idx, int from, int to) {
    if (in.type() != VecType.UTF8 || in.isDictionaryEncoded()) {
      throw new IllegalArgumentException("expected plain UTF8");
    }
    MemorySegment off = in.offsets();
    long total = 0;
    for (int o = from; o < to; o++) {
      int i = idx[o];
      if (i >= 0 && !in.isNull(i)) {
        total += off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) i << 2);
      }
    }
    return total;
  }

  /** Gathers a plain UTF8 column; {@code outData} must hold {@link #gatherUtf8Bytes}. */
  public static void gatherUtf8(
      VectorBuffers in, int[] idx, int from, int to,
      MemorySegment outOffsets, MemorySegment outData, MemorySegment outValidity) {
    if (in.type() != VecType.UTF8 || in.isDictionaryEncoded()) {
      throw new IllegalArgumentException("expected plain UTF8");
    }
    MemorySegment off = in.offsets();
    MemorySegment data = in.data();
    int count = to - from;
    int pos = 0;
    for (int o = 0; o < count; o++) {
      int i = idx[from + o];
      outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
      if (i >= 0 && !in.isNull(i)) {
        int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
        int len = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - start;
        ByteCopy.copy(data, start, outData, pos, len);
        pos += len;
      }
    }
    outOffsets.set(VectorBuffers.LE_INT, (long) count << 2, pos);
    if (outValidity != null) {
      gatherValidity(in.validity(), idx, from, to, outValidity);
    }
  }
}
