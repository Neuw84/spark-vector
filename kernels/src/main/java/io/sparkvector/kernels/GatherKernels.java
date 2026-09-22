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
    if (outValidity == null) {
      // No validity asked for: the input has no nulls and no index is padded -- the plain loop.
      gatherFixedPlain(type, in.data(), idx, from, to, outData);
    } else {
      gatherFixed(type, in.data(), idx, from, to, outData);
      gatherValidity(in.validity(), idx, from, to, outValidity);
    }
  }

  /** The data lanes of real indices only (no {@code -1}): the loop is a load and a store per row. */
  public static void gatherFixedPlain(VecType type, MemorySegment data, int[] idx, int from, int to, MemorySegment outData) {
    int count = to - from;
    switch (type) {
      case INT32 -> {
        for (int o = 0; o < count; o++) {
          outData.set(VectorBuffers.LE_INT, (long) o << 2, data.get(VectorBuffers.LE_INT, (long) idx[from + o] << 2));
        }
      }
      case INT64, FLOAT64 -> {
        for (int o = 0; o < count; o++) {
          outData.set(VectorBuffers.LE_LONG, (long) o << 3, data.get(VectorBuffers.LE_LONG, (long) idx[from + o] << 3));
        }
      }
      case BOOL -> gatherBits(data, idx, from, to, outData, false);
      case DECIMAL128 -> {
        for (int o = 0; o < count; o++) {
          Decimal128.copy(data, idx[from + o], outData, o);
        }
      }
      default -> throw new IllegalArgumentException("not fixed width: " + type);
    }
  }

  /**
   * The data lanes alone, from a segment, indices possibly padded ({@code -1}). The loops carry no
   * data-dependent branch: a padded index reads slot 0 and masks the value to zero (#416 -- the
   * {@code i < 0} branch, never taken in a sort's gathers and taken in an outer join's, had C2
   * speculate on it and deoptimise the kernel forty times in q18's first iteration, every task on the
   * executor running it interpreted until the recompile; the same for the null branch of the
   * dictionary decode). Gathers that produce no validity take {@link #gatherFixedPlain}, whose loops
   * are the unmasked load and store.
   */
  public static void gatherFixed(VecType type, MemorySegment data, int[] idx, int from, int to, MemorySegment outData) {
    int count = to - from;
    switch (type) {
      case INT32 -> {
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          int keep = ~(i >> 31); // all ones for a real index, zero for -1
          outData.set(VectorBuffers.LE_INT, (long) o << 2, data.get(VectorBuffers.LE_INT, (long) (i & keep) << 2) & keep);
        }
      }
      case INT64, FLOAT64 -> {
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          long keep = ~(long) (i >> 31);
          outData.set(VectorBuffers.LE_LONG, (long) o << 3, data.get(VectorBuffers.LE_LONG, (long) (i & (int) keep) << 3) & keep);
        }
      }
      case BOOL -> gatherBits(data, idx, from, to, outData, false);
      case DECIMAL128 -> {
        for (int o = 0; o < count; o++) {
          int i = idx[from + o];
          long keep = ~(long) (i >> 31);
          long src = (long) (i & (int) keep) << 4;
          long dst = (long) o << 4;
          outData.set(VectorBuffers.LE_LONG, dst, data.get(VectorBuffers.LE_LONG, src) & keep);
          outData.set(VectorBuffers.LE_LONG, dst + 8, data.get(VectorBuffers.LE_LONG, src + 8) & keep);
        }
      }
      default -> throw new IllegalArgumentException("not fixed width: " + type);
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
          word |= ((~idx[from + base + j] >>> 31) & 1L) << j;
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
    int pad = padValue ? 1 : 0;
    for (int base = 0; base < count; base += 64) {
      int limit = Math.min(64, count - base);
      long word = 0L;
      for (int j = 0; j < limit; j++) {
        int i = idx[from + base + j];
        int neg = i >>> 31; // 1 for a padded index
        int ci = i & ~(i >> 31);
        int bit = ((bits.get(ValueLayout.JAVA_BYTE, ci >>> 3) >>> (ci & 7)) & 1 & (neg ^ 1)) | (neg & pad);
        word |= (long) bit << j;
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
    MemorySegment validity = in.validity();
    long total = 0;
    for (int o = from; o < to; o++) {
      int i = idx[o];
      if (i >= 0 && (validity == null || Bitmap.isSet(validity, i))) {
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
    // The validity as a local segment: `in.isNull(i)` per row is an interface call whose receiver
    // profile mixes every buffer implementation, so it stays virtual and repeats the segment checks
    // (#377: 3.5x on this loop at SF10).
    MemorySegment validity = in.validity();
    int count = to - from;
    int pos = 0;
    for (int o = 0; o < count; o++) {
      int i = idx[from + o];
      outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
      if (i >= 0 && (validity == null || Bitmap.isSet(validity, i))) {
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
