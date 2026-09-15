package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Row hashing for group keys: a 32-bit multiplicative mix per key column, combined column by
 * column into an {@code int[]} of row hashes. Fixed-width columns are mixed with {@link IntVector}
 * lanes (32-bit multiply is native everywhere, unlike 64-bit lane multiply on NEON); rows in
 * blocks that contain nulls, and string or boolean columns, use the scalar {@link #mix32} so both
 * paths are bit-identical.
 */
public final class HashKernels {

  static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
  static final VectorSpecies<Long> L = LongVector.SPECIES_PREFERRED;
  static final VectorSpecies<Integer> IH = CastKernels.IH;
  static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

  public static final int SEED = 0x2545F491;
  static final int MULT = 0x9E3779B1;
  /** Mixed in for a null key so that null and 0 land in different groups. */
  static final int NULL_MARK = 0x7F4A7C15;

  private HashKernels() {}

  public static int mix32(int h, int v) {
    h = (h ^ v) * MULT;
    return h ^ (h >>> 15);
  }

  /** Final avalanche before probing (Murmur3 fmix32). */
  public static int finish(int h) {
    h ^= h >>> 16;
    h *= 0x85EBCA6B;
    h ^= h >>> 13;
    h *= 0xC2B2AE35;
    h ^= h >>> 16;
    return h;
  }

  public static void init(int[] hashes, int n) {
    java.util.Arrays.fill(hashes, 0, n, SEED);
  }

  /** Mixes one key column into the row hashes. */
  public static void mixColumn(VectorBuffers col, int[] hashes) {
    int n = col.length();
    if (col.isDictionaryEncoded()) {
      mixDictionary(col, hashes, n);
      return;
    }
    switch (col.type()) {
      case INT32 -> mixInt32(col, hashes, n);
      case INT64, FLOAT64 -> mixInt64(col, hashes, n);
      case BOOL -> {
        for (int i = 0; i < n; i++) {
          hashes[i] = mix32(hashes[i], col.isNull(i) ? NULL_MARK : (col.getBoolean(i) ? 1 : 0));
        }
      }
      case UTF8 -> mixUtf8(col, hashes, n);
      default -> throw new IllegalArgumentException("unsupported key type " + col.type());
    }
  }

  static void mixInt32(VectorBuffers col, int[] hashes, int n) {
    MemorySegment d = col.data();
    MemorySegment validity = col.validity();
    int lanes = I.length();
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      int base = w << 6;
      int limit = Math.min(64, n - base);
      long word = validity == null ? -1L : Bitmap.wordAt(validity, w, n);
      boolean full = limit == 64 ? word == -1L : word == ((1L << limit) - 1);
      if (full) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          int off = base + k;
          IntVector h = IntVector.fromArray(I, hashes, off);
          IntVector v = IntVector.fromMemorySegment(I, d, (long) off << 2, LE);
          h = h.lanewise(VectorOperators.XOR, v).mul(MULT);
          h = h.lanewise(VectorOperators.XOR, h.lanewise(VectorOperators.LSHR, 15));
          h.intoArray(hashes, off);
        }
        for (; k < limit; k++) {
          int i = base + k;
          hashes[i] = mix32(hashes[i], col.getInt(i));
        }
      } else {
        for (int k = 0; k < limit; k++) {
          int i = base + k;
          hashes[i] = mix32(hashes[i], ((word >>> k) & 1L) == 0 ? NULL_MARK : col.getInt(i));
        }
      }
    }
  }

  /** 64-bit keys (and raw double bits) are folded to 32 bits with an xor-shift before mixing. */
  static void mixInt64(VectorBuffers col, int[] hashes, int n) {
    MemorySegment d = col.data();
    MemorySegment validity = col.validity();
    int lanes = L.length(); // == IH.length()
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      int base = w << 6;
      int limit = Math.min(64, n - base);
      long word = validity == null ? -1L : Bitmap.wordAt(validity, w, n);
      boolean full = limit == 64 ? word == -1L : word == ((1L << limit) - 1);
      if (full) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          int off = base + k;
          LongVector v = LongVector.fromMemorySegment(L, d, (long) off << 3, LE);
          v = v.lanewise(VectorOperators.XOR, v.lanewise(VectorOperators.LSHR, 32));
          IntVector folded = v.convertShape(VectorOperators.L2I, IH, 0).reinterpretAsInts();
          IntVector h = IntVector.fromArray(IH, hashes, off);
          h = h.lanewise(VectorOperators.XOR, folded).mul(MULT);
          h = h.lanewise(VectorOperators.XOR, h.lanewise(VectorOperators.LSHR, 15));
          h.intoArray(hashes, off);
        }
        for (; k < limit; k++) {
          int i = base + k;
          hashes[i] = mix32(hashes[i], fold(col.getLong(i)));
        }
      } else {
        for (int k = 0; k < limit; k++) {
          int i = base + k;
          hashes[i] = mix32(hashes[i], ((word >>> k) & 1L) == 0 ? NULL_MARK : fold(col.getLong(i)));
        }
      }
    }
  }

  public static int fold(long v) {
    return (int) (v ^ (v >>> 32));
  }

  /** Hash of a UTF-8 byte range, used for both plain and dictionary strings. */
  public static int hashBytes(MemorySegment data, long start, int len) {
    int h = 0x1B873593;
    long end = start + len;
    long p = start;
    for (; p + 4 <= end; p += 4) {
      h = mix32(h, data.get(VectorBuffers.LE_INT, p));
    }
    int tail = 0;
    for (int shift = 0; p < end; p++, shift += 8) {
      tail |= (data.get(ValueLayout.JAVA_BYTE, p) & 0xFF) << shift;
    }
    return mix32(h, tail ^ len);
  }

  static void mixUtf8(VectorBuffers col, int[] hashes, int n) {
    MemorySegment off = col.offsets();
    MemorySegment data = col.data();
    for (int i = 0; i < n; i++) {
      if (col.isNull(i)) {
        hashes[i] = mix32(hashes[i], NULL_MARK);
      } else {
        int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
        int len = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - start;
        hashes[i] = mix32(hashes[i], hashBytes(data, start, len));
      }
    }
  }

  /** Dictionary strings: hash each dictionary entry once, then gather per row. */
  static void mixDictionary(VectorBuffers col, int[] hashes, int n) {
    VectorBuffers dict = col.dictionary();
    int[] dictHashes = new int[dict.length()];
    MemorySegment off = dict.offsets();
    for (int k = 0; k < dictHashes.length; k++) {
      int start = off.get(VectorBuffers.LE_INT, (long) k << 2);
      int len = off.get(VectorBuffers.LE_INT, (long) (k + 1) << 2) - start;
      dictHashes[k] = dict.isNull(k) ? NULL_MARK : hashBytes(dict.data(), start, len);
    }
    for (int i = 0; i < n; i++) {
      hashes[i] = mix32(hashes[i], col.isNull(i) ? NULL_MARK : dictHashes[col.getInt(i)]);
    }
  }
}
