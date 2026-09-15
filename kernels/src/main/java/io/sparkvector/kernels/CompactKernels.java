package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * Compaction: copy the elements selected by a bitmap into a dense output column.
 *
 * <p>Fixed-width data uses {@code Vector.compress(mask)} per lane group; bitmaps (validity and
 * BOOL values) use {@link Long#compress(long, long)} on whole 64-bit words, which is the same
 * PEXT-style operation at bit granularity.
 */
public final class CompactKernels {

  private static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Long> L = LongVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Double> D = DoubleVector.SPECIES_PREFERRED;
  private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

  /**
   * {@code Vector.compress} is only a native instruction on AVX-512 and SVE; on NEON and AVX2 the
   * JDK falls back to scalar code. For species of up to 8 lanes we instead index a precomputed
   * table of {@link VectorShuffle}s by the selection bits and use {@code rearrange}, which every
   * platform implements with a single permute. Wider species (16 int lanes = AVX-512) use compress.
   */
  private static final int MAX_TABLE_LANES = 8;

  private static final VectorShuffle<Integer>[] I_SHUFFLES = shuffles(I);
  private static final VectorShuffle<Long>[] L_SHUFFLES = shuffles(L);
  private static final VectorShuffle<Double>[] D_SHUFFLES = shuffles(D);

  @SuppressWarnings("unchecked")
  private static <E> VectorShuffle<E>[] shuffles(VectorSpecies<E> species) {
    int lanes = species.length();
    if (lanes > MAX_TABLE_LANES) {
      return null;
    }
    VectorShuffle<E>[] table = (VectorShuffle<E>[]) new VectorShuffle<?>[1 << lanes];
    int[] idx = new int[lanes];
    for (int bits = 0; bits < table.length; bits++) {
      int j = 0;
      for (int lane = 0; lane < lanes; lane++) {
        if ((bits & (1 << lane)) != 0) {
          idx[j++] = lane;
        }
      }
      for (; j < lanes; j++) {
        idx[j] = 0;
      }
      table[bits] = VectorShuffle.fromArray(species, idx, 0);
    }
    return table;
  }

  private CompactKernels() {}

  public static int selectedCount(MemorySegment selection, int n) {
    return Bitmap.popcount(selection, n);
  }

  /**
   * Compacts a fixed-width or BOOL column (or the int32 indices of a dictionary-encoded column).
   * {@code outValidity} must be non-null iff the input has nulls; {@code outCount} is the number of
   * selected elements (see {@link #selectedCount}).
   */
  public static void compactFixed(
      VectorBuffers in,
      MemorySegment selection,
      int outCount,
      MemorySegment outData,
      MemorySegment outValidity) {
    int n = in.length();
    VecType type = in.isDictionaryEncoded() ? VecType.INT32 : in.type();
    switch (type) {
      case INT32 -> compactInt32(in.data(), n, selection, outData);
      case INT64 -> compactInt64(in.data(), n, selection, outData);
      case FLOAT64 -> compactFloat64(in.data(), n, selection, outData);
      case BOOL -> compactBits(in.data(), n, selection, outData, outCount);
      default -> throw new IllegalArgumentException("not fixed width: " + type);
    }
    if (in.validity() != null) {
      if (outValidity == null) {
        throw new IllegalArgumentException("input has nulls but no output validity given");
      }
      compactBits(in.validity(), n, selection, outValidity, outCount);
    }
  }

  static void compactInt32(MemorySegment data, int n, MemorySegment sel, MemorySegment out) {
    int lanes = I.length();
    long laneMask = lanes == 64 ? -1L : (1L << lanes) - 1;
    long outCap = out.byteSize() >>> 2;
    int o = 0;
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      long word = Bitmap.wordAt(sel, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      for (int k = 0; k < limit; k += lanes) {
        long bits = (word >>> k) & laneMask;
        if (bits == 0L) {
          continue;
        }
        long off = (long) (base + k) << 2;
        IntVector v =
            k + lanes <= limit
                ? IntVector.fromMemorySegment(I, data, off, LE)
                : IntVector.fromMemorySegment(I, data, off, LE, I.indexInRange(k, limit));
        IntVector c =
            I_SHUFFLES != null
                ? v.rearrange(I_SHUFFLES[(int) bits])
                : v.compress(VectorMask.fromLong(I, bits));
        int count = Long.bitCount(bits);
        if (o + lanes <= outCap) {
          c.intoMemorySegment(out, (long) o << 2, LE);
        } else {
          c.intoMemorySegment(out, (long) o << 2, LE, I.indexInRange(0, count));
        }
        o += count;
      }
    }
  }

  static void compactInt64(MemorySegment data, int n, MemorySegment sel, MemorySegment out) {
    int lanes = L.length();
    long laneMask = (1L << lanes) - 1;
    long outCap = out.byteSize() >>> 3;
    int o = 0;
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      long word = Bitmap.wordAt(sel, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      for (int k = 0; k < limit; k += lanes) {
        long bits = (word >>> k) & laneMask;
        if (bits == 0L) {
          continue;
        }
        long off = (long) (base + k) << 3;
        LongVector v =
            k + lanes <= limit
                ? LongVector.fromMemorySegment(L, data, off, LE)
                : LongVector.fromMemorySegment(L, data, off, LE, L.indexInRange(k, limit));
        LongVector c =
            L_SHUFFLES != null
                ? v.rearrange(L_SHUFFLES[(int) bits])
                : v.compress(VectorMask.fromLong(L, bits));
        int count = Long.bitCount(bits);
        if (o + lanes <= outCap) {
          c.intoMemorySegment(out, (long) o << 3, LE);
        } else {
          c.intoMemorySegment(out, (long) o << 3, LE, L.indexInRange(0, count));
        }
        o += count;
      }
    }
  }

  static void compactFloat64(MemorySegment data, int n, MemorySegment sel, MemorySegment out) {
    int lanes = D.length();
    long laneMask = (1L << lanes) - 1;
    long outCap = out.byteSize() >>> 3;
    int o = 0;
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      long word = Bitmap.wordAt(sel, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      for (int k = 0; k < limit; k += lanes) {
        long bits = (word >>> k) & laneMask;
        if (bits == 0L) {
          continue;
        }
        long off = (long) (base + k) << 3;
        DoubleVector v =
            k + lanes <= limit
                ? DoubleVector.fromMemorySegment(D, data, off, LE)
                : DoubleVector.fromMemorySegment(D, data, off, LE, D.indexInRange(k, limit));
        DoubleVector c =
            D_SHUFFLES != null
                ? v.rearrange(D_SHUFFLES[(int) bits])
                : v.compress(VectorMask.fromLong(D, bits));
        int count = Long.bitCount(bits);
        if (o + lanes <= outCap) {
          c.intoMemorySegment(out, (long) o << 3, LE);
        } else {
          c.intoMemorySegment(out, (long) o << 3, LE, D.indexInRange(0, count));
        }
        o += count;
      }
    }
  }

  /** Compacts a bitmap by a selection bitmap using {@code Long.compress} per word. */
  static void compactBits(
      MemorySegment bits, int n, MemorySegment sel, MemorySegment out, int outCount) {
    BitWriter writer = new BitWriter(out, outCount);
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      long s = Bitmap.wordAt(sel, w, n);
      if (s == 0L) {
        continue;
      }
      writer.append(Long.compress(Bitmap.wordAt(bits, w, n), s), Long.bitCount(s));
    }
    writer.finish();
  }

  /** Appends variable-length bit runs into an output bitmap. */
  static final class BitWriter {
    private final MemorySegment out;
    private final int totalBits;
    private long acc;
    private int accBits;
    private int wordIndex;

    BitWriter(MemorySegment out, int totalBits) {
      this.out = out;
      this.totalBits = totalBits;
    }

    void append(long value, int count) {
      if (count == 0) {
        return;
      }
      acc |= value << accBits;
      if (accBits + count >= 64) {
        Bitmap.setWord(out, wordIndex++, totalBits, acc);
        int spill = accBits + count - 64;
        acc = accBits == 0 ? 0L : value >>> (64 - accBits);
        if (spill == 0) {
          acc = 0L;
        }
        accBits = spill;
      } else {
        accBits += count;
      }
    }

    void finish() {
      if (accBits > 0) {
        Bitmap.setWord(out, wordIndex, totalBits, acc);
      }
    }
  }

  // ------------------------------------------------------------------ UTF8

  /** Bytes needed for {@code outData} when compacting a plain UTF8 column. */
  public static long selectedUtf8Bytes(VectorBuffers in, MemorySegment selection) {
    if (in.type() != VecType.UTF8 || in.isDictionaryEncoded()) {
      throw new IllegalArgumentException("expected plain UTF8");
    }
    MemorySegment off = in.offsets();
    MemorySegment validity = in.validity();
    int n = in.length();
    long total = 0;
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      long s = Bitmap.wordAt(selection, w, n);
      if (validity != null) {
        s &= Bitmap.wordAt(validity, w, n);
      }
      while (s != 0L) {
        int i = (w << 6) + Long.numberOfTrailingZeros(s);
        s &= s - 1;
        total += off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2)
            - off.get(VectorBuffers.LE_INT, (long) i << 2);
      }
    }
    return total;
  }

  /**
   * Compacts a plain UTF8 column. Variable-length bytes are copied element by element;
   * {@code outData} must hold at least {@link #selectedUtf8Bytes} bytes.
   */
  public static void compactUtf8(
      VectorBuffers in,
      MemorySegment selection,
      int outCount,
      MemorySegment outOffsets,
      MemorySegment outData,
      MemorySegment outValidity) {
    if (in.type() != VecType.UTF8 || in.isDictionaryEncoded()) {
      throw new IllegalArgumentException("expected plain UTF8");
    }
    MemorySegment off = in.offsets();
    MemorySegment validity = in.validity();
    MemorySegment data = in.data();
    int n = in.length();
    int o = 0;
    int pos = 0;
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      long s = Bitmap.wordAt(selection, w, n);
      while (s != 0L) {
        int i = (w << 6) + Long.numberOfTrailingZeros(s);
        s &= s - 1;
        outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
        if (validity == null || Bitmap.isSet(validity, i)) {
          int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
          int len = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - start;
          MemorySegment.copy(data, ValueLayout.JAVA_BYTE, start, outData, ValueLayout.JAVA_BYTE, pos, len);
          pos += len;
        }
        o++;
      }
    }
    outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
    if (validity != null) {
      compactBits(validity, n, selection, outValidity, outCount);
    }
  }
}
