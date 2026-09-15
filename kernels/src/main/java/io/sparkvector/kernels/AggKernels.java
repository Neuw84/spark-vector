package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Whole-column reductions for ungrouped aggregation. Nulls are skipped. Callers keep the running
 * state across batches (sum, count, min, max) and decide null results from the valid count.
 *
 * <p>Each 64-element block is processed with plain unmasked loads and lanewise ops when its validity
 * word is full (the common case), and with lane masks otherwise. Masks are derived from the
 * validity bits with a broadcast-AND-compare rather than {@code VectorMask.fromLong}, which has no
 * fast path on NEON/AVX2 and made the null path several times slower than scalar code.
 *
 * <p>Doubles follow Spark: max treats NaN as the largest value (Java's max already propagates NaN),
 * min ignores NaN unless every valid value is NaN.
 */
public final class AggKernels {

  static final VectorSpecies<Integer> I = IntVector.SPECIES_PREFERRED;
  static final VectorSpecies<Long> L = LongVector.SPECIES_PREFERRED;
  static final VectorSpecies<Double> D = DoubleVector.SPECIES_PREFERRED;
  /** Int species with as many lanes as {@link #L}, for widening int sums. */
  static final VectorSpecies<Integer> IH = CastKernels.IH;
  static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

  private static final LongVector L_LANE_BITS = LongVector.fromArray(L, laneBits(L.length()), 0);
  private static final IntVector I_LANE_BITS = IntVector.fromArray(I, intLaneBits(I.length()), 0);
  private static final IntVector IH_LANE_BITS = IntVector.fromArray(IH, intLaneBits(IH.length()), 0);

  private AggKernels() {}

  private static long[] laneBits(int lanes) {
    long[] b = new long[lanes];
    for (int i = 0; i < lanes; i++) {
      b[i] = 1L << i;
    }
    return b;
  }

  private static int[] intLaneBits(int lanes) {
    int[] b = new int[lanes];
    for (int i = 0; i < lanes; i++) {
      b[i] = 1 << i;
    }
    return b;
  }

  /** Lane mask from the low {@code lanes} bits of {@code bits}. */
  static VectorMask<Long> maskL(long bits) {
    return LongVector.broadcast(L, bits).and(L_LANE_BITS).compare(VectorOperators.NE, 0L);
  }

  static VectorMask<Double> maskD(long bits) {
    return maskL(bits).cast(D);
  }

  static VectorMask<Integer> maskI(long bits) {
    return IntVector.broadcast(I, (int) bits).and(I_LANE_BITS).compare(VectorOperators.NE, 0);
  }

  static VectorMask<Integer> maskIH(long bits) {
    return IntVector.broadcast(IH, (int) bits).and(IH_LANE_BITS).compare(VectorOperators.NE, 0);
  }

  /** Number of non-null elements. */
  public static long countValid(VectorBuffers a) {
    MemorySegment v = a.validity();
    return v == null ? a.length() : Bitmap.popcount(v, a.length());
  }

  private static long wordOf(MemorySegment validity, int w, int n) {
    return validity == null ? fullWord(Math.min(64, n - (w << 6))) : Bitmap.wordAt(validity, w, n);
  }

  private static long fullWord(int bits) {
    return bits >= 64 ? -1L : (1L << bits) - 1;
  }

  // ================================================================== sums

  /** Sum of the valid doubles (0.0 when there are none). */
  public static double sumDouble(VectorBuffers a) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = D.length();
    long laneMask = fullWord(lanes);
    DoubleVector acc = DoubleVector.zero(D);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          acc = acc.add(DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE));
        }
        if (k < limit) {
          acc = acc.add(DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, D.indexInRange(k, limit)));
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            // Masked load zero-fills the null lanes, which is the identity for ADD.
            acc = acc.add(DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, maskD(bits)));
          }
        }
      }
    }
    return acc.reduceLanes(VectorOperators.ADD);
  }

  /** Sum of the valid longs (wrapping on overflow, like Spark's legacy mode). */
  public static long sumLong(VectorBuffers a) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = L.length();
    long laneMask = fullWord(lanes);
    LongVector acc = LongVector.zero(L);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          acc = acc.add(LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE));
        }
        if (k < limit) {
          acc = acc.add(LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, L.indexInRange(k, limit)));
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            acc = acc.add(LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, maskL(bits)));
          }
        }
      }
    }
    return acc.reduceLanes(VectorOperators.ADD);
  }

  /** Sum of the valid ints, accumulated in 64-bit lanes (cannot overflow for realistic sizes). */
  public static long sumInt(VectorBuffers a) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = L.length(); // == IH.length()
    long laneMask = fullWord(lanes);
    LongVector acc = LongVector.zero(L);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          acc = acc.add(IntVector.fromMemorySegment(IH, d, (long) (base + k) << 2, LE).convertShape(VectorOperators.I2L, L, 0).reinterpretAsLongs());
        }
        if (k < limit) {
          acc = acc.add(IntVector.fromMemorySegment(IH, d, (long) (base + k) << 2, LE, IH.indexInRange(k, limit)).convertShape(VectorOperators.I2L, L, 0).reinterpretAsLongs());
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            acc = acc.add(IntVector.fromMemorySegment(IH, d, (long) (base + k) << 2, LE, maskIH(bits)).convertShape(VectorOperators.I2L, L, 0).reinterpretAsLongs());
          }
        }
      }
    }
    return acc.reduceLanes(VectorOperators.ADD);
  }

  // ================================================================== min / max

  /** Max of the valid doubles; NaN if any valid value is NaN. Undefined when none are valid. */
  public static double maxDouble(VectorBuffers a) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = D.length();
    long laneMask = fullWord(lanes);
    DoubleVector acc = DoubleVector.broadcast(D, Double.NEGATIVE_INFINITY);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          acc = acc.max(DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE));
        }
        if (k < limit) {
          VectorMask<Double> m = D.indexInRange(k, limit);
          acc = acc.lanewise(VectorOperators.MAX, DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m), m);
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            VectorMask<Double> m = maskD(bits);
            acc = acc.lanewise(VectorOperators.MAX, DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m), m);
          }
        }
      }
    }
    // Java's max propagates NaN, which is exactly Spark's "NaN is the largest double".
    return acc.reduceLanes(VectorOperators.MAX);
  }

  /** Min of the valid doubles, ignoring NaN unless every valid value is NaN. */
  public static double minDouble(VectorBuffers a) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = D.length();
    long laneMask = fullWord(lanes);
    DoubleVector acc = DoubleVector.broadcast(D, Double.POSITIVE_INFINITY);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          acc = acc.min(DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE));
        }
        if (k < limit) {
          VectorMask<Double> m = D.indexInRange(k, limit);
          acc = acc.lanewise(VectorOperators.MIN, DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m), m);
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            VectorMask<Double> m = maskD(bits);
            acc = acc.lanewise(VectorOperators.MIN, DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m), m);
          }
        }
      }
    }
    double r = acc.reduceLanes(VectorOperators.MIN);
    // Java's min propagates NaN; Spark ignores NaN unless nothing else is there. NaN is rare, so
    // the fast path assumes none and only then does the careful pass.
    return Double.isNaN(r) ? minDoubleIgnoringNaN(a) : r;
  }

  private static double minDoubleIgnoringNaN(VectorBuffers a) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = D.length();
    long laneMask = fullWord(lanes);
    DoubleVector acc = DoubleVector.broadcast(D, Double.POSITIVE_INFINITY);
    boolean sawNumber = false;
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
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
        VectorMask<Double> m = maskD(bits);
        DoubleVector v = DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m);
        VectorMask<Double> numbers = v.compare(VectorOperators.EQ, v).and(m); // false on NaN
        sawNumber |= numbers.anyTrue();
        acc = acc.lanewise(VectorOperators.MIN, v, numbers);
      }
    }
    return sawNumber ? acc.reduceLanes(VectorOperators.MIN) : Double.NaN;
  }

  public static long maxLong(VectorBuffers a) {
    return minMaxLong(a, false);
  }

  public static long minLong(VectorBuffers a) {
    return minMaxLong(a, true);
  }

  private static long minMaxLong(VectorBuffers a, boolean min) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = L.length();
    long laneMask = fullWord(lanes);
    LongVector acc = LongVector.broadcast(L, min ? Long.MAX_VALUE : Long.MIN_VALUE);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          LongVector v = LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE);
          acc = min ? acc.min(v) : acc.max(v);
        }
        if (k < limit) {
          VectorMask<Long> m = L.indexInRange(k, limit);
          LongVector v = LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, m);
          acc = min ? acc.lanewise(VectorOperators.MIN, v, m) : acc.lanewise(VectorOperators.MAX, v, m);
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            VectorMask<Long> m = maskL(bits);
            LongVector v = LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, m);
            acc = min ? acc.lanewise(VectorOperators.MIN, v, m) : acc.lanewise(VectorOperators.MAX, v, m);
          }
        }
      }
    }
    return min ? acc.reduceLanes(VectorOperators.MIN) : acc.reduceLanes(VectorOperators.MAX);
  }

  public static int maxInt(VectorBuffers a) {
    return minMaxInt(a, false);
  }

  public static int minInt(VectorBuffers a) {
    return minMaxInt(a, true);
  }

  private static int minMaxInt(VectorBuffers a, boolean min) {
    MemorySegment d = a.data();
    MemorySegment validity = a.validity();
    int n = a.length();
    int lanes = I.length();
    long laneMask = fullWord(lanes);
    IntVector acc = IntVector.broadcast(I, min ? Integer.MAX_VALUE : Integer.MIN_VALUE);
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      long word = wordOf(validity, w, n);
      if (word == 0L) {
        continue;
      }
      int base = w << 6;
      int limit = Math.min(64, n - base);
      if (word == fullWord(limit)) {
        int k = 0;
        for (; k + lanes <= limit; k += lanes) {
          IntVector v = IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE);
          acc = min ? acc.min(v) : acc.max(v);
        }
        if (k < limit) {
          VectorMask<Integer> m = I.indexInRange(k, limit);
          IntVector v = IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE, m);
          acc = min ? acc.lanewise(VectorOperators.MIN, v, m) : acc.lanewise(VectorOperators.MAX, v, m);
        }
      } else {
        for (int k = 0; k < limit; k += lanes) {
          long bits = (word >>> k) & laneMask;
          if (bits != 0L) {
            VectorMask<Integer> m = maskI(bits);
            IntVector v = IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE, m);
            acc = min ? acc.lanewise(VectorOperators.MIN, v, m) : acc.lanewise(VectorOperators.MAX, v, m);
          }
        }
      }
    }
    return min ? acc.reduceLanes(VectorOperators.MIN) : acc.reduceLanes(VectorOperators.MAX);
  }
}
