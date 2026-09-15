package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

/** Random column generators and bitmap assertions shared by kernel tests. */
final class TestData {

  static final int[] LENGTHS = {0, 1, 3, 7, 8, 15, 16, 31, 63, 64, 65, 100, 127, 128, 129, 1000, 1023, 4097};

  private TestData() {}

  static boolean[] nulls(Random rnd, int n, double fraction) {
    boolean[] nulls = new boolean[n];
    for (int i = 0; i < n; i++) {
      nulls[i] = rnd.nextDouble() < fraction;
    }
    return nulls;
  }

  /** Small-range ints so that equality comparisons hit often. */
  static SegmentVectorBuffers ints(Arena arena, Random rnd, int n, boolean[] nulls) {
    int[] v = new int[n];
    for (int i = 0; i < n; i++) {
      v[i] = rnd.nextInt(-20, 20);
    }
    return ArrowLayout.ofInts(arena, v, nulls);
  }

  static SegmentVectorBuffers longs(Arena arena, Random rnd, int n, boolean[] nulls) {
    long[] v = new long[n];
    for (int i = 0; i < n; i++) {
      v[i] = rnd.nextInt(3) == 0 ? rnd.nextLong() : rnd.nextLong(-20, 20);
    }
    return ArrowLayout.ofLongs(arena, v, nulls);
  }

  /** Doubles including NaN, infinities and signed zeros to exercise NaN-safe ordering. */
  static SegmentVectorBuffers doubles(Arena arena, Random rnd, int n, boolean[] nulls) {
    double[] v = new double[n];
    for (int i = 0; i < n; i++) {
      v[i] =
          switch (rnd.nextInt(12)) {
            case 0 -> Double.NaN;
            case 1 -> Double.POSITIVE_INFINITY;
            case 2 -> Double.NEGATIVE_INFINITY;
            case 3 -> -0.0;
            case 4 -> 0.0;
            case 5 -> 1.5;
            default -> rnd.nextInt(-5, 5) + (rnd.nextBoolean() ? 0.0 : 0.5);
          };
    }
    return ArrowLayout.ofDoubles(arena, v, nulls);
  }

  static MemorySegment randomBitmap(Arena arena, Random rnd, int n) {
    MemorySegment bm = ArrowLayout.allocateBitmap(arena, n);
    for (int i = 0; i < n; i++) {
      if (rnd.nextBoolean()) {
        Bitmap.set(bm, i);
      }
    }
    return bm;
  }

  static void assertBitmapEquals(MemorySegment expected, MemorySegment actual, int n, String what) {
    for (int i = 0; i < n; i++) {
      assertEquals(Bitmap.isSet(expected, i), Bitmap.isSet(actual, i), what + " bit " + i + " of " + n);
    }
  }
}
