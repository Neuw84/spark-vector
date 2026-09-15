package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.reference.ScalarReference;
import java.lang.foreign.Arena;
import java.util.Random;
import org.junit.jupiter.api.Test;

class AggKernelsTest {

  private static void assertDoubleAgg(double expected, double actual, String what) {
    if (Double.isNaN(expected)) {
      assertTrue(Double.isNaN(actual), what + " expected NaN, got " + actual);
    } else if (Double.isInfinite(expected) || expected == 0.0) {
      assertEquals(expected, actual, what);
    } else {
      assertEquals(expected, actual, Math.abs(expected) * 1e-12, what);
    }
  }

  @Test
  void sumsMatchReferenceWithAndWithoutNulls() {
    Random rnd = new Random(21);
    for (int n : TestData.LENGTHS) {
      for (double nullFraction : new double[] {0.0, 0.01, 0.3, 1.0}) {
        try (Arena arena = Arena.ofConfined()) {
          boolean[] nulls = TestData.nulls(rnd, n, nullFraction);
          // Finite doubles only so that the sequential and SIMD sums agree to a tolerance.
          double[] dv = new double[n];
          int[] iv = new int[n];
          long[] lv = new long[n];
          for (int i = 0; i < n; i++) {
            dv[i] = rnd.nextInt(-1000, 1000) / 8.0;
            iv[i] = rnd.nextInt();
            lv[i] = rnd.nextLong(-1L << 40, 1L << 40);
          }
          VectorBuffers d = ArrowLayout.ofDoubles(arena, dv, nulls);
          VectorBuffers ints = ArrowLayout.ofInts(arena, iv, nulls);
          VectorBuffers longs = ArrowLayout.ofLongs(arena, lv, nulls);
          String what = "n=" + n + " nulls=" + nullFraction;
          assertEquals(ScalarReference.countValid(d), AggKernels.countValid(d), what);
          assertDoubleAgg(ScalarReference.sumDouble(d), AggKernels.sumDouble(d), "sumDouble " + what);
          assertEquals(ScalarReference.sumLong(ints), AggKernels.sumInt(ints), "sumInt " + what);
          assertEquals(ScalarReference.sumLong(longs), AggKernels.sumLong(longs), "sumLong " + what);
        }
      }
    }
  }

  @Test
  void minMaxMatchReferenceIncludingNaNAndInfinities() {
    Random rnd = new Random(22);
    for (int n : TestData.LENGTHS) {
      for (double nullFraction : new double[] {0.0, 0.2, 1.0}) {
        try (Arena arena = Arena.ofConfined()) {
          boolean[] nulls = TestData.nulls(rnd, n, nullFraction);
          VectorBuffers d = TestData.doubles(arena, rnd, n, nulls); // has NaN, +/-Inf, -0.0
          VectorBuffers ints = TestData.ints(arena, rnd, n, nulls);
          VectorBuffers longs = TestData.longs(arena, rnd, n, nulls);
          String what = "n=" + n + " nulls=" + nullFraction;
          if (AggKernels.countValid(d) > 0) {
            assertDoubleAgg(ScalarReference.minDouble(d), AggKernels.minDouble(d), "minDouble " + what);
            assertDoubleAgg(ScalarReference.maxDouble(d), AggKernels.maxDouble(d), "maxDouble " + what);
            assertEquals(ScalarReference.minLong(ints), AggKernels.minInt(ints), "minInt " + what);
            assertEquals(ScalarReference.maxLong(ints), AggKernels.maxInt(ints), "maxInt " + what);
            assertEquals(ScalarReference.minLong(longs), AggKernels.minLong(longs), "minLong " + what);
            assertEquals(ScalarReference.maxLong(longs), AggKernels.maxLong(longs), "maxLong " + what);
          }
        }
      }
    }
  }

  @Test
  void nanHandlingFollowsSpark() {
    try (Arena arena = Arena.ofConfined()) {
      double nan = Double.NaN;
      VectorBuffers mixed = ArrowLayout.ofDoubles(arena, new double[] {3.0, nan, 1.0, 2.0, nan}, null);
      assertEquals(1.0, AggKernels.minDouble(mixed), "min ignores NaN");
      assertTrue(Double.isNaN(AggKernels.maxDouble(mixed)), "max is NaN when any NaN");

      VectorBuffers allNaN = ArrowLayout.ofDoubles(arena, new double[] {nan, nan, nan}, null);
      assertTrue(Double.isNaN(AggKernels.minDouble(allNaN)), "min of only NaNs is NaN");

      // NaN in a null slot must be ignored entirely.
      VectorBuffers nullNaN = ArrowLayout.ofDoubles(arena, new double[] {nan, 5.0, 7.0}, new boolean[] {true, false, false});
      assertEquals(7.0, AggKernels.maxDouble(nullNaN));
      assertEquals(5.0, AggKernels.minDouble(nullNaN));
      assertEquals(12.0, AggKernels.sumDouble(nullNaN));
    }
  }
}
