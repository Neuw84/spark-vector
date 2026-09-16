package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.RoundKernels.Mode;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** {@link RoundKernels} against {@link BigDecimal} (Spark's own reference) and Spark's documented cases. */
class RoundKernelsTest {

  @Test
  void doublesFollowSparkNotMathRound() {
    double[] xs = {0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.675, -2.675, 1234.5678, -1234.5678, 0.0, -0.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1e300, 0.1 + 0.2};
    // round: HALF_UP on the shortest decimal representation.
    double[] halfUp0 = {1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 3.0, -3.0, 1235.0, -1235.0, 0.0, 0.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1e300, 0.0};
    double[] halfEven0 = {0.0, 2.0, 2.0, 0.0, -2.0, -2.0, 3.0, -3.0, 1235.0, -1235.0, 0.0, 0.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1e300, 0.0};
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofDoubles(arena, xs, null);
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.FLOAT64, xs.length);
      RoundKernels.roundDouble(a, 0, Mode.HALF_UP, out);
      for (int i = 0; i < xs.length; i++) assertSameDouble(halfUp0[i], out.getAtIndex(VectorBuffers.LE_DOUBLE, i), "round(" + xs[i] + ")");
      RoundKernels.roundDouble(a, 0, Mode.HALF_EVEN, out);
      for (int i = 0; i < xs.length; i++) assertSameDouble(halfEven0[i], out.getAtIndex(VectorBuffers.LE_DOUBLE, i), "bround(" + xs[i] + ")");
      RoundKernels.roundDouble(a, 2, Mode.HALF_UP, out);
      assertEquals(2.68, out.getAtIndex(VectorBuffers.LE_DOUBLE, 6), "round(2.675, 2) rounds the decimal string, not the binary value");
      assertEquals(-2.68, out.getAtIndex(VectorBuffers.LE_DOUBLE, 7));
      assertEquals(1234.57, out.getAtIndex(VectorBuffers.LE_DOUBLE, 8));
      assertEquals(0.3, out.getAtIndex(VectorBuffers.LE_DOUBLE, 16), "0.30000000000000004 rounds to 0.3");
      RoundKernels.roundDouble(a, -2, Mode.HALF_UP, out);
      assertEquals(1200.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, 8));
      assertEquals(-1200.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, 9));
      assertEquals(0.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, 0));
      RoundKernels.roundDouble(a, -2, Mode.HALF_EVEN, out);
      assertEquals(1200.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, 8));
      assertTrue(Double.isNaN(out.getAtIndex(VectorBuffers.LE_DOUBLE, 12)));
      // Every scale, every mode against the reference for a random sample.
      Random rnd = new Random(3);
      for (int t = 0; t < 2000; t++) {
        double d = rnd.nextGaussian() * Math.pow(10, rnd.nextInt(8) - 2);
        int k = rnd.nextInt(9) - 4;
        for (Mode mode : Mode.values()) {
          double expected = new BigDecimal(Double.toString(d)).setScale(k, mode.java).doubleValue();
          assertSameDouble(expected, RoundKernels.roundDouble(d, k, mode), mode + "(" + d + ", " + k + ")");
        }
      }
      // ceil / floor to long and rint.
      MemorySegment outl = ArrowLayout.allocateData(arena, VecType.INT64, xs.length);
      RoundKernels.ceilFloorToLong(a, true, outl);
      long[] ceils = {1, 2, 3, 0, -1, -2, 3, -2, 1235, -1234, 0, 0, 0, Long.MAX_VALUE, Long.MIN_VALUE, Long.MAX_VALUE, 1};
      for (int i = 0; i < xs.length; i++) assertEquals(ceils[i], outl.getAtIndex(VectorBuffers.LE_LONG, i), "ceil(" + xs[i] + ")");
      RoundKernels.ceilFloorToLong(a, false, outl);
      long[] floors = {0, 1, 2, -1, -2, -3, 2, -3, 1234, -1235, 0, 0, 0, Long.MAX_VALUE, Long.MIN_VALUE, Long.MAX_VALUE, 0};
      for (int i = 0; i < xs.length; i++) assertEquals(floors[i], outl.getAtIndex(VectorBuffers.LE_LONG, i), "floor(" + xs[i] + ")");
      RoundKernels.rint(a, out);
      for (int i = 0; i < xs.length; i++) assertSameDouble(Math.rint(xs[i]), out.getAtIndex(VectorBuffers.LE_DOUBLE, i), "rint(" + xs[i] + ")");
    }
  }

  @Test
  void integersToNegativeScalesWrapOrFlagLikeBigDecimal() {
    try (Arena arena = Arena.ofConfined()) {
      Random rnd = new Random(11);
      int n = 3000;
      long[] xl = new long[n];
      int[] xi = new int[n];
      for (int i = 0; i < n; i++) {
        xl[i] = switch (i % 4) {
          case 0 -> rnd.nextLong();
          case 1 -> rnd.nextInt(2000) - 1000;
          case 2 -> (long) rnd.nextInt() * 1000L + rnd.nextInt(1000);
          default -> rnd.nextBoolean() ? Long.MAX_VALUE - rnd.nextInt(100) : Long.MIN_VALUE + rnd.nextInt(100);
        };
        xi[i] = i % 3 == 0 ? (rnd.nextBoolean() ? Integer.MAX_VALUE - rnd.nextInt(100) : Integer.MIN_VALUE + rnd.nextInt(100)) : rnd.nextInt();
      }
      xl[0] = 5_000_000_000_000_000_000L; // rounds up to 10^19 at power 19: overflow
      xl[1] = 4_999_999_999_999_999_999L; // stays 0 at power 19
      xl[2] = 15L; xl[3] = 25L; xl[4] = -15L; xl[5] = -25L; // the halves at power 1
      xi[0] = 2_147_483_647; xi[1] = 2_147_483_600; xi[2] = -2_147_483_648; xi[3] = 15; xi[4] = 25; xi[5] = -25;
      VectorBuffers al = ArrowLayout.ofLongs(arena, xl, null);
      VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null);
      MemorySegment outl = ArrowLayout.allocateData(arena, VecType.INT64, n);
      MemorySegment outi = ArrowLayout.allocateData(arena, VecType.INT32, n);
      MemorySegment ov = ArrowLayout.allocateBitmap(arena, n);
      for (Mode mode : new Mode[] {Mode.HALF_UP, Mode.HALF_EVEN}) {
        for (int power : new int[] {1, 2, 3, 9, 10, 17, 18, 19, 20, 25}) {
          Bitmap.fill(ov, n, false);
          RoundKernels.roundIntegral(al, power, mode, outl, ov);
          for (int i = 0; i < n; i++) {
            BigDecimal r = new BigDecimal(xl[i]).setScale(-power, mode.java);
            boolean overflow;
            try { r.longValueExact(); overflow = false; } catch (ArithmeticException e) { overflow = true; }
            assertEquals(r.longValue(), outl.getAtIndex(VectorBuffers.LE_LONG, i), mode + " long " + xl[i] + " power " + power);
            assertEquals(overflow, Bitmap.isSet(ov, i), mode + " long overflow " + xl[i] + " power " + power);
          }
          Bitmap.fill(ov, n, false);
          RoundKernels.roundIntegral(ai, power, mode, outi, ov);
          for (int i = 0; i < n; i++) {
            BigDecimal r = new BigDecimal(xi[i]).setScale(-power, mode.java);
            boolean overflow;
            try { r.intValueExact(); overflow = false; } catch (ArithmeticException e) { overflow = true; }
            assertEquals(r.intValue(), outi.getAtIndex(VectorBuffers.LE_INT, i), mode + " int " + xi[i] + " power " + power);
            assertEquals(overflow, Bitmap.isSet(ov, i), mode + " int overflow " + xi[i] + " power " + power);
          }
        }
      }
      Bitmap.fill(ov, n, false);
      RoundKernels.roundIntegral(al, 1, Mode.HALF_UP, outl, ov);
      assertEquals(20L, outl.getAtIndex(VectorBuffers.LE_LONG, 2));
      assertEquals(30L, outl.getAtIndex(VectorBuffers.LE_LONG, 3));
      assertEquals(-20L, outl.getAtIndex(VectorBuffers.LE_LONG, 4));
      assertEquals(-30L, outl.getAtIndex(VectorBuffers.LE_LONG, 5));
      RoundKernels.roundIntegral(al, 1, Mode.HALF_EVEN, outl, ov);
      assertEquals(20L, outl.getAtIndex(VectorBuffers.LE_LONG, 2));
      assertEquals(20L, outl.getAtIndex(VectorBuffers.LE_LONG, 3));
      assertEquals(-20L, outl.getAtIndex(VectorBuffers.LE_LONG, 4));
      assertEquals(-20L, outl.getAtIndex(VectorBuffers.LE_LONG, 5));
      Bitmap.fill(ov, n, false);
      RoundKernels.roundIntegral(ai, 3, Mode.HALF_UP, outi, ov);
      assertTrue(Bitmap.isSet(ov, 1), "2147483600 rounds to 2147484000, above the int range");
      assertEquals((int) 2_147_484_000L, outi.getAtIndex(VectorBuffers.LE_INT, 1), "wrapped like BigDecimal.intValue()");
      assertFalse(Bitmap.isSet(ov, 3));
    }
  }

  @Test
  void decimalsRoundOnTheUnscaledValue() {
    Random rnd = new Random(5);
    for (int t = 0; t < 20000; t++) {
      long v = rnd.nextInt(4) == 0 ? rnd.nextLong() / 10 : rnd.nextInt(2_000_000) - 1_000_000;
      int s = rnd.nextInt(19); // the decimal's scale
      int k = rnd.nextInt(26) - 8; // target scale
      for (Mode mode : Mode.values()) {
        BigDecimal expected = BigDecimal.valueOf(v, s).setScale(k, mode.java);
        long got;
        if (k >= 0) {
          int newScale = Math.min(s, k);
          got = RoundKernels.roundDiv(v, s - newScale, mode);
          assertEquals(expected.setScale(newScale, mode.java).unscaledValue().longValueExact(), got, mode + " " + v + " scale " + s + " to " + k);
        } else {
          got = RoundKernels.roundDiv(v, s - k, mode) * DecimalKernels.POW10[-k];
          assertEquals(expected.setScale(0).unscaledValue().longValueExact(), got, mode + " " + v + " scale " + s + " to " + k);
        }
      }
    }
    // The column form, and a divisor above 18 digits.
    try (Arena arena = Arena.ofConfined()) {
      long[] unscaled = {12345, -12345, 12350, -12350, 12250, 5, -5, 0, 999_999_999_999_999_999L};
      VectorBuffers a = ArrowLayout.ofLongs(arena, unscaled, null);
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT64, unscaled.length);
      RoundKernels.roundDecimal(a, 2, 0, Mode.HALF_UP, out); // decimal(_, 2) -> scale 0
      long[] halfUp = {123, -123, 124, -124, 123, 0, 0, 0, 10_000_000_000_000_000L};
      for (int i = 0; i < unscaled.length; i++) assertEquals(halfUp[i], out.getAtIndex(VectorBuffers.LE_LONG, i), "half up " + unscaled[i]);
      RoundKernels.roundDecimal(a, 2, 0, Mode.HALF_EVEN, out);
      long[] halfEven = {123, -123, 124, -124, 122, 0, 0, 0, 10_000_000_000_000_000L};
      for (int i = 0; i < unscaled.length; i++) assertEquals(halfEven[i], out.getAtIndex(VectorBuffers.LE_LONG, i), "half even " + unscaled[i]);
      RoundKernels.roundDecimal(a, 2, 0, Mode.CEILING, out);
      long[] ceiling = {124, -123, 124, -123, 123, 1, 0, 0, 10_000_000_000_000_000L};
      for (int i = 0; i < unscaled.length; i++) assertEquals(ceiling[i], out.getAtIndex(VectorBuffers.LE_LONG, i), "ceiling " + unscaled[i]);
      RoundKernels.roundDecimal(a, 2, 0, Mode.FLOOR, out);
      long[] floor = {123, -124, 123, -124, 122, 0, -1, 0, 9_999_999_999_999_999L};
      for (int i = 0; i < unscaled.length; i++) assertEquals(floor[i], out.getAtIndex(VectorBuffers.LE_LONG, i), "floor " + unscaled[i]);
      RoundKernels.roundDecimal(a, 4, 2, Mode.HALF_UP, out); // decimal(_, 2) rounded to scale -2: divide by 10^4, times 10^2
      assertEquals(100L, out.getAtIndex(VectorBuffers.LE_LONG, 0));
      assertEquals(-100L, out.getAtIndex(VectorBuffers.LE_LONG, 1));
      RoundKernels.roundDecimal(a, 25, 0, Mode.CEILING, out);
      assertEquals(1L, out.getAtIndex(VectorBuffers.LE_LONG, 0));
      assertEquals(0L, out.getAtIndex(VectorBuffers.LE_LONG, 1));
      RoundKernels.roundDecimal(a, 25, 0, Mode.HALF_UP, out);
      assertEquals(0L, out.getAtIndex(VectorBuffers.LE_LONG, 8));
    }
  }

  private static void assertSameDouble(double expected, double actual, String what) {
    if (Double.isNaN(expected)) {
      assertTrue(Double.isNaN(actual), what);
    } else {
      assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual), what + ": expected " + expected + " got " + actual);
    }
  }
}
