package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.MathKernels.Pick;
import io.sparkvector.kernels.MathKernels.RemOp;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** {@link MathKernels} against Spark's definitions, at the type extremes and with zero divisors. */
class MathKernelsTest {

  private static long sparkPmod(long a, long n) {
    long r = a % n;
    return r < 0 ? (r + n) % n : r;
  }

  private static double sparkPmod(double a, double n) {
    double r = a % n;
    return r < 0 ? (r + n) % n : r;
  }

  @Test
  void absAndSignum() {
    try (Arena arena = Arena.ofConfined()) {
      int[] xi = {0, 1, -1, Integer.MAX_VALUE, Integer.MIN_VALUE, -12345};
      VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null);
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT32, xi.length);
      MathKernels.abs(ai, out);
      for (int i = 0; i < xi.length; i++) {
        assertEquals(Math.abs(xi[i]), out.getAtIndex(VectorBuffers.LE_INT, i));
      }
      assertEquals(Integer.MIN_VALUE, out.getAtIndex(VectorBuffers.LE_INT, 4), "MIN_VALUE wraps; ANSI checks it via the negate-overflow mask");
      long[] xl = {0L, -7L, Long.MIN_VALUE, Long.MAX_VALUE};
      VectorBuffers al = ArrowLayout.ofLongs(arena, xl, null);
      MemorySegment outl = ArrowLayout.allocateData(arena, VecType.INT64, xl.length);
      MathKernels.abs(al, outl);
      assertEquals(7L, outl.getAtIndex(VectorBuffers.LE_LONG, 1));
      assertEquals(Long.MIN_VALUE, outl.getAtIndex(VectorBuffers.LE_LONG, 2));
      double[] xd = {-0.0, -2.5, Double.NaN, Double.NEGATIVE_INFINITY, 3.0, 0.0};
      VectorBuffers ad = ArrowLayout.ofDoubles(arena, xd, null);
      MemorySegment outd = ArrowLayout.allocateData(arena, VecType.FLOAT64, xd.length);
      MathKernels.abs(ad, outd);
      assertEquals(2.5, outd.getAtIndex(VectorBuffers.LE_DOUBLE, 1));
      assertTrue(Double.isNaN(outd.getAtIndex(VectorBuffers.LE_DOUBLE, 2)));
      assertEquals(Double.POSITIVE_INFINITY, outd.getAtIndex(VectorBuffers.LE_DOUBLE, 3));
      MathKernels.signum(ad, outd);
      double[] expected = {-0.0, -1.0, Double.NaN, -1.0, 1.0, 0.0};
      for (int i = 0; i < xd.length; i++) {
        double got = outd.getAtIndex(VectorBuffers.LE_DOUBLE, i);
        if (Double.isNaN(expected[i])) assertTrue(Double.isNaN(got)); else assertEquals(expected[i], got, "signum of " + xd[i]);
      }
      double[] xs = {4.0, 2.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY, 0.0, -0.0};
      VectorBuffers as = ArrowLayout.ofDoubles(arena, xs, null);
      MemorySegment outs = ArrowLayout.allocateData(arena, VecType.FLOAT64, xs.length);
      MathKernels.sqrt(as, outs);
      double[] expectedSqrt = {2.0, Math.sqrt(2.0), Double.NaN, Double.NaN, Double.POSITIVE_INFINITY, 0.0, -0.0};
      for (int i = 0; i < xs.length; i++) {
        double got = outs.getAtIndex(VectorBuffers.LE_DOUBLE, i);
        if (Double.isNaN(expectedSqrt[i])) assertTrue(Double.isNaN(got)); else assertEquals(expectedSqrt[i], got, "sqrt of " + xs[i]);
      }
    }
  }

  @Test
  void remainderPmodAndIntegralDivideFollowSparkAndSkipZeroDivisors() {
    try (Arena arena = Arena.ofConfined()) {
      Random rnd = new Random(9);
      int n = 2000;
      long[] xl = new long[n], yl = new long[n];
      int[] xi = new int[n], yi = new int[n];
      double[] xd = new double[n], yd = new double[n];
      for (int i = 0; i < n; i++) {
        xl[i] = rnd.nextBoolean() ? rnd.nextLong() : rnd.nextInt(200) - 100;
        yl[i] = i % 7 == 0 ? 0L : (rnd.nextBoolean() ? rnd.nextLong() : rnd.nextInt(20) - 10);
        xi[i] = (int) xl[i];
        yi[i] = (int) yl[i];
        xd[i] = rnd.nextGaussian() * 50;
        yd[i] = i % 7 == 0 ? 0.0 : rnd.nextGaussian() * 5;
      }
      xl[1] = Long.MIN_VALUE; yl[1] = -1L; // the one overflowing integral division
      xi[2] = Integer.MIN_VALUE; yi[2] = -1;
      xl[3] = -7L; yl[3] = 3L; // pmod(-7, 3) = 2, -7 % 3 = -1
      xl[4] = 7L; yl[4] = -3L; // pmod(7, -3) = 1 (r = 1 is not negative), 7 % -3 = 1
      VectorBuffers al = ArrowLayout.ofLongs(arena, xl, null), bl = ArrowLayout.ofLongs(arena, yl, null);
      VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null), bi = ArrowLayout.ofInts(arena, yi, null);
      VectorBuffers ad = ArrowLayout.ofDoubles(arena, xd, null), bd = ArrowLayout.ofDoubles(arena, yd, null);
      MemorySegment outl = ArrowLayout.allocateData(arena, VecType.INT64, n);
      MemorySegment outi = ArrowLayout.allocateData(arena, VecType.INT32, n);
      MemorySegment outd = ArrowLayout.allocateData(arena, VecType.FLOAT64, n);
      for (RemOp op : RemOp.values()) {
        MathKernels.remainder(op, al, bl, outl);
        MathKernels.remainder(op, ai, bi, outi);
        MathKernels.remainder(op, ad, bd, outd);
        for (int i = 0; i < n; i++) {
          if (yl[i] != 0L) {
            long e = op == RemOp.REM ? xl[i] % yl[i] : sparkPmod(xl[i], yl[i]);
            assertEquals(e, outl.getAtIndex(VectorBuffers.LE_LONG, i), op + " long row " + i);
          } else {
            assertEquals(0L, outl.getAtIndex(VectorBuffers.LE_LONG, i), "zero divisor lane left 0");
          }
          if (yi[i] != 0) {
            int r = xi[i] % yi[i]; long e = op == RemOp.REM ? r : (r < 0 ? (r + yi[i]) % yi[i] : r); // Spark pmod in int arithmetic (wraps)
            assertEquals((int) e, outi.getAtIndex(VectorBuffers.LE_INT, i), op + " int row " + i);
          }
          if (yd[i] != 0.0) {
            double e = op == RemOp.REM ? xd[i] % yd[i] : sparkPmod(xd[i], yd[i]);
            assertEquals(e, outd.getAtIndex(VectorBuffers.LE_DOUBLE, i), 0.0, op + " double row " + i);
          }
        }
        // Scalar forms, both orders.
        MathKernels.remainderScalar(op, al, 3L, false, outl);
        MathKernels.remainderScalar(op, al, 3L, true, outi.byteSize() >= outl.byteSize() ? outl : outl);
        MathKernels.remainderScalar(op, al, 3L, false, outl);
        for (int i = 0; i < n; i++) {
          long e = op == RemOp.REM ? xl[i] % 3L : sparkPmod(xl[i], 3L);
          assertEquals(e, outl.getAtIndex(VectorBuffers.LE_LONG, i), op + " scalar row " + i);
        }
        MathKernels.remainderScalar(op, al, 100L, true, outl);
        for (int i = 0; i < n; i++) {
          long e = xl[i] == 0L ? 0L : (op == RemOp.REM ? 100L % xl[i] : sparkPmod(100L, xl[i]));
          assertEquals(e, outl.getAtIndex(VectorBuffers.LE_LONG, i), op + " reversed scalar row " + i);
        }
      }
      assertEquals(2L, MathKernels.rem(RemOp.PMOD, -7L, 3L));
      assertEquals(-1L, MathKernels.rem(RemOp.REM, -7L, 3L));
      assertEquals(1L, MathKernels.rem(RemOp.PMOD, 7L, -3L)); // r = 1 is not negative: Spark leaves it
      assertEquals(1L, MathKernels.rem(RemOp.REM, 7L, -3L));

      MathKernels.integralDivide(al, bl, outl);
      MemorySegment ov = ArrowLayout.allocateBitmap(arena, n);
      MathKernels.integralDivideOverflow(al, bl, ov);
      for (int i = 0; i < n; i++) {
        if (yl[i] != 0L) assertEquals(xl[i] / yl[i], outl.getAtIndex(VectorBuffers.LE_LONG, i), "div row " + i);
        assertEquals(xl[i] == Long.MIN_VALUE && yl[i] == -1L, Bitmap.isSet(ov, i), "overflow row " + i);
      }
      assertTrue(Bitmap.isSet(ov, 1));
      MathKernels.integralDivide(ai, bi, outl);
      for (int i = 0; i < n; i++) {
        if (yi[i] != 0) assertEquals((long) xi[i] / yi[i], outl.getAtIndex(VectorBuffers.LE_LONG, i), "int div row " + i);
      }
      assertEquals(-(long) Integer.MIN_VALUE, outl.getAtIndex(VectorBuffers.LE_LONG, 2), "int MIN / -1 fits a long, no overflow");
      MathKernels.integralDivideOverflow(ai, bi, ov);
      assertEquals(0, Bitmap.popcount(ov, n));
      MathKernels.integralDivideScalar(al, -1L, false, outl);
      MathKernels.integralDivideOverflowScalar(al, -1L, false, ov);
      for (int i = 0; i < n; i++) {
        assertEquals(-xl[i], outl.getAtIndex(VectorBuffers.LE_LONG, i));
        assertEquals(xl[i] == Long.MIN_VALUE, Bitmap.isSet(ov, i));
      }
    }
  }

  @Test
  void greatestLeastIgnoreNullsAndUseSparkDoubleOrder() {
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofInts(arena, new int[] {1, 5, 0, 9}, new boolean[] {false, false, true, true});
      VectorBuffers b = ArrowLayout.ofInts(arena, new int[] {3, 0, 7, 0}, new boolean[] {false, true, false, true});
      VectorBuffers c = ArrowLayout.ofInts(arena, new int[] {2, 2, 2, 0}, new boolean[] {false, false, false, true});
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT32, 4);
      MemorySegment valid = ArrowLayout.allocateBitmap(arena, 4);
      MathKernels.pick(Pick.GREATEST, new VectorBuffers[] {a, b, c}, out, valid);
      assertEquals(3, out.getAtIndex(VectorBuffers.LE_INT, 0));
      assertEquals(5, out.getAtIndex(VectorBuffers.LE_INT, 1));
      assertEquals(7, out.getAtIndex(VectorBuffers.LE_INT, 2));
      assertTrue(Bitmap.isSet(valid, 0) && Bitmap.isSet(valid, 1) && Bitmap.isSet(valid, 2));
      assertFalse(Bitmap.isSet(valid, 3), "all null -> null");
      MathKernels.pick(Pick.LEAST, new VectorBuffers[] {a, b, c}, out, valid);
      assertEquals(1, out.getAtIndex(VectorBuffers.LE_INT, 0));
      assertEquals(2, out.getAtIndex(VectorBuffers.LE_INT, 1));
      assertEquals(2, out.getAtIndex(VectorBuffers.LE_INT, 2));
      VectorBuffers d1 = ArrowLayout.ofDoubles(arena, new double[] {Double.NaN, -0.0, 1.0}, null);
      VectorBuffers d2 = ArrowLayout.ofDoubles(arena, new double[] {1.0, 0.0, Double.NEGATIVE_INFINITY}, null);
      MemorySegment outd = ArrowLayout.allocateData(arena, VecType.FLOAT64, 3);
      MathKernels.pick(Pick.GREATEST, new VectorBuffers[] {d1, d2}, outd, valid);
      assertTrue(Double.isNaN(outd.getAtIndex(VectorBuffers.LE_DOUBLE, 0)), "NaN is greatest");
      assertEquals(1.0, outd.getAtIndex(VectorBuffers.LE_DOUBLE, 2));
      MathKernels.pick(Pick.LEAST, new VectorBuffers[] {d1, d2}, outd, valid);
      assertEquals(1.0, outd.getAtIndex(VectorBuffers.LE_DOUBLE, 0), "least of NaN and 1 is 1");
      assertEquals(Double.NEGATIVE_INFINITY, outd.getAtIndex(VectorBuffers.LE_DOUBLE, 2));
      // -0.0 == 0.0: the first operand wins the tie.
      assertEquals(0.0, Math.abs(outd.getAtIndex(VectorBuffers.LE_DOUBLE, 1)));
    }
  }

  /** After normalisation, values Spark treats as equal grouping/join keys have identical bits. */
  @Test
  void normalizeNaNAndZeroCanonicalisesBits() {
    try (Arena arena = Arena.ofConfined()) {
      double otherNaN = Double.longBitsToDouble(0x7ff8000000000001L);
      double[] x = {1.5, -0.0, 0.0, Double.NaN, otherNaN, -2.25, Double.NEGATIVE_INFINITY, Double.MIN_VALUE};
      VectorBuffers a = ArrowLayout.ofDoubles(arena, x, null);
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.FLOAT64, x.length);
      MathKernels.normalizeNaNAndZero(a, out);
      long[] expected = {
        Double.doubleToRawLongBits(1.5), Double.doubleToRawLongBits(0.0), Double.doubleToRawLongBits(0.0),
        Double.doubleToRawLongBits(Double.NaN), Double.doubleToRawLongBits(Double.NaN), Double.doubleToRawLongBits(-2.25),
        Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY), Double.doubleToRawLongBits(Double.MIN_VALUE)
      };
      for (int i = 0; i < x.length; i++) {
        assertEquals(expected[i], out.getAtIndex(VectorBuffers.LE_LONG, i), "bits of normalised " + x[i]);
      }
    }
  }

  @Test
  void nanvlFollowsSparkEval() {
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofDoubles(arena, new double[] {1.5, Double.NaN, Double.NaN, 0.0, 2.0}, new boolean[] {false, false, false, true, false});
      VectorBuffers b = ArrowLayout.ofDoubles(arena, new double[] {9.0, 9.0, 0.0, 9.0, 0.0}, new boolean[] {false, false, true, false, true});
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.FLOAT64, 5);
      MemorySegment valid = ArrowLayout.allocateBitmap(arena, 5);
      MathKernels.nanvl(a, b, out, valid);
      assertEquals(1.5, out.getAtIndex(VectorBuffers.LE_DOUBLE, 0)); // not NaN: a wins
      assertEquals(9.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, 1)); // NaN: b
      assertFalse(Bitmap.isSet(valid, 2), "NaN with a null replacement is null");
      assertFalse(Bitmap.isSet(valid, 3), "null a is null");
      assertTrue(Bitmap.isSet(valid, 4), "non-NaN a with a null b keeps a");
      assertEquals(2.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, 4));
    }
  }
}
