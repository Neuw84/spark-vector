package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.TranscendentalKernels.Fn;
import io.sparkvector.kernels.TranscendentalKernels.Fn2;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * {@link TranscendentalKernels} bit for bit against the {@code Math} / {@code StrictMath} calls Spark's
 * generated code makes, over a grid that includes every domain edge (negative arguments, zero, the
 * log asymptotes, infinities, NaN, huge magnitudes).
 */
class TranscendentalKernelsTest {

  private static double[] grid() {
    Random rnd = new Random(35);
    double[] fixed = {
      0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 10.0, 100.0, 1e-300, -1e-300, 1e300, -1e300,
      Math.PI, -Math.PI, Math.PI / 2, Math.E, 1e-9, 0.9999999, 1.0000001, 3.0, 27.0, 710.0, -745.0,
      Double.MIN_VALUE, Double.MAX_VALUE, -Double.MAX_VALUE, Math.sqrt(Double.MAX_VALUE),
      Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN
    };
    double[] all = new double[fixed.length + 200];
    System.arraycopy(fixed, 0, all, 0, fixed.length);
    for (int i = fixed.length; i < all.length; i++) {
      // A mix of magnitudes and signs.
      double m = Math.pow(10, rnd.nextInt(24) - 12);
      all[i] = (rnd.nextBoolean() ? 1 : -1) * rnd.nextDouble() * m;
    }
    return all;
  }

  private static void assertBits(double expected, double got, String what) {
    assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(got), what + ": " + expected + " vs " + got);
  }

  @Test
  void unaryFunctionsAreBitIdenticalToSparksCalls() {
    double[] xs = grid();
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofDoubles(arena, xs, null);
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.FLOAT64, xs.length);
      for (Fn fn : Fn.values()) {
        TranscendentalKernels.unary(fn, a, out);
        for (int i = 0; i < xs.length; i++) {
          double x = xs[i];
          double expected = switch (fn) {
            case EXP -> StrictMath.exp(x);
            case EXPM1 -> StrictMath.expm1(x);
            case LOG -> StrictMath.log(x);
            case LOG2 -> StrictMath.log(x) / StrictMath.log(2);
            case LOG10 -> StrictMath.log10(x);
            case LOG1P -> StrictMath.log1p(x);
            case CBRT -> Math.cbrt(x);
            case SIN -> Math.sin(x);
            case COS -> Math.cos(x);
            case TAN -> Math.tan(x);
            case ASIN -> Math.asin(x);
            case ACOS -> Math.acos(x);
            case ATAN -> Math.atan(x);
            case SINH -> Math.sinh(x);
            case COSH -> Math.cosh(x);
            case TANH -> Math.tanh(x);
            case ASINH -> Math.abs(x) >= Math.sqrt(Double.MAX_VALUE) - 1
                ? Math.signum(x) * (StrictMath.log(2) + StrictMath.log(Math.abs(x)))
                : StrictMath.log(x + Math.sqrt(x * x + 1.0));
            case ACOSH -> x >= Math.sqrt(Double.MAX_VALUE) ? StrictMath.log(2) + StrictMath.log(x)
                : x < 1 ? Double.NaN : StrictMath.log(x + Math.sqrt(x * x - 1.0));
            case ATANH -> 0.5 * (StrictMath.log1p(x) - StrictMath.log1p(-x));
            case COT -> 1 / Math.tan(x);
            case SEC -> 1 / Math.cos(x);
            case CSC -> 1 / Math.sin(x);
            case DEGREES -> Math.toDegrees(x);
            case RADIANS -> Math.toRadians(x);
          };
          assertBits(expected, out.getAtIndex(VectorBuffers.LE_DOUBLE, i), fn + "(" + x + ")");
        }
      }
      // The asymptote mask is Spark's `!(x <= asymptote)`: NaN passes (log(NaN) is NaN, not null).
      MemorySegment bits = ArrowLayout.allocateBitmap(arena, xs.length);
      TranscendentalKernels.above(a, 0.0, bits);
      for (int i = 0; i < xs.length; i++) {
        assertEquals(!(xs[i] <= 0.0), Bitmap.isSet(bits, i), "above 0 for " + xs[i]);
      }
      TranscendentalKernels.above(a, -1.0, bits);
      for (int i = 0; i < xs.length; i++) {
        assertEquals(!(xs[i] <= -1.0), Bitmap.isSet(bits, i), "above -1 for " + xs[i]);
      }
      assertTrue(Double.isNaN(TranscendentalKernels.asymptote(Fn.SIN)));
      assertEquals(0.0, TranscendentalKernels.asymptote(Fn.LOG10));
      assertEquals(-1.0, TranscendentalKernels.asymptote(Fn.LOG1P));
    }
  }

  @Test
  void binaryFunctionsIncludingScalarOperands() {
    double[] xs = grid();
    double[] ys = new double[xs.length];
    for (int i = 0; i < xs.length; i++) ys[i] = xs[(i * 7 + 3) % xs.length];
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofDoubles(arena, xs, null);
      VectorBuffers b = ArrowLayout.ofDoubles(arena, ys, null);
      MemorySegment out = ArrowLayout.allocateData(arena, VecType.FLOAT64, xs.length);
      for (Fn2 fn : Fn2.values()) {
        TranscendentalKernels.binary(fn, a, b, out);
        for (int i = 0; i < xs.length; i++) {
          assertBits(reference(fn, xs[i], ys[i]), out.getAtIndex(VectorBuffers.LE_DOUBLE, i), fn + "(" + xs[i] + ", " + ys[i] + ")");
        }
        TranscendentalKernels.binaryScalar(fn, a, 2.5, false, out);
        for (int i = 0; i < xs.length; i++) {
          assertBits(reference(fn, xs[i], 2.5), out.getAtIndex(VectorBuffers.LE_DOUBLE, i), fn + "(" + xs[i] + ", 2.5)");
        }
        TranscendentalKernels.binaryScalar(fn, a, 2.5, true, out);
        for (int i = 0; i < xs.length; i++) {
          assertBits(reference(fn, 2.5, xs[i]), out.getAtIndex(VectorBuffers.LE_DOUBLE, i), fn + "(2.5, " + xs[i] + ")");
        }
      }
      // atan2's signed zeros: Spark's + 0.0 folds -0.0 into 0.0 on both operands.
      double[] zs = {-0.0, 0.0, -0.0, 0.0};
      double[] ws = {-0.0, -0.0, 0.0, 0.0};
      VectorBuffers za = ArrowLayout.ofDoubles(arena, zs, null);
      VectorBuffers wa = ArrowLayout.ofDoubles(arena, ws, null);
      TranscendentalKernels.binary(Fn2.ATAN2, za, wa, out);
      for (int i = 0; i < zs.length; i++) {
        assertBits(0.0, out.getAtIndex(VectorBuffers.LE_DOUBLE, i), "atan2 of signed zeros");
      }
      assertFalse(Double.isNaN(TranscendentalKernels.apply(Fn2.POW, 0.0, 0.0)));
      assertEquals(1.0, TranscendentalKernels.apply(Fn2.POW, 0.0, 0.0));
    }
  }

  private static double reference(Fn2 fn, double a, double b) {
    return switch (fn) {
      case POW -> StrictMath.pow(a, b);
      case ATAN2 -> Math.atan2(a + 0.0, b + 0.0);
      case HYPOT -> Math.hypot(a, b);
      case LOG_BASE -> StrictMath.log(b) / StrictMath.log(a);
    };
  }
}
