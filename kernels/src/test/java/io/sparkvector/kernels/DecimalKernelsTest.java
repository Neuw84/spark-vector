package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;

import io.sparkvector.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecimalKernelsTest {

    /**
     * Unscaled values of up to {@code digits} digits, with the awkward ones
     * over-represented.
     */
    private static SegmentVectorBuffers decimals(Arena arena, Random rnd, int n,
            int digits, boolean[] nulls) {
        long bound = DecimalKernels.maxUnscaled(digits);
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
            v[i] = switch (rnd.nextInt(10)) {
                        case 0 -> bound;
                        case 1 -> -bound;
                        case 2 -> 0L;
                        case 3 -> rnd.nextLong(-1000, 1000) * 10 + 5; // exact halves after a rescale by 10
                        case 4 -> rnd.nextLong(-1000, 1000) * 100 + 50;
                        default -> rnd.nextLong(-bound, bound + 1);
                    };
        }
        return ArrowLayout.ofLongs(arena, v, nulls);
    }

    private static void assertSameLongs(MemorySegment exp, MemorySegment act, int n,
            String what) {
        for (int i = 0; i < n; i++) {
            assertEquals(exp.get(VectorBuffers.LE_LONG, (long) i << 3),
                    act.get(VectorBuffers.LE_LONG, (long) i << 3), what + " @" + i);
        }
    }

    @Test
    void rescalingUpAndDownMatchesBigDecimal() {
        Random rnd = new Random(1);
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                VectorBuffers a = decimals(arena, rnd, n, 12, null);
                for (int power = 0; power <= 6; power++) {
                    MemorySegment up = ArrowLayout.allocateData(arena, VecType.INT64, n);
                    DecimalKernels.mulPow10(a, power, up);
                    for (int i = 0; i < n; i++) {
                        assertEquals(a.getLong(i) * DecimalKernels.POW10[power], up.get(VectorBuffers.LE_LONG, (long) i << 3));
                    }
                    MemorySegment exp = ArrowLayout.allocateData(arena, VecType.INT64, n);
                    MemorySegment act = ArrowLayout.allocateData(arena, VecType.INT64, n);
                    ScalarReference.divPow10HalfUp(a, power, exp);
                    DecimalKernels.divPow10HalfUp(a, power, act);
                    assertSameLongs(exp, act, n, "div 10^" + power);
                }
            }
        }
    }

    @Test
    void halfUpRoundsAwayFromZero() {
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofLongs(
                    arena,
                    new long[] {15, -15, 14, -14, 25, -25,
                            5, -5, 0, 149, -151},
                    null);
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT64, a.length());
            DecimalKernels.divPow10HalfUp(a, 1, out);
            long[] expected = {2, -2, 1, -1, 3, -3,
                    1, -1, 0, 15, -15};
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], out.get(VectorBuffers.LE_LONG, (long) i << 3), "@" + i);
            }
        }
    }

    @Test
    void rangeCheckMatchesReferenceIncludingExtremes() {
        Random rnd = new Random(2);
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                long[] v = new long[n];
                for (int i = 0; i < n; i++) {
                    v[i] = switch (rnd.nextInt(6)) {
                                case 0 -> Long.MIN_VALUE;
                                case 1 -> Long.MAX_VALUE;
                                case 2 -> 999_999L;
                                case 3 -> -1_000_000L;
                                default -> rnd.nextLong(-2_000_000L, 2_000_000L);
                            };
                }
                VectorBuffers a = ArrowLayout.ofLongs(arena, v, null);
                MemorySegment exp = ArrowLayout.allocateBitmap(arena, n);
                MemorySegment act = ArrowLayout.allocateBitmap(arena, n);
                act.fill((byte) 0xFF); // every bit must be written
                int ec = ScalarReference.outOfRange(a, 999_999L, exp);
                int ac = DecimalKernels.outOfRange(a, 999_999L, act);
                assertEquals(ec, ac);
                TestData.assertBitmapEquals(exp, act, n, "outOfRange");
            }
        }
    }

    @Test
    void divisionMatchesSparkSemanticsOnBothPaths() {
        Random rnd = new Random(3);
        int[][] shapes = {{2, 2, 8, 13}, {4, 0, 6, 12}, {2, 3, 5, 10},
                {0, 0, 6, 18}, {6, 2, 2, 7}};
        for (int n : TestData.LENGTHS) {
            for (int[] shape : shapes) {
                int s1 = shape[0], s2 = shape[1], rs = shape[2], rp = shape[3];
                try (Arena arena = Arena.ofConfined()) {
                    VectorBuffers a = decimals(arena, rnd, n, 12, null);
                    long[] bv = new long[n];
                    for (int i = 0; i < n; i++) {
                        bv[i] = switch (rnd.nextInt(8)) {
                                    case 0 -> 0L;
                                    case 1 -> 3L;
                                    case 2 -> -7L;
                                    default -> rnd.nextLong(-100000, 100000);
                                };
                    }
                    VectorBuffers b = ArrowLayout.ofLongs(arena, bv, null);
                    MemorySegment exp = ArrowLayout.allocateData(arena, VecType.INT64, n);
                    MemorySegment act = ArrowLayout.allocateData(arena, VecType.INT64, n);
                    MemorySegment expBits = ArrowLayout.allocateBitmap(arena, n);
                    MemorySegment actBits = ArrowLayout.allocateBitmap(arena, n);
                    int ec = ScalarReference.divide(a, b, s1, s2, rs, rp,
                            exp, expBits);
                    int ac = DecimalKernels.divide(a, b, s1, s2, rs, rp,
                            act, actBits);
                    assertEquals(ec, ac, "overflow count " + java.util.Arrays.toString(shape));
                    assertSameLongs(exp, act, n, "divide " + java.util.Arrays.toString(shape));
                    TestData.assertBitmapEquals(expBits, actBits, n, "overflow bits");
                }
            }
        }
    }

    @Test
    void divisionSlowPathIsTakenWhenTheScaledDividendOverflows() {
        // 10^17 scaled by 10^6 does not fit a long: the BigDecimal path must agree with the oracle.
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofLongs(arena, new long[] {100_000_000_000_000_000L, -99_999_999_999_999_999L, 7}, null);
            VectorBuffers b = ArrowLayout.ofLongs(arena, new long[] {30_000_000_000_000_000L, 7, 3}, null);
            assertFalse(DecimalKernels.fitsWhenScaled(a.getLong(0), 6));
            assertTrue(DecimalKernels.fitsWhenScaled(7, 6));
            MemorySegment exp = ArrowLayout.allocateData(arena, VecType.INT64, 3);
            MemorySegment act = ArrowLayout.allocateData(arena, VecType.INT64, 3);
            MemorySegment eb = ArrowLayout.allocateBitmap(arena, 3);
            MemorySegment ab = ArrowLayout.allocateBitmap(arena, 3);
            ScalarReference.divide(a, b, 2, 2, 6, 18,
                    exp, eb);
            DecimalKernels.divide(a, b, 2, 2, 6, 18,
                    act, ab);
            assertSameLongs(exp, act, 3, "slow path");
            TestData.assertBitmapEquals(eb, ab, 3, "slow path bits");
        }
    }

    @Test
    void toDoubleMatchesBigDecimalDoubleValue() {
        Random rnd = new Random(4);
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                VectorBuffers a = decimals(arena, rnd, n, 18, null);
                for (int scale : new int[] {0, 2, 9, 18}) {
                    MemorySegment exp = ArrowLayout.allocateData(arena, VecType.FLOAT64, n);
                    MemorySegment act = ArrowLayout.allocateData(arena, VecType.FLOAT64, n);
                    ScalarReference.toDouble(a, scale, exp);
                    DecimalKernels.toDouble(a, scale, act);
                    for (int i = 0; i < n; i++) {
                        assertEquals(Double.doubleToLongBits(exp.get(VectorBuffers.LE_DOUBLE, (long) i << 3)), Double.doubleToLongBits(act.get(VectorBuffers.LE_DOUBLE, (long) i << 3)),
                                "scale "
                                        + scale
                                        + " @"
                                        + i
                                        + " value "
                                        + a.getLong(i));
                    }
                }
            }
        }
    }

    @Test
    void integralToDecimalMatchesReference() {
        Random rnd = new Random(5);
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                int[] iv = new int[n];
                long[] lv = new long[n];
                for (int i = 0; i < n; i++) {
                    iv[i] = rnd.nextInt(4) == 0
                            ? (rnd.nextBoolean() ? Integer.MAX_VALUE : Integer.MIN_VALUE)
                            : rnd.nextInt(-100000, 100000);
                    lv[i] = rnd.nextInt(4) == 0
                            ? (rnd.nextBoolean() ? Long.MAX_VALUE : Long.MIN_VALUE)
                            : rnd.nextLong(-100000, 100000);
                }
                for (VectorBuffers a : new VectorBuffers[] {ArrowLayout.ofInts(arena, iv, null), ArrowLayout.ofLongs(arena, lv, null)}) {
                    for (int[] ps : new int[][] {{10, 2}, {7, 3}, {18, 0},
                            {18, 10}}) {
                        MemorySegment exp = ArrowLayout.allocateData(arena, VecType.INT64, n);
                        MemorySegment act = ArrowLayout.allocateData(arena, VecType.INT64, n);
                        MemorySegment eb = ArrowLayout.allocateBitmap(arena, n);
                        MemorySegment ab = ArrowLayout.allocateBitmap(arena, n);
                        int ec = ScalarReference.fromIntegral(a, ps[0], ps[1], exp, eb);
                        int ac = DecimalKernels.fromIntegral(a, ps[0], ps[1], act, ab);
                        assertEquals(ec, ac, a.type() + " -> decimal" + java.util.Arrays.toString(ps));
                        assertSameLongs(exp, act, n, a.type() + " -> decimal" + java.util.Arrays.toString(ps));
                        TestData.assertBitmapEquals(eb, ab, n, "invalid bits");
                    }
                }
            }
        }
    }

    @Test
    void doubleToDecimalUsesSparksStringRoundTrip() {
        try (Arena arena = Arena.ofConfined()) {
            double[] d = {0.1, 1.005, -2.5, 1e10, Double.NaN,
                    Double.POSITIVE_INFINITY, 123456.789, -0.0};
            VectorBuffers a = ArrowLayout.ofDoubles(arena, d, null);
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT64, d.length);
            MemorySegment bad = ArrowLayout.allocateBitmap(arena, d.length);
            int count = DecimalKernels.fromDouble(a, 10, 2, out, bad);
            assertEquals(3, count); // NaN, Infinity, 1e10 out of precision 10
            for (int i = 0; i < d.length; i++) {
                if (!Bitmap.isSet(bad, i)) {
                    long expected = new BigDecimal(Double.toString(d[i]))
                            .setScale(2, RoundingMode.HALF_UP)
                            .unscaledValue()
                            .longValueExact();
                    assertEquals(expected, out.get(VectorBuffers.LE_LONG, (long) i << 3), "@" + i);
                }
            }
            assertEquals(101L, out.get(VectorBuffers.LE_LONG, 8)); // 1.005 -> "1.005" -> 1.01
            assertEquals(-250L, out.get(VectorBuffers.LE_LONG, 16));
        }
    }

    @Test
    void toIntegralTruncatesTowardZero() {
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofLongs(
                    arena,
                    new long[] {1999, -1999, 50, -50, 300_000_000_000L},
                    null);
            MemorySegment l = ArrowLayout.allocateData(arena, VecType.INT64, 5);
            DecimalKernels.toLong(a, 2, l);
            assertEquals(19L, l.get(VectorBuffers.LE_LONG, 0));
            assertEquals(-19L, l.get(VectorBuffers.LE_LONG, 8));
            assertEquals(0L, l.get(VectorBuffers.LE_LONG, 16));
            assertEquals(0L, l.get(VectorBuffers.LE_LONG, 24));
            MemorySegment ints = ArrowLayout.allocateData(arena, VecType.INT32, 5);
            MemorySegment bad = ArrowLayout.allocateBitmap(arena, 5);
            assertEquals(1, DecimalKernels.toInt(a, 2, ints, bad));
            assertTrue(Bitmap.isSet(bad, 4));
            assertEquals(19, ints.get(VectorBuffers.LE_INT, 0));
            assertEquals(-19, ints.get(VectorBuffers.LE_INT, 4));
        }
    }
}
