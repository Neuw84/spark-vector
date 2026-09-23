package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Random column generators and bitmap assertions shared by kernel tests. */
final class TestData {

    static final int[] LENGTHS = {
        0,
        1,
        3,
        7,
        8,
        15,
        16,
        31,
        63,
        64,
        65,
        100,
        127,
        128,
        129,
        1000,
        1023,
        4097
    };

    private TestData() {}

    static boolean[] nulls(Random rnd, int n, double fraction) {
        boolean[] nulls = new boolean[n];
        for (int i = 0; i < n; i++) {
            nulls[i] = rnd.nextDouble() < fraction;
        }
        return nulls;
    }

    /** Small-range ints so that equality comparisons hit often. */
    static SegmentVectorBuffers ints(Arena arena, Random rnd, int n,
            boolean[] nulls) {
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextInt(-20, 20);
        }
        return ArrowLayout.ofInts(arena, v, nulls);
    }

    static SegmentVectorBuffers longs(Arena arena, Random rnd, int n,
            boolean[] nulls) {
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextInt(3) == 0 ? rnd.nextLong() : rnd.nextLong(-20, 20);
        }
        return ArrowLayout.ofLongs(arena, v, nulls);
    }

    /*
     * Doubles including NaN, infinities and signed zeros to exercise NaN-safe
     * ordering.
     */
/**
            * Random 128-bit unscaled values: a mix of small values, values straddling
            * the limb boundary, and the extremes (+-(10^38 - 1), -2^127,
            * Long.MIN_VALUE in either limb).
            */
    static SegmentVectorBuffers decimal128s(Arena arena, Random rnd, int n,
            boolean[] nulls) {
        java.math.BigInteger[] v = new java.math.BigInteger[n];
        for (int i = 0; i < n; i++) {
            v[i] = randomDecimal128(rnd);
        }
        return ArrowLayout.ofDecimal128(arena, v, nulls);
    }

    static final java.math.BigInteger MAX_DECIMAL38 = java.math.BigInteger.TEN.pow(38).subtract(java.math.BigInteger.ONE);

    static java.math.BigInteger randomDecimal128(Random rnd) {
        switch (rnd.nextInt(12)) {
            case 0:
                return MAX_DECIMAL38;
            case 1:
                return MAX_DECIMAL38.negate();
            case 2:
                return java.math.BigInteger.ONE.shiftLeft(127).negate();
            case 3:
                return java.math.BigInteger.valueOf(Long.MIN_VALUE);
            case 4:
                return java.math.BigInteger.valueOf(Long.MIN_VALUE).shiftLeft(64);
            case 5:
                return java.math.BigInteger.valueOf(Long.MAX_VALUE).add(java.math.BigInteger.ONE);
            case 6:
                return java.math.BigInteger.ZERO;
            case 7:
                return java.math.BigInteger.valueOf(rnd.nextInt(2000) - 1000);
            default:
                return new java.math.BigInteger(rnd.nextInt(127), rnd).multiply(java.math.BigInteger.valueOf(rnd.nextBoolean() ? 1 : -1));
        }
    }

    static SegmentVectorBuffers doubles(Arena arena, Random rnd, int n,
            boolean[] nulls) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = switch (rnd.nextInt(12)) {
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

    static void assertBitmapEquals(MemorySegment expected, MemorySegment actual, int n,
            String what) {
        for (int i = 0; i < n; i++) {
            assertEquals(Bitmap.isSet(expected, i), Bitmap.isSet(actual, i), what + " bit " + i + " of " + n);
        }
    }
}
