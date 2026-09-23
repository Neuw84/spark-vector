package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import io.sparkvector.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link OverflowKernels} against {@code Math.*Exact} at the type boundaries
 * and on random data.
 */
class OverflowKernelsTest {

    private static void checkColumns(ArithOp op, VectorBuffers a, VectorBuffers b,
            Arena arena) {
        int n = a.length();
        MemorySegment r = ArrowLayout.allocateData(arena, a.type(), n);
        ArithKernels.arith(op, a, b, r);
        MemorySegment mask = ArrowLayout.allocateBitmap(arena, n);
        OverflowKernels.overflow(op, a, b, r, mask);
        for (int i = 0; i < n; i++) {
            assertEquals(ScalarReference.overflows(op, a, b, null, false, i),
                    Bitmap.isSet(mask, i), op + " row " + i);
        }
    }

    private static void checkScalar(ArithOp op, VectorBuffers a, Number s,
            Arena arena) {
        int n = a.length();
        MemorySegment r = ArrowLayout.allocateData(arena, a.type(), n);
        ArithKernels.arithScalar(op, a, s, r);
        MemorySegment mask = ArrowLayout.allocateBitmap(arena, n);
        OverflowKernels.overflowScalar(op, a, s, r, mask);
        for (int i = 0; i < n; i++) {
            assertEquals(ScalarReference.overflows(op, a, null, s, false, i),
                    Bitmap.isSet(mask, i), op + " " + s + " row " + i);
        }
        MemorySegment r2 = ArrowLayout.allocateData(arena, a.type(), n);
        ArithKernels.scalarArith(op, s, a, r2);
        MemorySegment mask2 = ArrowLayout.allocateBitmap(arena, n);
        OverflowKernels.scalarOverflow(op, s, a, r2, mask2);
        for (int i = 0; i < n; i++) {
            assertEquals(ScalarReference.overflows(op, a, null, s, true, i),
                    Bitmap.isSet(mask2, i), s + " " + op + " row " + i);
        }
    }

    @Test
    void intBoundaries() {
        try (Arena arena = Arena.ofConfined()) {
            int max = Integer.MAX_VALUE;
            int min = Integer.MIN_VALUE;
            int[] xs = {
                max,
                min,
                max,
                min,
                0,
                -1,
                46341,
                -46341,
                65536,
                max,
                min,
                7,
                max - 1,
                min + 1,
                1 << 30,
                -(1 << 30)
            };
            int[] ys = {
                1,
                -1,
                -1,
                1,
                0,
                min,
                46341,
                46341,
                65536,
                max,
                min,
                -7,
                1,
                -1,
                2,
                2
            };
            VectorBuffers a = ArrowLayout.ofInts(arena, xs, null);
            VectorBuffers b = ArrowLayout.ofInts(arena, ys, null);
            for (ArithOp op : new ArithOp[] {ArithOp.ADD, ArithOp.SUB, ArithOp.MUL}) {
                checkColumns(op, a, b, arena);
                checkScalar(op, a, 2, arena);
                checkScalar(op, a, -1, arena);
                checkScalar(op, a, max, arena);
                checkScalar(op, a, min, arena);
            }
            // Spot checks the reference agrees with: MAX + 1 and MIN - 1 overflow, MAX + 0 does not,
            // 46341 * 46341 overflows an int, 65536 * 65536 too, -1 * MIN too.
            MemorySegment r = ArrowLayout.allocateData(arena, VecType.INT32, xs.length);
            ArithKernels.arith(ArithOp.ADD, a, b, r);
            MemorySegment mask = ArrowLayout.allocateBitmap(arena, xs.length);
            OverflowKernels.overflow(ArithOp.ADD, a, b, r, mask);
            assertTrue(Bitmap.isSet(mask, 0), "MAX + 1");
            assertTrue(Bitmap.isSet(mask, 1), "MIN + -1");
            assertFalse(Bitmap.isSet(mask, 2), "MAX + -1");
            assertFalse(Bitmap.isSet(mask, 4), "0 + 0");
            ArithKernels.arith(ArithOp.MUL, a, b, r);
            OverflowKernels.overflow(ArithOp.MUL, a, b, r, mask);
            assertTrue(Bitmap.isSet(mask, 6), "46341^2");
            assertTrue(Bitmap.isSet(mask, 8), "65536^2");
            assertTrue(Bitmap.isSet(mask, 5), "-1 * MIN");
            assertFalse(Bitmap.isSet(mask, 11), "7 * -7");

            MemorySegment neg = ArrowLayout.allocateBitmap(arena, xs.length);
            OverflowKernels.negateOverflow(a, neg);
            for (int i = 0; i < xs.length; i++) {
                assertEquals(ScalarReference.negateOverflows(a, i),
                        Bitmap.isSet(neg, i), "negate row " + i);
            }
            assertTrue(Bitmap.isSet(neg, 1));
            assertFalse(Bitmap.isSet(neg, 0));
        }
    }

    @Test
    void longBoundaries() {
        try (Arena arena = Arena.ofConfined()) {
            long max = Long.MAX_VALUE;
            long min = Long.MIN_VALUE;
            long[] xs = {
                max,
                min,
                max,
                min,
                0,
                -1,
                3037000500L,
                -3037000500L,
                1L << 32,
                max,
                min,
                7,
                max - 1,
                min + 1,
                1L << 62,
                -(1L << 62)
            };
            long[] ys = {
                1,
                -1,
                -1,
                1,
                0,
                min,
                3037000500L,
                3037000500L,
                1L << 32,
                max,
                min,
                -7,
                1,
                -1,
                2,
                2
            };
            VectorBuffers a = ArrowLayout.ofLongs(arena, xs, null);
            VectorBuffers b = ArrowLayout.ofLongs(arena, ys, null);
            for (ArithOp op : new ArithOp[] {ArithOp.ADD, ArithOp.SUB, ArithOp.MUL}) {
                checkColumns(op, a, b, arena);
                checkScalar(op, a, 2L, arena);
                checkScalar(op, a, -1L, arena);
                checkScalar(op, a, max, arena);
                checkScalar(op, a, min, arena);
            }
            MemorySegment neg = ArrowLayout.allocateBitmap(arena, xs.length);
            OverflowKernels.negateOverflow(a, neg);
            for (int i = 0; i < xs.length; i++) {
                assertEquals(ScalarReference.negateOverflows(a, i),
                        Bitmap.isSet(neg, i), "negate row " + i);
            }
        }
    }

    @Test
    void randomDataMatchesTheReferenceAcrossFullAndPartialLaneBlocks() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(5);
            for (int n : new int[] {1, 7, 64, 65, 1000, 4096}) {
                int[] xi = new int[n];
                int[] yi = new int[n];
                long[] xl = new long[n];
                long[] yl = new long[n];
                for (int i = 0; i < n; i++) {
                    // Mix magnitudes so about half the products and a fair share of the sums overflow.
                    xi[i] = rnd.nextBoolean() ? rnd.nextInt() : rnd.nextInt(100000) - 50000;
                    yi[i] = rnd.nextBoolean() ? rnd.nextInt() : rnd.nextInt(100000) - 50000;
                    xl[i] = rnd.nextBoolean() ? rnd.nextLong() : rnd.nextInt();
                    yl[i] = rnd.nextBoolean() ? rnd.nextLong() : rnd.nextInt();
                }
                VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null);
                VectorBuffers bi = ArrowLayout.ofInts(arena, yi, null);
                VectorBuffers al = ArrowLayout.ofLongs(arena, xl, null);
                VectorBuffers bl = ArrowLayout.ofLongs(arena, yl, null);
                for (ArithOp op : new ArithOp[] {ArithOp.ADD, ArithOp.SUB, ArithOp.MUL}) {
                    checkColumns(op, ai, bi, arena);
                    checkColumns(op, al, bl, arena);
                    checkScalar(op, ai, rnd.nextInt(), arena);
                    checkScalar(op, al, rnd.nextLong(), arena);
                }
            }
        }
    }
}
