/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link PredicateKernels} against scalar definitions at lengths that are not
 * lane or word multiples.
 */
class PredicateKernelsTest {

    @Test
    void isNaNPacksEveryLaneShape() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(7);
            for (int n : new int[] {1, 3, 7, 64, 65, 100,
                    129, 1000, 4096, 4099}) {
                double[] xs = new double[n];
                for (int i = 0; i < n; i++) {
                    xs[i] = switch (rnd.nextInt(6)) {
                                case 0 -> Double.NaN;
                                case 1 -> Double.longBitsToDouble(0x7ff8000000000001L); // a non-canonical NaN
                                case 2 -> -0.0;
                                case 3 -> Double.POSITIVE_INFINITY;
                                default -> rnd.nextGaussian();
                            };
                }
                VectorBuffers a = ArrowLayout.ofDoubles(arena, xs, null);
                MemorySegment out = ArrowLayout.allocateBitmap(arena, n);
                Bitmap.fill(out, n, true); // the kernel must overwrite every word
                PredicateKernels.isNaN(a, out);
                for (int i = 0; i < n; i++) {
                    assertEquals(Double.isNaN(xs[i]), Bitmap.isSet(out, i), "n=" + n + " row " + i);
                }
            }
        }
    }

    @Test
    void inSetIsBoxedEqualityPerLaneType() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(9);
            int n = 777;
            int[] xi = new int[n];
            long[] xl = new long[n];
            double[] xd = new double[n];
            for (int i = 0; i < n; i++) {
                xi[i] = rnd.nextInt(50) - 25;
                xl[i] = rnd.nextBoolean() ? rnd.nextLong() : rnd.nextInt(50) - 25;
                xd[i] = switch (rnd.nextInt(5)) {
                            case 0 -> Double.NaN;
                            case 1 -> -0.0;
                            case 2 -> 0.0;
                            default -> (rnd.nextInt(40) - 20) / 4.0;
                        };
            }
            xi[0] = Integer.MIN_VALUE;
            xl[0] = Long.MIN_VALUE;
            long[] intKeys = {-25, -3, 0, 7, 12, 24,
                    Integer.MIN_VALUE, Integer.MAX_VALUE};
            Arrays.sort(intKeys);
            long[] longKeys = {Long.MIN_VALUE, -1L, 0L, 5L, 1L << 40,
                    xl[3], xl[10]};
            Arrays.sort(longKeys);
            double[] doubleSet = {Double.NaN, 0.0, 1.25, -2.5, 4.0};
            long[] doubleKeys = PredicateKernels.doubleKeys(doubleSet);
            Set<Long> intSet = new HashSet<>();
            for (long k : intKeys) {
                intSet.add(k);
            }
            Set<Long> longSet = new HashSet<>();
            for (long k : longKeys) {
                longSet.add(k);
            }
            Set<Double> boxed = new HashSet<>();
            for (double d : doubleSet) {
                boxed.add(d); // boxed Double.equals: NaN in, -0.0 distinct from 0.0
            }
            VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null), al = ArrowLayout.ofLongs(arena, xl, null), ad = ArrowLayout.ofDoubles(arena, xd, null);
            MemorySegment out = ArrowLayout.allocateBitmap(arena, n);
            PredicateKernels.inSet(ai, intKeys, out);
            for (int i = 0; i < n; i++) {
                assertEquals(intSet.contains((long) xi[i]), Bitmap.isSet(out, i), "int row " + i);
            }
            PredicateKernels.inSet(al, longKeys, out);
            for (int i = 0; i < n; i++) {
                assertEquals(longSet.contains(xl[i]), Bitmap.isSet(out, i), "long row " + i);
            }
            PredicateKernels.inSet(ad, doubleKeys, out);
            for (int i = 0; i < n; i++) {
                assertEquals(boxed.contains(xd[i]), Bitmap.isSet(out, i), "double row " + i + " = " + xd[i]);
            }
            // Spot checks of the semantics the doc promises.
            VectorBuffers edge = ArrowLayout.ofDoubles(
                    arena,
                    new double[] {Double.NaN, -0.0, 0.0, Double.longBitsToDouble(0x7ff8000000000001L)},
                    null);
            PredicateKernels.inSet(edge, doubleKeys, out);
            assertTrue(Bitmap.isSet(out, 0), "NaN is a member of a set holding NaN");
            assertFalse(Bitmap.isSet(out, 1), "-0.0 is not a member of a set holding 0.0");
            assertTrue(Bitmap.isSet(out, 2));
            assertTrue(Bitmap.isSet(out, 3), "every NaN payload canonicalises like Double.equals");
        }
    }

    @Test
    void booleanComparisonsMatchTheTruthTable() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(3);
            int n = 203;
            boolean[] x = new boolean[n], y = new boolean[n];
            MemorySegment a = ArrowLayout.allocateBitmap(arena, n), b = ArrowLayout.allocateBitmap(arena, n), out = ArrowLayout.allocateBitmap(arena, n);
            for (int i = 0; i < n; i++) {
                x[i] = rnd.nextBoolean();
                y[i] = rnd.nextBoolean();
                Bitmap.setTo(a, i, x[i]);
                Bitmap.setTo(b, i, y[i]);
            }
            for (CompareOp op : CompareOp.values()) {
                PredicateKernels.compareBool(a, b, op, out, n);
                for (int i = 0; i < n; i++) {
                    assertEquals(ref(op, x[i], y[i]), Bitmap.isSet(out, i), op + " row " + i);
                }
                for (boolean v : new boolean[] {false, true}) {
                    PredicateKernels.compareBoolScalar(a, v, op, out, n);
                    for (int i = 0; i < n; i++) {
                        assertEquals(ref(op, x[i], v), Bitmap.isSet(out, i), op + " scalar " + v + " row " + i);
                    }
                }
            }
        }
    }

    private static boolean ref(CompareOp op, boolean x, boolean y) {
        int c = Boolean.compare(x, y);
        return switch (op) {
            case EQ -> c == 0;
            case NE -> c != 0;
            case LT -> c < 0;
            case LE -> c <= 0;
            case GT -> c > 0;
            case GE -> c >= 0;
        };
    }
}
