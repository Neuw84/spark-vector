/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package io.vecruntime.kernels;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Arrays;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;

import static io.vecruntime.kernels.Species.D;

/**
 * Predicates the comparison kernels do not cover: {@code isnan}, membership in
 * a literal set ({@code InSet}, the optimizer's rewrite of a long {@code IN}
 * list) and comparisons between BOOL columns, whose values are already packed
 * bitmaps. Every kernel writes a result bitmap over all {@code n} rows and
 * leaves validity to the caller; null lanes hold arbitrary values.
 */
public final class PredicateKernels {
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private PredicateKernels() {}

    // ---------------------------------------------------------------- isnan

    /**
     * {@code out[i] = isNaN(a[i])} for FLOAT64 lanes, as a Vector API {@code
     * IS_NAN} test packed into the bitmap.
     */
    public static void isNaN(VectorBuffers a, MemorySegment out) {
        if (a.type() != VecType.FLOAT64) {
            throw new IllegalArgumentException("isnan needs FLOAT64 lanes, got " + a.type());
        }
        int n = a.length();
        MemorySegment data = a.data();
        int lanes = D.length();
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            int base = w << 6;
            int end = Math.min(n, base + 64);
            long word = 0L;
            int i = base;
            for (; i + lanes <= end; i += lanes) {
                long bits = DoubleVector.fromMemorySegment(D, data, (long) i << 3, LE)
                        .test(VectorOperators.IS_NAN)
                        .toLong();
                word |= bits << (i - base);
            }
            if (i < end) {
                VectorMask<Double> m = D.indexInRange(i, end);
                long bits = DoubleVector.fromMemorySegment(D, data, (long) i << 3, LE, m)
                        .test(VectorOperators.IS_NAN)
                        .and(m)
                        .toLong();
                word |= bits << (i - base);
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    // ---------------------------------------------------------------- InSet

    /**
     * {@code out[i] = a[i] in keys} for INT32 / INT64 / FLOAT64 lanes. {@code
     * keys} is sorted and holds the set's values widened to longs -- for
     * FLOAT64 the {@link Double#doubleToLongBits} images, so membership is
     * boxed {@code Double.equals}: NaN is a member of a set holding NaN and
     * {@code -0.0} is not a member of a set holding {@code 0.0}, exactly as
     * Spark's {@code InSet} over its {@code Set[Any]}. One binary search per
     * lane.
     */
    public static void inSet(VectorBuffers a, long[] keys, MemorySegment out) {
        int n = a.length();
        MemorySegment data = a.data();
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            int base = w << 6;
            int end = Math.min(n, base + 64);
            long word = 0L;
            switch (a.type()) {
                case INT32 -> {
                    for (int i = base; i < end; i++) {
                        if (Arrays.binarySearch(keys, data.getAtIndex(VectorBuffers.LE_INT, i)) >= 0) {
                            word |= 1L << (i - base);
                        }
                    }
                }
                case INT64 -> {
                    for (int i = base; i < end; i++) {
                        if (Arrays.binarySearch(keys, data.getAtIndex(VectorBuffers.LE_LONG, i)) >= 0) {
                            word |= 1L << (i - base);
                        }
                    }
                }
                case FLOAT64 -> {
                    for (int i = base; i < end; i++) {
                        long bits = Double.doubleToLongBits(data.getAtIndex(VectorBuffers.LE_DOUBLE, i));
                        if (Arrays.binarySearch(keys, bits) >= 0) {
                            word |= 1L << (i - base);
                        }
                    }
                }
                default -> throw new IllegalArgumentException("InSet over " + a.type());
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /** The sorted key array {@link #inSet} expects for a set of doubles. */
    public static long[] doubleKeys(double[] values) {
        long[] keys = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            keys[i] = Double.doubleToLongBits(values[i]);
        }
        Arrays.sort(keys);
        return keys;
    }

    // ---------------------------------------------------------------- booleans

    /** {@code out[i] = a[i] op b[i]} on two BOOL bitmaps ({@code false < true}). */
    public static void compareBool(MemorySegment a, MemorySegment b, CompareOp op,
            MemorySegment out, int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            long x = Bitmap.wordAt(a, w, n);
            long y = Bitmap.wordAt(b, w, n);
            Bitmap.setWord(out, w, n, boolWord(op, x, y));
        }
    }

    /** {@code out[i] = a[i] op value} on a BOOL bitmap. */
    public static void compareBoolScalar(MemorySegment a, boolean value, CompareOp op,
            MemorySegment out, int n) {
        int words = Bitmap.wordsFor(n);
        long y = value ? -1L : 0L;
        for (int w = 0; w < words; w++) {
            long x = Bitmap.wordAt(a, w, n);
            Bitmap.setWord(out, w, n, boolWord(op, x, y));
        }
    }

    /** One 64-row word of a boolean comparison. */
    static long boolWord(CompareOp op, long x, long y) {
        return switch (op) {
            case EQ -> ~(x ^ y);
            case NE -> x ^ y;
            case LT -> ~x & y;
            case LE -> ~x | y;
            case GT -> x & ~y;
            case GE -> x | ~y;
        };
    }
}
