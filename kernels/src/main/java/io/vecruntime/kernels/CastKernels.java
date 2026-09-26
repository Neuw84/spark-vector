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

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Widening numeric casts (INT32 to INT64/FLOAT64, INT64 to FLOAT64). These
 * never fail or lose integer precision beyond what Spark itself does (long to
 * double rounds), so they are valid in both legacy and ANSI mode. Validity is
 * unchanged and is shared by the caller.
 *
 * <p>Widening halves the lane count, so the int source is loaded with a species
 * of half the preferred bit size and converted with {@code convertShape} into
 * the full-size destination.
 */
public final class CastKernels {

    static final VectorSpecies<Long> L = Species.L;
    static final VectorSpecies<Double> D = Species.D;

    /** Int species with as many lanes as {@link #D} / {@link #L}. */
    static final VectorSpecies<Integer> IH = Species.IH;

    static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private CastKernels() {}

    /** Casts {@code a} to {@code target}, writing values into {@code out}. */
    public static void cast(VectorBuffers a, VecType target, MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                switch (target) {
                    case INT64 -> i32ToI64(a.data(), n, out);
                    case FLOAT64 -> i32ToF64(a.data(), n, out);
                    default -> throw unsupported(a.type(), target);
                }
            }
            case INT64 -> {
                if (target != VecType.FLOAT64) {
                    throw unsupported(a.type(), target);
                }
                i64ToF64(a.data(), n, out);
            }
            default -> throw unsupported(a.type(), target);
        }
    }

    public static boolean isSupported(VecType from, VecType to) {
        return (from == VecType.INT32 && (to == VecType.INT64 || to == VecType.FLOAT64))
                || (from == VecType.INT64 && to == VecType.FLOAT64);
    }

    // ---------------------------------------------------------------- narrowing, booleans, days (Spark's Cast)

    /**
     * Narrowing casts with Spark's legacy value rule -- Java's {@code (int)} /
     * {@code (long)}: a long wraps, a double truncates toward zero and
     * saturates, NaN becomes 0 -- and, when {@code overflow} is given, a bit
     * per row whose value is outside the target's range by Spark's ANSI test
     * ({@code v != (int) v} for a long; {@code floor(d) <= MAX && ceil(d) >=
     * MIN} for a double, so NaN and the infinities are out of range).
     */
    public static void narrow(VectorBuffers a, VecType target, MemorySegment out,
            MemorySegment overflow) {
        int n = a.length();
        MemorySegment d = a.data();
        switch (a.type()) {
            case INT64 -> {
                if (target != VecType.INT32) {
                    throw unsupported(a.type(), target);
                }
                for (int i = 0; i < n; i++) {
                    long v = d.getAtIndex(VectorBuffers.LE_LONG, i);
                    int r = (int) v;
                    out.setAtIndex(VectorBuffers.LE_INT, i, r);
                    if (overflow != null && r != v) {
                        Bitmap.set(overflow, i);
                    }
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    double v = d.getAtIndex(VectorBuffers.LE_DOUBLE, i);
                    switch (target) {
                        case INT32 -> {
                            out.setAtIndex(VectorBuffers.LE_INT, i, (int) v);
                            if (overflow != null && !(Math.floor(v) <= Integer.MAX_VALUE && Math.ceil(v) >= Integer.MIN_VALUE)) {
                                Bitmap.set(overflow, i);
                            }
                        }
                        case INT64 -> {
                            out.setAtIndex(VectorBuffers.LE_LONG, i, (long) v);
                            if (overflow != null && !(Math.floor(v) <= Long.MAX_VALUE && Math.ceil(v) >= Long.MIN_VALUE)) {
                                Bitmap.set(overflow, i);
                            }
                        }
                        default -> throw unsupported(a.type(), target);
                    }
                }
            }
            default -> throw unsupported(a.type(), target);
        }
    }

    /**
     * Spark's numeric-to-boolean cast: {@code v != 0}, so NaN is true. Sets
     * bits in {@code outBits}.
     */
    public static void toBool(VectorBuffers a, MemorySegment outBits) {
        int n = a.length();
        MemorySegment d = a.data();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    if (d.getAtIndex(VectorBuffers.LE_INT, i) != 0) {
                        Bitmap.set(outBits, i);
                    }
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    if (d.getAtIndex(VectorBuffers.LE_LONG, i) != 0L) {
                        Bitmap.set(outBits, i);
                    }
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    if (d.getAtIndex(VectorBuffers.LE_DOUBLE, i) != 0.0) {
                        Bitmap.set(outBits, i);
                    }
                }
            }
            default -> throw unsupported(a.type(), VecType.BOOL);
        }
    }

    /** Spark's boolean-to-numeric cast: true is 1, false is 0. */
    public static void fromBool(VectorBuffers a, VecType target, MemorySegment out) {
        int n = a.length();
        MemorySegment bits = a.data();
        switch (target) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_INT, i, Bitmap.isSet(bits, i) ? 1 : 0);
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_LONG, i, Bitmap.isSet(bits, i) ? 1L : 0L);
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_DOUBLE, i, Bitmap.isSet(bits, i) ? 1.0 : 0.0);
                }
            }
            default -> throw unsupported(VecType.BOOL, target);
        }
    }

    /**
     * Spark's date-to-timestamp cast under a fixed offset: local midnight as
     * micros since the epoch.
     */
    public static void daysToMicros(VectorBuffers a, long offsetMicros, MemorySegment out) {
        int n = a.length();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_LONG, i, d.getAtIndex(VectorBuffers.LE_INT, i) * 86_400_000_000L - offsetMicros);
        }
    }

    private static IllegalArgumentException unsupported(VecType from, VecType to) {
        return new IllegalArgumentException("unsupported cast " + from + " -> " + to);
    }

    /**
     * Widens {@code n} little-endian int32 values of {@code in} into int64
     * lanes of {@code out} (Vector API {@code I2L}): the Iceberg adapter's lane
     * for a small decimal kept as an IntVector.
     */
    public static void widenInt32(MemorySegment in, int n, MemorySegment out) {
        i32ToI64(in, n, out);
    }

    static void i32ToI64(MemorySegment a, int n, MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            IntVector.fromMemorySegment(IH, a, (long) i << 2, LE)
                    .convertShape(VectorOperators.I2L, L, 0)
                    .reinterpretAsLongs()
                    .intoMemorySegment(out, (long) i << 3, LE);
        }
        if (i < n) {
            VectorMask<Integer> mi = IH.indexInRange(i, n);
            VectorMask<Long> ml = L.indexInRange(i, n);
            IntVector.fromMemorySegment(IH, a, (long) i << 2, LE, mi)
                    .convertShape(VectorOperators.I2L, L, 0)
                    .reinterpretAsLongs()
                    .intoMemorySegment(out, (long) i << 3, LE, ml);
        }
    }

    static void i32ToF64(MemorySegment a, int n, MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            IntVector.fromMemorySegment(IH, a, (long) i << 2, LE)
                    .convertShape(VectorOperators.I2D, D, 0)
                    .reinterpretAsDoubles()
                    .intoMemorySegment(out, (long) i << 3, LE);
        }
        if (i < n) {
            VectorMask<Integer> mi = IH.indexInRange(i, n);
            VectorMask<Double> md = D.indexInRange(i, n);
            IntVector.fromMemorySegment(IH, a, (long) i << 2, LE, mi)
                    .convertShape(VectorOperators.I2D, D, 0)
                    .reinterpretAsDoubles()
                    .intoMemorySegment(out, (long) i << 3, LE, md);
        }
    }

    static void i64ToF64(MemorySegment a, int n, MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            LongVector.fromMemorySegment(L, a, (long) i << 3, LE)
                    .convertShape(VectorOperators.L2D, D, 0)
                    .reinterpretAsDoubles()
                    .intoMemorySegment(out, (long) i << 3, LE);
        }
        if (i < n) {
            VectorMask<Long> ml = L.indexInRange(i, n);
            VectorMask<Double> md = D.indexInRange(i, n);
            LongVector.fromMemorySegment(L, a, (long) i << 3, LE, ml)
                    .convertShape(VectorOperators.L2D, D, 0)
                    .reinterpretAsDoubles()
                    .intoMemorySegment(out, (long) i << 3, LE, md);
        }
    }
}
