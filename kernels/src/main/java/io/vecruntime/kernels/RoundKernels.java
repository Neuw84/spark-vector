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
import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * The rounding family -- {@code ceil} / {@code floor}, {@code rint}, {@code
 * round} (half up) and {@code bround} (half even), with and without a scale --
 * over FLOAT64, INT32 / INT64 and decimal (unscaled INT64) lanes. None of the
 * kernels touches validity; null lanes hold arbitrary values.
 *
 * <p>Every definition is Spark's, not {@code Math.round}'s:
 *
 * <ul>
 *   <li>{@code ceil} / {@code floor} of a double is {@code (long)
 *       Math.ceil(d)}, so NaN is 0 and the infinities saturate exactly as
 *       Java's cast does;
 *   <li>{@code round(d, k)} / {@code bround(d, k)} of a double is {@code
 *       BigDecimal(Double.toString(d)).setScale(k, mode).doubleValue()} -- the
 *       SHORTEST decimal representation is rounded, which is why {@code
 *       round(2.675, 2)} is 2.68 although the binary value is below the half;
 *       NaN and the infinities pass through. This is inherently a scalar path;
 *   <li>rounding an integer to a negative scale is {@code
 *       BigDecimal(v).setScale(k, mode)} narrowed back: the exact result may
 *       not fit, and the caller either raises (ANSI, the lane is flagged in a
 *       mask) or keeps the wrapped low bits like {@code BigDecimal.intValue()}
 *       / {@code longValue()};
 *   <li>rounding a decimal is an unscaled-value operation: divide by a power of
 *       ten with the rounding mode, then (for a negative scale) multiply the
 *       integer result back up.
 * </ul>
 */
public final class RoundKernels {
    /** Rounding modes Spark's rounding expressions use. */
    public enum Mode {
        /** {@code round}: half away from zero. */
        HALF_UP(RoundingMode.HALF_UP),
        /** {@code bround}: half to the even neighbour. */
        HALF_EVEN(RoundingMode.HALF_EVEN),
        /** {@code ceil}. */
        CEILING(RoundingMode.CEILING),
        /** {@code floor}. */
        FLOOR(RoundingMode.FLOOR);

        final RoundingMode java;

        Mode(RoundingMode java) {
            this.java = java;
        }
    }

    private RoundKernels() {}

    // ---------------------------------------------------------------- doubles

    /** {@code out[i] = (long) Math.ceil(a[i])} or {@code floor}; INT64 output. */
    public static void ceilFloorToLong(VectorBuffers a, boolean ceil, MemorySegment out) {
        requireType(a, VecType.FLOAT64, "ceil/floor to long");
        int n = a.length();
        MemorySegment data = a.data();
        if (ceil) {
            for (int i = 0; i < n; i++) {
                out.setAtIndex(VectorBuffers.LE_LONG, i, (long) Math.ceil(data.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
            }
        } else {
            for (int i = 0; i < n; i++) {
                out.setAtIndex(VectorBuffers.LE_LONG, i, (long) Math.floor(data.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
            }
        }
    }

    /**
     * {@code out[i] = Math.rint(a[i])}: the nearest integral double, ties to
     * even.
     */
    public static void rint(VectorBuffers a, MemorySegment out) {
        requireType(a, VecType.FLOAT64, "rint");
        int n = a.length();
        MemorySegment data = a.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, Math.rint(data.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
        }
    }

    /**
     * {@code out[i] = round(a[i], scale)} in {@code mode} as Spark's {@code
     * RoundBase} does for doubles.
     */
    public static void roundDouble(VectorBuffers a, int scale, Mode mode,
            MemorySegment out) {
        requireType(a, VecType.FLOAT64, "round");
        int n = a.length();
        MemorySegment data = a.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, roundDouble(data.getAtIndex(VectorBuffers.LE_DOUBLE, i), scale, mode));
        }
    }

    /** Spark's double rounding for one value. */
    public static double roundDouble(double d, int scale, Mode mode) {
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            return d;
        }
        return new BigDecimal(Double.toString(d)).setScale(scale, mode.java).doubleValue();
    }

    // ---------------------------------------------------------------- integers, negative scale

    /**
     * {@code out[i] = round(a[i], -power)} for INT32 / INT64 lanes (same lane
     * type out), {@code power >= 1}: the value rounded to a multiple of {@code
     * 10^power} in {@code mode}. Where the exact result does not fit the lane,
     * {@code overflowBits} is set and the lane holds the wrapped low bits, like
     * {@code BigDecimal.intValue()} / {@code longValue()}. The mask must be
     * cleared by the caller.
     */
    public static void roundIntegral(VectorBuffers a, int power, Mode mode,
            MemorySegment out, MemorySegment overflowBits) {
        int n = a.length();
        MemorySegment data = a.data();
        if (a.type() == VecType.INT32) {
            if (power >= 19) {
                for (int i = 0; i < n; i++) {
                    BigDecimal r = new BigDecimal(data.getAtIndex(VectorBuffers.LE_INT, i)).setScale(-power, mode.java);
                    if (r.compareTo(INT_MAX) > 0 || r.compareTo(INT_MIN) < 0) {
                        Bitmap.set(overflowBits, i);
                    }
                    out.setAtIndex(VectorBuffers.LE_INT, i, r.intValue());
                }
                return;
            }
            long p = DecimalKernels.POW10[power];
            for (int i = 0; i < n; i++) {
                long exact = roundDiv(data.getAtIndex(VectorBuffers.LE_INT, i), power, mode) * p; // |q| * p <= |v| + p, fits a long
                int narrowed = (int) exact;
                if (narrowed != exact) {
                    Bitmap.set(overflowBits, i);
                }
                out.setAtIndex(VectorBuffers.LE_INT, i, narrowed);
            }
            return;
        }
        requireType(a, VecType.INT64, "round");
        if (power >= 19) {
            // 10^19 is above Long.MAX_VALUE: only power 19 can still round a value up to +-10^19 (which
            // overflows); the BigDecimal path is exact and rare.
            for (int i = 0; i < n; i++) {
                BigDecimal r = new BigDecimal(data.getAtIndex(VectorBuffers.LE_LONG, i)).setScale(-power, mode.java);
                if (r.compareTo(LONG_MAX) > 0 || r.compareTo(LONG_MIN) < 0) {
                    Bitmap.set(overflowBits, i);
                }
                out.setAtIndex(VectorBuffers.LE_LONG, i, r.longValue());
            }
            return;
        }
        long p = DecimalKernels.POW10[power];
        for (int i = 0; i < n; i++) {
            long q = roundDiv(data.getAtIndex(VectorBuffers.LE_LONG, i), power, mode);
            long wrapped = q * p;
            if (Math.multiplyHigh(q, p) != (wrapped >> 63)) {
                Bitmap.set(overflowBits, i);
            }
            out.setAtIndex(VectorBuffers.LE_LONG, i, wrapped);
        }
    }

    private static final BigDecimal LONG_MAX = BigDecimal.valueOf(Long.MAX_VALUE);
    private static final BigDecimal LONG_MIN = BigDecimal.valueOf(Long.MIN_VALUE);
    private static final BigDecimal INT_MAX = BigDecimal.valueOf(Integer.MAX_VALUE);
    private static final BigDecimal INT_MIN = BigDecimal.valueOf(Integer.MIN_VALUE);

    // ---------------------------------------------------------------- decimals

    /**
     * Decimal rounding on unscaled INT64 lanes: {@code out[i] = roundDiv(a[i],
     * 10^divPower, mode) * 10^mulPower}. A decimal(p, s) rounded to scale
     * {@code k >= 0} has {@code divPower = s - min(s, k)} and {@code mulPower =
     * 0}; to a negative scale {@code k} it has {@code divPower = s - k} and
     * {@code mulPower = -k} (result scale 0). {@code ceil} / {@code floor} are
     * {@code divPower = s}. {@code divPower} may exceed 18 (the divisor is then
     * above every unscaled value, see {@link #roundDiv}). The caller has
     * already checked that the result type holds the exact result, which
     * Spark's result precision guarantees.
     */
    public static void roundDecimal(VectorBuffers a, int divPower, int mulPower,
            Mode mode, MemorySegment out) {
        requireType(a, VecType.INT64, "decimal round");
        int n = a.length();
        MemorySegment data = a.data();
        long m = DecimalKernels.POW10[mulPower];
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_LONG, i, roundDiv(data.getAtIndex(VectorBuffers.LE_LONG, i), divPower, mode) * m);
        }
    }

    /**
     * {@code v / 10^power} rounded in {@code mode}. For {@code power > 18} the
     * divisor is above twice any long's magnitude, so the half modes give 0 and
     * the directed modes give the sign's unit.
     */
    public static long roundDiv(long v, int power, Mode mode) {
        if (power == 0) {
            return v;
        }
        if (power > 18) {
            return roundDivWide(v, mode);
        }
        long p = DecimalKernels.POW10[power];
        long q = v / p;
        long r = v % p;
        if (r == 0L) {
            return q;
        }
        switch (mode) {
            case HALF_UP:
                {
                    long twiceR = Math.abs(r) << 1; // |r| < 10^18 so no overflow
                    return twiceR >= p ? q + Long.signum(v) : q;
                }
            case HALF_EVEN:
                {
                    long twiceR = Math.abs(r) << 1;
                    if (twiceR > p || (twiceR == p && (q & 1L) != 0L)) {
                        return q + Long.signum(v);
                    }
                    return q;
                }
            case CEILING:
                return r > 0L ? q + 1L : q;
            case FLOOR:
                return r < 0L ? q - 1L : q;
            default:
                throw new IllegalArgumentException(mode.toString());
        }
    }

    /**
     * {@link #roundDiv} for a divisor above twice any long's magnitude: the
     * quotient is 0 or the sign's unit.
     */
    private static long roundDivWide(long v, Mode mode) {
        switch (mode) {
            case CEILING:
                return v > 0L ? 1L : 0L;
            case FLOOR:
                return v < 0L ? -1L : 0L;
            default:
                return 0L;
        }
    }

    private static void requireType(VectorBuffers a, VecType type, String what) {
        if (a.type() != type) {
            throw new IllegalArgumentException(what + " needs " + type + " lanes, got " + a.type());
        }
    }
}
