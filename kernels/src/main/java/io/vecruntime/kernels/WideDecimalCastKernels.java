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
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;

/**
 * Casts to and from the {@link VecType#DECIMAL128} lane with Spark's {@code
 * Cast} semantics (#258). Every value is an unscaled integer at its declared
 * scale. Rows that do not fit the target are reported in an {@code invalid}
 * bitmap (null in legacy mode, an error in ANSI mode -- the caller decides) and
 * their data is unspecified. Like {@link WideDecimalKernels}, the loops are
 * scalar over the limbs with the exact {@code BigInteger} / {@code BigDecimal}
 * path wherever a value leaves 64 bits or the operation rounds.
 */
public final class WideDecimalCastKernels {

    private WideDecimalCastKernels() {}

    private static final long[] LONG_POW10 = new long[19];

    static {
        long p = 1L;
        for (int i = 0; i <= 18; i++) {
            LONG_POW10[i] = p;
            p *= 10L;
        }
    }

    private static BigInteger big(VectorBuffers a, int i) {
        return a.type() == VecType.DECIMAL128 ? a.getDecimal128(i) : BigInteger.valueOf(a.type() == VecType.INT32 ? a.getInt(i) : a.getLong(i));
    }

    private static boolean isNull(VectorBuffers a, int i) {
        return a.validity() != null && !Bitmap.isSet(a.validity(), i);
    }

    /**
     * Spark's {@code Decimal.changePrecision}: rescale (half up when digits
     * drop), then the precision check.
     */
    private static BigInteger rescaled(BigInteger unscaled, int fromScale, int toScale) {
        if (toScale == fromScale) {
            return unscaled;
        }
        if (toScale > fromScale) {
            return unscaled.multiply(WideDecimalKernels.pow10(toScale - fromScale));
        }
        return new BigDecimal(unscaled, fromScale).setScale(toScale, RoundingMode.HALF_UP).unscaledValue();
    }

    private static boolean fits(BigInteger u, int precision) {
        return u.abs().compareTo(WideDecimalKernels.pow10(precision)) < 0;
    }

    /**
     * A decimal (DECIMAL128 or INT64 lane) to a DECIMAL128 of another type. On
     * the limbs when the value fits a long and the scale grows within a long;
     * the exact path otherwise.
     */
    public static void toWide(
            VectorBuffers a,
            int fromScale,
            int toScale,
            int toPrecision,
            int n,
            MemorySegment out,
            MemorySegment invalid) {
        int k = toScale - fromScale;
        boolean wide = a.type() == VecType.DECIMAL128;
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            if (k >= 0 && k <= 18) {
                long lo = wide ? Decimal128.lo(a.data(), i) : a.getLong(i);
                long hi = wide ? Decimal128.hi(a.data(), i) : (lo >> 63);
                if (WideDecimalKernels.fitsLong(hi, lo)) {
                    long p = LONG_POW10[k];
                    long r = lo * p;
                    if (Math.multiplyHigh(lo, p) == (r >> 63)) {
                        if (WideDecimalKernels.fitsPrecision(r >> 63, r, toPrecision)) {
                            Decimal128.set(out, i, r >> 63, r);
                        } else {
                            Bitmap.set(invalid, i);
                        }
                        continue;
                    }
                }
            }
            BigInteger u = rescaled(big(a, i), fromScale, toScale);
            if (fits(u, toPrecision)) {
                Decimal128.set(out, i, Decimal128.hiOf(u), Decimal128.loOf(u));
            } else {
                Bitmap.set(invalid, i);
            }
        }
    }

    /**
     * A DECIMAL128 to a decimal of at most 18 digits (the INT64 lane): rescale,
     * then the precision check guarantees it fits a long.
     */
    public static void toNarrow(
            VectorBuffers a,
            int fromScale,
            int toScale,
            int toPrecision,
            int n,
            MemorySegment out,
            MemorySegment invalid) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            BigInteger u = rescaled(a.getDecimal128(i), fromScale, toScale);
            if (fits(u, toPrecision)) {
                out.setAtIndex(VectorBuffers.LE_LONG, i, u.longValue());
            } else {
                Bitmap.set(invalid, i);
            }
        }
    }

    /**
     * An int or long column to a DECIMAL128: {@code value * 10^scale}, checked
     * against the precision.
     */
    public static void fromIntegral(VectorBuffers a, int toScale, int toPrecision,
            int n, MemorySegment out, MemorySegment invalid) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            long v = a.type() == VecType.INT32 ? a.getInt(i) : a.getLong(i);
            if (toScale <= 18) {
                long p = LONG_POW10[toScale];
                long r = v * p;
                if (Math.multiplyHigh(v, p) == (r >> 63)) {
                    if (WideDecimalKernels.fitsPrecision(r >> 63, r, toPrecision)) {
                        Decimal128.set(out, i, r >> 63, r);
                    } else {
                        Bitmap.set(invalid, i);
                    }
                    continue;
                }
            }
            BigInteger u = BigInteger.valueOf(v).multiply(WideDecimalKernels.pow10(toScale));
            if (fits(u, toPrecision)) {
                Decimal128.set(out, i, Decimal128.hiOf(u), Decimal128.loOf(u));
            } else {
                Bitmap.set(invalid, i);
            }
        }
    }

    /**
     * A double to a DECIMAL128 as Spark's {@code Decimal(double)} then {@code
     * changePrecision}: the double's exact decimal expansion rounded half up to
     * the scale. NaN and infinities are marked invalid here; the caller makes
     * them null in every mode, as Spark does.
     */
    public static void fromDouble(VectorBuffers a, int toScale, int toPrecision,
            int n, MemorySegment out, MemorySegment invalid) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            double d = a.getDouble(i);
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                Bitmap.set(invalid, i);
                continue;
            }
            // Spark builds the Decimal from the double's shortest string (BigDecimal.valueOf), not its binary expansion.
            BigInteger u = BigDecimal.valueOf(d)
                    .setScale(toScale, RoundingMode.HALF_UP)
                    .unscaledValue();
            if (fits(u, toPrecision)) {
                Decimal128.set(out, i, Decimal128.hiOf(u), Decimal128.loOf(u));
            } else {
                Bitmap.set(invalid, i);
            }
        }
    }

    /**
     * A DECIMAL128 to double: {@code BigDecimal.doubleValue()}, which is what
     * Spark's {@code Decimal.toDouble} does.
     */
    public static void toDouble(VectorBuffers a, int scale, int n,
            MemorySegment out) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, new BigDecimal(a.getDecimal128(i), scale).doubleValue());
        }
    }

    /**
     * A DECIMAL128 to long, truncating toward zero as Spark's {@code
     * Decimal.toLong} does. A value outside the long range is invalid (Spark
     * raises in ANSI mode and wraps -- the low 64 bits -- in legacy mode; the
     * wrapped value is written so the legacy caller can keep it).
     */
    public static void toLong(VectorBuffers a, int scale, int n,
            MemorySegment out, MemorySegment invalid) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            BigInteger t = scale == 0 ? a.getDecimal128(i) : a.getDecimal128(i).divide(WideDecimalKernels.pow10(scale));
            if (t.bitLength() >= 64) {
                Bitmap.set(invalid, i);
            }
            out.setAtIndex(VectorBuffers.LE_LONG, i, t.longValue());
        }
    }

    /**
     * A DECIMAL128 to int: the long above, then the int range; the wrapped
     * value is written.
     */
    public static void toInt(VectorBuffers a, int scale, int n,
            MemorySegment out, MemorySegment invalid) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            BigInteger t = scale == 0 ? a.getDecimal128(i) : a.getDecimal128(i).divide(WideDecimalKernels.pow10(scale));
            if (t.bitLength() >= 32) {
                Bitmap.set(invalid, i);
            }
            out.setAtIndex(VectorBuffers.LE_INT, i, t.intValue());
        }
    }

    /**
     * A DECIMAL128 to its string as Spark's {@code Cast} prints it: {@code
     * BigDecimal.toPlainString} under ANSI ({@code plain}), {@code
     * BigDecimal.toString} otherwise (scientific when the adjusted exponent is
     * below -6: a zero at scale 10 is {@code 0E-10}).
     */
    public static SegmentVectorBuffers toUtf8(Arena arena, VectorBuffers a, int scale,
            int n, boolean plain) {
        byte[][] strings = new byte[n][];
        int total = 0;
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                strings[i] = new byte[0];
            } else {
                BigDecimal d = new BigDecimal(a.getDecimal128(i), scale);
                strings[i] = (plain ? d.toPlainString() : d.toString()).getBytes(StandardCharsets.US_ASCII);
                total += strings[i].length;
            }
        }
        byte[] bytes = new byte[total];
        int[] offsets = new int[n + 1];
        int pos = 0;
        for (int i = 0; i < n; i++) {
            offsets[i] = pos;
            System.arraycopy(strings[i], 0, bytes, pos, strings[i].length);
            pos += strings[i].length;
        }
        offsets[n] = pos;
        MemorySegment offsetSegment = ArrowLayout.allocateOffsets(arena, n);
        MemorySegment.copy(offsets, 0, offsetSegment, VectorBuffers.LE_INT, 0,
                n + 1);
        MemorySegment data = ArrowLayout.allocateBytes(arena, Math.max(1, total));
        MemorySegment.copy(bytes, 0, data, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, total);
        return SegmentVectorBuffers.utf8(n, a.validity(), offsetSegment, data);
    }

    /**
     * Two's-complement negation on the limbs. A decimal's range is symmetric,
     * so a valid value never overflows.
     */
    public static void negate(VectorBuffers a, int n, MemorySegment out) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            long lo = Decimal128.lo(a.data(), i), hi = Decimal128.hi(a.data(), i);
            long nl = ~lo + 1L;
            long nh = ~hi + (lo == 0L ? 1L : 0L);
            Decimal128.set(out, i, nh, nl);
        }
    }

    /**
     * |value| on the limbs: the value when the high limb is non-negative, its
     * negation otherwise.
     */
    public static void abs(VectorBuffers a, int n, MemorySegment out) {
        for (int i = 0; i < n; i++) {
            if (isNull(a, i)) {
                continue;
            }
            long lo = Decimal128.lo(a.data(), i), hi = Decimal128.hi(a.data(), i);
            if (hi >= 0) {
                Decimal128.set(out, i, hi, lo);
            } else {
                long nl = ~lo + 1L;
                long nh = ~hi + (lo == 0L ? 1L : 0L);
                Decimal128.set(out, i, nh, nl);
            }
        }
    }
}
