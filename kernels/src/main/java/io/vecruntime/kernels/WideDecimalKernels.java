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
import java.math.BigInteger;
import java.math.RoundingMode;

/**
 * Arithmetic over the {@link VecType#DECIMAL128} lane with Spark's decimal
 * semantics (#258, option 1 of #28: scalar two-{@code long} limb arithmetic, no
 * species -- the lane deliberately has no SIMD path). An operand is a
 * DECIMAL128 lane, an INT64 lane (a narrow decimal beside a wide one: Spark
 * does not cast the operands of an arithmetic to one type) or a scalar {@code
 * BigInteger}; every value is an unscaled integer at its declared scale.
 *
 * <p>Spark evaluates a decimal {@code + - * /} exactly, then {@code
 * toPrecision(p, s, HALF_UP)} to the result type: the value is rounded half up
 * to the result scale and, if it still has more than {@code p} digits, the row
 * overflows (null, or an error under ANSI). The kernels do the same. The common
 * shapes -- a sum whose result scale is the wider operand scale, a product
 * whose result scale is the sum of the scales, which is every {@code + - *}
 * whose precision Spark did not have to cap at 38 -- stay on the limbs: rescale
 * by a power of ten, add with carry, or multiply two values that fit a {@code
 * long} into an exact 128-bit product. A value that would leave 128 bits, or a
 * result scale below the exact one (a capped precision, a division), takes the
 * exact {@code BigInteger} / {@code BigDecimal} path for that row, so no row is
 * ever wrong -- only slower. Division is {@code BigDecimal} per row: Spark's
 * own is ({@code divide(divisor, 38, HALF_UP)} then {@code toPrecision}), and a
 * two-limb long division would have to reproduce that double rounding.
 *
 * <p>Overflow and division by zero are reported as bitmaps for the caller to
 * turn into nulls or an error for the active rows; the data of such rows is
 * unspecified.
 */
public final class WideDecimalKernels {

    private WideDecimalKernels() {}

    /** Powers of ten up to {@code 10^38} as (hi, lo) limbs. */
    private static final long[] POW10_HI = new long[39];

    private static final long[] POW10_LO = new long[39];
    private static final BigInteger[] POW10 = new BigInteger[39];

    static {
        BigInteger p = BigInteger.ONE;
        for (int i = 0; i <= 38; i++) {
            POW10[i] = p;
            POW10_HI[i] = Decimal128.hiOf(p);
            POW10_LO[i] = Decimal128.loOf(p);
            p = p.multiply(BigInteger.TEN);
        }
    }

    public static BigInteger pow10(int k) {
        return POW10[k];
    }

    /**
     * An operand read row by row as limbs: a DECIMAL128 lane, an INT64 lane, or
     * a scalar.
     */
    public static final class Operand {
        private final VectorBuffers column;
        private final boolean wideLane;
        private final long scalarHi;
        private final long scalarLo;
        private final BigInteger scalar;

        /** Digits of scale this operand is shifted by to reach the working scale. */
        final int shift;

        private Operand(VectorBuffers column, BigInteger scalar, int shift) {
            this.column = column;
            this.wideLane = column != null && column.type() == VecType.DECIMAL128;
            this.scalar = scalar;
            this.scalarHi = scalar == null ? 0L : Decimal128.hiOf(scalar);
            this.scalarLo = scalar == null ? 0L : Decimal128.loOf(scalar);
            this.shift = shift;
        }

        public static Operand of(VectorBuffers column, int shift) {
            if (column.type() != VecType.DECIMAL128 && column.type() != VecType.INT64) {
                throw new IllegalArgumentException("decimal operand lane " + column.type());
            }
            return new Operand(column, null, shift);
        }

        public static Operand of(BigInteger scalar, int shift) {
            return new Operand(null, scalar, shift);
        }

        boolean isNull(int i) {
            return column != null && column.validity() != null && !Bitmap.isSet(column.validity(), i);
        }

        long hi(int i) {
            if (column == null) {
                return scalarHi;
            }
            if (wideLane) {
                return Decimal128.hi(column.data(), i);
            }
            return column.getLong(i) >> 63;
        }

        long lo(int i) {
            if (column == null) {
                return scalarLo;
            }
            if (wideLane) {
                return Decimal128.lo(column.data(), i);
            }
            return column.getLong(i);
        }

        BigInteger big(int i) {
            if (column == null) {
                return scalar;
            }
            if (wideLane) {
                return column.getDecimal128(i);
            }
            return BigInteger.valueOf(column.getLong(i));
        }

        /** The value at the working scale, exactly. */
        BigInteger shifted(int i) {
            BigInteger v = big(i);
            return shift == 0 ? v : v.multiply(POW10[shift]);
        }
    }

    /** Whether (hi, lo) fits in a {@code long}. */
    static boolean fitsLong(long hi, long lo) {
        return hi == (lo >> 63);
    }

    /**
     * Whether |value| has at most {@code precision} digits: {@code -(10^p - 1)
     * <= v <= 10^p - 1}.
     */
    static boolean fitsPrecision(long hi, long lo, int precision) {
        // 10^p as limbs; the bound is 10^p - 1, so the test is |v| < 10^p, i.e. -10^p < v < 10^p.
        long ph = POW10_HI[precision], pl = POW10_LO[precision];
        if (Decimal128.compare(hi, lo, ph, pl) >= 0) {
            return false;
        }
        // -10^p as limbs
        long nl = -pl;
        long nh = ~ph + (pl == 0L ? 1L : 0L);
        return Decimal128.compare(hi, lo, nh, nl) > 0;
    }

    /**
     * {@code a +- b} at the result scale, per row: both operands shifted to the
     * working scale ({@code max(s1, s2)}), added on the limbs with overflow
     * detection, then Spark's {@code toPrecision}: rounded half up to {@code
     * resultScale} when it is below the working scale (a capped precision --
     * the exact path), and checked against {@code precision}.
     *
     * @param overflow set for rows whose result does not fit the result
     *     precision
     */
    public static void addSub(
            Operand a,
            Operand b,
            boolean subtract,
            int workingScale,
            int resultScale,
            int precision,
            int n,
            MemorySegment out,
            MemorySegment overflow) {
        int drop = workingScale - resultScale;
        for (int i = 0; i < n; i++) {
            if (a.isNull(i) || b.isNull(i)) {
                continue;
            }
            long ah, al, bh, bl;
            boolean exact = drop > 0;
            // Rescale on the limbs when the value fits a long after the shift.
            long xh = a.hi(i), xl = a.lo(i);
            if (a.shift > 0) {
                if (fitsLong(xh, xl) && a.shift <= 18 && Math.multiplyHigh(xl, LONG_POW10[a.shift]) == ((xl * LONG_POW10[a.shift]) >> 63)) {
                    xl = xl * LONG_POW10[a.shift];
                    xh = xl >> 63;
                } else {
                    exact = true;
                }
            }
            long yh = b.hi(i), yl = b.lo(i);
            if (!exact && b.shift > 0) {
                if (fitsLong(yh, yl) && b.shift <= 18 && Math.multiplyHigh(yl, LONG_POW10[b.shift]) == ((yl * LONG_POW10[b.shift]) >> 63)) {
                    yl = yl * LONG_POW10[b.shift];
                    yh = yl >> 63;
                } else {
                    exact = true;
                }
            }
            if (!exact) {
                ah = xh;
                al = xl;
                bh = yh;
                bl = yl;
                if (subtract) {
                    // negate b: two's complement over the limbs
                    bl = ~bl + 1L;
                    bh = ~bh + (bl == 0L ? 1L : 0L);
                    if (yh == Long.MIN_VALUE && yl == 0L) {
                        exact = true; // -2^127 has no negation
                    }
                }
                if (!exact) {
                    long rl = al + bl;
                    long carry = Long.compareUnsigned(rl, al) < 0 ? 1L : 0L;
                    long rh = ah + bh + carry;
                    // Signed overflow: operands of the same sign, result of the other.
                    boolean ovf = ((ah ^ rh) & (bh ^ rh)) < 0;
                    if (!ovf) {
                        if (fitsPrecision(rh, rl, precision)) {
                            Decimal128.set(out, i, rh, rl);
                        } else {
                            Bitmap.set(overflow, i);
                        }
                        continue;
                    }
                }
            }
            // The exact path: BigInteger at the working scale, Spark's rounding and check.
            BigInteger x = a.shifted(i);
            BigInteger y = b.shifted(i);
            BigInteger r = subtract ? x.subtract(y) : x.add(y);
            finish(r, drop, precision, i, out, overflow);
        }
    }

    /** Powers of ten that fit a long. */
    private static final long[] LONG_POW10 = new long[19];

    static {
        long p = 1L;
        for (int i = 0; i <= 18; i++) {
            LONG_POW10[i] = p;
            p *= 10L;
        }
    }

    /**
     * {@code a * b} per row: the exact product (at scale {@code s1 + s2}) then
     * {@code toPrecision} to {@code resultScale} / {@code precision}. Two
     * operands that fit a long multiply into an exact 128-bit product on the
     * limbs; anything wider goes through {@code BigInteger}.
     */
    public static void mul(
            Operand a,
            Operand b,
            int exactScale,
            int resultScale,
            int precision,
            int n,
            MemorySegment out,
            MemorySegment overflow) {
        int drop = exactScale - resultScale;
        for (int i = 0; i < n; i++) {
            if (a.isNull(i) || b.isNull(i)) {
                continue;
            }
            long ah = a.hi(i), al = a.lo(i), bh = b.hi(i), bl = b.lo(i);
            if (drop == 0 && fitsLong(ah, al) && fitsLong(bh, bl)) {
                long rl = al * bl;
                long rh = Math.multiplyHigh(al, bl);
                if (fitsPrecision(rh, rl, precision)) {
                    Decimal128.set(out, i, rh, rl);
                } else {
                    Bitmap.set(overflow, i);
                }
                continue;
            }
            finish(a.big(i).multiply(b.big(i)), drop,
                    precision, i, out, overflow);
        }
    }

    /**
     * {@code a / b} per row as Spark's {@code Decimal./}: {@code
     * BigDecimal.divide(divisor, 38, HALF_UP)} then {@code
     * toPrecision(precision, resultScale, HALF_UP)}. A zero divisor sets {@code
     * divisorZero} and writes nothing.
     */
    public static void divide(
            Operand a,
            int scaleA,
            Operand b,
            int scaleB,
            int resultScale,
            int precision,
            int n,
            MemorySegment out,
            MemorySegment overflow,
            MemorySegment divisorZero) {
        for (int i = 0; i < n; i++) {
            if (a.isNull(i) || b.isNull(i)) {
                continue;
            }
            BigInteger y = b.big(i);
            if (y.signum() == 0) {
                Bitmap.set(divisorZero, i);
                continue;
            }
            BigDecimal x = new BigDecimal(a.big(i), scaleA);
            BigDecimal q = x.divide(new BigDecimal(y, scaleB), 38, RoundingMode.HALF_UP);
            BigDecimal r = q.setScale(resultScale, RoundingMode.HALF_UP);
            BigInteger u = r.unscaledValue();
            if (u.abs().compareTo(POW10[precision]) < 0) {
                Decimal128.set(out, i, Decimal128.hiOf(u), Decimal128.loOf(u));
            } else {
                Bitmap.set(overflow, i);
            }
        }
    }

    /**
     * The exact value at the working scale, dropped {@code drop} digits half
     * up, checked against {@code precision}.
     */
    private static void finish(BigInteger r, int drop, int precision,
            int i, MemorySegment out, MemorySegment overflow) {
        BigInteger u = drop == 0 ? r : new BigDecimal(r)
                .movePointLeft(drop)
                .setScale(0, RoundingMode.HALF_UP)
                .unscaledValue();
        if (u.abs().compareTo(POW10[precision]) < 0) {
            Decimal128.set(out, i, Decimal128.hiOf(u), Decimal128.loOf(u));
        } else {
            Bitmap.set(overflow, i);
        }
    }
}
