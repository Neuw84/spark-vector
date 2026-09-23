package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.ByteOrder;

import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Kernels for decimals of at most 18 digits, held as unscaled {@code INT64}
 * lanes; the scale lives in the caller's logical type. Additions, subtractions
 * and multiplications whose result precision (in Spark's sense: {@code
 * max(p1-s1, p2-s2) + max(s1,s2) + 1}, resp. {@code p1+p2+1}) stays within 18
 * digits cannot overflow, so they are plain {@link ArithKernels} long
 * operations after the operands have been brought to a common scale with {@link
 * #mulPow10}. What this class adds is:
 *
 * <ul>
 *   <li>rescaling: multiply by a power of ten (SIMD) or divide by one with
 *       Spark's {@code ROUND_HALF_UP} (scalar; integer division has no lanes);
 *   <li>range checks against a precision bound, producing a bitmap of the
 *       offending rows for the caller to turn into nulls (legacy mode) or an
 *       error (ANSI);
 *   <li>division, which Spark defines as {@code BigDecimal.divide(divisor, 39,
 *       HALF_UP)} followed by {@code setScale(resultScale, HALF_UP)}. When the
 *       scaled dividend fits in a long, one integer division with a half-up
 *       correction gives the same result (with a divisor below 10^18 the two
 *       roundings cannot disagree); otherwise the row takes the {@link
 *       BigDecimal} path that reproduces Spark exactly;
 *   <li>conversions to and from doubles and integers with Spark's rounding.
 * </ul>
 *
 * <p>Validity is never touched here: the caller shares or combines the operand
 * bitmaps and folds the range-check bitmaps in.
 */
public final class DecimalKernels {

    static final VectorSpecies<Long> L = Species.L;
    static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    /**
     * {@code POW10[k] == 10^k} for {@code 0 <= k <= 18}; a lookup table, never
     * written after the static initialiser.
     */
    @SuppressWarnings("MutablePublicArray")
    public static final long[] POW10 = new long[19];

    static {
        POW10[0] = 1L;
        for (int i = 1; i < POW10.length; i++) {
            POW10[i] = POW10[i - 1] * 10L;
        }
    }

    /**
     * Largest magnitude a decimal of {@code precision} digits can hold: {@code
     * 10^p - 1}.
     */
    public static long maxUnscaled(int precision) {
        if (precision < 1 || precision > 18) {
            throw new IllegalArgumentException("precision out of the 64-bit range: " + precision);
        }
        return POW10[precision] - 1;
    }

    private DecimalKernels() {}

    // ------------------------------------------------------------------ rescaling

    /**
     * {@code out[i] = a[i] * 10^power} (INT64 lanes; the caller guarantees no
     * overflow).
     */
    public static void mulPow10(VectorBuffers a, int power, MemorySegment out) {
        requireInt64(a);
        if (power == 0) {
            MemorySegment.copy(a.data(), 0, out, 0,
                    (long) a.length() << 3);
            return;
        }
        ArithKernels.i64MulS(a.data(), POW10[power], a.length(), out);
    }

    /**
     * {@code out[i] = round_half_up(a[i] / 10^power)}, rounding half away from
     * zero like Spark's {@code Decimal.changePrecision} with {@code
     * ROUND_HALF_UP}.
     */
    public static void divPow10HalfUp(VectorBuffers a, int power, MemorySegment out) {
        requireInt64(a);
        int n = a.length();
        if (power == 0) {
            MemorySegment.copy(a.data(), 0, out, 0,
                    (long) n << 3);
            return;
        }
        long p = POW10[power];
        long half = p >>> 1; // p is even for power >= 1, so 2|r| >= p  <=>  |r| >= p/2
        MemorySegment data = a.data();
        for (int i = 0; i < n; i++) {
            long v = data.get(VectorBuffers.LE_LONG, (long) i << 3);
            long q = v / p;
            long r = v % p;
            if (r >= half) {
                q++;
            } else if (r <= -half) {
                q--;
            }
            out.set(VectorBuffers.LE_LONG, (long) i << 3, q);
        }
    }

    /**
     * Sets bit {@code i} of {@code outBits} where {@code |a[i]| > bound} (INT64
     * lanes). Bits of other rows are cleared. Returns the number of rows out of
     * range.
     */
    public static int outOfRange(VectorBuffers a, long bound, MemorySegment outBits) {
        requireInt64(a);
        int n = a.length();
        MemorySegment data = a.data();
        int lanes = L.length();
        int count = 0;
        long negBound = -bound;
        // Rows are processed 64 at a time so a whole bitmap word is written per iteration. Two signed
        // compares rather than abs(): abs(Long.MIN_VALUE) is negative and would slip through.
        for (int base = 0; base < n; base += 64) {
            int limit = Math.min(64, n - base);
            long word = 0L;
            int i = 0;
            for (; i + lanes <= limit; i += lanes) {
                LongVector v = LongVector.fromMemorySegment(L, data, (long) (base + i) << 3, LE);
                VectorMask<Long> m = v.compare(VectorOperators.GT, bound).or(v.compare(VectorOperators.LT, negBound));
                word |= m.toLong() << i;
            }
            for (; i < limit; i++) {
                long v = data.get(VectorBuffers.LE_LONG, (long) (base + i) << 3);
                if (v > bound || v < negBound) {
                    word |= 1L << i;
                }
            }
            Bitmap.setWord(outBits, base >>> 6, n, word);
            count += Long.bitCount(word);
        }
        return count;
    }

    // ------------------------------------------------------------------ division

    /**
     * Decimal division {@code a(s1) / b(s2)} into a result of {@code
     * resultScale}. Rows whose divisor is zero produce 0 and must be masked by
     * the caller (see {@link CompareKernels}); rows whose quotient does not fit
     * {@code resultPrecision} digits are flagged in {@code overflowBits} (all
     * other bits cleared). Returns the number of overflowing rows.
     */
    public static int divide(
            VectorBuffers a,
            VectorBuffers b,
            int s1,
            int s2,
            int resultScale,
            int resultPrecision,
            MemorySegment out,
            MemorySegment overflowBits) {
        requireInt64(a);
        requireInt64(b);
        int n = a.length();
        Div div = new Div(s1, s2, resultScale, resultPrecision, n, out,
                overflowBits);
        MemorySegment ad = a.data();
        MemorySegment bd = b.data();
        for (int i = 0; i < n; i++) {
            div.row(i, ad.get(VectorBuffers.LE_LONG, (long) i << 3), bd.get(VectorBuffers.LE_LONG, (long) i << 3));
        }
        return div.overflows;
    }

    /** {@link #divide} with a constant divisor. */
    public static int divideScalar(
            VectorBuffers a,
            long divisor,
            int s1,
            int s2,
            int resultScale,
            int resultPrecision,
            MemorySegment out,
            MemorySegment overflowBits) {
        requireInt64(a);
        int n = a.length();
        Div div = new Div(s1, s2, resultScale, resultPrecision, n, out,
                overflowBits);
        MemorySegment ad = a.data();
        for (int i = 0; i < n; i++) {
            div.row(i, ad.get(VectorBuffers.LE_LONG, (long) i << 3), divisor);
        }
        return div.overflows;
    }

    /** {@link #divide} with a constant dividend. */
    public static int scalarDivide(
            long dividend,
            VectorBuffers b,
            int s1,
            int s2,
            int resultScale,
            int resultPrecision,
            MemorySegment out,
            MemorySegment overflowBits) {
        requireInt64(b);
        int n = b.length();
        Div div = new Div(s1, s2, resultScale, resultPrecision, n, out,
                overflowBits);
        MemorySegment bd = b.data();
        for (int i = 0; i < n; i++) {
            div.row(i, dividend, bd.get(VectorBuffers.LE_LONG, (long) i << 3));
        }
        return div.overflows;
    }

    /** Per-row division state shared by the three entry points. */
    private static final class Div {
        private final int s1;
        private final int s2;
        private final int resultScale;
        private final int k; // 10^k scales the dividend so that num / den is the result's unscaled value
        private final long bound;
        private final MemorySegment out;
        private final MemorySegment overflowBits;
        int overflows;

        Div(
                int s1,
                int s2,
                int resultScale,
                int resultPrecision,
                int n,
                MemorySegment out,
                MemorySegment overflowBits) {
            this.s1 = s1;
            this.s2 = s2;
            this.resultScale = resultScale;
            this.k = s2 + resultScale - s1;
            this.bound = maxUnscaled(resultPrecision);
            this.out = out;
            this.overflowBits = overflowBits;
            Bitmap.fill(overflowBits, n, false);
        }

        void row(int i, long a, long b) {
            long result;
            if (b == 0L) {
                result = 0L;
            } else if (k >= 0 && k <= 18 && fitsWhenScaled(a, k)) {
                result = divideHalfUp(a * POW10[k], b);
            } else {
                result = divideBig(a, b);
            }
            if (Math.abs(result) > bound) {
                Bitmap.set(overflowBits, i);
                overflows++;
                result = 0L;
            }
            out.set(VectorBuffers.LE_LONG, (long) i << 3, result);
        }

        /**
         * Spark's two roundings, exactly: divide at scale 39 half-up, then to
         * the result scale.
         */
        private long divideBig(long a, long b) {
            BigDecimal q = new BigDecimal(BigInteger.valueOf(a), s1).divide(new BigDecimal(BigInteger.valueOf(b), s2), 39, RoundingMode.HALF_UP).setScale(resultScale, RoundingMode.HALF_UP);
            BigInteger unscaled = q.unscaledValue();
            if (unscaled.bitLength() > 63) {
                return Long.MAX_VALUE; // flagged as out of range by the bound check
            }
            return unscaled.longValue();
        }
    }

    /** True when {@code a * 10^k} fits in a long. */
    static boolean fitsWhenScaled(long a, int k) {
        long p = POW10[k];
        long hi = Math.multiplyHigh(a, p);
        long lo = a * p;
        return hi == (lo >> 63);
    }

    /** Integer division rounding half away from zero. */
    static long divideHalfUp(long num, long den) {
        long q = num / den;
        long r = num % den;
        if (r != 0L) {
            long twiceR = Math.abs(r) << 1; // |r| < |den| <= Long.MAX so no overflow
            if (twiceR >= Math.abs(den) && twiceR >= 0) {
                q += ((num ^ den) >= 0) ? 1 : -1;
            }
        }
        return q;
    }

    // ------------------------------------------------------------------ conversions

    /**
     * Decimal to double, exactly as {@code BigDecimal.doubleValue()}: for
     * unscaled values below 2^53 the correctly rounded IEEE division {@code v /
     * 10^scale} is the same number; larger values take the {@link BigDecimal}
     * path.
     */
    public static void toDouble(VectorBuffers a, int scale, MemorySegment out) {
        requireInt64(a);
        int n = a.length();
        MemorySegment data = a.data();
        double p = (double) POW10[scale];
        for (int i = 0; i < n; i++) {
            long v = data.get(VectorBuffers.LE_LONG, (long) i << 3);
            double d;
            if (Math.abs(v) < (1L << 53)) {
                d = scale == 0 ? (double) v : (double) v / p;
            } else {
                d = new BigDecimal(BigInteger.valueOf(v), scale).doubleValue();
            }
            out.set(VectorBuffers.LE_DOUBLE, (long) i << 3, d);
        }
    }

    /**
     * Double to decimal with Spark's semantics ({@code Decimal(double)} goes
     * through {@code Double.toString}, then {@code changePrecision} rounds half
     * up). NaN, infinities and values out of range are flagged in {@code
     * invalidBits} (other bits cleared) and produce 0. Returns the number of
     * flagged rows.
     */
    public static int fromDouble(VectorBuffers a, int precision, int scale,
            MemorySegment out, MemorySegment invalidBits) {
        if (a.type() != VecType.FLOAT64) {
            throw new IllegalArgumentException("expected FLOAT64, got " + a.type());
        }
        int n = a.length();
        long bound = maxUnscaled(precision);
        Bitmap.fill(invalidBits, n, false);
        int count = 0;
        MemorySegment data = a.data();
        for (int i = 0; i < n; i++) {
            double d = data.get(VectorBuffers.LE_DOUBLE, (long) i << 3);
            long result = 0L;
            boolean bad = Double.isNaN(d) || Double.isInfinite(d);
            if (!bad) {
                BigInteger unscaled = new BigDecimal(Double.toString(d)).setScale(scale, RoundingMode.HALF_UP).unscaledValue();
                if (unscaled.bitLength() > 63 || Math.abs(unscaled.longValue()) > bound) {
                    bad = true;
                } else {
                    result = unscaled.longValue();
                }
            }
            if (bad) {
                Bitmap.set(invalidBits, i);
                count++;
            }
            out.set(VectorBuffers.LE_LONG, (long) i << 3, result);
        }
        return count;
    }

    /**
     * Integer (INT32 or INT64) to decimal: widens and multiplies by {@code
     * 10^scale}. Rows whose value would exceed {@code precision} digits are
     * flagged in {@code invalidBits} (other bits cleared) and produce 0; the
     * check happens before the multiplication so it cannot wrap. Returns the
     * number of flagged rows.
     */
    public static int fromIntegral(VectorBuffers a, int precision, int scale,
            MemorySegment out, MemorySegment invalidBits) {
        int n = a.length();
        if (a.type() == VecType.INT32) {
            CastKernels.i32ToI64(a.data(), n, out);
        } else if (a.type() == VecType.INT64) {
            MemorySegment.copy(a.data(), 0, out, 0,
                    (long) n << 3);
        } else {
            throw new IllegalArgumentException("expected INT32 or INT64, got " + a.type());
        }
        long bound = maxUnscaled(precision) / POW10[scale];
        SegmentVectorBuffers widened = SegmentVectorBuffers.fixedWidth(VecType.INT64, n, null, out);
        int count = outOfRange(widened, bound, invalidBits);
        if (count > 0) {
            // Zero the flagged lanes so the multiplication below cannot wrap.
            for (int w = 0, words = Bitmap.wordsFor(n);
                 w < words;
                 w++) {
                long bits = Bitmap.wordAt(invalidBits, w, n);
                while (bits != 0L) {
                    int i = (w << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    out.set(VectorBuffers.LE_LONG, (long) i << 3, 0L);
                }
            }
        }
        if (scale > 0) {
            ArithKernels.i64MulS(out, POW10[scale], n, out);
        }
        return count;
    }

    /**
     * Decimal to a long, truncating toward zero ({@code BigDecimal.longValue()}
     * and Spark's ANSI {@code roundToLong} agree for 18-digit values, which
     * always fit).
     */
    public static void toLong(VectorBuffers a, int scale, MemorySegment out) {
        requireInt64(a);
        int n = a.length();
        if (scale == 0) {
            MemorySegment.copy(a.data(), 0, out, 0,
                    (long) n << 3);
            return;
        }
        long p = POW10[scale];
        MemorySegment data = a.data();
        for (int i = 0; i < n; i++) {
            out.set(VectorBuffers.LE_LONG, (long) i << 3, data.get(VectorBuffers.LE_LONG, (long) i << 3) / p);
        }
    }

    /**
     * Decimal to an int, truncating toward zero. Values outside the int range
     * are flagged in {@code invalidBits} (other bits cleared) and written
     * wrapped, which is what Spark's legacy cast does; ANSI callers raise
     * instead. Returns the number of flagged rows.
     */
    public static int toInt(VectorBuffers a, int scale, MemorySegment out,
                            MemorySegment invalidBits) {
        requireInt64(a);
        int n = a.length();
        long p = POW10[scale];
        Bitmap.fill(invalidBits, n, false);
        int count = 0;
        MemorySegment data = a.data();
        for (int i = 0; i < n; i++) {
            long v = data.get(VectorBuffers.LE_LONG, (long) i << 3) / p;
            if (v != (int) v) {
                Bitmap.set(invalidBits, i);
                count++;
            }
            out.set(VectorBuffers.LE_INT, (long) i << 2, (int) v);
        }
        return count;
    }

    private static void requireInt64(VectorBuffers a) {
        if (a.type() != VecType.INT64) {
            throw new IllegalArgumentException("decimal lanes must be INT64, got " + a.type());
        }
    }
}
