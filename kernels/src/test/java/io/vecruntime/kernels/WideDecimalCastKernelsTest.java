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
import java.nio.charset.StandardCharsets;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The wide casts against BigDecimal, which is what Spark's Decimal does
 * underneath.
 */
class WideDecimalCastKernelsTest {

    private static BigInteger pick(Random rnd, int precision) {
        BigInteger bound = BigInteger.TEN.pow(precision).subtract(BigInteger.ONE);
        switch (rnd.nextInt(10)) {
            case 0:
                return bound;
            case 1:
                return bound.negate();
            case 2:
                return BigInteger.valueOf(Long.MAX_VALUE).min(bound);
            case 3:
                return BigInteger.valueOf(Long.MIN_VALUE).max(bound.negate());
            case 4:
                return BigInteger.ZERO;
            case 5:
                return BigInteger.valueOf(rnd.nextInt(20000) - 10000);
            default:
                {
                    BigInteger v = new BigInteger(rnd.nextInt(precision * 3 + 8), rnd);
                    return (rnd.nextBoolean() ? v.negate() : v).min(bound).max(bound.negate());
                }
        }
    }

    private static boolean[] nulls(Random rnd, int n) {
        boolean[] b = new boolean[n];
        for (int i = 0; i < n; i++) {
            b[i] = rnd.nextInt(6) == 0;
        }
        return b;
    }

    private static BigInteger expectedRescale(BigInteger u, int from, int to,
            int precision) {
        BigInteger r = new BigDecimal(u, from).setScale(to, java.math.RoundingMode.HALF_UP).unscaledValue();
        return r.abs().compareTo(BigInteger.TEN.pow(precision)) < 0 ? r : null;
    }

    @Test
    void decimalToDecimalInEveryDirectionMatchesBigDecimal() {
        Random rnd = new Random(31);
        int[][] types = {
            {38, 10},
            {27, 2},
            {20, 0},
            {38, 0},
            {18, 4},
            {10, 2},
            {38, 30},
            {19, 19}
        };
        int checked = 0, invalidRows = 0;
        for (int[] f : types) {
            for (int[] t : types) {
                if (f[0] <= 18 && t[0] <= 18) {
                    continue; // the INT64 lane's cast
                }
                for (int n : new int[] {1, 64, 65, 257}) {
                    try (Arena arena = Arena.ofConfined()) {
                        boolean[] nl = nulls(rnd, n);
                        VectorBuffers a;
                        if (f[0] > 18) {
                            BigInteger[] v = new BigInteger[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = pick(rnd, f[0]);
                            }
                            a = ArrowLayout.ofDecimal128(arena, v, nl);
                        } else {
                            long[] v = new long[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = pick(rnd, f[0]).longValue();
                            }
                            a = ArrowLayout.ofLongs(arena, v, nl);
                        }
                        MemorySegment invalid = ArrowLayout.allocateBitmap(arena, n);
                        if (t[0] > 18) {
                            MemorySegment out = ArrowLayout.allocateData(arena, VecType.DECIMAL128, n);
                            WideDecimalCastKernels.toWide(a, f[1], t[1], t[0], n, out,
                                    invalid);
                            for (int i = 0; i < n; i++) {
                                if (nl[i]) {
                                    continue;
                                }
                                BigInteger u = a.type() == VecType.DECIMAL128 ? a.getDecimal128(i) : BigInteger.valueOf(a.getLong(i));
                                BigInteger e = expectedRescale(u, f[1], t[1], t[0]);
                                if (e == null) {
                                    assertTrue(Bitmap.isSet(invalid, i), "invalid at " + i);
                                    invalidRows++;
                                } else {
                                    assertFalse(Bitmap.isSet(invalid, i), "valid at " + i + ": " + u);
                                    assertEquals(
                                            e,
                                            Decimal128.toBigInteger(Decimal128.hi(out, i), Decimal128.lo(out, i)),
                                            u
                                                    + " ("
                                                    + f[0]
                                                    + ","
                                                    + f[1]
                                                    + ") -> ("
                                                    + t[0]
                                                    + ","
                                                    + t[1]
                                                    + ")");
                                }
                                checked++;
                            }
                        } else {
                            MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT64, n);
                            WideDecimalCastKernels.toNarrow(a, f[1], t[1], t[0], n, out,
                                    invalid);
                            for (int i = 0; i < n; i++) {
                                if (nl[i]) {
                                    continue;
                                }
                                BigInteger e = expectedRescale(a.getDecimal128(i), f[1], t[1], t[0]);
                                if (e == null) {
                                    assertTrue(Bitmap.isSet(invalid, i));
                                    invalidRows++;
                                } else {
                                    assertFalse(Bitmap.isSet(invalid, i));
                                    assertEquals(e.longValueExact(), out.getAtIndex(VectorBuffers.LE_LONG, i));
                                }
                                checked++;
                            }
                        }
                    }
                }
            }
        }
        assertTrue(checked > 5000 && invalidRows > 100, checked + " checked, " + invalidRows + " invalid");
    }

    @Test
    void integralsDoublesStringsLongsIntsNegateAbs() {
        Random rnd = new Random(5);
        int n = 300;
        try (Arena arena = Arena.ofConfined()) {
            boolean[] nl = nulls(rnd, n);
            long[] lv = new long[n];
            for (int i = 0; i < n; i++) {
                lv[i] = rnd.nextInt(4) == 0
                        ? (rnd.nextBoolean() ? Long.MAX_VALUE : Long.MIN_VALUE)
                        : rnd.nextLong() >> rnd.nextInt(60);
            }
            VectorBuffers longs = ArrowLayout.ofLongs(arena, lv, nl);
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.DECIMAL128, n);
            MemorySegment invalid = ArrowLayout.allocateBitmap(arena, n);
            // long -> decimal(25,3) and decimal(38,20)
            for (int[] t : new int[][] {{25, 3}, {38, 20}, {20, 0}}) {
                Bitmap.fill(invalid, n, false);
                WideDecimalCastKernels.fromIntegral(longs, t[1], t[0], n, out, invalid);
                for (int i = 0; i < n; i++) {
                    if (nl[i]) {
                        continue;
                    }
                    BigInteger e = BigInteger.valueOf(lv[i]).multiply(BigInteger.TEN.pow(t[1]));
                    boolean fits = e.abs().compareTo(BigInteger.TEN.pow(t[0])) < 0;
                    assertEquals(!fits, Bitmap.isSet(invalid, i), "row " + i + " " + lv[i]);
                    if (fits) {
                        assertEquals(e,
                                Decimal128.toBigInteger(Decimal128.hi(out, i), Decimal128.lo(out, i)));
                    }
                }
            }
            // doubles -> decimal(30,8)
            double[] dv = new double[n];
            for (int i = 0; i < n; i++) {
                dv[i] = switch (rnd.nextInt(6)) {
                            case 0 -> Double.NaN;
                            case 1 -> Double.POSITIVE_INFINITY;
                            case 2 -> 1e25;
                            default -> (rnd.nextDouble() - 0.5) * 1e12;
                        };
            }
            VectorBuffers doubles = ArrowLayout.ofDoubles(arena, dv, nl);
            Bitmap.fill(invalid, n, false);
            WideDecimalCastKernels.fromDouble(doubles, 8, 30, n, out, invalid);
            for (int i = 0; i < n; i++) {
                if (nl[i]) {
                    continue;
                }
                if (Double.isNaN(dv[i]) || Double.isInfinite(dv[i])) {
                    assertTrue(Bitmap.isSet(invalid, i));
                    continue;
                }
                BigInteger e = BigDecimal.valueOf(dv[i])
                        .setScale(8, java.math.RoundingMode.HALF_UP)
                        .unscaledValue();
                boolean fits = e.abs().compareTo(BigInteger.TEN.pow(30)) < 0;
                assertEquals(!fits, Bitmap.isSet(invalid, i), "row " + i + " " + dv[i]);
                if (fits) {
                    assertEquals(e,
                            Decimal128.toBigInteger(Decimal128.hi(out, i), Decimal128.lo(out, i)));
                }
            }
            // wide -> double, long, int, string, -x, |x|
            BigInteger[] wv = new BigInteger[n];
            for (int i = 0; i < n; i++) {
                wv[i] = pick(rnd, 38);
            }
            VectorBuffers wide = ArrowLayout.ofDecimal128(arena, wv, nl);
            int scale = 10;
            MemorySegment dbl = ArrowLayout.allocateData(arena, VecType.FLOAT64, n);
            WideDecimalCastKernels.toDouble(wide, scale, n, dbl);
            MemorySegment lng = ArrowLayout.allocateData(arena, VecType.INT64, n);
            MemorySegment lngInvalid = ArrowLayout.allocateBitmap(arena, n);
            WideDecimalCastKernels.toLong(wide, scale, n, lng, lngInvalid);
            MemorySegment ints = ArrowLayout.allocateData(arena, VecType.INT32, n);
            MemorySegment intInvalid = ArrowLayout.allocateBitmap(arena, n);
            WideDecimalCastKernels.toInt(wide, scale, n, ints, intInvalid);
            VectorBuffers strings = WideDecimalCastKernels.toUtf8(arena, wide, scale, n, false);
            VectorBuffers plain = WideDecimalCastKernels.toUtf8(arena, wide, scale, n, true);
            MemorySegment neg = ArrowLayout.allocateData(arena, VecType.DECIMAL128, n);
            WideDecimalCastKernels.negate(wide, n, neg);
            MemorySegment abs = ArrowLayout.allocateData(arena, VecType.DECIMAL128, n);
            WideDecimalCastKernels.abs(wide, n, abs);
            for (int i = 0; i < n; i++) {
                if (nl[i]) {
                    continue;
                }
                BigDecimal d = new BigDecimal(wv[i], scale);
                assertEquals(d.doubleValue(), dbl.getAtIndex(VectorBuffers.LE_DOUBLE, i), "double at " + i);
                BigInteger trunc = d.toBigInteger();
                assertEquals(trunc.bitLength() >= 64, Bitmap.isSet(lngInvalid, i), "long range at " + i);
                assertEquals(trunc.longValue(), lng.getAtIndex(VectorBuffers.LE_LONG, i), "long at " + i);
                assertEquals(trunc.bitLength() >= 32, Bitmap.isSet(intInvalid, i), "int range at " + i);
                assertEquals(trunc.intValue(), ints.getAtIndex(VectorBuffers.LE_INT, i), "int at " + i);
                int s = strings.offsets().getAtIndex(VectorBuffers.LE_INT, i), e = strings.offsets().getAtIndex(VectorBuffers.LE_INT, i + 1);
                assertEquals(
                        d.toString(),
                        new String(strings.data()
                                          .asSlice(s, e - s)
                                          .toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),
                                StandardCharsets.US_ASCII),
                        "string at " + i);
                int ps = plain.offsets().getAtIndex(VectorBuffers.LE_INT, i), pe = plain.offsets().getAtIndex(VectorBuffers.LE_INT, i + 1);
                assertEquals(
                        d.toPlainString(),
                        new String(plain.data()
                                        .asSlice(ps, pe - ps)
                                        .toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),
                                StandardCharsets.US_ASCII),
                        "plain string at " + i);
                assertEquals(
                        wv[i].negate(),
                        Decimal128.toBigInteger(Decimal128.hi(neg, i), Decimal128.lo(neg, i)),
                        "negate at " + i);
                assertEquals(
                        wv[i].abs(),
                        Decimal128.toBigInteger(Decimal128.hi(abs, i), Decimal128.lo(abs, i)),
                        "abs at " + i);
            }
        }
    }
}
