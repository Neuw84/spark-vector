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
import java.util.Random;

import io.sparkvector.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The two-limb arithmetic against Spark's semantics as BigDecimal, row by row,
 * over the awkward values.
 */
class WideDecimalKernelsTest {

    /**
     * A value: the extremes, both limb boundaries, small values, and random
     * 128-bit ones; nulls in a fifth of the rows.
     */
    private static BigInteger pick(Random rnd, int precision) {
        BigInteger bound = BigInteger.TEN.pow(precision).subtract(BigInteger.ONE);
        switch (rnd.nextInt(12)) {
            case 0:
                return bound;
            case 1:
                return bound.negate();
            case 2:
                return BigInteger.valueOf(Long.MAX_VALUE);
            case 3:
                return BigInteger.valueOf(Long.MIN_VALUE);
            case 4:
                return BigInteger.valueOf(Long.MIN_VALUE).subtract(BigInteger.ONE);
            case 5:
                return BigInteger.ZERO;
            case 6:
                return BigInteger.valueOf(rnd.nextInt(2000) - 1000);
            case 7:
                return BigInteger.ONE
                        .shiftLeft(127)
                        .negate()
                        .max(bound.negate());
            default:
                {
                    BigInteger v = new BigInteger(rnd.nextInt(precision * 3 + 10), rnd);
                    if (rnd.nextBoolean()) {
                        v = v.negate();
                    }
                    return v.min(bound).max(bound.negate());
                }
        }
    }

    private static SegmentVectorBuffers column(Arena arena, Random rnd, int n,
            int precision, boolean wide, boolean[] nulls) {
        if (wide) {
            BigInteger[] v = new BigInteger[n];
            for (int i = 0; i < n; i++) {
                v[i] = pick(rnd, precision);
            }
            return ArrowLayout.ofDecimal128(arena, v, nulls);
        } else {
            long[] v = new long[n];
            for (int i = 0; i < n; i++) {
                v[i] = pick(rnd, Math.min(precision, 18)).max(BigInteger.valueOf(Long.MIN_VALUE))
                                .min(BigInteger.valueOf(Long.MAX_VALUE))
                                .longValue();
            }
            return ArrowLayout.ofLongs(arena, v, nulls);
        }
    }

    private static boolean[] nulls(Random rnd, int n) {
        boolean[] b = new boolean[n];
        for (int i = 0; i < n; i++) {
            b[i] = rnd.nextInt(5) == 0;
        }
        return b;
    }

    /**
     * Spark's result type for the op over (p1, s1) and (p2, s2), with precision
     * loss allowed (the default).
     */
    private static int[] resultType(char op, int p1, int s1,
            int p2, int s2) {
        int p, s;
        switch (op) {
            case '+':
            case '-':
                s = Math.max(s1, s2);
                p = Math.max(p1 - s1, p2 - s2) + s + 1;
                break;
            case '*':
                s = s1 + s2;
                p = p1 + p2 + 1;
                break;
            default:
                { // '/'
                    int intDig = p1 - s1 + s2;
                    s = Math.max(6, s1 + p2 + 1);
                    p = intDig + s;
                }
        }
        if (p > 38) {
            int intDigits = p - s;
            int minScale = Math.min(s, 6);
            int adjusted = Math.max(38 - intDigits, minScale);
            s = Math.min(s, adjusted);
            p = 38;
        }
        return new int[] {p, s};
    }

    @Test
    void everyOperatorMatchesSparkSemanticsOverWideAndNarrowOperands() {
        Random rnd = new Random(258);
        int[][] types = {
            {38, 10},
            {27, 2},
            {20, 0},
            {38, 0},
            {18, 4},
            {10, 2},
            {38, 37}
        };
        char[] ops = {'+', '-', '*', '/'};
        int checked = 0, overflowed = 0, zeros = 0;
        for (char op : ops) {
            for (int[] ta : types) {
                for (int[] tb : types) {
                    for (int n : new int[] {1, 63, 64, 65, 300}) {
                        try (Arena arena = Arena.ofConfined()) {
                            boolean wideA = ta[0] > 18, wideB = tb[0] > 18;
                            if (!wideA && !wideB) {
                                continue; // both narrow is the INT64 lane's business
                            }
                            int[] rt = resultType(op, ta[0], ta[1], tb[0], tb[1]);
                            boolean[] na = nulls(rnd, n), nb = nulls(rnd, n);
                            VectorBuffers a = column(arena, rnd, n, ta[0], wideA, na);
                            VectorBuffers b = column(arena, rnd, n, tb[0], wideB, nb);
                            MemorySegment out = ArrowLayout.allocateData(arena, VecType.DECIMAL128, n);
                            MemorySegment overflow = ArrowLayout.allocateBitmap(arena, n);
                            MemorySegment zero = ArrowLayout.allocateBitmap(arena, n);
                            run(op, a, ta, b, tb, rt,
                                    n, out, overflow, zero);
                            for (int i = 0; i < n; i++) {
                                if (na[i] || nb[i]) {
                                    continue;
                                }
                                BigDecimal x = new BigDecimal(a.type() == VecType.DECIMAL128 ? a.getDecimal128(i) : BigInteger.valueOf(a.getLong(i)),
                                        ta[1]);
                                BigDecimal y = new BigDecimal(b.type() == VecType.DECIMAL128 ? b.getDecimal128(i) : BigInteger.valueOf(b.getLong(i)),
                                        tb[1]);
                                if (op == '/' && y.signum() == 0) {
                                    assertTrue(Bitmap.isSet(zero, i), "divisor zero at " + i);
                                    zeros++;
                                    continue;
                                }
                                BigInteger expected = ScalarReference.decimalOp(op, x, y, rt[1], rt[0]);
                                if (expected == null) {
                                    assertTrue(Bitmap.isSet(overflow, i), op
                                            + " overflow expected at "
                                            + i
                                            + ": "
                                            + x
                                            + " "
                                            + y
                                            + " -> "
                                            + rt[0]
                                            + ","
                                            + rt[1]);
                                    overflowed++;
                                } else {
                                    assertFalse(Bitmap.isSet(overflow, i), op
                                            + " no overflow expected at "
                                            + i
                                            + ": "
                                            + x
                                            + " "
                                            + y);
                                    BigInteger actual = Decimal128.toBigInteger(Decimal128.hi(out, i), Decimal128.lo(out, i));
                                    assertEquals(expected, actual, op
                                            + " at "
                                            + i
                                            + ": "
                                            + x
                                            + " "
                                            + op
                                            + " "
                                            + y
                                            + " as decimal("
                                            + rt[0]
                                            + ","
                                            + rt[1]
                                            + ")");
                                }
                                checked++;
                            }
                        }
                    }
                }
            }
        }
        assertTrue(checked > 20000 && overflowed > 100 && zeros > 10,
                checked + " checked, " + overflowed + " overflowed, " + zeros + " zero divisors");
    }

    @Test
    void scalarOperandsAndPureLimbSums() {
        Random rnd = new Random(7);
        try (Arena arena = Arena.ofConfined()) {
            int n = 200;
            BigInteger[] v = new BigInteger[n];
            for (int i = 0; i < n; i++) {
                v[i] = BigInteger.valueOf(rnd.nextLong() >> 3).multiply(BigInteger.valueOf(rnd.nextInt(1000)));
            }
            VectorBuffers a = ArrowLayout.ofDecimal128(arena, v, null);
            BigInteger s = new BigInteger("123456789012345678901234567");
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.DECIMAL128, n);
            MemorySegment overflow = ArrowLayout.allocateBitmap(arena, n);
            // a (38,10) + literal (27,2): the literal shifts by 8 digits, the sum stays within 38 digits here.
            WideDecimalKernels.addSub(WideDecimalKernels.Operand.of(a, 0), WideDecimalKernels.Operand.of(s, 8), false, 10,
                    10, 38, n, out, overflow);
            for (int i = 0; i < n; i++) {
                BigInteger expected = ScalarReference.decimalOp('+', new BigDecimal(v[i], 10), new BigDecimal(s, 2), 10,
                        38);
                assertEquals(
                        expected,
                        Decimal128.toBigInteger(Decimal128.hi(out, i), Decimal128.lo(out, i)),
                        "row " + i);
                assertFalse(Bitmap.isSet(overflow, i));
            }
            // literal - a
            WideDecimalKernels.addSub(WideDecimalKernels.Operand.of(s, 8), WideDecimalKernels.Operand.of(a, 0), true, 10,
                    10, 38, n, out, overflow);
            for (int i = 0; i < n; i++) {
                BigInteger expected = ScalarReference.decimalOp('-', new BigDecimal(s, 2), new BigDecimal(v[i], 10), 10,
                        38);
                assertEquals(
                        expected,
                        Decimal128.toBigInteger(Decimal128.hi(out, i), Decimal128.lo(out, i)),
                        "row " + i);
            }
        }
    }

    private static void run(
            char op,
            VectorBuffers a,
            int[] ta,
            VectorBuffers b,
            int[] tb,
            int[] rt,
            int n,
            MemorySegment out,
            MemorySegment overflow,
            MemorySegment zero) {
        switch (op) {
            case '+', '-' -> {
                int working = Math.max(ta[1], tb[1]);
                WideDecimalKernels.addSub(
                        WideDecimalKernels.Operand.of(a, working - ta[1]),
                        WideDecimalKernels.Operand.of(b, working - tb[1]),
                        op == '-',
                        working,
                        rt[1],
                        rt[0],
                        n,
                        out,
                        overflow);
            }
            case '*' -> WideDecimalKernels.mul(WideDecimalKernels.Operand.of(a, 0), WideDecimalKernels.Operand.of(b, 0), ta[1] + tb[1],
                    rt[1], rt[0], n, out, overflow);
            default -> WideDecimalKernels.divide(
                    WideDecimalKernels.Operand.of(a, 0),
                    ta[1],
                    WideDecimalKernels.Operand.of(b, 0),
                    tb[1],
                    rt[1],
                    rt[0],
                    n,
                    out,
                    overflow,
                    zero);
        }
    }
}
