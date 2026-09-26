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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import io.vecruntime.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ArithKernelsTest {

    private static SegmentVectorBuffers wideInts(Arena arena, Random rnd, int n) {
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            // Include extremes so wrap-around is exercised.
            v[i] = switch (rnd.nextInt(8)) {
                        case 0 -> Integer.MAX_VALUE;
                        case 1 -> Integer.MIN_VALUE;
                        default -> rnd.nextInt();
                    };
        }
        return ArrowLayout.ofInts(arena, v, null);
    }

    private static SegmentVectorBuffers wideLongs(Arena arena, Random rnd, int n) {
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
            v[i] = switch (rnd.nextInt(8)) {
                        case 0 -> Long.MAX_VALUE;
                        case 1 -> Long.MIN_VALUE;
                        default -> rnd.nextLong();
                    };
        }
        return ArrowLayout.ofLongs(arena, v, null);
    }

    private static void assertSameData(VecType t, MemorySegment exp, MemorySegment act,
            int n, String what) {
        for (int i = 0; i < n; i++) {
            switch (t) {
                case INT32 -> assertEquals(exp.get(VectorBuffers.LE_INT, (long) i << 2),
                        act.get(VectorBuffers.LE_INT, (long) i << 2), what + " @" + i);
                case INT64 -> assertEquals(exp.get(VectorBuffers.LE_LONG, (long) i << 3),
                        act.get(VectorBuffers.LE_LONG, (long) i << 3), what + " @" + i);
                case FLOAT64 -> assertEquals(Double.doubleToLongBits(exp.get(VectorBuffers.LE_DOUBLE, (long) i << 3)), Double.doubleToLongBits(act.get(VectorBuffers.LE_DOUBLE, (long) i << 3)),
                        what + " @" + i);
                default -> throw new IllegalStateException();
            }
        }
    }

    @ParameterizedTest
    @EnumSource(ArithOp.class)
    void columnColumnMatchesReference(ArithOp op) {
        Random rnd = new Random(op.name()
                .hashCode());
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                VectorBuffers[] as = {wideInts(arena, rnd, n), wideLongs(arena, rnd, n), TestData.doubles(arena, rnd, n, null)};
                VectorBuffers[] bs = {wideInts(arena, rnd, n), wideLongs(arena, rnd, n), TestData.doubles(arena, rnd, n, null)};
                for (int t = 0; t < 3; t++) {
                    if (op == ArithOp.DIV && as[t].type() != VecType.FLOAT64) {
                        continue;
                    }
                    MemorySegment exp = ArrowLayout.allocateData(arena, as[t].type(), n);
                    MemorySegment act = ArrowLayout.allocateData(arena, as[t].type(), n);
                    act.fill((byte) 0x7F);
                    ScalarReference.arith(op, as[t], bs[t], exp);
                    ArithKernels.arith(op, as[t], bs[t], act);
                    assertSameData(as[t].type(), exp, act, n,
                            as[t].type() + " " + op);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(ArithOp.class)
    void scalarOperandsMatchReference(ArithOp op) {
        Random rnd = new Random(100L + op.name().hashCode());
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                VectorBuffers[] as = {wideInts(arena, rnd, n), wideLongs(arena, rnd, n), TestData.doubles(arena, rnd, n, null)};
                Number[] scalars = {rnd.nextBoolean() ? Integer.MIN_VALUE : rnd.nextInt(-9, 9),
                        rnd.nextLong(-9, 9), rnd.nextInt(-3, 4) * 0.5};
                for (int t = 0; t < 3; t++) {
                    if (op == ArithOp.DIV && as[t].type() != VecType.FLOAT64) {
                        continue;
                    }
                    MemorySegment exp = ArrowLayout.allocateData(arena, as[t].type(), n);
                    MemorySegment act = ArrowLayout.allocateData(arena, as[t].type(), n);
                    ScalarReference.arithScalar(op, as[t], scalars[t], exp);
                    ArithKernels.arithScalar(op, as[t], scalars[t], act);
                    assertSameData(as[t].type(), exp, act, n,
                            as[t].type() + " col " + op + " scalar");

                    ScalarReference.scalarArith(op, scalars[t], as[t], exp);
                    ArithKernels.scalarArith(op, scalars[t], as[t], act);
                    assertSameData(as[t].type(), exp, act, n,
                            as[t].type() + " scalar " + op + " col");
                }
            }
        }
    }

    @Test
    void negateMatchesReferenceIncludingSignedZeroAndMinValue() {
        Random rnd = new Random(7);
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                VectorBuffers[] as = {wideInts(arena, rnd, n), wideLongs(arena, rnd, n), TestData.doubles(arena, rnd, n, null)};
                for (VectorBuffers a : as) {
                    MemorySegment exp = ArrowLayout.allocateData(arena, a.type(), n);
                    MemorySegment act = ArrowLayout.allocateData(arena, a.type(), n);
                    ScalarReference.negate(a, exp);
                    ArithKernels.negate(a, act);
                    assertSameData(a.type(), exp, act, n,
                            a.type() + " neg");
                }
            }
        }
    }

    @Test
    void divisionByZeroProducesInfOrNanValuesForCallerToMaskOut() {
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofDoubles(
                    arena,
                    new double[] {1.0, -1.0, 0.0, 5.0},
                    null);
            VectorBuffers b = ArrowLayout.ofDoubles(
                    arena,
                    new double[] {0.0, 0.0, 0.0, -0.0},
                    null);
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.FLOAT64, 4);
            ArithKernels.arith(ArithOp.DIV, a, b, out);
            assertEquals(Double.POSITIVE_INFINITY, out.get(VectorBuffers.LE_DOUBLE, 0));
            assertEquals(Double.NEGATIVE_INFINITY, out.get(VectorBuffers.LE_DOUBLE, 8));
            assertEquals(true, Double.isNaN(out.get(VectorBuffers.LE_DOUBLE, 16)));
            // The caller derives the null mask with a compare against zero; -0.0 == 0.0 there.
            MemorySegment zero = ArrowLayout.allocateBitmap(arena, 4);
            CompareKernels.compareScalar(b, 0.0, CompareOp.EQ, zero);
            assertEquals(4, Bitmap.popcount(zero, 4));
        }
    }
}
