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

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Overflow detection for integer {@code + - *} and negation, the piece ANSI
 * mode needs on top of {@link ArithKernels}: the arithmetic is computed
 * wrapping, as before, and this kernel sets a bit in {@code out} for every lane
 * whose true result did not fit the lane. The caller ANDs that mask with the
 * validity and the active rows and raises Spark's error only if anything
 * survives, so rows a filter removed or an earlier conjunct decided never raise
 * -- Spark's short-circuit rule.
 *
 * <p>Add and subtract use the sign trick ({@code ((a ^ r) & (b ^ r)) < 0} for
 * add, {@code ((a ^ b) & (a ^ r)) < 0} for subtract) over full lanes. Multiply
 * has no lane-wide equivalent: INT32 products are checked through the exact
 * 64-bit product, INT64 products through {@link Math#multiplyHigh}, both scalar
 * per lane. Negation overflows exactly at {@code MIN_VALUE}. Bits beyond {@code
 * n} in the last word are left clear.
 */
public final class OverflowKernels {
    static final VectorSpecies<Integer> I = Species.I;
    static final VectorSpecies<Long> L = Species.L;
    static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private OverflowKernels() {}

    /**
     * Lanes where {@code a <op> b} overflowed; {@code r} is the wrapped result
     * already computed.
     */
    public static void overflow(ArithOp op, VectorBuffers a, VectorBuffers b,
            MemorySegment r, MemorySegment out) {
        int n = a.length();
        Bitmap.fill(out, n, false);
        switch (a.type()) {
            case INT32 -> {
                switch (op) {
                    case ADD, SUB -> i32AddSub(op, a.data(), b.data(), r,
                            n, out);
                    case MUL -> i32Mul(a.data(), b.data(), n, out);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            case INT64 -> {
                switch (op) {
                    case ADD, SUB -> i64AddSub(op, a.data(), b.data(), r,
                            n, out);
                    case MUL -> i64Mul(a.data(), b.data(), n, out);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            default -> throw new IllegalArgumentException("no overflow check for " + a.type());
        }
    }

    /** Lanes where {@code a <op> s} overflowed. */
    public static void overflowScalar(ArithOp op, VectorBuffers a, Number s,
            MemorySegment r, MemorySegment out) {
        int n = a.length();
        Bitmap.fill(out, n, false);
        switch (a.type()) {
            case INT32 -> {
                int sv = s.intValue();
                switch (op) {
                    case ADD, SUB -> i32AddSubScalar(op, a.data(), sv, r, n,
                            out, false);
                    case MUL -> i32MulScalar(a.data(), sv, n, out);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            case INT64 -> {
                long sv = s.longValue();
                switch (op) {
                    case ADD, SUB -> i64AddSubScalar(op, a.data(), sv, r, n,
                            out, false);
                    case MUL -> i64MulScalar(a.data(), sv, n, out);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            default -> throw new IllegalArgumentException("no overflow check for " + a.type());
        }
    }

    /** Lanes where {@code s <op> b} overflowed. */
    public static void scalarOverflow(ArithOp op, Number s, VectorBuffers b,
            MemorySegment r, MemorySegment out) {
        int n = b.length();
        Bitmap.fill(out, n, false);
        switch (b.type()) {
            case INT32 -> {
                int sv = s.intValue();
                switch (op) {
                    case ADD -> i32AddSubScalar(op, b.data(), sv, r, n,
                            out, false);
                    case SUB -> i32AddSubScalar(op, b.data(), sv, r, n,
                            out, true);
                    case MUL -> i32MulScalar(b.data(), sv, n, out);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            case INT64 -> {
                long sv = s.longValue();
                switch (op) {
                    case ADD -> i64AddSubScalar(op, b.data(), sv, r, n,
                            out, false);
                    case SUB -> i64AddSubScalar(op, b.data(), sv, r, n,
                            out, true);
                    case MUL -> i64MulScalar(b.data(), sv, n, out);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            default -> throw new IllegalArgumentException("no overflow check for " + b.type());
        }
    }

    /** Lanes where {@code -a} overflowed, i.e. {@code a == MIN_VALUE}. */
    public static void negateOverflow(VectorBuffers a, MemorySegment out) {
        int n = a.length();
        Bitmap.fill(out, n, false);
        MemorySegment d = a.data();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    if (d.getAtIndex(VectorBuffers.LE_INT, i) == Integer.MIN_VALUE) {
                        Bitmap.set(out, i);
                    }
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    if (d.getAtIndex(VectorBuffers.LE_LONG, i) == Long.MIN_VALUE) {
                        Bitmap.set(out, i);
                    }
                }
            }
            default -> throw new IllegalArgumentException("no overflow check for " + a.type());
        }
    }

    // ---------------------------------------------------------------- add / subtract, sign trick

    private static void i32AddSub(ArithOp op, MemorySegment a, MemorySegment b,
            MemorySegment r, int n, MemorySegment out) {
        int lanes = I.length();
        int i = 0;
        for (; i + lanes <= n; i += lanes) {
            IntVector va = IntVector.fromMemorySegment(I, a, (long) i << 2, LE);
            IntVector vb = IntVector.fromMemorySegment(I, b, (long) i << 2, LE);
            IntVector vr = IntVector.fromMemorySegment(I, r, (long) i << 2, LE);
            IntVector ov = op == ArithOp.ADD ? va.lanewise(VectorOperators.XOR, vr).and(vb.lanewise(VectorOperators.XOR, vr)) : va.lanewise(VectorOperators.XOR, vb).and(va.lanewise(VectorOperators.XOR, vr));
            setBits(out, i,
                    ov.compare(VectorOperators.LT, 0).toLong(), lanes);
        }
        for (; i < n; i++) {
            int x = a.getAtIndex(VectorBuffers.LE_INT, i);
            int y = b.getAtIndex(VectorBuffers.LE_INT, i);
            int z = r.getAtIndex(VectorBuffers.LE_INT, i);
            if (addSubOverflow(op, x, y, z)) {
                Bitmap.set(out, i);
            }
        }
    }

    private static void i64AddSub(ArithOp op, MemorySegment a, MemorySegment b,
            MemorySegment r, int n, MemorySegment out) {
        int lanes = L.length();
        int i = 0;
        for (; i + lanes <= n; i += lanes) {
            LongVector va = LongVector.fromMemorySegment(L, a, (long) i << 3, LE);
            LongVector vb = LongVector.fromMemorySegment(L, b, (long) i << 3, LE);
            LongVector vr = LongVector.fromMemorySegment(L, r, (long) i << 3, LE);
            LongVector ov = op == ArithOp.ADD ? va.lanewise(VectorOperators.XOR, vr).and(vb.lanewise(VectorOperators.XOR, vr)) : va.lanewise(VectorOperators.XOR, vb).and(va.lanewise(VectorOperators.XOR, vr));
            setBits(out, i,
                    ov.compare(VectorOperators.LT, 0L).toLong(), lanes);
        }
        for (; i < n; i++) {
            long x = a.getAtIndex(VectorBuffers.LE_LONG, i);
            long y = b.getAtIndex(VectorBuffers.LE_LONG, i);
            long z = r.getAtIndex(VectorBuffers.LE_LONG, i);
            if (addSubOverflow(op, x, y, z)) {
                Bitmap.set(out, i);
            }
        }
    }

    /**
     * Column against a scalar. {@code reversed} is the {@code s - a} form,
     * whose sign trick is the subtract rule with the operands swapped.
     */
    private static void i32AddSubScalar(
            ArithOp op,
            MemorySegment a,
            int s,
            MemorySegment r,
            int n,
            MemorySegment out,
            boolean reversed) {
        for (int i = 0; i < n; i++) {
            int x = a.getAtIndex(VectorBuffers.LE_INT, i);
            int z = r.getAtIndex(VectorBuffers.LE_INT, i);
            boolean ov = reversed ? addSubOverflow(op, s, x, z) : addSubOverflow(op, x, s, z);
            if (ov) {
                Bitmap.set(out, i);
            }
        }
    }

    private static void i64AddSubScalar(
            ArithOp op,
            MemorySegment a,
            long s,
            MemorySegment r,
            int n,
            MemorySegment out,
            boolean reversed) {
        for (int i = 0; i < n; i++) {
            long x = a.getAtIndex(VectorBuffers.LE_LONG, i);
            long z = r.getAtIndex(VectorBuffers.LE_LONG, i);
            boolean ov = reversed ? addSubOverflow(op, s, x, z) : addSubOverflow(op, x, s, z);
            if (ov) {
                Bitmap.set(out, i);
            }
        }
    }

    static boolean addSubOverflow(ArithOp op, int a, int b,
            int r) {
        return op == ArithOp.ADD ? ((a ^ r) & (b ^ r)) < 0 : ((a ^ b) & (a ^ r)) < 0;
    }

    static boolean addSubOverflow(ArithOp op, long a, long b,
            long r) {
        return op == ArithOp.ADD ? ((a ^ r) & (b ^ r)) < 0 : ((a ^ b) & (a ^ r)) < 0;
    }

    // ---------------------------------------------------------------- multiply, exact product

    private static void i32Mul(MemorySegment a, MemorySegment b, int n,
            MemorySegment out) {
        for (int i = 0; i < n; i++) {
            long p = (long) a.getAtIndex(VectorBuffers.LE_INT, i) * b.getAtIndex(VectorBuffers.LE_INT, i);
            if ((int) p != p) {
                Bitmap.set(out, i);
            }
        }
    }

    private static void i32MulScalar(MemorySegment a, int s, int n,
            MemorySegment out) {
        for (int i = 0; i < n; i++) {
            long p = (long) a.getAtIndex(VectorBuffers.LE_INT, i) * s;
            if ((int) p != p) {
                Bitmap.set(out, i);
            }
        }
    }

    private static void i64Mul(MemorySegment a, MemorySegment b, int n,
            MemorySegment out) {
        for (int i = 0; i < n; i++) {
            if (mulOverflow(a.getAtIndex(VectorBuffers.LE_LONG, i), b.getAtIndex(VectorBuffers.LE_LONG, i))) {
                Bitmap.set(out, i);
            }
        }
    }

    private static void i64MulScalar(MemorySegment a, long s, int n,
            MemorySegment out) {
        for (int i = 0; i < n; i++) {
            if (mulOverflow(a.getAtIndex(VectorBuffers.LE_LONG, i), s)) {
                Bitmap.set(out, i);
            }
        }
    }

    /**
     * The 128-bit product fits a long iff its high word is the sign extension
     * of its low word.
     */
    static boolean mulOverflow(long x, long y) {
        long lo = x * y;
        long hi = Math.multiplyHigh(x, y);
        return hi != (lo >> 63);
    }

    /**
     * Writes {@code lanes} mask bits starting at bit {@code i} (an aligned run
     * inside one word).
     */
    private static void setBits(MemorySegment out, int i, long bits,
            int lanes) {
        if (bits == 0L) {
            return;
        }
        for (int k = 0; k < lanes; k++) {
            if ((bits & (1L << k)) != 0L) {
                Bitmap.set(out, i + k);
            }
        }
    }
}
