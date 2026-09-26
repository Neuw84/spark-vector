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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD comparisons producing a result bitmap (one bit per element, Arrow LSB
 * order).
 *
 * <p>Null lanes are compared like any other lane; their result bits are
 * meaningless and the caller combines the result with the operands' validity
 * (see {@link BitmapKernels#selection} and the Kleene operators). Doubles
 * follow Spark's NaN-safe ordering, not IEEE.
 *
 * <p>Only EQ, LT and LE are implemented as vector loops; NE, GE and GT are
 * their complements (this holds for the NaN-safe ordering too). C2 only
 * intrinsifies {@code Vector.compare} when the operator is a compile-time
 * constant, so every loop spells its operator out literally instead of taking
 * it as a parameter.
 *
 * <p>Each loop fills one 64-element output word at a time so lane masks can be
 * OR-ed straight into the word; the tail uses masked loads so no read runs past
 * the segment.
 */
public final class CompareKernels {

    static final VectorSpecies<Integer> I = Species.I;
    static final VectorSpecies<Long> L = Species.L;
    static final VectorSpecies<Double> D = Species.D;
    static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private CompareKernels() {}

    /** {@code a <op> scalar}. The scalar is widened/narrowed to the column's type. */
    public static void compareScalar(VectorBuffers a, Number scalar, CompareOp op,
            MemorySegment out) {
        compareScalar(a, scalar, op, null, out);
    }

    /**
     * As {@link #compareScalar(VectorBuffers, Number, CompareOp,
     * MemorySegment)}, but 64-row blocks whose {@code active} word is zero are
     * skipped (their output bits are cleared). {@code active} may be {@code
     * null}.
     */
    public static void compareScalar(VectorBuffers a, Number scalar, CompareOp op,
            MemorySegment active, MemorySegment out) {
        MemorySegment d = a.data();
        int n = a.length();
        boolean negate = op == CompareOp.NE || op == CompareOp.GT || op == CompareOp.GE;
        switch (a.type()) {
            case INT32 -> {
                int s = scalar.intValue();
                switch (op) {
                    case EQ, NE -> i32EqScalar(d, n, s, negate, active, out);
                    case LT, GE -> i32LtScalar(d, n, s, negate, active, out);
                    case LE, GT -> i32LeScalar(d, n, s, negate, active, out);
                }
            }
            case INT64 -> {
                long s = scalar.longValue();
                switch (op) {
                    case EQ, NE -> i64EqScalar(d, n, s, negate, active, out);
                    case LT, GE -> i64LtScalar(d, n, s, negate, active, out);
                    case LE, GT -> i64LeScalar(d, n, s, negate, active, out);
                }
            }
            case FLOAT64 -> {
                double s = scalar.doubleValue();
                if (Double.isNaN(s)) {
                    f64NaNScalar(d, n, op, active, out);
                } else {
                    switch (op) {
                        case EQ, NE -> f64EqScalar(d, n, s, negate, active, out);
                        case LT, GE -> f64LtScalar(d, n, s, negate, active, out);
                        case LE, GT -> f64LeScalar(d, n, s, negate, active, out);
                    }
                }
            }
            case DECIMAL128 -> {
                // The scalar is the unscaled value as a BigInteger (already at the column's scale).
                java.math.BigInteger s = (java.math.BigInteger) scalar;
                d128Scalar(d, n, Decimal128.hiOf(s), Decimal128.loOf(s),
                        op, active, out);
            }
            default -> throw new IllegalArgumentException("unsupported compare type " + a.type());
        }
    }

    /**
     * DECIMAL128 against a two-limb scalar, a word of 64 rows at a time: the
     * signed order of the high limbs, then the unsigned order of the low ones
     * ({@link Decimal128#compare}). No species -- the lane deliberately has no
     * SIMD path (#28); this is the scalar loop the reference also runs.
     */
    private static void d128Scalar(
            MemorySegment d,
            int n,
            long hi,
            long lo,
            CompareOp op,
            MemorySegment active,
            MemorySegment out) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int start = w << 6;
            int end = Math.min(n, start + 64);
            long word = 0L;
            for (int i = start; i < end; i++) {
                if (op.test(Decimal128.compare(Decimal128.hi(d, i), Decimal128.lo(d, i), hi, lo))) {
                    word |= 1L << (i - start);
                }
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /** DECIMAL128 against DECIMAL128 at the same scale, row by row over the limbs. */
    static void d128(MemorySegment da, MemorySegment db, int n,
                     CompareOp op, MemorySegment active, MemorySegment out) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int start = w << 6;
            int end = Math.min(n, start + 64);
            long word = 0L;
            for (int i = start; i < end; i++) {
                int c = Decimal128.compare(Decimal128.hi(da, i), Decimal128.lo(da, i), Decimal128.hi(db, i),
                        Decimal128.lo(db, i));
                if (op.test(c)) {
                    word |= 1L << (i - start);
                }
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /** {@code a <op> b}; both columns must share a fixed-width type and length. */
    public static void compare(VectorBuffers a, VectorBuffers b, CompareOp op,
            MemorySegment out) {
        compare(a, b, op, null, out);
    }

    /**
     * As {@link #compare(VectorBuffers, VectorBuffers, CompareOp,
     * MemorySegment)} with block skipping.
     */
    public static void compare(VectorBuffers a, VectorBuffers b, CompareOp op,
            MemorySegment active, MemorySegment out) {
        if (a.type() != b.type() || a.length() != b.length()) {
            throw new IllegalArgumentException("operands differ: " + a.type() + "/" + b.type());
        }
        MemorySegment da = a.data();
        MemorySegment db = b.data();
        int n = a.length();
        boolean negate = op == CompareOp.NE || op == CompareOp.GT || op == CompareOp.GE;
        switch (a.type()) {
            case INT32 -> {
                switch (op) {
                    case EQ, NE -> i32Eq(da, db, n, negate, active, out);
                    case LT, GE -> i32Lt(da, db, n, negate, active, out);
                    case LE, GT -> i32Le(da, db, n, negate, active, out);
                }
            }
            case INT64 -> {
                switch (op) {
                    case EQ, NE -> i64Eq(da, db, n, negate, active, out);
                    case LT, GE -> i64Lt(da, db, n, negate, active, out);
                    case LE, GT -> i64Le(da, db, n, negate, active, out);
                }
            }
            case FLOAT64 -> {
                switch (op) {
                    case EQ, NE -> f64Eq(da, db, n, negate, active, out);
                    case LT, GE -> f64Lt(da, db, n, negate, active, out);
                    case LE, GT -> f64Le(da, db, n, negate, active, out);
                }
            }
            case DECIMAL128 -> d128(da, db, n, op, active, out);
            default -> throw new IllegalArgumentException("unsupported compare type " + a.type());
        }
    }

    private static long finish(long word, boolean negate, int limit) {
        if (negate) {
            word = ~word;
            if (limit < 64) {
                word &= (1L << limit) - 1;
            }
        }
        return word;
    }

    // ================================================================== int32

    static void i32EqScalar(MemorySegment d, int n, int s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = I.length();
        IntVector sv = IntVector.broadcast(I, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE)
                        .compare(VectorOperators.EQ, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Integer> m = I.indexInRange(k, limit);
                word |= IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE, m)
                        .compare(VectorOperators.EQ, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i32LtScalar(MemorySegment d, int n, int s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = I.length();
        IntVector sv = IntVector.broadcast(I, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE)
                        .compare(VectorOperators.LT, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Integer> m = I.indexInRange(k, limit);
                word |= IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE, m)
                        .compare(VectorOperators.LT, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i32LeScalar(MemorySegment d, int n, int s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = I.length();
        IntVector sv = IntVector.broadcast(I, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE)
                        .compare(VectorOperators.LE, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Integer> m = I.indexInRange(k, limit);
                word |= IntVector.fromMemorySegment(I, d, (long) (base + k) << 2, LE, m)
                        .compare(VectorOperators.LE, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i32Eq(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = I.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 2;
                word |= IntVector.fromMemorySegment(I, a, off, LE)
                        .compare(VectorOperators.EQ, IntVector.fromMemorySegment(I, b, off, LE))
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Integer> m = I.indexInRange(k, limit);
                long off = (long) (base + k) << 2;
                word |= IntVector.fromMemorySegment(I, a, off, LE, m)
                        .compare(VectorOperators.EQ, IntVector.fromMemorySegment(I, b, off, LE, m), m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i32Lt(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = I.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 2;
                word |= IntVector.fromMemorySegment(I, a, off, LE)
                        .compare(VectorOperators.LT, IntVector.fromMemorySegment(I, b, off, LE))
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Integer> m = I.indexInRange(k, limit);
                long off = (long) (base + k) << 2;
                word |= IntVector.fromMemorySegment(I, a, off, LE, m)
                        .compare(VectorOperators.LT, IntVector.fromMemorySegment(I, b, off, LE, m), m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i32Le(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = I.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 2;
                word |= IntVector.fromMemorySegment(I, a, off, LE)
                        .compare(VectorOperators.LE, IntVector.fromMemorySegment(I, b, off, LE))
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Integer> m = I.indexInRange(k, limit);
                long off = (long) (base + k) << 2;
                word |= IntVector.fromMemorySegment(I, a, off, LE, m)
                        .compare(VectorOperators.LE, IntVector.fromMemorySegment(I, b, off, LE, m), m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    // ================================================================== int64

    static void i64EqScalar(MemorySegment d, int n, long s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = L.length();
        LongVector sv = LongVector.broadcast(L, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE)
                        .compare(VectorOperators.EQ, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Long> m = L.indexInRange(k, limit);
                word |= LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, m)
                        .compare(VectorOperators.EQ, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i64LtScalar(MemorySegment d, int n, long s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = L.length();
        LongVector sv = LongVector.broadcast(L, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE)
                        .compare(VectorOperators.LT, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Long> m = L.indexInRange(k, limit);
                word |= LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, m)
                        .compare(VectorOperators.LT, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i64LeScalar(MemorySegment d, int n, long s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = L.length();
        LongVector sv = LongVector.broadcast(L, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE)
                        .compare(VectorOperators.LE, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Long> m = L.indexInRange(k, limit);
                word |= LongVector.fromMemorySegment(L, d, (long) (base + k) << 3, LE, m)
                        .compare(VectorOperators.LE, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i64Eq(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = L.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 3;
                word |= LongVector.fromMemorySegment(L, a, off, LE)
                        .compare(VectorOperators.EQ, LongVector.fromMemorySegment(L, b, off, LE))
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Long> m = L.indexInRange(k, limit);
                long off = (long) (base + k) << 3;
                word |= LongVector.fromMemorySegment(L, a, off, LE, m)
                        .compare(VectorOperators.EQ, LongVector.fromMemorySegment(L, b, off, LE, m), m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i64Lt(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = L.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 3;
                word |= LongVector.fromMemorySegment(L, a, off, LE)
                        .compare(VectorOperators.LT, LongVector.fromMemorySegment(L, b, off, LE))
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Long> m = L.indexInRange(k, limit);
                long off = (long) (base + k) << 3;
                word |= LongVector.fromMemorySegment(L, a, off, LE, m)
                        .compare(VectorOperators.LT, LongVector.fromMemorySegment(L, b, off, LE, m), m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void i64Le(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = L.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 3;
                word |= LongVector.fromMemorySegment(L, a, off, LE)
                        .compare(VectorOperators.LE, LongVector.fromMemorySegment(L, b, off, LE))
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Long> m = L.indexInRange(k, limit);
                long off = (long) (base + k) << 3;
                word |= LongVector.fromMemorySegment(L, a, off, LE, m)
                        .compare(VectorOperators.LE, LongVector.fromMemorySegment(L, b, off, LE, m), m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    // ================================================================== float64 (NaN-safe)
    //
    // Against a non-NaN scalar the IEEE result is already correct for EQ/LT/LE (NaN lanes are false
    // and NaN sorts above the scalar, so NaN < s and NaN <= s are false, and so is NaN == s). Their
    // complements (NE/GE/GT) therefore come out right too. Column-vs-column needs the NaN fix-ups.

    static void f64EqScalar(MemorySegment d, int n, double s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = D.length();
        DoubleVector sv = DoubleVector.broadcast(D, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE)
                        .compare(VectorOperators.EQ, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                word |= DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m)
                        .compare(VectorOperators.EQ, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void f64LtScalar(MemorySegment d, int n, double s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = D.length();
        DoubleVector sv = DoubleVector.broadcast(D, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE)
                        .compare(VectorOperators.LT, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                word |= DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m)
                        .compare(VectorOperators.LT, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void f64LeScalar(MemorySegment d, int n, double s,
                            boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = D.length();
        DoubleVector sv = DoubleVector.broadcast(D, s);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                word |= DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE)
                        .compare(VectorOperators.LE, sv)
                        .toLong()
                        << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                word |= DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m)
                        .compare(VectorOperators.LE, sv, m)
                        .toLong()
                        << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    /** Comparisons against a NaN scalar only depend on whether each lane is NaN. */
    static void f64NaNScalar(MemorySegment d, int n, CompareOp op,
            MemorySegment active, MemorySegment out) {
        // isNaN bitmap, then a fixed truth table: NaN equals NaN and is greater than everything.
        int lanes = D.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long nan = 0L;
            for (; k + lanes <= limit; k += lanes) {
                DoubleVector v = DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE);
                nan |= v.compare(VectorOperators.NE, v).toLong() << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                DoubleVector v = DoubleVector.fromMemorySegment(D, d, (long) (base + k) << 3, LE, m);
                nan |= v.compare(VectorOperators.NE, v, m).toLong() << k;
            }
            long all = limit == 64 ? -1L : (1L << limit) - 1;
            long word = switch (op) {
                case EQ, GE -> nan;
                case NE, LT -> ~nan & all;
                case LE -> all;
                case GT -> 0L;
            };
            Bitmap.setWord(out, w, n, word);
        }
    }

    static void f64Eq(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = D.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 3;
                DoubleVector va = DoubleVector.fromMemorySegment(D, a, off, LE);
                DoubleVector vb = DoubleVector.fromMemorySegment(D, b, off, LE);
                // ieeeEQ | (aNaN & bNaN)
                VectorMask<Double> r = va.compare(VectorOperators.EQ, vb).or(va.compare(VectorOperators.NE, va)
                        .and(vb.compare(VectorOperators.NE, vb)));
                word |= r.toLong() << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                long off = (long) (base + k) << 3;
                DoubleVector va = DoubleVector.fromMemorySegment(D, a, off, LE, m);
                DoubleVector vb = DoubleVector.fromMemorySegment(D, b, off, LE, m);
                VectorMask<Double> r = va.compare(VectorOperators.EQ, vb)
                        .or(va.compare(VectorOperators.NE, va).and(vb.compare(VectorOperators.NE, vb)))
                        .and(m);
                word |= r.toLong() << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void f64Lt(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = D.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 3;
                DoubleVector va = DoubleVector.fromMemorySegment(D, a, off, LE);
                DoubleVector vb = DoubleVector.fromMemorySegment(D, b, off, LE);
                // ieeeLT | (!aNaN & bNaN)
                VectorMask<Double> r = va.compare(VectorOperators.LT, vb).or(va.compare(VectorOperators.EQ, va)
                        .and(vb.compare(VectorOperators.NE, vb)));
                word |= r.toLong() << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                long off = (long) (base + k) << 3;
                DoubleVector va = DoubleVector.fromMemorySegment(D, a, off, LE, m);
                DoubleVector vb = DoubleVector.fromMemorySegment(D, b, off, LE, m);
                VectorMask<Double> r = va.compare(VectorOperators.LT, vb)
                        .or(va.compare(VectorOperators.EQ, va).and(vb.compare(VectorOperators.NE, vb)))
                        .and(m);
                word |= r.toLong() << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }

    static void f64Le(MemorySegment a, MemorySegment b, int n,
                      boolean neg, MemorySegment active, MemorySegment out) {
        int lanes = D.length();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L); // no live row in this block: skip the compares
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base), k = 0;
            long word = 0L;
            for (; k + lanes <= limit; k += lanes) {
                long off = (long) (base + k) << 3;
                DoubleVector va = DoubleVector.fromMemorySegment(D, a, off, LE);
                DoubleVector vb = DoubleVector.fromMemorySegment(D, b, off, LE);
                // ieeeLE | bNaN
                VectorMask<Double> r = va.compare(VectorOperators.LE, vb).or(vb.compare(VectorOperators.NE, vb));
                word |= r.toLong() << k;
            }
            if (k < limit) {
                VectorMask<Double> m = D.indexInRange(k, limit);
                long off = (long) (base + k) << 3;
                DoubleVector va = DoubleVector.fromMemorySegment(D, a, off, LE, m);
                DoubleVector vb = DoubleVector.fromMemorySegment(D, b, off, LE, m);
                VectorMask<Double> r = va.compare(VectorOperators.LE, vb)
                        .or(vb.compare(VectorOperators.NE, vb))
                        .and(m);
                word |= r.toLong() << k;
            }
            Bitmap.setWord(out, w, n, finish(word, neg, limit));
        }
    }
}
