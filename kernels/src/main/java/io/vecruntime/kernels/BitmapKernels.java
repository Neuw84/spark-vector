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

/**
 * Word-at-a-time logic over bitmaps: boolean algebra, Spark's three-valued
 * AND/OR, null tests and selection masks. A {@code null} validity segment means
 * "all valid" throughout.
 *
 * <p>These deliberately use 64-bit scalar words rather than {@code LongVector}:
 * bitmaps are 1/32 to 1/64 the size of the data they describe, and validity
 * segments coming from Arrow are not guaranteed to be padded to a full SIMD
 * register, so the tail handling would cost more than the vectorization saves.
 */
public final class BitmapKernels {

    private BitmapKernels() {}

    public static void and(MemorySegment a, MemorySegment b, MemorySegment out,
                           int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            Bitmap.setWord(out, w, n, Bitmap.wordAt(a, w, n) & Bitmap.wordAt(b, w, n));
        }
    }

    public static void or(MemorySegment a, MemorySegment b, MemorySegment out,
                          int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            Bitmap.setWord(out, w, n, Bitmap.wordAt(a, w, n) | Bitmap.wordAt(b, w, n));
        }
    }

    /** {@code a & ~b}. */
    public static void andNot(MemorySegment a, MemorySegment b, MemorySegment out,
            int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            Bitmap.setWord(out, w, n, Bitmap.wordAt(a, w, n) & ~Bitmap.wordAt(b, w, n));
        }
    }

    public static void not(MemorySegment a, MemorySegment out, int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            Bitmap.setWord(out, w, n, ~Bitmap.wordAt(a, w, n));
        }
    }

    public static void copy(MemorySegment a, MemorySegment out, int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            Bitmap.setWord(out, w, n, Bitmap.wordAt(a, w, n));
        }
    }

    /**
     * Validity of a result that is null whenever any operand is null: the AND
     * of the operands' validity bitmaps. Returns {@code false} and leaves
     * {@code out} untouched if neither operand has nulls (the caller then uses
     * a {@code null} validity).
     */
    public static boolean combineValidity(MemorySegment aValid, MemorySegment bValid, MemorySegment out,
            int n) {
        if (aValid == null && bValid == null) {
            return false;
        }
        if (aValid == null) {
            copy(bValid, out, n);
        } else if (bValid == null) {
            copy(aValid, out, n);
        } else {
            and(aValid, bValid, out, n);
        }
        return true;
    }

    /**
     * Selection mask: element is true and not null (nulls are treated as false,
     * as in WHERE).
     */
    public static void selection(MemorySegment bits, MemorySegment validity, MemorySegment out,
            int n) {
        if (validity == null) {
            copy(bits, out, n);
        } else {
            and(bits, validity, out, n);
        }
    }

    /**
     * {@code IS NULL}: result bits are the inverted validity; the result itself
     * is never null.
     */
    public static void isNull(MemorySegment validity, MemorySegment out, int n) {
        if (validity == null) {
            Bitmap.fill(out, n, false);
        } else {
            not(validity, out, n);
        }
    }

    public static void isNotNull(MemorySegment validity, MemorySegment out, int n) {
        if (validity == null) {
            Bitmap.fill(out, n, true);
        } else {
            copy(validity, out, n);
        }
    }

    /**
     * Three-valued AND. {@code false AND null = false}; {@code true AND null =
     * null}. Output validity is always written.
     */
    public static void kleeneAnd(
            MemorySegment aBits,
            MemorySegment aValid,
            MemorySegment bBits,
            MemorySegment bValid,
            MemorySegment outBits,
            MemorySegment outValid,
            int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            long av = aValid == null ? -1L : Bitmap.wordAt(aValid, w, n);
            long bv = bValid == null ? -1L : Bitmap.wordAt(bValid, w, n);
            long ab = Bitmap.wordAt(aBits, w, n);
            long bb = Bitmap.wordAt(bBits, w, n);
            long aTrue = ab & av;
            long aFalse = ~ab & av;
            long bTrue = bb & bv;
            long bFalse = ~bb & bv;
            long t = aTrue & bTrue;
            Bitmap.setWord(outBits, w, n, t);
            Bitmap.setWord(outValid, w, n, t | aFalse | bFalse);
        }
    }

    /** Three-valued OR. {@code true OR null = true}; {@code false OR null = null}. */
    public static void kleeneOr(
            MemorySegment aBits,
            MemorySegment aValid,
            MemorySegment bBits,
            MemorySegment bValid,
            MemorySegment outBits,
            MemorySegment outValid,
            int n) {
        int words = Bitmap.wordsFor(n);
        for (int w = 0; w < words; w++) {
            long av = aValid == null ? -1L : Bitmap.wordAt(aValid, w, n);
            long bv = bValid == null ? -1L : Bitmap.wordAt(bValid, w, n);
            long ab = Bitmap.wordAt(aBits, w, n);
            long bb = Bitmap.wordAt(bBits, w, n);
            long aTrue = ab & av;
            long aFalse = ~ab & av;
            long bTrue = bb & bv;
            long bFalse = ~bb & bv;
            long t = aTrue | bTrue;
            Bitmap.setWord(outBits, w, n, t);
            Bitmap.setWord(outValid, w, n, t | (aFalse & bFalse));
        }
    }
}
