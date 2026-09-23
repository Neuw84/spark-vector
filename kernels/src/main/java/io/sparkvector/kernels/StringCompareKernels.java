package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * String comparisons in Spark's default {@code UTF8_BINARY} order: unsigned
 * byte-wise lexicographic, shorter prefix first -- exactly {@code
 * UTF8String.compareTo}, which is also code-point order. One result bit per
 * row; null lanes get arbitrary bits, the caller shares the validity.
 *
 * <p>Column vs literal: a dictionary-encoded column is compared once per
 * dictionary entry and the verdicts gathered through the indices, so a Parquet
 * dictionary page costs one compare per distinct value. A plain column is
 * compared row by row; equality checks the length first and then finds the
 * first mismatching byte with {@link MemorySegment#mismatch}, ordered compares
 * decide on that byte (or on the lengths when one string is a prefix of the
 * other). 64-row blocks whose {@code active} word is zero are skipped, like the
 * fixed-width kernels.
 *
 * <p>Column vs column resolves each side's bytes (through the dictionary where
 * there is one) and compares the two ranges the same way.
 */
public final class StringCompareKernels {
    private StringCompareKernels() {}

    /**
     * {@code a <op> s}; {@code s} is the literal's UTF-8 bytes. {@code active}
     * may be null.
     */
    public static void compareScalar(VectorBuffers a, byte[] s, CompareOp op,
            MemorySegment active, MemorySegment out) {
        if (a.type() != VecType.UTF8) {
            throw new IllegalArgumentException("expected UTF8, got " + a.type());
        }
        int n = a.length();
        MemorySegment lit = MemorySegment.ofArray(s);
        VectorBuffers dict = a.dictionary();
        if (dict != null) {
            // One verdict per dictionary entry, then a gather through the indices.
            int m = dict.length();
            boolean[] verdict = new boolean[m];
            MemorySegment doff = dict.offsets();
            MemorySegment ddata = dict.data();
            for (int j = 0; j < m; j++) {
                int start = doff.get(VectorBuffers.LE_INT, (long) j << 2);
                int end = doff.get(VectorBuffers.LE_INT, (long) (j + 1) << 2);
                verdict[j] = op.test(compareBytes(ddata, start, end, lit, 0, s.length));
            }
            gather(a.data(), n, verdict, active, out);
            return;
        }
        MemorySegment off = a.offsets();
        MemorySegment data = a.data();
        boolean eqOnly = op == CompareOp.EQ || op == CompareOp.NE;
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base);
            long word = 0L;
            int start = off.get(VectorBuffers.LE_INT, (long) base << 2);
            for (int k = 0; k < limit; k++) {
                int end = off.get(VectorBuffers.LE_INT, (long) (base + k + 1) << 2);
                boolean hit;
                if (eqOnly) {
                    boolean eq = end - start == s.length && MemorySegment.mismatch(data, start, end, lit, 0, s.length) < 0;
                    hit = (op == CompareOp.EQ) == eq;
                } else {
                    hit = op.test(compareBytes(data, start, end, lit, 0, s.length));
                }
                if (hit) {
                    word |= 1L << k;
                }
                start = end;
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /**
     * {@code a <op> b}; both UTF8 of the same length, either side dictionary
     * encoded or plain.
     */
    public static void compare(VectorBuffers a, VectorBuffers b, CompareOp op,
            MemorySegment active, MemorySegment out) {
        if (a.type() != VecType.UTF8 || b.type() != VecType.UTF8 || a.length() != b.length()) {
            throw new IllegalArgumentException("operands differ: " + a.type() + "/" + b.type());
        }
        int n = a.length();
        Side sa = new Side(a);
        Side sb = new Side(b);
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base);
            long word = 0L;
            for (int k = 0; k < limit; k++) {
                int i = base + k;
                if (!sa.resolve(i) || !sb.resolve(i)) {
                    continue; // out-of-range dictionary index on a null row: validity masks it
                }
                if (op.test(compareBytes(sa.bytes, sa.start, sa.end, sb.bytes, sb.start, sb.end))) {
                    word |= 1L << k;
                }
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /**
     * Gathers one verdict per dictionary entry through a column of {@code n}
     * indices into a result bitmap, skipping blocks with no active row. A null
     * row may carry any index: out-of-range ones are left clear and validity
     * masks them.
     */
    static void gather(MemorySegment idx, int n, boolean[] verdict,
                       MemorySegment active, MemorySegment out) {
        int m = verdict.length;
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base);
            long word = 0L;
            for (int k = 0; k < limit; k++) {
                int j = idx.get(VectorBuffers.LE_INT, (long) (base + k) << 2);
                if (j >= 0 && j < m && verdict[j]) {
                    word |= 1L << k;
                }
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /**
     * Unsigned byte-wise comparison of {@code x[xs, xe)} against {@code y[ys,
     * ye)}: negative, zero or positive like {@code Comparable.compareTo}.
     */
    public static int compareBytes(MemorySegment x, long xs, long xe,
            MemorySegment y, long ys, long ye) {
        long lx = xe - xs, ly = ye - ys;
        long common = Math.min(lx, ly);
        long at = MemorySegment.mismatch(x, xs, xs + common, y, ys,
                ys + common);
        if (at >= 0) {
            return Byte.toUnsignedInt(x.get(ValueLayout.JAVA_BYTE, xs + at)) - Byte.toUnsignedInt(y.get(ValueLayout.JAVA_BYTE, ys + at));
        }
        return Long.compare(lx, ly);
    }

    /**
     * Per-row byte range of one operand, resolved through its dictionary when
     * it has one.
     */
    private static final class Side {
        final MemorySegment indices;
        final MemorySegment offsets;
        final MemorySegment bytes;
        final int dictSize;
        int start;
        int end;

        Side(VectorBuffers v) {
            VectorBuffers dict = v.dictionary();
            if (dict != null) {
                indices = v.data();
                offsets = dict.offsets();
                bytes = dict.data();
                dictSize = dict.length();
            } else {
                indices = null;
                offsets = v.offsets();
                bytes = v.data();
                dictSize = -1;
            }
        }

        boolean resolve(int i) {
            int j = i;
            if (indices != null) {
                j = indices.get(VectorBuffers.LE_INT, (long) i << 2);
                if (j < 0 || j >= dictSize) {
                    return false;
                }
            }
            start = offsets.get(VectorBuffers.LE_INT, (long) j << 2);
            end = offsets.get(VectorBuffers.LE_INT, (long) (j + 1) << 2);
            return true;
        }
    }
}
