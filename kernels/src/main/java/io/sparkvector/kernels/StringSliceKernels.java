package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * String functions that <em>write</em> new UTF8 data: {@code substring}, {@code
 * lpad}/{@code rpad}, {@code repeat}, {@code space} and {@code overlay}. Every
 * kernel runs in two passes -- the first computes each row's output byte length
 * (and, for the slices, where its bytes come from), the second writes into a
 * data buffer sized once from the prefix sum of those lengths -- and returns a
 * fresh UTF8 {@link SegmentVectorBuffers} carrying the caller's validity.
 *
 * <p>Semantics are Spark's {@code UTF8String} to the letter: positions are
 * 1-based code points, {@code substringSQL}'s negative start counts from the
 * end and an out-of-range range clamps silently ({@link #substringRange}); code
 * point widths come from the same first-byte table ({@link
 * #numBytesForFirstByte}); padding truncates to {@code len} code points when
 * the string is long enough and repeats the pad string, plus a partial one,
 * otherwise. Dictionary-encoded input is read through the dictionary; the
 * output is always plain.
 *
 * <p>Integer arguments are either a scalar or an INT32 lane ({@code col ==
 * null} means scalar). Null rows (validity clear) produce no bytes; the caller
 * supplies the combined validity. The total output is capped at {@link
 * #MAX_OUTPUT_BYTES} per batch: past it the kernel refuses rather than allocate
 * unbounded native memory (a {@code repeat} or {@code space} with a runaway
 * argument).
 */
public final class StringSliceKernels {
    private StringSliceKernels() {}

    /** The most output bytes one batch may produce. */
    public static final long MAX_OUTPUT_BYTES = 1L << 30;

    // ---------------------------------------------------------------- UTF-8 helpers

    /**
     * Bytes of the code point starting with {@code b}, as Spark's {@code
     * UTF8String.numBytesForFirstByte}.
     */
    public static int numBytesForFirstByte(byte b) {
        int u = b & 0xFF;
        if (u < 0x80) {
            return 1;
        }
        if (u >= 0xC2 && u <= 0xDF) {
            return 2;
        }
        if (u >= 0xE0 && u <= 0xEF) {
            return 3;
        }
        if (u >= 0xF0 && u <= 0xF4) {
            return 4;
        }
        return 1; // continuation bytes and the disallowed leaders step one byte, as Spark's table (0 -> 1)
    }

    /** Code points in {@code len} bytes at {@code start}. */
    public static int numChars(MemorySegment data, long start, int len) {
        int i = 0;
        int n = 0;
        while (i < len) {
            i += numBytesForFirstByte(data.get(ValueLayout.JAVA_BYTE, start + i));
            n++;
        }
        return n;
    }

    /**
     * Byte offset (within the string) after {@code chars} code points, clamped
     * to {@code len}.
     */
    static int byteOffsetOfChar(MemorySegment data, long start, int len,
            int chars) {
        int i = 0;
        int c = 0;
        while (i < len && c < chars) {
            i += numBytesForFirstByte(data.get(ValueLayout.JAVA_BYTE, start + i));
            c++;
        }
        return Math.min(i, len);
    }

    /**
     * Spark's {@code substringSQL(pos, length)} then {@code substring(start,
     * until)} over the string at {@code start}/{@code len}: returns the byte
     * range as {@code (offset << 32) | byteLength}, both relative to the
     * string.
     */
    public static long substringRange(MemorySegment data, long start, int len,
            int pos, int length) {
        int chars = pos < 0 ? numChars(data, start, len) : 0;
        int from = pos > 0
                ? pos - 1
                : (pos < 0 ? chars + pos : 0);
        long untilL = (long) from + length;
        int until = untilL > Integer.MAX_VALUE
                ? Integer.MAX_VALUE
                : (untilL < Integer.MIN_VALUE ? Integer.MIN_VALUE : (int) untilL);
        if (until <= from || from >= len) {
            return 0L;
        }
        // Walk to `from` code points (a negative `from` walks nothing, as Spark's loop), then to `until`.
        int i = 0;
        int c = 0;
        while (i < len && c < from) {
            i += numBytesForFirstByte(data.get(ValueLayout.JAVA_BYTE, start + i));
            c++;
        }
        int j = i;
        if (until == Integer.MAX_VALUE) {
            i = len;
        } else {
            while (i < len && c < until) {
                i += numBytesForFirstByte(data.get(ValueLayout.JAVA_BYTE, start + i));
                c++;
            }
        }
        i = Math.min(i, len);
        return i > j ? (((long) j) << 32) | (i - j) : 0L;
    }

    // ---------------------------------------------------------------- input access

    static VectorBuffers store(VectorBuffers v) {
        return v.dictionary() != null ? v.dictionary() : v;
    }

    static int index(VectorBuffers v, int i) {
        return v.dictionary() != null ? v.getInt(i) : i;
    }

    static int startOf(VectorBuffers store, int idx) {
        return store.offsets().getAtIndex(VectorBuffers.LE_INT, idx);
    }

    static int lengthOf(VectorBuffers store, int idx) {
        return store.offsets().getAtIndex(VectorBuffers.LE_INT, idx + 1) - startOf(store, idx);
    }

    private static int arg(VectorBuffers col, int scalar, int i) {
        return col == null ? scalar : col.getInt(i);
    }

    private static boolean live(MemorySegment validity, int i) {
        return validity == null || Bitmap.isSet(validity, i);
    }

    /** Allocates the output offsets from per-row lengths; returns the total. */
    private static long prefix(int[] outLen, MemorySegment offsets, int n) {
        long total = 0;
        offsets.setAtIndex(VectorBuffers.LE_INT, 0, 0);
        for (int i = 0; i < n; i++) {
            total += outLen[i];
            if (total > MAX_OUTPUT_BYTES) {
                throw new IllegalStateException("string output of more than " + MAX_OUTPUT_BYTES + " bytes in one batch");
            }
            offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, (int) total);
        }
        return total;
    }

    private static void fillRepeated(MemorySegment out, long at, MemorySegment src,
            long srcStart, int srcLen, int times) {
        for (int k = 0; k < times; k++) {
            MemorySegment.copy(src, srcStart, out, at, srcLen);
            at += srcLen;
        }
    }

    // ---------------------------------------------------------------- substring

    /**
     * {@code substring(str, pos, len)}: a scalar or INT32 lane for each of
     * {@code pos} and {@code len}.
     */
    public static SegmentVectorBuffers substring(
            VectorBuffers str,
            VectorBuffers posCol,
            int pos,
            VectorBuffers lenCol,
            int len,
            MemorySegment validity,
            Arena arena) {
        int n = str.length();
        VectorBuffers s = store(str);
        int m = s.length();
        // The store's offsets and bytes on the heap once, the ranges computed there and the slices
        // gathered with System.arraycopy into one output copied out once (#400): a MemorySegment.copy
        // and two offset reads per value were 17.5% of an executor's time in q8 at SF10. With a
        // dictionary-encoded input and literal arguments the range is computed once per dictionary
        // entry, not once per row.
        int[] off = s.offsets()
                     .asSlice(0, ((long) m + 1) << 2)
                     .toArray(VectorBuffers.LE_INT);
        byte[] bytes = s.data()
                        .asSlice(0, off[m])
                        .toArray(ValueLayout.JAVA_BYTE);
        MemorySegment heapData = MemorySegment.ofArray(bytes);
        int[] srcOff = new int[n];
        int[] outLen = new int[n];
        boolean encoded = str.dictionary() != null;
        int[] codes = encoded ? str.data()
                .asSlice(0, (long) n << 2)
                .toArray(VectorBuffers.LE_INT)
                : null;
        if (encoded && posCol == null && lenCol == null) {
            int[] dOff = new int[m];
            int[] dLen = new int[m];
            for (int k = 0; k < m; k++) {
                long range = substringRange(heapData, off[k], off[k + 1] - off[k], pos, len);
                dOff[k] = off[k] + (int) (range >>> 32);
                dLen[k] = (int) range;
            }
            for (int i = 0; i < n; i++) {
                if (!live(validity, i)) {
                    continue;
                }
                int k = codes[i];
                srcOff[i] = dOff[k];
                outLen[i] = dLen[k];
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (!live(validity, i)) {
                    continue;
                }
                int idx = encoded ? codes[i] : i;
                long range = substringRange(heapData, off[idx], off[idx + 1] - off[idx], arg(posCol, pos, i),
                        arg(lenCol, len, i));
                srcOff[i] = off[idx] + (int) (range >>> 32);
                outLen[i] = (int) range;
            }
        }
        int[] outOff = new int[n + 1];
        long total = 0;
        for (int i = 0; i < n; i++) {
            total += outLen[i];
            if (total > MAX_OUTPUT_BYTES) {
                throw new IllegalStateException("string output of more than " + MAX_OUTPUT_BYTES + " bytes in one batch");
            }
            outOff[i + 1] = (int) total;
        }
        byte[] outBytes = new byte[(int) total];
        for (int i = 0; i < n; i++) {
            if (outLen[i] > 0) {
                System.arraycopy(bytes, srcOff[i], outBytes, outOff[i], outLen[i]);
            }
        }
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        MemorySegment.copy(outOff, 0, offsets, VectorBuffers.LE_INT, 0,
                n + 1);
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        MemorySegment.copy(outBytes, 0, out, ValueLayout.JAVA_BYTE, 0, outBytes.length);
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    // ---------------------------------------------------------------- lpad / rpad

    /**
     * {@code lpad} / {@code rpad(str, len, pad)}: {@code padCol} is a UTF8 lane
     * or {@code null} with {@code padBytes} the literal pad. Spark's rule: a
     * non-positive {@code len} gives the empty string; a string of at least
     * {@code len} code points is cut to {@code len}; otherwise whole copies of
     * the pad and then a prefix of it fill the gap (no padding at all when the
     * pad is empty).
     */
    public static SegmentVectorBuffers pad(
            VectorBuffers str,
            VectorBuffers lenCol,
            int len,
            VectorBuffers padCol,
            byte[] padBytes,
            boolean left,
            MemorySegment validity,
            Arena arena) {
        int n = str.length();
        VectorBuffers s = store(str);
        MemorySegment data = s.data();
        VectorBuffers p = padCol == null ? null : store(padCol);
        MemorySegment padLit = padCol == null ? MemorySegment.ofArray(padBytes) : null;
        int[] keep = new int[n]; // bytes of the string kept
        int[] copies = new int[n]; // whole pad copies
        int[] remain = new int[n]; // bytes of the partial pad
        int[] outLen = new int[n];
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            int idx = index(str, i);
            int sStart = startOf(s, idx);
            int sLen = lengthOf(s, idx);
            int target = arg(lenCol, len, i);
            if (target <= 0) {
                continue;
            }
            int chars = numChars(data, sStart, sLen);
            int spaces = target - chars;
            MemorySegment pd = p == null ? padLit : p.data();
            long pStart = p == null ? 0L : startOf(p, index(padCol, i));
            int pLen = p == null ? padBytes.length : lengthOf(p, index(padCol, i));
            if (spaces <= 0 || pLen == 0) {
                keep[i] = byteOffsetOfChar(data, sStart, sLen, target);
                outLen[i] = keep[i];
            } else {
                int padChars = numChars(pd, pStart, pLen);
                int count = spaces / padChars;
                keep[i] = sLen;
                copies[i] = count;
                remain[i] = byteOffsetOfChar(pd, pStart, pLen, spaces - padChars * count);
                long total = (long) sLen + (long) pLen * count + remain[i];
                if (total > Integer.MAX_VALUE) {
                    throw new IllegalStateException("padded string exceeds the maximum length");
                }
                outLen[i] = (int) total;
            }
        }
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = prefix(outLen, offsets, n);
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        for (int i = 0; i < n; i++) {
            if (outLen[i] == 0) {
                continue;
            }
            int idx = index(str, i);
            long at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
            MemorySegment pd = p == null ? padLit : p.data();
            long pStart = p == null ? 0L : startOf(p, index(padCol, i));
            int pLen = p == null ? padBytes.length : lengthOf(p, index(padCol, i));
            if (left) {
                fillRepeated(out, at, pd, pStart, pLen, copies[i]);
                at += (long) pLen * copies[i];
                MemorySegment.copy(pd, pStart, out, at, remain[i]);
                at += remain[i];
                MemorySegment.copy(data, startOf(s, idx), out, at, keep[i]);
            } else {
                MemorySegment.copy(data, startOf(s, idx), out, at, keep[i]);
                at += keep[i];
                fillRepeated(out, at, pd, pStart, pLen, copies[i]);
                at += (long) pLen * copies[i];
                MemorySegment.copy(pd, pStart, out, at, remain[i]);
            }
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    // ---------------------------------------------------------------- repeat / space

    /** {@code repeat(str, times)}: empty for a non-positive count. */
    public static SegmentVectorBuffers repeat(VectorBuffers str, VectorBuffers timesCol, int times,
            MemorySegment validity, Arena arena) {
        int n = str.length();
        VectorBuffers s = store(str);
        MemorySegment data = s.data();
        int[] outLen = new int[n];
        int[] count = new int[n];
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            int idx = index(str, i);
            int t = arg(timesCol, times, i);
            int sLen = lengthOf(s, idx);
            if (t <= 0 || sLen == 0) {
                continue;
            }
            long total = (long) sLen * t;
            if (total > MAX_OUTPUT_BYTES) {
                throw new IllegalStateException("repeat output of more than " + MAX_OUTPUT_BYTES + " bytes");
            }
            count[i] = t;
            outLen[i] = (int) total;
        }
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = prefix(outLen, offsets, n);
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        for (int i = 0; i < n; i++) {
            if (outLen[i] == 0) {
                continue;
            }
            int idx = index(str, i);
            fillRepeated(out, offsets.getAtIndex(VectorBuffers.LE_INT, i), data, startOf(s, idx),
                    lengthOf(s, idx), count[i]);
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    /** {@code space(n)}: {@code n} spaces, none for a negative count. */
    public static SegmentVectorBuffers space(int n, VectorBuffers countCol, int count,
            MemorySegment validity, Arena arena) {
        int[] outLen = new int[n];
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            int c = arg(countCol, count, i);
            outLen[i] = Math.max(c, 0);
        }
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = prefix(outLen, offsets, n);
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        out.fill((byte) ' ');
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    // ---------------------------------------------------------------- overlay

    /**
     * {@code overlay(input, replace, pos, len)}: Spark's {@code
     * Overlay.calculate} -- the first {@code pos - 1} code points, the
     * replacement, then the input from code point {@code pos + length} where
     * {@code length} is {@code len} when non-negative and the replacement's
     * code point count otherwise. {@code replCol} is a UTF8 lane or {@code
     * null} with {@code replBytes} the literal.
     */
    public static SegmentVectorBuffers overlay(
            VectorBuffers input,
            VectorBuffers replCol,
            byte[] replBytes,
            VectorBuffers posCol,
            int pos,
            VectorBuffers lenCol,
            int len,
            MemorySegment validity,
            Arena arena) {
        int n = input.length();
        VectorBuffers s = store(input);
        MemorySegment data = s.data();
        VectorBuffers r = replCol == null ? null : store(replCol);
        MemorySegment replLit = replCol == null ? MemorySegment.ofArray(replBytes) : null;
        long[] head = new long[n];
        long[] tail = new long[n];
        int[] outLen = new int[n];
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            int idx = index(input, i);
            int sStart = startOf(s, idx);
            int sLen = lengthOf(s, idx);
            MemorySegment rd = r == null ? replLit : r.data();
            long rStart = r == null ? 0L : startOf(r, index(replCol, i));
            int rLen = r == null ? replBytes.length : lengthOf(r, index(replCol, i));
            int p = arg(posCol, pos, i);
            int l = arg(lenCol, len, i);
            int length = l >= 0 ? l : numChars(rd, rStart, rLen);
            head[i] = substringRange(data, sStart, sLen, 1, p - 1);
            // Spark computes `pos + length` in int arithmetic.
            tail[i] = substringRange(data, sStart, sLen, p + length, Integer.MAX_VALUE);
            long total = (long) (int) head[i]
                    + rLen
                    + (int) tail[i];
            if (total > Integer.MAX_VALUE) {
                throw new IllegalStateException("overlay output exceeds the maximum length");
            }
            outLen[i] = (int) total;
        }
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = prefix(outLen, offsets, n);
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            int idx = index(input, i);
            int sStart = startOf(s, idx);
            MemorySegment rd = r == null ? replLit : r.data();
            long rStart = r == null ? 0L : startOf(r, index(replCol, i));
            int rLen = r == null ? replBytes.length : lengthOf(r, index(replCol, i));
            long at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
            int hLen = (int) head[i];
            MemorySegment.copy(data, sStart + (head[i] >>> 32), out, at, hLen);
            at += hLen;
            MemorySegment.copy(rd, rStart, out, at, rLen);
            at += rLen;
            MemorySegment.copy(data, sStart + (tail[i] >>> 32), out, at,
                    tail[i] & 0xFFFFFFFFL);
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }
}
