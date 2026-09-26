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
import java.lang.foreign.ValueLayout;

/**
 * Case mapping and trimming.
 *
 * <p>Case mapping runs an ASCII fast path -- a byte-range test and a {@code
 * 0x20} flip, so the output has the input's length -- and flags the rows it
 * cannot decide ({@link #slowRows}): any byte at or above {@code 0x80}, and for
 * {@code initcap} under ICU word titlecasing anything but letters and spaces
 * (ICU moves the capital past leading digits and breaks words on punctuation).
 * The caller computes those rows with Spark's own implementation and hands them
 * back as {@code overrides}, which {@link #caseMap} copies verbatim; every
 * other row is mapped in place. Dictionary-encoded input is decided per
 * dictionary entry.
 *
 * <p>Trimming is offset arithmetic: scan in from each end while the code point
 * is in the trim set (the single space by default, as Spark's {@code trim()}
 * trims only ASCII 32); the output borrows nothing and is written once from the
 * surviving ranges.
 */
public final class StringCaseKernels {
    private StringCaseKernels() {}

    public enum Kind {
        UPPER,
        LOWER,
        INITCAP
    }

    public enum Side {
        LEFT,
        RIGHT,
        BOTH
    }

    private static boolean isUpper(int b) {
        return b >= 'A' && b <= 'Z';
    }

    private static boolean isLower(int b) {
        return b >= 'a' && b <= 'z';
    }

    /**
     * Whether the ASCII path cannot decide the string at {@code start}/{@code
     * len}.
     */
    static boolean slow(MemorySegment data, long start, int len,
                        Kind kind, boolean strict) {
        for (int i = 0; i < len; i++) {
            int b = data.get(ValueLayout.JAVA_BYTE, start + i) & 0xFF;
            if (b >= 0x80) {
                return true;
            }
            if (strict
                    && kind == Kind.INITCAP
                    && b != ' '
                    && !isUpper(b)
                    && !isLower(b)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Flags the rows the ASCII path cannot decide into {@code out} (a zeroed
     * bitmap of {@code n}); returns how many. Null rows are never flagged.
     */
    public static int slowRows(VectorBuffers v, Kind kind, boolean strict,
            MemorySegment validity, MemorySegment out) {
        int n = v.length();
        int count = 0;
        VectorBuffers dict = v.dictionary();
        if (dict != null) {
            int k = dict.length();
            boolean[] perEntry = new boolean[k];
            for (int e = 0; e < k; e++) {
                perEntry[e] = slow(dict.data(), StringSliceKernels.startOf(dict, e), StringSliceKernels.lengthOf(dict, e),
                        kind, strict);
            }
            for (int i = 0; i < n; i++) {
                if (validity != null && !Bitmap.isSet(validity, i)) {
                    continue;
                }
                if (perEntry[v.getInt(i)]) {
                    Bitmap.set(out, i);
                    count++;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (validity != null && !Bitmap.isSet(validity, i)) {
                    continue;
                }
                if (slow(v.data(), StringSliceKernels.startOf(v, i), StringSliceKernels.lengthOf(v, i),
                        kind, strict)) {
                    Bitmap.set(out, i);
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * The ASCII mapping of one byte given the previous byte (for {@code
     * initcap}'s word starts).
     */
    private static byte map(Kind kind, int b, int prev) {
        switch (kind) {
            case UPPER:
                return (byte) (isLower(b) ? b - 0x20 : b);
            case LOWER:
                return (byte) (isUpper(b) ? b + 0x20 : b);
            default:
                boolean wordStart = prev < 0 || prev == ' ';
                if (wordStart) {
                    return (byte) (isLower(b) ? b - 0x20 : b);
                }
                return (byte) (isUpper(b) ? b + 0x20 : b);
        }
    }

    /**
     * Maps every live row: rows with a non-null entry in {@code overrides} (may
     * be {@code null} when there are none) are copied from it; the rest through
     * the ASCII mapping, same length.
     */
    public static SegmentVectorBuffers caseMap(VectorBuffers v, Kind kind, byte[][] overrides,
            MemorySegment validity, Arena arena) {
        int n = v.length();
        VectorBuffers store = StringSliceKernels.store(v);
        MemorySegment data = store.data();
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = 0;
        for (int i = 0; i < n; i++) {
            if (validity == null || Bitmap.isSet(validity, i)) {
                byte[] o = overrides == null ? null : overrides[i];
                total += o != null ? o.length : StringSliceKernels.lengthOf(store, StringSliceKernels.index(v, i));
                if (total > StringSliceKernels.MAX_OUTPUT_BYTES) {
                    throw new IllegalStateException("string output of more than " + StringSliceKernels.MAX_OUTPUT_BYTES + " bytes in one batch");
                }
            }
            offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, (int) total);
        }
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        for (int i = 0; i < n; i++) {
            int at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
            int len = offsets.getAtIndex(VectorBuffers.LE_INT, i + 1) - at;
            if (len == 0) {
                continue;
            }
            byte[] o = overrides == null ? null : overrides[i];
            if (o != null) {
                MemorySegment.copy(MemorySegment.ofArray(o), 0, out, at, len);
                continue;
            }
            int idx = StringSliceKernels.index(v, i);
            long src = StringSliceKernels.startOf(store, idx);
            int prev = -1;
            for (int k = 0; k < len; k++) {
                int b = data.get(ValueLayout.JAVA_BYTE, src + k) & 0xFF;
                out.set(ValueLayout.JAVA_BYTE, at + k, map(kind, b, prev));
                prev = b;
            }
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    /**
     * Decodes the code point at {@code at} (Spark's width table; malformed
     * bytes decode to themselves).
     */
    private static int codePointAt(MemorySegment data, long at, int nb) {
        int b0 = data.get(ValueLayout.JAVA_BYTE, at) & 0xFF;
        if (nb == 1) {
            return b0;
        }
        int cp = b0 & (0xFF >> (nb + 1));
        for (int k = 1; k < nb; k++) {
            cp = (cp << 6) | (data.get(ValueLayout.JAVA_BYTE, at + k) & 0x3F);
        }
        return cp;
    }

    private static boolean inSet(int cp, int[] set) {
        for (int s : set) {
            if (s == cp) {
                return true;
            }
        }
        return false;
    }

    /**
     * The surviving byte range of the string at {@code start}/{@code len} after
     * trimming {@code set} (code points; {@code null} = the space) from {@code
     * side}: {@code (offset << 32) | length}.
     */
    static long trimRange(MemorySegment data, long start, int len,
                          Side side, int[] set) {
        int from = 0;
        int to = len;
        if (side != Side.RIGHT) {
            while (from < to) {
                int nb = StringSliceKernels.numBytesForFirstByte(data.get(ValueLayout.JAVA_BYTE, start + from));
                if (from + nb > to) {
                    break;
                }
                int cp = codePointAt(data, start + from, nb);
                if (set == null ? cp != ' ' : !inSet(cp, set)) {
                    break;
                }
                from += nb;
            }
        }
        if (side != Side.LEFT) {
            while (to > from) {
                // Step back to the start of the last code point: skip continuation bytes.
                int begin = to - 1;
                while (begin > from && (data.get(ValueLayout.JAVA_BYTE, start + begin) & 0xC0) == 0x80) {
                    begin--;
                }
                int nb = to - begin;
                int cp = nb <= 4 ? codePointAt(data, start + begin, nb) : -1;
                if (set == null ? cp != ' ' : !inSet(cp, set)) {
                    break;
                }
                to = begin;
            }
        }
        return (((long) from) << 32) | (to - from);
    }

    /**
     * {@code trim} / {@code ltrim} / {@code rtrim} of every live row; {@code
     * set} null trims spaces.
     */
    public static SegmentVectorBuffers trim(VectorBuffers v, Side side, int[] set,
            MemorySegment validity, Arena arena) {
        int n = v.length();
        VectorBuffers store = StringSliceKernels.store(v);
        MemorySegment data = store.data();
        long[] ranges = new long[n];
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = 0;
        for (int i = 0; i < n; i++) {
            if (validity == null || Bitmap.isSet(validity, i)) {
                int idx = StringSliceKernels.index(v, i);
                long start = StringSliceKernels.startOf(store, idx);
                long r = trimRange(data, start, StringSliceKernels.lengthOf(store, idx), side, set);
                ranges[i] = ((start + (r >>> 32)) << 32) | (int) r;
                total += (int) r;
            }
            offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, (int) total);
        }
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        for (int i = 0; i < n; i++) {
            int len = (int) ranges[i];
            if (len == 0) {
                continue;
            }
            MemorySegment.copy(data, ranges[i] >>> 32, out, offsets.getAtIndex(VectorBuffers.LE_INT, i),
                    len);
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }
}
