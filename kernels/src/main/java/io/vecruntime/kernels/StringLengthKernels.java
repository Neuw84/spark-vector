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
 * The string functions computable from the offsets and a scan of the bytes
 * alone: {@code length} (code points), {@code octet_length} / {@code
 * bit_length} (bytes), {@code ascii} (the first code point) -- all INT32 out --
 * and {@code chr}, the one that writes a string.
 *
 * <p>Dictionary-encoded input is handled the way every string function should:
 * the per-string value is computed once per dictionary entry, then gathered
 * through the indices ({@link #perEntry}). Code points are counted with Spark's
 * {@code UTF8String.numChars} walk over its first-byte width table ({@link
 * StringSliceKernels#numBytesForFirstByte}), so malformed input counts exactly
 * as Spark counts it. Null rows (validity clear) get 0 in an INT32 output and
 * no bytes in a string output; the caller supplies the validity.
 */
public final class StringLengthKernels {
    private StringLengthKernels() {}

    /** What to compute per string. */
    public enum Measure {
        /** Code points, Spark's {@code numChars}. */
        CHARS,
        /** Bytes. */
        BYTES,
        /** Bytes times eight. */
        BITS,
        /** The first code point, 0 for the empty string. */
        ASCII
    }

    /** The replacement character Java's decoder yields for a malformed sequence. */
    static final int REPLACEMENT = 0xFFFD;

    private static int measure(Measure m, MemorySegment data, long start,
            int len) {
        switch (m) {
            case CHARS:
                return StringSliceKernels.numChars(data, start, len);
            case BYTES:
                return len;
            case BITS:
                return len * 8;
            default:
                return firstCodePoint(data, start, len);
        }
    }

    /**
     * Spark's {@code ascii}: the code point of {@code substring(0,
     * 1).toString()}, which decodes the first sequence as Java does -- a lone
     * continuation byte, a disallowed lead byte, a truncated or ill-formed
     * sequence gives U+FFFD.
     */
    static int firstCodePoint(MemorySegment data, long start, int len) {
        if (len == 0) {
            return 0;
        }
        int b0 = data.get(ValueLayout.JAVA_BYTE, start) & 0xFF;
        if (b0 < 0x80) {
            return b0;
        }
        int nb = StringSliceKernels.numBytesForFirstByte((byte) b0);
        if (nb == 1 || nb > len) {
            return REPLACEMENT;
        }
        int cp = b0 & (0xFF >> (nb + 1));
        for (int k = 1; k < nb; k++) {
            int b = data.get(ValueLayout.JAVA_BYTE, start + k) & 0xFF;
            if ((b & 0xC0) != 0x80) {
                return REPLACEMENT;
            }
            cp = (cp << 6) | (b & 0x3F);
        }
        // Overlong forms, surrogates and values past U+10FFFF are what the decoder rejects.
        if ((nb == 2 && cp < 0x80)
                || (nb == 3
                        && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)))
                || (nb == 4 && (cp < 0x10000 || cp > 0x10FFFF))) {
            return REPLACEMENT;
        }
        return cp;
    }

    /**
     * {@code m} over every string of {@code v} into an INT32 lane; dictionary
     * input once per entry.
     */
    public static SegmentVectorBuffers measure(Measure m, VectorBuffers v, MemorySegment validity,
            Arena arena) {
        int n = v.length();
        MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT32, n);
        VectorBuffers dict = v.dictionary();
        if (dict != null) {
            int[] perEntry = perEntry(m, dict);
            for (int i = 0; i < n; i++) {
                if (validity == null || Bitmap.isSet(validity, i)) {
                    out.setAtIndex(VectorBuffers.LE_INT, i, perEntry[v.getInt(i)]);
                }
            }
        } else {
            MemorySegment offsets = v.offsets();
            MemorySegment data = v.data();
            for (int i = 0; i < n; i++) {
                if (validity != null && !Bitmap.isSet(validity, i)) {
                    continue;
                }
                int start = offsets.getAtIndex(VectorBuffers.LE_INT, i);
                int len = offsets.getAtIndex(VectorBuffers.LE_INT, i + 1) - start;
                out.setAtIndex(VectorBuffers.LE_INT, i, measure(m, data, start, len));
            }
        }
        return SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, out);
    }

    /** {@code m} for each entry of a plain UTF8 dictionary. */
    static int[] perEntry(Measure m, VectorBuffers dict) {
        int k = dict.length();
        int[] values = new int[k];
        MemorySegment offsets = dict.offsets();
        MemorySegment data = dict.data();
        for (int e = 0; e < k; e++) {
            int start = offsets.getAtIndex(VectorBuffers.LE_INT, e);
            values[e] = measure(m, data, start, offsets.getAtIndex(VectorBuffers.LE_INT, e + 1) - start);
        }
        return values;
    }

    /**
     * Spark's {@code chr(n)} over an INT32 or INT64 lane: a negative {@code n}
     * gives {@code ''}, the low byte {@code n & 0xFF} is the character -- 0 is
     * the NUL character, 1..127 one byte, 128..255 the two-byte UTF-8 encoding
     * of that Latin-1 code point.
     */
    public static SegmentVectorBuffers chr(VectorBuffers v, MemorySegment validity, Arena arena) {
        int n = v.length();
        boolean wide = v.type() == VecType.INT64;
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        int total = 0;
        offsets.setAtIndex(VectorBuffers.LE_INT, 0, 0);
        for (int i = 0; i < n; i++) {
            if (validity == null || Bitmap.isSet(validity, i)) {
                long x = wide ? v.getLong(i) : v.getInt(i);
                total += x < 0
                        ? 0
                        : ((x & 0xFF) < 0x80 ? 1 : 2);
            }
            offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, total);
        }
        MemorySegment out = ArrowLayout.allocateBytes(arena, total);
        for (int i = 0; i < n; i++) {
            int at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
            int len = offsets.getAtIndex(VectorBuffers.LE_INT, i + 1) - at;
            if (len == 0) {
                continue;
            }
            int c = (int) ((wide ? v.getLong(i) : v.getInt(i)) & 0xFF);
            if (len == 1) {
                out.set(ValueLayout.JAVA_BYTE, at, (byte) c);
            } else {
                out.set(ValueLayout.JAVA_BYTE, at, (byte) (0xC0 | (c >> 6)));
                out.set(ValueLayout.JAVA_BYTE, at + 1, (byte) (0x80 | (c & 0x3F)));
            }
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }
}
