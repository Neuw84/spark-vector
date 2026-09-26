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

/**
 * {@code concat}, {@code concat_ws} and {@code elt}: string functions over
 * <em>several</em> inputs. Each row's output length is a sum across the inputs,
 * so the offsets are the prefix sum of those sums and the second pass is a
 * per-input loop over byte ranges -- nothing per character. An input is a UTF8
 * lane (dictionary-encoded read through the dictionary) or a literal's bytes
 * ({@link Part}). Semantics are Spark's {@code UTF8String.concat} / {@code
 * concatWs} and {@code Elt}: {@code concat} is null when any input is null;
 * {@code concat_ws} is null only for a null separator, skips null inputs and
 * separates the live ones; {@code elt} picks input {@code index} (1-based) and
 * is null for a null index, an out-of-range index or a null pick. The per-batch
 * output is capped at {@link StringSliceKernels#MAX_OUTPUT_BYTES}.
 */
public final class StringConcatKernels {
    private StringConcatKernels() {}

    /** One input: a lane or a literal. */
    public static final class Part {
        final VectorBuffers lane;
        final VectorBuffers store;
        final MemorySegment literal;

        private Part(VectorBuffers lane, byte[] literal) {
            this.lane = lane;
            this.store = lane == null
                    ? null
                    : (lane.dictionary() != null ? lane.dictionary() : lane);
            this.literal = literal == null ? null : MemorySegment.ofArray(literal);
        }

        public static Part of(VectorBuffers lane) {
            return new Part(lane, null);
        }

        public static Part literal(byte[] bytes) {
            return new Part(null, bytes);
        }

        boolean live(int i) {
            return lane == null || lane.validity() == null || Bitmap.isSet(lane.validity(), i);
        }

        int length(int i) {
            if (lane == null) {
                return (int) literal.byteSize();
            }
            return StringSliceKernels.lengthOf(store, StringSliceKernels.index(lane, i));
        }

        /** The segment holding row {@code i}'s bytes. */
        MemorySegment seg(int i) {
            return lane == null ? literal : store.data();
        }

        /** The offset of row {@code i}'s first byte within {@link #seg}. */
        long at(int i) {
            return lane == null ? 0L : StringSliceKernels.startOf(store, StringSliceKernels.index(lane, i));
        }

        /**
         * Copies row {@code i}'s bytes to {@code out} at {@code at}; returns
         * the bytes copied.
         */
        int copy(int i, MemorySegment out, long at) {
            if (lane == null) {
                MemorySegment.copy(literal, 0, out, at, literal.byteSize());
                return (int) literal.byteSize();
            }
            int idx = StringSliceKernels.index(lane, i);
            int len = StringSliceKernels.lengthOf(store, idx);
            MemorySegment.copy(store.data(), StringSliceKernels.startOf(store, idx), out, at,
                    len);
            return len;
        }
    }

    private static boolean live(MemorySegment validity, int i) {
        return validity == null || Bitmap.isSet(validity, i);
    }

    private static MemorySegment offsetsOf(long[] lengths, int n, Arena arena) {
        MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
        long total = 0;
        for (int i = 0; i < n; i++) {
            total += lengths[i];
            if (total > StringSliceKernels.MAX_OUTPUT_BYTES) {
                throw new IllegalStateException("string output of more than " + StringSliceKernels.MAX_OUTPUT_BYTES + " bytes in one batch");
            }
            offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, (int) total);
        }
        return offsets;
    }

    /**
     * A UTF8 lane from per-row byte arrays -- the escape hatch for a per-row
     * string result computed outside the kernels (a digest, a formatter). A
     * null entry is a null row; {@code validity}, when given, is ANDed with the
     * entries' presence.
     */
    public static SegmentVectorBuffers fromRows(byte[][] rows, MemorySegment validity, Arena arena) {
        int n = rows.length;
        MemorySegment out = ArrowLayout.allocateBitmap(arena, n);
        long[] lengths = new long[n];
        for (int i = 0; i < n; i++) {
            if (rows[i] == null || !live(validity, i)) {
                continue;
            }
            Bitmap.set(out, i);
            lengths[i] = rows[i].length;
        }
        MemorySegment offsets = offsetsOf(lengths, n, arena);
        MemorySegment data = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
        for (int i = 0; i < n; i++) {
            if (lengths[i] == 0) {
                continue;
            }
            MemorySegment.copy(MemorySegment.ofArray(rows[i]), 0, data, offsets.getAtIndex(VectorBuffers.LE_INT, i),
                    rows[i].length);
        }
        return SegmentVectorBuffers.utf8(n, out, offsets, data);
    }

    /**
     * {@code concat(parts...)}: {@code validity} is the AND of the parts'
     * validities (the caller's).
     */
    public static SegmentVectorBuffers concat(Part[] parts, int n, MemorySegment validity,
            Arena arena) {
        long[] lengths = new long[n];
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            long sum = 0;
            for (Part p : parts) {
                sum += p.length(i);
            }
            lengths[i] = sum;
        }
        MemorySegment offsets = offsetsOf(lengths, n, arena);
        MemorySegment out = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
        for (int i = 0; i < n; i++) {
            if (lengths[i] == 0) {
                continue;
            }
            long at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
            for (Part p : parts) {
                at += p.copy(i, out, at);
            }
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    /**
     * {@code concat_ws(sep, parts...)}: the output validity is the separator's;
     * a null part is skipped and the separator sits only between the live
     * parts.
     */
    public static SegmentVectorBuffers concatWs(Part sep, Part[] parts, int n,
            Arena arena) {
        MemorySegment validity = sep.lane == null ? null : sep.lane.validity();
        long[] lengths = new long[n];
        for (int i = 0; i < n; i++) {
            if (!live(validity, i)) {
                continue;
            }
            long sum = 0;
            int liveParts = 0;
            for (Part p : parts) {
                if (!p.live(i)) {
                    continue;
                }
                sum += p.length(i);
                liveParts++;
            }
            if (liveParts > 1) {
                sum += (long) sep.length(i) * (liveParts - 1);
            }
            lengths[i] = sum;
        }
        MemorySegment offsets = offsetsOf(lengths, n, arena);
        MemorySegment out = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
        for (int i = 0; i < n; i++) {
            if (lengths[i] == 0) {
                continue;
            }
            long at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
            boolean first = true;
            for (Part p : parts) {
                if (!p.live(i)) {
                    continue;
                }
                if (!first) {
                    at += sep.copy(i, out, at);
                }
                at += p.copy(i, out, at);
                first = false;
            }
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }

    /**
     * The first row of {@code active} whose live index is outside {@code
     * 1..count}, or -1.
     */
    public static int firstInvalidIndex(VectorBuffers index, int count, MemorySegment active,
            int n) {
        for (int i = 0; i < n; i++) {
            if (active != null && !Bitmap.isSet(active, i)) {
                continue;
            }
            if (index.validity() != null && !Bitmap.isSet(index.validity(), i)) {
                continue;
            }
            int k = index.getInt(i);
            if (k <= 0 || k > count) {
                return i;
            }
        }
        return -1;
    }

    /**
     * {@code elt(index, parts...)} with its own validity: null index,
     * out-of-range index or null pick.
     */
    public static SegmentVectorBuffers elt(VectorBuffers index, Part[] parts, int n,
            Arena arena) {
        MemorySegment validity = ArrowLayout.allocateBitmap(arena, n);
        long[] lengths = new long[n];
        int[] pick = new int[n];
        for (int i = 0; i < n; i++) {
            if (index.validity() != null && !Bitmap.isSet(index.validity(), i)) {
                continue;
            }
            int k = index.getInt(i);
            if (k <= 0 || k > parts.length) {
                continue;
            }
            Part p = parts[k - 1];
            if (!p.live(i)) {
                continue;
            }
            Bitmap.setTo(validity, i, true);
            pick[i] = k - 1;
            lengths[i] = p.length(i);
        }
        MemorySegment offsets = offsetsOf(lengths, n, arena);
        MemorySegment out = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
        for (int i = 0; i < n; i++) {
            if (lengths[i] == 0) {
                continue;
            }
            parts[pick[i]].copy(i, out, offsets.getAtIndex(VectorBuffers.LE_INT, i));
        }
        return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }
}
