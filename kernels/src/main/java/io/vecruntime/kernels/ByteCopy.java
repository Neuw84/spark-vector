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
import java.lang.foreign.ValueLayout;

/**
 * Copies of the short byte runs UTF8 kernels move one string at a time. {@link
 * MemorySegment#copy} costs two session checks, two bounds checks and a call
 * into the vectorized copy stub whatever the length; on the 10 to 25-byte
 * strings TPC-H joins and filters carry (nation and region names, ship modes,
 * market segments) that set-up is most of the work, and gathering the 320k
 * output rows of Q9's nation join through it was 13% of the query's samples.
 * Runs of up to {@link #SHORT} bytes are moved as two overlapping unaligned
 * longs (or a handful of bytes below eight), which the JIT turns into plain
 * loads and stores; longer runs still take the bulk copy.
 */
public final class ByteCopy {
    /** Runs of at most this many bytes take the long-pair path. */
    public static final int SHORT = 16;

    private ByteCopy() {}

    /** {@code dst[dstPos, dstPos + len) = src[srcPos, srcPos + len)}. */
    public static void copy(MemorySegment src, long srcPos, MemorySegment dst,
                            long dstPos, int len) {
        if (len > SHORT) {
            MemorySegment.copy(src, ValueLayout.JAVA_BYTE, srcPos, dst, ValueLayout.JAVA_BYTE, dstPos,
                    len);
        } else if (len >= 8) {
            // Two 8-byte moves covering [0, 8) and [len - 8, len): they overlap for len < 16, which is fine.
            long head = src.get(VectorBuffers.LE_LONG, srcPos);
            long tail = src.get(VectorBuffers.LE_LONG, srcPos + len - 8);
            dst.set(VectorBuffers.LE_LONG, dstPos, head);
            dst.set(VectorBuffers.LE_LONG, dstPos + len - 8, tail);
        } else if (len >= 4) {
            int head = src.get(VectorBuffers.LE_INT, srcPos);
            int tail = src.get(VectorBuffers.LE_INT, srcPos + len - 4);
            dst.set(VectorBuffers.LE_INT, dstPos, head);
            dst.set(VectorBuffers.LE_INT, dstPos + len - 4, tail);
        } else {
            for (int k = 0; k < len; k++) {
                dst.set(ValueLayout.JAVA_BYTE, dstPos + k, src.get(ValueLayout.JAVA_BYTE, srcPos + k));
            }
        }
    }
}
