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
 * Scatters: value {@code i} of a column to slot {@code dest[i]} of the output
 * (#20). The shuffle writer's staged flush partitions a whole staging buffer at
 * a time. A gather through the partition order loads each value from wherever
 * it sits in a buffer of tens of megabytes, so every load misses the cache. A
 * scatter reads the column once, front to back, and its stores go to one
 * advancing cursor per partition -- a few hundred lines that stay in cache.
 * {@code dest} comes from {@link PartitionKernels#partitionDestinations}.
 */
public final class ScatterKernels {

    private ScatterKernels() {}

    /**
     * Scatters the first {@code n} values of a fixed-width or BOOL column. For
     * BOOL the output bits must be clear on entry.
     */
    public static void scatterFixed(VecType type, MemorySegment in, int n,
            int[] dest, MemorySegment out) {
        switch (type) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_INT, (long) dest[i] << 2, in.get(VectorBuffers.LE_INT, (long) i << 2));
                }
            }
            case INT64, FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_LONG, (long) dest[i] << 3, in.get(VectorBuffers.LE_LONG, (long) i << 3));
                }
            }
            case DECIMAL128 -> {
                for (int i = 0; i < n; i++) {
                    Decimal128.copy(in, i, out, dest[i]);
                }
            }
            case BOOL -> scatterBits(in, n, dest, out);
            default -> throw new IllegalArgumentException("not fixed width: " + type);
        }
    }

    /**
     * Scatters the first {@code n} bits of {@code in} (a validity bitmap or BOOL
     * data): bit {@code i} to bit {@code dest[i]} of {@code out}, whose bits must
     * be clear on entry. The input is read a word at a time; only the set bits
     * are stored.
     */
    public static void scatterBits(MemorySegment in, int n, int[] dest,
            MemorySegment out) {
        for (int base = 0; base < n; base += 64) {
            long word = Bitmap.wordAt(in, base >>> 6, n);
            while (word != 0L) {
                int j = Long.numberOfTrailingZeros(word);
                int d = dest[base + j];
                long at = (long) d >>> 3;
                out.set(ValueLayout.JAVA_BYTE, at, (byte) (out.get(ValueLayout.JAVA_BYTE, at) | (1 << (d & 7))));
                word &= word - 1;
            }
        }
    }
}
