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

import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Consecutive INT64 values written into a column: {@code start, start + 1,
 * ...}. Behind {@code monotonically_increasing_id()}, whose value is a
 * per-partition prefix plus the row's number in the partition. With a selection
 * only the selected rows are numbered (Spark numbers the rows that reach the
 * expression, and a forwarded selection carries rows a filter already dropped);
 * unselected lanes get an arbitrary value and are masked by the caller. Returns
 * the next value, so the caller can carry the counter across batches.
 */
public final class SequenceKernels {
    static final VectorSpecies<Long> L = Species.L;

    private SequenceKernels() {}

    /**
     * Fills lanes {@code [0, n)} with {@code start ..}; returns {@code start +
     * n}.
     */
    public static long iota(MemorySegment out, int n, long start) {
        int lanes = L.length();
        long[] step = new long[lanes];
        for (int k = 0; k < lanes; k++) {
            step[k] = k;
        }
        LongVector offsets = LongVector.fromArray(L, step, 0);
        int i = 0;
        for (; i + lanes <= n; i += lanes) {
            offsets.add(start + i).intoMemorySegment(out, (long) i << 3, java.nio.ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_LONG, i, start + i);
        }
        return start + n;
    }

    /**
     * Numbers only the selected lanes of {@code [0, n)} in order; returns the
     * next value.
     */
    public static long iotaSelected(MemorySegment out, int n, MemorySegment selection,
            long start) {
        if (selection == null) {
            return iota(out, n, start);
        }
        long next = start;
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            long word = Bitmap.wordAt(selection, w, n);
            int base = w << 6;
            while (word != 0L) {
                int k = Long.numberOfTrailingZeros(word);
                out.setAtIndex(VectorBuffers.LE_LONG, base + k, next++);
                word &= word - 1;
            }
        }
        return next;
    }
}
