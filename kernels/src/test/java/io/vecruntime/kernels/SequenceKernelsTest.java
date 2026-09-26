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
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link SequenceKernels}: consecutive values over all or selected lanes,
 * carried across batches.
 */
class SequenceKernelsTest {

    @Test
    void fillsFullLaneBlocksAndTails() {
        try (Arena arena = Arena.ofConfined()) {
            for (int n : new int[] {0, 1, 7, 8, 63, 64,
                    65, 1000, 4096}) {
                MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT64, Math.max(n, 1));
                long start = (7L << 33) + 12345;
                long next = SequenceKernels.iota(out, n, start);
                assertEquals(start + n, next, "next after " + n);
                for (int i = 0; i < n; i++) {
                    assertEquals(start + i, out.getAtIndex(VectorBuffers.LE_LONG, i), "lane " + i + " of " + n);
                }
            }
        }
    }

    @Test
    void numbersOnlySelectedLanesAndCarriesTheCounter() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(3);
            int n = 777;
            MemorySegment selection = ArrowLayout.allocateBitmap(arena, n);
            Bitmap.fill(selection, n, false);
            int selected = 0;
            for (int i = 0; i < n; i++) {
                if (rnd.nextInt(3) == 0) {
                    Bitmap.set(selection, i);
                    selected++;
                }
            }
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT64, n);
            long start = 2L << 33;
            long next = SequenceKernels.iotaSelected(out, n, selection, start);
            assertEquals(start + selected, next);
            long expected = start;
            for (int i = 0; i < n; i++) {
                if (Bitmap.isSet(selection, i)) {
                    assertEquals(expected++, out.getAtIndex(VectorBuffers.LE_LONG, i), "selected lane " + i);
                }
            }
            // A second batch continues where the first stopped, with and without a selection.
            long after = SequenceKernels.iota(out, 10, next);
            assertEquals(next + 10, after);
            assertEquals(next, out.getAtIndex(VectorBuffers.LE_LONG, 0));
            assertEquals(next + 9, out.getAtIndex(VectorBuffers.LE_LONG, 9));
            // No selection given: every lane is numbered.
            assertEquals(start + n, SequenceKernels.iotaSelected(out, n, null, start));
        }
    }
}
