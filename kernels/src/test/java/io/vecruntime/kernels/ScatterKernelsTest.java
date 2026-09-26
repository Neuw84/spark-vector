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
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * #20: the scatter the staged shuffle flush partitions by, against the gather
 * it replaces.
 */
class ScatterKernelsTest {

    private static int[] ids(Random r, int n, int partitions) {
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            // Skewed, with empty partitions: half the rows in partition 3.
            ids[i] = r.nextBoolean() ? 3 : r.nextInt(partitions / 2) * 2;
        }
        return ids;
    }

    @Test
    void destinationsAreTheInverseOfTheOrder() {
        Random r = new Random(20);
        for (int n : new int[] {0, 1, 63, 64, 65, 1000,
                8193}) {
            int partitions = 300;
            int[] ids = ids(r, n, partitions);
            int[] startsO = new int[partitions + 1];
            int[] startsD = new int[partitions + 1];
            int[] order = new int[n];
            int[] dest = new int[n];
            PartitionKernels.partitionOrder(ids, n, partitions, startsO, order);
            PartitionKernels.partitionDestinations(ids, n, partitions, startsD, dest);
            assertArrayEquals(startsO, startsD, "starts n=" + n);
            for (int k = 0; k < n; k++) {
                assertEquals(k, dest[order[k]], "n=" + n + " slot " + k);
            }
        }
    }

    @Test
    void scatterEqualsTheGatherThroughTheOrder() {
        Random r = new Random(21);
        int n = 5000;
        int partitions = 300;
        int[] ids = ids(r, n, partitions);
        int[] starts = new int[partitions + 1];
        int[] order = new int[n];
        int[] dest = new int[n];
        PartitionKernels.partitionOrder(ids, n, partitions, starts, order);
        PartitionKernels.partitionDestinations(ids, n, partitions, starts, dest);
        try (Arena arena = Arena.ofConfined()) {
            for (VecType type : new VecType[] {VecType.INT32, VecType.INT64, VecType.FLOAT64, VecType.DECIMAL128}) {
                int width = switch (type) {
                    case INT32 -> 4;
                    case DECIMAL128 -> 16;
                    default -> 8;
                };
                MemorySegment in = arena.allocate((long) n * width);
                for (long b = 0; b < in.byteSize(); b++) {
                    in.set(java.lang.foreign.ValueLayout.JAVA_BYTE,
                            b, (byte) r.nextInt());
                }
                MemorySegment gathered = arena.allocate((long) n * width);
                MemorySegment scattered = arena.allocate((long) n * width);
                GatherKernels.gatherFixedPlain(type, in, order, 0, n, gathered);
                ScatterKernels.scatterFixed(type, in, n, dest, scattered);
                assertEquals(-1L, gathered.mismatch(scattered), type.toString());
            }
            MemorySegment bits = Bitmap.allocate(arena, n);
            for (int i = 0; i < n; i++) {
                Bitmap.setTo(bits, i, r.nextInt(3) != 0);
            }
            MemorySegment gathered = Bitmap.allocate(arena, n);
            MemorySegment scattered = Bitmap.allocate(arena, n);
            GatherKernels.gatherFixedPlain(VecType.BOOL, bits, order, 0, n, gathered);
            ScatterKernels.scatterFixed(VecType.BOOL, bits, n, dest, scattered);
            for (int i = 0; i < n; i++) {
                assertEquals(Bitmap.isSet(gathered, i), Bitmap.isSet(scattered, i), "bit " + i);
            }
        }
    }

    @Test
    void chunkedScatterEqualsOneCall() {
        Random r = new Random(23);
        int partitions = 300;
        // over two chunks and a partial one, so every chunk boundary is crossed
        int n = 2 * ScatterKernels.CHUNK_ROWS + 1234;
        int[] ids = ids(r, n, partitions);
        int[] starts = new int[partitions + 1];
        int[] dest = new int[n];
        PartitionKernels.partitionDestinations(ids, n, partitions, starts, dest);
        try (Arena arena = Arena.ofConfined()) {
            for (VecType type : new VecType[] {VecType.INT32, VecType.INT64, VecType.DECIMAL128}) {
                int width = type == VecType.INT32
                        ? 4
                        : type == VecType.INT64 ? 8 : 16;
                MemorySegment in = arena.allocate((long) n * width);
                for (long b = 0; b < in.byteSize(); b++) {
                    in.set(java.lang.foreign.ValueLayout.JAVA_BYTE,
                            b, (byte) r.nextInt());
                }
                MemorySegment whole = arena.allocate((long) n * width);
                MemorySegment chunked = arena.allocate((long) n * width);
                ScatterKernels.scatterFixed(type, in, n, dest, whole);
                ScatterKernels.scatterFixedChunked(type, in, n, dest, chunked);
                assertEquals(-1L, whole.mismatch(chunked), type.toString());
            }
            MemorySegment bits = Bitmap.allocate(arena, n);
            for (int i = 0; i < n; i++) {
                Bitmap.setTo(bits, i, r.nextInt(3) != 0);
            }
            MemorySegment whole = Bitmap.allocate(arena, n);
            MemorySegment chunked = Bitmap.allocate(arena, n);
            ScatterKernels.scatterBits(bits, n, dest, whole);
            ScatterKernels.scatterBitsChunked(bits, n, dest, chunked);
            assertEquals(-1L, whole.mismatch(chunked), "bits");
        }
    }

    @Test
    void idRunsCountsChangesOfPartition() {
        assertEquals(0, ScatterKernels.idRuns(new int[0], 0));
        assertEquals(1, ScatterKernels.idRuns(new int[] {5}, 1));
        assertEquals(1, ScatterKernels.idRuns(new int[] {2, 2, 2, 2}, 4));
        assertEquals(4, ScatterKernels.idRuns(new int[] {1, 2, 1, 2}, 4));
        // only the first n ids count: 7 7 | 3 3 3
        assertEquals(
                2,
                ScatterKernels.idRuns(
                        new int[] {7, 7, 3, 3, 3, 7,
                                9},
                        5));
    }

    @Test
    void copyBitsFromAnyOffset() {
        Random r = new Random(22);
        try (Arena arena = Arena.ofConfined()) {
            int n = 1000;
            MemorySegment src = Bitmap.allocate(arena, n);
            for (int i = 0; i < n; i++) {
                Bitmap.setTo(src, i, r.nextBoolean());
            }
            for (int[] c : new int[][] {
                {0, 0, 1000},
                {8, 16, 300},
                {3, 0, 200},
                {61, 7, 130},
                {64, 1, 64},
                {999, 5, 1},
                {130, 130, 0}
            }) {
                int srcFrom = c[0], dstFrom = c[1], count = c[2];
                MemorySegment dst = Bitmap.allocate(arena, dstFrom + count + 70);
                Bitmap.fill(dst, dstFrom + count + 70, true);
                Bitmap.copyBitsFrom(src, srcFrom, dst, dstFrom, count);
                for (int i = 0; i < count; i++) {
                    assertEquals(Bitmap.isSet(src, srcFrom + i), Bitmap.isSet(dst, dstFrom + i), "offset case " + srcFrom + "/" + dstFrom + " bit " + i);
                }
                for (int i = 0; i < dstFrom; i++) {
                    assertEquals(true, Bitmap.isSet(dst, i), "bits before the copy untouched");
                }
                for (int i = dstFrom + count; i < dstFrom + count + 70; i++) {
                    assertEquals(true, Bitmap.isSet(dst, i), "bits after the copy untouched");
                }
            }
        }
    }
}
