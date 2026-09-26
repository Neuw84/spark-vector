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
     * Compiles the scatter loops before the first real flush (#20). A flush
     * calls each kernel once per column over the whole staging (~461k rows on
     * the CDC MERGE), so on a cold JVM those first calls ran the loops before
     * the JIT had compiled them: 3-5 s per flush against ~250 ms warm, for the
     * first flush of every task on every executor. Here every loop runs on
     * small native segments (global scope, like the Arrow buffers the writer
     * passes) enough times to be compiled. Runs once per JVM, on first use;
     * returns the milliseconds it took.
     */
    public static long warmUp() {
        return WarmUp.MILLIS;
    }

    private static final class WarmUp {
        static final long MILLIS = run();

        private static long run() {
            long start = System.nanoTime();
            int n = 64;
            java.lang.foreign.Arena arena = java.lang.foreign.Arena.global();
            MemorySegment in = arena.allocate((long) n * 16, 16);
            MemorySegment out = arena.allocate((long) n * 16, 16);
            int[] dest = new int[n];
            for (int i = 0; i < n; i++) {
                dest[i] = (i * 37) % n; // a permutation: 37 and 64 are coprime
            }
            for (int round = 0; round < 20_000; round++) {
                scatterFixed(VecType.INT32, in, 0, n, dest, out);
                scatterFixed(VecType.INT64, in, 0, n, dest, out);
                scatterFixed(VecType.DECIMAL128, in, 0, n, dest, out);
                out.asSlice(0, n / 8).fill((byte) 0);
                scatterBits(in, 0, n, dest, out);
            }
            return (System.nanoTime() - start) / 1_000_000L;
        }
    }

    /**
     * Rows per kernel call when a whole staging is scattered (#20). A flush
     * scatters ~461k rows per column; one call per chunk bounds what any call
     * run before the JIT has compiled it (or after a deoptimization) can cost,
     * and gives the kernels enough calls to be compiled normally. A multiple of
     * 64, so the bit scatter reads whole words.
     */
    public static final int CHUNK_ROWS = 1 << 16;

    /**
     * {@link #scatterFixed(VecType, MemorySegment, int, int, int[],
     * MemorySegment)} over the first {@code n} rows, in calls of at most {@link
     * #CHUNK_ROWS} rows.
     */
    public static void scatterFixedChunked(VecType type, MemorySegment in, int n,
            int[] dest, MemorySegment out) {
        for (int from = 0; from < n; from += CHUNK_ROWS) {
            scatterFixed(type, in, from, Math.min(n, from + CHUNK_ROWS), dest,
                    out);
        }
    }

    /**
     * {@link #scatterBits(MemorySegment, int, int, int[], MemorySegment)} over
     * the first {@code n} rows, in calls of at most {@link #CHUNK_ROWS} rows.
     */
    public static void scatterBitsChunked(MemorySegment in, int n, int[] dest,
            MemorySegment out) {
        for (int from = 0; from < n; from += CHUNK_ROWS) {
            scatterBits(in, from, Math.min(n, from + CHUNK_ROWS), dest, out);
        }
    }

    /**
     * The number of runs of equal partition ids in the first {@code n} ids:
     * {@code n} when every row changes partition, far fewer when the rows
     * arrive grouped (the reduce side of a join re-partitioning its input),
     * where a gather through the order reads nearly sequentially and a scatter
     * only adds a pass (#20).
     */
    public static int idRuns(int[] ids, int n) {
        if (n == 0) {
            return 0;
        }
        int runs = 1;
        for (int i = 1; i < n; i++) {
            runs += ids[i] != ids[i - 1] ? 1 : 0;
        }
        return runs;
    }

    /**
     * Scatters the first {@code n} values of a fixed-width or BOOL column. For
     * BOOL the output bits must be clear on entry.
     */
    public static void scatterFixed(VecType type, MemorySegment in, int n,
            int[] dest, MemorySegment out) {
        scatterFixed(type, in, 0, n, dest, out);
    }

    /**
     * Scatters values {@code [from, to)} of a fixed-width or BOOL column: value
     * {@code i} to slot {@code dest[i]}. For BOOL the output bits must be clear
     * on entry and {@code from} a multiple of 64.
     */
    public static void scatterFixed(VecType type, MemorySegment in, int from,
            int to, int[] dest, MemorySegment out) {
        switch (type) {
            case INT32 -> {
                for (int i = from; i < to; i++) {
                    out.set(VectorBuffers.LE_INT, (long) dest[i] << 2, in.get(VectorBuffers.LE_INT, (long) i << 2));
                }
            }
            case INT64, FLOAT64 -> {
                for (int i = from; i < to; i++) {
                    out.set(VectorBuffers.LE_LONG, (long) dest[i] << 3, in.get(VectorBuffers.LE_LONG, (long) i << 3));
                }
            }
            case DECIMAL128 -> {
                for (int i = from; i < to; i++) {
                    Decimal128.copy(in, i, out, dest[i]);
                }
            }
            case BOOL -> scatterBits(in, from, to, dest, out);
            default -> throw new IllegalArgumentException("not fixed width: " + type);
        }
    }

    /** Scatters the first {@code n} bits of {@code in}; see
     * {@link #scatterBits(MemorySegment, int, int, int[], MemorySegment)}. */
    public static void scatterBits(MemorySegment in, int n, int[] dest,
            MemorySegment out) {
        scatterBits(in, 0, n, dest, out);
    }

    /**
     * Scatters bits {@code [from, to)} of {@code in} (a validity bitmap or BOOL
     * data): bit {@code i} to bit {@code dest[i]} of {@code out}, whose bits
     * must be clear on entry. {@code from} must be a multiple of 64. The input
     * is read a word at a time; only the set bits are stored.
     */
    public static void scatterBits(MemorySegment in, int from, int to,
            int[] dest, MemorySegment out) {
        for (int base = from; base < to; base += 64) {
            long word = Bitmap.wordAt(in, base >>> 6, to);
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
