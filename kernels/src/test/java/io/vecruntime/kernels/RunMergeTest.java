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
import java.util.Random;

import io.vecruntime.kernels.reference.SortReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * The k-way merge over runs against the reference sort of the runs'
 * concatenation.
 */
class RunMergeTest {

    /**
     * Run sizes with the awkward cases: empty, single row, an odd length,
     * several equal lengths.
     */
    private static final int[][] RUN_SIZES = {{0, 1, 7}, {100, 100, 100, 100}, {1},
            {0, 0, 5}, {333, 1, 2000, 17}};

    @Test
    void mergeOfSortedRunsEqualsTheReferenceOverTheirConcatenation() {
        Random rnd = new Random(29);
        try (Arena arena = Arena.ofConfined()) {
            for (int[] sizes : RUN_SIZES) {
                for (VecType type : new VecType[] {VecType.INT32, VecType.INT64, VecType.FLOAT64, VecType.UTF8, VecType.BOOL}) {
                    for (boolean asc : new boolean[] {true, false}) {
                        for (boolean nf : new boolean[] {true, false}) {
                            check(arena, rnd, sizes, type, asc, nf);
                        }
                    }
                }
            }
        }
    }

    @Test
    void twoKeysAcrossRunsWithManyTies() {
        Random rnd = new Random(31);
        try (Arena arena = Arena.ofConfined()) {
            int[] sizes = {500, 500, 500};
            VectorBuffers[][] keys = new VectorBuffers[sizes.length][2];
            ColumnBuilder all0 = new ColumnBuilder(arena, VecType.INT32, 1500);
            ColumnBuilder all1 = new ColumnBuilder(arena, VecType.UTF8, 1500);
            for (int r = 0; r < sizes.length; r++) {
                int[] small = new int[sizes[r]];
                String[] flags = new String[sizes[r]];
                for (int i = 0; i < sizes[r]; i++) {
                    small[i] = rnd.nextInt(3);
                    flags[i] = rnd.nextInt(10) == 0 ? null : String.valueOf((char) ('a' + rnd.nextInt(3)));
                }
                keys[r][0] = ArrowLayout.ofInts(arena, small, TestData.nulls(rnd, sizes[r], 0.1));
                keys[r][1] = ArrowLayout.ofStrings(arena, flags);
                all0.append(keys[r][0]);
                all1.append(keys[r][1]);
            }
            boolean[] asc = {true, false};
            boolean[] nf = {false, true};
            int[][] perm = new int[sizes.length][];
            for (int r = 0; r < sizes.length; r++) {
                perm[r] = SortKernels.sortIndices(keys[r], asc, nf, sizes[r]);
            }
            int[] expected = SortReference.sortIndices(new VectorBuffers[] {all0.view(), all1.view()}, asc,
                    nf, 1500);
            assertArrayEquals(expected, merged(keys, perm, sizes, asc, nf, 64), "two keys, three runs");
        }
    }

    @Test
    void refillableRunsStreamInPiecesAndMergeLikeWholeRuns() {
        // #416: a spilled run comes back batch by batch. Run 1 is fed as sorted pieces of 5 rows through
        // refill(); the merge must equal the one over the whole runs, and every next() must stop at a
        // piece boundary so the caller's gather still sees the piece the rows refer to.
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(11);
            for (VecType type : new VecType[] {VecType.INT32, VecType.UTF8, VecType.FLOAT64}) {
                for (boolean asc : new boolean[] {true, false}) {
                    int[] sizes = {17, 23, 9};
                    VectorBuffers[][] keys = new VectorBuffers[sizes.length][1];
                    int total = 0;
                    for (int r = 0; r < sizes.length; r++) {
                        int n = sizes[r];
                        boolean[] nulls = TestData.nulls(rnd, n, 0.2);
                        keys[r][0] = switch (type) {
                                    case INT32 -> {
                                        int[] v = new int[n];
                                        for (int i = 0; i < n; i++) {
                                            v[i] = rnd.nextInt(5);
                                        }
                                        yield ArrowLayout.ofInts(arena, v, nulls);
                                    }
                                    case FLOAT64 -> {
                                        double[] v = new double[n];
                                        for (int i = 0; i < n; i++) {
                                            v[i] = new double[] {-1.5, 0.0, 2.25, Double.NaN}[rnd.nextInt(4)];
                                        }
                                        yield ArrowLayout.ofDoubles(arena, v, nulls);
                                    }
                                    default -> {
                                        String[] v = new String[n];
                                        for (int i = 0; i < n; i++) {
                                            v[i] = nulls[i] ? null : new String[] {"", "a", "ab", "b"}[rnd.nextInt(4)];
                                        }
                                        yield ArrowLayout.ofStrings(arena, v);
                                    }
                                };
                        total += n;
                    }
                    boolean[] ascending = {asc};
                    boolean[] nullsFirst = {!asc};
                    int[][] perm = new int[sizes.length][];
                    for (int r = 0; r < sizes.length; r++) {
                        perm[r] = SortKernels.sortIndices(keys[r], ascending, nullsFirst, sizes[r]);
                    }
                    int[] expected = merged(keys, perm, sizes.clone(), ascending, nullsFirst,
                            4);
                    // Run 1 as pieces: its sorted rows gathered into pieces of 5, each piece a column in order.
                    int pieceRows = 5;
                    java.util.List<VectorBuffers> pieces = new java.util.ArrayList<>();
                    java.util.List<int[]> pieceSource = new java.util.ArrayList<>(); // piece row -> row of run 1
                    for (int from = 0; from < sizes[1]; from += pieceRows) {
                        int n = Math.min(pieceRows, sizes[1] - from);
                        int[] src = java.util.Arrays.copyOfRange(perm[1], from, from + n);
                        ColumnBuilder b = new ColumnBuilder(arena, type, n);
                        for (int i = 0; i < n; i++) {
                            b.append(keys[1][0], ArrowLayout.selectionFromIndices(arena, new int[] {src[i]}, 1, sizes[1]), 1);
                        }
                        pieces.add(b.view());
                        pieceSource.add(src);
                    }
                    VectorBuffers[][] pk = {keys[0], new VectorBuffers[] {pieces.get(0)}, keys[2]};
                    int[][] pp = {perm[0], null, perm[2]};
                    int[] pr = {sizes[0], pieces.get(0).length(), sizes[2]};
                    RunMerge merge = new RunMerge(pk, pp, pr, ascending, nullsFirst,
                            new boolean[] {false, true, false});
                    int[] offset = {0, sizes[0], sizes[0] + sizes[1]};
                    int[] out = new int[total];
                    int[] runOf = new int[4];
                    int[] rowOf = new int[4];
                    int o = 0;
                    int piece = 0;
                    while (merge.hasNext()) {
                        int n = merge.next(runOf, rowOf, 4);
                        for (int j = 0; j < n; j++) {
                            int row = runOf[j] == 1 ? pieceSource.get(piece)[rowOf[j]] : rowOf[j];
                            out[o++] = offset[runOf[j]] + row;
                        }
                        int ex = merge.exhausted();
                        if (ex >= 0) {
                            assertEquals(1, ex);
                            piece++;
                            if (piece < pieces.size()) {
                                merge.refill(1, new VectorBuffers[] {pieces.get(piece)}, null,
                                        pieces.get(piece).length());
                            } else {
                                merge.finish(1);
                            }
                        }
                    }
                    assertEquals(total, o, "every row emitted once");
                    assertArrayEquals(expected, out, type + " asc=" + asc);
                }
            }
        }
    }

    private static void check(Arena arena, Random rnd, int[] sizes,
            VecType type, boolean asc, boolean nf) {
        int total = 0;
        for (int s : sizes) {
            total += s;
        }
        VectorBuffers[][] keys = new VectorBuffers[sizes.length][1];
        ColumnBuilder all = new ColumnBuilder(arena, type, Math.max(total, 1));
        for (int r = 0; r < sizes.length; r++) {
            int n = sizes[r];
            // Few distinct values, so that ties cross runs and the tie-break by run index matters.
            boolean[] nulls = TestData.nulls(rnd, n, 0.15);
            keys[r][0] = switch (type) {
                        case INT32 -> {
                            int[] v = new int[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = rnd.nextInt(6) - 3;
                            }
                            yield ArrowLayout.ofInts(arena, v, nulls);
                        }
                        case INT64 -> {
                            long[] v = new long[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = (rnd.nextInt(6) - 3) * 3_000_000_000L;
                            }
                            yield ArrowLayout.ofLongs(arena, v, nulls);
                        }
                        case FLOAT64 -> {
                            double[] v = new double[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = new double[] {-1.5, -0.0, 0.0, 2.25,
                                        Double.NaN, 7.0}[rnd.nextInt(6)];
                            }
                            yield ArrowLayout.ofDoubles(arena, v, nulls);
                        }
                        case UTF8 -> {
                            String[] v = new String[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = nulls[i] ? null : new String[] {"", "a", "ab", "b", "abcdefghij", "\u00e9"}[rnd.nextInt(6)];
                            }
                            yield ArrowLayout.ofStrings(arena, v);
                        }
                        case BOOL -> {
                            boolean[] v = new boolean[n];
                            for (int i = 0; i < n; i++) {
                                v[i] = rnd.nextBoolean();
                            }
                            yield ArrowLayout.ofBooleans(arena, v, nulls);
                        }
                        default -> throw new IllegalStateException(type.toString());
                    };
            all.append(keys[r][0]);
        }
        boolean[] ascending = {asc};
        boolean[] nullsFirst = {nf};
        int[][] perm = new int[sizes.length][];
        for (int r = 0; r < sizes.length; r++) {
            perm[r] = SortKernels.sortIndices(keys[r], ascending, nullsFirst, sizes[r]);
        }
        int[] expected = total == 0 ? new int[0] : SortReference.sortIndices(new VectorBuffers[] {all.view()}, ascending, nullsFirst, total);
        String what = type
                + " runs="
                + java.util.Arrays.toString(sizes)
                + " asc="
                + asc
                + " nullsFirst="
                + nf;
        assertArrayEquals(expected, merged(keys, perm, sizes, ascending, nullsFirst, 7), what);
        assertArrayEquals(expected, merged(keys, perm, sizes, ascending, nullsFirst, 4096), what + " (one batch)");
    }

    /**
     * Runs the merge in batches of {@code batch} and maps (run, row) to the
     * row's index in the concatenation.
     */
    private static int[] merged(VectorBuffers[][] keys, int[][] perm, int[] sizes,
            boolean[] asc, boolean[] nf, int batch) {
        int[] offset = new int[sizes.length];
        int total = 0;
        for (int r = 0; r < sizes.length; r++) {
            offset[r] = total;
            total += sizes[r];
        }
        RunMerge merge = new RunMerge(keys, perm, sizes, asc, nf);
        int[] out = new int[total];
        int[] runOf = new int[batch];
        int[] rowOf = new int[batch];
        int o = 0;
        while (merge.hasNext()) {
            int n = merge.next(runOf, rowOf, batch);
            for (int j = 0; j < n; j++) {
                out[o++] = offset[runOf[j]] + rowOf[j];
            }
        }
        assertEquals(total, o, "every row emitted once");
        assertFalse(merge.hasNext());
        return out;
    }
}
