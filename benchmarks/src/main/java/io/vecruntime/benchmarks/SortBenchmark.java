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
package io.vecruntime.benchmarks;

import java.lang.foreign.Arena;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.vecruntime.kernels.ArrowLayout;
import io.vecruntime.kernels.RunMerge;
import io.vecruntime.kernels.SegmentVectorBuffers;
import io.vecruntime.kernels.SortKernels;
import io.vecruntime.kernels.VectorBuffers;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * The index sort (#285): the radix kernel against the previous one ({@link
 * LegacySortKernels}, one {@code Arrays.sort(long[])} per 32-bit key pass) over
 * the key types, key counts, null shares and input orders the issue lists. The
 * unit is milliseconds per sort of {@code rows} rows; the number the docs quote
 * is ns/row at 10M random INT64 rows.
 *
 * <pre>
 * java --add-modules=jdk.incubator.vector -jar benchmarks/target/benchmarks.jar SortBenchmark \
 *   -p rows=1000000 -p key=INT64 -p order=random -wi 2 -i 3 -w 1 -r 1 -f 1
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xmx8g"})
@State(Scope.Thread)
public class SortBenchmark {

    @Param({"1000000", "10000000"})
    int rows;

    /**
     * INT32, INT64, FLOAT64, UTF8_SHORT (up to 8 bytes), UTF8_LONG (12-40
     * bytes), UTF8_DICT (dictionary-encoded, 1000 values).
     */
    @Param({"INT32", "INT64", "FLOAT64", "UTF8_SHORT", "UTF8_LONG", "UTF8_DICT"})
    String key;

    /** One key, or three (the named key twice around a low-cardinality int). */
    @Param({"1", "3"})
    int keys;

    /** Share of null rows. */
    @Param({"0", "0.1"})
    double nulls;

    /** random, presorted, reverse, lowcard (100 distinct values). */
    @Param({"random", "presorted", "reverse", "lowcard"})
    String order;

    Arena arena;
    VectorBuffers[] keyColumns;
    boolean[] ascending;
    boolean[] nullsFirst;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(7);
        boolean[] nullFlags = null;
        if (nulls > 0) {
            nullFlags = new boolean[rows];
            for (int i = 0; i < rows; i++) {
                nullFlags[i] = rnd.nextDouble() < nulls;
            }
        }
        // The value's rank in the requested order, from which every key type is derived.
        long[] shape = new long[rows];
        for (int i = 0; i < rows; i++) {
            shape[i] = switch (order) {
                        case "presorted" -> i;
                        case "reverse" -> rows - 1 - i;
                        case "lowcard" -> rnd.nextInt(100);
                        default -> rnd.nextLong();
                    };
        }
        keyColumns = keySet(shape, nullFlags, rnd);
        ascending = keys == 1 ? new boolean[] {true} : new boolean[] {true, false, true};
        nullsFirst = keys == 1 ? new boolean[] {true} : new boolean[] {true, false, true};
        // The same rows cut into eight runs, for the runs-and-merge variant.
        int k = RUNS;
        int per = (rows + k - 1) / k;
        runKeyColumns = new VectorBuffers[k][];
        runSizes = new int[k];
        for (int r = 0; r < k; r++) {
            int from = r * per;
            int to = Math.min(rows, from + per);
            runSizes[r] = to - from;
            long[] part = java.util.Arrays.copyOfRange(shape, from, to);
            boolean[] partNulls = nullFlags == null ? null : java.util.Arrays.copyOfRange(nullFlags, from, to);
            runKeyColumns[r] = keySet(part, partNulls, rnd);
        }
    }

    static final int RUNS = 8;
    VectorBuffers[][] runKeyColumns;
    int[] runSizes;

    /**
     * The key columns over the given rows: the named key alone, or it twice
     * around a four-value int.
     */
    private VectorBuffers[] keySet(long[] shape, boolean[] nullFlags, Random rnd) {
        VectorBuffers main = column(key, shape, nullFlags, rnd);
        if (keys == 1) {
            return new VectorBuffers[] {main};
        }
        int[] small = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            small[i] = rnd.nextInt(4);
        }
        return new VectorBuffers[] {main, ArrowLayout.ofInts(arena, small, null), column(key, shape, nullFlags, rnd)};
    }

    private VectorBuffers column(String type, long[] shape, boolean[] nullFlags,
            Random rnd) {
        int rows = shape.length;
        switch (type) {
            case "INT32" -> {
                int[] v = new int[rows];
                for (int i = 0; i < rows; i++) {
                    v[i] = (int) shape[i];
                }
                return ArrowLayout.ofInts(arena, v, nullFlags);
            }
            case "INT64" -> {
                return ArrowLayout.ofLongs(arena, shape, nullFlags);
            }
            case "FLOAT64" -> {
                double[] v = new double[rows];
                for (int i = 0; i < rows; i++) {
                    v[i] = shape[i] * 1.5;
                }
                return ArrowLayout.ofDoubles(arena, v, nullFlags);
            }
            case "UTF8_SHORT" -> {
                return ArrowLayout.ofStrings(arena, strings(shape, nullFlags, 8));
            }
            case "UTF8_LONG" -> {
                return ArrowLayout.ofStrings(arena, strings(shape, nullFlags, 40));
            }
            case "UTF8_DICT" -> {
                String[] dict = new String[1000];
                for (int d = 0; d < dict.length; d++) {
                    dict[d] = "value-" + Long.toHexString(rnd.nextLong());
                }
                int[] ids = new int[rows];
                for (int i = 0; i < rows; i++) {
                    ids[i] = (int) Math.floorMod(shape[i], (long) dict.length);
                }
                SegmentVectorBuffers idx = ArrowLayout.ofInts(arena, ids, nullFlags);
                return SegmentVectorBuffers.dictionaryUtf8(rows, idx.validity(), idx.data(),
                        ArrowLayout.ofStrings(arena, dict));
            }
            default -> throw new IllegalArgumentException(type);
        }
    }

    /**
     * Strings ordered like the shape value: its hex digits, zero-padded, with a
     * tail to reach {@code length}.
     */
    private static String[] strings(long[] shape, boolean[] nullFlags, int length) {
        String[] v = new String[shape.length];
        for (int i = 0; i < shape.length; i++) {
            if (nullFlags != null && nullFlags[i]) {
                continue;
            }
            String hex = Long.toHexString(shape[i]);
            String s = length <= 8 ? hex.substring(0, Math.min(hex.length(), length)) : hex;
            if (length > 8) {
                StringBuilder sb = new StringBuilder(s);
                while (sb.length() < length) {
                    sb.append('x');
                }
                s = sb.toString();
            }
            v[i] = s;
        }
        return v;
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public int[] radix() {
        return SortKernels.sortIndices(keyColumns, ascending, nullsFirst, rows);
    }

    /**
     * The sort in eight runs of {@code rows / 8} rows each, then the k-way
     * merge ({@link RunMerge}) walked to the end, emitting the merged order as
     * (run, row) pairs -- the operator's path when a partition exceeds {@code
     * spark.vector.sort.runRows}.
     */
    @Benchmark
    public int runs8() {
        int k = RUNS;
        int[][] perm = new int[k][];
        for (int r = 0; r < k; r++) {
            perm[r] = SortKernels.sortIndices(runKeyColumns[r], ascending, nullsFirst, runSizes[r]);
        }
        RunMerge merge = new RunMerge(runKeyColumns, perm, runSizes, ascending, nullsFirst);
        int[] runOf = new int[4096];
        int[] rowOf = new int[4096];
        int sum = 0;
        while (merge.hasNext()) {
            int n = merge.next(runOf, rowOf, 4096);
            sum += runOf[n - 1] + rowOf[n - 1];
        }
        return sum;
    }

    @Benchmark
    public int[] legacy() {
        return LegacySortKernels.sortIndices(keyColumns, ascending, nullsFirst, rows);
    }
}
