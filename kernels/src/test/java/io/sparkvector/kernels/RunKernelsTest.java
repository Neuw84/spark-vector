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
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

import io.sparkvector.kernels.reference.SortReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Run boundaries on sorted keys against a scalar reference built on {@link
 * RunMerge#compareKeys}.
 */
class RunKernelsTest {

    private static final int[] SIZES = {0, 1, 2, 63, 64, 65,
            127, 128, 1000, 4097};

    @Test
    void boundariesMatchTheReferenceOnEveryKeyType() {
        Random rnd = new Random(41);
        try (Arena arena = Arena.ofConfined()) {
            for (int n : SIZES) {
                for (boolean nullsFirst : new boolean[] {true, false}) {
                    int nulls = n == 0 ? 0 : rnd.nextInt(Math.max(1, n / 4));
                    check(sortedInts(arena, rnd, n, nulls, nullsFirst), n, "int32 n=" + n);
                    check(sortedLongs(arena, rnd, n, nulls, nullsFirst), n, "int64 n=" + n);
                    check(sortedDoubles(arena, rnd, n, nulls, nullsFirst), n, "float64 n=" + n);
                    check(sortedStrings(arena, rnd, n, nulls, nullsFirst, false),
                            n, "utf8 n=" + n);
                    check(sortedStrings(arena, rnd, n, nulls, nullsFirst, true),
                            n, "dictionary n=" + n);
                    check(sortedBooleans(arena, rnd, n, nulls, nullsFirst), n, "bool n=" + n);
                }
            }
        }
    }

    @Test
    void twoKeysAndTheDoubleTotalOrder() {
        try (Arena arena = Arena.ofConfined()) {
            // -0.0 and 0.0 are one run, NaN and NaN one run: Spark's join equality.
            double[] d = {-1.0, -0.0, 0.0, 0.0,
                    2.5, Double.NaN, Double.NaN};
            VectorBuffers key = ArrowLayout.ofDoubles(arena, d, null);
            assertArrayEquals(new int[] {0, 1, 4, 5, 7},
                    starts(new VectorBuffers[] {key}, d.length));
            // Two keys: a boundary on either is a boundary.
            int[] a = {1, 1, 1, 2, 2, 2};
            int[] b = {7, 7, 8, 8, 8, 9};
            VectorBuffers[] keys = {ArrowLayout.ofInts(arena, a, null), ArrowLayout.ofInts(arena, b, null)};
            assertArrayEquals(new int[] {0, 2, 3, 5, 6},
                    starts(keys, a.length));
            // Nulls: a block of nulls is one run, null against a value a boundary, on a key with no data change.
            int[] c = {5, 5, 5, 5};
            boolean[] nulls = {true, true, false, false};
            VectorBuffers withNulls = ArrowLayout.ofInts(arena, c, nulls);
            assertArrayEquals(new int[] {0, 2, 4}, starts(new VectorBuffers[] {withNulls}, c.length));
        }
    }

    private static void check(VectorBuffers key, int n, String what) {
        VectorBuffers[] keys = {key};
        int[] expected = referenceStarts(keys, n);
        assertArrayEquals(expected, starts(keys, n), what);
    }

    private static int[] starts(VectorBuffers[] keys, int n) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = Bitmap.allocate(arena, Math.max(n, 1));
            RunKernels.boundaries(keys, n, out);
            int[] starts = RunKernels.runStarts(out, n);
            assertEquals(n, starts[starts.length - 1], "end sentinel");
            return starts;
        }
    }

    private static int[] referenceStarts(VectorBuffers[] keys, int n) {
        int[] tmp = new int[n + 1];
        int o = 0;
        for (int i = 0; i < n; i++) {
            boolean boundary = i == 0;
            for (int k = 0;
                 k < keys.length && !boundary;
                 k++) {
                boundary = RunMerge.compareKeys(keys[k], i, keys[k], i - 1, false,
                        true)
                           != 0;
            }
            if (boundary) {
                tmp[o++] = i;
            }
        }
        tmp[o++] = n;
        return Arrays.copyOf(tmp, o);
    }

    // ---- sorted fixtures: few distinct values (long runs), nulls as a block first or last

    private static boolean[] nullBlock(int n, int nulls, boolean first) {
        if (nulls == 0) {
            return null;
        }
        boolean[] b = new boolean[n];
        for (int i = 0; i < nulls; i++) {
            b[first ? i : n - 1 - i] = true;
        }
        return b;
    }

    private static VectorBuffers sortedInts(Arena arena, Random rnd, int n,
            int nulls, boolean nullsFirst) {
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextInt(7) - 3;
        }
        Arrays.sort(v);
        boolean[] b = nullBlock(n, nulls, nullsFirst);
        if (b != null) {
            // Null rows hold arbitrary values: the kernel must not read them as keys.
            for (int i = 0; i < n; i++) {
                if (b[i]) {
                    v[i] = rnd.nextInt();
                }
            }
        }
        return ArrowLayout.ofInts(arena, v, b);
    }

    private static VectorBuffers sortedLongs(Arena arena, Random rnd, int n,
            int nulls, boolean nullsFirst) {
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rnd.nextInt(5) - 2) * 3_000_000_000L;
        }
        Arrays.sort(v);
        return ArrowLayout.ofLongs(arena, v, nullBlock(n, nulls, nullsFirst));
    }

    private static VectorBuffers sortedDoubles(Arena arena, Random rnd, int n,
            int nulls, boolean nullsFirst) {
        double[] pool = {-2.5, -0.0, 0.0, 1.25,
                Double.POSITIVE_INFINITY, Double.NaN};
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = pool[rnd.nextInt(pool.length)];
        }
        // Sort in Spark's order: by the total-order key.
        Double[] boxed = new Double[n];
        for (int i = 0; i < n; i++) {
            boxed[i] = v[i];
        }
        Arrays.sort(boxed,
                (x, y) -> Long.compareUnsigned(SortKernels.doubleKey(x), SortKernels.doubleKey(y)));
        for (int i = 0; i < n; i++) {
            v[i] = boxed[i];
        }
        return ArrowLayout.ofDoubles(arena, v, nullBlock(n, nulls, nullsFirst));
    }

    private static VectorBuffers sortedStrings(Arena arena, Random rnd, int n,
            int nulls, boolean nullsFirst, boolean dictionary) {
        String[] pool = {"", "a", "ab", "abc", "b", "ba",
                "\u00e9", "a longer value than eight bytes"};
        String[] v = new String[n];
        for (int i = 0; i < n; i++) {
            v[i] = pool[rnd.nextInt(pool.length)];
        }
        Arrays.sort(v,
                (x, y) -> SortReference.compareBytes(x.getBytes(StandardCharsets.UTF_8), y.getBytes(StandardCharsets.UTF_8)));
        boolean[] b = nullBlock(n, nulls, nullsFirst);
        if (!dictionary) {
            if (b != null) {
                for (int i = 0; i < n; i++) {
                    if (b[i]) {
                        v[i] = null;
                    }
                }
            }
            return ArrowLayout.ofStrings(arena, v);
        }
        VectorBuffers dict = ArrowLayout.ofStrings(arena, pool);
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            ids[i] = Arrays.asList(pool).indexOf(v[i]);
        }
        SegmentVectorBuffers idx = ArrowLayout.ofInts(arena, ids, b);
        return SegmentVectorBuffers.dictionaryUtf8(n, idx.validity(), idx.data(), dict);
    }

    private static VectorBuffers sortedBooleans(Arena arena, Random rnd, int n,
            int nulls, boolean nullsFirst) {
        boolean[] v = new boolean[n];
        int falses = n == 0 ? 0 : rnd.nextInt(n + 1);
        for (int i = falses; i < n; i++) {
            v[i] = true;
        }
        return ArrowLayout.ofBooleans(arena, v, nullBlock(n, nulls, nullsFirst));
    }
}
