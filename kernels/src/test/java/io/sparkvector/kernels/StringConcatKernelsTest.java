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

import io.sparkvector.kernels.StringConcatKernels.Part;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * {@link StringConcatKernels} against a {@code java.lang.String} oracle with
 * Spark's null rules.
 */
class StringConcatKernelsTest {

    private static final String[] A = {"Spark", "", null, "日本", "x", "😀",
            "a", null};
    private static final String[] B = {"SQL", "b", "c", null, "", "y",
            "b", null};
    private static final String[] C = {"!", "", "?", "語", "z", null,
            "c", "only"};

    private static String read(VectorBuffers v, int i) {
        if (v.validity() != null && !Bitmap.isSet(v.validity(), i)) {
            return null;
        }
        return new String(v.getUtf8Bytes(i), StandardCharsets.UTF_8);
    }

    private static VectorBuffers dictionaryOf(Arena arena, String[] values) {
        java.util.List<String> distinct = new java.util.ArrayList<>();
        for (String v : values) {
            if (v != null && !distinct.contains(v)) {
                distinct.add(v);
            }
        }
        int[] idx = new int[values.length];
        boolean[] nulls = new boolean[values.length];
        for (int i = 0; i < values.length; i++) {
            if (values[i] == null) {
                nulls[i] = true;
            } else {
                idx[i] = distinct.indexOf(values[i]);
            }
        }
        VectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
        return SegmentVectorBuffers.dictionaryUtf8(values.length, indices.validity(), indices.data(),
                ArrowLayout.ofStrings(arena, distinct.toArray(new String[0])));
    }

    private static String concat(String... parts) {
        StringBuilder sb = new StringBuilder();
        for (String p : parts) {
            if (p == null) {
                return null;
            }
            sb.append(p);
        }
        return sb.toString();
    }

    private static String concatWs(String sep, String... parts) {
        if (sep == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (String p : parts) {
            if (p == null) {
                continue;
            }
            if (!first) {
                sb.append(sep);
            }
            sb.append(p);
            first = false;
        }
        return sb.toString();
    }

    @Test
    void concatIsNullIntolerantAndConcatWsSkipsNulls() {
        int n = A.length;
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofStrings(arena, A);
            VectorBuffers b = dictionaryOf(arena, B);
            VectorBuffers c = ArrowLayout.ofStrings(arena, C);
            Part[] parts = {Part.of(a), Part.literal("-".getBytes(StandardCharsets.UTF_8)), Part.of(b),
                    Part.of(c)};
            MemorySegment validity = ArrowLayout.allocateBitmap(arena, n);
            BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n);
            BitmapKernels.and(validity, c.validity(), validity, n);
            VectorBuffers out = StringConcatKernels.concat(parts, n, validity, arena);
            for (int i = 0; i < n; i++) {
                assertEquals(concat(A[i], "-", B[i], C[i]), read(out, i), "concat row " + i);
            }
            // A single part and many parts.
            VectorBuffers one = StringConcatKernels.concat(new Part[] {Part.of(a)}, n, a.validity(),
                    arena);
            for (int i = 0; i < n; i++) {
                assertEquals(A[i], read(one, i));
            }
            Part[] many = new Part[12];
            for (int k = 0; k < 12; k++) {
                many[k] = k % 2 == 0 ? Part.of(c) : Part.literal("|".getBytes(StandardCharsets.UTF_8));
            }
            VectorBuffers wide = StringConcatKernels.concat(many, n, c.validity(), arena);
            for (int i = 0; i < n; i++) {
                String expected = C[i] == null ? null : (C[i] + "|").repeat(6);
                assertEquals(expected, read(wide, i), "12 parts row " + i);
            }
            // concat_ws with a literal separator, a lane separator (with a null) and no live parts.
            VectorBuffers ws = StringConcatKernels.concatWs(
                    Part.literal(", ".getBytes(StandardCharsets.UTF_8)),
                    new Part[] {Part.of(a), Part.of(b), Part.of(c)},
                    n,
                    arena);
            for (int i = 0; i < n; i++) {
                assertEquals(concatWs(", ", A[i], B[i], C[i]), read(ws, i), "concat_ws row " + i);
            }
            String[] seps = {"/", null, "", "・", "--", "s",
                    null, "+"};
            VectorBuffers sep = ArrowLayout.ofStrings(arena, seps);
            VectorBuffers ws2 = StringConcatKernels.concatWs(
                    Part.of(sep),
                    new Part[] {Part.of(a), Part.literal("L".getBytes(StandardCharsets.UTF_8)), Part.of(b)},
                    n,
                    arena);
            for (int i = 0; i < n; i++) {
                assertEquals(concatWs(seps[i], A[i], "L", B[i]), read(ws2, i), "concat_ws lane sep row " + i);
            }
            String[] allNull = {null, null, null, null, null, null,
                    null, null};
            VectorBuffers nulls = ArrowLayout.ofStrings(arena, allNull);
            VectorBuffers ws3 = StringConcatKernels.concatWs(
                    Part.literal(",".getBytes(StandardCharsets.UTF_8)),
                    new Part[] {Part.of(nulls), Part.of(nulls)},
                    n,
                    arena);
            for (int i = 0; i < n; i++) {
                assertEquals("", read(ws3, i));
            }
            // The cap declines an oversized batch.
            VectorBuffers big = ArrowLayout.ofStrings(arena, new String[] {"x".repeat(1 << 20)});
            Part[] huge = new Part[1025];
            java.util.Arrays.fill(huge, Part.of(big));
            assertThrows(IllegalStateException.class, () -> StringConcatKernels.concat(huge, 1, null, arena));
        }
    }

    @Test
    void fromRowsBuildsALaneWithNullsFromEitherSide() {
        try (Arena arena = Arena.ofConfined()) {
            byte[][] rows = {"ab".getBytes(StandardCharsets.UTF_8), null, "".getBytes(StandardCharsets.UTF_8),
                    "日本".getBytes(StandardCharsets.UTF_8), "x".getBytes(StandardCharsets.UTF_8)};
            MemorySegment validity = ArrowLayout.allocateBitmap(arena, 5);
            for (int i : new int[] {0, 1, 2, 3}) {
                Bitmap.set(validity, i); // row 4 null from the caller's validity
            }
            VectorBuffers out = StringConcatKernels.fromRows(rows, validity, arena);
            String[] expected = {"ab", null, "", "日本", null};
            for (int i = 0; i < 5; i++) {
                assertEquals(expected[i], read(out, i), "row " + i);
            }
            VectorBuffers all = StringConcatKernels.fromRows(rows, null, arena);
            assertEquals("x", read(all, 4));
            assertEquals(null, read(all, 1));
        }
    }

    @Test
    void eltPicksByIndex() {
        int n = A.length;
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofStrings(arena, A);
            VectorBuffers b = dictionaryOf(arena, B);
            Part[] parts = {Part.of(a), Part.of(b), Part.literal("lit".getBytes(StandardCharsets.UTF_8))};
            int[] idx = {1, 2, 3, 0, 4, 1,
                    -1, 2};
            boolean[] idxNull = new boolean[n];
            idxNull[5] = true;
            VectorBuffers index = ArrowLayout.ofInts(arena, idx, idxNull);
            VectorBuffers out = StringConcatKernels.elt(index, parts, n, arena);
            String[] expected = {"Spark", "b", "lit", null, null, null,
                    null, null};
            for (int i = 0; i < n; i++) {
                assertEquals(expected[i], read(out, i), "elt row " + i);
            }
            // Out-of-range detection honours the active mask and skips null indices.
            assertEquals(3, StringConcatKernels.firstInvalidIndex(index, 3, null, n));
            MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
            for (int i : new int[] {0, 1, 2, 5, 7}) {
                Bitmap.set(active, i);
            }
            assertEquals(-1, StringConcatKernels.firstInvalidIndex(index, 3, active, n));
            Bitmap.set(active, 6);
            assertEquals(6, StringConcatKernels.firstInvalidIndex(index, 3, active, n));
        }
    }
}
