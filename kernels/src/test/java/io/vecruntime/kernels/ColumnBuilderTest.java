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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ColumnBuilder} on UTF8: the bulk append of a plain batch (#394) and
 * the selected path.
 */
class ColumnBuilderTest {

    @Test
    void utf8BulkAppendMatchesRowByRow() {
        Random rnd = new Random(94);
        try (Arena arena = Arena.ofConfined()) {
            ColumnBuilder b = new ColumnBuilder(arena, VecType.UTF8, 4);
            List<String> expected = new ArrayList<>();
            for (int batch = 0; batch < 5; batch++) {
                int n = 1 + rnd.nextInt(700);
                String[] values = new String[n];
                for (int i = 0; i < n; i++) {
                    int kind = rnd.nextInt(10);
                    values[i] = kind == 0
                            ? null
                            : kind == 1 ? "" : randomString(rnd, 1 + rnd.nextInt(40));
                    expected.add(values[i]);
                }
                b.append(ArrowLayout.ofStrings(arena, values));
            }
            VectorBuffers out = b.view();
            assertEquals(expected.size(), out.length());
            for (int i = 0; i < expected.size(); i++) {
                String e = expected.get(i);
                if (e == null) {
                    assertTrue(out.isNull(i), "null at " + i);
                } else {
                    assertFalse(out.isNull(i), "non-null at " + i);
                    assertEquals(e, out.getString(i), "value at " + i);
                }
            }
        }
    }

    @Test
    void utf8AppendOfASelectionStillGathers() {
        try (Arena arena = Arena.ofConfined()) {
            String[] values = {"alpha", null, "", "delta", "echo", "foxtrot"};
            VectorBuffers in = ArrowLayout.ofStrings(arena, values);
            var selection = ArrowLayout.allocateBitmap(arena, values.length);
            Bitmap.set(selection, 0);
            Bitmap.set(selection, 1);
            Bitmap.set(selection, 3);
            Bitmap.set(selection, 5);
            ColumnBuilder b = new ColumnBuilder(arena, VecType.UTF8, 2);
            b.append(in, selection, 4);
            b.append(in); // and a bulk batch after it
            VectorBuffers out = b.view();
            assertEquals(10, out.length());
            assertEquals("alpha", out.getString(0));
            assertTrue(out.isNull(1));
            assertEquals("delta", out.getString(2));
            assertEquals("foxtrot", out.getString(3));
            for (int i = 0; i < values.length; i++) {
                if (values[i] == null) {
                    assertTrue(out.isNull(4 + i));
                } else {
                    assertEquals(values[i], out.getString(4 + i));
                }
            }
        }
    }

    @Test
    void owningBuilderGrowsAndReleasesAndPredictsItsFootprint() {
        Random rnd = new Random(416);
        try (Arena arena = Arena.ofConfined()) {
            for (VecType type : new VecType[] {VecType.INT32, VecType.INT64, VecType.BOOL, VecType.UTF8}) {
                ColumnBuilder b = ColumnBuilder.owning(type, 16);
                long footprint = b.allocatedBytes();
                int rows = 0;
                for (int batch = 0; batch < 40; batch++) {
                    int n = 1 + rnd.nextInt(1500);
                    VectorBuffers in = randomColumn(arena, type, n, rnd, batch % 3 == 0);
                    // The prediction is the footprint after the append, growth and lazy validity included.
                    long predicted = b.bytesAfterAppend(in, null, n);
                    b.append(in);
                    rows += n;
                    assertEquals(predicted, b.allocatedBytes(), type + " after batch " + batch);
                    assertTrue(b.allocatedBytes() >= footprint, "a footprint never shrinks while appending");
                    footprint = b.allocatedBytes();
                }
                VectorBuffers out = b.view();
                assertEquals(rows, out.length());
                // The view is the whole column, in memory the builder owns: readable until close.
                valueAt(out, rows - 1);
                b.close();
                assertThrows(IllegalStateException.class, () -> valueAt(out, 0), "closed with the builder");
                // A no-op for a builder over a caller's arena.
                ColumnBuilder shared = new ColumnBuilder(arena, type, 16);
                shared.append(randomColumn(arena, type, 100, rnd, true));
                shared.close();
                assertEquals(100, shared.view()
                                        .length());
            }
        }
    }

    @Test
    void owningBuilderPredictsASelectedAppend() {
        try (Arena arena = Arena.ofConfined()) {
            String[] values = {"alpha", null, "", "delta", "echo", "foxtrot"};
            VectorBuffers in = ArrowLayout.ofStrings(arena, values);
            var selection = ArrowLayout.allocateBitmap(arena, values.length);
            Bitmap.set(selection, 0);
            Bitmap.set(selection, 1);
            Bitmap.set(selection, 3);
            Bitmap.set(selection, 5);
            ColumnBuilder b = ColumnBuilder.owning(VecType.UTF8, 2);
            long predicted = b.bytesAfterAppend(in, selection, 4);
            b.append(in, selection, 4);
            assertEquals(predicted, b.allocatedBytes());
            assertEquals("foxtrot", b.view()
                    .getString(3));
            b.close();
        }
    }

    @Test
    void sliceReadsTheSameRows() {
        Random rnd = new Random(7);
        try (Arena arena = Arena.ofConfined()) {
            for (VecType type : new VecType[] {VecType.INT32, VecType.INT64, VecType.FLOAT64, VecType.BOOL, VecType.UTF8}) {
                for (boolean nulls : new boolean[] {false, true}) {
                    VectorBuffers col = randomColumn(arena, type, 1000, rnd, nulls);
                    int[][] ranges = {{0, 1000}, {64, 1000}, {128, 129},
                            {896, 1000}, {320, 320}};
                    for (int[] r : ranges) {
                        VectorBuffers s = col.slice(r[0], r[1]);
                        assertEquals(r[1] - r[0], s.length());
                        for (int i = 0; i < s.length(); i++) {
                            assertEquals(col.isNull(r[0] + i), s.isNull(i), type + " null at " + i);
                            if (!s.isNull(i)) {
                                assertEquals(valueAt(col, r[0] + i), valueAt(s, i), type + " value at " + i);
                            }
                        }
                    }
                    assertThrows(IllegalArgumentException.class, () -> col.slice(1, 10));
                    assertThrows(IllegalArgumentException.class, () -> col.slice(64, 1001));
                }
            }
            // A dictionary-encoded slice keeps the dictionary and slices the indices.
            VectorBuffers dict = ArrowLayout.ofStrings(arena, new String[] {"a", "bb", "ccc"});
            var ids = ArrowLayout.allocateData(arena, VecType.INT32, 130);
            for (int i = 0; i < 130; i++) {
                ids.set(VectorBuffers.LE_INT, (long) i << 2, i % 3);
            }
            VectorBuffers encoded = SegmentVectorBuffers.dictionaryUtf8(130, null, ids, dict);
            VectorBuffers s = encoded.slice(64, 130);
            assertTrue(s.isDictionaryEncoded());
            for (int i = 0; i < 66; i++) {
                assertEquals(encoded.getString(64 + i), s.getString(i));
            }
        }
    }

    private static Object valueAt(VectorBuffers col, int i) {
        return switch (col.type()) {
            case INT32 -> col.getInt(i);
            case INT64 -> col.getLong(i);
            case FLOAT64 -> col.getDouble(i);
            case BOOL -> col.getBoolean(i);
            case UTF8 -> col.getString(i);
            case DECIMAL128 -> col.getDecimal128(i);
        };
    }

    private static VectorBuffers randomColumn(Arena arena, VecType type, int n,
            Random rnd, boolean nulls) {
        boolean[] nn = new boolean[n];
        boolean any = false;
        for (int i = 0; i < n; i++) {
            if (nulls && rnd.nextInt(5) == 0) {
                nn[i] = true;
                any = true;
            }
        }
        boolean[] nullable = any ? nn : null;
        switch (type) {
            case INT32 -> {
                int[] v = new int[n];
                for (int i = 0; i < n; i++) {
                    v[i] = rnd.nextInt();
                }
                return ArrowLayout.ofInts(arena, v, nullable);
            }
            case INT64 -> {
                long[] v = new long[n];
                for (int i = 0; i < n; i++) {
                    v[i] = rnd.nextLong();
                }
                return ArrowLayout.ofLongs(arena, v, nullable);
            }
            case FLOAT64 -> {
                double[] v = new double[n];
                for (int i = 0; i < n; i++) {
                    v[i] = rnd.nextDouble();
                }
                return ArrowLayout.ofDoubles(arena, v, nullable);
            }
            case BOOL -> {
                boolean[] v = new boolean[n];
                for (int i = 0; i < n; i++) {
                    v[i] = rnd.nextBoolean();
                }
                return ArrowLayout.ofBooleans(arena, v, nullable);
            }
            case UTF8 -> {
                String[] v = new String[n];
                for (int i = 0; i < n; i++) {
                    v[i] = nullable != null && nullable[i]
                            ? null
                            : randomString(rnd, rnd.nextInt(12));
                }
                return ArrowLayout.ofStrings(arena, v);
            }
            default -> throw new IllegalArgumentException(type.toString());
        }
    }

    private static String randomString(Random rnd, int len) {
        StringBuilder sb = new StringBuilder(len);
        for (int i = 0; i < len; i++) {
            int k = rnd.nextInt(30);
            sb.append(
                    k < 26
                            ? (char) ('a' + k)
                            : k == 26
                                    ? 'é'
                                    : k == 27
                                            ? '∑'
                                            : k == 28 ? ' ' : '9');
        }
        return sb.toString();
    }
}
