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
import java.nio.charset.StandardCharsets;

import io.vecruntime.kernels.StringConcatKernels.Part;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link StringSearchKernels} against {@code java.lang.String} oracles with
 * Spark's rules.
 */
class StringSearchKernelsTest {

    private static final String[] HAY = {"Spark SQL", "", "aaaa", "日本語テキスト日本", "a,b,,c", "www.apache.org",
            null, "😀x😀y😀", "abcabcabc", "x"};
    private static final String[] NEEDLE = {"SQL", "", "aa", "日本", ",", ".",
            "z", "😀", "bc", "xyz"};

    private static String read(VectorBuffers v, int i) {
        if (v.validity() != null && !Bitmap.isSet(v.validity(), i)) {
            return null;
        }
        return new String(v.getUtf8Bytes(i), StandardCharsets.UTF_8);
    }

    private static Integer readInt(VectorBuffers v, int i) {
        if (v.validity() != null && !Bitmap.isSet(v.validity(), i)) {
            return null;
        }
        return v.getInt(i);
    }

    private static Part lit(String s) {
        return Part.literal(s.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Spark's indexOf over code points: the empty needle is at 0; otherwise
     * Java's indexOf from code point start.
     */
    private static int indexOf(String h, String n, int start) {
        if (n.isEmpty()) {
            return 0;
        }
        int[] cps = h.codePoints().toArray();
        if (start > cps.length) {
            return -1;
        }
        int from = h.offsetByCodePoints(0, Math.min(start, cps.length));
        int at = h.indexOf(n, from);
        return at < 0 ? -1 : h.codePointCount(0, at);
    }

    private static String substringIndex(String s, String d, int count) {
        if (d.isEmpty() || count == 0) {
            return "";
        }
        if (count > 0) {
            int idx = -1;
            for (int k = 0; k < count; k++) {
                idx = s.indexOf(d, idx + 1);
                if (idx < 0) {
                    return s;
                }
            }
            return s.substring(0, idx);
        }
        int idx = s.length();
        for (int k = 0; k < -count; k++) {
            idx = s.lastIndexOf(d, idx - 1);
            if (idx < 0) {
                return s;
            }
        }
        return s.substring(idx + d.length());
    }

    private static String splitPart(String s, String d, int k) {
        String[] parts = d.isEmpty() ? new String[] {s} : s.split(java.util.regex.Pattern.quote(d), -1);
        int t = k > 0 ? k : parts.length + k + 1;
        return t < 1 || t > parts.length
                ? ""
                : parts[t - 1];
    }

    @Test
    void locateReplaceAndFindInSet() {
        int n = HAY.length;
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers h = ArrowLayout.ofStrings(arena, HAY);
            VectorBuffers nd = ArrowLayout.ofStrings(arena, NEEDLE);
            for (int pos : new int[] {1, 0, -3, 2, 3, 100}) {
                VectorBuffers out = StringSearchKernels.locate(Part.of(nd), Part.of(h), null, pos,
                        n, arena);
                for (int i = 0; i < n; i++) {
                    Integer expected = HAY[i] == null
                            ? null
                            : pos < 1 ? 0 : indexOf(HAY[i], NEEDLE[i], pos - 1) + 1;
                    assertEquals(expected, readInt(out, i), "locate("
                            + NEEDLE[i]
                            + ", "
                            + HAY[i]
                            + ", "
                            + pos
                            + ")");
                }
            }
            // A null position gives 0 even for null strings; a literal needle.
            int[] positions = {1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1};
            boolean[] posNull = new boolean[n];
            posNull[0] = true;
            posNull[6] = true;
            VectorBuffers posCol = ArrowLayout.ofInts(arena, positions, posNull);
            VectorBuffers l2 = StringSearchKernels.locate(lit("a"), Part.of(h), posCol, 0,
                    n, arena);
            assertEquals(0, readInt(l2, 0));
            assertEquals(0, readInt(l2, 6));
            assertEquals(1, readInt(l2, 2));
            assertEquals(5, readInt(l2, 5));
            // replace: overlapping-looking matches advance past the match; empty search leaves the row.
            VectorBuffers r = StringSearchKernels.replace(Part.of(h), Part.of(nd), lit("<>"),
                    n, h.validity(), arena);
            for (int i = 0; i < n; i++) {
                String expected = HAY[i] == null
                        ? null
                        : NEEDLE[i].isEmpty() ? HAY[i] : HAY[i].replace(NEEDLE[i], "<>");
                assertEquals(expected, read(r, i), "replace row " + i);
            }
            VectorBuffers r2 = StringSearchKernels.replace(Part.of(h), lit("a"), lit(""),
                    n, h.validity(), arena);
            assertEquals("", read(r2, 2));
            assertEquals(",b,,c", read(r2, 4));
            // find_in_set
            String[] sets = {"a,Spark SQL,b", "", ",,", "x,日本語テキスト日本", "a,b,,c", "org,www.apache.org",
                    "z", "😀x😀y😀,z", "a,b", ",x"};
            VectorBuffers st = ArrowLayout.ofStrings(arena, sets);
            VectorBuffers f = StringSearchKernels.findInSet(Part.of(h), Part.of(st), n,
                    h.validity(), arena);
            Integer[] expected = {2, 1, 0, 2, 0, 2,
                    null, 1, 0, 2};
            for (int i = 0; i < n; i++) {
                assertEquals(expected[i], readInt(f, i), "find_in_set row " + i);
            }
        }
    }

    @Test
    void translateSubstringIndexAndSplitPart() {
        int n = HAY.length;
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers h = ArrowLayout.ofStrings(arena, HAY);
            int[] from = {'a', 'S', '日', 0x1F600, ','};
            byte[][] to = {"1".getBytes(StandardCharsets.UTF_8), "ß".getBytes(StandardCharsets.UTF_8), null,
                    "!".getBytes(StandardCharsets.UTF_8), ";".getBytes(StandardCharsets.UTF_8)};
            VectorBuffers t = StringSearchKernels.translate(Part.of(h), from, to, n,
                    h.validity(), arena);
            String[] expected = {"ßp1rk ßQL", "", "1111", "本語テキスト本", "1;b;;c", "www.1p1che.org",
                    null, "!x!y!", "1bc1bc1bc", "x"};
            for (int i = 0; i < n; i++) {
                assertEquals(expected[i], read(t, i), "translate row " + i);
            }
            // substring_index over several counts and delimiters.
            for (int count : new int[] {-3, -2, -1, 0, 1, 2,
                    3}) {
                VectorBuffers s = StringSearchKernels.substringIndex(Part.of(h), Part.of(ArrowLayout.ofStrings(arena, NEEDLE)), null, count,
                        n, h.validity(), arena);
                for (int i = 0; i < n; i++) {
                    String exp = HAY[i] == null ? null : substringIndex(HAY[i], NEEDLE[i], count);
                    assertEquals(exp, read(s, i), "substring_index("
                            + HAY[i]
                            + ", "
                            + NEEDLE[i]
                            + ", "
                            + count
                            + ")");
                }
            }
            VectorBuffers dot = StringSearchKernels.substringIndex(Part.of(h), lit("."), null, 2,
                    n, h.validity(), arena);
            assertEquals("www.apache", read(dot, 5));
            // split_part from either end, out of range, empty delimiter; the zero-part probe.
            for (int k : new int[] {-4, -1, 1, 2, 3, 5}) {
                VectorBuffers s = StringSearchKernels.splitPart(Part.of(h), Part.of(ArrowLayout.ofStrings(arena, NEEDLE)), null, k,
                        n, h.validity(), arena);
                for (int i = 0; i < n; i++) {
                    String exp = HAY[i] == null ? null : splitPart(HAY[i], NEEDLE[i], k);
                    assertEquals(exp, read(s, i), "split_part("
                            + HAY[i]
                            + ", "
                            + NEEDLE[i]
                            + ", "
                            + k
                            + ")");
                }
            }
            int[] parts = {1, 0, 2, 0, 1, 1,
                    0, 1, 1, 1};
            VectorBuffers partCol = ArrowLayout.ofInts(arena, parts, null);
            assertEquals(1, StringSearchKernels.firstZeroPart(partCol, 0, h.validity(), null, n));
            MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
            Bitmap.set(active, 6); // a null haystack row with part 0 must not count
            Bitmap.set(active, 4);
            assertEquals(-1, StringSearchKernels.firstZeroPart(partCol, 0, h.validity(), active, n));
            Bitmap.set(active, 3);
            assertEquals(3, StringSearchKernels.firstZeroPart(partCol, 0, h.validity(), active, n));
        }
    }
}
