package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link StringSliceKernels} against an independent oracle written over Java
 * code point arrays with Spark's rules: 1-based positions, negative starts from
 * the end, silent clamping, whole pad copies plus a partial one. Multi-byte
 * input, empty strings, nulls, a dictionary-encoded column.
 */
class StringSliceKernelsTest {

    private static final String[] VALUES = {"Spark SQL", "", "a", "héllo wörld", "日本語テキスト", "😀x😀y😀",
            null, "ab", "  trims  ", "ñ", "0123456789"};

    // ---- the oracle: Spark's substringSQL over code points

    private static String substringSQL(String s, int pos, int length) {
        int[] cps = s.codePoints().toArray();
        int len = cps.length;
        int start = pos > 0
                ? pos - 1
                : (pos < 0 ? len + pos : 0);
        long untilL = (long) start + length;
        int until = untilL > Integer.MAX_VALUE
                ? Integer.MAX_VALUE
                : (untilL < Integer.MIN_VALUE ? Integer.MIN_VALUE : (int) untilL);
        if (until <= start || start >= len) {
            return "";
        }
        int from = Math.max(start, 0);
        int to = Math.min(until, len);
        if (to <= from) {
            return "";
        }
        return new String(cps, from, to - from);
    }

    private static String pad(String s, int len, String pad,
            boolean left) {
        if (len <= 0) {
            return "";
        }
        int[] cps = s.codePoints().toArray();
        int spaces = len - cps.length;
        if (spaces <= 0 || pad.isEmpty()) {
            return new String(cps, 0, Math.min(len, cps.length));
        }
        int[] p = pad.codePoints().toArray();
        int count = spaces / p.length;
        StringBuilder padding = new StringBuilder();
        for (int k = 0; k < count; k++) {
            padding.append(pad);
        }
        padding.append(new String(p, 0, spaces - p.length * count));
        return left ? padding + s : s + padding;
    }

    private static String overlay(String input, String replace, int pos,
            int len) {
        int length = len >= 0 ? len : replace.codePointCount(0, replace.length());
        return substringSQL(input, 1, pos - 1) + replace + substringSQL(input, pos + length, Integer.MAX_VALUE);
    }

    private static String read(VectorBuffers v, int i) {
        if (v.validity() != null && !Bitmap.isSet(v.validity(), i)) {
            return null;
        }
        return new String(v.getUtf8Bytes(i), StandardCharsets.UTF_8);
    }

    private static VectorBuffers dictionaryOf(Arena arena, String[] values) {
        // Every distinct non-null value once in the dictionary; indices map rows to it (nulls keep index 0).
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

    @Test
    void substringMatchesSparkRules() {
        int[][] cases = {
            {1, 2},
            {1, Integer.MAX_VALUE},
            {0, 3},
            {-3, Integer.MAX_VALUE},
            {-5, 2},
            {-50, 4},
            {5, 1},
            {50, 2},
            {2, 0},
            {2, -1},
            {3, Integer.MAX_VALUE},
            {-1, 1},
            {Integer.MIN_VALUE, 5},
            {1, Integer.MIN_VALUE}
        };
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers plain = ArrowLayout.ofStrings(arena, VALUES);
            VectorBuffers dict = dictionaryOf(arena, VALUES);
            for (VectorBuffers a : new VectorBuffers[] {plain, dict}) {
                for (int[] c : cases) {
                    VectorBuffers out = StringSliceKernels.substring(a, null, c[0], null, c[1],
                            a.validity(), arena);
                    for (int i = 0; i < VALUES.length; i++) {
                        String expected = VALUES[i] == null ? null : substringSQL(VALUES[i], c[0], c[1]);
                        assertEquals(expected, read(out, i), "substring("
                                + VALUES[i]
                                + ", "
                                + c[0]
                                + ", "
                                + c[1]
                                + ")");
                    }
                }
            }
            // Positions and lengths as lanes, with a null in each.
            int[] pos = {1, 2, -1, 3, -2, 1,
                    1, 2, 4, 1, -4};
            int[] len = {3, 1, 5, 2, 2, 1,
                    1, 1, 100, 0, 2};
            boolean[] posNull = new boolean[VALUES.length];
            posNull[3] = true;
            VectorBuffers posCol = ArrowLayout.ofInts(arena, pos, posNull);
            VectorBuffers lenCol = ArrowLayout.ofInts(arena, len, null);
            MemorySegment validity = ArrowLayout.allocateBitmap(arena, VALUES.length);
            BitmapKernels.combineValidity(plain.validity(), posCol.validity(), validity, VALUES.length);
            VectorBuffers out = StringSliceKernels.substring(plain, posCol, 0, lenCol, 0, validity,
                    arena);
            for (int i = 0; i < VALUES.length; i++) {
                String expected = VALUES[i] == null || posNull[i]
                        ? null
                        : substringSQL(VALUES[i], pos[i], len[i]);
                assertEquals(expected, read(out, i), "row " + i);
            }
        }
    }

    @Test
    void padRepeatSpaceAndOverlay() {
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofStrings(arena, VALUES);
            String[] pads = {"??", " ", "", "ñ", "日本"};
            int[] lens = {0, 1, 3, 5, 11, 12,
                    20, -4};
            for (String p : pads) {
                for (int l : lens) {
                    for (boolean left : new boolean[] {true, false}) {
                        VectorBuffers out = StringSliceKernels.pad(a, null, l, null, p.getBytes(StandardCharsets.UTF_8),
                                left, a.validity(), arena);
                        for (int i = 0; i < VALUES.length; i++) {
                            String expected = VALUES[i] == null ? null : pad(VALUES[i], l, p, left);
                            assertEquals(expected, read(out, i), (left ? "lpad" : "rpad")
                                    + "("
                                    + VALUES[i]
                                    + ", "
                                    + l
                                    + ", '"
                                    + p
                                    + "')");
                        }
                    }
                }
            }
            // A pad column and a length column.
            String[] padValues = {"ab", "-", "xyz", "é", "", "😀",
                    "z", "..", "p", "q", "r"};
            int[] lenValues = {12, 3, 4, 20, 2, 7,
                    5, 6, 0, 3, 15};
            VectorBuffers padCol = ArrowLayout.ofStrings(arena, padValues);
            VectorBuffers lenCol = ArrowLayout.ofInts(arena, lenValues, null);
            VectorBuffers out = StringSliceKernels.pad(a, lenCol, 0, padCol, null, true,
                    a.validity(), arena);
            for (int i = 0; i < VALUES.length; i++) {
                String expected = VALUES[i] == null ? null : pad(VALUES[i], lenValues[i], padValues[i], true);
                assertEquals(expected, read(out, i), "lpad row " + i);
            }
            // repeat and space.
            for (int t : new int[] {-1, 0, 1, 2, 7}) {
                VectorBuffers rep = StringSliceKernels.repeat(a, null, t, a.validity(), arena);
                for (int i = 0; i < VALUES.length; i++) {
                    String expected = VALUES[i] == null
                            ? null
                            : (t <= 0 ? "" : VALUES[i].repeat(t));
                    assertEquals(expected, read(rep, i), "repeat(" + VALUES[i] + ", " + t + ")");
                }
            }
            int[] counts = {0, 1, 5, -3, 40};
            VectorBuffers countCol = ArrowLayout.ofInts(arena, counts, null);
            VectorBuffers sp = StringSliceKernels.space(counts.length, countCol, 0, null, arena);
            for (int i = 0; i < counts.length; i++) {
                assertEquals(" ".repeat(Math.max(counts[i], 0)), read(sp, i));
            }
            // overlay: literal replacement, several positions and lengths incl. the default -1.
            int[][] pl = {
                {6, -1},
                {7, 0},
                {2, 4},
                {1, 2},
                {100, 1},
                {0, -1},
                {-2, 1},
                {3, 100}
            };
            for (int[] c : pl) {
                VectorBuffers ov = StringSliceKernels.overlay(a, null, "_Ü_".getBytes(StandardCharsets.UTF_8), null, c[0],
                        null, c[1], a.validity(), arena);
                for (int i = 0; i < VALUES.length; i++) {
                    String expected = VALUES[i] == null ? null : overlay(VALUES[i], "_Ü_", c[0], c[1]);
                    assertEquals(expected, read(ov, i), "overlay("
                            + VALUES[i]
                            + ", '_Ü_', "
                            + c[0]
                            + ", "
                            + c[1]
                            + ")");
                }
            }
            // The output cap declines a runaway repeat instead of allocating it.
            VectorBuffers big = ArrowLayout.ofStrings(arena, new String[] {"x".repeat(1024)});
            assertThrows(IllegalStateException.class, () -> StringSliceKernels.repeat(big, null, 1 << 21, null, arena));
            assertTrue(StringSliceKernels.numBytesForFirstByte((byte) 0xE6) == 3 && StringSliceKernels.numBytesForFirstByte((byte) 0x80) == 1);
        }
    }
}
