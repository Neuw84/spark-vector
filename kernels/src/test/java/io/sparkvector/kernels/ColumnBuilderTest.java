package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
