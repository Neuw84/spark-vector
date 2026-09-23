package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.Random;

import io.sparkvector.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link StringCompareKernels} against the byte-array reference, plain and
 * dictionary encoded.
 */
class StringCompareKernelsTest {

    private static final String[] ALPHABET = {
        "",
        "a",
        "A",
        "ab",
        "abc",
        "abd",
        "b",
        "s1",
        "s10",
        "s2",
        "MAIL",
        "SHIP",
        "AIR",
        "REG AIR",
        "BUILDING",
        "BUILDINGS",
        "é",
        "e\u0301",
        "日本",
        "日本語",
        "\u00ff",
        "\u0100",
        "zz"
    };

    private static String[] randomStrings(Random rnd, int n, double nullRate) {
        String[] out = new String[n];
        for (int i = 0; i < n; i++) {
            out[i] = rnd.nextDouble() < nullRate ? null : ALPHABET[rnd.nextInt(ALPHABET.length)];
        }
        return out;
    }

    /**
     * The same values as a dictionary-encoded column over ALPHABET; null rows
     * carry index 0.
     */
    private static VectorBuffers dictionaryEncoded(Arena arena, String[] values) {
        int n = values.length;
        int[] idx = new int[n];
        boolean[] nulls = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (values[i] == null) {
                nulls[i] = true;
            } else {
                for (int j = 0; j < ALPHABET.length; j++) {
                    if (ALPHABET[j].equals(values[i])) {
                        idx[i] = j;
                    }
                }
            }
        }
        SegmentVectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
        return SegmentVectorBuffers.dictionaryUtf8(n, indices.validity(), indices.data(),
                ArrowLayout.ofStrings(arena, ALPHABET));
    }

    private static void assertSameValidBits(VectorBuffers a, MemorySegment expected, MemorySegment actual,
            String what) {
        for (int i = 0; i < a.length(); i++) {
            if (a.isNull(i)) {
                continue;
            }
            assertEquals(Bitmap.isSet(expected, i), Bitmap.isSet(actual, i), what
                    + " row "
                    + i
                    + " ("
                    + a.getString(i)
                    + ")");
        }
    }

    private static void checkScalar(Arena arena, VectorBuffers a, String lit) {
        byte[] s = lit.getBytes(StandardCharsets.UTF_8);
        for (CompareOp op : CompareOp.values()) {
            MemorySegment expected = ArrowLayout.allocateBitmap(arena, a.length());
            MemorySegment actual = ArrowLayout.allocateBitmap(arena, a.length());
            ScalarReference.compareUtf8Scalar(a, s, op, expected);
            StringCompareKernels.compareScalar(a, s, op, null, actual);
            assertSameValidBits(a, expected, actual, op + " '" + lit + "'");
        }
    }

    private static void checkColumns(Arena arena, VectorBuffers a, VectorBuffers b) {
        for (CompareOp op : CompareOp.values()) {
            MemorySegment expected = ArrowLayout.allocateBitmap(arena, a.length());
            MemorySegment actual = ArrowLayout.allocateBitmap(arena, a.length());
            ScalarReference.compareUtf8(a, b, op, expected);
            StringCompareKernels.compare(a, b, op, null, actual);
            for (int i = 0; i < a.length(); i++) {
                if (a.isNull(i) || b.isNull(i)) {
                    continue;
                }
                assertEquals(Bitmap.isSet(expected, i), Bitmap.isSet(actual, i), op
                        + " row "
                        + i
                        + " ("
                        + a.getString(i)
                        + " vs "
                        + b.getString(i)
                        + ")");
            }
        }
    }

    @Test
    void orderingIsUnsignedByteWiseWithPrefixesFirst() {
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = ArrowLayout.ofStrings(
                    arena,
                    new String[] {"", "a", "ab", "abc", "b", "s10",
                            "s2", "\u00ff", "\u0100", "é", "e\u0301"});
            MemorySegment out = ArrowLayout.allocateBitmap(arena, a.length());
            StringCompareKernels.compareScalar(a, "ab".getBytes(StandardCharsets.UTF_8), CompareOp.LT, null, out);
            assertTrue(Bitmap.isSet(out, 0), "'' < 'ab'");
            assertTrue(Bitmap.isSet(out, 1), "'a' < 'ab' (prefix first)");
            assertFalse(Bitmap.isSet(out, 2), "'ab' < 'ab'");
            assertFalse(Bitmap.isSet(out, 3), "'abc' < 'ab'");
            assertFalse(Bitmap.isSet(out, 4), "'b' < 'ab'");
            StringCompareKernels.compareScalar(a, "s2".getBytes(StandardCharsets.UTF_8), CompareOp.LT, null, out);
            assertTrue(Bitmap.isSet(out, 5), "'s10' < 's2' (byte order, not numeric)");
            // U+00FF is C3 BF, U+0100 is C4 80: byte order equals code-point order.
            StringCompareKernels.compareScalar(a, "\u0100".getBytes(StandardCharsets.UTF_8), CompareOp.LT, null, out);
            assertTrue(Bitmap.isSet(out, 7));
            assertFalse(Bitmap.isSet(out, 8));
            // Canonically equivalent but byte-different strings are not equal (UTF8_BINARY).
            StringCompareKernels.compareScalar(a, "é".getBytes(StandardCharsets.UTF_8), CompareOp.EQ, null, out);
            assertTrue(Bitmap.isSet(out, 9));
            assertFalse(Bitmap.isSet(out, 10));
            // The empty literal.
            StringCompareKernels.compareScalar(a, new byte[0], CompareOp.EQ, null, out);
            assertTrue(Bitmap.isSet(out, 0));
            assertFalse(Bitmap.isSet(out, 1));
            StringCompareKernels.compareScalar(a, new byte[0], CompareOp.GT, null, out);
            assertFalse(Bitmap.isSet(out, 0));
            assertTrue(Bitmap.isSet(out, 1));
        }
    }

    @Test
    void plainAndDictionaryColumnsMatchTheReferenceForEveryOperator() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(11);
            for (int n : new int[] {1, 63, 64, 65, 500, 4096}) {
                String[] values = randomStrings(rnd, n, 0.1);
                VectorBuffers plain = ArrowLayout.ofStrings(arena, values);
                VectorBuffers dict = dictionaryEncoded(arena, values);
                for (String lit : new String[] {"", "a", "abc", "s1", "MAIL", "BUILDING",
                        "日本", "\u00ff", "not there", "zzz"}) {
                    checkScalar(arena, plain, lit);
                    checkScalar(arena, dict, lit);
                }
                String[] other = randomStrings(rnd, n, 0.1);
                VectorBuffers plainB = ArrowLayout.ofStrings(arena, other);
                VectorBuffers dictB = dictionaryEncoded(arena, other);
                checkColumns(arena, plain, plainB);
                checkColumns(arena, dict, dictB);
                checkColumns(arena, plain, dictB);
                checkColumns(arena, dict, plainB);
                checkColumns(arena, plain, plain);
            }
        }
    }

    @Test
    void inactiveBlocksAreSkippedAndCleared() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 200;
            String[] values = new String[n];
            for (int i = 0; i < n; i++) {
                values[i] = "MAIL";
            }
            VectorBuffers a = ArrowLayout.ofStrings(arena, values);
            MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
            Bitmap.fill(active, n, false);
            Bitmap.set(active, 70); // only block 1 is live
            MemorySegment out = ArrowLayout.allocateBitmap(arena, n);
            Bitmap.fill(out, n, true);
            StringCompareKernels.compareScalar(a, "MAIL".getBytes(StandardCharsets.UTF_8), CompareOp.EQ, active, out);
            assertEquals(64, Bitmap.popcount(out, n), "only the live block was compared");
            assertTrue(Bitmap.isSet(out, 70));
            assertFalse(Bitmap.isSet(out, 0));
            assertFalse(Bitmap.isSet(out, 199));
            VectorBuffers b = dictionaryEncoded(arena, values);
            Bitmap.fill(out, n, true);
            StringCompareKernels.compare(a, b, CompareOp.EQ, active, out);
            assertEquals(64, Bitmap.popcount(out, n));
            Bitmap.fill(out, n, true);
            StringCompareKernels.compareScalar(b, "MAIL".getBytes(StandardCharsets.UTF_8), CompareOp.EQ, active, out);
            assertEquals(64, Bitmap.popcount(out, n));
        }
    }
}
