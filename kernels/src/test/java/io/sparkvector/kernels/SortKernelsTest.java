package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import io.sparkvector.kernels.reference.SortReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The permutation produced by {@link SortKernels} must equal the stable
 * comparator sort in {@link SortReference} for every key type, direction, null
 * ordering and multi-key combination; the gathers must reproduce the
 * reference's rows.
 */
class SortKernelsTest {

    private static final int[] SIZES = {0, 1, 2, 7, 64, 65,
            1000, 4097};

    private static void check(VectorBuffers[] keys, boolean[] ascending, boolean[] nullsFirst,
            int n, String what) {
        int[] expected = SortReference.sortIndices(keys, ascending, nullsFirst, n);
        int[] actual = SortKernels.sortIndices(keys, ascending, nullsFirst, n);
        assertArrayEquals(expected, actual, what);
    }

    private static void checkAllOrders(VectorBuffers key, int n, String what) {
        for (boolean asc : new boolean[] {true, false}) {
            for (boolean nf : new boolean[] {false, true}) {
                check(new VectorBuffers[] {key}, new boolean[] {asc}, new boolean[] {nf},
                        n, what + " asc=" + asc + " nullsFirst=" + nf);
            }
        }
    }

    @Test
    void fixedWidthKeysMatchReference() {
        Random rnd = new Random(7);
        try (Arena arena = Arena.ofConfined()) {
            for (int n : SIZES) {
                for (double nullFraction : new double[] {0.0, 0.1}) {
                    boolean[] nulls = TestData.nulls(rnd, n, nullFraction);
                    checkAllOrders(TestData.ints(arena, rnd, n, nulls), n, "int32 n=" + n);
                    checkAllOrders(TestData.longs(arena, rnd, n, nulls), n, "int64 n=" + n);
                    checkAllOrders(TestData.doubles(arena, rnd, n, nulls), n, "float64 n=" + n);
                    boolean[] bools = new boolean[n];
                    for (int i = 0; i < n; i++) {
                        bools[i] = rnd.nextBoolean();
                    }
                    checkAllOrders(ArrowLayout.ofBooleans(arena, bools, nulls), n, "bool n=" + n);
                }
            }
        }
    }

    @Test
    void doublesUseSparkTotalOrder() {
        try (Arena arena = Arena.ofConfined()) {
            double[] values = {
                1.0,
                -0.0,
                Double.NaN,
                0.0,
                Double.NEGATIVE_INFINITY,
                -1.5,
                Double.POSITIVE_INFINITY,
                Double.MIN_VALUE,
                -Double.MIN_VALUE,
                Double.longBitsToDouble(0x7ff8000000000001L),
                2.5,
                -2.5
            };
            VectorBuffers key = ArrowLayout.ofDoubles(arena, values, null);
            checkAllOrders(key, values.length, "special doubles");
            int[] asc = SortKernels.sortIndices(new VectorBuffers[] {key}, new boolean[] {true}, new boolean[] {true},
                    values.length);
            // NaNs (both payloads) are the greatest and equal; -0.0 and 0.0 are equal, so stability keeps
            // -0.0 (index 1) before 0.0 (index 3).
            assertTrue(Double.isNaN(values[asc[values.length - 1]]) && Double.isNaN(values[asc[values.length - 2]]));
            assertEquals(2, asc[values.length - 2], "first NaN in input order");
            int posNegZero = indexOf(asc, 1);
            int posZero = indexOf(asc, 3);
            assertEquals(posNegZero + 1, posZero, "-0.0 and 0.0 adjacent, input order kept");
        }
    }

    @Test
    void extremeIntegersAndLongs() {
        try (Arena arena = Arena.ofConfined()) {
            int[] ints = {Integer.MAX_VALUE, Integer.MIN_VALUE, 0, -1, 1,
                    Integer.MIN_VALUE + 1, Integer.MAX_VALUE - 1};
            checkAllOrders(ArrowLayout.ofInts(arena, ints, null), ints.length, "extreme ints");
            long[] longs = {
                Long.MAX_VALUE,
                Long.MIN_VALUE,
                0L,
                -1L,
                1L,
                1L << 32,
                -(1L << 32),
                (1L << 32) - 1,
                Integer.MIN_VALUE,
                Integer.MAX_VALUE
            };
            checkAllOrders(ArrowLayout.ofLongs(arena, longs, null), longs.length, "extreme longs");
        }
    }

    @Test
    void shortStringsTakeThePrefixPath() {
        try (Arena arena = Arena.ofConfined()) {
            // All at most 8 bytes: prefixes, shared prefixes, empty, non-ASCII (bytes above 0x7F must
            // order as unsigned), null.
            String[] values = {
                "b",
                "a",
                "ab",
                "",
                "abc",
                "A",
                "\u00e9",
                "aa",
                "abcdefgh",
                "abcdefg",
                null,
                "a",
                "z",
                "\u00e8"
            };
            VectorBuffers key = ArrowLayout.ofStrings(arena, values);
            checkAllOrders(key, values.length, "short strings");
        }
    }

    @Test
    void longStringsTakeTheRankPath() {
        Random rnd = new Random(11);
        try (Arena arena = Arena.ofConfined()) {
            for (int n : SIZES) {
                String[] values = new String[n];
                for (int i = 0; i < n; i++) {
                    if (rnd.nextInt(10) == 0) {
                        values[i] = null;
                    } else {
                        int len = rnd.nextInt(20);
                        StringBuilder sb = new StringBuilder();
                        for (int j = 0; j < len; j++) {
                            sb.append((char) ('a' + rnd.nextInt(3)));
                        }
                        values[i] = sb.toString();
                    }
                }
                if (n > 0) {
                    values[0] = "a string that is definitely longer than eight bytes"; // the chunked path (#377)
                }
                checkAllOrders(ArrowLayout.ofStrings(arena, values), n, "long strings n=" + n);
                if (n > 1) {
                    values[1] = "a string longer than sixty-four bytes forces the merge-sort rank path, still exact"; // 84 bytes
                    checkAllOrders(ArrowLayout.ofStrings(arena, values), n, "very long strings n=" + n);
                }
            }
        }
    }

    @Test
    void chunkedStringsAtTheBoundaries() {
        // Lengths around the 8-byte chunks, shared prefixes across a chunk edge, embedded zero bytes
        // (a shorter prefix sorts first, a zero byte after it does not tie with the end), and a
        // partition whose strings are all equal -- q67's window partitioned by its first sort key.
        try (Arena arena = Arena.ofConfined()) {
            String[] values = {
                "abcdefgh",
                "abcdefghi",
                "abcdefg",
                "abcdefgh\u0000",
                "abcdefghijklmnop",
                "abcdefghijklmnopq",
                "abcdefghijklmno",
                "ABCDEFGHIJKLMNOPQRSTUVWX",
                "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "",
                null,
                "z",
                "abcdefgh",
                "\u00ff\u00ff\u00ff\u00ff\u00ff\u00ff\u00ff\u00ff\u00ff",
                "日本語日本語",
                "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXY"
            };
            checkAllOrders(ArrowLayout.ofStrings(arena, values), values.length, "chunk boundaries");
            Random rnd = new Random(377);
            for (int n : SIZES) {
                String[] same = new String[n];
                for (int i = 0; i < n; i++) {
                    same[i] = rnd.nextInt(50) == 0 ? null : "Electronics"; // 11 bytes: two chunks
                }
                checkAllOrders(ArrowLayout.ofStrings(arena, same), n, "all equal n=" + n);
            }
        }
    }

    @Test
    void dictionaryEncodedStringsSortByValue() {
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers dict = ArrowLayout.ofStrings(arena, new String[] {"R", "A", "N", "a much longer dictionary value"});
            int[] indices = {0, 1, 2, 1, 0, 3,
                    2, 3};
            boolean[] nulls = {false, false, false, true, false, false,
                    false, false};
            SegmentVectorBuffers idx = ArrowLayout.ofInts(arena, indices, nulls);
            VectorBuffers key = SegmentVectorBuffers.dictionaryUtf8(indices.length, idx.validity(), idx.data(), dict);
            checkAllOrders(key, indices.length, "dictionary strings");
            // A plain copy of the same values must give the same permutation.
            String[] plain = new String[indices.length];
            for (int i = 0; i < plain.length; i++) {
                plain[i] = nulls[i] ? null : dict.getString(indices[i]);
            }
            VectorBuffers plainKey = ArrowLayout.ofStrings(arena, plain);
            assertArrayEquals(
                    SortKernels.sortIndices(new VectorBuffers[] {plainKey}, new boolean[] {true}, new boolean[] {true},
                            plain.length),
                    SortKernels.sortIndices(new VectorBuffers[] {key}, new boolean[] {true}, new boolean[] {true},
                            plain.length));
        }
    }

    @Test
    void radixPassesMatchTheReferenceOnLargeAndSkewedInputs() {
        Random rnd = new Random(11);
        try (Arena arena = Arena.ofConfined()) {
            int n = 100_003; // an awkward length, several digit passes per key deep
            boolean[] nulls = TestData.nulls(rnd, n, 0.02);
            checkAllOrders(TestData.ints(arena, rnd, n, nulls), n, "int32 large");
            checkAllOrders(TestData.longs(arena, rnd, n, nulls), n, "int64 large");
            checkAllOrders(TestData.doubles(arena, rnd, n, nulls), n, "float64 large");
            // Low cardinality: most digits of the key have a single bucket and are skipped; ties keep
            // their input order (the reference is stable).
            int[] small = new int[n];
            long[] smallLong = new long[n];
            for (int i = 0; i < n; i++) {
                small[i] = rnd.nextInt(5) - 2;
                smallLong[i] = (rnd.nextInt(3) - 1) * 4_000_000_000L; // straddles the 32-bit halves
            }
            checkAllOrders(ArrowLayout.ofInts(arena, small, nulls), n, "int32 low cardinality");
            checkAllOrders(ArrowLayout.ofLongs(arena, smallLong, null), n, "int64 low cardinality");
            // Presorted and reverse inputs, every key equal, and a single distinct value with nulls.
            int[] sorted = new int[n];
            int[] reverse = new int[n];
            int[] constant = new int[n];
            for (int i = 0; i < n; i++) {
                sorted[i] = i - n / 2;
                reverse[i] = n / 2 - i;
                constant[i] = 42;
            }
            checkAllOrders(ArrowLayout.ofInts(arena, sorted, null), n, "int32 presorted");
            checkAllOrders(ArrowLayout.ofInts(arena, reverse, null), n, "int32 reverse");
            checkAllOrders(ArrowLayout.ofInts(arena, constant, nulls), n, "int32 constant");
        }
    }

    @Test
    void equalKeysKeepInputOrderAcrossEveryPass() {
        Random rnd = new Random(13);
        try (Arena arena = Arena.ofConfined()) {
            int n = 5000;
            // Keys drawn from a small set so that every pass sees long runs of ties; the sort must
            // return them in input order, which is what makes the later (more significant) passes
            // correct at all.
            long[] longs = new long[n];
            double[] doubles = new double[n];
            String[] strings = new String[n];
            for (int i = 0; i < n; i++) {
                longs[i] = (rnd.nextInt(4) - 2) * 3_000_000_000L;
                doubles[i] = new double[] {-0.0, 0.0, 1.5, Double.NaN,
                        -2.25}[rnd.nextInt(5)];
                strings[i] = new String[] {"a", "ab", "b", "", "abcdefgh", null}[rnd.nextInt(6)];
            }
            VectorBuffers[] keys = {ArrowLayout.ofLongs(arena, longs, null), ArrowLayout.ofDoubles(arena, doubles, null),
                    ArrowLayout.ofStrings(arena, strings)};
            for (VectorBuffers key : keys) {
                for (boolean asc : new boolean[] {true, false}) {
                    int[] order = SortKernels.sortIndices(new VectorBuffers[] {key}, new boolean[] {asc}, new boolean[] {true},
                            n);
                    for (int j = 1; j < n; j++) {
                        int a = order[j - 1];
                        int b = order[j];
                        if (SortReference.compareKey(key, a, b, !asc, true) == 0) {
                            assertTrue(a < b, "ties in input order at "
                                    + j
                                    + " ("
                                    + key.type()
                                    + ", asc="
                                    + asc
                                    + ")");
                        }
                    }
                }
            }
        }
    }

    @Test
    void multipleKeysWithMixedDirections() {
        Random rnd = new Random(3);
        try (Arena arena = Arena.ofConfined()) {
            int n = 2000;
            // Low-cardinality leading keys so later keys decide many ties.
            String[] flags = new String[n];
            int[] small = new int[n];
            for (int i = 0; i < n; i++) {
                flags[i] = rnd.nextInt(12) == 0 ? null : String.valueOf((char) ('A' + rnd.nextInt(3)));
                small[i] = rnd.nextInt(4);
            }
            VectorBuffers k1 = ArrowLayout.ofStrings(arena, flags);
            VectorBuffers k2 = ArrowLayout.ofInts(arena, small, TestData.nulls(rnd, n, 0.2));
            VectorBuffers k3 = TestData.doubles(arena, rnd, n, TestData.nulls(rnd, n, 0.05));
            VectorBuffers k4 = TestData.longs(arena, rnd, n, null);
            VectorBuffers[] keys = {k1, k2, k3, k4};
            for (int mask = 0; mask < 16; mask++) {
                boolean[] asc = {(mask & 1) == 0, (mask & 2) == 0, (mask & 4) == 0,
                        (mask & 8) == 0};
                boolean[] nf = {(mask & 8) == 0, (mask & 4) == 0, (mask & 2) == 0,
                        (mask & 1) == 0};
                check(keys, asc, nf, n, "4 keys mask=" + mask);
            }
        }
    }

    @Test
    void gathersReproduceTheSortedRows() {
        Random rnd = new Random(5);
        try (Arena arena = Arena.ofConfined()) {
            int n = 1500;
            boolean[] nulls = TestData.nulls(rnd, n, 0.1);
            VectorBuffers ints = TestData.ints(arena, rnd, n, nulls);
            VectorBuffers doubles = TestData.doubles(arena, rnd, n, nulls);
            boolean[] bools = new boolean[n];
            String[] strings = new String[n];
            for (int i = 0; i < n; i++) {
                bools[i] = rnd.nextBoolean();
                strings[i] = nulls[i] ? null : "s" + rnd.nextInt(500);
            }
            VectorBuffers boolCol = ArrowLayout.ofBooleans(arena, bools, nulls);
            VectorBuffers strCol = ArrowLayout.ofStrings(arena, strings);
            int[] perm = SortKernels.sortIndices(new VectorBuffers[] {ints}, new boolean[] {true}, new boolean[] {true},
                    n);

            // Gather in two output batches and compare element by element with the permutation.
            int split = 700;
            for (int[] range : new int[][] {{0, split}, {split, n}}) {
                int from = range[0];
                int to = range[1];
                int count = to - from;
                MemorySegment d = ArrowLayout.allocateData(arena, VecType.FLOAT64, count);
                MemorySegment v = ArrowLayout.allocateBitmap(arena, count);
                GatherKernels.gatherFixed(doubles, perm, from, to, d, v);
                MemorySegment b = ArrowLayout.allocateBitmap(arena, count);
                MemorySegment bv = ArrowLayout.allocateBitmap(arena, count);
                GatherKernels.gatherFixed(boolCol, perm, from, to, b, bv);
                long bytes = GatherKernels.gatherUtf8Bytes(strCol, perm, from, to);
                MemorySegment off = ArrowLayout.allocateOffsets(arena, count);
                MemorySegment sd = ArrowLayout.allocateBytes(arena, bytes);
                MemorySegment sv = ArrowLayout.allocateBitmap(arena, count);
                GatherKernels.gatherUtf8(strCol, perm, from, to, off, sd,
                        sv);
                VectorBuffers gatheredStrings = SegmentVectorBuffers.utf8(count, sv, off, sd);
                for (int o = 0; o < count; o++) {
                    int row = perm[from + o];
                    assertEquals(!nulls[row], Bitmap.isSet(v, o), "double validity " + o);
                    if (!nulls[row]) {
                        assertEquals(doubles.getDouble(row), d.get(VectorBuffers.LE_DOUBLE, (long) o << 3), "double " + o);
                    }
                    assertEquals(bools[row], Bitmap.isSet(b, o), "bool " + o);
                    assertEquals(!nulls[row], Bitmap.isSet(bv, o), "bool validity " + o);
                    assertEquals(!nulls[row], !gatheredStrings.isNull(o), "string validity " + o);
                    if (!nulls[row]) {
                        assertEquals(strings[row], gatheredStrings.getString(o), "string " + o);
                    }
                }
            }
        }
    }

    @Test
    void materializeAndConcatBuildOneColumnFromChunks() {
        Random rnd = new Random(9);
        try (Arena arena = Arena.ofConfined()) {
            // Three chunks of strings, one dictionary encoded, one with a selection, joined into a plain
            // column whose rows are the selected rows of every chunk in order.
            String[] c1 = {"x", null, "yy", "zzz"};
            VectorBuffers dict = ArrowLayout.ofStrings(arena, new String[] {"R", "A"});
            SegmentVectorBuffers idx = ArrowLayout.ofInts(arena, new int[] {1, 0, 1},
                    new boolean[] {false, true, false});
            VectorBuffers c2 = SegmentVectorBuffers.dictionaryUtf8(3, idx.validity(), idx.data(), dict);
            String[] c3 = {"keep", "drop", "keep2", "drop2", "keep3"};
            MemorySegment sel = ArrowLayout.allocateBitmap(arena, 5);
            Bitmap.set(sel, 0);
            Bitmap.set(sel, 2);
            Bitmap.set(sel, 4);
            ColumnBuilder builder = new ColumnBuilder(arena, VecType.UTF8, 4);
            builder.append(ArrowLayout.ofStrings(arena, c1));
            builder.append(c2);
            builder.append(ArrowLayout.ofStrings(arena, c3), sel, 3);
            VectorBuffers all = builder.view();
            List<String> expected = new ArrayList<>(List.of("x"));
            expected.add(null);
            expected.addAll(List.of("yy", "zzz", "A"));
            expected.add(null);
            expected.addAll(List.of("A", "keep", "keep2", "keep3"));
            assertEquals(10, all.length());
            for (int i = 0; i < 10; i++) {
                assertEquals(expected.get(i) == null, all.isNull(i), "null " + i);
                if (expected.get(i) != null) {
                    assertEquals(expected.get(i), all.getString(i), "value " + i);
                }
            }

            // Fixed width and booleans, with and without nulls.
            int n1 = 100;
            int n2 = 37;
            boolean[] nulls2 = TestData.nulls(rnd, n2, 0.3);
            VectorBuffers d1 = TestData.doubles(arena, rnd, n1, null);
            VectorBuffers d2 = TestData.doubles(arena, rnd, n2, nulls2);
            ColumnBuilder doubles2 = new ColumnBuilder(arena, VecType.FLOAT64, 8);
            doubles2.append(d1);
            doubles2.append(d2);
            VectorBuffers joined = doubles2.view();
            for (int i = 0; i < n1; i++) {
                assertEquals(d1.getDouble(i), joined.getDouble(i));
                assertTrue(!joined.isNull(i));
            }
            for (int i = 0; i < n2; i++) {
                assertEquals(nulls2[i], joined.isNull(n1 + i));
                if (!nulls2[i]) {
                    assertEquals(d2.getDouble(i), joined.getDouble(n1 + i));
                }
            }
        }
    }

    private static int indexOf(int[] a, int value) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] == value) {
                return i;
            }
        }
        return -1;
    }
}
