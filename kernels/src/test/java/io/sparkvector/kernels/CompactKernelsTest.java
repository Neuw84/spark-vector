package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import io.sparkvector.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;

import static io.sparkvector.kernels.TestData.assertBitmapEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class CompactKernelsTest {

    /** Selection patterns: random, all, none, sparse, dense, alternating. */
    private static MemorySegment selection(Arena arena, Random rnd, int n,
            int pattern) {
        MemorySegment sel = ArrowLayout.allocateBitmap(arena, n);
        for (int i = 0; i < n; i++) {
            boolean s = switch (pattern) {
                case 0 -> rnd.nextBoolean();
                case 1 -> true;
                case 2 -> false;
                case 3 -> rnd.nextInt(50) == 0;
                case 4 -> rnd.nextInt(50) != 0;
                default -> (i & 1) == 0;
            };
            if (s) {
                Bitmap.set(sel, i);
            }
        }
        return sel;
    }

    private static void checkFixed(Arena arena, VectorBuffers in, MemorySegment sel) {
        int n = in.length();
        int outCount = CompactKernels.selectedCount(sel, n);
        assertEquals(ScalarReference.selectedCount(sel, n), outCount);
        VecType t = in.type();
        MemorySegment expData = t == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, outCount) : ArrowLayout.allocateData(arena, t, outCount);
        MemorySegment actData = t == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, outCount) : ArrowLayout.allocateData(arena, t, outCount);
        MemorySegment expValid = in.hasNulls() ? ArrowLayout.allocateBitmap(arena, outCount) : null;
        MemorySegment actValid = in.hasNulls() ? ArrowLayout.allocateBitmap(arena, outCount) : null;
        actData.fill((byte) 0xEE);
        ScalarReference.compactFixed(in, sel, expData, expValid);
        CompactKernels.compactFixed(in, sel, outCount, actData, actValid);

        for (int o = 0; o < outCount; o++) {
            switch (t) {
                case INT32 -> assertEquals(expData.get(VectorBuffers.LE_INT, (long) o << 2),
                        actData.get(VectorBuffers.LE_INT, (long) o << 2), t + " out " + o);
                case INT64 -> assertEquals(expData.get(VectorBuffers.LE_LONG, (long) o << 3),
                        actData.get(VectorBuffers.LE_LONG, (long) o << 3), t + " out " + o);
                case FLOAT64 -> assertEquals(Double.doubleToRawLongBits(expData.get(VectorBuffers.LE_DOUBLE, (long) o << 3)), Double.doubleToRawLongBits(actData.get(VectorBuffers.LE_DOUBLE, (long) o << 3)),
                        t + " out " + o);
                case BOOL -> assertEquals(Bitmap.isSet(expData, o), Bitmap.isSet(actData, o), "bool out " + o);
                case DECIMAL128 -> {
                    assertEquals(expData.get(VectorBuffers.LE_LONG, (long) o << 4),
                            actData.get(VectorBuffers.LE_LONG, (long) o << 4), t + " lo out " + o);
                    assertEquals(expData.get(VectorBuffers.LE_LONG, ((long) o << 4) + 8),
                            actData.get(VectorBuffers.LE_LONG, ((long) o << 4) + 8), t + " hi out " + o);
                }
                default -> throw new IllegalStateException();
            }
        }
        if (expValid != null) {
            assertBitmapEquals(expValid, actValid, outCount, t + " validity");
        }
    }

    @Test
    void fixedWidthAndBoolCompactionMatchesReference() {
        Random rnd = new Random(11);
        for (int n : TestData.LENGTHS) {
            for (int pattern = 0; pattern < 6; pattern++) {
                for (boolean withNulls : new boolean[] {false, true}) {
                    try (Arena arena = Arena.ofConfined()) {
                        boolean[] nulls = withNulls ? TestData.nulls(rnd, n, 0.25) : null;
                        MemorySegment sel = selection(arena, rnd, n, pattern);
                        checkFixed(arena, TestData.ints(arena, rnd, n, nulls), sel);
                        checkFixed(arena, TestData.longs(arena, rnd, n, nulls), sel);
                        checkFixed(arena, TestData.doubles(arena, rnd, n, nulls), sel);
                        checkFixed(arena, TestData.decimal128s(arena, rnd, n, nulls), sel);
                        boolean[] bools = new boolean[n];
                        for (int i = 0; i < n; i++) {
                            bools[i] = rnd.nextBoolean();
                        }
                        checkFixed(arena, ArrowLayout.ofBooleans(arena, bools, nulls), sel);
                    }
                }
            }
        }
    }

    @Test
    void compressIsThePathWhereverThePlatformHasIt() {
        // #283, decision 2: the shuffle table stands in for compress only where compress is not native,
        // and only for species it can index (up to 8 lanes).
        for (int lanes : new int[] {2, 4, 8}) {
            assertEquals(!Platform.NATIVE_COMPRESS, CompactKernels.usesTable(lanes), "lanes " + lanes);
        }
        assertFalse(CompactKernels.usesTable(16), "no table for 16 lanes");
    }

    @Test
    void compactionIntoTightlySizedOutputDoesNotOverrun() {
        // Output segment exactly outCount * width bytes: the masked tail store must be used.
        Random rnd = new Random(12);
        try (Arena arena = Arena.ofConfined()) {
            int n = 1000;
            VectorBuffers in = TestData.ints(arena, rnd, n, null);
            MemorySegment sel = selection(arena, rnd, n, 0);
            int outCount = CompactKernels.selectedCount(sel, n);
            MemorySegment tight = arena.allocate((long) outCount * 4L, 4);
            MemorySegment exp = ArrowLayout.allocateData(arena, VecType.INT32, outCount);
            ScalarReference.compactFixed(in, sel, exp, null);
            CompactKernels.compactFixed(in, sel, outCount, tight, null);
            for (int o = 0; o < outCount; o++) {
                assertEquals(exp.get(VectorBuffers.LE_INT, (long) o << 2), tight.get(VectorBuffers.LE_INT, (long) o << 2));
            }
        }
    }

    @Test
    void utf8CompactionMatchesReference() {
        Random rnd = new Random(13);
        for (int n : TestData.LENGTHS) {
            for (int pattern = 0; pattern < 6; pattern++) {
                try (Arena arena = Arena.ofConfined()) {
                    String[] values = new String[n];
                    for (int i = 0; i < n; i++) {
                        values[i] = rnd.nextInt(5) == 0 ? null : "v"
                                + rnd.nextInt(1000)
                                + (rnd.nextBoolean() ? "ü" : "");
                    }
                    VectorBuffers in = ArrowLayout.ofStrings(arena, values);
                    MemorySegment sel = selection(arena, rnd, n, pattern);
                    int outCount = CompactKernels.selectedCount(sel, n);
                    long bytes = CompactKernels.selectedUtf8Bytes(in, sel);
                    assertEquals(ScalarReference.selectedUtf8Bytes(in, sel), bytes);

                    MemorySegment expOff = ArrowLayout.allocateOffsets(arena, outCount);
                    MemorySegment actOff = ArrowLayout.allocateOffsets(arena, outCount);
                    MemorySegment expData = ArrowLayout.allocateBytes(arena, bytes);
                    MemorySegment actData = ArrowLayout.allocateBytes(arena, bytes);
                    MemorySegment expValid = in.hasNulls() ? ArrowLayout.allocateBitmap(arena, outCount) : null;
                    MemorySegment actValid = in.hasNulls() ? ArrowLayout.allocateBitmap(arena, outCount) : null;
                    ScalarReference.compactUtf8(in, sel, expOff, expData, expValid);
                    CompactKernels.compactUtf8(in, sel, outCount, actOff, actData, actValid);

                    VectorBuffers exp = SegmentVectorBuffers.utf8(outCount, expValid, expOff, expData);
                    VectorBuffers act = SegmentVectorBuffers.utf8(outCount, actValid, actOff, actData);
                    for (int o = 0; o < outCount; o++) {
                        assertEquals(exp.isNull(o), act.isNull(o), "null " + o);
                        if (!exp.isNull(o)) {
                            assertEquals(exp.getString(o), act.getString(o), "value " + o);
                        }
                    }
                }
            }
        }
    }

    @Test
    void dictionaryIndicesCompactLikeInt32() {
        Random rnd = new Random(14);
        try (Arena arena = Arena.ofConfined()) {
            int n = 333;
            VectorBuffers dict = ArrowLayout.ofStrings(arena,
                    new String[] {"A", "F", "N", "O", "R"});
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) {
                idx[i] = rnd.nextInt(5);
            }
            boolean[] nulls = TestData.nulls(rnd, n, 0.1);
            SegmentVectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
            VectorBuffers in = SegmentVectorBuffers.dictionaryUtf8(n, indices.validity(), indices.data(), dict);
            MemorySegment sel = selection(arena, rnd, n, 0);
            int outCount = CompactKernels.selectedCount(sel, n);
            MemorySegment outIdx = ArrowLayout.allocateData(arena, VecType.INT32, outCount);
            MemorySegment outValid = ArrowLayout.allocateBitmap(arena, outCount);
            CompactKernels.compactFixed(in, sel, outCount, outIdx, outValid);
            VectorBuffers out = SegmentVectorBuffers.dictionaryUtf8(outCount, outValid, outIdx, dict);
            int o = 0;
            for (int i = 0; i < n; i++) {
                if (Bitmap.isSet(sel, i)) {
                    assertEquals(in.isNull(i), out.isNull(o));
                    if (!in.isNull(i)) {
                        assertEquals(in.getString(i), out.getString(o));
                    }
                    o++;
                }
            }
            assertEquals(outCount, o);
        }
    }
}
