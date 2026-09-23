package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import java.util.function.Function;

import io.sparkvector.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static io.sparkvector.kernels.TestData.assertBitmapEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CompareKernelsTest {

    interface Gen {
        SegmentVectorBuffers make(Arena arena, Random rnd, int n);
    }

    private static final Gen[] GENS = {
        (a, r, n) -> TestData.ints(a, r, n, null), (a, r, n) -> TestData.longs(a, r, n, null), (a, r, n) -> TestData.doubles(a, r, n, null),
        // Wide decimals: random 128-bit values plus the extremes, so both limbs are exercised (#258).
        (a, r, n) -> TestData.decimal128s(a, r, n, null),};

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static final Function<Random, Number>[] SCALARS = new Function[] {
        (Function<Random, Number>) r -> r.nextInt(-20, 20),
        (Function<Random, Number>) r -> r.nextLong(-20, 20),
        (Function<Random, Number>) r -> switch (r.nextInt(6)) {
            case 0 -> Double.NaN;
            case 1 -> Double.POSITIVE_INFINITY;
            case 2 -> -0.0;
            default -> r.nextInt(-5, 5) + 0.5;
        },
        (Function<Random, Number>) r -> switch (r.nextInt(5)) {
            case 0 -> java.math.BigInteger.ZERO;
            case 1 -> java.math.BigInteger.ONE.shiftLeft(127).negate(); // -2^127
            case 2 -> java.math.BigInteger.TEN.pow(38).subtract(java.math.BigInteger.ONE);
            case 3 -> java.math.BigInteger.valueOf(Long.MIN_VALUE); // the low limb alone
            default -> TestData.randomDecimal128(r);
        },
    };

    @ParameterizedTest
    @EnumSource(CompareOp.class)
    void columnVersusScalarMatchesReference(CompareOp op) {
        Random rnd = new Random(op.name().hashCode() * 31L);
        for (int t = 0; t < GENS.length; t++) {
            for (int n : TestData.LENGTHS) {
                for (int rep = 0; rep < 3; rep++) {
                    try (Arena arena = Arena.ofConfined()) {
                        VectorBuffers a = GENS[t].make(arena, rnd, n);
                        Number s = SCALARS[t].apply(rnd);
                        MemorySegment expected = ArrowLayout.allocateBitmap(arena, n);
                        MemorySegment actual = ArrowLayout.allocateBitmap(arena, n);
                        // Poison the output to prove every bit is written.
                        actual.fill((byte) 0xAA);
                        ScalarReference.compareScalar(a, s, op, expected);
                        CompareKernels.compareScalar(a, s, op, actual);
                        assertBitmapEquals(expected, actual, n, a.type() + " " + op + " " + s);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(CompareOp.class)
    void columnVersusColumnMatchesReference(CompareOp op) {
        Random rnd = new Random(1000L + op.name().hashCode());
        for (Gen gen : GENS) {
            for (int n : TestData.LENGTHS) {
                try (Arena arena = Arena.ofConfined()) {
                    VectorBuffers a = gen.make(arena, rnd, n);
                    VectorBuffers b = gen.make(arena, rnd, n);
                    MemorySegment expected = ArrowLayout.allocateBitmap(arena, n);
                    MemorySegment actual = ArrowLayout.allocateBitmap(arena, n);
                    actual.fill((byte) 0x55);
                    ScalarReference.compare(a, b, op, expected);
                    CompareKernels.compare(a, b, op, actual);
                    assertBitmapEquals(expected, actual, n, a.type() + " " + op);
                }
            }
        }
    }

    @Test
    void nanSafeSemanticsMatchSpark() {
        try (Arena arena = Arena.ofConfined()) {
            double nan = Double.NaN;
            VectorBuffers a = ArrowLayout.ofDoubles(
                    arena,
                    new double[] {nan, 1.0, nan, -0.0, Double.POSITIVE_INFINITY},
                    null);
            VectorBuffers b = ArrowLayout.ofDoubles(
                    arena,
                    new double[] {nan, nan, 1.0, 0.0, nan},
                    null);
            MemorySegment out = ArrowLayout.allocateBitmap(arena, 5);

            CompareKernels.compare(a, b, CompareOp.EQ, out);
            assertTrue(Bitmap.isSet(out, 0), "NaN = NaN");
            assertFalse(Bitmap.isSet(out, 1));
            assertTrue(Bitmap.isSet(out, 3), "-0.0 = 0.0");

            CompareKernels.compare(a, b, CompareOp.LT, out);
            assertTrue(Bitmap.isSet(out, 1), "1.0 < NaN");
            assertFalse(Bitmap.isSet(out, 2), "NaN < 1.0 is false");
            assertTrue(Bitmap.isSet(out, 4), "+Inf < NaN");

            CompareKernels.compare(a, b, CompareOp.GE, out);
            assertTrue(Bitmap.isSet(out, 0));
            assertTrue(Bitmap.isSet(out, 2), "NaN >= 1.0");
            assertFalse(Bitmap.isSet(out, 4), "+Inf >= NaN is false");

            CompareKernels.compareScalar(a, nan, CompareOp.EQ, out);
            assertTrue(Bitmap.isSet(out, 0));
            assertFalse(Bitmap.isSet(out, 1));
            CompareKernels.compareScalar(a, nan, CompareOp.LE, out);
            assertTrue(Bitmap.allSet(out, 5), "everything <= NaN");
            CompareKernels.compareScalar(a, nan, CompareOp.GT, out);
            assertTrue(Bitmap.noneSet(out, 5), "nothing > NaN");
        }
    }

    @Test
    void resultOnlyDependsOnValuesNotValidity() {
        // Null lanes must not blow up and must produce the same bits as if they were valid.
        Random rnd = new Random(9);
        try (Arena arena = Arena.ofConfined()) {
            int n = 200;
            SegmentVectorBuffers withNulls = TestData.ints(arena, rnd, n, TestData.nulls(rnd, n, 0.3));
            VectorBuffers noNulls = withNulls.withValidity(null);
            MemorySegment o1 = ArrowLayout.allocateBitmap(arena, n);
            MemorySegment o2 = ArrowLayout.allocateBitmap(arena, n);
            CompareKernels.compareScalar(withNulls, 3, CompareOp.GT, o1);
            CompareKernels.compareScalar(noNulls, 3, CompareOp.GT, o2);
            assertBitmapEquals(o1, o2, n, "validity independence");
        }
    }

    @Test
    void activeBitmapSkipsBlocksAndClearsTheirBits() {
        Random rnd = new Random(21);
        int n = 1000;
        try (Arena arena = Arena.ofConfined()) {
            VectorBuffers a = TestData.doubles(arena, rnd, n, null);
            VectorBuffers b = TestData.doubles(arena, rnd, n, null);
            // Active: whole blocks on or off, plus one sparse block.
            MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
            for (int i = 0; i < n; i++) {
                int w = i >> 6;
                boolean on = w % 3 != 1 && (w != 4 || i % 17 == 0);
                if (on) {
                    Bitmap.set(active, i);
                }
            }
            for (CompareOp op : CompareOp.values()) {
                MemorySegment full = ArrowLayout.allocateBitmap(arena, n);
                MemorySegment skipped = ArrowLayout.allocateBitmap(arena, n);
                CompareKernels.compareScalar(a, 0.25, op, full);
                CompareKernels.compareScalar(a, 0.25, op, active, skipped);
                check(full, skipped, active, n, op + " scalar");
                CompareKernels.compare(a, b, op, full);
                CompareKernels.compare(a, b, op, active, skipped);
                check(full, skipped, active, n, op + " column");
            }
        }
    }

    private static void check(MemorySegment full, MemorySegment skipped, MemorySegment active,
            int n, String what) {
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            long act = Bitmap.wordAt(active, w, n);
            long expected = act == 0L ? 0L : Bitmap.wordAt(full, w, n);
            assertEquals(expected, Bitmap.wordAt(skipped, w, n), what + " word " + w);
        }
    }
}
