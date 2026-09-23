package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import io.sparkvector.kernels.BitKernels.BitOp;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** {@link BitKernels} against Java's operators, which are Spark's definitions. */
class BitKernelsTest {

    private static long ref(BitOp op, long x, long y,
                            boolean isInt) {
        return switch (op) {
            case AND -> x & y;
            case OR -> x | y;
            case XOR -> x ^ y;
            case SHL -> isInt ? (int) x << (int) y : x << (int) y;
            case SHR -> isInt ? (int) x >> (int) y : x >> (int) y;
            case USHR -> isInt ? (int) x >>> (int) y : x >>> (int) y;
        };
    }

    @Test
    void binaryColumnColumnAndColumnScalar() {
        try (Arena arena = Arena.ofConfined()) {
            Random rnd = new Random(21);
            int n = 1003; // not a lane multiple: exercises the masked tail
            int[] xi = new int[n], yi = new int[n], amounts = new int[n];
            long[] xl = new long[n], yl = new long[n];
            for (int i = 0; i < n; i++) {
                xi[i] = rnd.nextInt();
                yi[i] = rnd.nextInt();
                xl[i] = rnd.nextLong();
                yl[i] = rnd.nextLong();
                amounts[i] = switch (i % 8) {
                            case 0 -> 0;
                            case 1 -> 31;
                            case 2 -> 32;
                            case 3 -> 63;
                            case 4 -> 64;
                            case 5 -> -1;
                            case 6 -> 100;
                            default -> rnd.nextInt(70) - 5;
                        };
            }
            xi[0] = Integer.MIN_VALUE;
            xi[1] = -1;
            xl[0] = Long.MIN_VALUE;
            xl[1] = -1L;
            VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null), bi = ArrowLayout.ofInts(arena, yi, null), am = ArrowLayout.ofInts(arena, amounts, null);
            VectorBuffers al = ArrowLayout.ofLongs(arena, xl, null), bl = ArrowLayout.ofLongs(arena, yl, null);
            MemorySegment outi = ArrowLayout.allocateData(arena, VecType.INT32, n);
            MemorySegment outl = ArrowLayout.allocateData(arena, VecType.INT64, n);
            for (BitOp op : BitOp.values()) {
                VectorBuffers rightI = op.isShift() ? am : bi;
                VectorBuffers rightL = op.isShift() ? am : bl;
                BitKernels.binary(op, ai, rightI, outi);
                BitKernels.binary(op, al, rightL, outl);
                for (int i = 0; i < n; i++) {
                    long yI = op.isShift() ? amounts[i] : yi[i];
                    long yL = op.isShift() ? amounts[i] : yl[i];
                    assertEquals((int) ref(op, xi[i], yI, true), outi.getAtIndex(VectorBuffers.LE_INT, i),
                            op + " int row " + i);
                    assertEquals(ref(op, xl[i], yL, false), outl.getAtIndex(VectorBuffers.LE_LONG, i), op + " long row " + i);
                }
                for (long s : new long[] {0, 1, 5, 31, 32, 33,
                        63, 64, -1, 0xF0F0F0F0L, -6148914691236517206L}) {
                    BitKernels.binaryScalar(op, ai, s, outi);
                    BitKernels.binaryScalar(op, al, s, outl);
                    for (int i = 0; i < n; i++) {
                        assertEquals((int) ref(op, xi[i], (int) s, true), outi.getAtIndex(VectorBuffers.LE_INT, i),
                                op + " int scalar " + s + " row " + i);
                        assertEquals(
                                ref(op, xl[i],
                                        op.isShift() ? (int) s : s, false),
                                outl.getAtIndex(VectorBuffers.LE_LONG, i),
                                op + " long scalar " + s + " row " + i);
                    }
                }
                if (op.isShift()) {
                    BitKernels.shiftScalarValue(op, VecType.INT32, -12345, am, outi);
                    BitKernels.shiftScalarValue(op, VecType.INT64, Long.MIN_VALUE + 7, am, outl);
                    for (int i = 0; i < n; i++) {
                        assertEquals((int) ref(op, -12345, amounts[i], true), outi.getAtIndex(VectorBuffers.LE_INT, i),
                                op + " literal value int row " + i);
                        assertEquals(ref(op, Long.MIN_VALUE + 7, amounts[i], false),
                                outl.getAtIndex(VectorBuffers.LE_LONG, i), op + " literal value long row " + i);
                    }
                }
            }
            // >>> on INT32 is computed in 32 bits: -1 >>> 1 is Integer.MAX_VALUE, not 0x7FFF_FFFF_FFFF_FFFF narrowed.
            BitKernels.binaryScalar(BitOp.USHR, ai, 1, outi);
            assertEquals(Integer.MAX_VALUE, outi.getAtIndex(VectorBuffers.LE_INT, 1));
            assertEquals(1 << 30, outi.getAtIndex(VectorBuffers.LE_INT, 0));
            // Shift amounts wrap by the width: 32 on an int is 0, 33 is 1; 64 on a long is 0.
            BitKernels.binaryScalar(BitOp.SHL, ai, 33, outi);
            assertEquals(-2, outi.getAtIndex(VectorBuffers.LE_INT, 1));
            BitKernels.binaryScalar(BitOp.SHL, al, 64, outl);
            assertEquals(-1L, outl.getAtIndex(VectorBuffers.LE_LONG, 1));
        }
    }

    @Test
    void notAndBitCount() {
        try (Arena arena = Arena.ofConfined()) {
            int[] xi = {
                0,
                1,
                -1,
                Integer.MIN_VALUE,
                Integer.MAX_VALUE,
                0x0F0F0F0F,
                12345,
                -12345,
                7,
                8,
                9,
                10,
                11
            };
            long[] xl = {
                0L,
                1L,
                -1L,
                Long.MIN_VALUE,
                Long.MAX_VALUE,
                0x0F0F0F0F0F0F0F0FL,
                12345L,
                -12345L,
                7,
                8,
                9,
                10,
                11
            };
            VectorBuffers ai = ArrowLayout.ofInts(arena, xi, null), al = ArrowLayout.ofLongs(arena, xl, null);
            MemorySegment outi = ArrowLayout.allocateData(arena, VecType.INT32, xi.length);
            MemorySegment outl = ArrowLayout.allocateData(arena, VecType.INT64, xl.length);
            BitKernels.not(ai, outi);
            BitKernels.not(al, outl);
            for (int i = 0; i < xi.length; i++) {
                assertEquals(~xi[i], outi.getAtIndex(VectorBuffers.LE_INT, i));
                assertEquals(~xl[i], outl.getAtIndex(VectorBuffers.LE_LONG, i));
            }
            MemorySegment counts = ArrowLayout.allocateData(arena, VecType.INT32, xi.length);
            BitKernels.bitCount(ai, counts);
            for (int i = 0; i < xi.length; i++) {
                assertEquals(Long.bitCount(xi[i]), counts.getAtIndex(VectorBuffers.LE_INT, i), "bit_count(int " + xi[i] + ")");
            }
            assertEquals(64, counts.getAtIndex(VectorBuffers.LE_INT, 2), "Spark widens an int to a long first: bit_count(-1) is 64");
            assertEquals(33, counts.getAtIndex(VectorBuffers.LE_INT, 3), "Integer.MIN_VALUE widened has 33 set bits");
            BitKernels.bitCount(al, counts);
            for (int i = 0; i < xl.length; i++) {
                assertEquals(Long.bitCount(xl[i]), counts.getAtIndex(VectorBuffers.LE_INT, i), "bit_count(long " + xl[i] + ")");
            }
        }
    }
}
