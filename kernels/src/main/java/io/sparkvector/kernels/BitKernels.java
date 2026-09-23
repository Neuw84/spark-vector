package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;

import static io.sparkvector.kernels.Species.I;
import static io.sparkvector.kernels.Species.L;

/**
 * Bitwise operators over INT32 / INT64 lanes: {@code & | ^ ~}, the three shifts
 * and {@code bit_count}. None of the kernels touches validity; null lanes hold
 * arbitrary values.
 *
 * <p>Semantics are Java's, which is what Spark's {@code BitwiseAnd} / {@code
 * ShiftLeft} / ... compute: a shift amount is masked by the lane width ({@code
 * x << 33} on an int is {@code x << 1}, and a negative amount wraps the same
 * way), the unsigned shift on INT32 is computed in 32 bits (widening to 64
 * would fill the sign differently), and {@code bit_count} is Spark's {@code
 * java.lang.Long.bitCount(v)} on the value <em>widened to a long</em> -- so a
 * negative int counts its 32 sign-extension bits too, exactly as Spark does.
 * The and/or/xor/not kernels are Vector API lanewise operations; the shifts are
 * per-lane loops the JIT vectorises where the hardware has variable shifts.
 */
public final class BitKernels {
    /**
     * Binary operators; the shift amount operand is always INT32 (Spark casts
     * it).
     */
    public enum BitOp {
        AND(VectorOperators.AND),
        OR(VectorOperators.OR),
        XOR(VectorOperators.XOR),
        /** {@code shiftleft}. */
        SHL(null),
        /** {@code shiftright}, arithmetic. */
        SHR(null),
        /** {@code shiftrightunsigned}. */
        USHR(null);

        final VectorOperators.Binary lanewise;

        BitOp(VectorOperators.Binary lanewise) {
            this.lanewise = lanewise;
        }

        public boolean isShift() {
            return lanewise == null;
        }
    }

    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private BitKernels() {}

    // ---------------------------------------------------------------- column / column

    /**
     * {@code out[i] = a[i] op b[i]}; for and/or/xor {@code b} has {@code a}'s
     * lane type, for the shifts {@code b} is INT32 whatever {@code a} is.
     * {@code out} has {@code a}'s lane type.
     */
    public static void binary(BitOp op, VectorBuffers a, VectorBuffers b,
            MemorySegment out) {
        int n = a.length();
        MemorySegment ad = a.data(), bd = b.data();
        switch (a.type()) {
            case INT32 -> {
                if (op.isShift()) {
                    requireType(b, VecType.INT32, "shift amount");
                    for (int i = 0; i < n; i++) {
                        out.setAtIndex(VectorBuffers.LE_INT, i,
                                shift(op, ad.getAtIndex(VectorBuffers.LE_INT, i), bd.getAtIndex(VectorBuffers.LE_INT, i)));
                    }
                } else {
                    requireType(b, VecType.INT32, op.toString());
                    int lanes = I.length(), i = 0;
                    for (; i + lanes <= n; i += lanes) {
                        long off = (long) i << 2;
                        IntVector.fromMemorySegment(I, ad, off, LE)
                                .lanewise(op.lanewise, IntVector.fromMemorySegment(I, bd, off, LE))
                                .intoMemorySegment(out, off, LE);
                    }
                    if (i < n) {
                        VectorMask<Integer> m = I.indexInRange(i, n);
                        long off = (long) i << 2;
                        IntVector.fromMemorySegment(I, ad, off, LE, m)
                                .lanewise(op.lanewise, IntVector.fromMemorySegment(I, bd, off, LE, m))
                                .intoMemorySegment(out, off, LE, m);
                    }
                }
            }
            case INT64 -> {
                if (op.isShift()) {
                    requireType(b, VecType.INT32, "shift amount");
                    for (int i = 0; i < n; i++) {
                        out.setAtIndex(VectorBuffers.LE_LONG, i,
                                shift(op, ad.getAtIndex(VectorBuffers.LE_LONG, i), bd.getAtIndex(VectorBuffers.LE_INT, i)));
                    }
                } else {
                    requireType(b, VecType.INT64, op.toString());
                    int lanes = L.length(), i = 0;
                    for (; i + lanes <= n; i += lanes) {
                        long off = (long) i << 3;
                        LongVector.fromMemorySegment(L, ad, off, LE)
                                .lanewise(op.lanewise, LongVector.fromMemorySegment(L, bd, off, LE))
                                .intoMemorySegment(out, off, LE);
                    }
                    if (i < n) {
                        VectorMask<Long> m = L.indexInRange(i, n);
                        long off = (long) i << 3;
                        LongVector.fromMemorySegment(L, ad, off, LE, m)
                                .lanewise(op.lanewise, LongVector.fromMemorySegment(L, bd, off, LE, m))
                                .intoMemorySegment(out, off, LE, m);
                    }
                }
            }
            default -> throw new IllegalArgumentException("bitwise " + op + " over " + a.type());
        }
    }

    // ---------------------------------------------------------------- column / scalar

    /**
     * {@code out[i] = a[i] op s}. And/or/xor commute, so a literal on either
     * side takes this form; a literal <em>shifted value</em> ({@code 1 << i})
     * takes {@link #shiftScalarValue} instead.
     */
    public static void binaryScalar(BitOp op, VectorBuffers a, long s,
            MemorySegment out) {
        int n = a.length();
        MemorySegment ad = a.data();
        switch (a.type()) {
            case INT32 -> {
                int v = (int) s;
                if (op.isShift()) {
                    for (int i = 0; i < n; i++) {
                        out.setAtIndex(VectorBuffers.LE_INT, i, shift(op, ad.getAtIndex(VectorBuffers.LE_INT, i), v));
                    }
                } else {
                    int lanes = I.length(), i = 0;
                    IntVector sv = IntVector.broadcast(I, v);
                    for (; i + lanes <= n; i += lanes) {
                        long off = (long) i << 2;
                        IntVector.fromMemorySegment(I, ad, off, LE)
                                .lanewise(op.lanewise, sv)
                                .intoMemorySegment(out, off, LE);
                    }
                    if (i < n) {
                        VectorMask<Integer> m = I.indexInRange(i, n);
                        long off = (long) i << 2;
                        IntVector.fromMemorySegment(I, ad, off, LE, m)
                                .lanewise(op.lanewise, sv)
                                .intoMemorySegment(out, off, LE, m);
                    }
                }
            }
            case INT64 -> {
                if (op.isShift()) {
                    int v = (int) s;
                    for (int i = 0; i < n; i++) {
                        out.setAtIndex(VectorBuffers.LE_LONG, i, shift(op, ad.getAtIndex(VectorBuffers.LE_LONG, i), v));
                    }
                } else {
                    int lanes = L.length(), i = 0;
                    LongVector sv = LongVector.broadcast(L, s);
                    for (; i + lanes <= n; i += lanes) {
                        long off = (long) i << 3;
                        LongVector.fromMemorySegment(L, ad, off, LE)
                                .lanewise(op.lanewise, sv)
                                .intoMemorySegment(out, off, LE);
                    }
                    if (i < n) {
                        VectorMask<Long> m = L.indexInRange(i, n);
                        long off = (long) i << 3;
                        LongVector.fromMemorySegment(L, ad, off, LE, m)
                                .lanewise(op.lanewise, sv)
                                .intoMemorySegment(out, off, LE, m);
                    }
                }
            }
            default -> throw new IllegalArgumentException("bitwise " + op + " over " + a.type());
        }
    }

    /**
     * A literal value shifted by a column of amounts: {@code out[i] = value op
     * amounts[i]}, with {@code out} in {@code type} (INT32 or INT64, the
     * literal's type). {@code amounts} is INT32.
     */
    public static void shiftScalarValue(BitOp op, VecType type, long value,
            VectorBuffers amounts, MemorySegment out) {
        requireType(amounts, VecType.INT32, "shift amount");
        int n = amounts.length();
        MemorySegment bd = amounts.data();
        switch (type) {
            case INT32 -> {
                int v = (int) value;
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_INT, i, shift(op, v, bd.getAtIndex(VectorBuffers.LE_INT, i)));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_LONG, i, shift(op, value, bd.getAtIndex(VectorBuffers.LE_INT, i)));
                }
            }
            default -> throw new IllegalArgumentException("shift over " + type);
        }
    }

    // ---------------------------------------------------------------- unary

    /** {@code out[i] = ~a[i]}. */
    public static void not(VectorBuffers a, MemorySegment out) {
        int n = a.length();
        MemorySegment ad = a.data();
        switch (a.type()) {
            case INT32 -> {
                int lanes = I.length(), i = 0;
                for (; i + lanes <= n; i += lanes) {
                    long off = (long) i << 2;
                    IntVector.fromMemorySegment(I, ad, off, LE)
                            .not()
                            .intoMemorySegment(out, off, LE);
                }
                if (i < n) {
                    VectorMask<Integer> m = I.indexInRange(i, n);
                    long off = (long) i << 2;
                    IntVector.fromMemorySegment(I, ad, off, LE, m)
                            .not()
                            .intoMemorySegment(out, off, LE, m);
                }
            }
            case INT64 -> {
                int lanes = L.length(), i = 0;
                for (; i + lanes <= n; i += lanes) {
                    long off = (long) i << 3;
                    LongVector.fromMemorySegment(L, ad, off, LE)
                            .not()
                            .intoMemorySegment(out, off, LE);
                }
                if (i < n) {
                    VectorMask<Long> m = L.indexInRange(i, n);
                    long off = (long) i << 3;
                    LongVector.fromMemorySegment(L, ad, off, LE, m)
                            .not()
                            .intoMemorySegment(out, off, LE, m);
                }
            }
            default -> throw new IllegalArgumentException("bitwise not over " + a.type());
        }
    }

    /**
     * {@code out[i] = Long.bitCount((long) a[i])} into INT32 lanes -- Spark
     * widens first, so a negative int counts 64 bits' worth.
     */
    public static void bitCount(VectorBuffers a, MemorySegment out) {
        int n = a.length();
        MemorySegment ad = a.data();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_INT, i, Long.bitCount(ad.getAtIndex(VectorBuffers.LE_INT, i)));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_INT, i, Long.bitCount(ad.getAtIndex(VectorBuffers.LE_LONG, i)));
                }
            }
            default -> throw new IllegalArgumentException("bit_count over " + a.type());
        }
    }

    // ---------------------------------------------------------------- scalar definitions

    /** Java's shift on an int: the amount is masked to 5 bits. */
    public static int shift(BitOp op, int x, int amount) {
        return switch (op) {
            case SHL -> x << amount;
            case SHR -> x >> amount;
            case USHR -> x >>> amount;
            default -> throw new IllegalArgumentException(op + " is not a shift");
        };
    }

    /** Java's shift on a long: the amount is masked to 6 bits. */
    public static long shift(BitOp op, long x, int amount) {
        return switch (op) {
            case SHL -> x << amount;
            case SHR -> x >> amount;
            case USHR -> x >>> amount;
            default -> throw new IllegalArgumentException(op + " is not a shift");
        };
    }

    private static void requireType(VectorBuffers a, VecType type, String what) {
        if (a.type() != type) {
            throw new IllegalArgumentException(what + " needs " + type + " lanes, got " + a.type());
        }
    }
}
