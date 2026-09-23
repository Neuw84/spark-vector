package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD arithmetic over fixed-width columns. Values only: result validity is the
 * AND of the operand validities (see {@link BitmapKernels#combineValidity}),
 * and for DIV additionally excludes lanes whose divisor is zero (Spark returns
 * null there in legacy mode and raises in ANSI mode).
 *
 * <p>Integer ADD/SUB/MUL wrap on overflow exactly like Java, which matches
 * Spark's legacy mode. Each operator has its own loop because C2 only
 * intrinsifies {@code lanewise} with a constant operator; the {@code
 * add/sub/mul/div} convenience methods provide that constant.
 */
public final class ArithKernels {

    static final VectorSpecies<Integer> I = Species.I;
    static final VectorSpecies<Long> L = Species.L;
    static final VectorSpecies<Double> D = Species.D;
    static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    private ArithKernels() {}

    /** {@code out = a <op> b}; same type and length. */
    public static void arith(ArithOp op, VectorBuffers a, VectorBuffers b,
            MemorySegment out) {
        if (a.type() != b.type() || a.length() != b.length()) {
            throw new IllegalArgumentException("operands differ: " + a.type() + "/" + b.type());
        }
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                switch (op) {
                    case ADD -> i32Add(a.data(), b.data(), n, out);
                    case SUB -> i32Sub(a.data(), b.data(), n, out);
                    case MUL -> i32Mul(a.data(), b.data(), n, out);
                    case DIV -> throw new IllegalArgumentException("integer DIV");
                }
            }
            case INT64 -> {
                switch (op) {
                    case ADD -> i64Add(a.data(), b.data(), n, out);
                    case SUB -> i64Sub(a.data(), b.data(), n, out);
                    case MUL -> i64Mul(a.data(), b.data(), n, out);
                    case DIV -> throw new IllegalArgumentException("integer DIV");
                }
            }
            case FLOAT64 -> {
                switch (op) {
                    case ADD -> f64Add(a.data(), b.data(), n, out);
                    case SUB -> f64Sub(a.data(), b.data(), n, out);
                    case MUL -> f64Mul(a.data(), b.data(), n, out);
                    case DIV -> f64Div(a.data(), b.data(), n, out);
                }
            }
            default -> throw new IllegalArgumentException("unsupported arithmetic type " + a.type());
        }
    }

    /** {@code out = a <op> s}. */
    public static void arithScalar(ArithOp op, VectorBuffers a, Number s,
            MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                int v = s.intValue();
                switch (op) {
                    case ADD -> i32AddS(a.data(), v, n, out);
                    case SUB -> i32AddS(a.data(), -v, n, out); // a - s == a + (-s), also for MIN_VALUE (wraps)
                    case MUL -> i32MulS(a.data(), v, n, out);
                    case DIV -> throw new IllegalArgumentException("integer DIV");
                }
            }
            case INT64 -> {
                long v = s.longValue();
                switch (op) {
                    case ADD -> i64AddS(a.data(), v, n, out);
                    case SUB -> i64AddS(a.data(), -v, n, out);
                    case MUL -> i64MulS(a.data(), v, n, out);
                    case DIV -> throw new IllegalArgumentException("integer DIV");
                }
            }
            case FLOAT64 -> {
                double v = s.doubleValue();
                switch (op) {
                    case ADD -> f64AddS(a.data(), v, n, out);
                    case SUB -> f64SubS(a.data(), v, n, out);
                    case MUL -> f64MulS(a.data(), v, n, out);
                    case DIV -> f64DivS(a.data(), v, n, out);
                }
            }
            default -> throw new IllegalArgumentException("unsupported arithmetic type " + a.type());
        }
    }

    /**
     * {@code out = s <op> b}; commutative operators delegate to {@link
     * #arithScalar}.
     */
    public static void scalarArith(ArithOp op, Number s, VectorBuffers b,
            MemorySegment out) {
        if (op.isCommutative()) {
            arithScalar(op, b, s, out);
            return;
        }
        int n = b.length();
        switch (b.type()) {
            case INT32 -> {
                if (op != ArithOp.SUB) {
                    throw new IllegalArgumentException("integer DIV");
                }
                i32SSub(s.intValue(), b.data(), n, out);
            }
            case INT64 -> {
                if (op != ArithOp.SUB) {
                    throw new IllegalArgumentException("integer DIV");
                }
                i64SSub(s.longValue(), b.data(), n, out);
            }
            case FLOAT64 -> {
                if (op == ArithOp.SUB) {
                    f64SSub(s.doubleValue(), b.data(), n, out);
                } else {
                    f64SDiv(s.doubleValue(), b.data(), n, out);
                }
            }
            default -> throw new IllegalArgumentException("unsupported arithmetic type " + b.type());
        }
    }

    /** {@code out = -a}. */
    public static void negate(VectorBuffers a, MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> i32Neg(a.data(), n, out);
            case INT64 -> i64Neg(a.data(), n, out);
            case FLOAT64 -> f64Neg(a.data(), n, out);
            default -> throw new IllegalArgumentException("unsupported arithmetic type " + a.type());
        }
    }

    // ================================================================== int32

    static void i32Add(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = I.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE)
                    .add(IntVector.fromMemorySegment(I, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE, m)
                    .add(IntVector.fromMemorySegment(I, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i32Sub(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = I.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE)
                    .sub(IntVector.fromMemorySegment(I, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE, m)
                    .sub(IntVector.fromMemorySegment(I, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i32Mul(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = I.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE)
                    .mul(IntVector.fromMemorySegment(I, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE, m)
                    .mul(IntVector.fromMemorySegment(I, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i32AddS(MemorySegment a, int s, int n,
                        MemorySegment out) {
        int lanes = I.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE)
                    .add(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE, m)
                    .add(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i32MulS(MemorySegment a, int s, int n,
                        MemorySegment out) {
        int lanes = I.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE)
                    .mul(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE, m)
                    .mul(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i32SSub(int s, MemorySegment b, int n,
                        MemorySegment out) {
        int lanes = I.length(), i = 0;
        IntVector sv = IntVector.broadcast(I, s);
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            sv.sub(IntVector.fromMemorySegment(I, b, off, LE)).intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            sv.sub(IntVector.fromMemorySegment(I, b, off, LE, m)).intoMemorySegment(out, off, LE, m);
        }
    }

    static void i32Neg(MemorySegment a, int n, MemorySegment out) {
        int lanes = I.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE)
                    .neg()
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Integer> m = I.indexInRange(i, n);
            long off = (long) i << 2;
            IntVector.fromMemorySegment(I, a, off, LE, m)
                    .neg()
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    // ================================================================== int64

    static void i64Add(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE)
                    .add(LongVector.fromMemorySegment(L, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE, m)
                    .add(LongVector.fromMemorySegment(L, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i64Sub(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE)
                    .sub(LongVector.fromMemorySegment(L, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE, m)
                    .sub(LongVector.fromMemorySegment(L, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i64Mul(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE)
                    .mul(LongVector.fromMemorySegment(L, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE, m)
                    .mul(LongVector.fromMemorySegment(L, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i64AddS(MemorySegment a, long s, int n,
                        MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE)
                    .add(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE, m)
                    .add(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i64MulS(MemorySegment a, long s, int n,
                        MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE)
                    .mul(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE, m)
                    .mul(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void i64SSub(long s, MemorySegment b, int n,
                        MemorySegment out) {
        int lanes = L.length(), i = 0;
        LongVector sv = LongVector.broadcast(L, s);
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            sv.sub(LongVector.fromMemorySegment(L, b, off, LE)).intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            sv.sub(LongVector.fromMemorySegment(L, b, off, LE, m)).intoMemorySegment(out, off, LE, m);
        }
    }

    static void i64Neg(MemorySegment a, int n, MemorySegment out) {
        int lanes = L.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE)
                    .neg()
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Long> m = L.indexInRange(i, n);
            long off = (long) i << 3;
            LongVector.fromMemorySegment(L, a, off, LE, m)
                    .neg()
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    // ================================================================== float64

    static void f64Add(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .add(DoubleVector.fromMemorySegment(D, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .add(DoubleVector.fromMemorySegment(D, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64Sub(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .sub(DoubleVector.fromMemorySegment(D, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .sub(DoubleVector.fromMemorySegment(D, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64Mul(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .mul(DoubleVector.fromMemorySegment(D, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .mul(DoubleVector.fromMemorySegment(D, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64Div(MemorySegment a, MemorySegment b, int n,
                       MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .div(DoubleVector.fromMemorySegment(D, b, off, LE))
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .div(DoubleVector.fromMemorySegment(D, b, off, LE, m))
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64AddS(MemorySegment a, double s, int n,
                        MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .add(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .add(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64SubS(MemorySegment a, double s, int n,
                        MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .sub(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .sub(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64MulS(MemorySegment a, double s, int n,
                        MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .mul(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .mul(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64DivS(MemorySegment a, double s, int n,
                        MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .div(s)
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .div(s)
                    .intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64SSub(double s, MemorySegment b, int n,
                        MemorySegment out) {
        int lanes = D.length(), i = 0;
        DoubleVector sv = DoubleVector.broadcast(D, s);
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            sv.sub(DoubleVector.fromMemorySegment(D, b, off, LE)).intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            sv.sub(DoubleVector.fromMemorySegment(D, b, off, LE, m)).intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64SDiv(double s, MemorySegment b, int n,
                        MemorySegment out) {
        int lanes = D.length(), i = 0;
        DoubleVector sv = DoubleVector.broadcast(D, s);
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            sv.div(DoubleVector.fromMemorySegment(D, b, off, LE)).intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            sv.div(DoubleVector.fromMemorySegment(D, b, off, LE, m)).intoMemorySegment(out, off, LE, m);
        }
    }

    static void f64Neg(MemorySegment a, int n, MemorySegment out) {
        int lanes = D.length(), i = 0;
        for (; i + lanes <= n; i += lanes) {
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE)
                    .neg()
                    .intoMemorySegment(out, off, LE);
        }
        if (i < n) {
            VectorMask<Double> m = D.indexInRange(i, n);
            long off = (long) i << 3;
            DoubleVector.fromMemorySegment(D, a, off, LE, m)
                    .neg()
                    .intoMemorySegment(out, off, LE, m);
        }
    }
}
