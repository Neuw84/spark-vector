package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;

/**
 * Transcendental and trigonometric functions over FLOAT64 lanes, one scalar
 * call per lane.
 *
 * <p>The call is exactly the one Spark's generated code makes -- {@code
 * java.lang.Math} for the trigonometric and hyperbolic functions, {@code cbrt},
 * {@code degrees}/{@code radians}, {@code atan2} and {@code hypot}; {@code
 * java.lang.StrictMath} for {@code exp}/{@code expm1}, the log family and
 * {@code pow}; Spark's own formulas for the inverse hyperbolics -- so a lane is
 * bit-identical to Spark's value and no ulp tolerance enters. The Vector API's
 * {@code EXP}/{@code LOG}/{@code SIN}... operators are 1-2 ulp off {@code
 * StrictMath} and are deliberately not used here (a measured follow-up, see the
 * expression matrix). Nothing here touches validity: the caller nulls the lanes
 * Spark nulls -- the log family at or below its asymptote ({@link #asymptote}),
 * {@code log(base, x)} for a non-positive operand ({@link #positive}); every
 * other domain edge is {@code Math}'s NaN.
 */
public final class TranscendentalKernels {
    private TranscendentalKernels() {}

    /** The unary functions, named as Spark's expression classes. */
    public enum Fn {
        EXP,
        EXPM1,
        LOG,
        LOG2,
        LOG10,
        LOG1P,
        CBRT,
        SIN,
        COS,
        TAN,
        ASIN,
        ACOS,
        ATAN,
        SINH,
        COSH,
        TANH,
        ASINH,
        ACOSH,
        ATANH,
        COT,
        SEC,
        CSC,
        DEGREES,
        RADIANS
    }

    /** The binary functions. */
    public enum Fn2 {
        POW,
        ATAN2,
        HYPOT,
        LOG_BASE
    }

    private static final double LOG_2 = StrictMath.log(2);
    private static final double SQRT_MAX = Math.sqrt(Double.MAX_VALUE);

    /** Spark's value of {@code fn} at {@code x}, for a lane above the asymptote. */
    public static double apply(Fn fn, double x) {
        return switch (fn) {
            case EXP -> StrictMath.exp(x);
            case EXPM1 -> StrictMath.expm1(x);
            case LOG -> StrictMath.log(x);
            case LOG2 -> StrictMath.log(x) / LOG_2;
            case LOG10 -> StrictMath.log10(x);
            case LOG1P -> StrictMath.log1p(x);
            case CBRT -> Math.cbrt(x);
            case SIN -> Math.sin(x);
            case COS -> Math.cos(x);
            case TAN -> Math.tan(x);
            case ASIN -> Math.asin(x);
            case ACOS -> Math.acos(x);
            case ATAN -> Math.atan(x);
            case SINH -> Math.sinh(x);
            case COSH -> Math.cosh(x);
            case TANH -> Math.tanh(x);
            // Spark's Asinh / Acosh / Atanh formulas (SPARK-28519 for atanh), StrictMath where Spark uses it.
            case ASINH ->
                    Math.abs(x) >= SQRT_MAX - 1 ? Math.signum(x) * (StrictMath.log(2) + StrictMath.log(Math.abs(x))) : StrictMath.log(x + Math.sqrt(x * x + 1.0));
            case ACOSH ->
                    x >= SQRT_MAX
                            ? StrictMath.log(2) + StrictMath.log(x)
                            : x < 1 ? Double.NaN : StrictMath.log(x + Math.sqrt(x * x - 1.0));
            case ATANH -> 0.5 * (StrictMath.log1p(x) - StrictMath.log1p(-x));
            case COT -> 1 / Math.tan(x);
            case SEC -> 1 / Math.cos(x);
            case CSC -> 1 / Math.sin(x);
            case DEGREES -> Math.toDegrees(x);
            case RADIANS -> Math.toRadians(x);
        };
    }

    /**
     * The value at or below which Spark's {@code UnaryLogExpression} is null
     * instead of NaN or -Infinity: 0 for the logarithms, -1 for {@code log1p};
     * NaN for a function without one.
     */
    public static double asymptote(Fn fn) {
        return switch (fn) {
            case LOG, LOG2, LOG10 -> 0.0;
            case LOG1P -> -1.0;
            default -> Double.NaN;
        };
    }

    /** Spark's value of {@code fn} at {@code (a, b)}. */
    public static double apply(Fn2 fn, double a, double b) {
        return switch (fn) {
            case POW -> StrictMath.pow(a, b);
            // Spark adds 0.0 so that -0.0 and 0.0 agree between codegen and interpretation.
            case ATAN2 -> Math.atan2(a + 0.0, b + 0.0);
            case HYPOT -> Math.hypot(a, b);
            case LOG_BASE -> StrictMath.log(b) / StrictMath.log(a);
        };
    }

    /**
     * {@code out[i] = fn(a[i])} over a FLOAT64 lane. Null lanes hold arbitrary
     * values.
     */
    public static void unary(Fn fn, VectorBuffers a, MemorySegment out) {
        requireDouble(a, fn.name());
        int n = a.length();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, apply(fn, d.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
        }
    }

    /** {@code out[i] = fn(a[i], b[i])}. */
    public static void binary(Fn2 fn, VectorBuffers a, VectorBuffers b,
            MemorySegment out) {
        requireDouble(a, fn.name());
        requireDouble(b, fn.name());
        int n = a.length();
        MemorySegment da = a.data();
        MemorySegment db = b.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i,
                    apply(fn, da.getAtIndex(VectorBuffers.LE_DOUBLE, i), db.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
        }
    }

    /** {@code out[i] = fn(a[i], s)}, or {@code fn(s, a[i])} when {@code reversed}. */
    public static void binaryScalar(Fn2 fn, VectorBuffers a, double s,
            boolean reversed, MemorySegment out) {
        requireDouble(a, fn.name());
        int n = a.length();
        MemorySegment da = a.data();
        for (int i = 0; i < n; i++) {
            double x = da.getAtIndex(VectorBuffers.LE_DOUBLE, i);
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i,
                    reversed ? apply(fn, s, x) : apply(fn, x, s));
        }
    }

    /**
     * Sets bit {@code i} of {@code outBits} where {@code !(a[i] <= threshold)}
     * -- Spark's domain test, which keeps NaN (NaN is not {@code <=} anything,
     * so {@code log(NaN)} is NaN, not null) -- and clears it otherwise.
     */
    public static void above(VectorBuffers a, double threshold, MemorySegment outBits) {
        requireDouble(a, "above");
        int n = a.length();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(outBits, i, !(d.getAtIndex(VectorBuffers.LE_DOUBLE, i) <= threshold));
        }
    }

    private static void requireDouble(VectorBuffers a, String what) {
        if (a.type() != VecType.FLOAT64) {
            throw new IllegalArgumentException(what + " over " + a.type());
        }
    }
}
