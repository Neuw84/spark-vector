package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;

/**
 * The mainstream arithmetic surface beyond {@code + - * /}: {@code abs}, {@code
 * signum}, {@code %} / {@code pmod}, integral division, {@code greatest} /
 * {@code least} and {@code nanvl}, over INT32 / INT64 / FLOAT64 lanes. Every
 * kernel is a per-lane loop the JIT vectorises where the operation allows; none
 * touches validity -- the caller combines the operands' validity and masks
 * divisor-zero lanes as the ANSI rules require. Null lanes hold arbitrary
 * values.
 *
 * <p>Semantics follow Spark's implementations, not intuition: {@code %} is
 * Java's truncated remainder (sign of the dividend); {@code pmod(a, n)} is
 * {@code r = a % n; r < 0 ? (r + n) % n : r}; integral division is {@code
 * Integral.quot}, truncation toward zero, with a long result; a zero divisor
 * never reaches the arithmetic (the lane is skipped and left 0 -- the caller
 * nulls or raises it). {@code greatest} / {@code least} ignore null operands
 * and use Spark's double ordering (NaN greatest, {@code -0.0 == 0.0}).
 */
public final class MathKernels {
    /** Remainder flavours. */
    public enum RemOp {
        /** Java / Spark {@code %}. */
        REM,
        /** Spark's non-negative {@code pmod}. */
        PMOD
    }

    /** Reductions across columns. */
    public enum Pick {
        GREATEST,
        LEAST
    }

    private MathKernels() {}

    // ---------------------------------------------------------------- abs / signum

    /**
     * {@code out = |a|}, same lane type; {@code MIN_VALUE} wraps (the caller
     * checks it in ANSI mode).
     */
    public static void abs(VectorBuffers a, MemorySegment out) {
        int n = a.length();
        MemorySegment d = a.data();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_INT, i, Math.abs(d.getAtIndex(VectorBuffers.LE_INT, i)));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_LONG, i, Math.abs(d.getAtIndex(VectorBuffers.LE_LONG, i)));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.setAtIndex(VectorBuffers.LE_DOUBLE, i, Math.abs(d.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
                }
            }
            default -> throw new IllegalArgumentException("abs over " + a.type());
        }
    }

    /**
     * {@code out = signum(a)} over FLOAT64 (Spark casts the argument to
     * double); NaN stays NaN.
     */
    public static void signum(VectorBuffers a, MemorySegment out) {
        requireType(a, VecType.FLOAT64, "signum");
        int n = a.length();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, Math.signum(d.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
        }
    }

    /**
     * {@code out = sqrt(a)} over FLOAT64 (Spark casts the argument to double);
     * negative lanes give NaN, as Spark's.
     */
    public static void sqrt(VectorBuffers a, MemorySegment out) {
        requireType(a, VecType.FLOAT64, "sqrt");
        int n = a.length();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, Math.sqrt(d.getAtIndex(VectorBuffers.LE_DOUBLE, i)));
        }
    }

    // ---------------------------------------------------------------- remainder

    /** {@code out = a <op> b}; lanes whose divisor is zero are left 0. */
    public static void remainder(RemOp op, VectorBuffers a, VectorBuffers b,
            MemorySegment out) {
        int n = a.length();
        MemorySegment da = a.data(), db = b.data();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    int y = db.getAtIndex(VectorBuffers.LE_INT, i);
                    out.setAtIndex(VectorBuffers.LE_INT, i,
                            y == 0 ? 0 : rem(op, da.getAtIndex(VectorBuffers.LE_INT, i), y));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    long y = db.getAtIndex(VectorBuffers.LE_LONG, i);
                    out.setAtIndex(VectorBuffers.LE_LONG, i,
                            y == 0L ? 0L : rem(op, da.getAtIndex(VectorBuffers.LE_LONG, i), y));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    double y = db.getAtIndex(VectorBuffers.LE_DOUBLE, i);
                    out.setAtIndex(VectorBuffers.LE_DOUBLE, i,
                            y == 0.0 ? 0.0 : rem(op, da.getAtIndex(VectorBuffers.LE_DOUBLE, i), y));
                }
            }
            default -> throw new IllegalArgumentException("remainder over " + a.type());
        }
    }

    /**
     * {@code out = a <op> s} ({@code reversed}: {@code s <op> a}); a zero
     * divisor lane is left 0.
     */
    public static void remainderScalar(RemOp op, VectorBuffers a, Number s,
            boolean reversed, MemorySegment out) {
        int n = a.length();
        MemorySegment d = a.data();
        switch (a.type()) {
            case INT32 -> {
                int sv = s.intValue();
                for (int i = 0; i < n; i++) {
                    int x = d.getAtIndex(VectorBuffers.LE_INT, i);
                    int r = reversed
                            ? (x == 0 ? 0 : rem(op, sv, x))
                            : (sv == 0 ? 0 : rem(op, x, sv));
                    out.setAtIndex(VectorBuffers.LE_INT, i, r);
                }
            }
            case INT64 -> {
                long sv = s.longValue();
                for (int i = 0; i < n; i++) {
                    long x = d.getAtIndex(VectorBuffers.LE_LONG, i);
                    long r = reversed
                            ? (x == 0L ? 0L : rem(op, sv, x))
                            : (sv == 0L ? 0L : rem(op, x, sv));
                    out.setAtIndex(VectorBuffers.LE_LONG, i, r);
                }
            }
            case FLOAT64 -> {
                double sv = s.doubleValue();
                for (int i = 0; i < n; i++) {
                    double x = d.getAtIndex(VectorBuffers.LE_DOUBLE, i);
                    double r = reversed
                            ? (x == 0.0 ? 0.0 : rem(op, sv, x))
                            : (sv == 0.0 ? 0.0 : rem(op, x, sv));
                    out.setAtIndex(VectorBuffers.LE_DOUBLE, i, r);
                }
            }
            default -> throw new IllegalArgumentException("remainder over " + a.type());
        }
    }

    static int rem(RemOp op, int a, int n) {
        int r = a % n;
        return op == RemOp.PMOD && r < 0
                ? (r + n) % n
                : r;
    }

    static long rem(RemOp op, long a, long n) {
        long r = a % n;
        return op == RemOp.PMOD && r < 0
                ? (r + n) % n
                : r;
    }

    static double rem(RemOp op, double a, double n) {
        double r = a % n;
        return op == RemOp.PMOD && r < 0
                ? (r + n) % n
                : r;
    }

    // ---------------------------------------------------------------- integral divide

    /**
     * {@code out = a div b} as INT64 (Spark's {@code IntegralDivide} result
     * type) over INT32 or INT64 operands; a zero divisor lane is left 0; {@code
     * Long.MIN_VALUE / -1} wraps (the caller checks it in ANSI mode via {@link
     * #integralDivideOverflow}).
     */
    public static void integralDivide(VectorBuffers a, VectorBuffers b, MemorySegment out) {
        int n = a.length();
        MemorySegment da = a.data(), db = b.data();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    int y = db.getAtIndex(VectorBuffers.LE_INT, i);
                    out.setAtIndex(VectorBuffers.LE_LONG, i,
                            y == 0 ? 0L : (long) da.getAtIndex(VectorBuffers.LE_INT, i) / y);
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    long y = db.getAtIndex(VectorBuffers.LE_LONG, i);
                    out.setAtIndex(VectorBuffers.LE_LONG, i,
                            y == 0L ? 0L : da.getAtIndex(VectorBuffers.LE_LONG, i) / y);
                }
            }
            default -> throw new IllegalArgumentException("integral divide over " + a.type());
        }
    }

    /**
     * As {@link #integralDivide} against a scalar ({@code reversed}: {@code s
     * div a}).
     */
    public static void integralDivideScalar(VectorBuffers a, Number s, boolean reversed,
            MemorySegment out) {
        int n = a.length();
        MemorySegment d = a.data();
        switch (a.type()) {
            case INT32 -> {
                long sv = s.longValue();
                for (int i = 0; i < n; i++) {
                    long x = d.getAtIndex(VectorBuffers.LE_INT, i);
                    out.setAtIndex(
                            VectorBuffers.LE_LONG,
                            i,
                            reversed
                                    ? (x == 0L ? 0L : sv / x)
                                    : (sv == 0L ? 0L : x / sv));
                }
            }
            case INT64 -> {
                long sv = s.longValue();
                for (int i = 0; i < n; i++) {
                    long x = d.getAtIndex(VectorBuffers.LE_LONG, i);
                    out.setAtIndex(
                            VectorBuffers.LE_LONG,
                            i,
                            reversed
                                    ? (x == 0L ? 0L : sv / x)
                                    : (sv == 0L ? 0L : x / sv));
                }
            }
            default -> throw new IllegalArgumentException("integral divide over " + a.type());
        }
    }

    /**
     * Sets a bit where an INT64 {@code a div b} overflows: dividend {@code
     * Long.MIN_VALUE}, divisor -1.
     */
    public static void integralDivideOverflow(VectorBuffers a, VectorBuffers b, MemorySegment out) {
        int n = a.length();
        Bitmap.fill(out, n, false);
        if (a.type() != VecType.INT64) {
            return;
        }
        MemorySegment da = a.data(), db = b.data();
        for (int i = 0; i < n; i++) {
            if (da.getAtIndex(VectorBuffers.LE_LONG, i) == Long.MIN_VALUE && db.getAtIndex(VectorBuffers.LE_LONG, i) == -1L) {
                Bitmap.set(out, i);
            }
        }
    }

    /** Scalar form of {@link #integralDivideOverflow}. */
    public static void integralDivideOverflowScalar(VectorBuffers a, Number s, boolean reversed,
            MemorySegment out) {
        int n = a.length();
        Bitmap.fill(out, n, false);
        if (a.type() != VecType.INT64) {
            return;
        }
        long sv = s.longValue();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            long x = d.getAtIndex(VectorBuffers.LE_LONG, i);
            boolean ov = reversed
                    ? (sv == Long.MIN_VALUE && x == -1L)
                    : (x == Long.MIN_VALUE && sv == -1L);
            if (ov) {
                Bitmap.set(out, i);
            }
        }
    }

    // ---------------------------------------------------------------- greatest / least

    /**
     * Lane-wise {@code greatest} / {@code least} across {@code cols} (all the
     * same type and length), ignoring null operands; {@code outValidity} gets a
     * bit where at least one operand was valid. Doubles use Spark's ordering
     * (NaN greatest, {@code -0.0 == 0.0}).
     */
    public static void pick(Pick pick, VectorBuffers[] cols, MemorySegment out,
                            MemorySegment outValidity) {
        int n = cols[0].length();
        VecType type = cols[0].type();
        boolean greatest = pick == Pick.GREATEST;
        Bitmap.fill(outValidity, n, false);
        for (int i = 0; i < n; i++) {
            boolean any = false;
            switch (type) {
                case INT32 -> {
                    int best = 0;
                    for (VectorBuffers c : cols) {
                        if (c.isNull(i)) {
                            continue;
                        }
                        int v = c.getInt(i);
                        if (!any
                                || (greatest ? v > best : v < best)) {
                            best = v;
                        }
                        any = true;
                    }
                    out.setAtIndex(VectorBuffers.LE_INT, i, best);
                }
                case INT64 -> {
                    long best = 0L;
                    for (VectorBuffers c : cols) {
                        if (c.isNull(i)) {
                            continue;
                        }
                        long v = c.getLong(i);
                        if (!any
                                || (greatest ? v > best : v < best)) {
                            best = v;
                        }
                        any = true;
                    }
                    out.setAtIndex(VectorBuffers.LE_LONG, i, best);
                }
                case FLOAT64 -> {
                    double best = 0.0;
                    for (VectorBuffers c : cols) {
                        if (c.isNull(i)) {
                            continue;
                        }
                        double v = c.getDouble(i);
                        int cmp = CompareOp.nanSafeCompare(v, best);
                        if (!any
                                || (greatest ? cmp > 0 : cmp < 0)) {
                            best = v;
                        }
                        any = true;
                    }
                    out.setAtIndex(VectorBuffers.LE_DOUBLE, i, best);
                }
                default -> throw new IllegalArgumentException(pick + " over " + type);
            }
            if (any) {
                Bitmap.set(outValidity, i);
            }
        }
    }

    // ---------------------------------------------------------------- nanvl

    /**
     * {@code nanvl(a, b)} over FLOAT64: {@code a} unless it is NaN, then {@code
     * b}. Sets {@code outValidity} as Spark's eval does: null where {@code a}
     * is null, and where {@code a} is NaN and {@code b} is null; a non-NaN
     * {@code a} wins whatever {@code b} is.
     */
    public static void nanvl(VectorBuffers a, VectorBuffers b, MemorySegment out,
            MemorySegment outValidity) {
        requireType(a, VecType.FLOAT64, "nanvl");
        int n = a.length();
        MemorySegment da = a.data(), db = b.data();
        for (int i = 0; i < n; i++) {
            boolean valid;
            double v;
            if (a.isNull(i)) {
                valid = false;
                v = 0.0;
            } else {
                double x = da.getAtIndex(VectorBuffers.LE_DOUBLE, i);
                if (Double.isNaN(x)) {
                    valid = !b.isNull(i);
                    v = valid ? db.getAtIndex(VectorBuffers.LE_DOUBLE, i) : 0.0;
                } else {
                    valid = true;
                    v = x;
                }
            }
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i, v);
            Bitmap.setTo(outValidity, i, valid);
        }
    }

    // ---------------------------------------------------------------- NormalizeNaNAndZero

    /**
     * Spark's {@code NormalizeNaNAndZero} over FLOAT64: every NaN becomes the
     * canonical {@link Double#NaN}, {@code -0.0} becomes {@code 0.0},
     * everything else is copied. After it, values that Spark considers equal
     * have equal bits, which is what the key table compares. Nulls keep their
     * (unspecified) data.
     */
    public static void normalizeNaNAndZero(VectorBuffers a, MemorySegment out) {
        requireType(a, VecType.FLOAT64, "normalizeNaNAndZero");
        int n = a.length();
        MemorySegment d = a.data();
        for (int i = 0; i < n; i++) {
            double x = d.getAtIndex(VectorBuffers.LE_DOUBLE, i);
            out.setAtIndex(VectorBuffers.LE_DOUBLE, i,
                    Double.isNaN(x)
                            ? Double.NaN
                            : x == 0.0 ? 0.0 : x);
        }
    }

    private static void requireType(VectorBuffers a, VecType type, String what) {
        if (a.type() != type) {
            throw new IllegalArgumentException(what + " over " + a.type());
        }
    }
}
