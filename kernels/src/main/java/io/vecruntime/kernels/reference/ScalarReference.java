/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.vecruntime.kernels.reference;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.vecruntime.kernels.ArithOp;
import io.vecruntime.kernels.Bitmap;
import io.vecruntime.kernels.CompareOp;
import io.vecruntime.kernels.DateKernels;
import io.vecruntime.kernels.StringMatchKernels;
import io.vecruntime.kernels.VecType;
import io.vecruntime.kernels.VectorBuffers;

/**
 * Straightforward scalar implementations of every kernel. They define the
 * expected semantics and are used as oracles by the tests and as baselines by
 * the JMH benchmarks. Nothing here is tuned.
 */
public final class ScalarReference {

    private ScalarReference() {}

    // ---------------------------------------------------------------- comparisons

    /**
     * Element-wise compare; writes one result bit per element (null lanes get
     * arbitrary bits).
     */
    public static void compareScalar(VectorBuffers a, Number scalar, CompareOp op,
            MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                int s = scalar.intValue();
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i, op.test(Integer.compare(a.getInt(i), s)));
                }
            }
            case INT64 -> {
                long s = scalar.longValue();
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i, op.test(Long.compare(a.getLong(i), s)));
                }
            }
            case FLOAT64 -> {
                double s = scalar.doubleValue();
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i, op.test(a.getDouble(i), s));
                }
            }
            case DECIMAL128 -> {
                java.math.BigInteger s = (java.math.BigInteger) scalar;
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i,
                            op.test(a.getDecimal128(i).compareTo(s)));
                }
            }
            default -> throw new IllegalArgumentException("unsupported " + a.type());
        }
    }

    public static void compare(VectorBuffers a, VectorBuffers b, CompareOp op,
            MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i,
                            op.test(Integer.compare(a.getInt(i), b.getInt(i))));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i,
                            op.test(Long.compare(a.getLong(i), b.getLong(i))));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i,
                            op.test(a.getDouble(i), b.getDouble(i)));
                }
            }
            case DECIMAL128 -> {
                for (int i = 0; i < n; i++) {
                    Bitmap.setTo(out, i, op.test(compareDecimal128(a, i, b, i)));
                }
            }
            default -> throw new IllegalArgumentException("unsupported " + a.type());
        }
    }

    // ---------------------------------------------------------------- string comparison

    /**
     * {@code a[i] <op> s} in UTF8_BINARY order, via {@link
     * Arrays#compareUnsigned} on the bytes.
     */
    public static void compareUtf8Scalar(VectorBuffers a, byte[] s, CompareOp op,
            MemorySegment out) {
        int n = a.length();
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i, op.test(java.util.Arrays.compareUnsigned(a.getUtf8Bytes(i), s)));
        }
    }

    /** {@code a[i] <op> b[i]} in UTF8_BINARY order. */
    public static void compareUtf8(VectorBuffers a, VectorBuffers b, CompareOp op,
            MemorySegment out) {
        int n = a.length();
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i,
                    op.test(java.util.Arrays.compareUnsigned(a.getUtf8Bytes(i), b.getUtf8Bytes(i))));
        }
    }

    /**
     * Oracle for {@code StringMatchKernels}: prefix / suffix / substring on the
     * UTF-8 bytes, decided by a byte-array scan independent of the kernel's
     * {@code mismatch} tricks.
     */
    public static void matchUtf8(StringMatchKernels.Kind kind, VectorBuffers a, byte[] pattern,
            MemorySegment out) {
        int n = a.length();
        for (int i = 0; i < n; i++) {
            byte[] s = a.getUtf8Bytes(i);
            boolean hit;
            if (pattern.length > s.length) {
                hit = false;
            } else {
                switch (kind) {
                    case PREFIX -> hit = java.util.Arrays.equals(s, 0, pattern.length, pattern, 0, pattern.length);
                    case SUFFIX -> hit = java.util.Arrays.equals(s, s.length - pattern.length, s.length, pattern, 0,
                            pattern.length);
                    default -> {
                        hit = false;
                        for (int p = 0;
                             p + pattern.length <= s.length && !hit;
                             p++) {
                            hit = java.util.Arrays.equals(s, p, p + pattern.length, pattern, 0,
                                    pattern.length);
                        }
                    }
                }
            }
            Bitmap.setTo(out, i, hit);
        }
    }

    /**
     * Oracle for {@link StringMatchKernels#matchTokens}: the {@code LIKE}
     * pattern rebuilt as the regex Spark compiles it to ({@code %} is {@code
     * .*}, everything else quoted, DOTALL) and run on the decoded string -- a
     * different algorithm on a different representation.
     */
    public static void likeTokens(VectorBuffers a, byte[] prefix, byte[][] tokens,
            byte[] suffix, MemorySegment out) {
        StringBuilder re = new StringBuilder("^").append(java.util.regex.Pattern.quote(new String(prefix, java.nio.charset.StandardCharsets.UTF_8)));
        for (byte[] t : tokens) {
            re.append(".*").append(java.util.regex.Pattern.quote(new String(t, java.nio.charset.StandardCharsets.UTF_8)));
        }
        re.append(".*")
          .append(java.util.regex.Pattern.quote(new String(suffix, java.nio.charset.StandardCharsets.UTF_8)))
          .append("$");
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(re.toString(), java.util.regex.Pattern.DOTALL);
        int n = a.length();
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i, !a.isNull(i) && p.matcher(a.getString(i)).matches());
        }
    }

    // ---------------------------------------------------------------- dates

    /**
     * Oracle for {@code DateKernels.field}: {@code java.time.LocalDate} on each
     * lane.
     */
    public static void dateField(DateKernels.Field field, VectorBuffers a, MemorySegment out) {
        int n = a.length();
        for (int i = 0; i < n; i++) {
            java.time.LocalDate d = java.time.LocalDate.ofEpochDay(a.getInt(i));
            int v = switch (field) {
                case YEAR -> d.getYear();
                case MONTH -> d.getMonthValue();
                case DAY -> d.getDayOfMonth();
                case DAY_OF_YEAR -> d.getDayOfYear();
                case QUARTER -> (d.getMonthValue() - 1) / 3 + 1;
                case DAY_OF_WEEK ->
                        d.getDayOfWeek()
                         .plus(1)
                         .getValue(); // Sunday = 1, as Spark
                case WEEKDAY -> d.getDayOfWeek().getValue() - 1; // Monday = 0, as Spark
            };
            out.setAtIndex(VectorBuffers.LE_INT, i, v);
        }
    }

    /** Oracle for {@code DateKernels.trunc}. */
    public static void dateTrunc(DateKernels.TruncUnit unit, VectorBuffers a, MemorySegment out) {
        int n = a.length();
        for (int i = 0; i < n; i++) {
            java.time.LocalDate d = java.time.LocalDate.ofEpochDay(a.getInt(i));
            java.time.LocalDate t = switch (unit) {
                case YEAR -> d.withDayOfYear(1);
                case QUARTER -> d.withMonth((d.getMonthValue() - 1) / 3 * 3 + 1).withDayOfMonth(1);
                case MONTH -> d.withDayOfMonth(1);
                case WEEK -> d.with(java.time.DayOfWeek.MONDAY);
            };
            out.setAtIndex(VectorBuffers.LE_INT, i, (int) t.toEpochDay());
        }
    }

    /**
     * Oracle for {@code DateKernels.timestampToDate} / {@code timeField}:
     * {@code java.time} in the given fixed offset.
     */
    public static void timestampField(DateKernels.TimeField field, boolean toDate, VectorBuffers a,
            int offsetSeconds, MemorySegment out) {
        int n = a.length();
        java.time.ZoneOffset zone = java.time.ZoneOffset.ofTotalSeconds(offsetSeconds);
        for (int i = 0; i < n; i++) {
            long micros = a.getLong(i);
            java.time.Instant instant = java.time.Instant.ofEpochSecond(Math.floorDiv(micros, 1_000_000L), Math.floorMod(micros, 1_000_000L) * 1000L);
            java.time.LocalDateTime local = java.time.LocalDateTime.ofInstant(instant, zone);
            int v;
            if (toDate) {
                v = (int) local.toLocalDate().toEpochDay();
            } else {
                v = switch (field) {
                            case HOUR -> local.getHour();
                            case MINUTE -> local.getMinute();
                            case SECOND -> local.getSecond();
                        };
            }
            out.setAtIndex(VectorBuffers.LE_INT, i, v);
        }
    }

    // ---------------------------------------------------------------- arithmetic

    public static void arith(ArithOp op, VectorBuffers a, VectorBuffers b,
            MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_INT, (long) i << 2,
                            applyInt(op, a.getInt(i), b.getInt(i)));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_LONG, (long) i << 3,
                            applyLong(op, a.getLong(i), b.getLong(i)));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_DOUBLE, (long) i << 3,
                            applyDouble(op, a.getDouble(i), b.getDouble(i)));
                }
            }
            default -> throw new IllegalArgumentException("unsupported " + a.type());
        }
    }

    public static void arithScalar(ArithOp op, VectorBuffers a, Number s,
            MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_INT, (long) i << 2,
                            applyInt(op, a.getInt(i), s.intValue()));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_LONG, (long) i << 3,
                            applyLong(op, a.getLong(i), s.longValue()));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_DOUBLE, (long) i << 3,
                            applyDouble(op, a.getDouble(i), s.doubleValue()));
                }
            }
            default -> throw new IllegalArgumentException("unsupported " + a.type());
        }
    }

    public static void scalarArith(ArithOp op, Number s, VectorBuffers b,
            MemorySegment out) {
        int n = b.length();
        switch (b.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_INT, (long) i << 2,
                            applyInt(op, s.intValue(), b.getInt(i)));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_LONG, (long) i << 3,
                            applyLong(op, s.longValue(), b.getLong(i)));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_DOUBLE, (long) i << 3,
                            applyDouble(op, s.doubleValue(), b.getDouble(i)));
                }
            }
            default -> throw new IllegalArgumentException("unsupported " + b.type());
        }
    }

    public static void negate(VectorBuffers a, MemorySegment out) {
        int n = a.length();
        switch (a.type()) {
            case INT32 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_INT, (long) i << 2, -a.getInt(i));
                }
            }
            case INT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_LONG, (long) i << 3, -a.getLong(i));
                }
            }
            case FLOAT64 -> {
                for (int i = 0; i < n; i++) {
                    out.set(VectorBuffers.LE_DOUBLE, (long) i << 3, -a.getDouble(i));
                }
            }
            default -> throw new IllegalArgumentException("unsupported " + a.type());
        }
    }

    private static int applyInt(ArithOp op, int x, int y) {
        return switch (op) {
            case ADD -> x + y;
            case SUB -> x - y;
            case MUL -> x * y;
            case DIV -> throw new IllegalArgumentException("integer DIV");
        };
    }

    private static long applyLong(ArithOp op, long x, long y) {
        return switch (op) {
            case ADD -> x + y;
            case SUB -> x - y;
            case MUL -> x * y;
            case DIV -> throw new IllegalArgumentException("integer DIV");
        };
    }

    private static double applyDouble(ArithOp op, double x, double y) {
        return switch (op) {
            case ADD -> x + y;
            case SUB -> x - y;
            case MUL -> x * y;
            case DIV -> x / y;
        };
    }

    public static void cast(VectorBuffers a, VecType target, MemorySegment out) {
        int n = a.length();
        for (int i = 0; i < n; i++) {
            switch (a.type()) {
                case INT32 -> {
                    switch (target) {
                        case INT64 -> out.set(VectorBuffers.LE_LONG, (long) i << 3, (long) a.getInt(i));
                        case FLOAT64 -> out.set(VectorBuffers.LE_DOUBLE, (long) i << 3, (double) a.getInt(i));
                        default -> throw new IllegalArgumentException("unsupported cast");
                    }
                }
                case INT64 -> out.set(VectorBuffers.LE_DOUBLE, (long) i << 3, (double) a.getLong(i));
                default -> throw new IllegalArgumentException("unsupported cast");
            }
        }
    }

    // ---------------------------------------------------------------- overflow

    /**
     * Row-wise oracle for {@code OverflowKernels}: whether {@code a[i] <op>
     * b[i]} (or against the scalar {@code s} when {@code b} is null; {@code s
     * <op> a[i]} when {@code reversed}) overflows the lane, decided by {@link
     * Math#addExact} and friends.
     */
    public static boolean overflows(ArithOp op, VectorBuffers a, VectorBuffers b,
            Number s, boolean reversed, int i) {
        try {
            if (a.type() == VecType.INT32) {
                int x = a.getInt(i);
                int y = b != null ? b.getInt(i) : s.intValue();
                if (reversed) {
                    int t = x;
                    x = y;
                    y = t;
                }
                switch (op) {
                    case ADD -> Math.addExact(x, y);
                    case SUB -> Math.subtractExact(x, y);
                    case MUL -> Math.multiplyExact(x, y);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            } else {
                long x = a.getLong(i);
                long y = b != null ? b.getLong(i) : s.longValue();
                if (reversed) {
                    long t = x;
                    x = y;
                    y = t;
                }
                switch (op) {
                    case ADD -> Math.addExact(x, y);
                    case SUB -> Math.subtractExact(x, y);
                    case MUL -> Math.multiplyExact(x, y);
                    default -> throw new IllegalArgumentException("no overflow check for " + op);
                }
            }
            return false;
        } catch (ArithmeticException e) {
            return true;
        }
    }

    /** Whether {@code -a[i]} overflows. */
    public static boolean negateOverflows(VectorBuffers a, int i) {
        try {
            if (a.type() == VecType.INT32) {
                Math.negateExact(a.getInt(i));
            } else {
                Math.negateExact(a.getLong(i));
            }
            return false;
        } catch (ArithmeticException e) {
            return true;
        }
    }

    // ---------------------------------------------------------------- selection

    /**
     * Row-wise oracle for {@code SelectKernels.select}: the value of row {@code
     * i} as a boxed Java value ({@code Integer}, {@code Long}, {@code Double},
     * {@code Boolean} or {@code String}), or {@code null}. The first branch
     * whose mask bit is set wins; then {@code otherwise}; then null. Rows
     * outside {@code active} are null.
     */
    public static Object select(int i, MemorySegment[] wins, VectorBuffers[] branches,
            VectorBuffers otherwise, MemorySegment active) {
        if (active != null && !Bitmap.isSet(active, i)) {
            return null;
        }
        VectorBuffers src = otherwise;
        for (int k = 0; k < wins.length; k++) {
            if (Bitmap.isSet(wins[k], i)) {
                src = branches[k];
                break;
            }
        }
        if (src == null || src.isNull(i)) {
            return null;
        }
        return switch (src.type()) {
            case INT32 -> src.getInt(i);
            case INT64 -> src.getLong(i);
            case FLOAT64 -> src.getDouble(i);
            case BOOL -> src.getBoolean(i);
            case UTF8 -> src.getString(i);
            case DECIMAL128 -> src.getDecimal128(i);
        };
    }

    // ---------------------------------------------------------------- aggregation

    public static long countValid(VectorBuffers a) {
        long c = 0;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                c++;
            }
        }
        return c;
    }

    public static double sumDouble(VectorBuffers a) {
        double s = 0.0;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                s += a.getDouble(i);
            }
        }
        return s;
    }

    public static long sumLong(VectorBuffers a) {
        long s = 0;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                s += a.type() == VecType.INT32 ? a.getInt(i) : a.getLong(i);
            }
        }
        return s;
    }

    /**
     * Spark ordering: NaN is the largest double. Returns +Inf when nothing is
     * valid.
     */
    public static double minDouble(VectorBuffers a) {
        double best = Double.POSITIVE_INFINITY;
        boolean any = false;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                double v = a.getDouble(i);
                if (!any || CompareOp.nanSafeCompare(v, best) < 0) {
                    best = v;
                }
                any = true;
            }
        }
        return best;
    }

    public static double maxDouble(VectorBuffers a) {
        double best = Double.NEGATIVE_INFINITY;
        boolean any = false;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                double v = a.getDouble(i);
                if (!any || CompareOp.nanSafeCompare(v, best) > 0) {
                    best = v;
                }
                any = true;
            }
        }
        return best;
    }

    public static long minLong(VectorBuffers a) {
        long best = Long.MAX_VALUE;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                best = Math.min(best,
                        a.type() == VecType.INT32 ? a.getInt(i) : a.getLong(i));
            }
        }
        return best;
    }

    public static long maxLong(VectorBuffers a) {
        long best = Long.MIN_VALUE;
        for (int i = 0; i < a.length(); i++) {
            if (!a.isNull(i)) {
                best = Math.max(best,
                        a.type() == VecType.INT32 ? a.getInt(i) : a.getLong(i));
            }
        }
        return best;
    }

    // ---------------------------------------------------------------- bitmaps

    public static void and(MemorySegment a, MemorySegment b, MemorySegment out,
                           int n) {
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i, Bitmap.isSet(a, i) && Bitmap.isSet(b, i));
        }
    }

    public static void or(MemorySegment a, MemorySegment b, MemorySegment out,
                          int n) {
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i, Bitmap.isSet(a, i) || Bitmap.isSet(b, i));
        }
    }

    public static void not(MemorySegment a, MemorySegment out, int n) {
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i, !Bitmap.isSet(a, i));
        }
    }

    /** Bits set where the element is true and not null. */
    public static void selection(MemorySegment bits, MemorySegment validity, MemorySegment out,
            int n) {
        for (int i = 0; i < n; i++) {
            Bitmap.setTo(out, i,
                    Bitmap.isSet(bits, i) && (validity == null || Bitmap.isSet(validity, i)));
        }
    }

    /**
     * Three-valued AND: null if either side is null unless the other side is
     * false.
     */
    public static void kleeneAnd(
            MemorySegment aBits,
            MemorySegment aValid,
            MemorySegment bBits,
            MemorySegment bValid,
            MemorySegment outBits,
            MemorySegment outValid,
            int n) {
        for (int i = 0; i < n; i++) {
            Boolean a = value(aBits, aValid, i);
            Boolean b = value(bBits, bValid, i);
            Boolean r;
            if (Boolean.FALSE.equals(a) || Boolean.FALSE.equals(b)) {
                r = Boolean.FALSE;
            } else if (a == null || b == null) {
                r = null;
            } else {
                r = Boolean.TRUE;
            }
            Bitmap.setTo(outValid, i, r != null);
            Bitmap.setTo(outBits, i, Boolean.TRUE.equals(r));
        }
    }

    /** Three-valued OR: null if either side is null unless the other side is true. */
    public static void kleeneOr(
            MemorySegment aBits,
            MemorySegment aValid,
            MemorySegment bBits,
            MemorySegment bValid,
            MemorySegment outBits,
            MemorySegment outValid,
            int n) {
        for (int i = 0; i < n; i++) {
            Boolean a = value(aBits, aValid, i);
            Boolean b = value(bBits, bValid, i);
            Boolean r;
            if (Boolean.TRUE.equals(a) || Boolean.TRUE.equals(b)) {
                r = Boolean.TRUE;
            } else if (a == null || b == null) {
                r = null;
            } else {
                r = Boolean.FALSE;
            }
            Bitmap.setTo(outValid, i, r != null);
            Bitmap.setTo(outBits, i, Boolean.TRUE.equals(r));
        }
    }

    private static Boolean value(MemorySegment bits, MemorySegment valid, int i) {
        if (valid != null && !Bitmap.isSet(valid, i)) {
            return null;
        }
        return Bitmap.isSet(bits, i);
    }

    // ---------------------------------------------------------------- compaction

    /** Number of selected elements. */
    public static int selectedCount(MemorySegment selection, int n) {
        return Bitmap.popcount(selection, n);
    }

    /**
     * Copies selected elements of a fixed-width or BOOL column into the output
     * buffers. Output validity may be {@code null} when the input has no nulls.
     */
    public static void compactFixed(VectorBuffers in, MemorySegment selection, MemorySegment outData,
            MemorySegment outValidity) {
        int n = in.length();
        int o = 0;
        for (int i = 0; i < n; i++) {
            if (!Bitmap.isSet(selection, i)) {
                continue;
            }
            switch (in.type()) {
                case INT32 -> outData.set(VectorBuffers.LE_INT, (long) o << 2, in.getInt(i));
                case INT64 -> outData.set(VectorBuffers.LE_LONG, (long) o << 3, in.getLong(i));
                case FLOAT64 -> outData.set(VectorBuffers.LE_DOUBLE, (long) o << 3, in.getDouble(i));
                case BOOL -> Bitmap.setTo(outData, o, in.getBoolean(i));
                case DECIMAL128 -> setDecimal128(outData, o, in.getDecimal128(i));
                default -> throw new IllegalArgumentException("not fixed width: " + in.type());
            }
            if (outValidity != null) {
                Bitmap.setTo(outValidity, o, !in.isNull(i));
            }
            o++;
        }
    }

    /** Total UTF-8 bytes of the selected, non-null elements. */
    public static long selectedUtf8Bytes(VectorBuffers in, MemorySegment selection) {
        if (in.type() != VecType.UTF8 || in.isDictionaryEncoded()) {
            throw new IllegalArgumentException("expected plain UTF8");
        }
        long total = 0;
        MemorySegment off = in.offsets();
        for (int i = 0; i < in.length(); i++) {
            if (Bitmap.isSet(selection, i) && !in.isNull(i)) {
                total += off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) i << 2);
            }
        }
        return total;
    }

    public static void compactUtf8(VectorBuffers in, MemorySegment selection, MemorySegment outOffsets,
            MemorySegment outData, MemorySegment outValidity) {
        int n = in.length();
        int o = 0;
        int pos = 0;
        MemorySegment off = in.offsets();
        for (int i = 0; i < n; i++) {
            if (!Bitmap.isSet(selection, i)) {
                continue;
            }
            outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
            boolean valid = !in.isNull(i);
            if (valid) {
                int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
                int end = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2);
                MemorySegment.copy(in.data(), ValueLayout.JAVA_BYTE, start, outData, ValueLayout.JAVA_BYTE,
                        pos, end - start);
                pos += end - start;
            }
            if (outValidity != null) {
                Bitmap.setTo(outValidity, o, valid);
            }
            o++;
        }
        outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
    }

    // ---------------------------------------------------------------- decimals (BigDecimal oracle)

    private static java.math.BigDecimal decimal(VectorBuffers a, int i, int scale) {
        return new java.math.BigDecimal(java.math.BigInteger.valueOf(a.getLong(i)), scale);
    }

    /** {@code round_half_up(a / 10^power)}. */
    public static void divPow10HalfUp(VectorBuffers a, int power, MemorySegment out) {
        for (int i = 0; i < a.length(); i++) {
            long v = decimal(a, i, power).setScale(0, java.math.RoundingMode.HALF_UP).longValueExact();
            out.set(VectorBuffers.LE_LONG, (long) i << 3, v);
        }
    }

    /** {@code |a| > bound} as a bitmap; returns the count. */
    public static int outOfRange(VectorBuffers a, long bound, MemorySegment outBits) {
        int count = 0;
        for (int i = 0; i < a.length(); i++) {
            long v = a.getLong(i);
            boolean out = v > bound || v < -bound;
            Bitmap.setTo(outBits, i, out);
            if (out) {
                count++;
            }
        }
        return count;
    }

    /**
     * Spark's decimal division: {@code BigDecimal.divide(divisor, 39, HALF_UP)}
     * then {@code setScale(resultScale, HALF_UP)}; zero divisors give 0,
     * results beyond the precision are flagged.
     */
    public static int divide(
            VectorBuffers a,
            VectorBuffers b,
            int s1,
            int s2,
            int resultScale,
            int resultPrecision,
            MemorySegment out,
            MemorySegment overflowBits) {
        int count = 0;
        java.math.BigInteger bound = java.math.BigInteger.TEN.pow(resultPrecision).subtract(java.math.BigInteger.ONE);
        for (int i = 0; i < a.length(); i++) {
            long result = 0L;
            boolean overflow = false;
            if (b.getLong(i) != 0L) {
                java.math.BigInteger u = decimal(a, i, s1).divide(decimal(b, i, s2), 39, java.math.RoundingMode.HALF_UP)
                        .setScale(resultScale, java.math.RoundingMode.HALF_UP)
                        .unscaledValue();
                if (u.abs().compareTo(bound) > 0) {
                    overflow = true;
                } else {
                    result = u.longValueExact();
                }
            }
            Bitmap.setTo(overflowBits, i, overflow);
            if (overflow) {
                count++;
            }
            out.set(VectorBuffers.LE_LONG, (long) i << 3, result);
        }
        return count;
    }

    /** {@code BigDecimal.doubleValue()}. */
    public static void toDouble(VectorBuffers a, int scale, MemorySegment out) {
        for (int i = 0; i < a.length(); i++) {
            out.set(VectorBuffers.LE_DOUBLE, (long) i << 3, decimal(a, i, scale).doubleValue());
        }
    }

    /**
     * Integer to decimal: {@code v * 10^scale}, flagged when it exceeds the
     * precision.
     */
    public static int fromIntegral(VectorBuffers a, int precision, int scale,
            MemorySegment out, MemorySegment invalidBits) {
        int count = 0;
        java.math.BigInteger bound = java.math.BigInteger.TEN.pow(precision).subtract(java.math.BigInteger.ONE);
        for (int i = 0; i < a.length(); i++) {
            long v = a.type() == VecType.INT32 ? a.getInt(i) : a.getLong(i);
            java.math.BigInteger u = java.math.BigInteger.valueOf(v).multiply(java.math.BigInteger.TEN.pow(scale));
            boolean bad = u.abs().compareTo(bound) > 0;
            Bitmap.setTo(invalidBits, i, bad);
            if (bad) {
                count++;
            }
            out.set(VectorBuffers.LE_LONG, (long) i << 3,
                    bad ? 0L : u.longValueExact());
        }
        return count;
    }

    // ---------------------------------------------------------------- gather

    /**
     * Writes a 128-bit value through its big-endian two's complement bytes,
     * byte by byte, so the reference never touches the limb layout the kernels
     * use.
     */
    public static void setDecimal128(MemorySegment out, int o, java.math.BigInteger v) {
        byte[] be = v.toByteArray();
        long base = (long) o << 4;
        byte fill = v.signum() < 0 ? (byte) -1 : 0;
        for (int k = 0; k < 16; k++) {
            int fromEnd = 15 - k; // byte k of the big-endian 16-byte form counts from the most significant
            int idx = be.length - 1 - fromEnd;
            byte b = idx >= 0 ? be[idx] : fill;
            out.set(java.lang.foreign.ValueLayout.JAVA_BYTE, base + (15 - k), b); // little-endian slot
        }
    }

    /**
     * Spark's decimal arithmetic on one row, as BigDecimal: the exact result,
     * then {@code toPrecision(precision, resultScale, HALF_UP)}. Returns the
     * unscaled result, {@code null} when it overflows the precision; division
     * by zero throws. Division is Spark's {@code Decimal./}: {@code
     * divide(divisor, 38, HALF_UP)} first.
     */
    public static java.math.BigInteger decimalOp(char op, java.math.BigDecimal a, java.math.BigDecimal b,
            int resultScale, int precision) {
        java.math.BigDecimal exact = switch (op) {
            case '+' -> a.add(b);
            case '-' -> a.subtract(b);
            case '*' -> a.multiply(b);
            case '/' -> a.divide(b, 38, java.math.RoundingMode.HALF_UP);
            default -> throw new IllegalArgumentException("op " + op);
        };
        java.math.BigInteger u = exact.setScale(resultScale, java.math.RoundingMode.HALF_UP).unscaledValue();
        return u.abs().compareTo(java.math.BigInteger.TEN.pow(precision)) < 0 ? u : null;
    }

    /** Signed order of two 128-bit values via {@code BigInteger}. */
    public static int compareDecimal128(VectorBuffers a, int i, VectorBuffers b,
            int j) {
        return a.getDecimal128(i).compareTo(b.getDecimal128(j));
    }

    /** {@code out[o] = in[idx[from + o]]}; {@code -1} is a null. */
    public static void gatherFixed(VectorBuffers in, int[] idx, int from,
            int to, MemorySegment outData, MemorySegment outValidity) {
        VecType type = in.isDictionaryEncoded() ? VecType.INT32 : in.type();
        for (int o = 0; o < to - from; o++) {
            int i = idx[from + o];
            boolean valid = i >= 0 && !in.isNull(i);
            switch (type) {
                case INT32 -> outData.set(VectorBuffers.LE_INT, (long) o << 2,
                        i < 0 ? 0 : in.getInt(i));
                case INT64 -> outData.set(VectorBuffers.LE_LONG, (long) o << 3,
                        i < 0 ? 0L : in.getLong(i));
                case FLOAT64 -> outData.set(VectorBuffers.LE_DOUBLE, (long) o << 3,
                        i < 0 ? 0.0 : in.getDouble(i));
                case BOOL -> Bitmap.setTo(outData, o, i >= 0 && in.getBoolean(i));
                case DECIMAL128 -> setDecimal128(outData, o,
                        i < 0 ? java.math.BigInteger.ZERO : in.getDecimal128(i));
                default -> throw new IllegalArgumentException("not fixed width: " + type);
            }
            if (outValidity != null) {
                Bitmap.setTo(outValidity, o, valid);
            }
        }
    }

    public static void gatherUtf8(
            VectorBuffers in,
            int[] idx,
            int from,
            int to,
            MemorySegment outOffsets,
            MemorySegment outData,
            MemorySegment outValidity) {
        int pos = 0;
        for (int o = 0; o < to - from; o++) {
            int i = idx[from + o];
            outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
            boolean valid = i >= 0 && !in.isNull(i);
            if (valid) {
                byte[] bytes = in.getUtf8Bytes(i);
                MemorySegment.copy(bytes, 0, outData, ValueLayout.JAVA_BYTE, pos, bytes.length);
                pos += bytes.length;
            }
            if (outValidity != null) {
                Bitmap.setTo(outValidity, o, valid);
            }
        }
        outOffsets.set(VectorBuffers.LE_INT, (long) (to - from) << 2, pos);
    }
}
