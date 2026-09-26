/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;

/**
 * Date and time-of-day functions over Spark's lane representations: a date is
 * INT32 days since 1970-01-01, a timestamp INT64 microseconds since the epoch
 * (UTC). Every function is branch-free integer arithmetic per lane -- no {@code
 * LocalDate} -- using Howard Hinnant's civil-from-days / days-from-civil
 * algorithms, valid over the whole proleptic Gregorian range Spark uses. Null
 * lanes get arbitrary values; the caller shares the validity.
 *
 * <p>Timestamp functions take the session zone's fixed offset in microseconds:
 * the caller only compiles UTC and fixed-offset zones (a zone with rules would
 * need per-instant offsets), so the local time of every lane is {@code micros +
 * offset}.
 */
public final class DateKernels {
    /** Fields of a date, numbered as Spark returns them. */
    public enum Field {
        YEAR,
        MONTH,
        /** Day of the month, 1-31. */
        DAY,
        /** 1-366. */
        DAY_OF_YEAR,
        /** 1-4. */
        QUARTER,
        /** Spark's {@code dayofweek}: 1 = Sunday ... 7 = Saturday. */
        DAY_OF_WEEK,
        /** Spark's {@code weekday}: 0 = Monday ... 6 = Sunday. */
        WEEKDAY
    }

    /** Units of {@code trunc(date, unit)}. */
    public enum TruncUnit {
        YEAR,
        QUARTER,
        MONTH,
        /** Back to the Monday of the week, as Spark's {@code trunc(..., 'WEEK')}. */
        WEEK
    }

    /** Fields of a time of day. */
    public enum TimeField {
        HOUR,
        MINUTE,
        SECOND
    }

    private static final long MICROS_PER_DAY = 86_400_000_000L;
    private static final long MICROS_PER_HOUR = 3_600_000_000L;
    private static final long MICROS_PER_MINUTE = 60_000_000L;
    private static final long MICROS_PER_SECOND = 1_000_000L;

    private DateKernels() {}

    /** {@code out[i] = field(a[i])}; INT32 days in, INT32 out. */
    public static void field(Field field, VectorBuffers a, MemorySegment out) {
        requireInt32(a);
        MemorySegment d = a.data();
        int n = a.length();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_INT, i, field(field, d.getAtIndex(VectorBuffers.LE_INT, i)));
        }
    }

    /** {@code out[i] = trunc(a[i], unit)}; INT32 days in and out. */
    public static void trunc(TruncUnit unit, VectorBuffers a, MemorySegment out) {
        requireInt32(a);
        MemorySegment d = a.data();
        int n = a.length();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_INT, i, trunc(unit, d.getAtIndex(VectorBuffers.LE_INT, i)));
        }
    }

    /** {@code out[i] = date(a[i] + offset)}: INT64 micros to INT32 local days. */
    public static void timestampToDate(VectorBuffers a, long offsetMicros, MemorySegment out) {
        requireInt64(a);
        MemorySegment d = a.data();
        int n = a.length();
        for (int i = 0; i < n; i++) {
            out.setAtIndex(VectorBuffers.LE_INT, i, (int) Math.floorDiv(d.getAtIndex(VectorBuffers.LE_LONG, i) + offsetMicros, MICROS_PER_DAY));
        }
    }

    /**
     * {@code out[i] = hour/minute/second of (a[i] + offset)}: INT64 micros to
     * INT32.
     */
    public static void timeField(TimeField field, VectorBuffers a, long offsetMicros,
            MemorySegment out) {
        requireInt64(a);
        MemorySegment d = a.data();
        int n = a.length();
        for (int i = 0; i < n; i++) {
            long local = Math.floorMod(d.getAtIndex(VectorBuffers.LE_LONG, i) + offsetMicros, MICROS_PER_DAY);
            int v = switch (field) {
                case HOUR -> (int) (local / MICROS_PER_HOUR);
                case MINUTE -> (int) (local % MICROS_PER_HOUR / MICROS_PER_MINUTE);
                case SECOND -> (int) (local % MICROS_PER_MINUTE / MICROS_PER_SECOND);
            };
            out.setAtIndex(VectorBuffers.LE_INT, i, v);
        }
    }

    // ---------------------------------------------------------------- scalar core (shared with tests)

    /** The field of one date given as days since the epoch. */
    public static int field(Field field, int days) {
        switch (field) {
            case DAY_OF_WEEK:
                return Math.floorMod(days + 4, 7) + 1; // 1970-01-01 was a Thursday (5)
            case WEEKDAY:
                return Math.floorMod(days + 3, 7);
            default:
                break;
        }
        // Hinnant's civil_from_days, March-based year.
        long z = (long) days + 719468;
        long era = Math.floorDiv(z, 146097);
        long doe = z - era * 146097; // [0, 146096]
        long yoe = (doe
                - doe / 1460
                + doe / 36524
                - doe / 146096)
                / 365; // [0, 399]
        long y = yoe + era * 400;
        long doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], from March 1
        long mp = (5 * doy + 2) / 153; // [0, 11], 0 = March
        int d = (int) (doy - (153 * mp + 2) / 5 + 1); // [1, 31]
        int m = (int) (mp < 10 ? mp + 3 : mp - 9); // [1, 12]
        int year = (int) (m <= 2 ? y + 1 : y);
        return switch (field) {
            case YEAR -> year;
            case MONTH -> m;
            case DAY -> d;
            case QUARTER -> (m - 1) / 3 + 1;
            case DAY_OF_YEAR -> days - daysFromCivil(year, 1, 1) + 1;
            default -> throw new IllegalArgumentException(String.valueOf(field));
        };
    }

    /** One date truncated to the start of its unit, as days since the epoch. */
    public static int trunc(TruncUnit unit, int days) {
        if (unit == TruncUnit.WEEK) {
            return days - Math.floorMod(days + 3, 7);
        }
        int year = field(Field.YEAR, days);
        int month = field(Field.MONTH, days);
        return switch (unit) {
            case YEAR -> daysFromCivil(year, 1, 1);
            case QUARTER -> daysFromCivil(year, (month - 1) / 3 * 3 + 1, 1);
            case MONTH -> daysFromCivil(year, month, 1);
            default -> throw new IllegalArgumentException(String.valueOf(unit));
        };
    }

    // ---------------------------------------------------------------- month arithmetic (Spark's DateTimeUtils)

    private static boolean isLeap(long year) {
        return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    }

    /** Days in {@code month} of {@code year}. */
    public static int lengthOfMonth(long year, int month) {
        switch (month) {
            case 2:
                return isLeap(year) ? 29 : 28;
            case 4:
            case 6:
            case 9:
            case 11:
                return 30;
            default:
                return 31;
        }
    }

    /**
     * {@code last_day}: the last day of the date's month, as days since the
     * epoch.
     */
    public static int lastDay(int days) {
        int year = field(Field.YEAR, days);
        int month = field(Field.MONTH, days);
        return daysFromCivil(year, month, lengthOfMonth(year, month));
    }

    /**
     * {@code add_months}: {@code java.time}'s plusMonths -- the day clamped to
     * the target month's length.
     */
    public static int addMonths(int days, int months) {
        long year = field(Field.YEAR, days);
        int month = field(Field.MONTH, days);
        int day = field(Field.DAY, days);
        long total = year * 12 + (month - 1) + months;
        long y = Math.floorDiv(total, 12);
        int m = Math.floorMod(total, 12) + 1;
        return daysFromCivil((int) y, m, Math.min(day, lengthOfMonth(y, m)));
    }

    /**
     * Spark's {@code getDayOfWeekFromString} code: SU=3 MO=4 TU=5 WE=6 TH=0
     * FR=1 SA=2 (1970-01-01 was a Thursday), -1 when unknown.
     */
    public static int dayOfWeekCode(String name) {
        switch (name.toUpperCase(java.util.Locale.ROOT)) {
            case "SU":
            case "SUN":
            case "SUNDAY":
                return 3;
            case "MO":
            case "MON":
            case "MONDAY":
                return 4;
            case "TU":
            case "TUE":
            case "TUESDAY":
                return 5;
            case "WE":
            case "WED":
            case "WEDNESDAY":
                return 6;
            case "TH":
            case "THU":
            case "THURSDAY":
                return 0;
            case "FR":
            case "FRI":
            case "FRIDAY":
                return 1;
            case "SA":
            case "SAT":
            case "SATURDAY":
                return 2;
            default:
                return -1;
        }
    }

    /** {@code next_day}: Spark's {@code getNextDateForDayOfWeek}. */
    public static int nextDay(int days, int dayOfWeekCode) {
        return days + 1 + Math.floorMod(dayOfWeekCode - 1 - days, 7);
    }

    /** {@code weekofyear}: the ISO-8601 week of the week-based year. */
    public static int weekOfYear(int days) {
        int isoDow = Math.floorMod(days + 3, 7) + 1; // Monday 1 .. Sunday 7
        int thursday = days - isoDow + 4;
        int year = field(Field.YEAR, thursday);
        return (thursday - daysFromCivil(year, 1, 1)) / 7 + 1;
    }

    /**
     * {@code make_date}: days since the epoch, or {@code Integer.MIN_VALUE}
     * when the civil date is invalid.
     */
    public static int makeDate(int year, int month, int day) {
        if (year < -999999999
                || year > 999999999
                || month < 1
                || month > 12
                || day < 1
                || day > lengthOfMonth(year, month)) {
            return Integer.MIN_VALUE;
        }
        long y = month <= 2 ? (long) year - 1 : year;
        long era = Math.floorDiv(y, 400);
        long yoe = y - era * 400;
        long mp = month > 2 ? month - 3 : month + 9;
        long doy = (153 * mp + 2) / 5 + day - 1;
        long doe = yoe * 365
                + yoe / 4
                - yoe / 100
                + doy;
        long epochDay = era * 146097 + doe - 719468;
        return epochDay < Integer.MIN_VALUE + 1 || epochDay > Integer.MAX_VALUE
                ? Integer.MIN_VALUE
                : (int) epochDay;
    }

    private static final long SECONDS_PER_DAY = 86_400L;

    /**
     * {@code months_between} as Spark's {@code DateTimeUtils.monthsBetween}
     * over two instants under a fixed offset: whole months when the days of
     * month agree or both are month ends, otherwise the day-and-second
     * difference over a 31-day month, rounded to 8 places when {@code
     * roundOff}.
     */
    public static double monthsBetween(long micros1, long micros2, boolean roundOff,
            long offsetMicros) {
        int date1 = (int) Math.floorDiv(micros1 + offsetMicros, MICROS_PER_DAY);
        int date2 = (int) Math.floorDiv(micros2 + offsetMicros, MICROS_PER_DAY);
        int y1 = field(Field.YEAR, date1), m1 = field(Field.MONTH, date1), d1 = field(Field.DAY, date1);
        int y2 = field(Field.YEAR, date2), m2 = field(Field.MONTH, date2), d2 = field(Field.DAY, date2);
        int toEnd1 = lengthOfMonth(y1, m1) - d1;
        int toEnd2 = lengthOfMonth(y2, m2) - d2;
        double monthDiff = (y1 * 12 + m1) - (y2 * 12 + m2);
        if (d1 == d2 || (toEnd1 == 0 && toEnd2 == 0)) {
            return monthDiff;
        }
        long secondsInDay1 = Math.floorDiv(micros1 - (date1 * MICROS_PER_DAY - offsetMicros), 1_000_000L);
        long secondsInDay2 = Math.floorDiv(micros2 - (date2 * MICROS_PER_DAY - offsetMicros), 1_000_000L);
        long secondsDiff = (d1 - d2) * SECONDS_PER_DAY + secondsInDay1 - secondsInDay2;
        double diff = monthDiff + secondsDiff / (double) (31 * SECONDS_PER_DAY);
        if (!roundOff) {
            return diff;
        }
        return new java.math.BigDecimal(Double.toString(diff)).setScale(8, java.math.RoundingMode.HALF_UP).doubleValue();
    }

    /** Hinnant's days_from_civil. */
    public static int daysFromCivil(int year, int month, int day) {
        long y = month <= 2 ? (long) year - 1 : year;
        long era = Math.floorDiv(y, 400);
        long yoe = y - era * 400;
        long mp = month > 2 ? month - 3 : month + 9;
        long doy = (153 * mp + 2) / 5 + day - 1;
        long doe = yoe * 365
                + yoe / 4
                - yoe / 100
                + doy;
        return (int) (era * 146097 + doe - 719468);
    }

    private static void requireInt32(VectorBuffers a) {
        if (a.type() != VecType.INT32) {
            throw new IllegalArgumentException("expected INT32 days, got " + a.type());
        }
    }

    private static void requireInt64(VectorBuffers a) {
        if (a.type() != VecType.INT64) {
            throw new IllegalArgumentException("expected INT64 micros, got " + a.type());
        }
    }
}
