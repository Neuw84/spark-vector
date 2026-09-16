package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;

/**
 * Date and time-of-day functions over Spark's lane representations: a date is INT32 days since
 * 1970-01-01, a timestamp INT64 microseconds since the epoch (UTC). Every function is branch-free
 * integer arithmetic per lane -- no {@code LocalDate} -- using Howard Hinnant's civil-from-days /
 * days-from-civil algorithms, valid over the whole proleptic Gregorian range Spark uses. Null lanes
 * get arbitrary values; the caller shares the validity.
 *
 * <p>Timestamp functions take the session zone's fixed offset in microseconds: the caller only
 * compiles UTC and fixed-offset zones (a zone with rules would need per-instant offsets), so the
 * local time of every lane is {@code micros + offset}.
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

  /** {@code out[i] = hour/minute/second of (a[i] + offset)}: INT64 micros to INT32. */
  public static void timeField(TimeField field, VectorBuffers a, long offsetMicros, MemorySegment out) {
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
    long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
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

  /** Hinnant's days_from_civil. */
  public static int daysFromCivil(int year, int month, int day) {
    long y = month <= 2 ? (long) year - 1 : year;
    long era = Math.floorDiv(y, 400);
    long yoe = y - era * 400;
    long mp = month > 2 ? month - 3 : month + 9;
    long doy = (153 * mp + 2) / 5 + day - 1;
    long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
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
