package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.sparkvector.kernels.DateKernels.Field;
import io.sparkvector.kernels.DateKernels.TimeField;
import io.sparkvector.kernels.DateKernels.TruncUnit;
import io.sparkvector.kernels.reference.ScalarReference;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.LocalDate;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** {@link DateKernels} against {@code java.time}: every day of eight centuries, and fixed-offset timestamps. */
class DateKernelsTest {

  private static void assertSameInts(MemorySegment expected, MemorySegment actual, int n, String what) {
    for (int i = 0; i < n; i++) {
      assertEquals(expected.getAtIndex(VectorBuffers.LE_INT, i), actual.getAtIndex(VectorBuffers.LE_INT, i), what + " row " + i + " (" + LocalDate.ofEpochDay(i) + ")");
    }
  }

  @Test
  void everyDayFrom1600To2400MatchesJavaTime() {
    try (Arena arena = Arena.ofConfined()) {
      int from = (int) LocalDate.of(1600, 1, 1).toEpochDay(); // well before the epoch: negative days
      int to = (int) LocalDate.of(2400, 12, 31).toEpochDay();
      int n = to - from + 1;
      int[] days = new int[n];
      for (int i = 0; i < n; i++) {
        days[i] = from + i;
      }
      VectorBuffers a = ArrowLayout.ofInts(arena, days, null);
      MemorySegment expected = ArrowLayout.allocateData(arena, VecType.INT32, n);
      MemorySegment actual = ArrowLayout.allocateData(arena, VecType.INT32, n);
      for (Field f : Field.values()) {
        ScalarReference.dateField(f, a, expected);
        DateKernels.field(f, a, actual);
        for (int i = 0; i < n; i++) {
          int e = expected.getAtIndex(VectorBuffers.LE_INT, i), g = actual.getAtIndex(VectorBuffers.LE_INT, i);
          if (e != g) {
            assertEquals(e, g, f + " of " + LocalDate.ofEpochDay(days[i]));
          }
        }
      }
      for (TruncUnit u : TruncUnit.values()) {
        ScalarReference.dateTrunc(u, a, expected);
        DateKernels.trunc(u, a, actual);
        for (int i = 0; i < n; i++) {
          int e = expected.getAtIndex(VectorBuffers.LE_INT, i), g = actual.getAtIndex(VectorBuffers.LE_INT, i);
          if (e != g) {
            assertEquals(e, g, "trunc " + u + " of " + LocalDate.ofEpochDay(days[i]));
          }
        }
      }
    }
  }

  @Test
  void spotChecksAtTheKnownCorners() {
    int epoch = 0;
    assertEquals(1970, DateKernels.field(Field.YEAR, epoch));
    assertEquals(1, DateKernels.field(Field.MONTH, epoch));
    assertEquals(1, DateKernels.field(Field.DAY, epoch));
    assertEquals(5, DateKernels.field(Field.DAY_OF_WEEK, epoch), "1970-01-01 was a Thursday: Spark's dayofweek is 5");
    assertEquals(3, DateKernels.field(Field.WEEKDAY, epoch), "weekday Monday=0 -> Thursday=3");
    int dec31_1969 = -1;
    assertEquals(1969, DateKernels.field(Field.YEAR, dec31_1969));
    assertEquals(12, DateKernels.field(Field.MONTH, dec31_1969));
    assertEquals(31, DateKernels.field(Field.DAY, dec31_1969));
    assertEquals(365, DateKernels.field(Field.DAY_OF_YEAR, dec31_1969));
    assertEquals(4, DateKernels.field(Field.QUARTER, dec31_1969));
    int feb29_2000 = (int) LocalDate.of(2000, 2, 29).toEpochDay(); // leap year divisible by 400
    assertEquals(29, DateKernels.field(Field.DAY, feb29_2000));
    assertEquals(60, DateKernels.field(Field.DAY_OF_YEAR, feb29_2000));
    int mar1_1900 = (int) LocalDate.of(1900, 3, 1).toEpochDay(); // 1900 is not a leap year
    assertEquals(60, DateKernels.field(Field.DAY_OF_YEAR, mar1_1900));
    int feb29_2024 = (int) LocalDate.of(2024, 2, 29).toEpochDay();
    assertEquals((int) LocalDate.of(2024, 1, 1).toEpochDay(), DateKernels.trunc(TruncUnit.YEAR, feb29_2024));
    assertEquals((int) LocalDate.of(2024, 1, 1).toEpochDay(), DateKernels.trunc(TruncUnit.QUARTER, feb29_2024));
    assertEquals((int) LocalDate.of(2024, 2, 1).toEpochDay(), DateKernels.trunc(TruncUnit.MONTH, feb29_2024));
    assertEquals((int) LocalDate.of(2024, 2, 26).toEpochDay(), DateKernels.trunc(TruncUnit.WEEK, feb29_2024), "Thursday 2024-02-29 -> Monday 2024-02-26");
    int sunday = (int) LocalDate.of(2024, 3, 3).toEpochDay();
    assertEquals(1, DateKernels.field(Field.DAY_OF_WEEK, sunday));
    assertEquals(6, DateKernels.field(Field.WEEKDAY, sunday));
    assertEquals((int) LocalDate.of(2024, 2, 26).toEpochDay(), DateKernels.trunc(TruncUnit.WEEK, sunday));
    // days_from_civil round trips.
    for (int d = -800000; d <= 800000; d += 997) {
      assertEquals(d, DateKernels.daysFromCivil(DateKernels.field(Field.YEAR, d), DateKernels.field(Field.MONTH, d), DateKernels.field(Field.DAY, d)));
    }
  }

  @Test
  void timestampsUnderFixedOffsetsMatchJavaTime() {
    try (Arena arena = Arena.ofConfined()) {
      Random rnd = new Random(7);
      int n = 5000;
      long[] micros = new long[n];
      for (int i = 0; i < n; i++) {
        // +-300 years around the epoch, microsecond resolution, incl. negative values.
        micros[i] = (long) (rnd.nextGaussian() * 300.0 * 365.25 * 86_400_000_000L);
      }
      micros[0] = 0L;
      micros[1] = -1L; // 1969-12-31T23:59:59.999999 UTC
      micros[2] = 86_399_999_999L; // last micro of 1970-01-01 UTC
      micros[3] = -86_400_000_000L;
      VectorBuffers a = ArrowLayout.ofLongs(arena, micros, null);
      MemorySegment expected = ArrowLayout.allocateData(arena, VecType.INT32, n);
      MemorySegment actual = ArrowLayout.allocateData(arena, VecType.INT32, n);
      for (int offsetSeconds : new int[] {0, 3600, -18000, 19800, -3 * 3600 - 1800, 14 * 3600}) {
        long offsetMicros = offsetSeconds * 1_000_000L;
        ScalarReference.timestampField(TimeField.HOUR, true, a, offsetSeconds, expected);
        DateKernels.timestampToDate(a, offsetMicros, actual);
        assertSameInts(expected, actual, n, "to date at offset " + offsetSeconds);
        for (TimeField f : TimeField.values()) {
          ScalarReference.timestampField(f, false, a, offsetSeconds, expected);
          DateKernels.timeField(f, a, offsetMicros, actual);
          assertSameInts(expected, actual, n, f + " at offset " + offsetSeconds);
        }
      }
    }
  }

  @Test
  void monthArithmeticWeeksAndMakeDateMatchJavaTime() {
    java.time.LocalDate from = java.time.LocalDate.of(1890, 1, 1);
    java.time.LocalDate to = java.time.LocalDate.of(2110, 12, 31);
    for (java.time.LocalDate d = from; !d.isAfter(to); d = d.plusDays(7)) {
      int days = (int) d.toEpochDay();
      assertEquals((int) d.withDayOfMonth(d.lengthOfMonth()).toEpochDay(), DateKernels.lastDay(days), "last_day " + d);
      for (int months : new int[] {-25, -13, -1, 0, 1, 11, 12, 13, 47}) {
        assertEquals((int) d.plusMonths(months).toEpochDay(), DateKernels.addMonths(days, months), "add_months(" + d + ", " + months + ")");
      }
      assertEquals(d.get(java.time.temporal.IsoFields.WEEK_OF_WEEK_BASED_YEAR), DateKernels.weekOfYear(days), "weekofyear " + d);
      assertEquals(days, DateKernels.makeDate(d.getYear(), d.getMonthValue(), d.getDayOfMonth()), "make_date " + d);
      // next_day: the first strictly later date with that weekday.
      String[] names = {"SU", "MON", "tuesday", "We", "TH", "fri", "SATURDAY"};
      java.time.DayOfWeek[] dows = {java.time.DayOfWeek.SUNDAY, java.time.DayOfWeek.MONDAY, java.time.DayOfWeek.TUESDAY, java.time.DayOfWeek.WEDNESDAY, java.time.DayOfWeek.THURSDAY, java.time.DayOfWeek.FRIDAY, java.time.DayOfWeek.SATURDAY};
      for (int k = 0; k < names.length; k++) {
        java.time.LocalDate expected = d.with(java.time.temporal.TemporalAdjusters.next(dows[k]));
        assertEquals((int) expected.toEpochDay(), DateKernels.nextDay(days, DateKernels.dayOfWeekCode(names[k])), "next_day(" + d + ", " + names[k] + ")");
      }
    }
    // Month ends and leap days for add_months.
    assertEquals((int) java.time.LocalDate.of(2024, 2, 29).toEpochDay(), DateKernels.addMonths((int) java.time.LocalDate.of(2024, 1, 31).toEpochDay(), 1));
    assertEquals((int) java.time.LocalDate.of(2023, 2, 28).toEpochDay(), DateKernels.addMonths((int) java.time.LocalDate.of(2024, 2, 29).toEpochDay(), -12));
    // make_date rejects what LocalDate.of rejects.
    assertEquals(Integer.MIN_VALUE, DateKernels.makeDate(2023, 2, 29));
    assertEquals(Integer.MIN_VALUE, DateKernels.makeDate(2023, 13, 1));
    assertEquals(Integer.MIN_VALUE, DateKernels.makeDate(2023, 4, 0));
    assertEquals(-1, DateKernels.dayOfWeekCode("xyz"));
    // months_between: Spark's documented examples under UTC and a fixed offset.
    java.util.function.BiFunction<String, String, Long> micros = (dt, tm) -> java.time.LocalDateTime.parse(dt + "T" + tm).toEpochSecond(java.time.ZoneOffset.UTC) * 1_000_000L;
    assertEquals(3.94959677, DateKernels.monthsBetween(micros.apply("1997-02-28", "10:30:00"), micros.apply("1996-10-30", "00:00:00"), true, 0), 0.0);
    assertEquals(3.9495967741935485, DateKernels.monthsBetween(micros.apply("1997-02-28", "10:30:00"), micros.apply("1996-10-30", "00:00:00"), false, 0), 0.0);
    assertEquals(1.0, DateKernels.monthsBetween(micros.apply("2024-03-31", "00:00:00"), micros.apply("2024-02-29", "00:00:00"), true, 0), 0.0); // both month ends
    assertEquals(-1.0, DateKernels.monthsBetween(micros.apply("2024-02-15", "12:00:00"), micros.apply("2024-03-15", "12:00:00"), true, 0), 0.0);
    // Under +05:30 the civil day of an instant shifts: 2024-01-31T20:00Z is Feb 1 locally.
    long offset = (5 * 3600 + 30 * 60) * 1_000_000L;
    assertEquals(0.0, DateKernels.monthsBetween(micros.apply("2024-01-31", "20:00:00"), micros.apply("2024-02-01", "01:00:00"), true, offset), 1e-12);
  }
}
