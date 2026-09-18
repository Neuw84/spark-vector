package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;

/**
 * Runs of equal keys on sorted columns, for the merge join (#286). On a sorted key column a run
 * starts where a row's key differs from the previous row's: {@link #boundaries} finds every start
 * in a batch in one pass per key column -- the column compared against itself shifted by one row,
 * lane-parallel through {@link CompareKernels} for int32, int64, decimal128 and dictionary ids --
 * folded with the validity so that a block of nulls is one run and null against non-null is a
 * boundary, ORed across keys. Doubles compare through the sort's total order ({@code -0.0 == 0.0},
 * NaN equal to NaN, as Spark's join equality has it) and plain strings byte by byte, both scalar.
 *
 * <p>The equality here is the one {@link RunMerge#compareKeys} and {@link SortKernels} order by, so
 * the sort, the merge and the join agree on which rows are the same key.
 */
public final class RunKernels {

  private RunKernels() {}

  /**
   * Sets bit {@code i} of {@code out} (an {@code n}-bit bitmap, overwritten) when row {@code i}
   * starts a run of equal keys: row 0 always, then every row that differs from the previous one on
   * any of {@code keys}, all sorted by those keys.
   */
  public static void boundaries(VectorBuffers[] keys, int n, MemorySegment out) {
    int words = Bitmap.wordsFor(n);
    for (int w = 0; w < words; w++) {
      Bitmap.setWord(out, w, n, 0L);
    }
    if (n == 0) {
      return;
    }
    if (n == 1 || keys.length == 0) {
      Bitmap.setWord(out, 0, n, Bitmap.wordAt(out, 0, n) | 1L);
      return;
    }
    long[] diff = new long[Bitmap.wordsFor(n - 1)];
    MemorySegment diffSeg = MemorySegment.ofArray(diff);
    for (VectorBuffers key : keys) {
      java.util.Arrays.fill(diff, 0L);
      differsFromPrevious(key, n, diffSeg);
      fold(key, n, diff, out);
    }
    Bitmap.setWord(out, 0, n, Bitmap.wordAt(out, 0, n) | 1L);
  }

  /**
   * Bit {@code j} of {@code diff} ({@code n - 1} bits): the values of rows {@code j + 1} and
   * {@code j} differ. Null rows' values are whatever the buffer holds; {@link #fold} masks them.
   */
  private static void differsFromPrevious(VectorBuffers key, int n, MemorySegment diff) {
    int m = n - 1;
    MemorySegment d = key.data();
    switch (key.type()) {
      case INT32 -> CompareKernels.i32Eq(d, d.asSlice(4, (long) m << 2), m, true, null, diff);
      case INT64 -> CompareKernels.i64Eq(d, d.asSlice(8, (long) m << 3), m, true, null, diff);
      case DECIMAL128 -> CompareKernels.d128(d, d.asSlice(16, (long) m << 4), m, CompareOp.NE, null, diff);
      case FLOAT64 -> {
        // Spark's join equality on doubles is the sort's total order: -0.0 = 0.0, NaN = NaN.
        long prev = SortKernels.doubleKey(key.getDouble(0));
        for (int i = 1; i < n; i++) {
          long cur = SortKernels.doubleKey(key.getDouble(i));
          if (cur != prev) {
            Bitmap.set(diff, i - 1);
          }
          prev = cur;
        }
      }
      case BOOL -> {
        for (int i = 1; i < n; i++) {
          if (Bitmap.isSet(d, i) != Bitmap.isSet(d, i - 1)) {
            Bitmap.set(diff, i - 1);
          }
        }
      }
      case UTF8 -> {
        if (key.isDictionaryEncoded()) {
          // One id per distinct string in a dictionary: equal ids are equal strings and vice versa.
          CompareKernels.i32Eq(d, d.asSlice(4, (long) m << 2), m, true, null, diff);
        } else {
          MemorySegment off = key.offsets();
          int prevStart = off.get(VectorBuffers.LE_INT, 0);
          int start = off.get(VectorBuffers.LE_INT, 4);
          for (int i = 1; i < n; i++) {
            int end = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2);
            if (end - start != start - prevStart || MemorySegment.mismatch(d, prevStart, start, d, start, end) != -1) {
              Bitmap.set(diff, i - 1);
            }
            prevStart = start;
            start = end;
          }
        }
      }
      default -> throw new IllegalArgumentException("unsupported key type " + key.type());
    }
  }

  /**
   * ORs into {@code out}: bit {@code i} when rows {@code i} and {@code i - 1} are both valid and
   * their values differ, or when exactly one of them is null.
   */
  private static void fold(VectorBuffers key, int n, long[] diff, MemorySegment out) {
    MemorySegment validity = key.validity();
    int words = Bitmap.wordsFor(n);
    long prevDiff = 0L;
    long prevValid = 0L;
    for (int w = 0; w < words; w++) {
      long dw = w < diff.length ? diff[w] : 0L;
      // Shift the diff bitmap up by one: bit i of the result speaks of rows i and i - 1.
      long differs = (dw << 1) | (prevDiff >>> 63);
      prevDiff = dw;
      long boundary;
      if (validity == null) {
        boundary = differs;
      } else {
        long valid = Bitmap.wordAt(validity, w, n);
        long validPrev = (valid << 1) | (prevValid >>> 63);
        if (w == 0) {
          validPrev |= 1L; // row -1 does not exist; row 0 is forced to a boundary anyway
        }
        prevValid = valid;
        boundary = (differs & valid & validPrev) | (valid ^ validPrev);
      }
      if (boundary != 0L) {
        Bitmap.setWord(out, w, n, Bitmap.wordAt(out, w, n) | boundary);
      }
    }
  }

  /**
   * The start positions of the runs {@link #boundaries} marked, in order, followed by {@code n} as
   * the end sentinel: run {@code r} is rows {@code [starts[r], starts[r + 1])}.
   */
  public static int[] runStarts(MemorySegment boundaries, int n) {
    int words = Bitmap.wordsFor(n);
    int count = 0;
    for (int w = 0; w < words; w++) {
      count += Long.bitCount(Bitmap.wordAt(boundaries, w, n));
    }
    int[] starts = new int[count + 1];
    int o = 0;
    for (int w = 0; w < words; w++) {
      long word = Bitmap.wordAt(boundaries, w, n);
      while (word != 0L) {
        starts[o++] = (w << 6) + Long.numberOfTrailingZeros(word);
        word &= word - 1;
      }
    }
    starts[count] = n;
    return starts;
  }
}
