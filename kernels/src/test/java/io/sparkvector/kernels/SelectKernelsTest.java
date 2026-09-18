package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.reference.ScalarReference;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** {@link SelectKernels} against the row-wise oracle and against hand-computed expectations. */
class SelectKernelsTest {

  private static Object value(VectorBuffers v, int i) {
    if (v.isNull(i)) {
      return null;
    }
    return switch (v.type()) {
      case INT32 -> v.getInt(i);
      case INT64 -> v.getLong(i);
      case FLOAT64 -> v.getDouble(i);
      case BOOL -> v.getBoolean(i);
      case UTF8 -> v.getString(i);
      case DECIMAL128 -> v.getDecimal128(i);
    };
  }

  private static void assertMatchesReference(
      VecType type, int n, MemorySegment[] wins, VectorBuffers[] branches, VectorBuffers otherwise, MemorySegment active, Arena arena) {
    VectorBuffers out = SelectKernels.select(type, n, wins, branches, otherwise, active, arena);
    assertEquals(n, out.length());
    for (int i = 0; i < n; i++) {
      assertEquals(ScalarReference.select(i, wins, branches, otherwise, active), value(out, i), "row " + i);
    }
  }

  /** Bitmap with the given rows set. */
  private static MemorySegment rows(Arena arena, int n, int... set) {
    MemorySegment bm = ArrowLayout.allocateBitmap(arena, n);
    for (int i : set) {
      Bitmap.set(bm, i);
    }
    return bm;
  }

  @Test
  void firstWinningBranchThenOtherwiseThenNull() {
    try (Arena arena = Arena.ofConfined()) {
      int n = 6;
      VectorBuffers a = ArrowLayout.ofInts(arena, new int[] {10, 11, 12, 13, 14, 15}, new boolean[] {false, false, true, false, false, false});
      VectorBuffers b = ArrowLayout.ofInts(arena, new int[] {20, 21, 22, 23, 24, 25}, null);
      VectorBuffers c = ArrowLayout.ofInts(arena, new int[] {30, 31, 32, 33, 34, 35}, new boolean[] {false, false, false, false, false, true});
      // Disjoint masks, as the caller guarantees: rows 0 and 2 take a, rows 1 and 3 take b.
      MemorySegment[] wins = {rows(arena, n, 0, 2), rows(arena, n, 1, 3)};
      VectorBuffers out = SelectKernels.select(VecType.INT32, n, wins, new VectorBuffers[] {a, b}, c, null, arena);
      assertEquals(10, out.getInt(0));
      assertEquals(21, out.getInt(1));
      assertTrue(out.isNull(2), "winning branch is null there");
      assertEquals(23, out.getInt(3));
      assertEquals(34, out.getInt(4), "otherwise");
      assertTrue(out.isNull(5), "otherwise is null there");
      assertMatchesReference(VecType.INT32, n, wins, new VectorBuffers[] {a, b}, c, null, arena);

      // No ELSE: undecided rows are null. A null-literal branch (null column) is null where it wins.
      VectorBuffers none = SelectKernels.select(VecType.INT32, n, wins, new VectorBuffers[] {a, null}, null, null, arena);
      assertEquals(10, none.getInt(0));
      assertTrue(none.isNull(1), "NULL literal branch");
      assertTrue(none.isNull(4), "no branch, no ELSE");
      assertMatchesReference(VecType.INT32, n, wins, new VectorBuffers[] {a, null}, null, null, arena);

      // Rows outside the active set come out null whatever the masks say.
      MemorySegment active = rows(arena, n, 1, 4);
      VectorBuffers act = SelectKernels.select(VecType.INT32, n, wins, new VectorBuffers[] {a, b}, c, active, arena);
      assertTrue(act.isNull(0));
      assertEquals(21, act.getInt(1));
      assertEquals(34, act.getInt(4));
      assertMatchesReference(VecType.INT32, n, wins, new VectorBuffers[] {a, b}, c, active, arena);
    }
  }

  @Test
  void everyLaneTypeIncludingStringsAndDictionaries() {
    try (Arena arena = Arena.ofConfined()) {
      int n = 5;
      MemorySegment[] wins = {rows(arena, n, 0, 3), rows(arena, n, 1)};
      assertMatchesReference(VecType.INT64, n, wins,
          new VectorBuffers[] {ArrowLayout.ofLongs(arena, new long[] {1L << 40, 2, 3, 4, 5}, null), ArrowLayout.ofLongs(arena, new long[] {-1, -2, -3, -4, -5}, new boolean[] {false, true, false, false, false})},
          ArrowLayout.ofLongs(arena, new long[] {9, 9, 9, 9, 9}, null), null, arena);
      VectorBuffers d = SelectKernels.select(VecType.FLOAT64, n, wins,
          new VectorBuffers[] {ArrowLayout.ofDoubles(arena, new double[] {0.5, 1.5, 2.5, 3.5, 4.5}, null), ArrowLayout.ofDoubles(arena, new double[] {Double.NaN, -0.0, 0, 0, 0}, null)},
          null, null, arena);
      assertEquals(0.5, d.getDouble(0));
      assertEquals(-0.0, d.getDouble(1));
      assertTrue(d.isNull(2));
      assertEquals(3.5, d.getDouble(3));
      VectorBuffers b = SelectKernels.select(VecType.BOOL, n, wins,
          new VectorBuffers[] {ArrowLayout.ofBooleans(arena, new boolean[] {true, true, true, true, true}, null), ArrowLayout.ofBooleans(arena, new boolean[] {false, false, false, false, false}, null)},
          ArrowLayout.ofBooleans(arena, new boolean[] {false, false, true, false, true}, new boolean[] {false, false, false, false, true}), null, arena);
      assertTrue(b.getBoolean(0));
      assertEquals(false, b.getBoolean(1));
      assertTrue(b.getBoolean(2));
      assertTrue(b.isNull(4));

      // Strings: plain branch, dictionary-encoded branch, a null and an empty string.
      VectorBuffers plain = ArrowLayout.ofStrings(arena, new String[] {"alpha", "", "gamma", null, "epsilon"});
      String[] dictValues = {"UNITED KINGDOM", "FRANCE"};
      int[] idx = {1, 0, 0, 1, 1};
      VectorBuffers dict = SegmentVectorBuffers.dictionaryUtf8(n, null, MemorySegment.ofArray(idx), ArrowLayout.ofStrings(arena, dictValues));
      VectorBuffers otherwise = ArrowLayout.ofStrings(arena, new String[] {"x", "x", "x", "x", "x"});
      VectorBuffers s = SelectKernels.select(VecType.UTF8, n, wins, new VectorBuffers[] {plain, dict}, otherwise, null, arena);
      assertEquals("alpha", s.getString(0));
      assertEquals("UNITED KINGDOM", s.getString(1), "through the dictionary");
      assertEquals("x", s.getString(2));
      assertTrue(s.isNull(3), "winning branch null");
      assertEquals("x", s.getString(4));
      assertMatchesReference(VecType.UTF8, n, wins, new VectorBuffers[] {plain, dict}, otherwise, null, arena);
      VectorBuffers empty = SelectKernels.select(VecType.UTF8, n, new MemorySegment[] {rows(arena, n, 1)}, new VectorBuffers[] {plain}, null, null, arena);
      assertEquals("", empty.getString(1));
      assertNull(value(empty, 0));
    }
  }

  @Test
  void randomMasksAgreeWithTheReference() {
    try (Arena arena = Arena.ofConfined()) {
      Random rnd = new Random(11);
      int n = 4096;
      int[] x = new int[n];
      int[] y = new int[n];
      int[] z = new int[n];
      boolean[] nx = new boolean[n];
      boolean[] nz = new boolean[n];
      MemorySegment w0 = ArrowLayout.allocateBitmap(arena, n);
      MemorySegment w1 = ArrowLayout.allocateBitmap(arena, n);
      MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
      for (int i = 0; i < n; i++) {
        x[i] = rnd.nextInt();
        y[i] = rnd.nextInt();
        z[i] = rnd.nextInt();
        nx[i] = rnd.nextInt(7) == 0;
        nz[i] = rnd.nextInt(5) == 0;
        int pick = rnd.nextInt(3);
        if (pick == 0) Bitmap.set(w0, i);
        if (pick == 1) Bitmap.set(w1, i);
        if (rnd.nextInt(10) != 0) Bitmap.set(active, i);
      }
      VectorBuffers[] branches = {ArrowLayout.ofInts(arena, x, nx), ArrowLayout.ofInts(arena, y, null)};
      VectorBuffers otherwise = ArrowLayout.ofInts(arena, z, nz);
      assertMatchesReference(VecType.INT32, n, new MemorySegment[] {w0, w1}, branches, otherwise, active, arena);
      assertMatchesReference(VecType.INT32, n, new MemorySegment[] {w0, w1}, branches, null, null, arena);
    }
  }
}
