package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.reference.ScalarReference;
import io.sparkvector.kernels.reference.SortReference;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

class GatherSortKernelsTest {

  private static SegmentVectorBuffers strings(Arena arena, Random rnd, int n, double nullFraction) {
    String[] v = new String[n];
    for (int i = 0; i < n; i++) {
      v[i] = rnd.nextDouble() < nullFraction ? null : switch (rnd.nextInt(6)) {
        case 0 -> "";
        case 1 -> "a";
        case 2 -> "é" + rnd.nextInt(3); // multi-byte, sorts after ASCII in unsigned order
        default -> "s" + rnd.nextInt(30);
      };
    }
    return ArrowLayout.ofStrings(arena, v);
  }

  private static VectorBuffers dictionary(Arena arena, VectorBuffers plain) {
    // Re-encode a plain column as (indices, dictionary of the distinct non-null values).
    java.util.LinkedHashMap<String, Integer> ids = new java.util.LinkedHashMap<>();
    int n = plain.length();
    int[] idx = new int[n];
    boolean[] nulls = new boolean[n];
    for (int i = 0; i < n; i++) {
      if (plain.isNull(i)) {
        nulls[i] = true;
      } else {
        idx[i] = ids.computeIfAbsent(plain.getString(i), k -> ids.size());
      }
    }
    VectorBuffers dict = ArrowLayout.ofStrings(arena, ids.keySet().toArray(String[]::new));
    SegmentVectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
    return SegmentVectorBuffers.dictionaryUtf8(n, indices.validity(), indices.data(), dict);
  }

  private static int[] randomIndices(Random rnd, int count, int n, boolean withNulls) {
    int[] idx = new int[count];
    for (int i = 0; i < count; i++) {
      idx[i] = (withNulls && rnd.nextInt(5) == 0) || n == 0 ? -1 : rnd.nextInt(n);
    }
    return idx;
  }

  @Test
  void gatherFixedMatchesReferenceForEveryTypeWithNullsAndPadding() {
    Random rnd = new Random(11);
    for (int n : TestData.LENGTHS) {
      try (Arena arena = Arena.ofConfined()) {
        boolean[] nulls = TestData.nulls(rnd, n, 0.2);
        VectorBuffers[] cols = {
          TestData.ints(arena, rnd, n, nulls), TestData.longs(arena, rnd, n, nulls), TestData.doubles(arena, rnd, n, nulls),
          ArrowLayout.ofBooleans(arena, booleans(rnd, n), nulls), dictionary(arena, strings(arena, rnd, n, 0.2)),
          TestData.decimal128s(arena, rnd, n, nulls)
        };
        int count = rnd.nextInt(2 * n + 2);
        int[] idx = randomIndices(rnd, count + 5, n, true);
        int from = 3, to = 3 + count;
        for (VectorBuffers c : cols) {
          VecType physical = c.isDictionaryEncoded() ? VecType.INT32 : c.type();
          MemorySegment exp = physical == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, count) : ArrowLayout.allocateData(arena, physical, count);
          MemorySegment act = physical == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, count) : ArrowLayout.allocateData(arena, physical, count);
          MemorySegment ev = ArrowLayout.allocateBitmap(arena, count);
          MemorySegment av = ArrowLayout.allocateBitmap(arena, count);
          av.fill((byte) 0xFF);
          ScalarReference.gatherFixed(c, idx, from, to, exp, ev);
          GatherKernels.gatherFixed(c, idx, from, to, act, av);
          TestData.assertBitmapEquals(ev, av, count, c.type() + " validity");
          for (int o = 0; o < count; o++) {
            if (!Bitmap.isSet(ev, o)) {
              continue; // padded rows carry no value
            }
            switch (physical) {
              case INT32 -> assertEquals(exp.get(VectorBuffers.LE_INT, (long) o << 2), act.get(VectorBuffers.LE_INT, (long) o << 2));
              case INT64, FLOAT64 -> assertEquals(exp.get(VectorBuffers.LE_LONG, (long) o << 3), act.get(VectorBuffers.LE_LONG, (long) o << 3));
              case BOOL -> assertEquals(Bitmap.isSet(exp, o), Bitmap.isSet(act, o));
              case DECIMAL128 -> {
                assertEquals(exp.get(VectorBuffers.LE_LONG, (long) o << 4), act.get(VectorBuffers.LE_LONG, (long) o << 4), "lo @" + o);
                assertEquals(exp.get(VectorBuffers.LE_LONG, ((long) o << 4) + 8), act.get(VectorBuffers.LE_LONG, ((long) o << 4) + 8), "hi @" + o);
              }
              default -> throw new IllegalStateException();
            }
          }
        }
      }
    }
  }

  private static boolean[] booleans(Random rnd, int n) {
    boolean[] b = new boolean[n];
    for (int i = 0; i < n; i++) {
      b[i] = rnd.nextBoolean();
    }
    return b;
  }

  @Test
  void gatherUtf8MatchesReference() {
    Random rnd = new Random(12);
    for (int n : TestData.LENGTHS) {
      try (Arena arena = Arena.ofConfined()) {
        VectorBuffers s = strings(arena, rnd, n, 0.15);
        int count = rnd.nextInt(n + 3);
        int[] idx = randomIndices(rnd, count, n, true);
        long bytes = GatherKernels.gatherUtf8Bytes(s, idx, 0, count);
        MemorySegment eo = ArrowLayout.allocateOffsets(arena, count), ao = ArrowLayout.allocateOffsets(arena, count);
        MemorySegment ed = ArrowLayout.allocateBytes(arena, bytes), ad = ArrowLayout.allocateBytes(arena, bytes);
        MemorySegment ev = ArrowLayout.allocateBitmap(arena, count), av = ArrowLayout.allocateBitmap(arena, count);
        ScalarReference.gatherUtf8(s, idx, 0, count, eo, ed, ev);
        GatherKernels.gatherUtf8(s, idx, 0, count, ao, ad, av);
        TestData.assertBitmapEquals(ev, av, count, "validity");
        VectorBuffers e = SegmentVectorBuffers.utf8(count, ev, eo, ed);
        VectorBuffers a = SegmentVectorBuffers.utf8(count, av, ao, ad);
        for (int o = 0; o < count; o++) {
          if (!e.isNull(o)) {
            assertEquals(e.getString(o), a.getString(o), "@" + o);
          }
        }
      }
    }
  }

  @Test
  void sortMatchesReferenceForEveryKeyTypeDirectionAndNullOrder() {
    Random rnd = new Random(13);
    for (int n : TestData.LENGTHS) {
      try (Arena arena = Arena.ofConfined()) {
        boolean[] nulls = TestData.nulls(rnd, n, 0.15);
        VectorBuffers plain = strings(arena, rnd, n, 0.15);
        VectorBuffers[] single = {
          TestData.ints(arena, rnd, n, nulls), TestData.longs(arena, rnd, n, nulls), TestData.doubles(arena, rnd, n, nulls),
          ArrowLayout.ofBooleans(arena, booleans(rnd, n), nulls), plain, dictionary(arena, plain), TestData.ints(arena, rnd, n, null),
          TestData.decimal128s(arena, rnd, n, nulls), TestData.decimal128s(arena, rnd, n, null)
        };
        for (VectorBuffers k : single) {
          for (boolean asc : new boolean[] {true, false}) {
            for (boolean nullsFirst : new boolean[] {true, false}) {
              VectorBuffers[] keys = {k};
              int[] exp = SortReference.sortIndices(keys, new boolean[] {asc}, new boolean[] {nullsFirst}, n);
              int[] act = SortKernels.sortIndices(keys, new boolean[] {asc}, new boolean[] {nullsFirst}, n);
              assertArrayEquals(exp, act, k.type() + (k.isDictionaryEncoded() ? " dict" : "") + " asc=" + asc + " nullsFirst=" + nullsFirst + " n=" + n);
            }
          }
        }
        // Multi-key: a low-cardinality first key so the second one decides, mixed directions.
        VectorBuffers[] keys = {TestData.ints(arena, rnd, n, nulls), TestData.doubles(arena, rnd, n, null), plain, TestData.decimal128s(arena, rnd, n, nulls)};
        boolean[] asc = {true, false, true, false};
        boolean[] nf = {false, true, true, false};
        assertArrayEquals(SortReference.sortIndices(keys, asc, nf, n), SortKernels.sortIndices(keys, asc, nf, n), "multi-key n=" + n);
      }
    }
  }

  @Test
  void sortIsStable() {
    try (Arena arena = Arena.ofConfined()) {
      int n = 5000;
      int[] v = new int[n];
      for (int i = 0; i < n; i++) {
        v[i] = i % 7;
      }
      VectorBuffers k = ArrowLayout.ofInts(arena, v, null);
      int[] idx = SortKernels.sortIndices(new VectorBuffers[] {k}, new boolean[] {true}, new boolean[] {true}, n);
      for (int i = 1; i < n; i++) {
        int a = k.getInt(idx[i - 1]), b = k.getInt(idx[i]);
        assertTrue(a < b || (a == b && idx[i - 1] < idx[i]), "position " + i);
      }
    }
  }

  @Test
  void columnBuilderConcatenatesBatchesWithSelectionsAndDictionaries() {
    Random rnd = new Random(14);
    try (Arena arena = Arena.ofConfined()) {
      for (VecType type : VecType.values()) {
        ColumnBuilder builder = new ColumnBuilder(arena, type, 16);
        java.util.List<Object> expected = new java.util.ArrayList<>();
        for (int batch = 0; batch < 12; batch++) {
          int n = TestData.LENGTHS[rnd.nextInt(TestData.LENGTHS.length)];
          boolean[] nulls = batch % 3 == 0 ? null : TestData.nulls(rnd, n, 0.2);
          VectorBuffers in = switch (type) {
            case INT32 -> TestData.ints(arena, rnd, n, nulls);
            case INT64 -> TestData.longs(arena, rnd, n, nulls);
            case FLOAT64 -> TestData.doubles(arena, rnd, n, nulls);
            case BOOL -> ArrowLayout.ofBooleans(arena, booleans(rnd, n), nulls);
            case UTF8 -> batch % 2 == 0 ? strings(arena, rnd, n, 0.2) : dictionary(arena, strings(arena, rnd, n, 0.2));
            case DECIMAL128 -> TestData.decimal128s(arena, rnd, n, nulls);
          };
          MemorySegment selection = batch % 4 == 1 ? null : TestData.randomBitmap(arena, rnd, n);
          int count = selection == null ? n : Bitmap.popcount(selection, n);
          builder.append(in, selection, count);
          for (int i = 0; i < n; i++) {
            if (selection == null || Bitmap.isSet(selection, i)) {
              expected.add(in.isNull(i) ? null : value(in, i));
            }
          }
        }
        VectorBuffers v = builder.view();
        assertEquals(expected.size(), v.length(), type.toString());
        for (int i = 0; i < expected.size(); i++) {
          Object e = expected.get(i);
          assertEquals(e == null, v.isNull(i), type + " null @" + i);
          if (e != null) {
            assertEquals(e, value(v, i), type + " @" + i);
          }
        }
      }
    }
  }

  private static Object value(VectorBuffers v, int i) {
    return switch (v.type()) {
      case INT32 -> v.getInt(i);
      case INT64 -> v.getLong(i);
      case FLOAT64 -> Double.doubleToLongBits(v.getDouble(i));
      case BOOL -> v.getBoolean(i);
      case UTF8 -> v.getString(i);
      case DECIMAL128 -> v.getDecimal128(i);
    };
  }

  @Test
  void keyTableLookupFindsOnlyInsertedKeysAndNeverInserts() {
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.INT64, VecType.UTF8});
      VectorBuffers[] build = {
        ArrowLayout.ofLongs(arena, new long[] {1, 2, 3, 2}, null),
        ArrowLayout.ofStrings(arena, new String[] {"a", "b", "c", "b"})
      };
      int[] ids = new int[4];
      assertEquals(3, table.assign(build, 4, ids));
      VectorBuffers[] probe = {
        ArrowLayout.ofLongs(arena, new long[] {2, 9, 3, 1, 2}, new boolean[] {false, false, false, false, true}),
        ArrowLayout.ofStrings(arena, new String[] {"b", "b", "x", "a", "b"})
      };
      int[] out = new int[5];
      MemorySegment sel = ArrowLayout.allocateBitmap(arena, 5);
      Bitmap.fill(sel, 5, true);
      Bitmap.clear(sel, 4); // the null-key row is excluded by the caller
      assertEquals(2, table.lookup(probe, 5, out, sel));
      assertArrayEquals(new int[] {ids[1], -1, -1, ids[0], -1}, out);
      assertEquals(3, table.size(), "lookup must not insert");
    }
  }
}
