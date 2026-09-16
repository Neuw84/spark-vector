package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class GroupedAggregationTest {

  /** Reference state per key tuple. */
  private static final class Ref {
    double sum;
    long count;
    long rows;
    Double min;
    Double max;
    long lsum;
  }

  private static Ref ref(Map<List<Object>, Ref> m, List<Object> key) {
    return m.computeIfAbsent(key, k -> new Ref());
  }

  /**
   * Runs several batches through a GroupKeyTable with int + string keys of the given cardinality
   * and checks every accumulator against the reference. Cardinality <= 64 exercises the mask
   * path, more the scatter path, and a mid value crosses from one to the other.
   */
  @ParameterizedTest
  @ValueSource(ints = {1, 4, 64, 65, 300, 5000})
  void groupedAccumulatorsMatchReference(int cardinality) {
    run(cardinality, false);
  }

  /** Same, with a per-batch selection bitmap: unselected rows get id -1 and count nowhere. */
  @ParameterizedTest
  @ValueSource(ints = {1, 4, 64, 65, 300, 5000})
  void groupedAccumulatorsHonourSelection(int cardinality) {
    run(cardinality, true);
  }

  private void run(int cardinality, boolean withSelection) {
    Random rnd = new Random(cardinality);
    GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.INT32, VecType.UTF8});
    GroupedAccumulators.DoubleSum dsum = new GroupedAccumulators.DoubleSum();
    GroupedAccumulators.LongSum lsum = new GroupedAccumulators.LongSum();
    GroupedAccumulators.Count countAll = new GroupedAccumulators.Count();
    GroupedAccumulators.Count countNonNull = new GroupedAccumulators.Count();
    GroupedAccumulators.DoubleMinMax dmin = new GroupedAccumulators.DoubleMinMax(true);
    GroupedAccumulators.DoubleMinMax dmax = new GroupedAccumulators.DoubleMinMax(false);
    Map<List<Object>, Ref> reference = new HashMap<>();
    Map<List<Object>, Integer> idOf = new HashMap<>();

    int[] lengths = {1000, 37, 2048, 64, 999};
    for (int n : lengths) {
      try (Arena arena = Arena.ofConfined()) {
        int[] k1 = new int[n];
        String[] k2 = new String[n];
        double[] vals = new double[n];
        long[] lvals = new long[n];
        boolean[] keyNulls = new boolean[n];
        boolean[] valNulls = new boolean[n];
        for (int i = 0; i < n; i++) {
          int g = rnd.nextInt(cardinality);
          k1[i] = g % 7;
          k2[i] = g % 13 == 0 ? null : "k" + (g / 7);
          keyNulls[i] = k1[i] == 3 && rnd.nextInt(5) == 0; // some null int keys
          vals[i] = rnd.nextInt(6) == 0 ? Double.NaN : rnd.nextInt(-100, 100) / 4.0;
          lvals[i] = rnd.nextInt(-1000, 1000);
          valNulls[i] = rnd.nextInt(4) == 0;
        }
        VectorBuffers key1 = ArrowLayout.ofInts(arena, k1, keyNulls);
        VectorBuffers key2 = ArrowLayout.ofStrings(arena, k2);
        VectorBuffers values = ArrowLayout.ofDoubles(arena, vals, valNulls);
        VectorBuffers longs = ArrowLayout.ofLongs(arena, lvals, valNulls);

        java.lang.foreign.MemorySegment selection = withSelection ? TestData.randomBitmap(arena, rnd, n) : null;
        int[] ids = new int[n];
        int groups = table.assign(new VectorBuffers[] {key1, key2}, n, ids, selection);
        GroupAssignment a = GroupAssignment.of(ids, n, groups, arena, selection);
        dsum.update(values, a);
        lsum.update(longs, a);
        countAll.updateAll(a);
        countNonNull.updateNonNull(values, a);
        dmin.update(values, a);
        dmax.update(values, a);

        for (int i = 0; i < n; i++) {
          if (selection != null && !Bitmap.isSet(selection, i)) {
            assertEquals(-1, ids[i], "unselected row has no group");
            continue;
          }
          List<Object> key = List.of(keyNulls[i] ? "<null>" : k1[i], Objects.requireNonNullElse(k2[i], "<null>"));
          Integer prev = idOf.putIfAbsent(key, ids[i]);
          if (prev != null) {
            assertEquals(prev.intValue(), ids[i], "stable id for " + key);
          }
          Ref r = ref(reference, key);
          r.rows++;
          if (!valNulls[i]) {
            r.count++;
            r.sum += vals[i];
            r.lsum += lvals[i];
            r.min = r.min == null || CompareOp.nanSafeCompare(vals[i], r.min) < 0 ? vals[i] : r.min;
            r.max = r.max == null || CompareOp.nanSafeCompare(vals[i], r.max) > 0 ? vals[i] : r.max;
          }
        }
      }
    }

    assertEquals(reference.size(), table.size(), "group count");
    assertEquals(idOf.size(), table.size());
    for (Map.Entry<List<Object>, Integer> e : idOf.entrySet()) {
      Ref r = reference.get(e.getKey());
      int g = e.getValue();
      String what = "group " + e.getKey();
      assertEquals(r.rows, countAll.count(g), what + " rows");
      assertEquals(r.count, countNonNull.count(g), what + " non-null");
      assertEquals(r.count, dsum.count(g), what + " sum count");
      assertEquals(r.lsum, lsum.sum(g), what + " long sum");
      if (r.count > 0) {
        if (Double.isNaN(r.sum)) {
          assertTrue(Double.isNaN(dsum.sum(g)), what + " NaN sum");
        } else {
          assertEquals(r.sum, dsum.sum(g), 1e-9, what + " sum");
        }
        assertTrue(dmin.hasValue(g) && dmax.hasValue(g));
        if (Double.isNaN(r.min)) {
          assertTrue(Double.isNaN(dmin.value(g)), what + " min NaN");
        } else {
          assertEquals(r.min.doubleValue(), dmin.value(g), what + " min");
        }
        if (Double.isNaN(r.max)) {
          assertTrue(Double.isNaN(dmax.value(g)), what + " max NaN");
        } else {
          assertEquals(r.max.doubleValue(), dmax.value(g), what + " max");
        }
      } else {
        assertFalse(dmin.hasValue(g));
      }
      // Keys read back from the table.
      Object k1 = e.getKey().get(0);
      if (k1 instanceof String) {
        assertTrue(table.isNull(0, g));
      } else {
        assertFalse(table.isNull(0, g));
        assertEquals(k1, table.getInt(0, g));
      }
      Object k2 = e.getKey().get(1);
      if ("<null>".equals(k2)) {
        assertTrue(table.isNull(1, g));
      } else {
        assertEquals(k2, table.getString(1, g));
      }
    }
  }

  @Test
  void dictionaryAndPlainStringsGroupTogether() {
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.UTF8});
      VectorBuffers plain = ArrowLayout.ofStrings(arena, new String[] {"A", "N", "R", "A", null});
      int[] ids1 = new int[5];
      table.assign(new VectorBuffers[] {plain}, 5, ids1);

      VectorBuffers dict = ArrowLayout.ofStrings(arena, new String[] {"R", "A", "N"});
      SegmentVectorBuffers idx = ArrowLayout.ofInts(arena, new int[] {1, 2, 0, 0}, new boolean[] {false, false, false, true});
      VectorBuffers encoded = SegmentVectorBuffers.dictionaryUtf8(4, idx.validity(), idx.data(), dict);
      int[] ids2 = new int[4];
      int groups = table.assign(new VectorBuffers[] {encoded}, 4, ids2);

      assertEquals(4, groups, "A, N, R and null");
      assertEquals(ids1[0], ids2[0], "A");
      assertEquals(ids1[1], ids2[1], "N");
      assertEquals(ids1[2], ids2[2], "R");
      assertEquals(ids1[4], ids2[3], "null");
    }
  }

  @Test
  void memoisedDictionaryKeysMatchPlainKeys() {
    try (Arena arena = Arena.ofConfined()) {
      // Two dictionary-encoded keys take the memoised path; the same rows as plain strings must
      // land in the same groups, in a table that already holds them.
      String[] flags = {"A", "N", "R", "N", null, "A", "R", "R"};
      String[] status = {"F", "O", "F", "F", "O", null, "O", "F"};
      GroupKeyTable plainTable = new GroupKeyTable(new VecType[] {VecType.UTF8, VecType.UTF8});
      int[] plainIds = new int[flags.length];
      plainTable.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, flags), ArrowLayout.ofStrings(arena, status)}, flags.length, plainIds);

      VectorBuffers flagDict = ArrowLayout.ofStrings(arena, new String[] {"R", "A", "N"});
      VectorBuffers statusDict = ArrowLayout.ofStrings(arena, new String[] {"O", "F"});
      SegmentVectorBuffers flagIdx = ArrowLayout.ofInts(arena, new int[] {1, 2, 0, 2, 0, 1, 0, 0}, new boolean[] {false, false, false, false, true, false, false, false});
      SegmentVectorBuffers statusIdx = ArrowLayout.ofInts(arena, new int[] {1, 0, 1, 1, 0, 0, 0, 1}, new boolean[] {false, false, false, false, false, true, false, false});
      VectorBuffers[] encoded = {
        SegmentVectorBuffers.dictionaryUtf8(8, flagIdx.validity(), flagIdx.data(), flagDict),
        SegmentVectorBuffers.dictionaryUtf8(8, statusIdx.validity(), statusIdx.data(), statusDict)
      };
      int[] dictIds = new int[8];
      GroupKeyTable dictTable = new GroupKeyTable(new VecType[] {VecType.UTF8, VecType.UTF8});
      assertEquals(plainTable.size(), dictTable.assign(encoded, 8, dictIds), "group count");
      for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
          assertEquals(plainIds[i] == plainIds[j], dictIds[i] == dictIds[j], "rows " + i + " and " + j);
        }
        assertEquals(flags[i] == null, dictTable.isNull(0, dictIds[i]));
        assertEquals(status[i] == null, dictTable.isNull(1, dictIds[i]));
        if (flags[i] != null) {
          assertEquals(flags[i], dictTable.getString(0, dictIds[i]));
        }
      }
      // A later plain batch reuses the groups the dictionary batch created.
      int[] again = new int[8];
      assertEquals(plainTable.size(), dictTable.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, flags), ArrowLayout.ofStrings(arena, status)}, 8, again));
      assertArrayEquals(dictIds, again);
    }
  }

  @Test
  void plainStringsAreEncodedOnTheFlyWhateverTheirLength() {
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.UTF8});
      // All values fit in 8 bytes: the batch is dictionary encoded on the fly by packed bytes. Empty
      // strings, 8-byte values differing only in the last byte and a null must stay distinct.
      String[] shortKeys = {"", "abcdefgh", "abcdefgX", "a", null, "", "abcdefgh", "\u00e9"};
      int[] ids = new int[shortKeys.length];
      assertEquals(6, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, shortKeys)}, shortKeys.length, ids));
      assertEquals(ids[0], ids[5], "empty strings");
      assertEquals(ids[1], ids[6], "8-byte value");
      assertNotEquals(ids[1], ids[2], "last byte differs");
      assertTrue(table.isNull(0, ids[4]));
      assertEquals("\u00e9", table.getString(0, ids[7]));

      // A batch with a longer value is encoded too (hash plus byte compare) and must reuse the
      // groups the packed batch created.
      String[] mixed = {"a", "a value longer than eight bytes", "abcdefgh", null, ""};
      int[] ids2 = new int[mixed.length];
      assertEquals(7, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, mixed)}, mixed.length, ids2));
      assertEquals(ids[3], ids2[0], "a");
      assertEquals(ids[1], ids2[2], "abcdefgh");
      assertEquals(ids[4], ids2[3], "null");
      assertEquals(ids[0], ids2[4], "empty");
      assertEquals(mixed[1], table.getString(0, ids2[1]));

      // And a short batch after that finds the long group untouched and the short ones by memo.
      int[] ids3 = new int[shortKeys.length];
      assertEquals(7, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, shortKeys)}, shortKeys.length, ids3));
      assertArrayEquals(ids, ids3);
      assertTrue(table.encodesPlainStrings());
    }
  }

  @Test
  void longPlainStringsGroupByBytesNotByFingerprint() {
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.UTF8});
      // Nation-like keys: longer than 8 bytes, few distinct, same length and shared prefixes so an
      // equal (length, hash) pair is not enough and the byte compare has to decide.
      String[] keys = {"UNITED KINGDOM", "UNITED STATES", "UNITED KINGDOM", null, "UNITED KINGDOm", "UNITED STATES", "SAUDI ARABIA"};
      int[] ids = new int[keys.length];
      assertEquals(5, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, keys)}, keys.length, ids));
      assertEquals(ids[0], ids[2], "same long value");
      assertNotEquals(ids[0], ids[4], "differs in the last byte only");
      assertNotEquals(ids[0], ids[1], "same prefix, different length");
      assertTrue(table.isNull(0, ids[3]));
      assertEquals("UNITED KINGDOm", table.getString(0, ids[4]));

      // A later batch, plain again, finds the same groups through the dictionary kept across batches.
      String[] again = {"SAUDI ARABIA", "UNITED STATES", "UNITED KINGDOM", "UNITED KINGDOM"};
      int[] ids2 = new int[again.length];
      assertEquals(5, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, again)}, again.length, ids2));
      assertEquals(ids[6], ids2[0]);
      assertEquals(ids[1], ids2[1]);
      assertEquals(ids[0], ids2[2]);
      assertEquals(ids2[2], ids2[3]);
      assertTrue(table.encodesPlainStrings());

      // The join probe over the same table encodes with the same dictionary.
      int[] probe = new int[again.length];
      assertEquals(4, table.lookup(new VectorBuffers[] {ArrowLayout.ofStrings(arena, again)}, again.length, probe, null));
      assertArrayEquals(ids2, probe);
    }
  }

  @Test
  void highCardinalityPlainStringsStopBeingEncodedAboveTheCap() {
    System.setProperty("sparkvector.agg.plainDictMaxEntries", "4");
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.UTF8});
      String[] comments = new String[12];
      for (int i = 0; i < comments.length; i++) {
        comments[i] = "comment number " + (i % 6) + " of a high-cardinality column";
      }
      int[] ids = new int[comments.length];
      // Six distinct values exceed a cap of four inside the first batch: the batch is finished on
      // the hashing path and must still produce exactly the six groups, with the repeats matched.
      assertEquals(6, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, comments)}, comments.length, ids));
      for (int i = 0; i < 6; i++) {
        assertEquals(ids[i], ids[i + 6], "row " + i);
        assertEquals(comments[i], table.getString(0, ids[i]));
      }
      assertFalse(table.encodesPlainStrings(), "the table gave up on encoding this column");

      // Later batches, including short values that would have been encoded, take the hashing path
      // and keep grouping correctly.
      String[] more = {"a", comments[3], "a", comments[0]};
      int[] ids2 = new int[more.length];
      assertEquals(7, table.assign(new VectorBuffers[] {ArrowLayout.ofStrings(arena, more)}, more.length, ids2));
      assertEquals(ids2[0], ids2[2]);
      assertEquals(ids[3], ids2[1]);
      assertEquals(ids[0], ids2[3]);
      assertFalse(table.encodesPlainStrings());
    } finally {
      System.clearProperty("sparkvector.agg.plainDictMaxEntries");
    }
  }

  @Test
  void longKeysAndBooleanKeysDistinguishNullFromZero() {
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.INT64, VecType.BOOL});
      VectorBuffers l = ArrowLayout.ofLongs(arena, new long[] {0L, 0L, 1L << 40, 0L}, new boolean[] {false, true, false, false});
      VectorBuffers b = ArrowLayout.ofBooleans(arena, new boolean[] {false, false, true, false}, new boolean[] {false, false, false, true});
      int[] ids = new int[4];
      assertEquals(4, table.assign(new VectorBuffers[] {l, b}, 4, ids));
      assertEquals(0L, table.getLong(0, ids[0]));
      assertTrue(table.isNull(0, ids[1]));
      assertEquals(1L << 40, table.getLong(0, ids[2]));
      assertTrue(table.getBoolean(1, ids[2]));
      assertTrue(table.isNull(1, ids[3]));
    }
  }

  @Test
  void writeKeysProducesArrowLayoutColumns() {
    try (Arena arena = Arena.ofConfined()) {
      GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.INT32, VecType.UTF8});
      VectorBuffers k1 = ArrowLayout.ofInts(arena, new int[] {5, 6, 5, 7}, new boolean[] {false, false, false, true});
      VectorBuffers k2 = ArrowLayout.ofStrings(arena, new String[] {"xx", "y", "xx", null});
      int[] ids = new int[4];
      int groups = table.assign(new VectorBuffers[] {k1, k2}, 4, ids);
      assertEquals(3, groups);

      MemorySegmentHolder h = new MemorySegmentHolder(arena, groups);
      table.writeKeys(0, 0, groups, h.validity, h.intData, null);
      VectorBuffers outInts = SegmentVectorBuffers.fixedWidth(VecType.INT32, groups, h.validity, h.intData);
      assertEquals(5, outInts.getInt(0));
      assertEquals(6, outInts.getInt(1));
      assertTrue(outInts.isNull(2));

      long bytes = table.utf8Bytes(1, 0, groups);
      var offsets = ArrowLayout.allocateOffsets(arena, groups);
      var data = ArrowLayout.allocateBytes(arena, bytes);
      var validity = ArrowLayout.allocateBitmap(arena, groups);
      table.writeKeys(1, 0, groups, validity, data, offsets);
      VectorBuffers outStr = SegmentVectorBuffers.utf8(groups, validity, offsets, data);
      assertEquals("xx", outStr.getString(0));
      assertEquals("y", outStr.getString(1));
      assertTrue(outStr.isNull(2));
    }
  }

  private static final class MemorySegmentHolder {
    final java.lang.foreign.MemorySegment validity;
    final java.lang.foreign.MemorySegment intData;

    MemorySegmentHolder(Arena arena, int n) {
      validity = ArrowLayout.allocateBitmap(arena, n);
      intData = ArrowLayout.allocateData(arena, VecType.INT32, n);
    }
  }
}
