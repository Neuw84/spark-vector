package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.util.Random;
import org.junit.jupiter.api.Test;

class ArrowLayoutTest {

  @Test
  void fixedWidthRoundTripWithNulls() {
    Random rnd = new Random(1);
    int n = 133; // not a multiple of any lane count
    int[] ints = new int[n];
    long[] longs = new long[n];
    double[] doubles = new double[n];
    boolean[] nulls = new boolean[n];
    for (int i = 0; i < n; i++) {
      ints[i] = rnd.nextInt();
      longs[i] = rnd.nextLong();
      doubles[i] = rnd.nextDouble();
      nulls[i] = rnd.nextInt(4) == 0;
    }
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers vi = ArrowLayout.ofInts(arena, ints, nulls);
      VectorBuffers vl = ArrowLayout.ofLongs(arena, longs, nulls);
      VectorBuffers vd = ArrowLayout.ofDoubles(arena, doubles, nulls);
      assertEquals(VecType.INT32, vi.type());
      assertEquals(VecType.INT64, vl.type());
      assertEquals(VecType.FLOAT64, vd.type());
      int expectedNulls = 0;
      for (int i = 0; i < n; i++) {
        assertEquals(nulls[i], vi.isNull(i));
        assertEquals(nulls[i], vl.isNull(i));
        assertEquals(nulls[i], vd.isNull(i));
        assertEquals(ints[i], vi.getInt(i));
        assertEquals(longs[i], vl.getLong(i));
        assertEquals(doubles[i], vd.getDouble(i));
        if (nulls[i]) {
          expectedNulls++;
        }
      }
      assertEquals(expectedNulls, vi.nullCount());
      assertTrue(vi.hasNulls());
      assertTrue(vi.data().byteSize() % ArrowLayout.PAD == 0);
    }
  }

  @Test
  void noNullsDropsValidityBitmap() {
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofInts(arena, new int[] {1, 2, 3}, null);
      VectorBuffers b = ArrowLayout.ofInts(arena, new int[] {1, 2, 3}, new boolean[3]);
      assertNull(a.validity());
      assertNull(b.validity());
      assertFalse(a.hasNulls());
      assertEquals(0, b.nullCount());
    }
  }

  @Test
  void booleanRoundTrip() {
    boolean[] values = {true, false, true, true, false, false, false, true, true};
    boolean[] nulls = {false, false, true, false, false, false, false, false, true};
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers v = ArrowLayout.ofBooleans(arena, values, nulls);
      assertEquals(VecType.BOOL, v.type());
      for (int i = 0; i < values.length; i++) {
        assertEquals(nulls[i], v.isNull(i));
        if (!nulls[i]) {
          assertEquals(values[i], v.getBoolean(i));
        }
      }
    }
  }

  @Test
  void utf8RoundTripIncludingEmptyAndNull() {
    String[] values = {"a", "", null, "héllo", "wörld", null, "x".repeat(300)};
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers v = ArrowLayout.ofStrings(arena, values);
      assertEquals(VecType.UTF8, v.type());
      assertNotNull(v.offsets());
      assertEquals(2, v.nullCount());
      for (int i = 0; i < values.length; i++) {
        if (values[i] == null) {
          assertTrue(v.isNull(i));
          assertArrayEquals(new byte[0], v.getUtf8Bytes(i));
        } else {
          assertFalse(v.isNull(i));
          assertEquals(values[i], v.getString(i));
        }
      }
    }
  }

  @Test
  void dictionaryEncodedUtf8ResolvesThroughDictionary() {
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers dict = ArrowLayout.ofStrings(arena, new String[] {"A", "F", "N", "O", "R"});
      int[] idx = {4, 0, 2, 3, 1, 1};
      boolean[] nulls = {false, false, false, true, false, false};
      VectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
      VectorBuffers v =
          SegmentVectorBuffers.dictionaryUtf8(idx.length, indices.validity(), indices.data(), dict);
      assertTrue(v.isDictionaryEncoded());
      assertEquals("R", v.getString(0));
      assertEquals("A", v.getString(1));
      assertEquals("N", v.getString(2));
      assertTrue(v.isNull(3));
      assertEquals("F", v.getString(5));
    }
  }
}
