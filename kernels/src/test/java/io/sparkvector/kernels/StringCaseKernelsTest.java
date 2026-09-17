package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.StringCaseKernels.Kind;
import io.sparkvector.kernels.StringCaseKernels.Side;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/** {@link StringCaseKernels} against {@code java.lang.String} on the ASCII path; flags and trims. */
class StringCaseKernelsTest {

  private static final String[] VALUES = {"Spark SQL", "", "hello world", "MiXeD  case", null, "a b c", "x", "  padded  ", "ALL CAPS"};

  private static String read(VectorBuffers v, int i) {
    if (v.validity() != null && !Bitmap.isSet(v.validity(), i)) return null;
    return new String(v.getUtf8Bytes(i), StandardCharsets.UTF_8);
  }

  private static String initcapAscii(String s) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      boolean start = i == 0 || s.charAt(i - 1) == ' ';
      sb.append(start ? Character.toUpperCase(c) : Character.toLowerCase(c));
    }
    return sb.toString();
  }

  @Test
  void asciiCaseMappingAndFlags() {
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofStrings(arena, VALUES);
      MemorySegment flags = ArrowLayout.allocateBitmap(arena, VALUES.length);
      assertEquals(0, StringCaseKernels.slowRows(a, Kind.UPPER, true, a.validity(), flags));
      for (Kind k : Kind.values()) {
        VectorBuffers out = StringCaseKernels.caseMap(a, k, null, a.validity(), arena);
        for (int i = 0; i < VALUES.length; i++) {
          String s = VALUES[i];
          String expected = s == null ? null : k == Kind.UPPER ? s.toUpperCase(Locale.ROOT) : k == Kind.LOWER ? s.toLowerCase(Locale.ROOT) : initcapAscii(s);
          assertEquals(expected, read(out, i), k + "(" + s + ")");
        }
      }
      // Flags: non-ASCII bytes always; for a strict initcap anything but letters and spaces.
      String[] mixed = {"plain", "straße", "3abc", "o'neil", "a-b", null, "Über", "ok then"};
      VectorBuffers m = ArrowLayout.ofStrings(arena, mixed);
      MemorySegment f1 = ArrowLayout.allocateBitmap(arena, mixed.length);
      assertEquals(2, StringCaseKernels.slowRows(m, Kind.LOWER, true, m.validity(), f1));
      assertTrue(Bitmap.isSet(f1, 1) && Bitmap.isSet(f1, 6));
      assertFalse(Bitmap.isSet(f1, 2) || Bitmap.isSet(f1, 5));
      MemorySegment f2 = ArrowLayout.allocateBitmap(arena, mixed.length);
      assertEquals(5, StringCaseKernels.slowRows(m, Kind.INITCAP, true, m.validity(), f2));
      assertFalse(Bitmap.isSet(f2, 0) || Bitmap.isSet(f2, 7));
      MemorySegment f3 = ArrowLayout.allocateBitmap(arena, mixed.length);
      assertEquals(2, StringCaseKernels.slowRows(m, Kind.INITCAP, false, m.validity(), f3));
      // Overrides are copied verbatim (any length); the rest mapped.
      byte[][] overrides = new byte[mixed.length][];
      overrides[1] = "STRASSE".getBytes(StandardCharsets.UTF_8);
      overrides[6] = "ÜBER".getBytes(StandardCharsets.UTF_8);
      VectorBuffers up = StringCaseKernels.caseMap(m, Kind.UPPER, overrides, m.validity(), arena);
      String[] expected = {"PLAIN", "STRASSE", "3ABC", "O'NEIL", "A-B", null, "ÜBER", "OK THEN"};
      for (int i = 0; i < mixed.length; i++) assertEquals(expected[i], read(up, i), "row " + i);
      // A dictionary column decides per entry.
      int[] idx = {1, 0, 1, 2};
      VectorBuffers indices = ArrowLayout.ofInts(arena, idx, null);
      VectorBuffers dict = SegmentVectorBuffers.dictionaryUtf8(4, null, indices.data(), ArrowLayout.ofStrings(arena, new String[] {"abc", "ñ", "Q"}));
      MemorySegment f4 = ArrowLayout.allocateBitmap(arena, 4);
      assertEquals(2, StringCaseKernels.slowRows(dict, Kind.UPPER, true, null, f4));
      VectorBuffers du = StringCaseKernels.caseMap(dict, Kind.LOWER, null, null, arena);
      assertEquals("abc", read(du, 1));
      assertEquals("q", read(du, 3));
    }
  }

  @Test
  void trimsAreOffsetArithmetic() {
    String[] values = {"  Spark SQL  ", "", "   ", "no trim", " left", "right ", null, "xyxhixyx", "日本 語 ", "ééaéé", " \t tab"};
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers a = ArrowLayout.ofStrings(arena, values);
      for (Side side : Side.values()) {
        VectorBuffers out = StringCaseKernels.trim(a, side, null, a.validity(), arena);
        for (int i = 0; i < values.length; i++) {
          String s = values[i];
          String expected = s == null ? null : side == Side.LEFT ? s.replaceAll("^ +", "") : side == Side.RIGHT ? s.replaceAll(" +$", "") : s.replaceAll("^ +| +$", "");
          assertEquals(expected, read(out, i), side + "(" + s + ")");
        }
      }
      // A trim set of code points, including a multi-byte one.
      int[] set = {'x', 'y', 'é'};
      VectorBuffers both = StringCaseKernels.trim(a, Side.BOTH, set, a.validity(), arena);
      assertEquals("hi", read(both, 7));
      assertEquals("a", read(both, 9));
      assertEquals("  Spark SQL  ", read(both, 0));
      VectorBuffers left = StringCaseKernels.trim(a, Side.LEFT, set, a.validity(), arena);
      assertEquals("hixyx", read(left, 7));
      VectorBuffers right = StringCaseKernels.trim(a, Side.RIGHT, set, a.validity(), arena);
      assertEquals("xyxhi", read(right, 7));
      assertEquals("ééa", read(right, 9));
      // A multi-byte space-like set member and a set that empties the string.
      VectorBuffers jp = StringCaseKernels.trim(a, Side.BOTH, new int[] {' ', '日', '本', '語'}, a.validity(), arena);
      assertEquals("", read(jp, 8));
      assertEquals("Spark SQL", read(jp, 0));
    }
  }
}
