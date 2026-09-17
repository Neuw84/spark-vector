package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/** {@link StringLengthKernels} against {@code java.lang.String}: code points, bytes, first code point, chr. */
class StringLengthKernelsTest {

  private static final String[] VALUES = {
    "Spark SQL", "", "a", "héllo wörld", "日本語テキスト", "😀x😀y😀", null, "ab", "  trims  ", "ñ", "0123456789", "\u0000z"
  };

  private static Integer intAt(VectorBuffers v, int i) {
    if (v.validity() != null && !Bitmap.isSet(v.validity(), i)) return null;
    return v.getInt(i);
  }

  private static Integer expected(StringLengthKernels.Measure m, String s) {
    if (s == null) return null;
    switch (m) {
      case CHARS: return s.codePointCount(0, s.length());
      case BYTES: return s.getBytes(StandardCharsets.UTF_8).length;
      case BITS: return s.getBytes(StandardCharsets.UTF_8).length * 8;
      default: return s.isEmpty() ? 0 : s.codePointAt(0);
    }
  }

  @Test
  void measuresPlainAndDictionaryColumns() {
    try (Arena arena = Arena.ofConfined()) {
      VectorBuffers plain = ArrowLayout.ofStrings(arena, VALUES);
      // A dictionary with a repeated entry so the gather is exercised.
      String[] entries = {"héllo", "", "日本", "😀", "z"};
      int[] idx = {3, 0, 1, 2, 2, 4, 0, 3, 1, 4, 0, 2};
      boolean[] nulls = new boolean[idx.length];
      nulls[5] = true;
      VectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
      VectorBuffers dict = SegmentVectorBuffers.dictionaryUtf8(idx.length, indices.validity(), indices.data(), ArrowLayout.ofStrings(arena, entries));
      for (StringLengthKernels.Measure m : StringLengthKernels.Measure.values()) {
        VectorBuffers out = StringLengthKernels.measure(m, plain, plain.validity(), arena);
        assertEquals(VecType.INT32, out.type());
        for (int i = 0; i < VALUES.length; i++) assertEquals(expected(m, VALUES[i]), intAt(out, i), m + "(" + VALUES[i] + ")");
        // The dictionary path: per entry once, then gathered -- perEntry is what it gathers from.
        int[] per = StringLengthKernels.perEntry(m, dict.dictionary());
        int[] expectedPer = new int[entries.length];
        for (int e = 0; e < entries.length; e++) expectedPer[e] = expected(m, entries[e]);
        assertArrayEquals(expectedPer, per, m + " per entry");
        VectorBuffers gathered = StringLengthKernels.measure(m, dict, dict.validity(), arena);
        for (int i = 0; i < idx.length; i++) {
          assertEquals(nulls[i] ? null : expected(m, entries[idx[i]]), intAt(gathered, i), m + " dictionary row " + i);
        }
      }
      // Malformed leads decode to U+FFFD, as Spark's toString().codePointAt(0) does.
      MemorySegment bad = MemorySegment.ofArray(new byte[] {(byte) 0x80, 'a'});
      assertEquals(0xFFFD, StringLengthKernels.firstCodePoint(bad, 0, 2));
      MemorySegment truncated = MemorySegment.ofArray(new byte[] {(byte) 0xE6});
      assertEquals(0xFFFD, StringLengthKernels.firstCodePoint(truncated, 0, 1));
      assertEquals(0x65E5, StringLengthKernels.firstCodePoint(MemorySegment.ofArray("日".getBytes(StandardCharsets.UTF_8)), 0, 3));
      assertEquals(0x1F600, StringLengthKernels.firstCodePoint(MemorySegment.ofArray("😀".getBytes(StandardCharsets.UTF_8)), 0, 4));
    }
  }

  @Test
  void chrFollowsSparkRule() {
    try (Arena arena = Arena.ofConfined()) {
      int[] ints = {65, 0, -1, 256, 233, 321, 127, 128, 255, 97, 1000000};
      boolean[] nulls = new boolean[ints.length];
      nulls[9] = true;
      VectorBuffers in = ArrowLayout.ofInts(arena, ints, nulls);
      long[] longs = new long[ints.length];
      for (int i = 0; i < ints.length; i++) longs[i] = ints[i];
      longs[10] = 4294967361L; // & 0xFF == 65
      VectorBuffers inL = ArrowLayout.ofLongs(arena, longs, nulls);
      for (VectorBuffers v : new VectorBuffers[] {in, inL}) {
        boolean wide = v.type() == VecType.INT64;
        VectorBuffers out = StringLengthKernels.chr(v, v.validity(), arena);
        for (int i = 0; i < ints.length; i++) {
          long x = wide ? longs[i] : ints[i];
          String expected = nulls[i] ? null : (x < 0 ? "" : String.valueOf((char) (x & 0xFF)));
          String got = out.validity() != null && !Bitmap.isSet(out.validity(), i) ? null : new String(out.getUtf8Bytes(i), StandardCharsets.UTF_8);
          assertEquals(expected, got, "chr(" + x + ")");
          if (!nulls[i] && x >= 0 && (x & 0xFF) >= 128) assertEquals(2, out.getUtf8Bytes(i).length);
        }
        assertNull(nulls[9] ? null : out);
      }
    }
  }
}
