package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.sparkvector.kernels.StringMatchKernels.Kind;
import io.sparkvector.kernels.reference.ScalarReference;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** {@link StringMatchKernels} against the byte-array reference, plain and dictionary encoded. */
class StringMatchKernelsTest {

  private static final String[] ALPHABET = {
    "", "PROMO", "PROMO BRUSHED COPPER", "STANDARD PROMO", "green", "forest green metallic",
    "greenish", "special requests", "the special ones", "requests", "aaa", "aab", "aaab", "ab",
    "日本語", "本", "日日本", "caf\u00e9", "\u00e9", "e\u0301", "xyz", "zzzz"
  };

  private static String[] randomStrings(Random rnd, int n) {
    String[] out = new String[n];
    for (int i = 0; i < n; i++) {
      out[i] = rnd.nextInt(10) == 0 ? null : ALPHABET[rnd.nextInt(ALPHABET.length)];
    }
    return out;
  }

  private static VectorBuffers dictionaryEncoded(Arena arena, String[] values) {
    int n = values.length;
    int[] idx = new int[n];
    boolean[] nulls = new boolean[n];
    for (int i = 0; i < n; i++) {
      if (values[i] == null) {
        nulls[i] = true;
      } else {
        for (int j = 0; j < ALPHABET.length; j++) {
          if (ALPHABET[j].equals(values[i])) {
            idx[i] = j;
          }
        }
      }
    }
    SegmentVectorBuffers indices = ArrowLayout.ofInts(arena, idx, nulls);
    return SegmentVectorBuffers.dictionaryUtf8(n, indices.validity(), indices.data(), ArrowLayout.ofStrings(arena, ALPHABET));
  }

  private static void check(Arena arena, VectorBuffers a, String pattern) {
    byte[] p = pattern.getBytes(StandardCharsets.UTF_8);
    for (Kind kind : Kind.values()) {
      MemorySegment expected = ArrowLayout.allocateBitmap(arena, a.length());
      MemorySegment actual = ArrowLayout.allocateBitmap(arena, a.length());
      ScalarReference.matchUtf8(kind, a, p, expected);
      StringMatchKernels.match(kind, a, p, null, actual);
      for (int i = 0; i < a.length(); i++) {
        if (!a.isNull(i)) {
          assertEquals(Bitmap.isSet(expected, i), Bitmap.isSet(actual, i), kind + " '" + pattern + "' row " + i + " (" + a.getString(i) + ")");
        }
      }
    }
  }

  private static boolean one(Kind kind, String s, String pattern) {
    byte[] sb = s.getBytes(StandardCharsets.UTF_8);
    byte[] pb = pattern.getBytes(StandardCharsets.UTF_8);
    return StringMatchKernels.matches(kind, MemorySegment.ofArray(sb), 0, sb.length, MemorySegment.ofArray(pb), pb);
  }

  @Test
  void patternRules() {
    // Empty pattern matches everything, incl. the empty string; a longer pattern matches nothing.
    for (Kind k : Kind.values()) {
      assertTrue(one(k, "", ""), k + " empty/empty");
      assertTrue(one(k, "abc", ""), k + " empty pattern");
      assertFalse(one(k, "ab", "abc"), k + " pattern longer than the string");
      assertTrue(one(k, "abc", "abc"), k + " whole string");
    }
    assertTrue(one(Kind.PREFIX, "PROMO BRUSHED", "PROMO"));
    assertFalse(one(Kind.PREFIX, "STANDARD PROMO", "PROMO"));
    assertTrue(one(Kind.SUFFIX, "STANDARD PROMO", "PROMO"));
    assertFalse(one(Kind.SUFFIX, "PROMO BRUSHED", "PROMO"));
    assertTrue(one(Kind.CONTAINS, "forest green metallic", "green"));
    assertFalse(one(Kind.CONTAINS, "forest gren metallic", "green"));
    // The scan must not stop at a first-byte hit that fails to confirm.
    assertTrue(one(Kind.CONTAINS, "aaab", "aab"));
    assertTrue(one(Kind.CONTAINS, "abababc", "abc"));
    assertFalse(one(Kind.CONTAINS, "ababab", "abc"));
    // Multi-byte: '本' (E6 9C AC) inside '日本語'; the second byte of '日' (E6 97 A5) is not a boundary hit.
    assertTrue(one(Kind.CONTAINS, "日本語", "本"));
    assertTrue(one(Kind.PREFIX, "日本語", "日"));
    assertTrue(one(Kind.SUFFIX, "日本語", "語"));
    assertFalse(one(Kind.CONTAINS, "日日本", "本日"));
    // Byte semantics: a precomposed é does not contain the combining form.
    assertFalse(one(Kind.CONTAINS, "caf\u00e9", "e\u0301"));
    assertTrue(one(Kind.SUFFIX, "caf\u00e9", "\u00e9"));
  }

  @Test
  void plainAndDictionaryColumnsMatchTheReference() {
    try (Arena arena = Arena.ofConfined()) {
      Random rnd = new Random(3);
      for (int n : new int[] {1, 63, 64, 65, 500, 4096}) {
        String[] values = randomStrings(rnd, n);
        VectorBuffers plain = ArrowLayout.ofStrings(arena, values);
        VectorBuffers dict = dictionaryEncoded(arena, values);
        for (String pattern : new String[] {"", "PROMO", "green", "special", "requests", "aab", "本", "\u00e9", "z", "nowhere", "PROMO BRUSHED COPPER PLUS"}) {
          check(arena, plain, pattern);
          check(arena, dict, pattern);
        }
      }
    }
  }

  @Test
  void inactiveBlocksAreSkippedAndCleared() {
    try (Arena arena = Arena.ofConfined()) {
      int n = 200;
      String[] values = new String[n];
      for (int i = 0; i < n; i++) {
        values[i] = "PROMO ANODIZED";
      }
      VectorBuffers a = ArrowLayout.ofStrings(arena, values);
      MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
      Bitmap.fill(active, n, false);
      Bitmap.set(active, 130); // only block 2 is live
      byte[] p = "PROMO".getBytes(StandardCharsets.UTF_8);
      VectorBuffers dict = SegmentVectorBuffers.dictionaryUtf8(
          n, null, ArrowLayout.ofInts(arena, new int[n], null).data(), ArrowLayout.ofStrings(arena, new String[] {"PROMO ANODIZED"}));
      for (VectorBuffers col : new VectorBuffers[] {a, dict}) {
        MemorySegment out = ArrowLayout.allocateBitmap(arena, n);
        Bitmap.fill(out, n, true);
        StringMatchKernels.match(Kind.PREFIX, col, p, active, out);
        assertEquals(64, Bitmap.popcount(out, n), "only the live block was matched");
        assertTrue(Bitmap.isSet(out, 130));
        assertFalse(Bitmap.isSet(out, 0));
      }
    }
  }
}
