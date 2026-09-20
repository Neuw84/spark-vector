package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** {@link StringSetKernels} (#371) against a {@link HashSet} of strings, plain and dictionary encoded. */
class StringSetKernelsTest {

  private static final String[] ALPHABET = {
    "", "a", "A", "ab", "abc", "abd", "b", "s1", "s10", "s2", "MAIL", "SHIP", "AIR", "REG AIR",
    "BUILDING", "BUILDINGS", "é", "e\u0301", "日本", "日本語", "\u00ff", "\u0100", "zz", "10001", "10002", "99999"
  };

  private static String[] randomStrings(Random rnd, int n, double nullRate) {
    String[] out = new String[n];
    for (int i = 0; i < n; i++) {
      out[i] = rnd.nextDouble() < nullRate ? null : ALPHABET[rnd.nextInt(ALPHABET.length)];
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

  private static StringSetKernels.StringSet setOf(String... values) {
    byte[][] bytes = new byte[values.length][];
    for (int i = 0; i < values.length; i++) {
      bytes[i] = values[i].getBytes(StandardCharsets.UTF_8);
    }
    return new StringSetKernels.StringSet(bytes);
  }

  private static void check(Arena arena, VectorBuffers a, String[] values, String[] members, MemorySegment active) {
    Set<String> ref = new HashSet<>();
    for (String m : members) {
      ref.add(m);
    }
    MemorySegment out = ArrowLayout.allocateBitmap(arena, a.length());
    StringSetKernels.inSet(a, setOf(members), active, out);
    for (int i = 0; i < a.length(); i++) {
      if (values[i] == null || (active != null && !Bitmap.isSet(active, i))) {
        continue;
      }
      assertEquals(ref.contains(values[i]), Bitmap.isSet(out, i), "row " + i + " '" + values[i] + "'");
    }
  }

  @Test
  void membershipMatchesTheReferenceSet() {
    try (Arena arena = Arena.ofConfined()) {
      Random rnd = new Random(371);
      String[][] sets = {
        {"a"}, {"", "zz"}, {"s1", "s10", "s2", "MAIL"}, {"日本", "e\u0301", "\u00ff", "10001", "99999", "not there", "s1", "s1", "s1"},
        ALPHABET, {"nothing", "here", "matches"}
      };
      for (int n : new int[] {1, 63, 64, 65, 500, 4096}) {
        String[] values = randomStrings(rnd, n, 0.1);
        VectorBuffers plain = ArrowLayout.ofStrings(arena, values);
        VectorBuffers dict = dictionaryEncoded(arena, values);
        MemorySegment active = ArrowLayout.allocateBitmap(arena, n);
        for (int i = 0; i < n; i++) {
          if (rnd.nextBoolean()) {
            Bitmap.set(active, i);
          }
        }
        for (String[] members : sets) {
          check(arena, plain, values, members, null);
          check(arena, dict, values, members, null);
          check(arena, plain, values, members, active);
        }
      }
    }
  }

  @Test
  void manyLiteralsAndDuplicates() {
    String[] zips = new String[400];
    for (int i = 0; i < 400; i++) {
      zips[i] = String.format("%05d", 24000 + i * 7);
    }
    StringSetKernels.StringSet set = setOf(zips);
    assertEquals(400, set.size());
    MemorySegment probe = MemorySegment.ofArray("24007xx24008".getBytes(StandardCharsets.UTF_8));
    assertTrue(set.contains(probe, 0, 5)); // 24007
    assertFalse(set.contains(probe, 7, 12)); // 24008 is not a member (24000 + 7k)
    assertFalse(set.contains(probe, 0, 4)); // a prefix is not the value
    assertFalse(set.contains(probe, 0, 0)); // the empty string is not in this set
    StringSetKernels.StringSet dup = setOf("a", "a", "a", "b");
    assertEquals(4, dup.size()); // literals as given...
    assertTrue(dup.contains(MemorySegment.ofArray("ab".getBytes(StandardCharsets.UTF_8)), 0, 1)); // ...one entry answers for them
    assertTrue(dup.contains(MemorySegment.ofArray("ab".getBytes(StandardCharsets.UTF_8)), 1, 2));
  }
}
