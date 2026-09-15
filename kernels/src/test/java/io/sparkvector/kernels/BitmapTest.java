package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class BitmapTest {

  @Test
  void bytesForRoundsUpToWholeBytes() {
    assertEquals(0, Bitmap.bytesFor(0));
    assertEquals(1, Bitmap.bytesFor(1));
    assertEquals(1, Bitmap.bytesFor(8));
    assertEquals(2, Bitmap.bytesFor(9));
    assertEquals(8, Bitmap.bytesFor(64));
    assertEquals(9, Bitmap.bytesFor(65));
  }

  @Test
  void setClearAndIsSetUseArrowLsbBitOrder() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment bm = Bitmap.allocate(arena, 16);
      Bitmap.set(bm, 0);
      Bitmap.set(bm, 9);
      assertEquals(0b0000_0001, bm.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 0) & 0xFF);
      assertEquals(0b0000_0010, bm.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 1) & 0xFF);
      assertTrue(Bitmap.isSet(bm, 0));
      assertTrue(Bitmap.isSet(bm, 9));
      assertFalse(Bitmap.isSet(bm, 1));
      Bitmap.clear(bm, 9);
      assertFalse(Bitmap.isSet(bm, 9));
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 1000, 4097})
  void popcountMatchesScalarReferenceAndIgnoresTrailingGarbage(int numBits) {
    Random rnd = new Random(42L + numBits);
    try (Arena arena = Arena.ofConfined()) {
      // Allocate extra bytes filled with garbage to make sure the tail is masked.
      long bytes = Bitmap.bytesFor(numBits) + 16;
      MemorySegment bm = arena.allocate(bytes, 8);
      for (long i = 0; i < bytes; i++) {
        bm.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) rnd.nextInt(256));
      }
      int expected = 0;
      for (int i = 0; i < numBits; i++) {
        if (Bitmap.isSet(bm, i)) {
          expected++;
        }
      }
      assertEquals(expected, Bitmap.popcount(bm, numBits));
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 63, 64, 65, 130})
  void wordAtMasksBitsBeyondLength(int numBits) {
    try (Arena arena = Arena.ofConfined()) {
      long bytes = Bitmap.bytesFor(numBits) + 8;
      MemorySegment bm = arena.allocate(bytes, 8);
      bm.fill((byte) 0xFF);
      int words = (numBits + 63) >>> 6;
      for (int w = 0; w < words; w++) {
        long word = Bitmap.wordAt(bm, w, numBits);
        int validBitsInWord = Math.min(64, numBits - (w << 6));
        long expected = validBitsInWord == 64 ? -1L : (1L << validBitsInWord) - 1;
        assertEquals(expected, word, "word " + w);
      }
    }
  }

  @Test
  void allSetAndNoneSetFastPaths() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment bm = Bitmap.allocate(arena, 100);
      assertTrue(Bitmap.noneSet(bm, 100));
      assertFalse(Bitmap.allSet(bm, 100));
      Bitmap.fill(bm, 100, true);
      assertTrue(Bitmap.allSet(bm, 100));
      assertFalse(Bitmap.noneSet(bm, 100));
      Bitmap.clear(bm, 99);
      assertFalse(Bitmap.allSet(bm, 100));
      assertEquals(99, Bitmap.popcount(bm, 100));
    }
  }
}
