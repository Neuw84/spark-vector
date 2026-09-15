package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Utilities over Arrow-layout bitmaps (validity bitmaps and boolean/selection masks).
 *
 * <p>Arrow bitmaps are LSB-first: bit {@code i} lives in byte {@code i >>> 3} at bit position
 * {@code i & 7}. Reading the bitmap as little-endian 64-bit words therefore gives a word whose bit
 * {@code k} is element {@code wordIndex * 64 + k}, which is exactly the layout expected by
 * {@code VectorMask.fromLong}.
 *
 * <p>Bitmaps may be allocated with padding beyond {@code numBits}; padding bits are never assumed
 * to be zero and every bulk operation masks the tail word.
 */
public final class Bitmap {

  /** Little-endian unaligned long view, independent of the platform's native byte order. */
  static final ValueLayout.OfLong LE_LONG =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  static final ValueLayout.OfByte BYTE = ValueLayout.JAVA_BYTE;

  private Bitmap() {}

  /** Number of bytes needed to hold {@code numBits} bits. */
  public static long bytesFor(int numBits) {
    return ((long) numBits + 7) >>> 3;
  }

  /** Number of 64-bit words needed to hold {@code numBits} bits. */
  public static int wordsFor(int numBits) {
    return (numBits + 63) >>> 6;
  }

  /**
   * Allocates a zeroed bitmap for {@code numBits} bits, rounded up to a multiple of 8 bytes so
   * that whole-word reads never run past the end of the segment.
   */
  public static MemorySegment allocate(Arena arena, int numBits) {
    long bytes = Math.max(8L, (long) wordsFor(numBits) * 8L);
    return arena.allocate(bytes, 8);
  }

  public static boolean isSet(MemorySegment bm, int index) {
    return ((bm.get(BYTE, index >>> 3) >>> (index & 7)) & 1) != 0;
  }

  public static void set(MemorySegment bm, int index) {
    long byteIndex = index >>> 3;
    bm.set(BYTE, byteIndex, (byte) (bm.get(BYTE, byteIndex) | (1 << (index & 7))));
  }

  public static void clear(MemorySegment bm, int index) {
    long byteIndex = index >>> 3;
    bm.set(BYTE, byteIndex, (byte) (bm.get(BYTE, byteIndex) & ~(1 << (index & 7))));
  }

  public static void setTo(MemorySegment bm, int index, boolean value) {
    if (value) {
      set(bm, index);
    } else {
      clear(bm, index);
    }
  }

  /** Sets or clears the first {@code numBits} bits. */
  public static void fill(MemorySegment bm, int numBits, boolean value) {
    long bytes = bytesFor(numBits);
    if (bytes == 0) {
      return;
    }
    bm.asSlice(0, bytes).fill(value ? (byte) 0xFF : (byte) 0);
    if (value) {
      int rem = numBits & 7;
      if (rem != 0) {
        long last = bytes - 1;
        bm.set(BYTE, last, (byte) (bm.get(BYTE, last) & ((1 << rem) - 1)));
      }
    }
  }

  /**
   * Returns the 64-bit word at {@code wordIndex}, with any bits at or beyond {@code numBits}
   * cleared. Safe to call on a bitmap whose backing segment holds only {@code bytesFor(numBits)}
   * bytes: the tail word is assembled byte by byte when a full 8-byte read would overrun.
   */
  public static long wordAt(MemorySegment bm, int wordIndex, int numBits) {
    long byteOffset = (long) wordIndex << 3;
    int bitsInWord = numBits - (wordIndex << 6);
    if (bitsInWord >= 64) {
      return bm.get(LE_LONG, byteOffset);
    }
    if (bitsInWord <= 0) {
      return 0L;
    }
    long word;
    if (byteOffset + 8 <= bm.byteSize()) {
      word = bm.get(LE_LONG, byteOffset);
    } else {
      word = 0L;
      long bytes = bytesFor(bitsInWord);
      for (int b = 0; b < bytes; b++) {
        word |= (bm.get(BYTE, byteOffset + b) & 0xFFL) << (b << 3);
      }
    }
    return word & ((1L << bitsInWord) - 1);
  }

  /** Writes the low {@code min(64, numBits - wordIndex*64)} bits of {@code word}. */
  public static void setWord(MemorySegment bm, int wordIndex, int numBits, long word) {
    long byteOffset = (long) wordIndex << 3;
    int bitsInWord = numBits - (wordIndex << 6);
    if (bitsInWord >= 64 || byteOffset + 8 <= bm.byteSize()) {
      bm.set(LE_LONG, byteOffset, word);
      return;
    }
    long bytes = bytesFor(bitsInWord);
    for (int b = 0; b < bytes; b++) {
      bm.set(BYTE, byteOffset + b, (byte) (word >>> (b << 3)));
    }
  }

  /** Number of set bits among the first {@code numBits}. */
  public static int popcount(MemorySegment bm, int numBits) {
    int words = wordsFor(numBits);
    int count = 0;
    for (int w = 0; w < words; w++) {
      count += Long.bitCount(wordAt(bm, w, numBits));
    }
    return count;
  }

  public static boolean allSet(MemorySegment bm, int numBits) {
    return popcount(bm, numBits) == numBits;
  }

  public static boolean noneSet(MemorySegment bm, int numBits) {
    int words = wordsFor(numBits);
    for (int w = 0; w < words; w++) {
      if (wordAt(bm, w, numBits) != 0L) {
        return false;
      }
    }
    return true;
  }
}
