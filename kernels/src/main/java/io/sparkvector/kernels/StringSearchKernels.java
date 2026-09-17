package io.sparkvector.kernels;

import io.sparkvector.kernels.StringConcatKernels.Part;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * The search-based string functions: {@code instr} / {@code locate}, {@code replace},
 * {@code translate}, {@code substring_index}, {@code split_part} and {@code find_in_set}, all over one
 * primitive -- {@link #find}: the first occurrence of a needle at or after a byte offset (a first-byte
 * scan, candidates confirmed with {@code MemorySegment.mismatch}). Each function follows Spark's
 * {@code UTF8String} to the letter: {@code indexOf} steps by code points and reports a code-point
 * position (an empty needle is found at 0); {@code replace} and {@code substring_index} search bytes;
 * {@code translate} maps code points; {@code find_in_set} refuses a word containing a comma;
 * {@code split_part} counts parts from either end. Inputs are {@link Part}s (a lane read through its
 * dictionary, or a literal); the writers use the two-pass offsets/data pattern and the per-batch cap.
 */
public final class StringSearchKernels {
  private StringSearchKernels() {}

  /** Byte offset of the first occurrence of the needle in {@code [from, to)} of the haystack, or -1. */
  static int find(MemorySegment hay, long hStart, int from, int to, MemorySegment needle, long nStart, int nLen) {
    if (nLen == 0) return from <= to ? from : -1;
    byte first = needle.get(ValueLayout.JAVA_BYTE, nStart);
    int last = to - nLen;
    for (int i = from; i <= last; i++) {
      if (hay.get(ValueLayout.JAVA_BYTE, hStart + i) != first) continue;
      if (nLen == 1 || MemorySegment.mismatch(hay, hStart + i, hStart + i + nLen, needle, nStart, nStart + nLen) == -1) return i;
    }
    return -1;
  }

  /** Last occurrence starting at or before byte {@code start} (Spark's {@code rfind}), or -1. */
  private static int rfind(MemorySegment hay, long hStart, int len, int start, MemorySegment needle, long nStart, int nLen) {
    for (int i = Math.min(start, len - nLen); i >= 0; i--) {
      if (MemorySegment.mismatch(hay, hStart + i, hStart + i + nLen, needle, nStart, nStart + nLen) == -1) return i;
    }
    return -1;
  }

  private static int arg(VectorBuffers col, int scalar, int i) {
    return col == null ? scalar : col.getInt(i);
  }

  private static MemorySegment offsetsOf(int[] lengths, int n, Arena arena) {
    MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
    long total = 0;
    for (int i = 0; i < n; i++) {
      total += lengths[i];
      if (total > StringSliceKernels.MAX_OUTPUT_BYTES) throw new IllegalStateException("string output of more than " + StringSliceKernels.MAX_OUTPUT_BYTES + " bytes in one batch");
      offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, (int) total);
    }
    return offsets;
  }

  // ---------------------------------------------------------------- locate / instr

  /**
   * Spark's {@code UTF8String.indexOf(v, start)}: the code-point position of {@code needle} in the
   * haystack at or after code point {@code start}, or -1; an empty needle is at 0.
   */
  static int indexOf(MemorySegment hay, long hStart, int len, MemorySegment needle, long nStart, int nLen, int start) {
    if (nLen == 0) return 0;
    int i = 0;
    int c = 0;
    while (i < len && c < start) {
      i += StringSliceKernels.numBytesForFirstByte(hay.get(ValueLayout.JAVA_BYTE, hStart + i));
      c++;
    }
    do {
      if (i + nLen > len) return -1;
      if (MemorySegment.mismatch(hay, hStart + i, hStart + i + nLen, needle, nStart, nStart + nLen) == -1) return c;
      i += StringSliceKernels.numBytesForFirstByte(hay.get(ValueLayout.JAVA_BYTE, hStart + i));
      c++;
    } while (i < len);
    return -1;
  }

  /**
   * {@code locate(needle, hay, pos)} with Spark's null rules: a null {@code pos} gives 0 (not null),
   * then a null needle or haystack gives null, a {@code pos} below 1 gives 0, otherwise
   * {@code indexOf(needle, pos - 1) + 1}. {@code instr} is {@code locate} with {@code pos = 1}.
   */
  public static SegmentVectorBuffers locate(Part needle, Part hay, VectorBuffers posCol, int pos, int n, Arena arena) {
    MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT32, n);
    MemorySegment validity = ArrowLayout.allocateBitmap(arena, n);
    for (int i = 0; i < n; i++) {
      if (posCol != null && posCol.validity() != null && !Bitmap.isSet(posCol.validity(), i)) {
        Bitmap.set(validity, i);
        out.setAtIndex(VectorBuffers.LE_INT, i, 0);
        continue;
      }
      if (!needle.live(i) || !hay.live(i)) continue;
      Bitmap.set(validity, i);
      int p = arg(posCol, pos, i);
      int r = p < 1 ? 0 : indexOf(hay.seg(i), hay.at(i), hay.length(i), needle.seg(i), needle.at(i), needle.length(i), p - 1) + 1;
      out.setAtIndex(VectorBuffers.LE_INT, i, r);
    }
    return SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, out);
  }

  // ---------------------------------------------------------------- replace

  /** {@code replace(hay, search, repl)}: every byte-level occurrence; an empty haystack or search leaves the row unchanged. */
  public static SegmentVectorBuffers replace(Part hay, Part search, Part repl, int n, MemorySegment validity, Arena arena) {
    int[] lengths = new int[n];
    int[] counts = new int[n];
    for (int i = 0; i < n; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) continue;
      int len = hay.length(i);
      int sLen = search.length(i);
      int count = 0;
      if (len > 0 && sLen > 0) {
        int at = 0;
        while ((at = find(hay.seg(i), hay.at(i), at, len, search.seg(i), search.at(i), sLen)) >= 0) {
          count++;
          at += sLen;
        }
      }
      counts[i] = count;
      long total = (long) len + (long) count * (repl.length(i) - sLen);
      if (total > Integer.MAX_VALUE) throw new IllegalStateException("replace output exceeds the maximum length");
      lengths[i] = (int) total;
    }
    MemorySegment offsets = offsetsOf(lengths, n, arena);
    MemorySegment out = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
    for (int i = 0; i < n; i++) {
      if (lengths[i] == 0) continue;
      long at = offsets.getAtIndex(VectorBuffers.LE_INT, i);
      MemorySegment h = hay.seg(i);
      long hStart = hay.at(i);
      int len = hay.length(i);
      if (counts[i] == 0) {
        MemorySegment.copy(h, hStart, out, at, len);
        continue;
      }
      int sLen = search.length(i);
      int rLen = repl.length(i);
      int from = 0;
      int hit;
      while ((hit = find(h, hStart, from, len, search.seg(i), search.at(i), sLen)) >= 0) {
        MemorySegment.copy(h, hStart + from, out, at, hit - from);
        at += hit - from;
        MemorySegment.copy(repl.seg(i), repl.at(i), out, at, rLen);
        at += rLen;
        from = hit + sLen;
      }
      MemorySegment.copy(h, hStart + from, out, at, len - from);
    }
    return SegmentVectorBuffers.utf8(n, validity, offsets, out);
  }

  // ---------------------------------------------------------------- translate

  /**
   * {@code translate(hay, from, to)} with a prebuilt dictionary: code point {@code from[k]} maps to
   * {@code to[k]} ({@code null} = deleted); a code point not in the dictionary is copied.
   */
  public static SegmentVectorBuffers translate(Part hay, int[] from, byte[][] to, int n, MemorySegment validity, Arena arena) {
    int[] lengths = new int[n];
    for (int pass = 0; pass < 2; pass++) {
      MemorySegment offsets = pass == 0 ? null : offsetsOf(lengths, n, arena);
      MemorySegment out = pass == 0 ? null : ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
      for (int i = 0; i < n; i++) {
        if (validity != null && !Bitmap.isSet(validity, i)) continue;
        MemorySegment h = hay.seg(i);
        long hStart = hay.at(i);
        int len = hay.length(i);
        long at = pass == 0 ? 0 : offsets.getAtIndex(VectorBuffers.LE_INT, i);
        long total = 0;
        int k = 0;
        while (k < len) {
          int nb = StringSliceKernels.numBytesForFirstByte(h.get(ValueLayout.JAVA_BYTE, hStart + k));
          if (k + nb > len) nb = len - k;
          int cp = StringLengthKernels.firstCodePoint(h, hStart + k, nb);
          int m = -1;
          for (int d = 0; d < from.length; d++) if (from[d] == cp) { m = d; break; }
          if (m < 0) {
            if (pass == 1) MemorySegment.copy(h, hStart + k, out, at + total, nb);
            total += nb;
          } else if (to[m] != null) {
            if (pass == 1) MemorySegment.copy(MemorySegment.ofArray(to[m]), 0, out, at + total, to[m].length);
            total += to[m].length;
          }
          k += nb;
        }
        if (pass == 0) lengths[i] = (int) total;
      }
      if (pass == 1) return SegmentVectorBuffers.utf8(n, validity, offsets, out);
    }
    throw new IllegalStateException("unreachable");
  }

  // ---------------------------------------------------------------- substring_index

  /** Spark's {@code subStringIndex(delim, count)} as a byte range {@code (offset << 32) | length}. */
  static long substringIndexRange(MemorySegment h, long hStart, int len, MemorySegment d, long dStart, int dLen, int count) {
    if (dLen == 0 || count == 0) return 0L;
    if (count > 0) {
      int idx = -1;
      while (count > 0) {
        idx = find(h, hStart, idx + 1, len, d, dStart, dLen);
        if (idx >= 0) count--; else return len; // whole string
      }
      return idx == 0 ? 0L : (long) idx; // offset 0, length idx
    }
    int idx = len - dLen + 1;
    count = -count;
    while (count > 0) {
      idx = rfind(h, hStart, len, idx - 1, d, dStart, dLen);
      if (idx >= 0) count--; else return len;
    }
    if (idx + dLen == len) return 0L;
    return (((long) (idx + dLen)) << 32) | (len - dLen - idx);
  }

  public static SegmentVectorBuffers substringIndex(Part hay, Part delim, VectorBuffers countCol, int count, int n, MemorySegment validity, Arena arena) {
    long[] ranges = new long[n];
    int[] lengths = new int[n];
    for (int i = 0; i < n; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) continue;
      long r = substringIndexRange(hay.seg(i), hay.at(i), hay.length(i), delim.seg(i), delim.at(i), delim.length(i), arg(countCol, count, i));
      ranges[i] = r;
      lengths[i] = (int) r;
    }
    MemorySegment offsets = offsetsOf(lengths, n, arena);
    MemorySegment out = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
    for (int i = 0; i < n; i++) {
      if (lengths[i] == 0) continue;
      MemorySegment.copy(hay.seg(i), hay.at(i) + (ranges[i] >>> 32), out, offsets.getAtIndex(VectorBuffers.LE_INT, i), lengths[i]);
    }
    return SegmentVectorBuffers.utf8(n, validity, offsets, out);
  }

  // ---------------------------------------------------------------- split_part

  /** The first active row whose live part index is 0, or -1. */
  public static int firstZeroPart(VectorBuffers partCol, int part, MemorySegment validity, MemorySegment active, int n) {
    for (int i = 0; i < n; i++) {
      if (active != null && !Bitmap.isSet(active, i)) continue;
      if (validity != null && !Bitmap.isSet(validity, i)) continue;
      if (partCol != null && partCol.validity() != null && !Bitmap.isSet(partCol.validity(), i)) continue;
      if (arg(partCol, part, i) == 0) return i;
    }
    return -1;
  }

  /** The byte range of part {@code k} (1-based, negative from the end) after splitting on the delimiter; the empty range past the ends. */
  static long splitPartRange(MemorySegment h, long hStart, int len, MemorySegment d, long dStart, int dLen, int k) {
    if (dLen == 0) return k == 1 || k == -1 ? (long) len : 0L; // one part: the whole string
    int parts = 1;
    int at = 0;
    while ((at = find(h, hStart, at, len, d, dStart, dLen)) >= 0) { parts++; at += dLen; }
    int target = k > 0 ? k : parts + k + 1;
    if (target < 1 || target > parts) return 0L;
    int from = 0;
    for (int p = 1; p < target; p++) from = find(h, hStart, from, len, d, dStart, dLen) + dLen;
    int end = find(h, hStart, from, len, d, dStart, dLen);
    if (end < 0) end = len;
    return (((long) from) << 32) | (end - from);
  }

  public static SegmentVectorBuffers splitPart(Part hay, Part delim, VectorBuffers partCol, int part, int n, MemorySegment validity, Arena arena) {
    long[] ranges = new long[n];
    int[] lengths = new int[n];
    for (int i = 0; i < n; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) continue;
      long r = splitPartRange(hay.seg(i), hay.at(i), hay.length(i), delim.seg(i), delim.at(i), delim.length(i), arg(partCol, part, i));
      ranges[i] = r;
      lengths[i] = (int) r;
    }
    MemorySegment offsets = offsetsOf(lengths, n, arena);
    MemorySegment out = ArrowLayout.allocateBytes(arena, offsets.getAtIndex(VectorBuffers.LE_INT, n));
    for (int i = 0; i < n; i++) {
      if (lengths[i] == 0) continue;
      MemorySegment.copy(hay.seg(i), hay.at(i) + (ranges[i] >>> 32), out, offsets.getAtIndex(VectorBuffers.LE_INT, i), lengths[i]);
    }
    return SegmentVectorBuffers.utf8(n, validity, offsets, out);
  }

  // ---------------------------------------------------------------- find_in_set

  /** Spark's {@code findInSet}: the 1-based position of {@code word} among the comma-separated {@code set}, 0 when absent or when the word holds a comma. */
  static int findInSet(MemorySegment w, long wStart, int wLen, MemorySegment s, long sStart, int sLen) {
    for (int i = 0; i < wLen; i++) if (w.get(ValueLayout.JAVA_BYTE, wStart + i) == ',') return 0;
    int n = 1;
    int lastComma = -1;
    for (int i = 0; i < sLen; i++) {
      if (s.get(ValueLayout.JAVA_BYTE, sStart + i) == ',') {
        if (i - (lastComma + 1) == wLen && MemorySegment.mismatch(s, sStart + lastComma + 1, sStart + i, w, wStart, wStart + wLen) == -1) return n;
        lastComma = i;
        n++;
      }
    }
    if (sLen - (lastComma + 1) == wLen && MemorySegment.mismatch(s, sStart + lastComma + 1, sStart + sLen, w, wStart, wStart + wLen) == -1) return n;
    return 0;
  }

  public static SegmentVectorBuffers findInSet(Part word, Part set, int n, MemorySegment validity, Arena arena) {
    MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT32, n);
    for (int i = 0; i < n; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) continue;
      out.setAtIndex(VectorBuffers.LE_INT, i, findInSet(word.seg(i), word.at(i), word.length(i), set.seg(i), set.at(i), set.length(i)));
    }
    return SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, out);
  }
}
