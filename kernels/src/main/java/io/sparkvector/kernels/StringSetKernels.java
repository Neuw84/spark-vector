package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Membership of a UTF8 column's values in a fixed set of byte strings -- {@code col IN ('a', 'b', ...)}
 * with many literals (#371). The set is an open-addressing table over the literals, keyed by a hash of
 * their bytes, so a column is one pass whatever the set's size; {@code InExpr}'s one compare pass per
 * literal made a 400-way list 400 passes.
 *
 * <p>Equality is byte equality, as {@link StringCompareKernels#compareScalar} has it. Null rows are
 * the caller's business: the result bitmap shares the column's validity, and whatever bit this kernel
 * sets on a null row is masked by it.
 */
public final class StringSetKernels {
  private StringSetKernels() {}

  /** The set: literals' bytes in one array, a power-of-two table of entry indices (-1 empty). */
  public static final class StringSet {
    private final byte[] bytes;
    private final int[] starts; // entry i occupies bytes[starts[i], starts[i+1])
    private final int[] table;
    private final int mask;
    private final MemorySegment segment;

    public StringSet(byte[][] values) {
      int total = 0;
      for (byte[] v : values) {
        total += v.length;
      }
      bytes = new byte[total];
      starts = new int[values.length + 1];
      int pos = 0;
      for (int i = 0; i < values.length; i++) {
        System.arraycopy(values[i], 0, bytes, pos, values[i].length);
        pos += values[i].length;
        starts[i + 1] = pos;
      }
      segment = MemorySegment.ofArray(bytes);
      int capacity = Integer.highestOneBit(Math.max(4, values.length * 2) - 1) << 1;
      table = new int[capacity];
      Arrays.fill(table, -1);
      mask = capacity - 1;
      for (int i = 0; i < values.length; i++) {
        int h = hash(segment, starts[i], starts[i + 1]) & mask;
        while (table[h] >= 0) {
          if (sameBytes(table[h], segment, starts[i], starts[i + 1])) {
            break; // a duplicate literal: one entry is enough
          }
          h = (h + 1) & mask;
        }
        if (table[h] < 0) {
          table[h] = i;
        }
      }
    }

    public int size() {
      return starts.length - 1;
    }

    /** Whether {@code data[start, end)} is one of the literals. */
    public boolean contains(MemorySegment data, int start, int end) {
      int h = hash(data, start, end) & mask;
      int e;
      while ((e = table[h]) >= 0) {
        if (sameBytes(e, data, start, end)) {
          return true;
        }
        h = (h + 1) & mask;
      }
      return false;
    }

    private boolean sameBytes(int entry, MemorySegment data, int start, int end) {
      int es = starts[entry], ee = starts[entry + 1];
      return ee - es == end - start && MemorySegment.mismatch(segment, es, ee, data, start, end) < 0;
    }

    /** A 32-bit mix of the length and every byte (FNV-1a over the bytes, the length folded in). */
    static int hash(MemorySegment data, int start, int end) {
      int h = 0x811c9dc5 ^ (end - start);
      for (int i = start; i < end; i++) {
        h = (h ^ data.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i)) * 0x01000193;
      }
      return h ^ (h >>> 15);
    }
  }

  /** {@code out[i] = a[i] in set}; one verdict per dictionary entry on a dictionary column. {@code active} may be null. */
  public static void inSet(VectorBuffers a, StringSet set, MemorySegment active, MemorySegment out) {
    if (a.type() != VecType.UTF8) {
      throw new IllegalArgumentException("expected UTF8, got " + a.type());
    }
    int n = a.length();
    VectorBuffers dict = a.dictionary();
    if (dict != null) {
      int m = dict.length();
      boolean[] verdict = new boolean[m];
      MemorySegment doff = dict.offsets();
      MemorySegment ddata = dict.data();
      for (int j = 0; j < m; j++) {
        int start = doff.get(VectorBuffers.LE_INT, (long) j << 2);
        int end = doff.get(VectorBuffers.LE_INT, (long) (j + 1) << 2);
        verdict[j] = set.contains(ddata, start, end);
      }
      StringCompareKernels.gather(a.data(), n, verdict, active, out);
      return;
    }
    MemorySegment off = a.offsets();
    MemorySegment data = a.data();
    for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
      if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
        Bitmap.setWord(out, w, n, 0L);
        continue;
      }
      int base = w << 6, limit = Math.min(64, n - base);
      long word = 0L;
      int start = off.get(VectorBuffers.LE_INT, (long) base << 2);
      for (int k = 0; k < limit; k++) {
        int end = off.get(VectorBuffers.LE_INT, (long) (base + k + 1) << 2);
        if (set.contains(data, start, end)) {
          word |= 1L << k;
        }
        start = end;
      }
      Bitmap.setWord(out, w, n, word);
    }
  }
}
