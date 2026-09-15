package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.BitSet;

/**
 * Open-addressing hash table from grouping-key tuples to dense group ids, accumulated across the
 * batches of one task. Keys are stored column-wise (ints, longs/double bits, UTF-8 bytes) so they
 * can be written back out as columns when the partial aggregate is emitted.
 *
 * <p>Dictionary-encoded string keys are compared and stored by their bytes, so batches with
 * different dictionaries (or none) group consistently.
 */
public final class GroupKeyTable {

  private static final int INITIAL_CAPACITY = 1024;

  private final VecType[] types;
  private int[] slots; // group id or -1
  private int mask;
  private int size;
  private int[] groupHashes = new int[INITIAL_CAPACITY];

  private final int[][] intKeys; // INT32 and BOOL (0/1)
  private final long[][] longKeys; // INT64 and FLOAT64 (raw bits)
  private final byte[][] strBytes; // UTF8 byte store per column
  private final int[] strUsed; // bytes used per UTF8 column
  private final int[][] strOffsets; // per UTF8 column: size + 1 offsets
  private final BitSet[] nulls;

  private int[] hashScratch = new int[0];

  public GroupKeyTable(VecType[] types) {
    this.types = types.clone();
    this.slots = new int[INITIAL_CAPACITY * 2];
    Arrays.fill(slots, -1);
    this.mask = slots.length - 1;
    int k = types.length;
    intKeys = new int[k][];
    longKeys = new long[k][];
    strBytes = new byte[k][];
    strUsed = new int[k];
    strOffsets = new int[k][];
    nulls = new BitSet[k];
    for (int c = 0; c < k; c++) {
      nulls[c] = new BitSet();
      switch (types[c]) {
        case INT32, BOOL -> intKeys[c] = new int[INITIAL_CAPACITY];
        case INT64, FLOAT64 -> longKeys[c] = new long[INITIAL_CAPACITY];
        case UTF8 -> {
          strBytes[c] = new byte[INITIAL_CAPACITY * 8];
          strOffsets[c] = new int[INITIAL_CAPACITY + 1];
        }
      }
    }
  }

  public int size() {
    return size;
  }

  public int numKeys() {
    return types.length;
  }

  public VecType type(int c) {
    return types[c];
  }

  /**
   * Assigns a group id to each of the {@code n} rows described by the key columns, inserting new
   * groups as needed. Returns the number of groups after the batch.
   */
  public int assign(VectorBuffers[] keys, int n, int[] outIds) {
    if (hashScratch.length < n) {
      hashScratch = new int[Math.max(n, hashScratch.length * 2)];
    }
    int[] hashes = hashScratch;
    HashKernels.init(hashes, n);
    for (VectorBuffers key : keys) {
      HashKernels.mixColumn(key, hashes);
    }
    for (int i = 0; i < n; i++) {
      outIds[i] = lookupOrInsert(keys, i, HashKernels.finish(hashes[i]));
    }
    return size;
  }

  private int lookupOrInsert(VectorBuffers[] keys, int row, int hash) {
    int pos = hash & mask;
    while (true) {
      int gid = slots[pos];
      if (gid < 0) {
        return insert(keys, row, hash, pos);
      }
      if (groupHashes[gid] == hash && equals(gid, keys, row)) {
        return gid;
      }
      pos = (pos + 1) & mask;
    }
  }

  private boolean equals(int gid, VectorBuffers[] keys, int row) {
    for (int c = 0; c < types.length; c++) {
      VectorBuffers k = keys[c];
      boolean rowNull = k.isNull(row);
      if (rowNull != nulls[c].get(gid)) {
        return false;
      }
      if (rowNull) {
        continue;
      }
      switch (types[c]) {
        case INT32 -> {
          if (intKeys[c][gid] != k.getInt(row)) {
            return false;
          }
        }
        case BOOL -> {
          if (intKeys[c][gid] != (k.getBoolean(row) ? 1 : 0)) {
            return false;
          }
        }
        case INT64 -> {
          if (longKeys[c][gid] != k.getLong(row)) {
            return false;
          }
        }
        case FLOAT64 -> {
          if (longKeys[c][gid] != Double.doubleToRawLongBits(k.getDouble(row))) {
            return false;
          }
        }
        case UTF8 -> {
          if (!utf8Equals(c, gid, k, row)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  private boolean utf8Equals(int c, int gid, VectorBuffers k, int row) {
    int start = strOffsets[c][gid];
    int len = strOffsets[c][gid + 1] - start;
    MemorySegment data;
    long rowStart;
    int rowLen;
    if (k.isDictionaryEncoded()) {
      VectorBuffers dict = k.dictionary();
      int idx = k.getInt(row);
      rowStart = dict.offsets().get(VectorBuffers.LE_INT, (long) idx << 2);
      rowLen = dict.offsets().get(VectorBuffers.LE_INT, (long) (idx + 1) << 2) - (int) rowStart;
      data = dict.data();
    } else {
      rowStart = k.offsets().get(VectorBuffers.LE_INT, (long) row << 2);
      rowLen = k.offsets().get(VectorBuffers.LE_INT, (long) (row + 1) << 2) - (int) rowStart;
      data = k.data();
    }
    if (rowLen != len) {
      return false;
    }
    if (len == 0) {
      return true;
    }
    MemorySegment stored = MemorySegment.ofArray(strBytes[c]);
    return MemorySegment.mismatch(stored, start, start + len, data, rowStart, rowStart + len) == -1;
  }

  private int insert(VectorBuffers[] keys, int row, int hash, int pos) {
    int gid = size;
    ensureGroupCapacity(gid + 1);
    groupHashes[gid] = hash;
    for (int c = 0; c < types.length; c++) {
      VectorBuffers k = keys[c];
      boolean isNull = k.isNull(row);
      nulls[c].set(gid, isNull);
      switch (types[c]) {
        case INT32 -> intKeys[c][gid] = isNull ? 0 : k.getInt(row);
        case BOOL -> intKeys[c][gid] = isNull ? 0 : (k.getBoolean(row) ? 1 : 0);
        case INT64 -> longKeys[c][gid] = isNull ? 0L : k.getLong(row);
        case FLOAT64 -> longKeys[c][gid] = isNull ? 0L : Double.doubleToRawLongBits(k.getDouble(row));
        case UTF8 -> appendUtf8(c, gid, isNull ? null : k, row);
      }
    }
    slots[pos] = gid;
    size++;
    if (size * 10L > (long) slots.length * 7L) {
      rehash();
    }
    return gid;
  }

  private void appendUtf8(int c, int gid, VectorBuffers k, int row) {
    int used = strUsed[c];
    strOffsets[c][gid] = used;
    if (k != null) {
      MemorySegment data;
      long start;
      int len;
      if (k.isDictionaryEncoded()) {
        VectorBuffers dict = k.dictionary();
        int idx = k.getInt(row);
        start = dict.offsets().get(VectorBuffers.LE_INT, (long) idx << 2);
        len = dict.offsets().get(VectorBuffers.LE_INT, (long) (idx + 1) << 2) - (int) start;
        data = dict.data();
      } else {
        start = k.offsets().get(VectorBuffers.LE_INT, (long) row << 2);
        len = k.offsets().get(VectorBuffers.LE_INT, (long) (row + 1) << 2) - (int) start;
        data = k.data();
      }
      if (used + len > strBytes[c].length) {
        strBytes[c] = Arrays.copyOf(strBytes[c], Math.max(strBytes[c].length * 2, used + len));
      }
      MemorySegment.copy(data, ValueLayout.JAVA_BYTE, start, strBytes[c], used, len);
      used += len;
    }
    strUsed[c] = used;
    strOffsets[c][gid + 1] = used;
  }

  private void ensureGroupCapacity(int needed) {
    if (needed <= groupHashes.length) {
      return;
    }
    int cap = Math.max(needed, groupHashes.length * 2);
    groupHashes = Arrays.copyOf(groupHashes, cap);
    for (int c = 0; c < types.length; c++) {
      switch (types[c]) {
        case INT32, BOOL -> intKeys[c] = Arrays.copyOf(intKeys[c], cap);
        case INT64, FLOAT64 -> longKeys[c] = Arrays.copyOf(longKeys[c], cap);
        case UTF8 -> strOffsets[c] = Arrays.copyOf(strOffsets[c], cap + 1);
      }
    }
  }

  private void rehash() {
    int[] newSlots = new int[slots.length * 2];
    Arrays.fill(newSlots, -1);
    int newMask = newSlots.length - 1;
    for (int gid = 0; gid < size; gid++) {
      int pos = groupHashes[gid] & newMask;
      while (newSlots[pos] >= 0) {
        pos = (pos + 1) & newMask;
      }
      newSlots[pos] = gid;
    }
    slots = newSlots;
    mask = newMask;
  }

  // ------------------------------------------------------------------ reading keys back

  public boolean isNull(int c, int gid) {
    return nulls[c].get(gid);
  }

  public int getInt(int c, int gid) {
    return intKeys[c][gid];
  }

  public boolean getBoolean(int c, int gid) {
    return intKeys[c][gid] != 0;
  }

  public long getLong(int c, int gid) {
    return longKeys[c][gid];
  }

  public double getDouble(int c, int gid) {
    return Double.longBitsToDouble(longKeys[c][gid]);
  }

  public String getString(int c, int gid) {
    int start = strOffsets[c][gid];
    return new String(strBytes[c], start, strOffsets[c][gid + 1] - start, java.nio.charset.StandardCharsets.UTF_8);
  }

  /** Total UTF-8 bytes of the keys in {@code [from, to)} of column {@code c}. */
  public long utf8Bytes(int c, int from, int to) {
    return (long) strOffsets[c][to] - strOffsets[c][from];
  }

  /**
   * Writes the keys of groups {@code [from, to)} of column {@code c} into Arrow-layout output
   * buffers ({@code offsets} only for UTF8). Validity bits are written for every row.
   */
  public void writeKeys(int c, int from, int to, MemorySegment validity, MemorySegment data, MemorySegment offsets) {
    int count = to - from;
    for (int o = 0; o < count; o++) {
      Bitmap.setTo(validity, o, !nulls[c].get(from + o));
    }
    switch (types[c]) {
      case INT32 -> MemorySegment.copy(intKeys[c], from, data, VectorBuffers.LE_INT, 0, count);
      case BOOL -> {
        for (int o = 0; o < count; o++) {
          Bitmap.setTo(data, o, intKeys[c][from + o] != 0);
        }
      }
      case INT64, FLOAT64 -> MemorySegment.copy(longKeys[c], from, data, VectorBuffers.LE_LONG, 0, count);
      case UTF8 -> {
        int base = strOffsets[c][from];
        for (int o = 0; o <= count; o++) {
          offsets.set(VectorBuffers.LE_INT, (long) o << 2, strOffsets[c][from + o] - base);
        }
        MemorySegment.copy(strBytes[c], base, data, ValueLayout.JAVA_BYTE, 0, strOffsets[c][to] - base);
      }
    }
  }
}
