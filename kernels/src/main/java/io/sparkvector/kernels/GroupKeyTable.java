package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
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
  private final MemorySegment[] strSegments; // heap views of strBytes, refreshed on growth
  private final int[] strUsed; // bytes used per UTF8 column
  private final int[][] strOffsets; // per UTF8 column: size + 1 offsets
  private final BitSet[] nulls;

  private int[] hashScratch = new int[0]; // row hashes, or combined indices on the memoised path
  private int[] idxScratch = new int[0];
  private int[] memo = new int[0];

  /**
   * When every key is dictionary encoded and the dictionaries are small, group ids are memoised per
   * combination of dictionary indices for the batch, so most rows never probe the table.
   */
  private static final long MEMO_MAX_COMBINATIONS = 1 << 16;

  /** UTF8 keys up to this length are compared byte by byte rather than with MemorySegment.mismatch. */
  private static final int SHORT_KEY_BYTES = 16;

  /**
   * Plain (non-dictionary) UTF8 keys are dictionary-encoded on the fly, against a per-column
   * dictionary kept across batches, so the memoised path above applies to them too and the hashing
   * path compares dictionary indices rather than bytes. Comet's native scan hands over plain Arrow
   * strings for columns Spark's reader would keep dictionary encoded; without this, TPC-H Q1 over
   * that scan hashed and compared every row's group key. Values of up to this many bytes are keyed
   * in the dictionary by their packed bytes; longer values by a 64-bit hash plus a byte compare.
   */
  private static final int PACKED_KEY_BYTES = 8;

  /**
   * Once a plain-string dictionary would exceed this many distinct values the table stops encoding
   * plain strings and hashes and compares every row instead. The dictionary pays off through the
   * memoised path, whose per-batch memo reset and miss rate grow with the number of combinations:
   * measured on 4096-row batches it wins up to a few hundred distinct values and loses by 2x at
   * 4000, for 8-byte and 24-byte keys alike, so the default sits at the crossover. Read once per
   * table from the {@code sparkvector.agg.plainDictMaxEntries} system property.
   */
  static final int DEFAULT_PLAIN_DICT_MAX_ENTRIES = 512;

  private final int plainDictMaxEntries =
      Integer.getInteger("sparkvector.agg.plainDictMaxEntries", DEFAULT_PLAIN_DICT_MAX_ENTRIES);
  private boolean plainDictOverflowed;

  private PlainStringDict[] shortDicts;
  private VectorBuffers[] encodedKeys;
  private int[] offsetScratch = new int[0];
  private byte[] byteScratch = new byte[0];
  private int[][] indexScratch;

  public GroupKeyTable(VecType[] types) {
    this(types, true);
  }

  /**
   * @param encodePlainStrings whether plain UTF8 keys are dictionary-encoded on the fly (see
   *     {@link #plainDictMaxEntries}). Group-by keys want it; a hash join's build side, whose keys
   *     are mostly distinct, does not, and a table built without it is immutable once
   *     {@link #assign} is done, so {@link #lookup(VectorBuffers[], int, int[], MemorySegment,
   *     int[])} may run concurrently from several threads.
   */
  public GroupKeyTable(VecType[] types, boolean encodePlainStrings) {
    this.types = types.clone();
    this.plainDictOverflowed = !encodePlainStrings;
    this.slots = new int[INITIAL_CAPACITY * 2];
    Arrays.fill(slots, -1);
    this.mask = slots.length - 1;
    int k = types.length;
    intKeys = new int[k][];
    longKeys = new long[k][];
    strBytes = new byte[k][];
    strSegments = new MemorySegment[k];
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
          strSegments[c] = MemorySegment.ofArray(strBytes[c]);
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

  /** Whether plain UTF8 keys are still dictionary-encoded on the fly (false once a column overflowed the cap). */
  boolean encodesPlainStrings() {
    return !plainDictOverflowed;
  }

  /**
   * Assigns a group id to each of the {@code n} rows described by the key columns, inserting new
   * groups as needed. Returns the number of groups after the batch.
   */
  public int assign(VectorBuffers[] keys, int n, int[] outIds) {
    return assign(keys, n, outIds, null);
  }

  /**
   * As {@link #assign(VectorBuffers[], int, int[])} restricted to the rows set in {@code
   * selection} ({@code null} for all rows): unselected rows get id {@code -1} and never create a
   * group.
   */
  public int assign(VectorBuffers[] keys, int n, int[] outIds, MemorySegment selection) {
    VectorBuffers[] encoded = encodeShortStrings(keys, n);
    if (encoded != null) {
      keys = encoded;
    }
    long combinations = dictionaryCombinations(keys);
    if (combinations > 0 && combinations <= MEMO_MAX_COMBINATIONS) {
      return assignMemoised(keys, n, outIds, (int) combinations, selection);
    }
    if (hashScratch.length < n) {
      hashScratch = new int[Math.max(n, hashScratch.length * 2)];
    }
    int[] hashes = hashScratch;
    HashKernels.init(hashes, n);
    for (VectorBuffers key : keys) {
      HashKernels.mixColumn(key, hashes);
    }
    if (selection == null) {
      for (int i = 0; i < n; i++) {
        outIds[i] = lookupOrInsert(keys, i, HashKernels.finish(hashes[i]));
      }
    } else {
      Arrays.fill(outIds, 0, n, -1);
      for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
        long bits = Bitmap.wordAt(selection, w, n);
        while (bits != 0L) {
          int i = (w << 6) + Long.numberOfTrailingZeros(bits);
          bits &= bits - 1;
          outIds[i] = lookupOrInsert(keys, i, HashKernels.finish(hashes[i]));
        }
      }
    }
    return size;
  }

  /**
   * Probes without inserting: {@code outIds[i]} is the id of the group whose key equals row
   * {@code i}, or {@code -1} if none exists (or the row is not selected). This is a hash join's
   * probe side over a table built with {@link #assign}; returns the number of rows that matched.
   */
  public int lookup(VectorBuffers[] keys, int n, int[] outIds, MemorySegment selection) {
    if (hashScratch.length < n) {
      hashScratch = new int[Math.max(n, hashScratch.length * 2)];
    }
    return lookup(keys, n, outIds, selection, hashScratch);
  }

  /**
   * {@link #lookup(VectorBuffers[], int, int[], MemorySegment)} with the caller's row-hash scratch
   * ({@code hashes.length >= n}). On a table constructed without plain-string encoding this reads
   * the table only, so concurrent probes from several threads (a broadcast join's tasks sharing one
   * build table) are safe as long as nobody assigns to it any more.
   */
  public int lookup(VectorBuffers[] keys, int n, int[] outIds, MemorySegment selection, int[] hashes) {
    VectorBuffers[] encoded = encodeShortStrings(keys, n);
    if (encoded != null) {
      keys = encoded;
    }
    HashKernels.init(hashes, n);
    for (VectorBuffers key : keys) {
      HashKernels.mixColumn(key, hashes);
    }
    int matched = 0;
    if (selection == null) {
      for (int i = 0; i < n; i++) {
        int gid = lookupOnly(keys, i, HashKernels.finish(hashes[i]));
        outIds[i] = gid;
        if (gid >= 0) {
          matched++;
        }
      }
    } else {
      Arrays.fill(outIds, 0, n, -1);
      for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
        long bits = Bitmap.wordAt(selection, w, n);
        while (bits != 0L) {
          int i = (w << 6) + Long.numberOfTrailingZeros(bits);
          bits &= bits - 1;
          int gid = lookupOnly(keys, i, HashKernels.finish(hashes[i]));
          outIds[i] = gid;
          if (gid >= 0) {
            matched++;
          }
        }
      }
    }
    return matched;
  }

  private int lookupOnly(VectorBuffers[] keys, int row, int hash) {
    int pos = hash & mask;
    while (true) {
      int gid = slots[pos];
      if (gid < 0) {
        return -1;
      }
      if (groupHashes[gid] == hash && equals(gid, keys, row)) {
        return gid;
      }
      pos = (pos + 1) & mask;
    }
  }

  /** Product of (dictionary size + 1) over the keys, or 0 if any key is not dictionary encoded. */
  private static long dictionaryCombinations(VectorBuffers[] keys) {
    if (keys.length == 0) {
      return 0;
    }
    long combinations = 1;
    for (VectorBuffers k : keys) {
      if (!k.isDictionaryEncoded()) {
        return 0;
      }
      combinations *= k.dictionary().length() + 1L;
      if (combinations > MEMO_MAX_COMBINATIONS) {
        return combinations;
      }
    }
    return combinations;
  }

  /**
   * Returns {@code keys} with every plain UTF8 column replaced by a dictionary-encoded view over a
   * {@link PlainStringDict}, or {@code null} when that does not apply: no plain UTF8 key, a key of
   * another type (the memoised path needs every key dictionary encoded), or a dictionary that has
   * grown past {@link #plainDictMaxEntries} -- from then on every batch takes the hashing path.
   */
  private VectorBuffers[] encodeShortStrings(VectorBuffers[] keys, int n) {
    if (plainDictOverflowed) {
      return null;
    }
    boolean any = false;
    for (VectorBuffers k : keys) {
      if (k.isDictionaryEncoded()) {
        continue;
      }
      if (k.type() != VecType.UTF8) {
        return null;
      }
      any = true;
    }
    if (!any || n == 0) {
      return null;
    }
    int k = keys.length;
    if (shortDicts == null) {
      shortDicts = new PlainStringDict[k];
      encodedKeys = new VectorBuffers[k];
      indexScratch = new int[k][];
    }
    if (offsetScratch.length < n + 1) {
      offsetScratch = new int[Math.max(n + 1, offsetScratch.length * 2)];
    }
    int[] offs = offsetScratch;
    for (int c = 0; c < k; c++) {
      VectorBuffers key = keys[c];
      if (key.isDictionaryEncoded()) {
        encodedKeys[c] = key;
        continue;
      }
      MemorySegment.copy(key.offsets(), VectorBuffers.LE_INT, 0, offs, 0, n + 1);
      int first = offs[0];
      int total = offs[n] - first;
      if (byteScratch.length < total) {
        byteScratch = new byte[Math.max(total, byteScratch.length * 2)];
      }
      byte[] bytes = byteScratch;
      MemorySegment.copy(key.data(), ValueLayout.JAVA_BYTE, first, bytes, 0, total);
      if (shortDicts[c] == null) {
        shortDicts[c] = new PlainStringDict();
      }
      PlainStringDict dict = shortDicts[c];
      if (indexScratch[c] == null || indexScratch[c].length < n) {
        indexScratch[c] = new int[Math.max(n, indexScratch[c] == null ? 0 : indexScratch[c].length * 2)];
      }
      int[] idx = indexScratch[c];
      MemorySegment validity = key.validity();
      for (int i = 0; i < n; i++) {
        int start = offs[i] - first;
        int len = offs[i + 1] - offs[i];
        if (validity != null && !Bitmap.isSet(validity, i)) {
          idx[i] = 0;
          continue;
        }
        idx[i] = dict.indexOf(PlainStringDict.fingerprint(bytes, start, len), len, bytes, start);
        if (dict.size() > plainDictMaxEntries) {
          // Too many distinct values for the memo to pay: hash and compare from now on. The groups
          // already assigned are unaffected -- the dictionary only ever named this batch's rows,
          // never the table's keys.
          plainDictOverflowed = true;
          return null;
        }
      }
      encodedKeys[c] = SegmentVectorBuffers.dictionaryUtf8(n, validity, MemorySegment.ofArray(idx), dict.view());
    }
    return encodedKeys;
  }

  /**
   * Distinct plain strings of any length, stored contiguously so the dictionary can be read as a
   * UTF8 {@link VectorBuffers}. Each entry is keyed by its length and a 64-bit fingerprint: the
   * packed bytes themselves for values of up to {@link #PACKED_KEY_BYTES} bytes (so equality is a
   * long compare), a hash of the bytes for longer ones (confirmed by a byte compare on a hit).
   */
  static final class PlainStringDict {
    private long[] bits = new long[64];
    private int[] lens = new int[64];
    private int[] ids = new int[64];
    private int mask = 63;
    private int size;
    private byte[] data = new byte[256];
    private int used;
    private int[] offsets = new int[33];
    private MemorySegment dataSegment = MemorySegment.ofArray(data);
    private MemorySegment offsetSegment = MemorySegment.ofArray(offsets);

    int size() {
      return size;
    }

    /** Direct index for single-byte values (TPC-H's flag columns), bypassing the probe. */
    private final int[] singleByte = new int[256];

    PlainStringDict() {
      Arrays.fill(ids, -1);
      Arrays.fill(singleByte, -1);
    }

    private static final VarHandle LONG_LE = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /**
     * The 64-bit key of a value: its bytes packed little-endian when they fit in a long, otherwise
     * a multiply-xorshift mix over the bytes taken eight at a time (a collision between two long
     * values is caught by the byte compare).
     */
    static long fingerprint(byte[] src, int from, int len) {
      if (len <= PACKED_KEY_BYTES) {
        long packed = 0L;
        for (int b = len - 1; b >= 0; b--) {
          packed = (packed << 8) | (src[from + b] & 0xFFL);
        }
        return packed;
      }
      long h = 0x9E3779B97F4A7C15L ^ len;
      int end = from + len;
      int p = from;
      for (; p + 8 <= end; p += 8) {
        h = (h ^ (long) LONG_LE.get(src, p)) * 0xBF58476D1CE4E5B9L;
        h ^= h >>> 31;
      }
      if (p < end) {
        long tail = 0L;
        for (int b = end - 1; b >= p; b--) {
          tail = (tail << 8) | (src[b] & 0xFFL);
        }
        h = (h ^ tail) * 0x94D049BB133111EBL;
        h ^= h >>> 29;
      }
      return h;
    }

    int indexOf(long packed, int len, byte[] src, int from) {
      if (len == 1) {
        int b = (int) packed; // 0..255
        int id = singleByte[b];
        if (id < 0) {
          id = probe(packed, len, src, from);
          singleByte[b] = id;
        }
        return id;
      }
      return probe(packed, len, src, from);
    }

    private int probe(long packed, int len, byte[] src, int from) {
      int pos = HashKernels.finish(HashKernels.mix32(HashKernels.fold(packed), len)) & mask;
      while (true) {
        int id = ids[pos];
        if (id < 0) {
          return insert(packed, len, src, from, pos);
        }
        if (bits[id] == packed && lens[id] == len && (len <= PACKED_KEY_BYTES || sameBytes(id, src, from, len))) {
          return id;
        }
        pos = (pos + 1) & mask;
      }
    }

    private boolean sameBytes(int id, byte[] src, int from, int len) {
      int off = offsets[id];
      return Arrays.equals(data, off, off + len, src, from, from + len);
    }

    private int insert(long packed, int len, byte[] src, int from, int pos) {
      int id = size;
      if (id == bits.length) {
        bits = Arrays.copyOf(bits, id * 2);
        lens = Arrays.copyOf(lens, id * 2);
      }
      if (id + 1 >= offsets.length) {
        offsets = Arrays.copyOf(offsets, offsets.length * 2);
        offsetSegment = MemorySegment.ofArray(offsets);
      }
      if (used + len > data.length) {
        data = Arrays.copyOf(data, Math.max(data.length * 2, used + len));
        dataSegment = MemorySegment.ofArray(data);
      }
      bits[id] = packed;
      lens[id] = len;
      System.arraycopy(src, from, data, used, len);
      used += len;
      offsets[id + 1] = used;
      ids[pos] = id;
      size++;
      if (size * 2 > ids.length) {
        rehash();
      }
      return id;
    }

    private void rehash() {
      int[] newIds = new int[ids.length * 2];
      Arrays.fill(newIds, -1);
      int newMask = newIds.length - 1;
      for (int id = 0; id < size; id++) {
        int pos = HashKernels.finish(HashKernels.mix32(HashKernels.fold(bits[id]), lens[id])) & newMask;
        while (newIds[pos] >= 0) {
          pos = (pos + 1) & newMask;
        }
        newIds[pos] = id;
      }
      ids = newIds;
      mask = newMask;
    }

    /** The dictionary as a UTF8 column; valid until the next {@link #indexOf} that inserts. */
    VectorBuffers view() {
      return SegmentVectorBuffers.utf8(size, null, offsetSegment, dataSegment);
    }
  }

  private int assignMemoised(VectorBuffers[] keys, int n, int[] outIds, int combinations, MemorySegment selection) {
    if (memo.length < combinations) {
      memo = new int[Math.max(combinations, memo.length * 2)];
    }
    Arrays.fill(memo, 0, combinations, -1);
    int k = keys.length;
    // Fold the per-column dictionary indices (0 = null, i + 1 otherwise) into one combined index
    // per row, column by column, so the hot loop runs over plain int arrays.
    int[] combined = combinedScratch(n);
    Arrays.fill(combined, 0, n, 0);
    int[] idx = idxScratch(n);
    for (int c = 0; c < k; c++) {
      VectorBuffers key = keys[c];
      int size = key.dictionary().length() + 1;
      MemorySegment.copy(key.data(), VectorBuffers.LE_INT, 0, idx, 0, n);
      MemorySegment validity = key.validity();
      if (validity == null) {
        for (int i = 0; i < n; i++) {
          combined[i] = combined[i] * size + idx[i] + 1;
        }
      } else {
        for (int i = 0; i < n; i++) {
          combined[i] = combined[i] * size + (Bitmap.isSet(validity, i) ? idx[i] + 1 : 0);
        }
      }
    }
    int[] memo = this.memo;
    if (selection == null) {
      for (int i = 0; i < n; i++) {
        int gid = memo[combined[i]];
        if (gid < 0) {
          gid = lookupOrInsert(keys, i, dictionaryRowHash(keys, i));
          memo[combined[i]] = gid;
        }
        outIds[i] = gid;
      }
    } else {
      Arrays.fill(outIds, 0, n, -1);
      for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
        long bits = Bitmap.wordAt(selection, w, n);
        while (bits != 0L) {
          int i = (w << 6) + Long.numberOfTrailingZeros(bits);
          bits &= bits - 1;
          int gid = memo[combined[i]];
          if (gid < 0) {
            gid = lookupOrInsert(keys, i, dictionaryRowHash(keys, i));
            memo[combined[i]] = gid;
          }
          outIds[i] = gid;
        }
      }
    }
    return size;
  }

  private int[] combinedScratch(int n) {
    if (hashScratch.length < n) {
      hashScratch = new int[Math.max(n, hashScratch.length * 2)];
    }
    return hashScratch;
  }

  private int[] idxScratch(int n) {
    if (idxScratch.length < n) {
      idxScratch = new int[Math.max(n, idxScratch.length * 2)];
    }
    return idxScratch;
  }

  /** Same hash {@link HashKernels#mixColumn} produces for the row, computed for one row. */
  private static int dictionaryRowHash(VectorBuffers[] keys, int row) {
    int h = HashKernels.SEED;
    for (VectorBuffers k : keys) {
      int v;
      if (k.isNull(row)) {
        v = HashKernels.NULL_MARK;
      } else {
        VectorBuffers dict = k.dictionary();
        int idx = k.getInt(row);
        if (dict.isNull(idx)) {
          v = HashKernels.NULL_MARK;
        } else {
          int start = dict.offsets().get(VectorBuffers.LE_INT, (long) idx << 2);
          int len = dict.offsets().get(VectorBuffers.LE_INT, (long) (idx + 1) << 2) - start;
          v = HashKernels.hashBytes(dict.data(), start, len);
        }
      }
      h = HashKernels.mix32(h, v);
    }
    return HashKernels.finish(h);
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
    if (len <= SHORT_KEY_BYTES) {
      // Group-by strings are mostly codes of a few bytes. MemorySegment.mismatch costs more in
      // set-up (two liveness checks, a vectorized-mismatch call) than the comparison itself at
      // these lengths; TPC-H Q1 over plain (non-dictionary) Arrow strings spent a third of the
      // aggregate's time there.
      byte[] store = strBytes[c];
      for (int i = 0; i < len; i++) {
        if (store[start + i] != data.get(ValueLayout.JAVA_BYTE, rowStart + i)) {
          return false;
        }
      }
      return true;
    }
    return MemorySegment.mismatch(strSegments[c], start, start + len, data, rowStart, rowStart + len) == -1;
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
        strSegments[c] = MemorySegment.ofArray(strBytes[c]);
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
