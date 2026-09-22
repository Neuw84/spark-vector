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
 * <p>UTF8 keys are stored as int ids (#377): each UTF8 key column owns a dictionary of the distinct
 * values seen by this table, a row's string is mapped to its id once per batch (once per entry of
 * the input's dictionary when it has one), and from there the table hashes, compares and stores
 * ints -- five string keys probe like five int keys. The dictionary is what the aggregate emits
 * with the ids, and what the shuffle writer stages, so a string is hashed once in the whole stage.
 */
public final class GroupKeyTable {

  private static final int INITIAL_CAPACITY = 1024;

  private final VecType[] types;
  private int[] slots; // group id or -1
  private int mask;
  private int size;
  private int[] groupHashes = new int[INITIAL_CAPACITY];

  private final int[][] intKeys; // INT32 and BOOL (0/1)
  private final long[][] longKeys; // INT64 and FLOAT64 (raw bits); the low limb of DECIMAL128
  private final long[][] hiKeys; // the high limb of DECIMAL128
  private final int[][] strIds; // UTF8: the id of the group's value in dicts[c]
  private final PlainStringDict[] dicts; // UTF8: the column's distinct values
  private final int strCols; // number of UTF8 columns
  private final int[] strCol; // column -> index among the UTF8 columns, or -1
  private final BitSet[] nulls;

  private int[] hashScratch = new int[0]; // row hashes, or combined indices on the memoised path
  private byte[] emitScratch = new byte[0]; // a column's values gathered from the dictionary before one bulk copy out
  private int[] idxScratch = new int[0];
  private int[] memo = new int[0];

  /**
   * When every key is dictionary encoded and the dictionaries are small, group ids are memoised per
   * combination of dictionary indices for the batch, so most rows never probe the table.
   */
  private static final long MEMO_MAX_COMBINATIONS = 1 << 16;

  /**
   * Values of up to this many bytes are keyed in a column's dictionary by their packed bytes (so
   * equality is a long compare); longer values by a 64-bit hash plus a byte compare.
   */
  private static final int PACKED_KEY_BYTES = 8;

  public GroupKeyTable(VecType[] types) {
    this(types, true);
  }

  /**
   * @param encodePlainStrings kept for the callers' sake; every UTF8 key is stored by id now, and a
   *     table is immutable once {@link #assign} is done (a probe inserts nothing, not even into the
   *     dictionaries), so {@link #lookup(VectorBuffers[], int, int[], MemorySegment, int[])} may run
   *     concurrently from several threads whichever way it was built.
   */
  public GroupKeyTable(VecType[] types, boolean encodePlainStrings) {
    this.types = types.clone();
    this.slots = new int[INITIAL_CAPACITY * 2];
    Arrays.fill(slots, -1);
    this.mask = slots.length - 1;
    int k = types.length;
    intKeys = new int[k][];
    longKeys = new long[k][];
    hiKeys = new long[k][];
    strIds = new int[k][];
    dicts = new PlainStringDict[k];
    strCol = new int[k];
    nulls = new BitSet[k];
    int s = 0;
    for (int c = 0; c < k; c++) {
      nulls[c] = new BitSet();
      strCol[c] = -1;
      switch (types[c]) {
        case INT32, BOOL -> intKeys[c] = new int[INITIAL_CAPACITY];
        case INT64, FLOAT64 -> longKeys[c] = new long[INITIAL_CAPACITY];
        case DECIMAL128 -> {
          longKeys[c] = new long[INITIAL_CAPACITY];
          hiKeys[c] = new long[INITIAL_CAPACITY];
        }
        case UTF8 -> {
          strCol[c] = s++;
          strIds[c] = new int[INITIAL_CAPACITY];
          dicts[c] = new PlainStringDict();
        }
      }
    }
    strCols = s;
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
    return assign(keys, n, outIds, null);
  }

  /**
   * As {@link #assign(VectorBuffers[], int, int[])} restricted to the rows set in {@code
   * selection} ({@code null} for all rows): unselected rows get id {@code -1} and never create a
   * group.
   */
  public int assign(VectorBuffers[] keys, int n, int[] outIds, MemorySegment selection) {
    IdScratch ids = toIds(keys, n, true);
    if (ids != null) {
      keys = ids.keys;
    }
    long combinations = dictionaryCombinations();
    if (combinations > 0 && combinations <= MEMO_MAX_COMBINATIONS) {
      return assignMemoised(keys, ids, n, outIds, (int) combinations, selection);
    }
    if (hashScratch.length < n) {
      hashScratch = new int[Math.max(n, hashScratch.length * 2)];
    }
    int[] hashes = hashScratch;
    HashKernels.init(hashes, n);
    for (VectorBuffers key : keys) {
      HashKernels.mixColumn(key, hashes);
    }
    Bound b = Bound.of(keys, ids);
    if (selection == null) {
      for (int i = 0; i < n; i++) {
        outIds[i] = lookupOrInsert(b, i, HashKernels.finish(hashes[i]));
      }
    } else {
      Arrays.fill(outIds, 0, n, -1);
      for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
        long bits = Bitmap.wordAt(selection, w, n);
        while (bits != 0L) {
          int i = (w << 6) + Long.numberOfTrailingZeros(bits);
          bits &= bits - 1;
          outIds[i] = lookupOrInsert(b, i, HashKernels.finish(hashes[i]));
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
    IdScratch ids = toIds(keys, n, false);
    if (ids != null) {
      keys = ids.keys;
    }
    HashKernels.init(hashes, n);
    for (VectorBuffers key : keys) {
      HashKernels.mixColumn(key, hashes);
    }
    ProbeKeys heap = ProbeKeys.of(keys, n, types);
    Bound b = heap != null ? null : Bound.of(keys, ids);
    int matched = 0;
    if (selection == null) {
      for (int i = 0; i < n; i++) {
        int gid = heap != null ? lookupOnly(heap, i, HashKernels.finish(hashes[i])) : lookupOnly(b, i, HashKernels.finish(hashes[i]));
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
          int gid = heap != null ? lookupOnly(heap, i, HashKernels.finish(hashes[i])) : lookupOnly(b, i, HashKernels.finish(hashes[i]));
          outIds[i] = gid;
          if (gid >= 0) {
            matched++;
          }
        }
      }
    }
    return matched;
  }

  /**
   * The probe batch's key columns as Java arrays, for the probe of a join (#409). A join probes one
   * row at a time -- hash, slot, compare -- and each compare read the row's key through
   * {@link VectorBuffers#getInt} / {@link VectorBuffers#isNull}: a segment liveness check and a
   * bounds check per read, virtual when the receiver profile mixes heap and native segments, which
   * was 11% of an executor's samples in q88 at SF10 (store_sales probing three dimension tables).
   * Plain INT32, INT64 and FLOAT64 keys (a double is its raw bits) are copied out once per batch
   * with one bulk move per column and compared from the arrays; any other key type keeps the
   * segment path. Per thread, since a broadcast table is probed by several tasks at once.
   */
  /**
   * The batch's key columns bound once per call (#377): each column's data, offsets and validity
   * segments -- and its dictionary's -- as fields of the concrete segment type. The per-row path
   * (hash slot, compare, insert, append) read every key through {@link VectorBuffers#isNull},
   * {@link VectorBuffers#getInt} and {@code offsets()}/{@code data()}: interface calls whose receiver
   * profile mixes the adapters' buffers, Arrow-backed buffers and encoded short strings, so they
   * stayed virtual, and each carried a segment liveness and bounds check of its own -- 34% of an
   * executor's samples in q67's rollup at 1 TB were those checks. Per thread, as {@link ProbeKeys}.
   */
  private static final class Bound {
    /** Columns bound by the last {@link #of}; the arrays may be longer. */
    int count;
    VectorBuffers[] keys = new VectorBuffers[0];
    MemorySegment[] data = new MemorySegment[0];
    MemorySegment[] offsets = new MemorySegment[0];
    MemorySegment[] validity = new MemorySegment[0];
    int[][] ids = new int[0][]; // UTF8 columns: the rows' dictionary ids as a heap array

    private static final ThreadLocal<Bound> SCRATCH = ThreadLocal.withInitial(Bound::new);

    static Bound of(VectorBuffers[] keys, IdScratch idScratch) {
      Bound b = SCRATCH.get();
      int n = keys.length;
      b.count = n;
      if (b.data.length < n) {
        b.keys = new VectorBuffers[n];
        b.data = new MemorySegment[n];
        b.offsets = new MemorySegment[n];
        b.validity = new MemorySegment[n];
        b.ids = new int[n][];
      }
      for (int c = 0; c < n; c++) {
        VectorBuffers k = keys[c];
        b.keys[c] = k;
        b.data[c] = k.data();
        b.offsets[c] = k.offsets();
        b.validity[c] = k.validity();
        b.ids[c] = idScratch == null ? null : idScratch.ids[c];
      }
      return b;
    }

    int getId(int c, int row) {
      return ids[c][row];
    }

    boolean isNull(int c, int row) {
      MemorySegment v = validity[c];
      return v != null && !Bitmap.isSet(v, row);
    }

    int getInt(int c, int row) {
      return data[c].get(VectorBuffers.LE_INT, (long) row << 2);
    }

    long getLong(int c, int row) {
      return data[c].get(VectorBuffers.LE_LONG, (long) row << 3);
    }

    double getDouble(int c, int row) {
      return data[c].get(VectorBuffers.LE_DOUBLE, (long) row << 3);
    }

    boolean getBoolean(int c, int row) {
      return Bitmap.isSet(data[c], row);
    }

  }

  private static final class ProbeKeys {
    int[][] ints = new int[0][];
    long[][] longs = new long[0][];
    long[][] validity = new long[0][];

    private static final ThreadLocal<ProbeKeys> SCRATCH = ThreadLocal.withInitial(ProbeKeys::new);

    static ProbeKeys of(VectorBuffers[] keys, int n, VecType[] types) {
      for (int c = 0; c < keys.length; c++) {
        VecType t = types[c];
        if (keys[c].isDictionaryEncoded() || !(t == VecType.INT32 || t == VecType.INT64 || t == VecType.FLOAT64)) {
          return null;
        }
      }
      ProbeKeys p = SCRATCH.get();
      if (p.ints.length < keys.length) {
        p.ints = Arrays.copyOf(p.ints, keys.length);
        p.longs = Arrays.copyOf(p.longs, keys.length);
        p.validity = Arrays.copyOf(p.validity, keys.length);
      }
      for (int c = 0; c < keys.length; c++) {
        VectorBuffers k = keys[c];
        if (types[c] == VecType.INT32) {
          int[] a = p.ints[c];
          if (a == null || a.length < n) {
            a = new int[Math.max(n, 4096)];
            p.ints[c] = a;
          }
          MemorySegment.copy(k.data(), VectorBuffers.LE_INT, 0L, a, 0, n);
        } else {
          long[] a = p.longs[c];
          if (a == null || a.length < n) {
            a = new long[Math.max(n, 4096)];
            p.longs[c] = a;
          }
          MemorySegment.copy(k.data(), VectorBuffers.LE_LONG, 0L, a, 0, n);
        }
        if (k.hasNulls()) {
          int words = Bitmap.wordsFor(n);
          long[] v = p.validity[c];
          if (v == null || v.length < words) {
            v = new long[Math.max(words, 64)];
          }
          for (int w = 0; w < words; w++) {
            v[w] = Bitmap.wordAt(k.validity(), w, n);
          }
          p.validity[c] = v;
        } else {
          p.validity[c] = null;
        }
      }
      return p;
    }

    boolean isNull(int c, int row) {
      long[] v = validity[c];
      return v != null && (v[row >>> 6] & (1L << (row & 63))) == 0L;
    }
  }

  private int lookupOnly(ProbeKeys keys, int row, int hash) {
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

  private boolean equals(int gid, ProbeKeys keys, int row) {
    for (int c = 0; c < types.length; c++) {
      boolean rowNull = keys.isNull(c, row);
      if (rowNull != nulls[c].get(gid)) {
        return false;
      }
      if (rowNull) {
        continue;
      }
      if (types[c] == VecType.INT32) {
        if (intKeys[c][gid] != keys.ints[c][row]) {
          return false;
        }
      } else if (longKeys[c][gid] != keys.longs[c][row]) {
        return false;
      }
    }
    return true;
  }

  private int lookupOnly(Bound keys, int row, int hash) {
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

  /**
   * Product of (dictionary size + 1) over the UTF8 keys after the batch's ids were assigned, or 0
   * if a key is not UTF8 (the memoised path needs every key to have a small dictionary).
   */
  private long dictionaryCombinations() {
    if (types.length == 0 || strCols != types.length) {
      return 0;
    }
    long combinations = 1;
    for (PlainStringDict d : dicts) {
      combinations *= d.size() + 1L;
      if (combinations > MEMO_MAX_COMBINATIONS) {
        return combinations;
      }
    }
    return combinations;
  }

  /**
   * Per-thread scratch of {@link #toIds}: the key columns with every UTF8 column replaced by an
   * INT32 column of dictionary ids (a heap array, also kept as such for {@link Bound}), plus the
   * per-column maps from an input dictionary's entries to ids. Per thread because a broadcast
   * join's tasks probe one table concurrently.
   */
  private static final class IdScratch {
    VectorBuffers[] keys = new VectorBuffers[0];
    int[][] ids = new int[0][];
    int[][] entryIds = new int[0][]; // per column: input dictionary entry -> id
    int[][] entryGen = new int[0][]; // per column: the generation entryIds[c][e] was computed in
    int[] gen = new int[0];
    VectorBuffers[] lastDict = new VectorBuffers[0];
    int[] offs = new int[0];
    byte[] bytes = new byte[0];

    private static final ThreadLocal<IdScratch> SCRATCH = ThreadLocal.withInitial(IdScratch::new);

    static IdScratch get(int k) {
      IdScratch s = SCRATCH.get();
      if (s.keys.length != k) {
        s.keys = new VectorBuffers[k]; // exactly k: callers iterate it as the key columns
      }
      if (s.ids.length < k) {
        s.ids = Arrays.copyOf(s.ids, k);
        s.entryIds = Arrays.copyOf(s.entryIds, k);
        s.entryGen = Arrays.copyOf(s.entryGen, k);
        s.gen = Arrays.copyOf(s.gen, k);
        s.lastDict = Arrays.copyOf(s.lastDict, k);
      }
      return s;
    }
  }

  /**
   * Maps every UTF8 key column of the batch to ids in the column's dictionary: a plain column row
   * by row (one fingerprint and probe per row), a dictionary-encoded column entry by entry (one
   * probe per distinct entry the batch uses, remembered while the same dictionary keeps arriving)
   * and then a gather. With {@code insert} false (a join probe) an unknown value gets id -1, which
   * no group carries. Returns null when the table has no UTF8 key or the batch is empty.
   */
  private IdScratch toIds(VectorBuffers[] keys, int n, boolean insert) {
    if (strCols == 0 || n == 0) {
      return null;
    }
    int k = keys.length;
    IdScratch s = IdScratch.get(k);
    System.arraycopy(keys, 0, s.keys, 0, k);
    if (s.offs.length < n + 1) {
      s.offs = new int[Math.max(n + 1, s.offs.length * 2)];
    }
    for (int c = 0; c < k; c++) {
      if (types[c] != VecType.UTF8) {
        s.ids[c] = null;
        continue;
      }
      VectorBuffers key = keys[c];
      PlainStringDict dict = dicts[c];
      int[] ids = s.ids[c];
      if (ids == null || ids.length < n) {
        ids = new int[Math.max(n, ids == null ? 4096 : ids.length * 2)];
        s.ids[c] = ids;
      }
      MemorySegment validity = key.validity();
      if (key.isDictionaryEncoded()) {
        VectorBuffers d = key.dictionary();
        int m = d.length();
        int[] entryIds = s.entryIds[c];
        int[] entryGen = s.entryGen[c];
        if (entryIds == null || entryIds.length < m) {
          entryIds = new int[Math.max(m, entryIds == null ? 256 : entryIds.length * 2)];
          entryGen = new int[entryIds.length];
          s.entryIds[c] = entryIds;
          s.entryGen[c] = entryGen;
          s.gen[c] = 0;
          s.lastDict[c] = null;
        }
        // A new dictionary object invalidates the entry map by bumping the generation: entries are
        // mapped when first used, so a batch costs its rows plus its distinct entries, never the
        // whole dictionary. (Ids only ever grow, so a map of a dictionary that keeps arriving stays right.)
        if (s.lastDict[c] != d) {
          s.lastDict[c] = d;
          if (++s.gen[c] == 0) {
            Arrays.fill(entryGen, 0);
            s.gen[c] = 1;
          }
        }
        int gen = s.gen[c];
        MemorySegment dOff = d.offsets();
        MemorySegment dData = d.data();
        MemorySegment dValidity = d.validity();
        MemorySegment idx = key.data();
        for (int i = 0; i < n; i++) {
          if (validity != null && !Bitmap.isSet(validity, i)) {
            ids[i] = 0;
            continue;
          }
          int e = idx.get(VectorBuffers.LE_INT, (long) i << 2);
          if (entryGen[e] != gen) {
            int id;
            if (dValidity != null && !Bitmap.isSet(dValidity, e)) {
              id = 0; // a null entry: the row is null through the dictionary; not a key value
            } else {
              int start = dOff.get(VectorBuffers.LE_INT, (long) e << 2);
              int len = dOff.get(VectorBuffers.LE_INT, (long) (e + 1) << 2) - start;
              id = dict.indexOf(dData, start, len, insert, s);
            }
            entryIds[e] = id;
            entryGen[e] = gen;
          }
          ids[i] = entryIds[e];
        }
      } else {
        int[] offs = s.offs;
        MemorySegment.copy(key.offsets(), VectorBuffers.LE_INT, 0, offs, 0, n + 1);
        int first = offs[0];
        int total = offs[n] - first;
        if (s.bytes.length < total) {
          s.bytes = new byte[Math.max(total, s.bytes.length * 2)];
        }
        byte[] bytes = s.bytes;
        MemorySegment.copy(key.data(), ValueLayout.JAVA_BYTE, first, bytes, 0, total);
        for (int i = 0; i < n; i++) {
          if (validity != null && !Bitmap.isSet(validity, i)) {
            ids[i] = 0;
            continue;
          }
          int start = offs[i] - first;
          int len = offs[i + 1] - offs[i];
          ids[i] = dict.indexOf(PlainStringDict.fingerprint(bytes, start, len), len, bytes, start, insert);
        }
      }
      s.keys[c] = SegmentVectorBuffers.fixedWidth(VecType.INT32, n, validity, MemorySegment.ofArray(ids));
    }
    return s;
  }

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

    /** The id of the value, inserting it when absent if {@code insert}; -1 when absent otherwise. */
    int indexOf(long packed, int len, byte[] src, int from, boolean insert) {
      if (len == 1) {
        int b = (int) packed; // 0..255
        int id = singleByte[b];
        if (id < 0) {
          id = probe(packed, len, src, from, insert);
          if (id >= 0) {
            singleByte[b] = id;
          }
        }
        return id;
      }
      return probe(packed, len, src, from, insert);
    }

    /** {@link #indexOf} over a value that lives in a segment (an input dictionary's entry). */
    int indexOf(MemorySegment data, long start, int len, boolean insert, IdScratch s) {
      if (s.bytes.length < len) {
        s.bytes = new byte[Math.max(len, s.bytes.length * 2)];
      }
      MemorySegment.copy(data, ValueLayout.JAVA_BYTE, start, s.bytes, 0, len);
      return indexOf(fingerprint(s.bytes, 0, len), len, s.bytes, 0, insert);
    }

    private int probe(long packed, int len, byte[] src, int from, boolean insert) {
      int pos = HashKernels.finish(HashKernels.mix32(HashKernels.fold(packed), len)) & mask;
      while (true) {
        int id = ids[pos];
        if (id < 0) {
          return insert ? insert(packed, len, src, from, pos) : -1;
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

    int offset(int id) {
      return offsets[id];
    }

    int length(int id) {
      return offsets[id + 1] - offsets[id];
    }

    byte[] bytes() {
      return data;
    }

    /** Bytes of the distinct values held. */
    long valueBytes() {
      return used;
    }

    long memoryBytes() {
      return 8L * bits.length + 4L * lens.length + 4L * ids.length + data.length + 4L * offsets.length + 4L * singleByte.length;
    }
  }

  private int assignMemoised(VectorBuffers[] keys, IdScratch ids, int n, int[] outIds, int combinations, MemorySegment selection) {
    Bound b = Bound.of(keys, ids);
    if (memo.length < combinations) {
      memo = new int[Math.max(combinations, memo.length * 2)];
    }
    Arrays.fill(memo, 0, combinations, -1);
    int k = keys.length;
    // Fold the per-column ids (0 = null, id + 1 otherwise) into one combined index per row, column
    // by column, so the hot loop runs over plain int arrays.
    int[] combined = combinedScratch(n);
    Arrays.fill(combined, 0, n, 0);
    for (int c = 0; c < k; c++) {
      int size = dicts[c].size() + 1;
      int[] idx = ids.ids[c];
      MemorySegment validity = keys[c].validity();
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
          gid = lookupOrInsert(b, i, idRowHash(b, i));
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
            gid = lookupOrInsert(b, i, idRowHash(b, i));
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

  /** Same hash {@link HashKernels#mixColumn} produces for a row of INT32 id columns, computed for one row. */
  private static int idRowHash(Bound keys, int row) {
    int h = HashKernels.SEED;
    for (int c = 0; c < keys.count; c++) {
      h = HashKernels.mix32(h, keys.isNull(c, row) ? HashKernels.NULL_MARK : keys.getId(c, row));
    }
    return HashKernels.finish(h);
  }

  private int lookupOrInsert(Bound keys, int row, int hash) {
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

  private boolean equals(int gid, Bound keys, int row) {
    for (int c = 0; c < types.length; c++) {
      boolean rowNull = keys.isNull(c, row);
      if (rowNull != nulls[c].get(gid)) {
        return false;
      }
      if (rowNull) {
        continue;
      }
      switch (types[c]) {
        case INT32 -> {
          if (intKeys[c][gid] != keys.getInt(c, row)) {
            return false;
          }
        }
        case BOOL -> {
          if (intKeys[c][gid] != (keys.getBoolean(c, row) ? 1 : 0)) {
            return false;
          }
        }
        case INT64 -> {
          if (longKeys[c][gid] != keys.getLong(c, row)) {
            return false;
          }
        }
        case FLOAT64 -> {
          if (longKeys[c][gid] != Double.doubleToRawLongBits(keys.getDouble(c, row))) {
            return false;
          }
        }
        case DECIMAL128 -> {
          MemorySegment d = keys.data[c];
          if (longKeys[c][gid] != Decimal128.lo(d, row) || hiKeys[c][gid] != Decimal128.hi(d, row)) {
            return false;
          }
        }
        case UTF8 -> {
          if (strIds[c][gid] != keys.getId(c, row)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  private int insert(Bound keys, int row, int hash, int pos) {
    int gid = size;
    ensureGroupCapacity(gid + 1);
    groupHashes[gid] = hash;
    for (int c = 0; c < types.length; c++) {
      boolean isNull = keys.isNull(c, row);
      nulls[c].set(gid, isNull);
      switch (types[c]) {
        case INT32 -> intKeys[c][gid] = isNull ? 0 : keys.getInt(c, row);
        case BOOL -> intKeys[c][gid] = isNull ? 0 : (keys.getBoolean(c, row) ? 1 : 0);
        case INT64 -> longKeys[c][gid] = isNull ? 0L : keys.getLong(c, row);
        case FLOAT64 -> longKeys[c][gid] = isNull ? 0L : Double.doubleToRawLongBits(keys.getDouble(c, row));
        case DECIMAL128 -> {
          longKeys[c][gid] = isNull ? 0L : Decimal128.lo(keys.data[c], row);
          hiKeys[c][gid] = isNull ? 0L : Decimal128.hi(keys.data[c], row);
        }
        case UTF8 -> strIds[c][gid] = isNull ? 0 : keys.getId(c, row);
      }
    }
    slots[pos] = gid;
    size++;
    if (size * 10L > (long) slots.length * 7L) {
      rehash();
    }
    return gid;
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
        case DECIMAL128 -> {
          longKeys[c] = Arrays.copyOf(longKeys[c], cap);
          hiKeys[c] = Arrays.copyOf(hiKeys[c], cap);
        }
        case UTF8 -> strIds[c] = Arrays.copyOf(strIds[c], cap);
      }
    }
  }

  /**
   * The heap the table holds right now, as allocated (#367): the slots and hashes at capacity, the
   * key arrays at capacity, the string dictionaries as allocated. The next growth step doubles the array it
   * touches and holds both copies for its duration -- the caller adds that headroom.
   */
  public long memoryBytes() {
    long bytes = 4L * slots.length + 4L * groupHashes.length;
    for (int c = 0; c < types.length; c++) {
      switch (types[c]) {
        case INT32, BOOL -> bytes += 4L * intKeys[c].length;
        case INT64, FLOAT64 -> bytes += 8L * longKeys[c].length;
        case DECIMAL128 -> bytes += 16L * longKeys[c].length;
        case UTF8 -> bytes += 4L * strIds[c].length + dicts[c].memoryBytes();
      }
    }
    if (strCols > 0) {
      bytes += emitScratch.length;
    }
    return bytes;
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

  /** The 128-bit key of group {@code gid} of a DECIMAL128 column. */
  public java.math.BigInteger getDecimal128(int c, int gid) {
    return Decimal128.toBigInteger(hiKeys[c][gid], longKeys[c][gid]);
  }

  public String getString(int c, int gid) {
    PlainStringDict d = dicts[c];
    int id = strIds[c][gid];
    return new String(d.bytes(), d.offset(id), d.length(id), java.nio.charset.StandardCharsets.UTF_8);
  }

  /** The dictionary id of group {@code gid}'s value in UTF8 column {@code c} (0 for a null). */
  public int getStringId(int c, int gid) {
    return strIds[c][gid];
  }

  /** Number of distinct values UTF8 column {@code c} has seen. */
  public int dictionarySize(int c) {
    return dicts[c].size();
  }

  /** Bytes of the distinct values of UTF8 column {@code c}. */
  public long dictionaryBytes(int c) {
    return dicts[c].valueBytes();
  }

  /**
   * The distinct values of UTF8 column {@code c} as a plain UTF8 column, indexed by id (#377): the
   * dictionary the ids {@link #writeKeyIds} emits refer to. Valid until the next {@link #assign}.
   */
  public VectorBuffers dictionary(int c) {
    return dicts[c].view();
  }

  /**
   * Writes the dictionary ids of groups {@code [from, to)} of UTF8 column {@code c} as an INT32
   * Arrow-layout column (validity bits for every row, id 0 under a null).
   */
  public void writeKeyIds(int c, int from, int to, MemorySegment validity, MemorySegment ids) {
    int count = to - from;
    for (int o = 0; o < count; o++) {
      Bitmap.setTo(validity, o, !nulls[c].get(from + o));
    }
    MemorySegment.copy(strIds[c], from, ids, VectorBuffers.LE_INT, 0, count);
  }

  /** Total UTF-8 bytes of the keys in {@code [from, to)} of column {@code c}. */
  public long utf8Bytes(int c, int from, int to) {
    PlainStringDict d = dicts[c];
    int[] ids = strIds[c];
    BitSet nul = nulls[c];
    long total = 0;
    for (int gid = from; gid < to; gid++) {
      if (!nul.get(gid)) {
        total += d.length(ids[gid]);
      }
    }
    return total;
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
      case DECIMAL128 -> {
        for (int o = 0; o < count; o++) {
          Decimal128.set(data, o, hiKeys[c][from + o], longKeys[c][from + o]);
        }
      }
      case UTF8 -> {
        // Gather the values from the dictionary into a heap buffer (System.arraycopy, no per-value
        // segment checks) and copy them out once: one small MemorySegment.copy per value was 15% of
        // an executor's time in q67 at 1 TB, whose finest level emits nearly every row.
        PlainStringDict d = dicts[c];
        int[] ids = strIds[c];
        BitSet nul = nulls[c];
        int total = (int) utf8Bytes(c, from, to);
        if (emitScratch.length < total) {
          emitScratch = new byte[Math.max(total, emitScratch.length * 2)];
        }
        byte[] scratch = emitScratch;
        byte[] store = d.bytes();
        int out = 0;
        offsets.set(VectorBuffers.LE_INT, 0L, 0);
        for (int o = 0; o < count; o++) {
          int gid = from + o;
          if (!nul.get(gid)) {
            int id = ids[gid];
            int len = d.length(id);
            System.arraycopy(store, d.offset(id), scratch, out, len);
            out += len;
          }
          offsets.set(VectorBuffers.LE_INT, (long) (o + 1) << 2, out);
        }
        MemorySegment.copy(scratch, 0, data, ValueLayout.JAVA_BYTE, 0, out);
      }
    }
  }
}
