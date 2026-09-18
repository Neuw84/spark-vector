package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * The k-way merge of sorted runs (#285). A run is a set of columns and the permutation {@link
 * SortKernels#sortIndices} computed over them; the merge walks one cursor per run through its
 * permutation and emits, per output row, the run it came from and the row within that run, in the
 * total order of the keys -- the same order the sort kernel produces, ties broken by run index
 * then position, so the merge is stable across runs the way the sort is within one.
 *
 * <p>The merge is a loser tree over the runs' cursors: one compare per level per output row,
 * scalar. The key comparison reads the runs' key columns row by row ({@link #compareKeys}) with
 * Spark's rules -- doubles in Spark's total order, strings as unsigned bytes, nulls first or last
 * per key. The output gathers ({@link #gatherFixed}, {@link #gatherUtf8}) then copy each output
 * row from its run's column; they are the multi-source form of {@link GatherKernels}.
 */
public final class RunMerge {

  private final VectorBuffers[][] keys; // [run][key]
  private final int[][] perm;
  private final int[] rows;
  private final int[] pos;
  private final boolean[] descending;
  private final boolean[] nullsFirst;
  /**
   * For a fixed-width key: the run's keys in sorted order as order-preserving unsigned longs
   * (direction folded in), {@code norm[run][key][position]}, and the null flags in the same order
   * when the column has nulls -- so a compare is two array reads. {@code null} for a string or
   * decimal key, which compares through the columns.
   */
  private final long[][][] norm;
  private final boolean[][][] nullAt;

  /**
   * @param keys the key columns of every run, {@code keys[run][key]}
   * @param perm every run's permutation (its rows in sorted order)
   * @param rows the number of rows of every run
   */
  public RunMerge(VectorBuffers[][] keys, int[][] perm, int[] rows, boolean[] ascending, boolean[] nullsFirst) {
    this.keys = keys;
    this.perm = perm;
    this.rows = rows;
    this.pos = new int[rows.length];
    this.descending = new boolean[ascending.length];
    for (int k = 0; k < ascending.length; k++) {
      descending[k] = !ascending[k];
    }
    this.nullsFirst = nullsFirst;
    this.norm = new long[rows.length][ascending.length][];
    this.nullAt = new boolean[rows.length][ascending.length][];
    for (int r = 0; r < rows.length; r++) {
      for (int k = 0; k < ascending.length; k++) {
        if (rows[r] > 0) {
          normalise(r, k);
        }
      }
    }
    this.tree = new int[Math.max(rows.length, 1)];
    this.seen = new boolean[tree.length];
    java.util.Arrays.fill(tree, -1);
    // Leaves in order: a winner parks at the first empty node until its sibling subtree arrives,
    // so after the last leaf every internal node holds a loser and the root the winner.
    for (int r = 0; r < rows.length; r++) {
      int w = rows[r] > 0 ? r : -1;
      boolean parked = false;
      for (int i = (rows.length + r) >>> 1; i > 0; i >>>= 1) {
        if (tree[i] == -1 && !seen[i]) {
          tree[i] = w;
          seen[i] = true;
          parked = true;
          break;
        }
        int other = tree[i];
        if (lessTree(other, w)) {
          tree[i] = w;
          w = other;
        }
      }
      if (!parked) {
        tree[0] = w;
      }
    }
    boolean single = ascending.length == 1;
    boolean fixed = single;
    for (int r = 0; r < rows.length && fixed; r++) {
      fixed = rows[r] == 0 || norm[r][0] != null;
    }
    this.wideKey = fixed && rows.length > 1;
  }

  /** Internal nodes that already hold a parked winner or a loser during construction. */
  private final boolean[] seen;

  /** Builds the sorted-order key array of run {@code r}'s key {@code k} when the type allows it. */
  private void normalise(int r, int k) {
    VectorBuffers col = keys[r][k];
    int n = rows[r];
    int[] p = perm[r];
    long[] out;
    switch (col.type()) {
      case INT32 -> {
        out = new long[n];
        for (int i = 0; i < n; i++) {
          out[i] = (col.getInt(p[i]) ^ Integer.MIN_VALUE) & 0xFFFFFFFFL;
        }
      }
      case INT64 -> {
        out = new long[n];
        for (int i = 0; i < n; i++) {
          out[i] = col.getLong(p[i]) ^ Long.MIN_VALUE;
        }
      }
      case FLOAT64 -> {
        out = new long[n];
        for (int i = 0; i < n; i++) {
          out[i] = SortKernels.doubleKey(col.getDouble(p[i]));
        }
      }
      case BOOL -> {
        out = new long[n];
        for (int i = 0; i < n; i++) {
          out[i] = col.getBoolean(p[i]) ? 1L : 0L;
        }
      }
      default -> {
        return;
      }
    }
    if (descending[k]) {
      for (int i = 0; i < n; i++) {
        out[i] = ~out[i];
      }
    }
    norm[r][k] = out;
    if (col.hasNulls()) {
      boolean[] nulls = new boolean[n];
      for (int i = 0; i < n; i++) {
        nulls[i] = col.isNull(p[i]);
      }
      nullAt[r][k] = nulls;
    }
  }

  /** Rows left to emit. */
  public boolean hasNext() {
    return tree[0] >= 0;
  }

  /**
   * Emits up to {@code max} rows in order: {@code runOf[o]} is the run and {@code rowOf[o]} the row
   * within that run (the permutation already applied). Returns the number emitted.
   */
  public int next(int[] runOf, int[] rowOf, int max) {
    int o = 0;
    while (o < max && tree[0] >= 0) {
      int w = tree[0];
      int[] p = perm[w];
      int at = pos[w];
      int block = 1;
      // The widened leaf: rows of the winner still below the runner-up's key leave in one block,
      // without a replay each. Tried when the last block was wide or every 64th row; a block
      // needs a single fixed-width key and a non-null runner-up (#285).
      if (wideKey && (lastBlock > 1 || (emitted & 63) == 0)) {
        block = blockLength(w, max - o);
      }
      lastBlock = block;
      emitted += block;
      for (int j = 0; j < block; j++) {
        runOf[o] = w;
        rowOf[o] = p[at + j];
        o++;
      }
      pos[w] = at + block;
      replay(w, pos[w] < rows[w] ? w : -1);
    }
    return o;
  }

  private final int[] tree; // losers per internal node, tree[0] the winner; -1 an exhausted run
  private final boolean wideKey;
  private int lastBlock = 1;
  private long emitted;

  /**
   * Rows of run {@code w}, from its cursor, that precede the runner-up's current row: those with a
   * key below the runner-up's, plus the equal ones when {@code w} is the lower run (the tie rule).
   * At least one, at most {@code limit}.
   */
  private int blockLength(int w, int limit) {
    int k = rows.length;
    int best = -1;
    for (int i = (k + w) >>> 1; i > 0; i >>>= 1) {
      int c = tree[i];
      if (c >= 0 && (best < 0 || less(c, best))) {
        best = c;
      }
    }
    int end = Math.min(rows[w], pos[w] + limit);
    if (best < 0) {
      return end - pos[w]; // the only run left
    }
    boolean[] zb = nullAt[best][0];
    if (zb != null && zb[pos[best]]) {
      return 1;
    }
    if (nullAt[w][0] != null) {
      return 1; // null rows in the winner would break the search's monotonicity; one row at a time
    }
    long key2 = norm[best][0][pos[best]];
    long[] kw = norm[w][0];
    boolean equalToo = w < best;
    int lo = pos[w] + 1;
    int hi = end;
    while (lo < hi) {
      int mid = (lo + hi) >>> 1;
      int c = Long.compareUnsigned(kw[mid], key2);
      if (c < 0 || (c == 0 && equalToo)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo - pos[w];
  }

  /** Replays leaf {@code leaf} (its run now {@code run}, or -1 when exhausted) up to the root. */
  private void replay(int leaf, int run) {
    int k = rows.length;
    int w = run;
    for (int i = (k + leaf) >>> 1; i > 0; i >>>= 1) {
      int other = tree[i];
      if (lessTree(other, w)) {
        tree[i] = w;
        w = other;
      }
    }
    tree[0] = w;
  }

  /** {@link #less} with -1 as an exhausted run that never wins. */
  private boolean lessTree(int a, int b) {
    if (a < 0) {
      return false;
    }
    if (b < 0) {
      return true;
    }
    return less(a, b);
  }

  /** Whether run {@code a}'s current row sorts before run {@code b}'s (ties: the lower run first). */
  private boolean less(int a, int b) {
    int pa = pos[a];
    int pb = pos[b];
    long[][] na = norm[a];
    long[][] nb = norm[b];
    for (int k = 0; k < na.length; k++) {
      int c;
      if (na[k] != null) {
        boolean[] za = nullAt[a][k];
        boolean[] zb = nullAt[b][k];
        boolean nullA = za != null && za[pa];
        boolean nullB = zb != null && zb[pb];
        if (nullA || nullB) {
          c = nullA && nullB ? 0 : (nullA == nullsFirst[k] ? -1 : 1);
        } else {
          c = Long.compareUnsigned(na[k][pa], nb[k][pb]);
        }
      } else {
        c = compareKeys(keys[a][k], perm[a][pa], keys[b][k], perm[b][pb], descending[k], nullsFirst[k]);
      }
      if (c != 0) {
        return c < 0;
      }
    }
    return a < b;
  }

  /**
   * Compares row {@code i} of {@code a} with row {@code j} of {@code b} (two columns of the same
   * type) in the given direction, nulls first or last regardless of it, as Spark orders them.
   */
  public static int compareKeys(VectorBuffers a, int i, VectorBuffers b, int j, boolean descending, boolean nullsFirst) {
    boolean na = a.isNull(i);
    boolean nb = b.isNull(j);
    if (na || nb) {
      if (na && nb) {
        return 0;
      }
      return (na == nullsFirst) ? -1 : 1;
    }
    int c =
        switch (a.type()) {
          case INT32 -> Integer.compare(a.getInt(i), b.getInt(j));
          case INT64 -> Long.compare(a.getLong(i), b.getLong(j));
          case FLOAT64 -> Long.compareUnsigned(SortKernels.doubleKey(a.getDouble(i)), SortKernels.doubleKey(b.getDouble(j)));
          case BOOL -> Boolean.compare(a.getBoolean(i), b.getBoolean(j));
          case UTF8 -> compareUtf8(a, i, b, j);
          case DECIMAL128 -> Decimal128.compare(
              Decimal128.hi(a.data(), i), Decimal128.lo(a.data(), i), Decimal128.hi(b.data(), j), Decimal128.lo(b.data(), j));
          default -> throw new IllegalArgumentException("unsupported sort key type " + a.type());
        };
    return descending ? -c : c;
  }

  /** Unsigned lexicographic comparison of two non-null strings from two columns (dictionaries resolved). */
  static int compareUtf8(VectorBuffers a, int i, VectorBuffers b, int j) {
    if (a.isDictionaryEncoded()) {
      return compareUtf8(a.dictionary(), a.getInt(i), b, j);
    }
    if (b.isDictionaryEncoded()) {
      return compareUtf8(a, i, b.dictionary(), b.getInt(j));
    }
    MemorySegment offA = a.offsets();
    MemorySegment offB = b.offsets();
    int sa = offA.get(VectorBuffers.LE_INT, (long) i << 2);
    int ea = offA.get(VectorBuffers.LE_INT, (long) (i + 1) << 2);
    int sb = offB.get(VectorBuffers.LE_INT, (long) j << 2);
    int eb = offB.get(VectorBuffers.LE_INT, (long) (j + 1) << 2);
    int la = ea - sa;
    int lb = eb - sb;
    MemorySegment da = a.data();
    MemorySegment db = b.data();
    long mismatch = MemorySegment.mismatch(da, sa, ea, db, sb, eb);
    if (mismatch == -1) {
      return 0;
    }
    if (mismatch >= Math.min(la, lb)) {
      return Integer.compare(la, lb);
    }
    return Integer.compare(
        da.get(ValueLayout.JAVA_BYTE, sa + mismatch) & 0xFF, db.get(ValueLayout.JAVA_BYTE, sb + mismatch) & 0xFF);
  }

  /**
   * Gathers {@code count} rows of a fixed-width column from several runs: output row {@code o} is
   * row {@code idx[o]} of {@code sources[runOf[o]]}. All sources have the same type.
   */
  public static void gatherFixed(
      VectorBuffers[] sources, int[] runOf, int[] idx, int count, MemorySegment outData, MemorySegment outValidity) {
    VecType type = sources[0].isDictionaryEncoded() ? VecType.INT32 : sources[0].type();
    switch (type) {
      case INT32 -> {
        for (int o = 0; o < count; o++) {
          outData.set(VectorBuffers.LE_INT, (long) o << 2, sources[runOf[o]].data().get(VectorBuffers.LE_INT, (long) idx[o] << 2));
        }
      }
      case INT64, FLOAT64 -> {
        for (int o = 0; o < count; o++) {
          outData.set(VectorBuffers.LE_LONG, (long) o << 3, sources[runOf[o]].data().get(VectorBuffers.LE_LONG, (long) idx[o] << 3));
        }
      }
      case BOOL -> {
        for (int o = 0; o < count; o++) {
          Bitmap.setTo(outData, o, Bitmap.isSet(sources[runOf[o]].data(), idx[o]));
        }
      }
      case DECIMAL128 -> {
        for (int o = 0; o < count; o++) {
          Decimal128.copy(sources[runOf[o]].data(), idx[o], outData, o);
        }
      }
      default -> throw new IllegalArgumentException("not fixed width: " + type);
    }
    if (outValidity != null) {
      gatherValidity(sources, runOf, idx, count, outValidity);
    }
  }

  /** Validity bits of the gathered rows: set unless the source row is null. */
  public static void gatherValidity(VectorBuffers[] sources, int[] runOf, int[] idx, int count, MemorySegment outValidity) {
    for (int base = 0; base < count; base += 64) {
      int limit = Math.min(64, count - base);
      long word = 0L;
      for (int j = 0; j < limit; j++) {
        int o = base + j;
        if (!sources[runOf[o]].isNull(idx[o])) {
          word |= 1L << j;
        }
      }
      Bitmap.setWord(outValidity, base >>> 6, count, word);
    }
  }

  /** Bytes needed by {@link #gatherUtf8} for the given rows of plain UTF8 columns. */
  public static long gatherUtf8Bytes(VectorBuffers[] sources, int[] runOf, int[] idx, int count) {
    long total = 0;
    for (int o = 0; o < count; o++) {
      VectorBuffers in = sources[runOf[o]];
      int i = idx[o];
      if (!in.isNull(i)) {
        MemorySegment off = in.offsets();
        total += off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) i << 2);
      }
    }
    return total;
  }

  /** Gathers plain UTF8 columns from several runs; {@code outData} must hold {@link #gatherUtf8Bytes}. */
  public static void gatherUtf8(
      VectorBuffers[] sources, int[] runOf, int[] idx, int count,
      MemorySegment outOffsets, MemorySegment outData, MemorySegment outValidity) {
    int pos = 0;
    for (int o = 0; o < count; o++) {
      VectorBuffers in = sources[runOf[o]];
      if (in.isDictionaryEncoded()) {
        throw new IllegalArgumentException("expected plain UTF8");
      }
      int i = idx[o];
      outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
      if (!in.isNull(i)) {
        MemorySegment off = in.offsets();
        int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
        int len = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - start;
        ByteCopy.copy(in.data(), start, outData, pos, len);
        pos += len;
      }
    }
    outOffsets.set(VectorBuffers.LE_INT, (long) count << 2, pos);
    if (outValidity != null) {
      gatherValidity(sources, runOf, idx, count, outValidity);
    }
  }
}
