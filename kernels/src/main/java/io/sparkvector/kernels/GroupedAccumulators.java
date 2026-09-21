package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Per-group running state for the aggregate functions. Each accumulator grows with the group
 * table and updates from a batch either through per-group masked reductions (few groups) or a
 * scalar scatter over the group ids (many groups).
 */
public final class GroupedAccumulators {

  private GroupedAccumulators() {}

  private static int grow(int current, int needed) {
    return needed <= current ? current : Math.max(needed, current * 2);
  }

  /**
   * Number of independent accumulator sets the scatter loops rotate through. A single {@code
   * sum[g] += x} chain serialises on store-to-load forwarding whenever consecutive rows share a
   * group (about 1.5 ns per row); rotating over {@code INTERLEAVE} copies lets the CPU overlap
   * them (+40% at TPC-H Q1's four groups). Integer sums and counts are exact whatever the order.
   * Double sums are not: the copies are added on read, so they round differently from Spark's
   * sequential loop, and TPC-H Q15, which compares a double sum against the maximum of the same
   * sums computed by Spark, returned no rows. {@link DoubleSum} therefore takes a {@code strict}
   * flag (one accumulator, sequential reductions: Spark's rounding) that the operator sets from
   * {@code spark.vector.exec.strictFloatingPoint}; this property only sets the copies used when
   * strictness is off.
   */
  public static final int INTERLEAVE = interleave();

  /**
   * Whether the property was set to 1 explicitly: the legacy meaning of {@code
   * sparkvector.agg.interleave=1} is "Spark's order everywhere", which the ungrouped double sums
   * in {@link AggKernels} honour too. The platform default of one copy (below) does not carry that
   * meaning: it only picks the faster scatter loop.
   */
  public static final boolean SEQUENTIAL_SUMS = "1".equals(System.getProperty("sparkvector.agg.interleave"));

  /**
   * The copies the scatter loops rotate through when the property is unset. Measured in the x86
   * lab (#283, {@code docs/results.md} "Decision 3"): on AVX-512 one copy is 45-70% faster than
   * four from 4 groups up -- the only range where the scatter runs, the masked path owning the
   * groups below -- and four copies win only at 1-2 groups. The NEON measurement above (+40% for
   * four copies at 4 groups) stands for the other platforms.
   */
  static int defaultInterleave() {
    return Platform.NAME.equals(Platform.AVX512) ? 1 : 4;
  }

  private static int interleave() {
    int v = Integer.getInteger("sparkvector.agg.interleave", defaultInterleave());
    if (v != 1 && v != 2 && v != 4) {
      throw new IllegalArgumentException("sparkvector.agg.interleave must be 1, 2 or 4, got " + v);
    }
    return v;
  }

  /** Per-batch scratch shared by the scalar scatter loops (batches are bounded by the caller). */
  private static final class Scratch {
    long[] longs = new long[0];
    int[] ints = new int[0];

    long[] longs(MemorySegment data, int n) {
      if (longs.length < n) {
        longs = new long[Math.max(n, longs.length * 2)];
      }
      MemorySegment.copy(data, VectorBuffers.LE_LONG, 0, longs, 0, n);
      return longs;
    }

    long[] longsFromInts(MemorySegment data, int n) {
      if (ints.length < n) {
        ints = new int[Math.max(n, ints.length * 2)];
      }
      if (longs.length < n) {
        longs = new long[Math.max(n, longs.length * 2)];
      }
      MemorySegment.copy(data, VectorBuffers.LE_INT, 0, ints, 0, n);
      for (int i = 0; i < n; i++) {
        longs[i] = ints[i];
      }
      return longs;
    }
  }

  /**
   * SUM over doubles; also the (sum, count) buffer of AVG. Accumulators are {@code INTERLEAVE}
   * copies laid out {@code [copy][group]}; {@link #sum(int)} adds the copies.
   */
  public static final class DoubleSum {
    /** Copies of the accumulators; 1 when strict. */
    private final int interleave;
    /** Spark's rounding: one accumulator per group, sequential reductions on the masked path. */
    private final boolean strict;
    private int capacity = 64;
    private double[] sum;
    private long[] count;

    /** Fast rounding: {@link #INTERLEAVE} copies and lane-parallel reductions. */
    public DoubleSum() {
      this(false);
    }

    public DoubleSum(boolean strict) {
      this.strict = strict;
      this.interleave = strict ? 1 : INTERLEAVE;
      this.sum = new double[64 * interleave];
      this.count = new long[64 * interleave];
    }

    public void update(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          VectorBuffers sub = a.restrict(v, g);
          long c = AggKernels.countValid(sub);
          if (c > 0) {
            if (strict) {
              // Spark's `sum += x` chain continues from the running sum, not from a per-batch zero.
              sum[g] = AggKernels.sumDoubleSequential(sub, sum[g]);
            } else {
              sum[g] += AggKernels.sumDouble(sub);
            }
            count[g] += c;
          }
        }
        return;
      }
      int[] ids = a.ids();
      int n = a.numRows();
      MemorySegment x = v.data();
      MemorySegment validity = a.effectiveValidity(v);
      double[] sum = this.sum;
      long[] count = this.count;
      int cap = capacity;
      if (validity == null) {
        int i = 0;
        if (interleave == 4) {
          int c1 = cap, c2 = 2 * cap, c3 = 3 * cap;
          for (; i + 4 <= n; i += 4) {
            int g0 = ids[i], g1 = ids[i + 1] + c1, g2 = ids[i + 2] + c2, g3 = ids[i + 3] + c3;
            sum[g0] += x.get(VectorBuffers.LE_DOUBLE, (long) (i) << 3);
            sum[g1] += x.get(VectorBuffers.LE_DOUBLE, (long) (i + 1) << 3);
            sum[g2] += x.get(VectorBuffers.LE_DOUBLE, (long) (i + 2) << 3);
            sum[g3] += x.get(VectorBuffers.LE_DOUBLE, (long) (i + 3) << 3);
            count[g0]++;
            count[g1]++;
            count[g2]++;
            count[g3]++;
          }
        } else if (interleave == 2) {
          for (; i + 2 <= n; i += 2) {
            int g0 = ids[i], g1 = ids[i + 1] + cap;
            sum[g0] += x.get(VectorBuffers.LE_DOUBLE, (long) (i) << 3);
            sum[g1] += x.get(VectorBuffers.LE_DOUBLE, (long) (i + 1) << 3);
            count[g0]++;
            count[g1]++;
          }
        }
        for (; i < n; i++) {
          int g = ids[i];
          sum[g] += x.get(VectorBuffers.LE_DOUBLE, (long) (i) << 3);
          count[g]++;
        }
      } else {
        int slot = 0; // rotates over the copies for the valid rows only
        int total = cap * interleave;
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            int g = ids[i] + slot;
            sum[g] += x.get(VectorBuffers.LE_DOUBLE, (long) (i) << 3);
            count[g]++;
            slot += cap;
            if (slot == total) {
              slot = 0;
            }
          }
        }
      }
    }

    private void ensure(int groups) {
      if (groups > capacity) {
        int cap = grow(capacity, groups);
        sum = regroup(sum, capacity, cap, interleave);
        count = regroup(count, capacity, cap, interleave);
        capacity = cap;
      }
    }

    public double sum(int g) {
      double s = 0;
      for (int k = 0; k < interleave; k++) {
        s += sum[k * capacity + g];
      }
      return s;
    }

    public long count(int g) {
      long c = 0;
      for (int k = 0; k < interleave; k++) {
        c += count[k * capacity + g];
      }
      return c;
    }
  }

  /** Re-lays {@code [copy][group]} arrays out for a larger group capacity. */
  private static double[] regroup(double[] old, int oldCap, int newCap) {
    return regroup(old, oldCap, newCap, INTERLEAVE);
  }

  private static double[] regroup(double[] old, int oldCap, int newCap, int copies) {
    double[] out = new double[newCap * copies];
    for (int k = 0; k < copies; k++) {
      System.arraycopy(old, k * oldCap, out, k * newCap, oldCap);
    }
    return out;
  }

  private static long[] regroup(long[] old, int oldCap, int newCap) {
    return regroup(old, oldCap, newCap, INTERLEAVE);
  }

  private static long[] regroup(long[] old, int oldCap, int newCap, int copies) {
    long[] out = new long[newCap * copies];
    for (int k = 0; k < copies; k++) {
      System.arraycopy(old, k * oldCap, out, k * newCap, oldCap);
    }
    return out;
  }

  /**
   * SUM over ints or longs into long accumulators. Wraps on overflow like Spark's legacy mode, or,
   * when {@code checked}, detects it with {@code Math.addExact} (the scatter loops) and {@link
   * AggKernels#sumLongExact} (the masked path) and throws {@link ArithmeticException}, which is
   * what Spark's ANSI {@code sum(bigint)} needs.
   */
  public static final class LongSum {
    private int capacity = 64;
    private long[] sum = new long[64 * INTERLEAVE];
    private long[] count = new long[64 * INTERLEAVE];
    private final Scratch scratch = new Scratch();
    private final boolean checked;

    public LongSum() {
      this(false);
    }

    public LongSum(boolean checked) {
      this.checked = checked;
    }

    public void update(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      boolean ints = v.type() == VecType.INT32;
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          VectorBuffers sub = a.restrict(v, g);
          long c = AggKernels.countValid(sub);
          if (c > 0) {
            long s = ints ? AggKernels.sumInt(sub) : (checked ? AggKernels.sumLongExact(sub) : AggKernels.sumLong(sub));
            sum[g] = checked ? Math.addExact(sum[g], s) : sum[g] + s;
            count[g] += c;
          }
        }
        return;
      }
      int[] ids = a.ids();
      int n = a.numRows();
      // Ints are widened into the long scratch so one loop serves both types.
      long[] x = ints ? scratch.longsFromInts(v.data(), n) : scratch.longs(v.data(), n);
      MemorySegment validity = a.effectiveValidity(v);
      long[] sum = this.sum;
      long[] count = this.count;
      int cap = capacity;
      if (checked) {
        updateChecked(x, ids, n, validity, sum, count, cap);
        return;
      }
      if (validity == null) {
        int i = 0;
        if (INTERLEAVE == 4) {
          int c1 = cap, c2 = 2 * cap, c3 = 3 * cap;
          for (; i + 4 <= n; i += 4) {
            int g0 = ids[i], g1 = ids[i + 1] + c1, g2 = ids[i + 2] + c2, g3 = ids[i + 3] + c3;
            sum[g0] += x[i];
            sum[g1] += x[i + 1];
            sum[g2] += x[i + 2];
            sum[g3] += x[i + 3];
            count[g0]++;
            count[g1]++;
            count[g2]++;
            count[g3]++;
          }
        } else if (INTERLEAVE == 2) {
          for (; i + 2 <= n; i += 2) {
            int g0 = ids[i], g1 = ids[i + 1] + cap;
            sum[g0] += x[i];
            sum[g1] += x[i + 1];
            count[g0]++;
            count[g1]++;
          }
        }
        for (; i < n; i++) {
          int g = ids[i];
          sum[g] += x[i];
          count[g]++;
        }
      } else {
        int slot = 0;
        int total = cap * INTERLEAVE;
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            int g = ids[i] + slot;
            sum[g] += x[i];
            count[g]++;
            slot += cap;
            if (slot == total) {
              slot = 0;
            }
          }
        }
      }
    }

    private void ensure(int groups) {
      if (groups > capacity) {
        int cap = grow(capacity, groups);
        sum = regroup(sum, capacity, cap);
        count = regroup(count, capacity, cap);
        capacity = cap;
      }
    }

    /** Scatter with exact adds; the interleaving is kept so the sums are laid out identically. */
    private void updateChecked(long[] x, int[] ids, int n, MemorySegment validity, long[] sum, long[] count, int cap) {
      int slot = 0;
      int total = cap * INTERLEAVE;
      if (validity == null) {
        for (int i = 0; i < n; i++) {
          int g = ids[i] + slot;
          sum[g] = Math.addExact(sum[g], x[i]);
          count[g]++;
          slot += cap;
          if (slot == total) {
            slot = 0;
          }
        }
      } else {
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            int g = ids[i] + slot;
            sum[g] = Math.addExact(sum[g], x[i]);
            count[g]++;
            slot += cap;
            if (slot == total) {
              slot = 0;
            }
          }
        }
      }
    }

    public long sum(int g) {
      long s = 0;
      for (int k = 0; k < INTERLEAVE; k++) {
        s = checked ? Math.addExact(s, sum[k * capacity + g]) : s + sum[k * capacity + g];
      }
      return s;
    }

    public long count(int g) {
      long c = 0;
      for (int k = 0; k < INTERLEAVE; k++) {
        c += count[k * capacity + g];
      }
      return c;
    }
  }

  /**
   * SUM over INT64 lanes into a 128-bit signed accumulator per group: the unscaled values of a
   * decimal whose sum type is wider than 18 digits (Spark's {@code Decimal(p + 10, s)} buffer). An
   * accumulator is one value per group, not per row, so scalar two-word arithmetic with a carry is
   * all it takes -- no wide lane, no SIMD. Rows are added one at a time (a batch partial sum of
   * 18-digit values does not reliably fit a long), which also makes the ungrouped path the same
   * accumulator with every row in group 0. Exact, so the interleaving question of the double sums
   * does not arise and none is done.
   */
  public static final class WideLongSum {
    private int capacity = 64;
    private long[] hi = new long[64];
    private long[] lo = new long[64];
    private long[] count = new long[64];
    private boolean[] overflowed = new boolean[64];
    private final Scratch scratch = new Scratch();

    /**
     * Adds every valid row of {@code v} (INT32, INT64, or a DECIMAL128 lane whose limbs are added as
     * a signed 128-bit value) to the group {@code ids} says.
     */
    public void update(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      if (v.type() == VecType.DECIMAL128) {
        updateWide(v, a);
        return;
      }
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          addAll(a.restrict(v, g), g);
        }
        return;
      }
      int[] ids = a.ids();
      int n = a.numRows();
      long[] x = v.type() == VecType.INT32 ? scratch.longsFromInts(v.data(), n) : scratch.longs(v.data(), n);
      MemorySegment validity = a.effectiveValidity(v);
      if (validity == null) {
        for (int i = 0; i < n; i++) {
          add(ids[i], x[i]);
        }
      } else {
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            add(ids[i], x[i]);
          }
        }
      }
    }

    /** The ungrouped path: every valid row of {@code v} into group 0. */
    public void updateAll(VectorBuffers v) {
      ensure(1);
      if (v.type() == VecType.DECIMAL128) {
        addAllWide(v, 0);
      } else {
        addAll(v, 0);
      }
    }

    private void updateWide(VectorBuffers v, GroupAssignment a) {
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          addAllWide(a.restrict(v, g), g);
        }
        return;
      }
      int[] ids = a.ids();
      int n = a.numRows();
      MemorySegment data = v.data();
      MemorySegment validity = a.effectiveValidity(v);
      if (validity == null) {
        for (int i = 0; i < n; i++) {
          addWide(ids[i], Decimal128.hi(data, i), Decimal128.lo(data, i));
        }
      } else {
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            addWide(ids[i], Decimal128.hi(data, i), Decimal128.lo(data, i));
          }
        }
      }
    }

    private void addAllWide(VectorBuffers v, int g) {
      int n = v.length();
      MemorySegment data = v.data();
      MemorySegment validity = v.validity();
      if (validity == null) {
        for (int i = 0; i < n; i++) {
          addWide(g, Decimal128.hi(data, i), Decimal128.lo(data, i));
        }
      } else {
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            addWide(g, Decimal128.hi(data, i), Decimal128.lo(data, i));
          }
        }
      }
    }

    /**
     * {@code (hi, lo) += (xh, xl)} as signed 128-bit values. A total that leaves 128 bits sets the
     * group's sticky overflow flag: every decimal sum type is capped at 38 digits, below 2^127, so a
     * wrapped accumulator could only ever be reported as an overflowed sum, never as a value.
     */
    private void addWide(int g, long xh, long xl) {
      long l = lo[g];
      long sum = l + xl;
      long carry = ((l & xl) | ((l | xl) & ~sum)) >>> 63;
      long h = hi[g];
      long hsum = h + xh + carry;
      // Signed overflow of the high word: both operands share a sign the result does not have.
      if (((h ^ hsum) & (xh ^ hsum)) < 0) {
        overflowed[g] = true;
      }
      lo[g] = sum;
      hi[g] = hsum;
      count[g]++;
    }

    /** Whether the group's total left 128 bits at some point (the sum is then past any decimal precision). */
    public boolean overflowed(int g) {
      return overflowed[g];
    }

    private void addAll(VectorBuffers v, int g) {
      int n = v.length();
      long[] x = v.type() == VecType.INT32 ? scratch.longsFromInts(v.data(), n) : scratch.longs(v.data(), n);
      MemorySegment validity = v.validity();
      if (validity == null) {
        for (int i = 0; i < n; i++) {
          add(g, x[i]);
        }
      } else {
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(validity, w, n);
          while (bits != 0L) {
            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
            bits &= bits - 1;
            add(g, x[i]);
          }
        }
      }
    }

    /** {@code (hi, lo) += sign-extended x}: the carry out of the low word, then the sign word. */
    private void add(int g, long x) {
      long l = lo[g];
      long sum = l + x;
      long carry = ((l & x) | ((l | x) & ~sum)) >>> 63;
      lo[g] = sum;
      hi[g] += (x >> 63) + carry;
      count[g]++;
    }

    private void ensure(int groups) {
      if (groups > capacity) {
        int cap = grow(capacity, groups);
        hi = Arrays.copyOf(hi, cap);
        lo = Arrays.copyOf(lo, cap);
        count = Arrays.copyOf(count, cap);
        overflowed = Arrays.copyOf(overflowed, cap);
        capacity = cap;
      }
    }

    /** High word of the group's signed 128-bit sum. */
    public long hi(int g) {
      return hi[g];
    }

    /** Low word (unsigned) of the group's signed 128-bit sum. */
    public long lo(int g) {
      return lo[g];
    }

    public long count(int g) {
      return count[g];
    }

    /** The group's sum as a {@link java.math.BigInteger}. */
    public java.math.BigInteger sum(int g) {
      return toBigInteger(hi[g], lo[g]);
    }

    /**
     * {@code hi * 2^64 + lo} with {@code lo} unsigned: a signed 128-bit value. Built from the 16
     * big-endian bytes -- the previous form went through {@code Long.toUnsignedString} and a decimal
     * parse per group, 2% of an executor's time when a wide-decimal partial emits every row (#388).
     */
    public static java.math.BigInteger toBigInteger(long hi, long lo) {
      return Decimal128.toBigInteger(hi, lo);
    }
  }

  /** COUNT(*) or COUNT(expr). */
  public static final class Count {
    private int capacity = 64;
    private long[] count = new long[64 * INTERLEAVE];

    /** Counts every row of each group. */
    public void updateAll(GroupAssignment a) {
      ensure(a.numGroups());
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          count[g] += a.maskCount(g);
        }
        return;
      }
      int[] ids = a.ids();
      int n = a.numRows();
      long[] count = this.count;
      int cap = capacity;
      if (a.selection() != null) {
        int slot = 0;
        int total = cap * INTERLEAVE;
        MemorySegment selection = a.selection();
        for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
          long bits = Bitmap.wordAt(selection, w, n);
          while (bits != 0L) {
            count[ids[(w << 6) + Long.numberOfTrailingZeros(bits)] + slot]++;
            bits &= bits - 1;
            slot += cap;
            if (slot == total) {
              slot = 0;
            }
          }
        }
        return;
      }
      int i = 0;
      if (INTERLEAVE == 4) {
        int c1 = cap, c2 = 2 * cap, c3 = 3 * cap;
        for (; i + 4 <= n; i += 4) {
          count[ids[i]]++;
          count[ids[i + 1] + c1]++;
          count[ids[i + 2] + c2]++;
          count[ids[i + 3] + c3]++;
        }
      } else if (INTERLEAVE == 2) {
        for (; i + 2 <= n; i += 2) {
          count[ids[i]]++;
          count[ids[i + 1] + cap]++;
        }
      }
      for (; i < n; i++) {
        count[ids[i]]++;
      }
    }

    /** Counts the non-null values of {@code v} per group. */
    public void updateNonNull(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      MemorySegment validity = a.effectiveValidity(v);
      if (validity == null) {
        updateAll(a);
        return;
      }
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) > 0) {
            count[g] += AggKernels.countValid(a.restrict(v, g));
          }
        }
        return;
      }
      int[] ids = a.ids();
      int n = a.numRows();
      long[] count = this.count;
      int cap = capacity;
      int slot = 0;
      int total = cap * INTERLEAVE;
      for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
        long bits = Bitmap.wordAt(validity, w, n);
        while (bits != 0L) {
          count[ids[(w << 6) + Long.numberOfTrailingZeros(bits)] + slot]++;
          bits &= bits - 1;
          slot += cap;
          if (slot == total) {
            slot = 0;
          }
        }
      }
    }

    private void ensure(int groups) {
      if (groups > capacity) {
        int cap = grow(capacity, groups);
        count = regroup(count, capacity, cap);
        capacity = cap;
      }
    }

    public long count(int g) {
      long c = 0;
      for (int k = 0; k < INTERLEAVE; k++) {
        c += count[k * capacity + g];
      }
      return c;
    }
  }

  /** MIN / MAX over doubles with Spark's NaN ordering. */
  public static final class DoubleMinMax {
    private final boolean min;
    private double[] best = new double[64];
    private boolean[] any = new boolean[64];

    public DoubleMinMax(boolean min) {
      this.min = min;
    }

    public void update(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          VectorBuffers sub = a.restrict(v, g);
          if (AggKernels.countValid(sub) > 0) {
            offer(g, min ? AggKernels.minDouble(sub) : AggKernels.maxDouble(sub));
          }
        }
      } else {
        int[] ids = a.ids();
        MemorySegment data = v.data();
        MemorySegment validity = a.effectiveValidity(v);
        int n = a.numRows();
        if (validity == null) {
          for (int i = 0; i < n; i++) {
            offer(ids[i], data.get(VectorBuffers.LE_DOUBLE, (long) i << 3));
          }
        } else {
          for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
            long bits = Bitmap.wordAt(validity, w, n);
            while (bits != 0L) {
              int i = (w << 6) + Long.numberOfTrailingZeros(bits);
              bits &= bits - 1;
              offer(ids[i], data.get(VectorBuffers.LE_DOUBLE, (long) i << 3));
            }
          }
        }
      }
    }

    private void offer(int g, double x) {
      if (!any[g]) {
        best[g] = x;
        any[g] = true;
      } else {
        int cmp = CompareOp.nanSafeCompare(x, best[g]);
        if (min ? cmp < 0 : cmp > 0) {
          best[g] = x;
        }
      }
    }

    private void ensure(int groups) {
      if (groups > best.length) {
        int cap = grow(best.length, groups);
        best = Arrays.copyOf(best, cap);
        any = Arrays.copyOf(any, cap);
      }
    }

    public boolean hasValue(int g) {
      return any[g];
    }

    public double value(int g) {
      return best[g];
    }
  }

  /** MIN / MAX over ints (incl. dates) and longs (incl. timestamps). */
  public static final class LongMinMax {
    private final boolean min;
    private long[] best = new long[64];
    private boolean[] any = new boolean[64];

    public LongMinMax(boolean min) {
      this.min = min;
    }

    public void update(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      boolean ints = v.type() == VecType.INT32;
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          VectorBuffers sub = a.restrict(v, g);
          if (AggKernels.countValid(sub) > 0) {
            long x = ints
                ? (min ? AggKernels.minInt(sub) : AggKernels.maxInt(sub))
                : (min ? AggKernels.minLong(sub) : AggKernels.maxLong(sub));
            offer(g, x);
          }
        }
      } else {
        int[] ids = a.ids();
        MemorySegment data = v.data();
        MemorySegment validity = a.effectiveValidity(v);
        int n = a.numRows();
        if (validity == null) {
          for (int i = 0; i < n; i++) {
            offer(ids[i], ints ? data.get(VectorBuffers.LE_INT, (long) i << 2) : data.get(VectorBuffers.LE_LONG, (long) i << 3));
          }
        } else {
          for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
            long bits = Bitmap.wordAt(validity, w, n);
            while (bits != 0L) {
              int i = (w << 6) + Long.numberOfTrailingZeros(bits);
              bits &= bits - 1;
              offer(ids[i], ints ? data.get(VectorBuffers.LE_INT, (long) i << 2) : data.get(VectorBuffers.LE_LONG, (long) i << 3));
            }
          }
        }
      }
    }

    private void offer(int g, long x) {
      if (!any[g]) {
        best[g] = x;
        any[g] = true;
      } else if (min ? x < best[g] : x > best[g]) {
        best[g] = x;
      }
    }

    private void ensure(int groups) {
      if (groups > best.length) {
        int cap = grow(best.length, groups);
        best = Arrays.copyOf(best, cap);
        any = Arrays.copyOf(any, cap);
      }
    }

    public boolean hasValue(int g) {
      return any[g];
    }

    public long value(int g) {
      return best[g];
    }
  }

  /**
   * Per-group min or max over a DECIMAL128 lane: a two-limb compare per valid row (the lane is
   * scalar), results kept as limbs.
   */
  public static final class Decimal128MinMax {
    private final boolean min;
    private long[] bestHi = new long[64];
    private long[] bestLo = new long[64];
    private boolean[] any = new boolean[64];

    public Decimal128MinMax(boolean min) {
      this.min = min;
    }

    public void update(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      if (v.type() != VecType.DECIMAL128) {
        throw new IllegalArgumentException("expected DECIMAL128, got " + v.type());
      }
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          if (a.maskCount(g) == 0) {
            continue;
          }
          VectorBuffers sub = a.restrict(v, g);
          MemorySegment data = sub.data();
          int n = sub.length();
          for (int i = 0; i < n; i++) {
            if (!sub.isNull(i)) {
              offer(g, Decimal128.hi(data, i), Decimal128.lo(data, i));
            }
          }
        }
      } else {
        int[] ids = a.ids();
        MemorySegment data = v.data();
        MemorySegment validity = a.effectiveValidity(v);
        int n = a.numRows();
        if (validity == null) {
          for (int i = 0; i < n; i++) {
            offer(ids[i], Decimal128.hi(data, i), Decimal128.lo(data, i));
          }
        } else {
          for (int w = 0, words = Bitmap.wordsFor(n); w < words; w++) {
            long bits = Bitmap.wordAt(validity, w, n);
            while (bits != 0L) {
              int i = (w << 6) + Long.numberOfTrailingZeros(bits);
              bits &= bits - 1;
              offer(ids[i], Decimal128.hi(data, i), Decimal128.lo(data, i));
            }
          }
        }
      }
    }

    private void offer(int g, long hi, long lo) {
      if (!any[g]) {
        bestHi[g] = hi;
        bestLo[g] = lo;
        any[g] = true;
      } else {
        int c = Decimal128.compare(hi, lo, bestHi[g], bestLo[g]);
        if (min ? c < 0 : c > 0) {
          bestHi[g] = hi;
          bestLo[g] = lo;
        }
      }
    }

    private void ensure(int groups) {
      if (groups > bestHi.length) {
        int cap = grow(bestHi.length, groups);
        bestHi = Arrays.copyOf(bestHi, cap);
        bestLo = Arrays.copyOf(bestLo, cap);
        any = Arrays.copyOf(any, cap);
      }
    }

    public boolean hasValue(int g) {
      return any[g];
    }

    public long hi(int g) {
      return bestHi[g];
    }

    public long lo(int g) {
      return bestLo[g];
    }

    public java.math.BigInteger value(int g) {
      return Decimal128.toBigInteger(bestHi[g], bestLo[g]);
    }

    /** Writes the results of groups {@code [from, to)} as a DECIMAL128 lane with validity. */
    public void writeTo(int from, int to, MemorySegment validity, MemorySegment data) {
      for (int o = 0; o < to - from; o++) {
        int g = from + o;
        Bitmap.setTo(validity, o, any[g]);
        Decimal128.set(data, o, any[g] ? bestHi[g] : 0L, any[g] ? bestLo[g] : 0L);
      }
    }
  }
}
