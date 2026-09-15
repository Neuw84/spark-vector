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

  /** SUM over doubles; also the (sum, count) buffer of AVG. */
  public static final class DoubleSum {
    private double[] sum = new double[64];
    private long[] count = new long[64];

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
            sum[g] += AggKernels.sumDouble(sub);
            count[g] += c;
          }
        }
      } else {
        int[] ids = a.ids();
        MemorySegment validity = v.validity();
        for (int i = 0; i < a.numRows(); i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            int g = ids[i];
            sum[g] += v.getDouble(i);
            count[g]++;
          }
        }
      }
    }

    private void ensure(int groups) {
      if (groups > sum.length) {
        int cap = grow(sum.length, groups);
        sum = Arrays.copyOf(sum, cap);
        count = Arrays.copyOf(count, cap);
      }
    }

    public double sum(int g) {
      return sum[g];
    }

    public long count(int g) {
      return count[g];
    }
  }

  /** SUM over ints or longs into long accumulators (wrapping like Spark's legacy mode). */
  public static final class LongSum {
    private long[] sum = new long[64];
    private long[] count = new long[64];

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
            sum[g] += ints ? AggKernels.sumInt(sub) : AggKernels.sumLong(sub);
            count[g] += c;
          }
        }
      } else {
        int[] ids = a.ids();
        MemorySegment validity = v.validity();
        for (int i = 0; i < a.numRows(); i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            int g = ids[i];
            sum[g] += ints ? v.getInt(i) : v.getLong(i);
            count[g]++;
          }
        }
      }
    }

    private void ensure(int groups) {
      if (groups > sum.length) {
        int cap = grow(sum.length, groups);
        sum = Arrays.copyOf(sum, cap);
        count = Arrays.copyOf(count, cap);
      }
    }

    public long sum(int g) {
      return sum[g];
    }

    public long count(int g) {
      return count[g];
    }
  }

  /** COUNT(*) or COUNT(expr). */
  public static final class Count {
    private long[] count = new long[64];

    /** Counts every row of each group. */
    public void updateAll(GroupAssignment a) {
      ensure(a.numGroups());
      if (a.useMasks()) {
        for (int g = 0; g < a.numGroups(); g++) {
          count[g] += a.maskCount(g);
        }
      } else {
        int[] ids = a.ids();
        for (int i = 0; i < a.numRows(); i++) {
          count[ids[i]]++;
        }
      }
    }

    /** Counts the non-null values of {@code v} per group. */
    public void updateNonNull(VectorBuffers v, GroupAssignment a) {
      ensure(a.numGroups());
      MemorySegment validity = v.validity();
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
      } else {
        int[] ids = a.ids();
        for (int i = 0; i < a.numRows(); i++) {
          if (Bitmap.isSet(validity, i)) {
            count[ids[i]]++;
          }
        }
      }
    }

    private void ensure(int groups) {
      if (groups > count.length) {
        count = Arrays.copyOf(count, grow(count.length, groups));
      }
    }

    public long count(int g) {
      return count[g];
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
        MemorySegment validity = v.validity();
        for (int i = 0; i < a.numRows(); i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            offer(ids[i], v.getDouble(i));
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
        MemorySegment validity = v.validity();
        for (int i = 0; i < a.numRows(); i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            offer(ids[i], ints ? v.getInt(i) : v.getLong(i));
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
}
