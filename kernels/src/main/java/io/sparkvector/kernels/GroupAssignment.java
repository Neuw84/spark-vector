package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Group id per row of one batch, plus (for very few groups) one selection bitmap per group so
 * accumulators can reuse the masked SIMD reductions from {@link AggKernels}. Beyond
 * {@link #LOW_CARDINALITY} groups the accumulators scatter row by row instead.
 *
 * <p>The threshold is deliberately tiny. With interleaved groups every 64-row block is a partial
 * mask, and on 128-bit SIMD (2 double lanes) building a lane mask per pair of rows costs more than
 * the reduction saves: measured on Apple M3, the mask path is 2x faster with one group but 1.6x
 * slower with 4 groups and 5x slower with 16 (GroupedAggBenchmark). Wider registers move the
 * break-even point; the threshold can be tuned per platform.
 */
public final class GroupAssignment {

  /** Maximum number of groups for the masked-reduction path; scatter above it. */
  public static final int LOW_CARDINALITY =
      Integer.getInteger("sparkvector.agg.maskPathMaxGroups", DoubleVectorLanes() >= 8 ? 8 : 1);

  private static int DoubleVectorLanes() {
    return jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
  }

  private final int[] ids;
  private final int n;
  private final int numGroups;
  private final Arena arena;
  private final MemorySegment[] masks; // null when scattering
  private final int[] maskCounts;

  private GroupAssignment(int[] ids, int n, int numGroups, Arena arena, boolean lowCardinality) {
    this.ids = ids;
    this.n = n;
    this.numGroups = numGroups;
    this.arena = arena;
    if (lowCardinality) {
      masks = new MemorySegment[numGroups];
      maskCounts = new int[numGroups];
      for (int i = 0; i < n; i++) {
        int g = ids[i];
        MemorySegment m = masks[g];
        if (m == null) {
          m = ArrowLayout.allocateBitmap(arena, n);
          masks[g] = m;
        }
        Bitmap.set(m, i);
        maskCounts[g]++;
      }
    } else {
      masks = null;
      maskCounts = null;
    }
  }

  /** {@code numGroups} is the total number of groups seen so far in the task. */
  public static GroupAssignment of(int[] ids, int n, int numGroups, Arena arena) {
    return of(ids, n, numGroups, arena, numGroups <= LOW_CARDINALITY);
  }

  /** Explicit choice of path, for benchmarks and tests. */
  public static GroupAssignment of(int[] ids, int n, int numGroups, Arena arena, boolean useMasks) {
    return new GroupAssignment(ids, n, numGroups, arena, useMasks);
  }

  public int[] ids() {
    return ids;
  }

  public int numRows() {
    return n;
  }

  public int numGroups() {
    return numGroups;
  }

  public boolean useMasks() {
    return masks != null;
  }

  /** Rows of group {@code g} in this batch (mask path only). */
  public int maskCount(int g) {
    return masks[g] == null ? 0 : maskCounts[g];
  }

  /**
   * The column restricted to group {@code g}: its validity is the group mask ANDed with the
   * column's own validity. Only for the mask path and groups with at least one row.
   */
  public VectorBuffers restrict(VectorBuffers v, int g) {
    MemorySegment m = masks[g];
    MemorySegment validity = v.validity();
    MemorySegment combined;
    if (validity == null) {
      combined = m;
    } else {
      combined = ArrowLayout.allocateBitmap(arena, n);
      BitmapKernels.and(m, validity, combined, n);
    }
    return new SegmentVectorBuffers(v.type(), n, combined, v.data(), v.offsets(), v.dictionary());
  }
}
