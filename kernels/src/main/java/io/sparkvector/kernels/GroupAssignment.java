package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Group id per row of one batch, plus (for few groups) one selection bitmap per group so
 * accumulators can reuse the masked SIMD reductions from {@link AggKernels}. Beyond
 * {@link #LOW_CARDINALITY} groups the accumulators scatter row by row instead.
 */
public final class GroupAssignment {

  public static final int LOW_CARDINALITY = 64;

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
    return new GroupAssignment(ids, n, numGroups, arena, numGroups <= LOW_CARDINALITY);
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
