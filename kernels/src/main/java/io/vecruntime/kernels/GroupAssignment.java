/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.vecruntime.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Group id per row of one batch, plus (for very few groups) one selection
 * bitmap per group so accumulators can reuse the masked SIMD reductions from
 * {@link AggKernels}. Beyond {@link #LOW_CARDINALITY} groups the accumulators
 * scatter row by row instead.
 *
 * <p>The threshold is deliberately tiny. With interleaved groups every 64-row
 * block is a partial mask, and on 128-bit SIMD (2 double lanes) building a lane
 * mask per pair of rows costs more than the reduction saves: measured on Apple
 * M3, the mask path is 2x faster with one group but 1.6x slower with 4 groups
 * and 5x slower with 16 (GroupedAggBenchmark). Wider registers move the
 * break-even point; the threshold can be tuned per platform.
 */
public final class GroupAssignment {

    /**
     * Maximum number of groups for the masked-reduction path; scatter above it.
     * The default is the crossover the lab measured (#283, {@code
     * docs/results.md} "Decision 3"): 4 on AVX-512 at both widths -- the masked
     * path is never slower than the scatter up to 4 groups in either rounding
     * mode and loses from 6 in Spark's-order mode -- and 1 on the 2-lane NEON
     * species, where the masked path lost to the scatter from 2 groups. The
     * 4-lane AVX2 species is unmeasured and keeps the NEON value.
     */
    public static final int LOW_CARDINALITY = Integer.getInteger("sparkvector.agg.maskPathMaxGroups", defaultMaskPathMaxGroups());

    static int defaultMaskPathMaxGroups() {
        return Platform.MASK_REGISTERS && Species.DOUBLE_LANES >= 4 ? 4 : 1;
    }

    private final int[] ids;
    private final int n;
    private final int numGroups;
    private final Arena arena;
    private final MemorySegment[] masks; // null when scattering
    private final int[] maskCounts;
    private final MemorySegment selection; // rows of the batch that take part; null = all

    private GroupAssignment(int[] ids, int n, int numGroups,
                            Arena arena, boolean lowCardinality, MemorySegment selection) {
        this.ids = ids;
        this.n = n;
        this.numGroups = numGroups;
        this.arena = arena;
        this.selection = selection;
        if (lowCardinality) {
            masks = new MemorySegment[numGroups];
            maskCounts = new int[numGroups];
            for (int i = 0; i < n; i++) {
                int g = ids[i];
                if (g < 0) {
                    continue; // unselected row (see GroupKeyTable.assign with a selection)
                }
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
    public static GroupAssignment of(int[] ids, int n, int numGroups,
            Arena arena) {
        return of(ids, n, numGroups, arena, numGroups <= LOW_CARDINALITY,
                null);
    }

    /**
     * As {@link #of(int[], int, int, Arena)} for a batch carrying a selection
     * bitmap: rows outside it have id {@code -1} and are ignored by every
     * accumulator.
     */
    public static GroupAssignment of(int[] ids, int n, int numGroups,
            Arena arena, MemorySegment selection) {
        return of(ids, n, numGroups, arena, numGroups <= LOW_CARDINALITY,
                selection);
    }

    /** Explicit choice of path, for benchmarks and tests. */
    public static GroupAssignment of(int[] ids, int n, int numGroups,
            Arena arena, boolean useMasks) {
        return new GroupAssignment(ids, n, numGroups, arena, useMasks, null);
    }

    public static GroupAssignment of(int[] ids, int n, int numGroups,
            Arena arena, boolean useMasks, MemorySegment selection) {
        return new GroupAssignment(ids, n, numGroups, arena, useMasks, selection);
    }

    /** The batch's selection bitmap, or {@code null} when every row takes part. */
    public MemorySegment selection() {
        return selection;
    }

    /**
     * The rows of {@code v} an accumulator must visit: the column's validity
     * ANDed with the selection. {@code null} means every row. Scatter-path
     * accumulators use this instead of {@code v.validity()} so unselected rows
     * (id -1) are never touched.
     */
    public MemorySegment effectiveValidity(VectorBuffers v) {
        MemorySegment validity = v.validity();
        if (selection == null) {
            return validity;
        }
        if (validity == null) {
            return selection;
        }
        MemorySegment combined = ArrowLayout.allocateBitmap(arena, n);
        BitmapKernels.and(validity, selection, combined, n);
        return combined;
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
     * The column restricted to group {@code g}: its validity is the group mask
     * ANDed with the column's own validity. Only for the mask path and groups
     * with at least one row.
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
        return new SegmentVectorBuffers(v.type(), n, combined, v.data(),
                v.offsets(), v.dictionary());
    }
}
