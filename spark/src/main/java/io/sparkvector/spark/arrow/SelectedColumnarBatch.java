/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.spark.arrow;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.BitmapKernels;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A batch whose rows are qualified by a selection bitmap instead of being
 * compacted. Emitted by a spark-vector operator only when its parent is another
 * spark-vector operator (Spark's own consumers would read every row), so a
 * filter feeding an aggregate never copies the surviving rows: the aggregate
 * folds the selection into its validity masks.
 *
 * <p>{@link #numRows()} is the physical row count of the columns; {@link
 * #selectedCount()} the number of qualifying rows. The selection lives in this
 * batch's own arena, which is released when the batch is closed. Columns are
 * closed only if {@code ownsColumns}; a filter borrows its child's columns (the
 * child releases them when it produces its next batch), a projection owns the
 * vectors it built.
 */
public final class SelectedColumnarBatch extends ColumnarBatch {

    private final Arena arena;
    private final MemorySegment selection;
    private final int selectedCount;
    private final boolean ownsColumns;

    private SelectedColumnarBatch(ColumnVector[] columns, int numRows, Arena arena,
            MemorySegment selection, int selectedCount, boolean ownsColumns) {
        super(columns, numRows);
        this.arena = arena;
        this.selection = selection;
        this.selectedCount = selectedCount;
        this.ownsColumns = ownsColumns;
    }

    /**
     * Wraps {@code columns} with a copy of {@code selection} ({@code numRows}
     * bits, {@code selectedCount} set), taken into a fresh arena owned by the
     * batch.
     */
    public static SelectedColumnarBatch of(ColumnVector[] columns, int numRows, MemorySegment selection,
            int selectedCount, boolean ownsColumns) {
        Arena arena = Arena.ofConfined();
        MemorySegment copy = ArrowLayout.allocateBitmap(arena, numRows);
        BitmapKernels.copy(selection, copy, numRows);
        return new SelectedColumnarBatch(columns, numRows, arena, copy, selectedCount, ownsColumns);
    }

    /**
     * Wraps {@code columns} ({@code numRows} physical rows) selecting the rows
     * whose positions are {@code indices[0..selectedCount)}; the selection
     * bitmap lives in a fresh arena owned by the batch. This is how a foreign
     * reader's row-id mapping (Iceberg's {@code ColumnVectorWithFilter})
     * becomes a selection our operators evaluate over the full batch.
     */
    public static SelectedColumnarBatch ofIndices(ColumnVector[] columns, int numRows, int[] indices,
            int selectedCount, boolean ownsColumns) {
        Arena arena = Arena.ofConfined();
        MemorySegment selection = ArrowLayout.selectionFromIndices(arena, indices, selectedCount, numRows);
        return new SelectedColumnarBatch(columns, numRows, arena, selection, selectedCount, ownsColumns);
    }

    /** The columns of {@code batch}, in order (borrowed, not copied). */
    public static ColumnVector[] columnsOf(ColumnarBatch batch) {
        ColumnVector[] columns = new ColumnVector[batch.numCols()];
        for (int i = 0; i < columns.length; i++) {
            columns[i] = batch.column(i);
        }
        return columns;
    }

    public MemorySegment selection() {
        return selection;
    }

    public int selectedCount() {
        return selectedCount;
    }

    @Override
    public void close() {
        if (ownsColumns) {
            for (int i = 0; i < numCols(); i++) {
                column(i).close();
            }
        }
        arena.close();
    }
}
