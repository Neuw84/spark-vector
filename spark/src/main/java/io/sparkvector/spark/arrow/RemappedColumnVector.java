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

import java.lang.foreign.MemorySegment;

import io.sparkvector.kernels.Bitmap;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A dense view over a column the kernels cannot read (struct, array, map, wide
 * decimal): row {@code i} of the view is row {@code rows[i]} of the wrapped
 * Spark vector. This is how a selection is applied to a column that is passed
 * through untouched -- our own columns are compacted into new Arrow vectors, a
 * foreign one keeps Spark's vector and remaps the row ids, children included.
 * The wrapped vector stays owned by whoever produced it (the child keeps it
 * alive until its next batch), so closing the view closes nothing.
 */
public final class RemappedColumnVector extends ColumnVector {

    private final ColumnVector inner;
    private final int[] rows;
    private int numNulls = -1;

    private RemappedColumnVector(ColumnVector inner, int[] rows) {
        super(inner.dataType());
        this.inner = inner;
        this.rows = rows;
    }

    /**
     * The view of {@code cv} over the rows selected in the bitmap (the first
     * {@code count} set bits of {@code numRows}).
     */
    public static ColumnVector of(ColumnVector cv, int[] rows) {
        ColumnVector base = cv instanceof BorrowedColumnVector b ? b.inner() : cv;
        return new RemappedColumnVector(base, rows);
    }

    /**
     * The selected row ids of a bitmap, in order: the mapping every foreign
     * column of a batch shares.
     */
    public static int[] rowsOf(MemorySegment selection, int numRows, int count) {
        int[] rows = new int[count];
        int k = 0;
        for (int i = 0;
             i < numRows && k < count;
             i++) {
            if (Bitmap.isSet(selection, i)) {
                rows[k++] = i;
            }
        }
        if (k != count) {
            throw new IllegalStateException("selection has " + k + " rows, expected " + count);
        }
        return rows;
    }

    public ColumnVector inner() {
        return inner;
    }

    @Override
    public void close() {}

    @Override
    public boolean hasNull() {
        return inner.hasNull() && numNulls() > 0;
    }

    @Override
    public int numNulls() {
        if (numNulls < 0) {
            int n = 0;
            if (inner.hasNull()) {
                for (int r : rows) {
                    if (inner.isNullAt(r)) {
                        n++;
                    }
                }
            }
            numNulls = n;
        }
        return numNulls;
    }

    @Override
    public boolean isNullAt(int rowId) {
        return inner.isNullAt(rows[rowId]);
    }

    @Override
    public boolean getBoolean(int rowId) {
        return inner.getBoolean(rows[rowId]);
    }

    @Override
    public byte getByte(int rowId) {
        return inner.getByte(rows[rowId]);
    }

    @Override
    public short getShort(int rowId) {
        return inner.getShort(rows[rowId]);
    }

    @Override
    public int getInt(int rowId) {
        return inner.getInt(rows[rowId]);
    }

    @Override
    public long getLong(int rowId) {
        return inner.getLong(rows[rowId]);
    }

    @Override
    public float getFloat(int rowId) {
        return inner.getFloat(rows[rowId]);
    }

    @Override
    public double getDouble(int rowId) {
        return inner.getDouble(rows[rowId]);
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        return inner.getArray(rows[rowId]);
    }

    @Override
    public ColumnarMap getMap(int rowId) {
        return inner.getMap(rows[rowId]);
    }

    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        return inner.getDecimal(rows[rowId], precision, scale);
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        return inner.getUTF8String(rows[rowId]);
    }

    @Override
    public byte[] getBinary(int rowId) {
        return inner.getBinary(rows[rowId]);
    }

    @Override
    public org.apache.spark.unsafe.types.CalendarInterval getInterval(int rowId) {
        return inner.getInterval(rows[rowId]);
    }

    /** A struct's fields are read through the children, so they share the mapping. */
    @Override
    public ColumnVector getChild(int ordinal) {
        return new RemappedColumnVector(inner.getChild(ordinal), rows);
    }
}
