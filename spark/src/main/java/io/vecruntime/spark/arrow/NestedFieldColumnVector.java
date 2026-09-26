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

import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A struct field of a type without a lane (struct, array, map, wide decimal),
 * passed through as a column of its own: the field's child vector with every
 * ancestor struct's nulls folded in, since Spark's {@code GetStructField} is
 * null wherever the struct is. Children of the field share the same ancestors,
 * so a nested struct read through this view stays consistent. Owned by whoever
 * owns the wrapped vector; closing the view closes nothing.
 */
public final class NestedFieldColumnVector extends ColumnVector {

    private final ColumnVector inner;
    private final ColumnVector[] ancestors;
    private final int numRows;
    private int numNulls = -1;

    private NestedFieldColumnVector(ColumnVector inner, ColumnVector[] ancestors, int numRows) {
        super(inner.dataType());
        this.inner = inner;
        this.ancestors = ancestors;
        this.numRows = numRows;
    }

    /**
     * Walks {@code path} down the struct children of {@code root}; the
     * ancestors that carry nulls are kept for the null check.
     */
    public static ColumnVector of(ColumnVector root, int[] path, int numRows) {
        ColumnVector cv = root instanceof BorrowedColumnVector b ? b.inner() : root;
        java.util.ArrayList<ColumnVector> withNulls = new java.util.ArrayList<>();
        for (int step : path) {
            if (cv.hasNull()) {
                withNulls.add(cv);
            }
            cv = cv.getChild(step);
        }
        if (withNulls.isEmpty()) {
            return BorrowedColumnVector.of(cv);
        }
        return new NestedFieldColumnVector(cv, withNulls.toArray(new ColumnVector[0]), numRows);
    }

    private boolean ancestorNull(int rowId) {
        for (ColumnVector a : ancestors) {
            if (a.isNullAt(rowId)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void close() {}

    @Override
    public boolean hasNull() {
        return true;
    }

    @Override
    public int numNulls() {
        if (numNulls < 0) {
            int n = 0;
            for (int r = 0; r < numRows; r++) {
                if (isNullAt(r)) {
                    n++;
                }
            }
            numNulls = n;
        }
        return numNulls;
    }

    @Override
    public boolean isNullAt(int rowId) {
        return inner.isNullAt(rowId) || ancestorNull(rowId);
    }

    @Override
    public boolean getBoolean(int rowId) {
        return inner.getBoolean(rowId);
    }

    @Override
    public byte getByte(int rowId) {
        return inner.getByte(rowId);
    }

    @Override
    public short getShort(int rowId) {
        return inner.getShort(rowId);
    }

    @Override
    public int getInt(int rowId) {
        return inner.getInt(rowId);
    }

    @Override
    public long getLong(int rowId) {
        return inner.getLong(rowId);
    }

    @Override
    public float getFloat(int rowId) {
        return inner.getFloat(rowId);
    }

    @Override
    public double getDouble(int rowId) {
        return inner.getDouble(rowId);
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        return inner.getArray(rowId);
    }

    @Override
    public ColumnarMap getMap(int rowId) {
        return inner.getMap(rowId);
    }

    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        return inner.getDecimal(rowId, precision, scale);
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        return inner.getUTF8String(rowId);
    }

    @Override
    public byte[] getBinary(int rowId) {
        return inner.getBinary(rowId);
    }

    @Override
    public org.apache.spark.unsafe.types.CalendarInterval getInterval(int rowId) {
        return inner.getInterval(rowId);
    }

    /**
     * A field of this struct field: the same ancestors apply, plus this level
     * if it carries nulls.
     */
    @Override
    public ColumnVector getChild(int ordinal) {
        ColumnVector[] next = ancestors;
        if (inner.hasNull()) {
            next = java.util.Arrays.copyOf(ancestors, ancestors.length + 1);
            next[ancestors.length] = inner;
        }
        return new NestedFieldColumnVector(inner.getChild(ordinal), next, numRows);
    }
}
