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
package io.vecruntime.spark.arrow;

import org.apache.arrow.vector.IntVector;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A TINYINT or SMALLINT column over an INT32 lane (#327): the kernels compute
 * on 32-bit lanes and every value is within the declared type's range (the
 * compiler narrows casts and arithmetic), so Spark reads the declared type back
 * through {@link #getByte} / {@link #getShort} while the operators see an
 * ordinary {@link IntVector}. The counterpart of {@link
 * VectorDecimalColumnVector} for small integers.
 */
public final class VectorNarrowIntColumnVector extends ColumnVector {

    private final IntVector vector;
    private final boolean owns;

    public VectorNarrowIntColumnVector(IntVector vector, DataType type) {
        this(vector, type, true);
    }

    private VectorNarrowIntColumnVector(IntVector vector, DataType type, boolean owns) {
        super(type);
        if (!isNarrow(type)) {
            throw new IllegalArgumentException("not a narrow integer type: " + type);
        }
        this.vector = vector;
        this.owns = owns;
    }

    public static boolean isNarrow(DataType type) {
        return type instanceof ByteType || type instanceof ShortType;
    }

    public IntVector vector() {
        return vector;
    }

    public boolean ownsMemory() {
        return owns;
    }

    /** Same vector, not owned. */
    public VectorNarrowIntColumnVector borrow() {
        return new VectorNarrowIntColumnVector(vector, dataType(), false);
    }

    @Override
    public void close() {
        if (owns) {
            vector.close();
        }
    }

    @Override
    public boolean hasNull() {
        return vector.getNullCount() > 0;
    }

    @Override
    public int numNulls() {
        return vector.getNullCount();
    }

    @Override
    public boolean isNullAt(int rowId) {
        return vector.isNull(rowId);
    }

    @Override
    public byte getByte(int rowId) {
        return (byte) vector.get(rowId);
    }

    @Override
    public short getShort(int rowId) {
        return (short) vector.get(rowId);
    }

    @Override
    public int getInt(int rowId) {
        return vector.get(rowId);
    }

    @Override
    public long getLong(int rowId) {
        return vector.get(rowId);
    }

    @Override
    public boolean getBoolean(int rowId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float getFloat(int rowId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double getDouble(int rowId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ColumnarMap getMap(int ordinal) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        throw new UnsupportedOperationException();
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public byte[] getBinary(int rowId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ColumnVector getChild(int ordinal) {
        throw new UnsupportedOperationException();
    }
}
