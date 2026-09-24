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

import org.apache.arrow.vector.BigIntVector;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A decimal column of at most 18 digits over an Arrow {@link BigIntVector} of
 * unscaled values.
 *
 * <p>Arrow Java's own {@code DecimalVector} is 128 bits wide and Spark's {@code
 * ArrowColumnVector} reads it through {@code BigDecimal}; the kernels hold
 * small decimals as 64-bit lanes exactly like Spark's own {@code
 * WritableColumnVector} does for {@code Decimal(p <= 18)}. This class gives
 * Spark's row conversion {@link #getDecimal} over those lanes while
 * spark-vector operators downstream read the {@link BigIntVector} zero copy
 * (see {@code ColumnVectorAdapters}).
 */
public final class VectorDecimalColumnVector extends ColumnVector {

    private final BigIntVector vector;
    private final int precision;
    private final int scale;
    private final boolean owns;

    public VectorDecimalColumnVector(BigIntVector vector, DecimalType type) {
        this(vector, type, true);
    }

    private VectorDecimalColumnVector(BigIntVector vector, DecimalType type, boolean owns) {
        super(type);
        this.vector = vector;
        this.precision = type.precision();
        this.scale = type.scale();
        this.owns = owns;
    }

    public BigIntVector vector() {
        return vector;
    }

    public boolean ownsMemory() {
        return owns;
    }

    /** Same vector, not owned. */
    public VectorDecimalColumnVector borrow() {
        return new VectorDecimalColumnVector(vector, (DecimalType) dataType(), false);
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
    public Decimal getDecimal(int rowId, int precision, int scale) {
        if (vector.isNull(rowId)) {
            return null;
        }
        return Decimal.createUnsafe(vector.get(rowId), this.precision, this.scale);
    }

    /** The unscaled value, for consumers that know the scale. */
    @Override
    public long getLong(int rowId) {
        return vector.get(rowId);
    }

    @Override
    public boolean getBoolean(int rowId) {
        throw unsupported();
    }

    @Override
    public byte getByte(int rowId) {
        throw unsupported();
    }

    @Override
    public short getShort(int rowId) {
        throw unsupported();
    }

    @Override
    public int getInt(int rowId) {
        throw unsupported();
    }

    @Override
    public float getFloat(int rowId) {
        throw unsupported();
    }

    @Override
    public double getDouble(int rowId) {
        throw unsupported();
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        throw unsupported();
    }

    @Override
    public ColumnarMap getMap(int ordinal) {
        throw unsupported();
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        throw unsupported();
    }

    @Override
    public byte[] getBinary(int rowId) {
        throw unsupported();
    }

    @Override
    public ColumnVector getChild(int ordinal) {
        throw unsupported();
    }

    private static UnsupportedOperationException unsupported() {
        return new UnsupportedOperationException("decimal column");
    }
}
