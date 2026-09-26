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
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import org.apache.arrow.vector.BaseFixedWidthVector;
import org.apache.arrow.vector.BaseVariableWidthVector;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.VarCharVector;

/**
 * {@link VectorBuffers} over an unshaded Arrow Java {@link ValueVector}, zero
 * copy.
 *
 * <p>Two modes: {@link #forRead} drops the validity segment when the vector has
 * no nulls so kernels take their no-null fast path, while {@link #forWrite}
 * always exposes a validity segment so kernels can write validity bits directly
 * into a freshly allocated vector.
 */
public final class ArrowVectorBuffers implements VectorBuffers {

    private final ValueVector vector;
    private final VecType type;
    private final int length;
    private final MemorySegment validity;
    private final MemorySegment data;
    private final MemorySegment offsets;
    private org.apache.spark.sql.types.DataType sparkType;

    private ArrowVectorBuffers(ValueVector vector, VecType type, int length,
            MemorySegment validity, MemorySegment data, MemorySegment offsets) {
        this.vector = vector;
        this.type = type;
        this.length = length;
        this.validity = validity;
        this.data = data;
        this.offsets = offsets;
    }

    /**
     * Maps a supported Arrow vector class to its physical type, or {@code null}
     * if unsupported.
     */
    public static VecType vecTypeOf(ValueVector v) {
        if (v instanceof IntVector || v instanceof DateDayVector) {
            return VecType.INT32;
        }
        if (v instanceof BigIntVector || v instanceof TimeStampVector) {
            return VecType.INT64;
        }
        if (v instanceof Float8Vector) {
            return VecType.FLOAT64;
        }
        if (v instanceof BitVector) {
            return VecType.BOOL;
        }
        if (v instanceof org.apache.arrow.vector.DecimalVector) {
            // Arrow Decimal128: 16-byte little-endian values, our DECIMAL128 layout as is.
            return VecType.DECIMAL128;
        }
        if (v instanceof VarCharVector) {
            return VecType.UTF8;
        }
        return null;
    }

    public static boolean isSupported(ValueVector v) {
        return vecTypeOf(v) != null;
    }

    /** Read-only view over the vector's current {@code valueCount} elements. */
    public static ArrowVectorBuffers forRead(ValueVector v) {
        VecType type = requireSupported(v);
        int n = v.getValueCount();
        MemorySegment validity = v.getNullCount() == 0 ? null : ArrowSegments.of(v.getValidityBuffer());
        return wrap(v, type, n, validity);
    }

    /**
     * Writable view over a vector that has already been sized with {@code
     * allocateNew(length)} (or {@code allocateNew(bytes, length)} for variable
     * width). The caller writes values and validity bits through the segments
     * and then calls {@code vector.setValueCount(length)}.
     */
    public static ArrowVectorBuffers forWrite(ValueVector v, int length) {
        VecType type = requireSupported(v);
        if (v.getValidityBuffer().capacity() < Bitmap.bytesFor(length)) {
            throw new IllegalStateException("vector not allocated for " + length + " elements");
        }
        return wrap(v, type, length, ArrowSegments.of(v.getValidityBuffer()));
    }

    /**
     * {@link #forWrite(ValueVector, int)} remembering the Spark type the vector
     * stands for, which matters where the Arrow class alone is ambiguous (a
     * decimal is a {@link BigIntVector}).
     */
    public static ArrowVectorBuffers forWrite(ValueVector v, int length, org.apache.spark.sql.types.DataType sparkType) {
        ArrowVectorBuffers b = forWrite(v, length);
        b.sparkType = sparkType;
        return b;
    }

    /** The Spark type given at allocation, or {@code null}. */
    public org.apache.spark.sql.types.DataType sparkType() {
        return sparkType;
    }

    private static ArrowVectorBuffers wrap(ValueVector v, VecType type, int n,
            MemorySegment validity) {
        MemorySegment data = ArrowSegments.of(v.getDataBuffer());
        MemorySegment offsets = null;
        if (v instanceof BaseVariableWidthVector) {
            offsets = ArrowSegments.of(v.getOffsetBuffer());
        } else if (!(v instanceof BaseFixedWidthVector)) {
            throw new IllegalArgumentException("unexpected vector class " + v.getClass());
        }
        return new ArrowVectorBuffers(v, type, n, validity, data, offsets);
    }

    private static VecType requireSupported(ValueVector v) {
        VecType t = vecTypeOf(v);
        if (t == null) {
            throw new IllegalArgumentException("unsupported Arrow vector " + v.getClass().getName());
        }
        return t;
    }

    public ValueVector vector() {
        return vector;
    }

    @Override
    public VecType type() {
        return type;
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public MemorySegment validity() {
        return validity;
    }

    @Override
    public MemorySegment data() {
        return data;
    }

    @Override
    public MemorySegment offsets() {
        return offsets;
    }

    @Override
    public VectorBuffers dictionary() {
        return null;
    }
}
