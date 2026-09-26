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
package io.vecruntime.spark.adapter;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import io.vecruntime.kernels.VectorBuffers;
import io.vecruntime.spark.arrow.ArrowVectorBuffers;
import io.vecruntime.spark.arrow.BorrowedColumnVector;
import io.vecruntime.spark.arrow.VectorArrowColumnVector;
import io.vecruntime.spark.arrow.VectorDecimalColumnVector;
import io.vecruntime.spark.arrow.VectorDictionaryColumnVector;
import io.vecruntime.spark.arrow.VectorNarrowIntColumnVector;
import org.apache.spark.sql.vectorized.ColumnVector;

/**
 * Turns any Spark {@link ColumnVector} into {@link VectorBuffers}.
 *
 * <p>Zero-copy adapters are tried first (our own Arrow-backed vectors, plus any
 * registered adapter such as the Comet one, which is loaded reflectively so
 * this jar has no Comet dependency). Anything else is copied into the per-batch
 * {@code Arena}.
 */
public final class ColumnVectorAdapters {

    /**
     * Zero-copy adapter for a foreign columnar vector class. Returns {@code
     * null} to decline. The scratch arena lives as long as the batch is being
     * evaluated; adapters that need small side buffers (a validity bitmap
     * derived from a foreign nullability structure, a decoded dictionary)
     * allocate them there and still hand the column's own data buffers over
     * without a copy.
     */
    @FunctionalInterface
    public interface Adapter {
        VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch);
    }

    private static final List<Adapter> ADAPTERS = new CopyOnWriteArrayList<>();

    /**
     * Test-visible counters (executor side; local mode in the suites): columns
     * a registered adapter took zero-copy, and columns that fell through to the
     * copy.
     */
    private static final java.util.concurrent.atomic.LongAdder ADAPTED_COLUMNS = new java.util.concurrent.atomic.LongAdder();

    private static final java.util.concurrent.atomic.LongAdder COPIED_COLUMNS = new java.util.concurrent.atomic.LongAdder();

    public static long adaptedColumns() {
        return ADAPTED_COLUMNS.sum();
    }

    public static long copiedColumns() {
        return COPIED_COLUMNS.sum();
    }

    static {
        // Comet's and Iceberg's scan vectors are read zero-copy when their jars are on the classpath
        // (executor side); registration is a no-op otherwise.
        io.vecruntime.spark.comet.CometVectorAdapter.tryRegister();
        io.vecruntime.spark.iceberg.IcebergVectorAdapter.tryRegister();
    }

    private ColumnVectorAdapters() {}

    public static void register(Adapter adapter) {
        ADAPTERS.add(adapter);
    }

    public static VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch) {
        if (cv instanceof BorrowedColumnVector b) {
            return adapt(b.inner(), numRows, scratch);
        }
        if (cv instanceof VectorArrowColumnVector v) {
            return ArrowVectorBuffers.forRead(v.getValueVector());
        }
        if (cv instanceof VectorDictionaryColumnVector d) {
            return d.buffers();
        }
        if (cv instanceof VectorDecimalColumnVector d) {
            return ArrowVectorBuffers.forRead(d.vector());
        }
        if (cv instanceof VectorNarrowIntColumnVector n) {
            return ArrowVectorBuffers.forRead(n.vector()); // an INT32 lane under a TINYINT / SMALLINT type (#327)
        }
        for (Adapter adapter : ADAPTERS) {
            VectorBuffers vb = adapter.adapt(cv, numRows, scratch);
            if (vb != null) {
                ADAPTED_COLUMNS.increment();
                return vb;
            }
        }
        COPIED_COLUMNS.increment();
        return SparkColumnVectorBuffers.copy(cv, numRows, scratch);
    }

    /** True if {@link #adapt} would not need to copy. */
    public static boolean isZeroCopy(ColumnVector cv, int numRows) {
        if (cv instanceof BorrowedColumnVector b) {
            return isZeroCopy(b.inner(), numRows);
        }
        if (cv instanceof VectorArrowColumnVector
                || cv instanceof VectorDictionaryColumnVector
                || cv instanceof VectorDecimalColumnVector
                || cv instanceof VectorNarrowIntColumnVector) {
            return true;
        }
        try (Arena scratch = Arena.ofConfined()) {
            for (Adapter adapter : ADAPTERS) {
                if (adapter.adapt(cv, numRows, scratch) != null) {
                    return true;
                }
            }
        }
        return false;
    }
}
