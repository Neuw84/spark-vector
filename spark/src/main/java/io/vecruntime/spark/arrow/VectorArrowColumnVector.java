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

import org.apache.arrow.vector.ValueVector;
import org.apache.spark.sql.vectorized.ArrowColumnVector;

/**
 * Spark's {@link ArrowColumnVector} hides the wrapped {@link ValueVector}. Our
 * operators emit this subclass so downstream spark-vector operators can get
 * back to the Arrow buffers without a copy.
 *
 * <p>A column may be <em>borrowed</em>: a projection that forwards an input
 * column unchanged wraps the child's vector without taking ownership, so
 * closing the output batch leaves the child's memory alone (the child releases
 * it when it produces its next batch).
 */
public final class VectorArrowColumnVector extends ArrowColumnVector {

    private final ValueVector valueVector;
    private final boolean owns;

    public VectorArrowColumnVector(ValueVector vector) {
        this(vector, true);
    }

    public VectorArrowColumnVector(ValueVector vector, boolean owns) {
        super(vector);
        this.valueVector = vector;
        this.owns = owns;
    }

    @Override
    public ValueVector getValueVector() {
        return valueVector;
    }

    public boolean ownsMemory() {
        return owns;
    }

    /** Same vector, not owned. */
    public VectorArrowColumnVector borrow() {
        return new VectorArrowColumnVector(valueVector, false);
    }

    @Override
    public void close() {
        if (owns) {
            super.close();
        }
    }
}
