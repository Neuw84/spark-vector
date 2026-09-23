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
