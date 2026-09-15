package io.sparkvector.spark.arrow;

import org.apache.arrow.vector.ValueVector;
import org.apache.spark.sql.vectorized.ArrowColumnVector;

/**
 * Spark's {@link ArrowColumnVector} hides the wrapped {@link ValueVector}. Our operators emit this
 * subclass so downstream spark-vector operators can get back to the Arrow buffers without a copy.
 */
public final class VectorArrowColumnVector extends ArrowColumnVector {

  private final ValueVector valueVector;

  public VectorArrowColumnVector(ValueVector vector) {
    super(vector);
    this.valueVector = vector;
  }

  public ValueVector getValueVector() {
    return valueVector;
  }
}
