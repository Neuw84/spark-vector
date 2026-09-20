package io.sparkvector.spark.arrow;

import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A column forwarded from a child batch without taking ownership: every read delegates to the
 * wrapped vector, {@link #close()} does nothing (the producer releases the vector when it moves on
 * to its next batch). Used by projections that pass input columns through when the child's vectors
 * are not ours (Spark's scan vectors, Comet's); {@code ColumnVectorAdapters} unwraps it.
 */
public final class BorrowedColumnVector extends ColumnVector {

  private final ColumnVector inner;

  private BorrowedColumnVector(ColumnVector inner) {
    super(inner.dataType());
    this.inner = inner;
  }

  /** {@code cv} without ownership; our own vector classes have their own borrow methods. */
  public static ColumnVector of(ColumnVector cv) {
    if (cv instanceof VectorArrowColumnVector v) {
      return v.borrow();
    }
    if (cv instanceof VectorDictionaryColumnVector d) {
      return d.borrow();
    }
    if (cv instanceof VectorDecimalColumnVector d) {
      return d.borrow();
    }
    if (cv instanceof VectorNarrowIntColumnVector n) {
      return n.borrow();
    }
    if (cv instanceof BorrowedColumnVector b) {
      return b;
    }
    return new BorrowedColumnVector(cv);
  }

  public ColumnVector inner() {
    return inner;
  }

  @Override
  public void close() {}

  @Override
  public boolean hasNull() {
    return inner.hasNull();
  }

  @Override
  public int numNulls() {
    return inner.numNulls();
  }

  @Override
  public boolean isNullAt(int rowId) {
    return inner.isNullAt(rowId);
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
  public ColumnarMap getMap(int ordinal) {
    return inner.getMap(ordinal);
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
  public ColumnVector getChild(int ordinal) {
    return inner.getChild(ordinal);
  }
}
