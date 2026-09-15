package io.sparkvector.spark.arrow;

import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.VectorBuffers;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A dictionary-encoded string column: an Arrow {@link IntVector} of indices plus a {@link
 * VarCharVector} dictionary. Spark has no dictionary column vectors of its own, so this class
 * decodes on {@link #getUTF8String} for row-based consumers, while spark-vector operators downstream
 * read the indices and dictionary zero copy (see {@code ColumnVectorAdapters}).
 *
 * <p>Keeping strings dictionary encoded through a filter lets the compaction move int32 indices
 * instead of variable-width bytes, and lets a following aggregate hash and compare dictionary
 * entries rather than every row.
 */
public final class VectorDictionaryColumnVector extends ColumnVector {

  private final IntVector indices;
  private final VarCharVector dictionary;
  private final boolean owns;

  public VectorDictionaryColumnVector(IntVector indices, VarCharVector dictionary) {
    this(indices, dictionary, true);
  }

  private VectorDictionaryColumnVector(IntVector indices, VarCharVector dictionary, boolean owns) {
    super(DataTypes.StringType);
    this.indices = indices;
    this.dictionary = dictionary;
    this.owns = owns;
  }

  public IntVector indices() {
    return indices;
  }

  public VarCharVector dictionary() {
    return dictionary;
  }

  public boolean ownsMemory() {
    return owns;
  }

  /** Same vectors, not owned. */
  public VectorDictionaryColumnVector borrow() {
    return new VectorDictionaryColumnVector(indices, dictionary, false);
  }

  /** Zero-copy dictionary-encoded UTF8 buffers over the two vectors. */
  public VectorBuffers buffers() {
    ArrowVectorBuffers idx = ArrowVectorBuffers.forRead(indices);
    return SegmentVectorBuffers.dictionaryUtf8(
        idx.length(), idx.validity(), idx.data(), ArrowVectorBuffers.forRead(dictionary));
  }

  @Override
  public void close() {
    if (owns) {
      indices.close();
      dictionary.close();
    }
  }

  @Override
  public boolean hasNull() {
    return indices.getNullCount() > 0;
  }

  @Override
  public int numNulls() {
    return indices.getNullCount();
  }

  @Override
  public boolean isNullAt(int rowId) {
    return indices.isNull(rowId);
  }

  @Override
  public UTF8String getUTF8String(int rowId) {
    if (indices.isNull(rowId)) {
      return null;
    }
    int k = indices.get(rowId);
    ArrowBuf offsets = dictionary.getOffsetBuffer();
    int start = offsets.getInt((long) k << 2);
    int end = offsets.getInt((long) (k + 1) << 2);
    return UTF8String.fromAddress(null, dictionary.getDataBuffer().memoryAddress() + start, end - start);
  }

  @Override
  public byte[] getBinary(int rowId) {
    UTF8String s = getUTF8String(rowId);
    return s == null ? null : s.getBytes();
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
  public long getLong(int rowId) {
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
  public Decimal getDecimal(int rowId, int precision, int scale) {
    throw unsupported();
  }

  @Override
  public ColumnVector getChild(int ordinal) {
    throw unsupported();
  }

  private static UnsupportedOperationException unsupported() {
    return new UnsupportedOperationException("dictionary-encoded string column");
  }
}
