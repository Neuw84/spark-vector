package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * A column in Arrow memory layout, exposed as raw {@link MemorySegment}s so kernels can use
 * {@code Vector.fromMemorySegment} without caring where the memory came from (Arrow Java, Comet,
 * a copy of a Spark ColumnVector, or an {@code Arena} allocation).
 *
 * <p>Layout by {@link VecType}:
 *
 * <ul>
 *   <li>fixed width (INT32, INT64, FLOAT64): {@code data()} holds {@code length * byteWidth} bytes
 *       in little-endian order.
 *   <li>BOOL: {@code data()} is a bitmap with one bit per element (Arrow LSB order).
 *   <li>UTF8: {@code offsets()} holds {@code length + 1} little-endian int32 values; the bytes of
 *       element {@code i} are {@code data()[offsets[i], offsets[i+1])}. When {@link #dictionary()}
 *       is non-null the column is dictionary encoded: {@code data()} holds {@code length} int32
 *       indices into the dictionary, which is itself a UTF8 {@code VectorBuffers}.
 * </ul>
 *
 * <p>{@link #validity()} is {@code null} when the column has no nulls. Implementations must
 * guarantee that segments are at least as large as the layout requires; they may be larger
 * (Arrow buffers are padded) and any padding is ignored.
 */
public interface VectorBuffers {

  ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  ValueLayout.OfDouble LE_DOUBLE =
      ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  VecType type();

  int length();

  /** Validity bitmap, or {@code null} if every element is valid. */
  MemorySegment validity();

  MemorySegment data();

  /** Offsets buffer for UTF8; {@code null} otherwise. */
  MemorySegment offsets();

  /** Dictionary for dictionary-encoded UTF8; {@code null} otherwise. */
  VectorBuffers dictionary();

  default boolean hasNulls() {
    return validity() != null;
  }

  default boolean isDictionaryEncoded() {
    return dictionary() != null;
  }

  default int nullCount() {
    MemorySegment v = validity();
    return v == null ? 0 : length() - Bitmap.popcount(v, length());
  }

  default boolean isNull(int i) {
    MemorySegment v = validity();
    return v != null && !Bitmap.isSet(v, i);
  }

  // Element accessors. These are convenience methods for tests, adapters and scalar fallbacks;
  // kernels should read whole lanes from the segments instead.

  default int getInt(int i) {
    return data().get(LE_INT, (long) i << 2);
  }

  default long getLong(int i) {
    return data().get(LE_LONG, (long) i << 3);
  }

  default double getDouble(int i) {
    return data().get(LE_DOUBLE, (long) i << 3);
  }

  default boolean getBoolean(int i) {
    return Bitmap.isSet(data(), i);
  }

  /** Copies out the UTF-8 bytes of element {@code i}, resolving dictionary encoding. */
  default byte[] getUtf8Bytes(int i) {
    VectorBuffers dict = dictionary();
    if (dict != null) {
      return dict.getUtf8Bytes(getInt(i));
    }
    MemorySegment off = offsets();
    int start = off.get(LE_INT, (long) i << 2);
    int end = off.get(LE_INT, (long) (i + 1) << 2);
    return data().asSlice(start, end - start).toArray(ValueLayout.JAVA_BYTE);
  }

  default String getString(int i) {
    return new String(getUtf8Bytes(i), StandardCharsets.UTF_8);
  }
}
