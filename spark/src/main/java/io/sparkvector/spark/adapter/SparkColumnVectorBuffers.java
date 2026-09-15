package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * Copies a Spark {@link ColumnVector} (typically {@code OnHeapColumnVector} or {@code
 * OffHeapColumnVector} from the vectorized Parquet reader) into Arrow-layout segments.
 *
 * <p>This is the fallback input path when the scan does not already produce Arrow memory. Bulk
 * getters ({@code getInts} etc.) are used for the data buffer; the validity bitmap is built from
 * {@code isNullAt} only when {@code hasNull()} reports nulls.
 */
public final class SparkColumnVectorBuffers {

  private SparkColumnVectorBuffers() {}

  public static VectorBuffers copy(ColumnVector cv, int numRows, Arena arena) {
    VecType type = TypeMapping.vecTypeOf(cv.dataType());
    if (type == null) {
      throw new UnsupportedOperationException("unsupported Spark type " + cv.dataType());
    }
    MemorySegment validity = copyValidity(cv, numRows, arena);
    switch (type) {
      case INT32 -> {
        MemorySegment data = ArrowLayout.allocateData(arena, type, numRows);
        MemorySegment.copy(cv.getInts(0, numRows), 0, data, VectorBuffers.LE_INT, 0, numRows);
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
      }
      case INT64 -> {
        MemorySegment data = ArrowLayout.allocateData(arena, type, numRows);
        MemorySegment.copy(cv.getLongs(0, numRows), 0, data, VectorBuffers.LE_LONG, 0, numRows);
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
      }
      case FLOAT64 -> {
        MemorySegment data = ArrowLayout.allocateData(arena, type, numRows);
        MemorySegment.copy(
            cv.getDoubles(0, numRows), 0, data, VectorBuffers.LE_DOUBLE, 0, numRows);
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
      }
      case BOOL -> {
        MemorySegment data = ArrowLayout.allocateBitmap(arena, numRows);
        boolean[] values = cv.getBooleans(0, numRows);
        for (int i = 0; i < numRows; i++) {
          if (values[i]) {
            Bitmap.set(data, i);
          }
        }
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
      }
      case UTF8 -> {
        return copyUtf8(cv, numRows, arena, validity);
      }
      default -> throw new IllegalStateException(type.toString());
    }
  }

  private static MemorySegment copyValidity(ColumnVector cv, int numRows, Arena arena) {
    if (!cv.hasNull()) {
      return null;
    }
    MemorySegment validity = ArrowLayout.allocateBitmap(arena, numRows);
    boolean any = false;
    for (int i = 0; i < numRows; i++) {
      if (cv.isNullAt(i)) {
        any = true;
      } else {
        Bitmap.set(validity, i);
      }
    }
    // hasNull() may be conservative (e.g. nulls outside [0, numRows)); drop an all-valid bitmap.
    return any ? validity : null;
  }

  private static VectorBuffers copyUtf8(
      ColumnVector cv, int numRows, Arena arena, MemorySegment validity) {
    UTF8String[] strings = new UTF8String[numRows];
    long total = 0;
    for (int i = 0; i < numRows; i++) {
      if (validity == null || Bitmap.isSet(validity, i)) {
        strings[i] = cv.getUTF8String(i);
        total += strings[i].numBytes();
      }
    }
    MemorySegment offsets = ArrowLayout.allocateOffsets(arena, numRows);
    MemorySegment data = ArrowLayout.allocateBytes(arena, total);
    int pos = 0;
    for (int i = 0; i < numRows; i++) {
      offsets.set(VectorBuffers.LE_INT, (long) i << 2, pos);
      UTF8String s = strings[i];
      if (s != null) {
        byte[] bytes = s.getBytes();
        MemorySegment.copy(bytes, 0, data, ValueLayout.JAVA_BYTE, pos, bytes.length);
        pos += bytes.length;
      }
    }
    offsets.set(VectorBuffers.LE_INT, (long) numRows << 2, pos);
    return SegmentVectorBuffers.utf8(numRows, validity, offsets, data);
  }
}
