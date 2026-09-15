package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.apache.spark.sql.execution.vectorized.Dictionary;
import org.apache.spark.sql.execution.vectorized.OffHeapColumnVector;
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector;
import org.apache.spark.sql.execution.vectorized.WritableColumnVector;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * Adapts a Spark {@link ColumnVector} (typically {@code OnHeapColumnVector} or {@code
 * OffHeapColumnVector} from the vectorized Parquet reader) to Arrow-layout segments.
 *
 * <p>Fixed-width data is copied once into native memory: straight out of the backing {@code
 * int[]}/{@code long[]}/{@code double[]} (on heap) or native buffer (off heap) of Spark's own
 * writable vectors, or through the bulk getters ({@code getInts} etc., which copy once more) for
 * foreign vector classes. Kernels read native segments noticeably faster than heap-array
 * segments on the Vector API, which is why the arrays are not wrapped in place. Numeric columns the
 * Parquet reader left dictionary encoded are decoded through a per-batch lookup table instead of
 * one virtual call per row. The validity bitmap is rebuilt only when the vector has nulls.
 *
 * <p>String columns that the Parquet reader left dictionary encoded are copied as dictionary
 * indices plus the distinct values actually referenced by the batch, so downstream kernels never
 * touch the row strings; other string columns are copied byte-wise without materialising a {@code
 * UTF8String} per row.
 */
public final class SparkColumnVectorBuffers {

  private static final Field DICTIONARY_FIELD = field(WritableColumnVector.class, "dictionary");
  private static final Field ONHEAP_NULLS = field(OnHeapColumnVector.class, "nulls");
  private static final Field ONHEAP_INTS = field(OnHeapColumnVector.class, "intData");
  private static final Field ONHEAP_LONGS = field(OnHeapColumnVector.class, "longData");
  private static final Field ONHEAP_DOUBLES = field(OnHeapColumnVector.class, "doubleData");
  private static final Field OFFHEAP_NULLS = field(OffHeapColumnVector.class, "nulls");
  private static final Field OFFHEAP_DATA = field(OffHeapColumnVector.class, "data");

  private SparkColumnVectorBuffers() {}

  public static VectorBuffers copy(ColumnVector cv, int numRows, Arena arena) {
    VecType type = TypeMapping.vecTypeOf(cv.dataType());
    if (type == null) {
      throw new UnsupportedOperationException("unsupported Spark type " + cv.dataType());
    }
    MemorySegment validity = copyValidity(cv, numRows, arena);
    if (type.isFixedWidth() && cv instanceof WritableColumnVector w) {
      Dictionary dict = w.hasDictionary() ? dictionaryOf(w) : null;
      MemorySegment source =
          dict != null ? decodeDictionary(w, dict, type, numRows, validity) : wrapData(w, type, numRows);
      if (source != null) {
        MemorySegment data = ArrowLayout.allocateData(arena, type, numRows);
        MemorySegment.copy(source, 0, data, 0, source.byteSize());
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
      }
    }
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
        if (cv instanceof WritableColumnVector w) {
          Dictionary dict = w.hasDictionary() ? dictionaryOf(w) : null;
          if (dict != null) {
            return copyDictionaryUtf8(w, dict, numRows, arena, validity);
          }
          return copyWritableUtf8(w, numRows, arena, validity);
        }
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
    byte[] nulls = cv instanceof OnHeapColumnVector ? (byte[]) get(ONHEAP_NULLS, cv) : null;
    if (nulls != null && nulls.length >= numRows) {
      for (int i = 0; i < numRows; i++) {
        if (nulls[i] != 0) {
          any = true;
        } else {
          Bitmap.set(validity, i);
        }
      }
      return any ? validity : null;
    }
    MemorySegment nativeNulls = cv instanceof OffHeapColumnVector ? nativeSegment(OFFHEAP_NULLS, cv, numRows) : null;
    for (int i = 0; i < numRows; i++) {
      boolean isNull = nativeNulls != null ? nativeNulls.get(ValueLayout.JAVA_BYTE, i) != 0 : cv.isNullAt(i);
      if (isNull) {
        any = true;
      } else {
        Bitmap.set(validity, i);
      }
    }
    // hasNull() may be conservative (e.g. nulls outside [0, numRows)); drop an all-valid bitmap.
    return any ? validity : null;
  }

  /**
   * View of a writable vector's fixed-width storage, or {@code null} when the vector class is not
   * one we know how to read directly.
   */
  private static MemorySegment wrapData(WritableColumnVector cv, VecType type, int numRows) {
    long bytes = (long) numRows * type.byteWidth();
    if (cv instanceof OnHeapColumnVector) {
      Object array =
          switch (type) {
            case INT32 -> get(ONHEAP_INTS, cv);
            case INT64 -> get(ONHEAP_LONGS, cv);
            case FLOAT64 -> get(ONHEAP_DOUBLES, cv);
            default -> null;
          };
      MemorySegment heap =
          switch (array) {
            case int[] a when a.length >= numRows -> MemorySegment.ofArray(a);
            case long[] a when a.length >= numRows -> MemorySegment.ofArray(a);
            case double[] a when a.length >= numRows -> MemorySegment.ofArray(a);
            case null, default -> null;
          };
      return heap == null ? null : heap.asSlice(0, bytes);
    }
    if (cv instanceof OffHeapColumnVector) {
      return nativeSegment(OFFHEAP_DATA, cv, bytes);
    }
    return null;
  }

  /**
   * Numeric column left dictionary encoded by the Parquet reader: decodes the dictionary once (ids
   * are dense, so every id up to the largest one referenced exists) and gathers, instead of
   * Spark's virtual {@code decodeToX} call per row.
   */
  private static MemorySegment decodeDictionary(
      WritableColumnVector cv, Dictionary dict, VecType type, int numRows, MemorySegment validity) {
    int[] ids = cv.getDictionaryIds().getInts(0, numRows);
    if (validity != null) {
      for (int i = 0; i < numRows; i++) {
        if (!Bitmap.isSet(validity, i)) {
          ids[i] = 0; // any valid id; the value is masked by the validity bitmap
        }
      }
    }
    int maxId = -1;
    for (int i = 0; i < numRows; i++) {
      maxId = Math.max(maxId, ids[i]);
    }
    switch (type) {
      case INT32 -> {
        int[] table = new int[maxId + 1];
        for (int id = 0; id <= maxId; id++) {
          table[id] = dict.decodeToInt(id);
        }
        int[] out = new int[numRows];
        for (int i = 0; i < numRows; i++) {
          out[i] = table[ids[i]];
        }
        return MemorySegment.ofArray(out);
      }
      case INT64 -> {
        long[] table = new long[maxId + 1];
        for (int id = 0; id <= maxId; id++) {
          table[id] = dict.decodeToLong(id);
        }
        long[] out = new long[numRows];
        for (int i = 0; i < numRows; i++) {
          out[i] = table[ids[i]];
        }
        return MemorySegment.ofArray(out);
      }
      case FLOAT64 -> {
        double[] table = new double[maxId + 1];
        for (int id = 0; id <= maxId; id++) {
          table[id] = dict.decodeToDouble(id);
        }
        double[] out = new double[numRows];
        for (int i = 0; i < numRows; i++) {
          out[i] = table[ids[i]];
        }
        return MemorySegment.ofArray(out);
      }
      default -> {
        return null;
      }
    }
  }

  /**
   * Dictionary-encoded strings: Spark keeps Parquet dictionary ids per row and decodes on access.
   * The ids are remapped to a dense dictionary holding only the values this batch references.
   */
  private static VectorBuffers copyDictionaryUtf8(
      WritableColumnVector cv, Dictionary dict, int numRows, Arena arena, MemorySegment validity) {
    int[] ids = cv.getDictionaryIds().getInts(0, numRows); // a fresh array; remapped in place
    int[] remap = new int[64];
    Arrays.fill(remap, -1);
    int distinct = 0;
    byte[] bytes = new byte[256];
    int used = 0;
    int[] offsets = new int[65];
    for (int i = 0; i < numRows; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) {
        ids[i] = 0;
        continue;
      }
      int id = ids[i];
      if (id >= remap.length) {
        int old = remap.length;
        remap = Arrays.copyOf(remap, Math.max(id + 1, old * 2));
        Arrays.fill(remap, old, remap.length, -1);
      }
      int k = remap[id];
      if (k < 0) {
        byte[] value = dict.decodeToBinary(id);
        if (used + value.length > bytes.length) {
          bytes = Arrays.copyOf(bytes, Math.max(bytes.length * 2, used + value.length));
        }
        System.arraycopy(value, 0, bytes, used, value.length);
        used += value.length;
        k = distinct++;
        if (distinct >= offsets.length) {
          offsets = Arrays.copyOf(offsets, offsets.length * 2);
        }
        offsets[distinct] = used;
        remap[id] = k;
      }
      ids[i] = k;
    }
    MemorySegment indices = ArrowLayout.allocateData(arena, VecType.INT32, numRows);
    MemorySegment.copy(ids, 0, indices, VectorBuffers.LE_INT, 0, numRows);
    MemorySegment dictOffsets = ArrowLayout.allocateOffsets(arena, distinct);
    MemorySegment.copy(offsets, 0, dictOffsets, VectorBuffers.LE_INT, 0, distinct + 1);
    MemorySegment dictData = ArrowLayout.allocateBytes(arena, used);
    MemorySegment.copy(bytes, 0, dictData, ValueLayout.JAVA_BYTE, 0, used);
    VectorBuffers dictionary = SegmentVectorBuffers.utf8(distinct, null, dictOffsets, dictData);
    return SegmentVectorBuffers.dictionaryUtf8(numRows, validity, indices, dictionary);
  }

  /** Plain strings in a writable vector: copied straight out of its byte storage. */
  private static VectorBuffers copyWritableUtf8(
      WritableColumnVector cv, int numRows, Arena arena, MemorySegment validity) {
    WritableColumnVector bytes = cv.arrayData();
    long total = 0;
    for (int i = 0; i < numRows; i++) {
      if (validity == null || Bitmap.isSet(validity, i)) {
        total += cv.getArrayLength(i);
      }
    }
    MemorySegment offsets = ArrowLayout.allocateOffsets(arena, numRows);
    MemorySegment data = ArrowLayout.allocateBytes(arena, total);
    int pos = 0;
    for (int i = 0; i < numRows; i++) {
      offsets.set(VectorBuffers.LE_INT, (long) i << 2, pos);
      if (validity == null || Bitmap.isSet(validity, i)) {
        int len = cv.getArrayLength(i);
        if (len > 0) {
          ByteBuffer bb = bytes.getByteBuffer(cv.getArrayOffset(i), len);
          if (bb.hasArray()) {
            MemorySegment.copy(
                bb.array(), bb.arrayOffset() + bb.position(), data, ValueLayout.JAVA_BYTE, pos, len);
          } else {
            MemorySegment.copy(MemorySegment.ofBuffer(bb), 0, data, pos, len);
          }
          pos += len;
        }
      }
    }
    offsets.set(VectorBuffers.LE_INT, (long) numRows << 2, pos);
    return SegmentVectorBuffers.utf8(numRows, validity, offsets, data);
  }

  /** Any other column vector: one {@link UTF8String} per row. */
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

  private static Field field(Class<?> cls, String name) {
    try {
      Field f = cls.getDeclaredField(name);
      f.setAccessible(true);
      return f;
    } catch (ReflectiveOperationException | RuntimeException e) {
      return null;
    }
  }

  private static Object get(Field f, Object target) {
    if (f == null) {
      return null;
    }
    try {
      return f.get(target);
    } catch (IllegalAccessException e) {
      return null;
    }
  }

  /** Native memory a Spark off-heap vector points to, as a segment of {@code bytes}. */
  private static MemorySegment nativeSegment(Field addressField, Object target, long bytes) {
    Object address = get(addressField, target);
    if (!(address instanceof Long addr) || addr == 0L) {
      return null;
    }
    return MemorySegment.ofAddress(addr).reinterpret(bytes);
  }

  private static Dictionary dictionaryOf(WritableColumnVector cv) {
    return (Dictionary) get(DICTIONARY_FIELD, cv);
  }
}
