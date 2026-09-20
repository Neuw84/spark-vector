package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.Decimal128;
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
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.DecimalType;
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

  /** Test-visible counter: off-heap fixed-width columns handed over as views instead of copies. */
  private static final java.util.concurrent.atomic.LongAdder WRAPPED_OFFHEAP_COLUMNS = new java.util.concurrent.atomic.LongAdder();

  public static long wrappedOffHeapColumns() {
    return WRAPPED_OFFHEAP_COLUMNS.sum();
  }

  /** Test-visible counter: wide decimal columns converted into DECIMAL128 lanes. */
  private static final java.util.concurrent.atomic.LongAdder WIDE_DECIMAL_COLUMNS = new java.util.concurrent.atomic.LongAdder();

  public static long wideDecimalColumns() {
    return WIDE_DECIMAL_COLUMNS.sum();
  }

  public static VectorBuffers copy(ColumnVector cv, int numRows, Arena arena) {
    VecType type = TypeMapping.vecTypeOf(cv.dataType());
    if (type == null) {
      throw new UnsupportedOperationException("unsupported Spark type " + cv.dataType());
    }
    MemorySegment validity = copyValidity(cv, numRows, arena);
    if (cv.dataType() instanceof DecimalType dec) {
      return copyDecimal(cv, dec, numRows, arena, validity);
    }
    if (cv.dataType() instanceof ByteType || cv.dataType() instanceof ShortType) {
      // TINYINT / SMALLINT widen into the INT32 lane (#327); the per-row getters decode a Parquet
      // dictionary, which the bulk ones do not, and the int fast paths below would read the wrong array.
      MemorySegment data = ArrowLayout.allocateData(arena, VecType.INT32, numRows);
      boolean bytes = cv.dataType() instanceof ByteType;
      for (int i = 0; i < numRows; i++) {
        if (validity == null || Bitmap.isSet(validity, i)) {
          data.setAtIndex(VectorBuffers.LE_INT, i, bytes ? cv.getByte(i) : cv.getShort(i));
        }
      }
      return SegmentVectorBuffers.fixedWidth(VecType.INT32, numRows, validity, data);
    }
    if (type.isFixedWidth() && cv instanceof WritableColumnVector w) {
      Dictionary dict = w.hasDictionary() ? dictionaryOf(w) : null;
      if (dict == null && cv instanceof OffHeapColumnVector) {
        // Off-heap Parquet batches (spark.sql.columnVector.offheap.enabled): the data already sits in
        // native memory in Arrow's fixed-width layout, so the lane is a view of it, not a copy. The
        // batch outlives every read of it (operators drain a batch before pulling the next one), as
        // with the Comet and Iceberg zero-copy adapters.
        MemorySegment view = wrapData(w, type, numRows);
        if (view != null) {
          WRAPPED_OFFHEAP_COLUMNS.increment();
          return SegmentVectorBuffers.fixedWidth(type, numRows, validity, view);
        }
      }
      if (dict != null) {
        MemorySegment data = ArrowLayout.allocateData(arena, type, numRows);
        if (decodeDictionaryInto(w, dict, type, numRows, validity, data)) {
          return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
        }
      } else {
        MemorySegment source = wrapData(w, type, numRows);
        if (source != null) {
          MemorySegment data = ArrowLayout.allocateData(arena, type, numRows);
          MemorySegment.copy(source, 0, data, 0, source.byteSize());
          return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
        }
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

  /**
   * Decimals of up to 18 digits: Spark's writable vectors keep them as ints (precision <= 9) or
   * longs, the Parquet reader possibly dictionary encoded; both bulk getters decode the dictionary.
   * Foreign vectors (Comet's 128-bit decimals, for instance) are read through {@code getDecimal}.
   */
  private static VectorBuffers copyDecimal(
      ColumnVector cv, DecimalType dec, int numRows, Arena arena, MemorySegment validity) {
    if (dec.precision() > TypeMapping.MAX_DECIMAL_PRECISION) {
      return copyWideDecimal(cv, dec, numRows, arena, validity);
    }
    MemorySegment data = ArrowLayout.allocateData(arena, VecType.INT64, numRows);
    if (cv instanceof WritableColumnVector) {
      if (dec.precision() <= Decimal.MAX_INT_DIGITS()) {
        int[] ints = cv.getInts(0, numRows);
        for (int i = 0; i < numRows; i++) {
          data.setAtIndex(VectorBuffers.LE_LONG, i, ints[i]);
        }
      } else {
        MemorySegment.copy(cv.getLongs(0, numRows), 0, data, VectorBuffers.LE_LONG, 0, numRows);
      }
    } else {
      for (int i = 0; i < numRows; i++) {
        if (validity == null || Bitmap.isSet(validity, i)) {
          Decimal d = cv.getDecimal(i, dec.precision(), dec.scale());
          data.setAtIndex(VectorBuffers.LE_LONG, i, d == null ? 0L : d.toUnscaledLong());
        }
      }
    }
    return SegmentVectorBuffers.fixedWidth(VecType.INT64, numRows, validity, data);
  }

  /**
   * Decimals of 19 to 38 digits become a DECIMAL128 lane. Spark's writable vectors keep them as
   * big-endian two's complement byte strings ({@code getBinary}, which also resolves the Parquet
   * reader's dictionary), so each row is two sign-extended limbs without a {@code BigDecimal} on
   * the way; a foreign vector is read through {@code getDecimal}.
   */
  private static VectorBuffers copyWideDecimal(
      ColumnVector cv, DecimalType dec, int numRows, Arena arena, MemorySegment validity) {
    MemorySegment data = ArrowLayout.allocateData(arena, VecType.DECIMAL128, numRows);
    WIDE_DECIMAL_COLUMNS.increment();
    boolean writable = cv instanceof WritableColumnVector;
    for (int i = 0; i < numRows; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) {
        continue;
      }
      if (writable) {
        byte[] be = cv.getBinary(i);
        Decimal128.set(data, i, Decimal128.hiFromBigEndian(be, 0, be.length), Decimal128.loFromBigEndian(be, 0, be.length));
      } else {
        Decimal d = cv.getDecimal(i, dec.precision(), dec.scale());
        if (d != null) {
          java.math.BigInteger unscaled = d.toJavaBigDecimal().unscaledValue();
          Decimal128.set(data, i, Decimal128.hiOf(unscaled), Decimal128.loOf(unscaled));
        }
      }
    }
    return SegmentVectorBuffers.fixedWidth(VecType.DECIMAL128, numRows, validity, data);
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

  /** Per-thread scratch for the decoded dictionary table: no allocation per column per batch. */
  private static final ThreadLocal<long[]> DICTIONARY_TABLE = ThreadLocal.withInitial(() -> new long[256]);

  /**
   * Numeric column left dictionary encoded by the Parquet reader: decodes the dictionary once (ids
   * are dense, so every id up to the largest one referenced exists) and gathers straight into the
   * arena segment. Nothing is allocated on the heap: the earlier form built three arrays per column
   * per batch (ids, table, values) and was the GC behind the shuffle's short-query regressions.
   */
  private static boolean decodeDictionaryInto(
      WritableColumnVector cv, Dictionary dict, VecType type, int numRows, MemorySegment validity,
      MemorySegment data) {
    if (type != VecType.INT32 && type != VecType.INT64 && type != VecType.FLOAT64) {
      return false;
    }
    WritableColumnVector ids = cv.getDictionaryIds();
    int maxId = -1;
    for (int i = 0; i < numRows; i++) {
      if (validity == null || Bitmap.isSet(validity, i)) {
        maxId = Math.max(maxId, ids.getInt(i));
      }
    }
    long[] table = DICTIONARY_TABLE.get();
    if (table.length <= maxId) {
      table = new long[Integer.highestOneBit(maxId) << 1];
      DICTIONARY_TABLE.set(table);
    }
    switch (type) {
      case INT32 -> {
        for (int id = 0; id <= maxId; id++) {
          table[id] = dict.decodeToInt(id);
        }
        for (int i = 0; i < numRows; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            data.setAtIndex(VectorBuffers.LE_INT, i, (int) table[ids.getInt(i)]);
          }
        }
      }
      case INT64 -> {
        for (int id = 0; id <= maxId; id++) {
          table[id] = dict.decodeToLong(id);
        }
        for (int i = 0; i < numRows; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            data.setAtIndex(VectorBuffers.LE_LONG, i, table[ids.getInt(i)]);
          }
        }
      }
      default -> {
        for (int id = 0; id <= maxId; id++) {
          table[id] = Double.doubleToRawLongBits(dict.decodeToDouble(id));
        }
        for (int i = 0; i < numRows; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            data.setAtIndex(VectorBuffers.LE_LONG, i, table[ids.getInt(i)]);
          }
        }
      }
    }
    return true;
  }

  /**
   * Dictionary-encoded strings: Spark keeps Parquet dictionary ids per row and decodes on access.
   * The ids are remapped to a dense dictionary holding only the values this batch references.
   */
  private static VectorBuffers copyDictionaryUtf8(
      WritableColumnVector cv, Dictionary dict, int numRows, Arena arena, MemorySegment validity) {
    // Per-thread scratch, grown and kept: the four fresh arrays per column per batch this used to
    // allocate were GC on every dictionary string column adapted (the shuffle adapts every batch).
    Utf8Scratch sc = UTF8_SCRATCH.get();
    WritableColumnVector idVector = cv.getDictionaryIds();
    int[] remap = sc.remap;
    Arrays.fill(remap, -1);
    int distinct = 0;
    byte[] bytes = sc.bytes;
    int used = 0;
    int[] offsets = sc.offsets;
    MemorySegment indices = ArrowLayout.allocateData(arena, VecType.INT32, numRows);
    for (int i = 0; i < numRows; i++) {
      if (validity != null && !Bitmap.isSet(validity, i)) {
        indices.setAtIndex(VectorBuffers.LE_INT, i, 0);
        continue;
      }
      int id = idVector.getInt(i);
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
      indices.setAtIndex(VectorBuffers.LE_INT, i, k);
    }
    sc.remap = remap;
    sc.bytes = bytes;
    sc.offsets = offsets;
    MemorySegment dictOffsets = ArrowLayout.allocateOffsets(arena, distinct);
    MemorySegment.copy(offsets, 0, dictOffsets, VectorBuffers.LE_INT, 0, distinct + 1);
    MemorySegment dictData = ArrowLayout.allocateBytes(arena, used);
    MemorySegment.copy(bytes, 0, dictData, ValueLayout.JAVA_BYTE, 0, used);
    VectorBuffers dictionary = SegmentVectorBuffers.utf8(distinct, null, dictOffsets, dictData);
    return SegmentVectorBuffers.dictionaryUtf8(numRows, validity, indices, dictionary);
  }

  private static final class Utf8Scratch {
    int[] remap = new int[1024];
    byte[] bytes = new byte[1 << 16];
    int[] offsets = new int[1025];
  }

  private static final ThreadLocal<Utf8Scratch> UTF8_SCRATCH = ThreadLocal.withInitial(Utf8Scratch::new);

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
