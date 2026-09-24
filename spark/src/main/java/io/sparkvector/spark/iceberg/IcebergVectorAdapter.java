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
package io.sparkvector.spark.iceberg;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.concurrent.atomic.LongAdder;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.Decimal128;
import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.spark.adapter.ColumnVectorAdapters;
import io.sparkvector.spark.adapter.TypeMapping;
import io.sparkvector.spark.arrow.SelectedColumnarBatch;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Zero-copy adapter for the batches Iceberg's own JVM vectorized Parquet reader
 * produces (Spark's {@code BatchScanExec} over an Iceberg table when Comet's
 * native Iceberg scan is absent or falls back, e.g. Comet 1.0 on format version
 * 3 tables).
 *
 * <p>Two things differ from Comet's vectors. Iceberg does not maintain Arrow
 * validity buffers (it disables Arrow's null checking and tracks nulls in a
 * byte-per-row {@code NullabilityHolder}), so the validity bitmap is derived
 * from that array into the batch's scratch arena while the data and offset
 * buffers are wrapped in place. And on merge-on-read tables the reader applies
 * deletes by wrapping every column in a {@code ColumnVectorWithFilter} that
 * remaps row ids through a shared {@code int[] rowIdMapping} and reports the
 * live row count as the batch size. {@link #normalize} unwraps such a batch
 * into the delegate vectors plus a {@link SelectedColumnarBatch} whose
 * selection has one bit per live position, so the operators evaluate over the
 * physical batch with the selection they already understand, instead of paying
 * the indirection on every access.
 *
 * <p>Everything is resolved reflectively; the plugin has no compile-time
 * Iceberg dependency and the adapter registers only when the Iceberg runtime is
 * on the classpath. Any vector it does not recognise (constant partition
 * columns, dictionary-encoded non-string columns, oversized dictionaries) is
 * declined to the copying path, which reads through Iceberg's accessors and is
 * always correct.
 */
public final class IcebergVectorAdapter implements ColumnVectorAdapters.Adapter {

    private static final Logger LOG = LoggerFactory.getLogger(IcebergVectorAdapter.class);

    private static final String ARROW_COLUMN_VECTOR = "org.apache.iceberg.spark.data.vectorized.IcebergArrowColumnVector";
    private static final String COLUMN_VECTOR_WITH_FILTER = "org.apache.iceberg.spark.data.vectorized.ColumnVectorWithFilter";
    private static final String VECTOR_ACCESSOR = "org.apache.iceberg.arrow.vectorized.ArrowVectorAccessor";
    private static final String NULLABILITY_HOLDER = "org.apache.iceberg.arrow.vectorized.NullabilityHolder";
    private static final String SHADED_VALUE_VECTOR = "org.apache.iceberg.shaded.org.apache.arrow.vector.ValueVector";
    private static final String SHADED_ARROW_BUF = "org.apache.iceberg.shaded.org.apache.arrow.memory.ArrowBuf";
    private static final String SHADED_LARGE_VARCHAR = "org.apache.iceberg.shaded.org.apache.arrow.vector.LargeVarCharVector";
    private static final String DICTIONARY_DECIMAL_ACCESSOR = "org.apache.iceberg.arrow.vectorized.GenericArrowVectorAccessorFactory$DictionaryDecimalAccessor";
    private static final String DICTIONARY_STRING_ACCESSOR = "org.apache.iceberg.arrow.vectorized.GenericArrowVectorAccessorFactory$DictionaryStringAccessor";
    private static final String SHADED_PARQUET_DICTIONARY = "org.apache.iceberg.shaded.org.apache.parquet.column.Dictionary";
    private static final String SHADED_PARQUET_BINARY = "org.apache.iceberg.shaded.org.apache.parquet.io.api.Binary";

    private static volatile IcebergVectorAdapter instance;
    private static volatile Boolean registered;

    /** Test-visible counters (executor side; local mode in the suites). */
    private static final LongAdder NORMALIZED_BATCHES = new LongAdder();

    /**
     * Rows of the normalized batches: the positions read (physical) and the
     * ones the deletes left (live).
     */
    private static final LongAdder NORMALIZED_PHYSICAL_ROWS = new LongAdder();

    private static final LongAdder NORMALIZED_LIVE_ROWS = new LongAdder();
    private static final LongAdder ADAPTED_COLUMNS = new LongAdder();
    private static final LongAdder ADAPTED_DICTIONARY_COLUMNS = new LongAdder();

    /**
     * Columns whose 4-byte IntVector was widened into an INT64 lane (small
     * decimals).
     */
    private static final LongAdder WIDENED_INT_COLUMNS = new LongAdder();

    public static long widenedIntColumns() {
        return WIDENED_INT_COLUMNS.sum();
    }

    private final Class<?> arrowColumnVector;
    private final Class<?> columnVectorWithFilter;
    private final Class<?> largeVarChar;
    private final Method accessor; // IcebergArrowColumnVector.accessor() (protected)
    private final Method nullabilityHolder; // IcebergArrowColumnVector.nullabilityHolder() (protected)
    private final Method getVector; // ArrowVectorAccessor.getVector() -> shaded ValueVector
    private final Method hasNulls; // NullabilityHolder.hasNulls()
    private final Field isNull; // NullabilityHolder.isNull (byte[], non-zero = null)
    private final Field filterDelegate; // ColumnVectorWithFilter.delegate
    private final Field filterRowIdMapping; // ColumnVectorWithFilter.rowIdMapping
    private final Method getValueCount;
    private final Method getDataBuffer;
    private final Method getOffsetBuffer;
    private final Method memoryAddress;
    private final Method capacity;
    // Dictionary-encoded strings: resolved best effort, null when the Iceberg build differs.
    private final Class<?> dictionaryStringAccessor;
    private final Field accessorDictionary; // DictionaryStringAccessor.dictionary (shaded parquet)
    private final Field dictionaryParquetField; // DictionaryDecimalAccessor.parquetDictionary (shaded parquet), or null
    private final Method dictionaryMaxId;
    private final Method dictionaryDecodeToBinary;
    private final Method binaryGetBytes;

    private IcebergVectorAdapter(ClassLoader loader) throws ReflectiveOperationException {
        arrowColumnVector = Class.forName(ARROW_COLUMN_VECTOR, false, loader);
        columnVectorWithFilter = Class.forName(COLUMN_VECTOR_WITH_FILTER, false, loader);
        Class<?> vectorAccessor = Class.forName(VECTOR_ACCESSOR, false, loader);
        Class<?> holder = Class.forName(NULLABILITY_HOLDER, false, loader);
        Class<?> valueVector = Class.forName(SHADED_VALUE_VECTOR, false, loader);
        Class<?> arrowBuf = Class.forName(SHADED_ARROW_BUF, false, loader);
        largeVarChar = Class.forName(SHADED_LARGE_VARCHAR, false, loader);
        accessor = accessible(arrowColumnVector.getDeclaredMethod("accessor"));
        nullabilityHolder = accessible(arrowColumnVector.getDeclaredMethod("nullabilityHolder"));
        getVector = vectorAccessor.getMethod("getVector");
        hasNulls = holder.getMethod("hasNulls");
        isNull = accessible(holder.getDeclaredField("isNull"));
        filterDelegate = accessible(columnVectorWithFilter.getDeclaredField("delegate"));
        filterRowIdMapping = accessible(columnVectorWithFilter.getDeclaredField("rowIdMapping"));
        getValueCount = valueVector.getMethod("getValueCount");
        getDataBuffer = valueVector.getMethod("getDataBuffer");
        getOffsetBuffer = valueVector.getMethod("getOffsetBuffer");
        memoryAddress = arrowBuf.getMethod("memoryAddress");
        capacity = arrowBuf.getMethod("capacity");

        Class<?> dsa = null;
        Field dictField = null;
        Method maxId = null;
        Method decode = null;
        Method getBytes = null;
        try {
            dsa = Class.forName(DICTIONARY_STRING_ACCESSOR, false, loader);
            dictField = accessible(dsa.getDeclaredField("dictionary"));
            Class<?> dictionary = Class.forName(SHADED_PARQUET_DICTIONARY, false, loader);
            Class<?> binary = Class.forName(SHADED_PARQUET_BINARY, false, loader);
            maxId = dictionary.getMethod("getMaxId");
            decode = dictionary.getMethod("decodeToBinary", int.class);
            getBytes = binary.getMethod("getBytes");
        } catch (ReflectiveOperationException e) {
            LOG.debug("spark-vector: Iceberg dictionary accessor not resolvable, dictionary strings will be copied", e);
            dsa = null;
        }
        dictionaryStringAccessor = dsa;
        accessorDictionary = dictField;
        Field decimalDict = null;
        try {
            Class<?> dda = Class.forName(DICTIONARY_DECIMAL_ACCESSOR, false, loader);
            decimalDict = accessible(dda.getDeclaredField("parquetDictionary"));
        } catch (ReflectiveOperationException e) {
            LOG.debug("spark-vector: Iceberg decimal dictionary accessor not resolvable, dictionary wide decimals will be copied", e);
        }
        dictionaryParquetField = decimalDict;
        dictionaryMaxId = maxId;
        dictionaryDecodeToBinary = decode;
        binaryGetBytes = getBytes;
    }

    private static <T extends java.lang.reflect.AccessibleObject> T accessible(T member) {
        member.setAccessible(true);
        return member;
    }

    /**
     * Registers the adapter if the Iceberg Spark runtime is on the classpath.
     * Safe to call repeatedly; returns whether Iceberg batches are now handled
     * zero-copy.
     */
    public static synchronized boolean tryRegister() {
        if (registered != null) {
            return registered;
        }
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader == null) {
            loader = IcebergVectorAdapter.class.getClassLoader();
        }
        try {
            IcebergVectorAdapter adapter = new IcebergVectorAdapter(loader);
            ColumnVectorAdapters.register(adapter);
            instance = adapter;
            LOG.info("spark-vector: Iceberg vectors will be read zero-copy");
            registered = true;
        } catch (ClassNotFoundException e) {
            LOG.debug("spark-vector: Iceberg not on the classpath, adapter not registered");
            registered = false;
        } catch (ReflectiveOperationException | RuntimeException e) {
            LOG.warn("spark-vector: Iceberg found but its vectorized reader differs from the expected one; " + "Iceberg batches will be copied instead of read zero-copy", e);
            registered = false;
        }
        return registered;
    }

    public static boolean isRegistered() {
        return Boolean.TRUE.equals(registered);
    }

    public static long normalizedBatches() {
        return NORMALIZED_BATCHES.sum();
    }

    /**
     * Physical rows of every merge-on-read batch normalized so far (in this
     * JVM: the local runner sees its executors).
     */
    public static long normalizedPhysicalRows() {
        return NORMALIZED_PHYSICAL_ROWS.sum();
    }

    /**
     * Live rows of those batches: what the deletes left. {@code live /
     * physical} is the merge ratio.
     */
    public static long normalizedLiveRows() {
        return NORMALIZED_LIVE_ROWS.sum();
    }

    public static long adaptedColumns() {
        return ADAPTED_COLUMNS.sum();
    }

    public static long adaptedDictionaryColumns() {
        return ADAPTED_DICTIONARY_COLUMNS.sum();
    }

    /**
     * If {@code batch} is an Iceberg merge-on-read batch whose columns are
     * row-id-mapped, returns a {@link SelectedColumnarBatch} over the unwrapped
     * delegate vectors (physical row count, the live positions selected,
     * columns not owned); otherwise returns {@code batch} itself. The caller
     * closes the returned batch when it differs from the input (this releases
     * only the selection bitmap; Iceberg keeps owning the vectors).
     */
    public static ColumnarBatch normalize(ColumnarBatch batch) {
        IcebergVectorAdapter a = instance;
        if (a == null || batch.numCols() == 0 || !a.columnVectorWithFilter.isInstance(batch.column(0))) {
            return batch;
        }
        try {
            return a.unwrap(batch);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot read Iceberg row-id mapping", e);
        }
    }

    @SuppressWarnings("ReferenceEquality") // every column must share the one mapping array, by identity
    private ColumnarBatch unwrap(ColumnarBatch batch) throws ReflectiveOperationException {
        int numCols = batch.numCols();
        int live = batch.numRows();
        ColumnVector[] delegates = new ColumnVector[numCols];
        int[] mapping = null;
        for (int c = 0; c < numCols; c++) {
            ColumnVector cv = batch.column(c);
            if (!columnVectorWithFilter.isInstance(cv)) {
                return batch; // mixed batch we do not understand; the copy path handles the wrappers
            }
            int[] m = (int[]) filterRowIdMapping.get(cv);
            if (mapping == null) {
                mapping = m;
            } else if (m != mapping) {
                return batch; // every column is expected to share one mapping
            }
            delegates[c] = (ColumnVector) filterDelegate.get(cv);
        }
        if (mapping == null) {
            return batch;
        }
        // Iceberg allocates the mapping with the physical batch size and fills its first `live` slots.
        int physical = mapping.length;
        if (live > physical || (live > 0 && mapping[live - 1] >= physical)) {
            return batch;
        }
        NORMALIZED_BATCHES.increment();
        NORMALIZED_PHYSICAL_ROWS.add(physical);
        NORMALIZED_LIVE_ROWS.add(live);
        return SelectedColumnarBatch.ofIndices(delegates, physical, mapping, live, false);
    }

    @Override
    public VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch) {
        if (!arrowColumnVector.isInstance(cv)) {
            return null; // ColumnVectorWithFilter (not normalized), ConstantColumnVector, ... -> copy path
        }
        VecType type = TypeMapping.vecTypeOf(cv.dataType());
        if (type == null) {
            return null;
        }
        try {
            Object acc = accessor.invoke(cv);
            Object vector = getVector.invoke(acc);
            if ((Integer) getValueCount.invoke(vector) != numRows) {
                return null;
            }
            Object holder = nullabilityHolder.invoke(cv);
            MemorySegment validity = validityOf(holder, numRows, scratch);
            VectorBuffers result;
            if (dictionaryStringAccessor != null && dictionaryStringAccessor.isInstance(acc)) {
                result = adaptDictionaryUtf8(acc, vector, numRows, validity, scratch);
                if (result != null) {
                    ADAPTED_DICTIONARY_COLUMNS.increment();
                }
            } else if (type == VecType.DECIMAL128 && acc.getClass()
                    .getSimpleName()
                    .equals("DictionaryDecimalBinaryAccessor")) {
                result = adaptDictionaryWideDecimal(acc, vector, numRows, validity, scratch);
            } else if (acc.getClass()
                          .getSimpleName()
                          .startsWith("Dictionary")) {
                return null; // dictionary-encoded non-string column: Iceberg decodes it per row
            } else if (type == VecType.DECIMAL128) {
                result = adaptWideDecimal(vector, numRows, validity, scratch);
            } else {
                result = wrap(vector, numRows, type, validity, scratch);
            }
            if (result != null) {
                ADAPTED_COLUMNS.increment();
            }
            return result;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot read Iceberg vector", e);
        }
    }

    private MemorySegment validityOf(Object holder, int numRows, Arena scratch) throws ReflectiveOperationException {
        if (!(Boolean) hasNulls.invoke(holder)) {
            return null;
        }
        byte[] nulls = (byte[]) isNull.get(holder);
        if (nulls.length < numRows) {
            throw new IllegalStateException("Iceberg nullability holder shorter than the batch");
        }
        return ArrowLayout.validityFromNullBytes(scratch, nulls, numRows);
    }

    private VectorBuffers wrap(Object vector, int numRows, VecType type,
            MemorySegment validity, Arena scratch)
            throws ReflectiveOperationException {
        if (type == VecType.UTF8 && largeVarChar.isInstance(vector)) {
            return null; // 64-bit offsets: let the copy path handle it
        }
        MemorySegment data = segment(getDataBuffer.invoke(vector));
        String vectorClass = vector.getClass().getSimpleName();
        if (type == VecType.INT64 && vectorClass.equals("IntVector")) {
            // Iceberg keeps a decimal of up to 9 digits (e.g. TPC-DS DECIMAL(7,2)) as an IntVector of
            // 4-byte unscaled values, while the lane for every decimal up to 18 digits is INT64: widen
            // into the scratch arena. Wrapping the 4-byte buffer as 8-byte lanes read past its end.
            MemorySegment wide = scratch.allocate((long) numRows << 3, 8);
            io.sparkvector.kernels.CastKernels.widenInt32(data, numRows, wide);
            WIDENED_INT_COLUMNS.increment();
            return SegmentVectorBuffers.fixedWidth(type, numRows, validity, wide);
        }
        if (type == VecType.INT64 && !(vectorClass.equals("BigIntVector") || vectorClass.startsWith("TimeStamp"))) {
            return null; // not 8-byte values (e.g. a 16-byte DecimalVector): the copy path is always correct
        }
        if (type.isFixedWidth() && type != VecType.BOOL && data.byteSize() < (long) numRows * type.byteWidth()) {
            return null; // a buffer shorter than the rows it claims: decline rather than read past it
        }
        if (type == VecType.UTF8) {
            MemorySegment offsets = segment(getOffsetBuffer.invoke(vector));
            return SegmentVectorBuffers.utf8(numRows, validity, offsets, data);
        }
        if (type == VecType.DECIMAL128 && !vector.getClass()
                .getSimpleName()
                .equals("DecimalVector")) {
            return null; // converted by adaptWideDecimal (a FixedSizeBinaryVector of big-endian bytes)
        }
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
    }

    /**
     * Iceberg's reader keeps a decimal wider than 18 digits as a {@code
     * FixedSizeBinaryVector} of big-endian two's complement bytes, as many per
     * value as the precision needs (up to 16); each valid row becomes two
     * little-endian limbs in a scratch lane (#257).
     */
    private VectorBuffers adaptWideDecimal(Object vector, int numRows, MemorySegment validity,
            Arena scratch)
            throws ReflectiveOperationException {
        if (vector.getClass()
                  .getSimpleName()
                  .equals("DecimalVector")) {
            return wrap(vector, numRows, VecType.DECIMAL128, validity, scratch); // already Arrow Decimal128
        }
        if (!vector.getClass()
                   .getSimpleName()
                   .equals("FixedSizeBinaryVector")) {
            return null;
        }
        int width = (Integer) vector.getClass()
                .getMethod("getByteWidth")
                .invoke(vector);
        if (width <= 0 || width > Decimal128.WIDTH) {
            return null;
        }
        MemorySegment source = segment(getDataBuffer.invoke(vector));
        if (source.byteSize() < (long) numRows * width) {
            return null;
        }
        // Parquet stores a decimal in the fewest bytes its precision needs (decimal(27,2) in 12), so
        // each value is a big-endian string of `width` bytes, sign-extended into the limbs.
        MemorySegment data = ArrowLayout.allocateData(scratch, VecType.DECIMAL128, numRows);
        byte[] be = new byte[width];
        for (int i = 0; i < numRows; i++) {
            if (validity != null && !io.sparkvector.kernels.Bitmap.isSet(validity, i)) {
                continue;
            }
            MemorySegment.copy(source, java.lang.foreign.ValueLayout.JAVA_BYTE,
                    (long) i * width, be, 0, width);
            Decimal128.set(data, i, Decimal128.hiFromBigEndian(be, 0, width), Decimal128.loFromBigEndian(be, 0, width));
        }
        return SegmentVectorBuffers.fixedWidth(VecType.DECIMAL128, numRows, validity, data);
    }

    /**
     * Dictionary-encoded strings: the indices are the IntVector's data buffer
     * (wrapped in place), the Parquet dictionary is decoded once per batch into
     * the scratch arena. Declined when the dictionary is larger than the batch,
     * where decoding every entry would cost more than the copy path's per-row
     * decode.
     */
    private VectorBuffers adaptDictionaryUtf8(Object acc, Object indexVector, int numRows,
            MemorySegment validity, Arena scratch)
            throws ReflectiveOperationException {
        Object dictionary = accessorDictionary.get(acc);
        int size = (Integer) dictionaryMaxId.invoke(dictionary) + 1;
        if (size > numRows) {
            return null;
        }
        byte[][] values = new byte[size][];
        long total = 0;
        for (int i = 0; i < size; i++) {
            values[i] = (byte[]) binaryGetBytes.invoke(dictionaryDecodeToBinary.invoke(dictionary, i));
            total += values[i].length;
        }
        MemorySegment offsets = ArrowLayout.allocateOffsets(scratch, size);
        MemorySegment bytes = ArrowLayout.allocateBytes(scratch, total);
        int pos = 0;
        for (int i = 0; i < size; i++) {
            offsets.setAtIndex(VectorBuffers.LE_INT, i, pos);
            MemorySegment.copy(values[i], 0, bytes, java.lang.foreign.ValueLayout.JAVA_BYTE, pos, values[i].length);
            pos += values[i].length;
        }
        offsets.setAtIndex(VectorBuffers.LE_INT, size, pos);
        VectorBuffers dict = SegmentVectorBuffers.utf8(size, null, offsets, bytes);
        MemorySegment indices = segment(getDataBuffer.invoke(indexVector));
        return SegmentVectorBuffers.dictionaryUtf8(numRows, validity, indices, dict);
    }

    /**
     * Dictionary-encoded wide decimals ({@code
     * DictionaryDecimalBinaryAccessor}): the Parquet dictionary holds
     * big-endian byte strings and the Arrow vector the int32 indices. A 16-byte
     * dictionary lookup buys nothing downstream, so the column is decoded per
     * batch: the dictionary once into limbs, then one two-limb copy per row
     * (#257).
     */
    private VectorBuffers adaptDictionaryWideDecimal(Object acc, Object indexVector, int numRows,
            MemorySegment validity, Arena scratch)
            throws ReflectiveOperationException {
        if (dictionaryParquetField == null) {
            return null;
        }
        Object dictionary = dictionaryParquetField.get(acc);
        int size = (Integer) dictionaryMaxId.invoke(dictionary) + 1;
        long[] hi = new long[size];
        long[] lo = new long[size];
        for (int i = 0; i < size; i++) {
            byte[] be = (byte[]) binaryGetBytes.invoke(dictionaryDecodeToBinary.invoke(dictionary, i));
            if (be.length > Decimal128.WIDTH) {
                return null;
            }
            hi[i] = Decimal128.hiFromBigEndian(be, 0, be.length);
            lo[i] = Decimal128.loFromBigEndian(be, 0, be.length);
        }
        MemorySegment indices = segment(getDataBuffer.invoke(indexVector));
        if (indices.byteSize() < (long) numRows << 2) {
            return null;
        }
        MemorySegment data = ArrowLayout.allocateData(scratch, VecType.DECIMAL128, numRows);
        for (int i = 0; i < numRows; i++) {
            if (validity != null && !io.sparkvector.kernels.Bitmap.isSet(validity, i)) {
                continue;
            }
            int id = indices.getAtIndex(VectorBuffers.LE_INT, i);
            if (id < 0 || id >= size) {
                return null;
            }
            Decimal128.set(data, i, hi[id], lo[id]);
        }
        return SegmentVectorBuffers.fixedWidth(VecType.DECIMAL128, numRows, validity, data);
    }

    private MemorySegment segment(Object arrowBuf) throws ReflectiveOperationException {
        long address = (Long) memoryAddress.invoke(arrowBuf);
        long size = (Long) capacity.invoke(arrowBuf);
        if (size == 0) {
            return MemorySegment.NULL.reinterpret(0);
        }
        return MemorySegment.ofAddress(address).reinterpret(size);
    }
}
