package io.sparkvector.spark.comet;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.spark.adapter.ColumnVectorAdapters;
import io.sparkvector.spark.adapter.TypeMapping;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Zero-copy view of Comet's scan output ({@code
 * org.apache.comet.vector.CometVector}, backed by Arrow buffers that Comet's
 * native reader handed to the JVM through Arrow FFI).
 *
 * <p>Comet shades Arrow to {@code org.apache.comet.shaded.arrow}, so its
 * vectors are not our Arrow classes and this jar must not depend on Comet at
 * compile time. Everything is resolved reflectively once, on first use, and
 * only the buffer addresses are read afterwards. Comet's dictionary-encoded
 * strings become dictionary-encoded UTF8 buffers, so group-by keys hash the
 * dictionary rather than every row.
 */
public final class CometVectorAdapter implements ColumnVectorAdapters.Adapter {

    private static final Logger LOG = LoggerFactory.getLogger(CometVectorAdapter.class);
    private static final String COMET_VECTOR = "org.apache.comet.vector.CometVector";
    private static final String COMET_DICTIONARY_VECTOR = "org.apache.comet.vector.CometDictionaryVector";
    private static final String COMET_DICTIONARY = "org.apache.comet.vector.CometDictionary";
    private static final String SHADED_VALUE_VECTOR = "org.apache.comet.shaded.arrow.vector.ValueVector";
    private static final String SHADED_ARROW_BUF = "org.apache.comet.shaded.arrow.memory.ArrowBuf";
    private static final String SHADED_LARGE_VARCHAR = "org.apache.comet.shaded.arrow.vector.LargeVarCharVector";

    private static volatile Boolean registered;

    private final Class<?> cometVector;
    private final Class<?> cometDictionaryVector;
    private final Class<?> largeVarChar;
    private final Method getValueVector; // CometVector -> shaded ValueVector
    private final Field dictIndices; // CometDictionaryVector.indices (CometPlainVector)
    private final Field dictValues; // CometDictionaryVector.values (CometDictionary)
    private final Method dictionaryValueVector; // CometDictionary -> shaded ValueVector
    private final Method getValueCount;
    private final Method getNullCount;
    private final Method getValidityBuffer;
    private final Method getDataBuffer;
    private final Method getOffsetBuffer;
    private final Method memoryAddress;
    private final Method capacity;

    private CometVectorAdapter(ClassLoader loader) throws ReflectiveOperationException {
        cometVector = Class.forName(COMET_VECTOR, false, loader);
        cometDictionaryVector = Class.forName(COMET_DICTIONARY_VECTOR, false, loader);
        Class<?> cometDictionary = Class.forName(COMET_DICTIONARY, false, loader);
        Class<?> valueVector = Class.forName(SHADED_VALUE_VECTOR, false, loader);
        Class<?> arrowBuf = Class.forName(SHADED_ARROW_BUF, false, loader);
        largeVarChar = Class.forName(SHADED_LARGE_VARCHAR, false, loader);
        getValueVector = cometVector.getMethod("getValueVector");
        dictIndices = cometDictionaryVector.getField("indices");
        dictValues = cometDictionaryVector.getField("values");
        dictionaryValueVector = cometDictionary.getMethod("getValueVector");
        getValueCount = valueVector.getMethod("getValueCount");
        getNullCount = valueVector.getMethod("getNullCount");
        getValidityBuffer = valueVector.getMethod("getValidityBuffer");
        getDataBuffer = valueVector.getMethod("getDataBuffer");
        getOffsetBuffer = valueVector.getMethod("getOffsetBuffer");
        memoryAddress = arrowBuf.getMethod("memoryAddress");
        capacity = arrowBuf.getMethod("capacity");
    }

    /**
     * Registers the adapter if Comet is on the classpath. Safe to call
     * repeatedly; returns whether Comet vectors are now handled zero-copy.
     */
    public static synchronized boolean tryRegister() {
        if (registered != null) {
            return registered;
        }
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader == null) {
            loader = CometVectorAdapter.class.getClassLoader();
        }
        try {
            ColumnVectorAdapters.register(new CometVectorAdapter(loader));
            LOG.info("spark-vector: Comet vectors will be read zero-copy");
            registered = true;
        } catch (ClassNotFoundException e) {
            LOG.debug("spark-vector: Comet not on the classpath, adapter not registered");
            registered = false;
        } catch (ReflectiveOperationException e) {
            LOG.warn("spark-vector: Comet found but its vector API differs from the expected one; " + "Comet batches will be copied instead of read zero-copy", e);
            registered = false;
        }
        return registered;
    }

    public static boolean isRegistered() {
        return Boolean.TRUE.equals(registered);
    }

    @Override
    public VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch) {
        if (!cometVector.isInstance(cv)) {
            return null;
        }
        VecType type = TypeMapping.vecTypeOf(cv.dataType());
        if (type == null) {
            return null;
        }
        try {
            if (cometDictionaryVector.isInstance(cv)) {
                return adaptDictionary(cv, numRows, type);
            }
            Object vector = getValueVector.invoke(cv);
            if (cv.dataType() instanceof org.apache.spark.sql.types.DecimalType) {
                String vectorClass = vector.getClass().getName();
                boolean narrowLongs = type == VecType.INT64 && vectorClass.endsWith(".BigIntVector");
                // Comet's 128-bit decimals are Arrow Decimal128 (16 little-endian bytes per value): the
                // DECIMAL128 lane's own layout, wrapped in place (#257). 32-bit decimals (p <= 9 in an
                // IntVector) still take the copy path, which widens them to the INT64 lane via getDecimal.
                boolean wide = type == VecType.DECIMAL128 && vectorClass.endsWith(".DecimalVector");
                if (!narrowLongs && !wide) {
                    return null;
                }
            }
            return wrap(vector, numRows, type);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot read Comet vector", e);
        }
    }

    private VectorBuffers adaptDictionary(ColumnVector cv, int numRows, VecType type) throws ReflectiveOperationException {
        if (type != VecType.UTF8) {
            // Comet only dictionary-encodes binary-like types; anything else is decoded by Comet itself.
            return null;
        }
        Object indices = dictIndices.get(cv); // CometPlainVector
        Object dictionary = dictValues.get(cv); // CometDictionary
        Object indexVector = getValueVector.invoke(indices);
        Object dictVector = dictionaryValueVector.invoke(dictionary);
        if (largeVarChar.isInstance(dictVector)) {
            return null;
        }
        VectorBuffers dict = wrap(dictVector, (Integer) getValueCount.invoke(dictVector), VecType.UTF8);
        MemorySegment validity = validityOf(indexVector);
        MemorySegment data = segment(getDataBuffer.invoke(indexVector));
        return SegmentVectorBuffers.dictionaryUtf8(numRows, validity, data, dict);
    }

    private VectorBuffers wrap(Object vector, int numRows, VecType type) throws ReflectiveOperationException {
        if (type == VecType.UTF8 && largeVarChar.isInstance(vector)) {
            return null; // 64-bit offsets: let the copy path handle it
        }
        MemorySegment validity = validityOf(vector);
        MemorySegment data = segment(getDataBuffer.invoke(vector));
        if (type == VecType.UTF8) {
            MemorySegment offsets = segment(getOffsetBuffer.invoke(vector));
            return SegmentVectorBuffers.utf8(numRows, validity, offsets, data);
        }
        return SegmentVectorBuffers.fixedWidth(type, numRows, validity, data);
    }

    private MemorySegment validityOf(Object vector) throws ReflectiveOperationException {
        int nulls = (Integer) getNullCount.invoke(vector);
        if (nulls == 0) {
            return null;
        }
        return segment(getValidityBuffer.invoke(vector));
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
