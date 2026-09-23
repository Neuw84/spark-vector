package io.sparkvector.spark.comet;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.CompactKernels;
import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.spark.adapter.ColumnVectorAdapters;
import io.sparkvector.spark.adapter.TypeMapping;
import io.sparkvector.spark.arrow.SelectedColumnarBatch;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Turns one of our batches into a batch of Comet vectors, zero copy, so Comet's
 * native shuffle (and any Comet operator that consumes JVM batches) can take
 * it.
 *
 * <p>Comet shades Arrow into {@code org.apache.comet.shaded.arrow}, so its
 * vectors and ours share no class; what they do share is the process and the
 * Arrow C Data Interface. Each column is exported with {@link ArrowCData} (two
 * structs pointing at our buffers) and imported through Comet's own {@code
 * org.apache.arrow.c.ArrowImporter}, which wraps the same memory in a shaded
 * vector and calls our release callback when Comet is done with it. Comet is
 * reached reflectively, so this jar has no compile-time dependency on it.
 */
public final class CometBatchBridge {

    private final Object importer; // org.apache.arrow.c.ArrowImporter over Comet's root allocator
    private final Object dictionaryProvider; // org.apache.arrow.c.CDataDictionaryProvider
    private final Method wrapArray;
    private final Method wrapSchema;
    private final Method importVector;
    private final Method getVector; // CometVector.getVector(ValueVector, DictionaryProvider)

    /** Exports handed to Comet by this bridge, for {@link #releaseOutstanding}. */
    private final java.util.List<Long> exported = new java.util.ArrayList<>();

    private CometBatchBridge(ClassLoader loader) throws ReflectiveOperationException {
        Class<?> pkg = Class.forName("org.apache.comet.package$", true, loader);
        Object allocator = pkg.getMethod("CometArrowAllocator").invoke(pkg.getField("MODULE$")
                .get(null));
        Class<?> allocatorClass = Class.forName("org.apache.comet.shaded.arrow.memory.BufferAllocator", false, loader);
        Class<?> arrowArray = Class.forName("org.apache.arrow.c.ArrowArray", false, loader);
        Class<?> arrowSchema = Class.forName("org.apache.arrow.c.ArrowSchema", false, loader);
        Class<?> providerClass = Class.forName("org.apache.arrow.c.CDataDictionaryProvider", false, loader);
        Class<?> importerClass = Class.forName("org.apache.arrow.c.ArrowImporter", false, loader);
        importer = importerClass.getConstructor(allocatorClass).newInstance(allocator);
        dictionaryProvider = providerClass.getConstructor().newInstance();
        wrapArray = arrowArray.getMethod("wrap", long.class);
        wrapSchema = arrowSchema.getMethod("wrap", long.class);
        importVector = importerClass.getMethod("importVector", arrowArray, arrowSchema, providerClass);
        Class<?> valueVector = Class.forName("org.apache.comet.shaded.arrow.vector.ValueVector", false, loader);
        Class<?> dictProvider = Class.forName("org.apache.comet.shaded.arrow.vector.dictionary.DictionaryProvider", false, loader);
        getVector = Class.forName("org.apache.comet.vector.CometVector", false, loader).getMethod("getVector", valueVector, dictProvider);
    }

    /** A bridge if Comet's classes are loadable, else {@code null}. */
    public static CometBatchBridge tryCreate() {
        try {
            return new CometBatchBridge(CometBatchBridge.class.getClassLoader());
        } catch (ReflectiveOperationException | LinkageError e) {
            return null;
        }
    }

    public static boolean isCometLoaded() {
        try {
            Class.forName("org.apache.comet.vector.CometVector", false, CometBatchBridge.class.getClassLoader());
            return true;
        } catch (ClassNotFoundException | LinkageError e) {
            return false;
        }
    }

    /**
     * Whether {@link #convert} can carry a column of this type: any lane, the
     * DECIMAL128 one included -- a wide decimal crosses as its 16-byte Arrow
     * layout (#281); the bridge only moves columns, so the kernels' narrower
     * {@link TypeMapping#isSupported} does not apply.
     */
    public static boolean isSupported(DataType dt) {
        return TypeMapping.hasLane(dt);
    }

    /**
     * Converts {@code batch} (dense, or a {@link SelectedColumnarBatch} whose
     * selection is applied first) into a batch of Comet vectors over the same
     * memory. The returned batch owns its vectors; closing them releases our
     * buffers through the C Data release callback.
     */
    public ColumnarBatch convert(ColumnarBatch batch, String[] names, DataType[] types) {
        int n = batch.numRows();
        ColumnVector[] out = new ColumnVector[batch.numCols()];
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment selection = null;
            int count = n;
            if (batch instanceof SelectedColumnarBatch s) {
                selection = s.selection();
                count = s.selectedCount();
            }
            for (int c = 0; c < out.length; c++) {
                VectorBuffers in = ColumnVectorAdapters.adapt(batch.column(c), n, scratch);
                if (selection != null) {
                    in = compact(in, selection, count, scratch);
                }
                out[c] = toComet(in, names[c], types[c]);
            }
            return new ColumnarBatch(out, count);
        } catch (RuntimeException e) {
            for (ColumnVector v : out) {
                if (v != null) {
                    v.close();
                }
            }
            throw e;
        }
    }

    private ColumnVector toComet(VectorBuffers in, String name, DataType dt) {
        ArrowCData.Exported export = ArrowCData.export(in, name, dt);
        try {
            Object array = wrapArray.invoke(null, export.array());
            Object schema = wrapSchema.invoke(null, export.schema());
            Object fieldVector = importVector.invoke(importer, array, schema, dictionaryProvider);
            ColumnVector v = (ColumnVector) getVector.invoke(null, fieldVector, dictionaryProvider);
            exported.add(export.id());
            return v;
        } catch (InvocationTargetException e) {
            ArrowCData.abandon(export.id());
            throw new IllegalStateException("Comet import failed for column " + name, e.getCause());
        } catch (IllegalAccessException e) {
            ArrowCData.abandon(export.id());
            throw new IllegalStateException("Comet import failed for column " + name, e);
        }
    }

    /**
     * Releases the exports of this bridge that Comet has not released itself.
     * Called when the task completes, i.e. after every consumer in it is done:
     * Comet's range-partitioning sampler reads our batches through {@code
     * rowIterator()} and never closes the imported vectors, which would
     * otherwise pin the last batch of every task. Returns the number of exports
     * released.
     */
    public int releaseOutstanding() {
        int released = 0;
        for (long id : exported) {
            if (ArrowCData.abandon(id)) {
                released++;
            }
        }
        exported.clear();
        return released;
    }

    /** Applies a selection into scratch memory; the export then copies from there. */
    private static VectorBuffers compact(VectorBuffers in, MemorySegment selection, int count,
            Arena arena) {
        MemorySegment validity = in.hasNulls() ? ArrowLayout.allocateBitmap(arena, count) : null;
        if (in.type() == VecType.UTF8 && !in.isDictionaryEncoded()) {
            long bytes = CompactKernels.selectedUtf8Bytes(in, selection);
            MemorySegment offsets = ArrowLayout.allocateOffsets(arena, count);
            MemorySegment data = ArrowLayout.allocateBytes(arena, bytes);
            CompactKernels.compactUtf8(in, selection, count, offsets, data, validity);
            return SegmentVectorBuffers.utf8(count, validity, offsets, data);
        }
        VecType physical = in.isDictionaryEncoded() ? VecType.INT32 : in.type();
        MemorySegment data = physical == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, count) : ArrowLayout.allocateData(arena, physical, count);
        CompactKernels.compactFixed(in, selection, count, data, validity);
        if (in.isDictionaryEncoded()) {
            return SegmentVectorBuffers.dictionaryUtf8(count, validity, data, in.dictionary());
        }
        return SegmentVectorBuffers.fixedWidth(in.type(), count, validity, data);
    }
}
