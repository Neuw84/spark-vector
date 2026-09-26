/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package io.vecruntime.spark.comet;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import io.vecruntime.kernels.ArrowLayout;
import io.vecruntime.kernels.Bitmap;
import io.vecruntime.kernels.VecType;
import io.vecruntime.kernels.VectorBuffers;
import io.vecruntime.spark.arrow.ArrowVectorBuffers;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampType;

/**
 * Producer side of the <a
 * href="https://arrow.apache.org/docs/format/CDataInterface.html">Arrow C Data
 * Interface</a>, written with the FFM API instead of {@code arrow-c-data}.
 *
 * <p>The interface is two C structs ({@code ArrowSchema}, {@code ArrowArray})
 * whose fields point at the buffers, plus a release callback the consumer
 * invokes when it is done. Filling the structs is a few dozen stores; the
 * callback is an FFM upcall stub. That is enough for another Arrow
 * implementation in the same process (Comet's shaded one) to wrap our buffers
 * without copying, and it needs neither {@code arrow-c-data} nor its JNI
 * library on our side, which matters because Comet ships that library's classes
 * under their original names with shaded signatures.
 *
 * <p>Lifetime: an export retains the Arrow buffers it points at until the
 * consumer's release call; structs, strings and any copied buffers live in a
 * GC-managed arena that becomes unreachable at the same moment. Releases may
 * arrive on any thread, including native ones, so the registry is concurrent.
 */
public final class ArrowCData {

    /**
     * ArrowArray struct: length, null_count, offset, n_buffers, n_children,
     * buffers, children, dictionary, release, private_data.
     */
    static final long ARRAY_SIZE = 80;

    /**
     * ArrowSchema struct: format, name, metadata, flags, n_children, children,
     * dictionary, release, private_data.
     */
    static final long SCHEMA_SIZE = 72;

    static final long ARROW_FLAG_NULLABLE = 2;

    private static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG;

    private static final MemorySegment RELEASE_ARRAY;
    private static final MemorySegment RELEASE_SCHEMA;
    private static final ConcurrentHashMap<Long, Export> LIVE = new ConcurrentHashMap<>();
    private static final AtomicLong IDS = new AtomicLong(1);

    static {
        try {
            Linker linker = Linker.nativeLinker();
            MethodHandles.Lookup lookup = MethodHandles.lookup();
            RELEASE_ARRAY = linker.upcallStub(lookup.findStatic(ArrowCData.class, "releaseArray", MethodType.methodType(void.class, MemorySegment.class)),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS), Arena.global());
            RELEASE_SCHEMA = linker.upcallStub(lookup.findStatic(ArrowCData.class, "releaseSchema", MethodType.methodType(void.class, MemorySegment.class)),
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS), Arena.global());
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private ArrowCData() {}

    /**
     * One exported column: its arena (structs, strings, pointer arrays, any
     * copied buffers) and retained Arrow buffers.
     */
    private static final class Export {
        final Arena arena = Arena.ofAuto(); // freed by GC once released; no close handshake per column
        final List<ArrowBuf> retained = new ArrayList<>(3);

        void close() {
            for (ArrowBuf b : retained) {
                b.getReferenceManager().release();
            }
            retained.clear();
        }
    }

    /** Addresses of a filled ArrowArray and ArrowSchema for one column. */
    public record Exported(long array, long schema, long id) {}

    /**
     * Exports {@code in} as the C Data structs for a column named {@code name}
     * of Spark type {@code dt}. Buffers of our own Arrow vectors are pointed at
     * directly (and retained); anything else is copied into the export's arena.
     * Dictionary-encoded strings are decoded (the consumer would decode them
     * anyway).
     */
    public static Exported export(VectorBuffers in, String name, DataType dt) {
        Export export = new Export();
        long id = IDS.getAndIncrement();
        try {
            Arena arena = export.arena;
            int n = in.length();
            long nullCount = in.hasNulls() ? n - Bitmap.popcount(in.validity(), n) : 0;
            MemorySegment validity = null;
            MemorySegment data;
            MemorySegment offsets = null;
            if (dt instanceof DecimalType && in.type() == VecType.DECIMAL128) {
                // A wide decimal (19 to 38 digits) is already a DECIMAL128 lane: 16 little-endian bytes per
                // value, Arrow's own Decimal128 layout (#257). Point at an Arrow vector's buffers, copy any
                // other lane as is -- widening it word by word read every value as two rows (#281).
                if (in instanceof ArrowVectorBuffers a) {
                    FieldVector v = (FieldVector) a.vector();
                    for (ArrowBuf b : v.getBuffers(false)) {
                        b.getReferenceManager().retain();
                        export.retained.add(b);
                    }
                    if (nullCount > 0) {
                        validity = a.validity();
                    }
                    data = a.data();
                } else {
                    data = arena.allocate(Math.max((long) n << 4, 16), 16);
                    MemorySegment.copy(in.data(), 0, data, 0,
                            (long) n << 4);
                    if (nullCount > 0) {
                        validity = copyBitmap(arena, in.validity(), n);
                    }
                }
            } else if (dt instanceof DecimalType) {
                // Narrow decimals are 64-bit lanes; the C Data format "d:p,s" is 128-bit two's complement,
                // so widen into the export's arena (sign-extended high word).
                data = arena.allocate(Math.max((long) n << 4, 16), 16);
                for (int i = 0; i < n; i++) {
                    long v = in.getLong(i);
                    data.set(I64, (long) i << 4, v);
                    data.set(I64, ((long) i << 4) + 8, v >> 63);
                }
                if (nullCount > 0) {
                    validity = copyBitmap(arena, in.validity(), n);
                }
            } else if (in instanceof ArrowVectorBuffers a) {
                FieldVector v = (FieldVector) a.vector();
                for (ArrowBuf b : v.getBuffers(false)) {
                    b.getReferenceManager().retain();
                    export.retained.add(b);
                }
                if (nullCount > 0) {
                    validity = a.validity();
                }
                data = a.data();
                offsets = a.offsets();
            } else if (in.type() == VecType.UTF8 && in.isDictionaryEncoded()) {
                VectorBuffers dict = in.dictionary();
                long bytes = 0;
                for (int i = 0; i < n; i++) {
                    if (!in.isNull(i)) {
                        bytes += utf8Length(dict, in.getInt(i));
                    }
                }
                offsets = ArrowLayout.allocateOffsets(arena, n);
                data = ArrowLayout.allocateBytes(arena, bytes);
                int pos = 0;
                for (int i = 0; i < n; i++) {
                    offsets.set(VectorBuffers.LE_INT, (long) i << 2, pos);
                    if (!in.isNull(i)) {
                        int k = in.getInt(i);
                        int start = dict.offsets().get(VectorBuffers.LE_INT, (long) k << 2);
                        int len = utf8Length(dict, k);
                        MemorySegment.copy(dict.data(), ValueLayout.JAVA_BYTE, start, data, ValueLayout.JAVA_BYTE,
                                pos, len);
                        pos += len;
                    }
                }
                offsets.set(VectorBuffers.LE_INT, (long) n << 2, pos);
                if (nullCount > 0) {
                    validity = copyBitmap(arena, in.validity(), n);
                }
            } else {
                if (nullCount > 0) {
                    validity = copyBitmap(arena, in.validity(), n);
                }
                if (in.type() == VecType.UTF8) {
                    int end = in.offsets().get(VectorBuffers.LE_INT, (long) n << 2);
                    offsets = arena.allocate(((long) n + 1) << 2, 8);
                    MemorySegment.copy(in.offsets(), 0, offsets, 0,
                            ((long) n + 1) << 2);
                    data = arena.allocate(Math.max(end, 1), 8);
                    MemorySegment.copy(in.data(), 0, data, 0, end);
                } else if (in.type() == VecType.BOOL) {
                    data = copyBitmap(arena, in.data(), n);
                } else {
                    long bytes = (long) n * in.type().byteWidth();
                    data = arena.allocate(Math.max(bytes, 1), 8);
                    MemorySegment.copy(in.data(), 0, data, 0, bytes);
                }
            }

            // ArrowArray
            int nBuffers = offsets != null ? 3 : 2;
            MemorySegment buffers = arena.allocate(I64, nBuffers);
            buffers.setAtIndex(I64, 0,
                    validity == null ? 0L : validity.address());
            if (offsets != null) {
                buffers.setAtIndex(I64, 1, offsets.address());
                buffers.setAtIndex(I64, 2, data.address());
            } else {
                buffers.setAtIndex(I64, 1, data.address());
            }
            MemorySegment array = arena.allocate(ARRAY_SIZE, 8);
            array.set(I64, 0, n); // length
            array.set(I64, 8, nullCount);
            array.set(I64, 16, 0L); // offset
            array.set(I64, 24, nBuffers);
            array.set(I64, 32, 0L); // n_children
            array.set(I64, 40, buffers.address());
            array.set(I64, 48, 0L); // children
            array.set(I64, 56, 0L); // dictionary
            array.set(I64, 64, RELEASE_ARRAY.address());
            array.set(I64, 72, id); // private_data carries our registry key

            // ArrowSchema
            MemorySegment format = arena.allocateFrom(formatOf(dt), StandardCharsets.UTF_8);
            MemorySegment cname = arena.allocateFrom(name, StandardCharsets.UTF_8);
            MemorySegment schema = arena.allocate(SCHEMA_SIZE, 8);
            schema.set(I64, 0, format.address());
            schema.set(I64, 8, cname.address());
            schema.set(I64, 16, 0L); // metadata
            schema.set(I64, 24, ARROW_FLAG_NULLABLE);
            schema.set(I64, 32, 0L); // n_children
            schema.set(I64, 40, 0L); // children
            schema.set(I64, 48, 0L); // dictionary
            schema.set(I64, 56, RELEASE_SCHEMA.address());
            schema.set(I64, 64, id);

            LIVE.put(id, export);
            return new Exported(array.address(), schema.address(), id);
        } catch (RuntimeException e) {
            export.close();
            throw e;
        }
    }

    /**
     * Frees an export the consumer will not release (an import failed, or the
     * consumer never closes what it imported). Returns whether the export was
     * still live.
     */
    public static boolean abandon(long id) {
        Export e = LIVE.remove(id);
        if (e != null) {
            e.close();
            return true;
        }
        return false;
    }

    /** Exports still owned by consumers; for tests and leak checks. */
    public static int liveExports() {
        return LIVE.size();
    }

    /** C Data format string for the Spark type. */
    static String formatOf(DataType dt) {
        if (dt instanceof IntegerType) {
            return "i";
        }
        if (dt instanceof DateType) {
            return "tdD";
        }
        if (dt instanceof LongType) {
            return "l";
        }
        if (dt instanceof TimestampType) {
            return "tsu:UTC";
        }
        if (dt instanceof DoubleType) {
            return "g";
        }
        if (dt instanceof BooleanType) {
            return "b";
        }
        if (dt instanceof StringType) {
            return "u";
        }
        if (dt instanceof DecimalType d) {
            return "d:" + d.precision() + "," + d.scale();
        }
        throw new UnsupportedOperationException("no C Data format for " + dt);
    }

    private static int utf8Length(VectorBuffers dict, int k) {
        MemorySegment off = dict.offsets();
        return off.get(VectorBuffers.LE_INT, (long) (k + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) k << 2);
    }

    private static MemorySegment copyBitmap(Arena arena, MemorySegment bits, int n) {
        MemorySegment copy = ArrowLayout.allocateBitmap(arena, n);
        io.vecruntime.kernels.BitmapKernels.copy(bits, copy, n);
        return copy;
    }

    // ------------------------------------------------------------------ release callbacks

    /**
     * {@code void release(struct ArrowArray*)}: called by the consumer,
     * possibly from a native thread.
     */
    static void releaseArray(MemorySegment array) {
        try {
            MemorySegment s = array.reinterpret(ARRAY_SIZE);
            long id = s.get(I64, 72);
            s.set(I64, 64, 0L); // mark released, as the spec requires
            Export e = LIVE.remove(id);
            if (e != null) {
                e.close();
            }
        } catch (Throwable t) {
            // An exception escaping an upcall would terminate the JVM.
        }
    }

    /**
     * {@code void release(struct ArrowSchema*)}: the schema owns nothing beyond
     * the export's arena.
     */
    static void releaseSchema(MemorySegment schema) {
        try {
            schema.reinterpret(SCHEMA_SIZE).set(I64, 56, 0L);
        } catch (Throwable t) {
            // see above
        }
    }
}
