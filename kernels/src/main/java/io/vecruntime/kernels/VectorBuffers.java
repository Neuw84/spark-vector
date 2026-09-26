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
package io.vecruntime.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * A column in Arrow memory layout, exposed as raw {@link MemorySegment}s so
 * kernels can use {@code Vector.fromMemorySegment} without caring where the
 * memory came from (Arrow Java, Comet, a copy of a Spark ColumnVector, or an
 * {@code Arena} allocation).
 *
 * <p>Layout by {@link VecType}:
 *
 * <ul>
 *   <li>fixed width (INT32, INT64, FLOAT64): {@code data()} holds {@code length
 *       * byteWidth} bytes in little-endian order.
 *   <li>BOOL: {@code data()} is a bitmap with one bit per element (Arrow LSB
 *       order).
 *   <li>UTF8: {@code offsets()} holds {@code length + 1} little-endian int32
 *       values; the bytes of element {@code i} are {@code data()[offsets[i],
 *       offsets[i+1])}. When {@link #dictionary()} is non-null the column is
 *       dictionary encoded: {@code data()} holds {@code length} int32 indices
 *       into the dictionary, which is itself a UTF8 {@code VectorBuffers}.
 * </ul>
 *
 * <p>{@link #validity()} is {@code null} when the column has no nulls.
 * Implementations must guarantee that segments are at least as large as the
 * layout requires; they may be larger (Arrow buffers are padded) and any
 * padding is ignored.
 */
public interface VectorBuffers {

    ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    ValueLayout.OfDouble LE_DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

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

    /** The 128-bit unscaled value of element {@code i} of a DECIMAL128 lane. */
    default java.math.BigInteger getDecimal128(int i) {
        MemorySegment d = data();
        return Decimal128.toBigInteger(Decimal128.hi(d, i), Decimal128.lo(d, i));
    }

    /**
     * Copies out the UTF-8 bytes of element {@code i}, resolving dictionary
     * encoding.
     */
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

    /**
     * Rows {@code [from, to)} as a column over the same memory, nothing copied:
     * the fixed-width data and the bitmaps are sliced, a plain UTF8 column keeps
     * its data buffer whole and slices the offsets (they stay absolute into it,
     * which every kernel reads them as), a dictionary-encoded one slices its
     * indices. {@code from} must be a multiple of 64 so the bitmaps slice on a
     * byte, and the slice is valid as long as this column is.
     */
    default VectorBuffers slice(int from, int to) {
        if (from < 0 || to < from || to > length()) {
            throw new IllegalArgumentException("slice ["
                    + from
                    + ", "
                    + to
                    + ") of "
                    + length()
                    + " rows");
        }
        if ((from & 63) != 0) {
            throw new IllegalArgumentException("a slice starts on a 64-row boundary, not at " + from);
        }
        int n = to - from;
        MemorySegment validity = validity() == null ? null : sliceBitmap(validity(), from, n);
        switch (type()) {
            case UTF8 -> {
                if (isDictionaryEncoded()) {
                    return SegmentVectorBuffers.dictionaryUtf8(
                            n,
                            validity,
                            data().asSlice((long) from << 2, (long) n << 2),
                            dictionary());
                }
                return SegmentVectorBuffers.utf8(
                        n,
                        validity,
                        offsets().asSlice((long) from << 2, ((long) n + 1) << 2),
                        data());
            }
            case BOOL -> {
                return SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, sliceBitmap(data(), from, n));
            }
            default -> {
                int width = type().byteWidth();
                return SegmentVectorBuffers.fixedWidth(type(), n, validity,
                        data().asSlice((long) from * width, (long) n * width));
            }
        }
    }

    /**
     * Bits {@code [from, from + n)} of a bitmap, {@code from} a multiple of 64;
     * whole words where the bitmap has them, so word-wise readers see what they
     * would in the original.
     */
    private static MemorySegment sliceBitmap(MemorySegment bitmap, int from, int n) {
        long offset = (long) from >>> 3;
        long bytes = Math.min(bitmap.byteSize() - offset, (long) Bitmap.wordsFor(n) << 3);
        return bitmap.asSlice(offset, bytes);
    }
}
