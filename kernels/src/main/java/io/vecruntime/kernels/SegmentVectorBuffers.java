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
import java.util.Objects;

/** Plain {@link VectorBuffers} over caller-provided segments. */
public record SegmentVectorBuffers(VecType type, int length, MemorySegment validity,
        MemorySegment data, MemorySegment offsets, VectorBuffers dictionary)
        implements VectorBuffers {

    public SegmentVectorBuffers {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(data, "data");
        if (length < 0) {
            throw new IllegalArgumentException("negative length: " + length);
        }
        if (type == VecType.UTF8 && dictionary == null && offsets == null) {
            throw new IllegalArgumentException("UTF8 without dictionary requires offsets");
        }
        if (dictionary != null && dictionary.type() != VecType.UTF8) {
            throw new IllegalArgumentException("only UTF8 dictionaries are supported");
        }
    }

    public static SegmentVectorBuffers fixedWidth(VecType type, int length, MemorySegment validity,
            MemorySegment data) {
        return new SegmentVectorBuffers(type, length, validity, data, null, null);
    }

    public static SegmentVectorBuffers utf8(int length, MemorySegment validity, MemorySegment offsets,
            MemorySegment data) {
        return new SegmentVectorBuffers(VecType.UTF8, length, validity, data, offsets, null);
    }

    public static SegmentVectorBuffers dictionaryUtf8(int length, MemorySegment validity, MemorySegment indices,
            VectorBuffers dictionary) {
        return new SegmentVectorBuffers(VecType.UTF8, length, validity, indices, null, dictionary);
    }

    /** Same buffers with a different (or dropped) validity bitmap. */
    public SegmentVectorBuffers withValidity(MemorySegment newValidity) {
        return new SegmentVectorBuffers(type, length, newValidity, data, offsets, dictionary);
    }
}
