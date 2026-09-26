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
package io.vecruntime.spark.arrow;

import java.lang.foreign.MemorySegment;

import org.apache.arrow.memory.ArrowBuf;

/**
 * Zero-copy views of Arrow buffers as {@link MemorySegment}s.
 *
 * <p>{@code MemorySegment.ofAddress(...).reinterpret(...)} is a restricted FFM
 * method; the JVM must run with {@code --enable-native-access=ALL-UNNAMED}
 * (otherwise JDK 25 prints a warning).
 *
 * <p>The returned segment aliases the buffer's current address. It becomes
 * invalid if the buffer is reallocated (vector grows) or released, so callers
 * wrap buffers after the vector has been sized and drop the segment before the
 * vector is closed.
 */
public final class ArrowSegments {

    private ArrowSegments() {}

    /** View over {@code [0, buf.capacity())}. */
    public static MemorySegment of(ArrowBuf buf) {
        return of(buf, buf.capacity());
    }

    public static MemorySegment of(ArrowBuf buf, long byteSize) {
        if (byteSize == 0) {
            return MemorySegment.NULL.reinterpret(0);
        }
        return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(byteSize);
    }
}
