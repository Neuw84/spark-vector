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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ByteCopyTest {

    /**
     * Every length through the long-pair threshold, at every source and
     * destination misalignment.
     */
    @Test
    void copiesEveryShortLengthAtEveryAlignment() {
        Random rnd = new Random(5);
        try (Arena arena = Arena.ofConfined()) {
            byte[] source = new byte[256];
            rnd.nextBytes(source);
            MemorySegment src = arena.allocate(source.length);
            MemorySegment.copy(source, 0, src, ValueLayout.JAVA_BYTE, 0, source.length);
            for (int len = 0; len <= ByteCopy.SHORT + 20; len++) {
                for (int srcOff = 0; srcOff < 9; srcOff++) {
                    for (int dstOff = 0; dstOff < 9; dstOff++) {
                        MemorySegment dst = arena.allocate(len + 32);
                        dst.fill((byte) 0x5A);
                        ByteCopy.copy(src, srcOff, dst, dstOff, len);
                        byte[] expected = new byte[len + 32];
                        java.util.Arrays.fill(expected, (byte) 0x5A);
                        System.arraycopy(source, srcOff, expected, dstOff, len);
                        assertArrayEquals(expected, dst.toArray(ValueLayout.JAVA_BYTE), "len=" + len + " src+" + srcOff + " dst+" + dstOff);
                    }
                }
            }
        }
    }
}
