package io.sparkvector.kernels;

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
