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
package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.math.BigInteger;

/**
 * The {@link VecType#DECIMAL128} value layout and the conversions every adapter
 * and kernel needs.
 *
 * <p>A value is a two's complement 128-bit integer (the unscaled decimal)
 * stored as two little-endian {@code long} limbs: the low limb at byte offset
 * {@code 16 * i}, the high limb at {@code 16 * i + 8}. Both are read and
 * written through {@link VectorBuffers#LE_LONG}, never in native order, so a
 * buffer is Arrow {@code Decimal128} on every host and can be handed to Arrow
 * or Comet as is.
 *
 * <p>Arithmetic on the limbs (add, multiply, divide with Spark's rounding) is
 * #258's; this class only moves values in and out of the layout and orders and
 * hashes them.
 */
public final class Decimal128 {

    private Decimal128() {}

    /** Bytes per value. */
    public static final int WIDTH = 16;

    /** Low (least significant) limb of value {@code i}. */
    public static long lo(MemorySegment data, int i) {
        return data.get(VectorBuffers.LE_LONG, (long) i << 4);
    }

    /**
     * High (most significant) limb of value {@code i}; its sign bit is the
     * value's sign.
     */
    public static long hi(MemorySegment data, int i) {
        return data.get(VectorBuffers.LE_LONG, ((long) i << 4) + 8);
    }

    public static void set(MemorySegment data, int i, long hi,
                           long lo) {
        long off = (long) i << 4;
        data.set(VectorBuffers.LE_LONG, off, lo);
        data.set(VectorBuffers.LE_LONG, off + 8, hi);
    }

    /** Copies value {@code from} of {@code in} to slot {@code to} of {@code out}. */
    public static void copy(MemorySegment in, int from, MemorySegment out,
                            int to) {
        MemorySegment.copy(in, (long) from << 4, out, (long) to << 4,
                WIDTH);
    }

    /**
     * High limb of a two's complement big-endian byte string of at most 16
     * bytes (the form Spark's {@code WritableColumnVector.getBinary} and
     * Parquet's {@code FIXED_LEN_BYTE_ARRAY} decimals use), sign-extended from
     * its first byte. {@link #loFromBigEndian} gives the other limb.
     */
    public static long hiFromBigEndian(byte[] bytes, int offset, int length) {
        if (length > WIDTH) {
            throw new IllegalArgumentException("decimal wider than 16 bytes: " + length);
        }
        long sign = length > 0 && bytes[offset] < 0
                ? -1L
                : 0L;
        int highBytes = Math.max(0, length - 8);
        long hi = sign;
        for (int k = 0; k < highBytes; k++) {
            hi = (hi << 8) | (bytes[offset + k] & 0xFFL);
        }
        return hi;
    }

    public static long loFromBigEndian(byte[] bytes, int offset, int length) {
        if (length > WIDTH) {
            throw new IllegalArgumentException("decimal wider than 16 bytes: " + length);
        }
        long sign = length > 0 && bytes[offset] < 0
                ? -1L
                : 0L;
        int lowBytes = Math.min(8, length);
        int start = offset + length - lowBytes;
        long lo = lowBytes == 8 ? 0L : sign;
        for (int k = 0; k < lowBytes; k++) {
            lo = (lo << 8) | (bytes[start + k] & 0xFFL);
        }
        return lo;
    }

    /**
     * Writes the two's complement big-endian byte string of length {@code
     * length} (16 at most).
     */
    public static void toBigEndian(long hi, long lo, byte[] out,
            int offset, int length) {
        for (int k = length - 1; k >= 0; k--) {
            int shift = (length - 1 - k) << 3;
            long limb = shift < 64 ? lo : hi;
            int s = shift & 63;
            out[offset + k] = (byte) (limb >>> s);
        }
    }

    /** Big-endian bytes of the value, 16 wide. */
    public static byte[] toBigEndian(long hi, long lo) {
        byte[] out = new byte[WIDTH];
        toBigEndian(hi, lo, out, 0, WIDTH);
        return out;
    }

    public static BigInteger toBigInteger(long hi, long lo) {
        return new BigInteger(toBigEndian(hi, lo));
    }

    /**
     * High limb of a {@code BigInteger} that fits in 128 bits (it is not
     * range-checked).
     */
    public static long hiOf(BigInteger v) {
        return v.shiftRight(64).longValue();
    }

    public static long loOf(BigInteger v) {
        return v.longValue();
    }

    /** Signed order of two values: high limbs signed, then low limbs unsigned. */
    public static int compare(long hi1, long lo1, long hi2,
            long lo2) {
        int c = Long.compare(hi1, hi2);
        return c != 0 ? c : Long.compareUnsigned(lo1, lo2);
    }

    /**
     * Mixes both limbs into one {@code long}; equal values hash equal, as the
     * group key tables need.
     */
    public static long hash(long hi, long lo) {
        long h = lo * 0x9E3779B97F4A7C15L;
        h ^= (h >>> 32) ^ hi * 0xC2B2AE3D27D4EB4FL;
        h ^= h >>> 29;
        return h;
    }
}
