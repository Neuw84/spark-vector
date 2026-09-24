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

/**
 * Physical vector types understood by the kernels. Logical types are mapped
 * onto these by the Spark layer (e.g. DateType is INT32 days, TimestampType is
 * INT64 micros).
 */
public enum VecType {
    /** Arrow boolean: values are a bitmap, one bit per element. */
    BOOL(0),
    INT32(4),
    INT64(8),
    FLOAT64(8),
    /** Arrow Utf8: int32 offsets (length + 1 entries) plus a byte buffer. */
    UTF8(-1),
    /**
     * Arrow Decimal128: 16 bytes per value, a two's complement 128-bit integer
     * stored little-endian as two {@code long} limbs (low at offset 0, high at
     * offset 8; see {@link Decimal128}). Holds the unscaled value of a decimal
     * wider than 18 digits. The lane is deliberately scalar: no {@code Species}
     * covers it and every kernel walks it limb by limb (#28, option 1).
     */
    DECIMAL128(16);

    private final int byteWidth;

    VecType(int byteWidth) {
        this.byteWidth = byteWidth;
    }

    /**
     * Bytes per element for fixed-width types; 0 for BOOL (bit-packed); -1 for
     * variable width.
     */
    public int byteWidth() {
        return byteWidth;
    }

    public boolean isFixedWidth() {
        return byteWidth > 0;
    }
}
