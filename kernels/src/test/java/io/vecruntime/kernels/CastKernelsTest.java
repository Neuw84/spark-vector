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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Narrowing, boolean and day casts against Java's own conversions and Spark's
 * ANSI range test.
 */
class CastKernelsTest {

    @Test
    void narrowingMatchesJavaAndFlagsSparkOverflow() {
        try (Arena arena = Arena.ofConfined()) {
            long[] longs = {
                0,
                1,
                -1,
                Integer.MAX_VALUE,
                Integer.MIN_VALUE,
                (long) Integer.MAX_VALUE + 1,
                (long) Integer.MIN_VALUE - 1,
                Long.MAX_VALUE,
                Long.MIN_VALUE,
                1L << 40
            };
            VectorBuffers a = ArrowLayout.ofLongs(arena, longs, null);
            MemorySegment out = ArrowLayout.allocateData(arena, VecType.INT32, longs.length);
            MemorySegment overflow = ArrowLayout.allocateBitmap(arena, longs.length);
            CastKernels.narrow(a, VecType.INT32, out, overflow);
            for (int i = 0; i < longs.length; i++) {
                assertEquals((int) longs[i], out.getAtIndex(VectorBuffers.LE_INT, i),
                        "long " + longs[i]);
                assertEquals(longs[i] != (int) longs[i], Bitmap.isSet(overflow, i), "overflow " + longs[i]);
            }
            double[] doubles = {
                0.0,
                -0.0,
                3.7,
                -3.7,
                2147483647.0,
                2147483647.5,
                2147483648.0,
                -2147483648.0,
                -2147483648.9,
                -2147483649.0,
                9.3e18,
                -9.3e18,
                1e300,
                Double.NaN,
                Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                0.5,
                -0.999
            };
            VectorBuffers b = ArrowLayout.ofDoubles(arena, doubles, null);
            MemorySegment outI = ArrowLayout.allocateData(arena, VecType.INT32, doubles.length);
            MemorySegment ovI = ArrowLayout.allocateBitmap(arena, doubles.length);
            CastKernels.narrow(b, VecType.INT32, outI, ovI);
            MemorySegment outL = ArrowLayout.allocateData(arena, VecType.INT64, doubles.length);
            MemorySegment ovL = ArrowLayout.allocateBitmap(arena, doubles.length);
            CastKernels.narrow(b, VecType.INT64, outL, ovL);
            for (int i = 0; i < doubles.length; i++) {
                double v = doubles[i];
                assertEquals((int) v, outI.getAtIndex(VectorBuffers.LE_INT, i), "int of " + v);
                assertEquals(!(Math.floor(v) <= Integer.MAX_VALUE && Math.ceil(v) >= Integer.MIN_VALUE), Bitmap.isSet(ovI, i),
                        "int overflow of " + v);
                assertEquals((long) v, outL.getAtIndex(VectorBuffers.LE_LONG, i), "long of " + v);
                assertEquals(!(Math.floor(v) <= Long.MAX_VALUE && Math.ceil(v) >= Long.MIN_VALUE), Bitmap.isSet(ovL, i),
                        "long overflow of " + v);
            }
            // Spark's test is floor(v) <= MAX: 2147483647.5 truncates into range and is accepted; 2147483648.0 is not.
            assertFalse(Bitmap.isSet(ovI, 5));
            assertTrue(Bitmap.isSet(ovI, 6));
            assertTrue(Bitmap.isSet(ovI, 13)); // NaN
            assertEquals(0, outI.getAtIndex(VectorBuffers.LE_INT, 13));
        }
    }

    @Test
    void booleansAndDays() {
        try (Arena arena = Arena.ofConfined()) {
            double[] doubles = {0.0, -0.0, 1.5, -2.0,
                    Double.NaN, Double.NEGATIVE_INFINITY};
            MemorySegment bits = ArrowLayout.allocateBitmap(arena, doubles.length);
            CastKernels.toBool(ArrowLayout.ofDoubles(arena, doubles, null), bits);
            boolean[] expected = {false, false, true, true, true, true};
            for (int i = 0; i < doubles.length; i++) {
                assertEquals(expected[i], Bitmap.isSet(bits, i), "bool of " + doubles[i]);
            }
            MemorySegment ints = ArrowLayout.allocateBitmap(arena, 3);
            CastKernels.toBool(ArrowLayout.ofInts(arena, new int[] {0, -7, Integer.MIN_VALUE}, null), ints);
            assertFalse(Bitmap.isSet(ints, 0));
            assertTrue(Bitmap.isSet(ints, 1));
            assertTrue(Bitmap.isSet(ints, 2));
            // Back again as each numeric lane.
            VectorBuffers bools = SegmentVectorBuffers.fixedWidth(VecType.BOOL, doubles.length, null, bits);
            MemorySegment asInt = ArrowLayout.allocateData(arena, VecType.INT32, doubles.length);
            MemorySegment asLong = ArrowLayout.allocateData(arena, VecType.INT64, doubles.length);
            MemorySegment asDouble = ArrowLayout.allocateData(arena, VecType.FLOAT64, doubles.length);
            CastKernels.fromBool(bools, VecType.INT32, asInt);
            CastKernels.fromBool(bools, VecType.INT64, asLong);
            CastKernels.fromBool(bools, VecType.FLOAT64, asDouble);
            for (int i = 0; i < doubles.length; i++) {
                assertEquals(expected[i] ? 1 : 0, asInt.getAtIndex(VectorBuffers.LE_INT, i));
                assertEquals(expected[i] ? 1L : 0L, asLong.getAtIndex(VectorBuffers.LE_LONG, i));
                assertEquals(expected[i] ? 1.0 : 0.0, asDouble.getAtIndex(VectorBuffers.LE_DOUBLE, i));
            }
            // Days to micros under +05:30: local midnight is 18:30 UTC of the previous day.
            int[] days = {0, 1, -1, 19000, -25000};
            MemorySegment micros = ArrowLayout.allocateData(arena, VecType.INT64, days.length);
            long offset = (5 * 3600 + 30 * 60) * 1_000_000L;
            CastKernels.daysToMicros(ArrowLayout.ofInts(arena, days, null), offset, micros);
            for (int i = 0; i < days.length; i++) {
                long expectedMicros = java.time.LocalDate.ofEpochDay(days[i])
                        .atStartOfDay(java.time.ZoneOffset.ofHoursMinutes(5, 30))
                        .toInstant()
                        .toEpochMilli()
                        * 1000L;
                assertEquals(expectedMicros, micros.getAtIndex(VectorBuffers.LE_LONG, i), "day " + days[i]);
            }
        }
    }
}
