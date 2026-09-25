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

import java.util.Set;

import jdk.incubator.vector.VectorMask;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The platform probe (#283) and the equivalence of the two mask constructions
 * it chooses between.
 */
class PlatformTest {

    @Test
    void probeNamesAKnownPlatformAndItsConstantsAgree() {
        Set<String> known = Set.of(Platform.NEON, Platform.SVE, Platform.AVX2, Platform.AVX512, "unknown");
        assertTrue(known.contains(Platform.NAME), Platform.NAME);
        boolean predicated = Platform.NAME.equals(Platform.AVX512) || Platform.NAME.equals(Platform.SVE);
        assertEquals(predicated, Platform.NATIVE_COMPRESS);
        // #253: fromLong masks only on AVX-512 (not intrinsified at 128-bit SVE on JDK 25), unless overridden.
        String maskOverride = System.getProperty("sparkvector.maskRegisters");
        boolean expectedMasks = maskOverride == null ? Platform.NAME.equals(Platform.AVX512) : Boolean.parseBoolean(maskOverride);
        assertEquals(expectedMasks, Platform.MASK_REGISTERS);
        String arch = System.getProperty("os.arch", "");
        if (System.getProperty("sparkvector.platform") == null && (arch.equals("amd64") || arch.equals("x86_64"))) {
            // HotSpot on x86-64 always has some AVX level; the probe must not fall through to "unknown".
            assertTrue(Platform.NAME.equals(Platform.AVX2) || Platform.NAME.equals(Platform.AVX512), Platform.NAME);
            assertTrue(Platform.MAX_VECTOR_BYTES >= 16, "MaxVectorSize " + Platform.MAX_VECTOR_BYTES);
        }
    }

    @Test
    void groupedAggregationDefaultsFollowThePlatform() {
        // #283, decision 3: the masked path up to 4 groups where masks are registers and the species has
        // at least 4 lanes, 1 group elsewhere; one scatter copy on AVX-512, four elsewhere. The
        // properties override both, and only an explicit interleave=1 means Spark's order.
        int lanes = Species.DOUBLE_LANES;
        assertEquals(Platform.MASK_REGISTERS && lanes >= 4 ? 4 : 1,
                GroupAssignment.defaultMaskPathMaxGroups());
        assertEquals(Platform.NAME.equals(Platform.AVX512) ? 1 : 4, GroupedAccumulators.defaultInterleave());
        if (System.getProperty("sparkvector.agg.maskPathMaxGroups") == null) {
            assertEquals(GroupAssignment.defaultMaskPathMaxGroups(), GroupAssignment.LOW_CARDINALITY);
        }
        String interleave = System.getProperty("sparkvector.agg.interleave");
        assertEquals("1".equals(interleave), GroupedAccumulators.SEQUENTIAL_SUMS);
        if (interleave == null) {
            assertEquals(GroupedAccumulators.defaultInterleave(), GroupedAccumulators.INTERLEAVE);
        }
    }

    @Test
    void bothMaskFormsAgreeOnEveryLanePattern() {
        int longLanes = Species.L.length();
        for (long bits = 0; bits < (1L << longLanes); bits++) {
            VectorMask<Long> expected = VectorMask.fromLong(Species.L, bits);
            assertEquals(
                    expected.toLong(),
                    AggKernels.maskLBroadcast(bits).toLong(),
                    "long lanes " + bits);
            assertEquals(
                    expected.toLong(),
                    AggKernels.maskL(bits).toLong(),
                    "maskL " + bits);
            assertEquals(bits, AggKernels.maskD(bits).toLong(),
                    "maskD " + bits);
        }
        int intLanes = Species.I.length();
        long step = intLanes > 8 ? 4099 : 1; // every pattern for 8 lanes, a dense sample of the 65536 for 16
        for (long bits = 0; bits < (1L << intLanes); bits += step) {
            assertEquals(bits, AggKernels.maskIBroadcast(bits).toLong(),
                    "int lanes " + bits);
            assertEquals(bits, AggKernels.maskI(bits).toLong(),
                    "maskI " + bits);
        }
        int halfLanes = Species.IH.length();
        for (long bits = 0; bits < (1L << halfLanes); bits++) {
            assertEquals(bits, AggKernels.maskIHBroadcast(bits).toLong(),
                    "half int lanes " + bits);
            assertEquals(bits, AggKernels.maskIH(bits).toLong(),
                    "maskIH " + bits);
        }
    }
}
