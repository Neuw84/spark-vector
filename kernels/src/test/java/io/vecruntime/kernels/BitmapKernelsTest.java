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
import java.util.Random;

import io.vecruntime.kernels.reference.ScalarReference;
import org.junit.jupiter.api.Test;

import static io.vecruntime.kernels.TestData.assertBitmapEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BitmapKernelsTest {

    @Test
    void booleanAlgebraMatchesReference() {
        Random rnd = new Random(3);
        for (int n : TestData.LENGTHS) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment a = TestData.randomBitmap(arena, rnd, n);
                MemorySegment b = TestData.randomBitmap(arena, rnd, n);
                MemorySegment exp = ArrowLayout.allocateBitmap(arena, n);
                MemorySegment act = ArrowLayout.allocateBitmap(arena, n);

                ScalarReference.and(a, b, exp, n);
                BitmapKernels.and(a, b, act, n);
                assertBitmapEquals(exp, act, n, "and");

                ScalarReference.or(a, b, exp, n);
                BitmapKernels.or(a, b, act, n);
                assertBitmapEquals(exp, act, n, "or");

                ScalarReference.not(a, exp, n);
                BitmapKernels.not(a, act, n);
                assertBitmapEquals(exp, act, n, "not");

                ScalarReference.selection(a, b, exp, n);
                BitmapKernels.selection(a, b, act, n);
                assertBitmapEquals(exp, act, n, "selection");

                ScalarReference.selection(a, null, exp, n);
                BitmapKernels.selection(a, null, act, n);
                assertBitmapEquals(exp, act, n, "selection no validity");

                ScalarReference.not(b, exp, n);
                BitmapKernels.isNull(b, act, n);
                assertBitmapEquals(exp, act, n, "isNull");

                BitmapKernels.isNull(null, act, n);
                assertTrue(Bitmap.noneSet(act, n), "isNull with no nulls");
                BitmapKernels.isNotNull(null, act, n);
                assertTrue(Bitmap.allSet(act, n), "isNotNull with no nulls");
                BitmapKernels.isNotNull(b, act, n);
                assertBitmapEquals(b, act, n, "isNotNull");
            }
        }
    }

    @Test
    void kleeneAndOrMatchReferenceWithAndWithoutValidity() {
        Random rnd = new Random(5);
        for (int n : TestData.LENGTHS) {
            for (int nullMode = 0; nullMode < 4; nullMode++) {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment aBits = TestData.randomBitmap(arena, rnd, n);
                    MemorySegment bBits = TestData.randomBitmap(arena, rnd, n);
                    MemorySegment aValid = (nullMode & 1) != 0 ? TestData.randomBitmap(arena, rnd, n) : null;
                    MemorySegment bValid = (nullMode & 2) != 0 ? TestData.randomBitmap(arena, rnd, n) : null;
                    MemorySegment expBits = ArrowLayout.allocateBitmap(arena, n);
                    MemorySegment expValid = ArrowLayout.allocateBitmap(arena, n);
                    MemorySegment actBits = ArrowLayout.allocateBitmap(arena, n);
                    MemorySegment actValid = ArrowLayout.allocateBitmap(arena, n);

                    ScalarReference.kleeneAnd(aBits, aValid, bBits, bValid, expBits, expValid,
                            n);
                    BitmapKernels.kleeneAnd(aBits, aValid, bBits, bValid, actBits, actValid,
                            n);
                    assertBitmapEquals(expValid, actValid, n, "and validity mode " + nullMode);
                    // Bits only matter where valid.
                    for (int i = 0; i < n; i++) {
                        if (Bitmap.isSet(expValid, i)) {
                            assertTrue(Bitmap.isSet(expBits, i) == Bitmap.isSet(actBits, i), "and bit " + i);
                        }
                    }

                    ScalarReference.kleeneOr(aBits, aValid, bBits, bValid, expBits, expValid,
                            n);
                    BitmapKernels.kleeneOr(aBits, aValid, bBits, bValid, actBits, actValid,
                            n);
                    assertBitmapEquals(expValid, actValid, n, "or validity mode " + nullMode);
                    for (int i = 0; i < n; i++) {
                        if (Bitmap.isSet(expValid, i)) {
                            assertTrue(Bitmap.isSet(expBits, i) == Bitmap.isSet(actBits, i), "or bit " + i);
                        }
                    }
                }
            }
        }
    }

    @Test
    void kleeneTruthTable() {
        try (Arena arena = Arena.ofConfined()) {
            // a: T T T F F F N N N ; b: T F N T F N T F N
            boolean[] aB = {true, true, true, false, false, false,
                    false, false, false};
            boolean[] aV = {true, true, true, true, true, true,
                    false, false, false};
            boolean[] bB = {true, false, false, true, false, false,
                    true, false, false};
            boolean[] bV = {true, true, false, true, true, false,
                    true, true, false};
            MemorySegment ab = ArrowLayout.ofBooleans(arena, aB, null).data();
            MemorySegment av = ArrowLayout.ofBooleans(arena, aV, null).data();
            MemorySegment bb = ArrowLayout.ofBooleans(arena, bB, null).data();
            MemorySegment bv = ArrowLayout.ofBooleans(arena, bV, null).data();
            MemorySegment oB = ArrowLayout.allocateBitmap(arena, 9);
            MemorySegment oV = ArrowLayout.allocateBitmap(arena, 9);

            BitmapKernels.kleeneAnd(ab, av, bb, bv, oB, oV,
                    9);
            // Expected AND: T F N F F F N F N
            boolean[] andValid = {true, true, false, true, true, true,
                    false, true, false};
            boolean[] andBits = {true, false, false, false, false, false,
                    false, false, false};
            for (int i = 0; i < 9; i++) {
                assertTrue(Bitmap.isSet(oV, i) == andValid[i], "and valid " + i);
                if (andValid[i]) {
                    assertTrue(Bitmap.isSet(oB, i) == andBits[i], "and bit " + i);
                }
            }

            BitmapKernels.kleeneOr(ab, av, bb, bv, oB, oV,
                    9);
            // Expected OR: T T T T F N T N N
            boolean[] orValid = {true, true, true, true, true, false,
                    true, false, false};
            boolean[] orBits = {true, true, true, true, false, false,
                    true, false, false};
            for (int i = 0; i < 9; i++) {
                assertTrue(Bitmap.isSet(oV, i) == orValid[i], "or valid " + i);
                if (orValid[i]) {
                    assertTrue(Bitmap.isSet(oB, i) == orBits[i], "or bit " + i);
                }
            }
        }
    }

    @Test
    void combineValidityDropsBitmapWhenNeitherSideHasNulls() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = ArrowLayout.allocateBitmap(arena, 10);
            assertFalse(BitmapKernels.combineValidity(null, null, out, 10));
            MemorySegment v = ArrowLayout.ofBooleans(
                    arena,
                    new boolean[] {true, false, true, true, false, true,
                            true, true, true, false},
                    null)
                    .data();
            assertTrue(BitmapKernels.combineValidity(v, null, out, 10));
            assertBitmapEquals(v, out, 10, "copy left");
            assertTrue(BitmapKernels.combineValidity(null, v, out, 10));
            assertBitmapEquals(v, out, 10, "copy right");
        }
    }
}
