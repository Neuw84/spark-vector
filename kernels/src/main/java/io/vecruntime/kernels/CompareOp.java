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

import jdk.incubator.vector.VectorOperators;

/** Comparison operators, with their Vector API counterparts. */
@SuppressWarnings("ImmutableEnumChecker") // VectorOperators constants are immutable, just not annotated so
public enum CompareOp {
    EQ(VectorOperators.EQ),
    NE(VectorOperators.NE),
    LT(VectorOperators.LT),
    LE(VectorOperators.LE),
    GT(VectorOperators.GT),
    GE(VectorOperators.GE);

    final VectorOperators.Comparison vop;

    CompareOp(VectorOperators.Comparison vop) {
        this.vop = vop;
    }

    /** The operator with operands swapped ({@code a < b} is {@code b > a}). */
    public CompareOp flip() {
        return switch (this) {
            case EQ -> EQ;
            case NE -> NE;
            case LT -> GT;
            case LE -> GE;
            case GT -> LT;
            case GE -> LE;
        };
    }

    public boolean test(int cmp) {
        return switch (this) {
            case EQ -> cmp == 0;
            case NE -> cmp != 0;
            case LT -> cmp < 0;
            case LE -> cmp <= 0;
            case GT -> cmp > 0;
            case GE -> cmp >= 0;
        };
    }

    /**
     * Spark's NaN-safe double semantics (see {@code
     * Utils.nanSafeCompareDoubles}): NaN equals NaN and is greater than every
     * other value; {@code -0.0 == 0.0}.
     */
    public boolean test(double a, double b) {
        return test(nanSafeCompare(a, b));
    }

    public static int nanSafeCompare(double x, double y) {
        boolean xNaN = Double.isNaN(x);
        boolean yNaN = Double.isNaN(y);
        if ((xNaN && yNaN) || x == y) {
            return 0;
        }
        if (xNaN) {
            return 1;
        }
        if (yNaN) {
            return -1;
        }
        return x > y ? 1 : -1;
    }
}
