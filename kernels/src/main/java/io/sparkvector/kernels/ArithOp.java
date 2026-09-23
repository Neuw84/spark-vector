package io.sparkvector.kernels;

/**
 * Binary arithmetic operators. DIV is only defined for FLOAT64 (Spark's `/`
 * yields double).
 */
public enum ArithOp {
    ADD,
    SUB,
    MUL,
    DIV;

    public boolean isCommutative() {
        return this == ADD || this == MUL;
    }
}
