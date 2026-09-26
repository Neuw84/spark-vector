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
package io.vecruntime.benchmarks;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.vecruntime.kernels.ArrowLayout;
import io.vecruntime.kernels.Bitmap;
import io.vecruntime.kernels.CompareKernels;
import io.vecruntime.kernels.CompareOp;
import io.vecruntime.kernels.VecType;
import io.vecruntime.kernels.VectorBuffers;
import io.vecruntime.kernels.WideDecimalKernels;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Wide decimals (#258): the two-limb DECIMAL128 kernels against the per-row
 * {@code BigDecimal} loop Spark's row path runs, on {@code decimal(38,10)}
 * columns. {@code magnitude} picks the values: {@code long} -- unscaled values
 * that fit a long (TPC-H-sized amounts declared wide, the common case), where
 * the kernels stay on the limbs; {@code wide} -- unscaled values of 30-37
 * digits, where {@code +} still runs on the limbs and {@code *} takes the exact
 * path on every row (the product leaves 128 bits). Division is {@code
 * BigDecimal} by construction on both sides, so its two numbers show only the
 * columnar overhead.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(WideDecimalBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class WideDecimalBenchmark {

    /** Rows per batch; throughput is reported per element. */
    static final int N = 8192;

    static final int SCALE = 10;
    static final int PRECISION = 38;

    @Param({"long", "wide"})
    String magnitude;

    Arena arena;
    VectorBuffers a;
    VectorBuffers b;
    BigDecimal[] ra;
    BigDecimal[] rb;
    BigDecimal[] ra2;
    BigDecimal[] rb2;
    BigDecimal scalarDec;
    BigInteger scalar;
    MemorySegment out;
    MemorySegment overflow;
    MemorySegment zero;
    MemorySegment sel;

    /**
     * Sink for the reference loops; the results are summed into it so the JIT
     * keeps them.
     */
    long sink;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(7);
        BigInteger[] x = new BigInteger[N];
        BigInteger[] y = new BigInteger[N];
        for (int i = 0; i < N; i++) {
            if ("long".equals(magnitude)) {
                // Amounts up to a few hundred million at scale 10: 18 digits, the limbs' fast path.
                x[i] = BigInteger.valueOf(rnd.nextLong(1, 400_000_000_000_000_000L));
                y[i] = BigInteger.valueOf(rnd.nextLong(1, 400_000_000_000_000_000L));
            } else {
                x[i] = new BigInteger(30 + rnd.nextInt(8) * 3, rnd).add(BigInteger.ONE);
                y[i] = new BigInteger(30 + rnd.nextInt(8) * 3, rnd).add(BigInteger.ONE);
                if (rnd.nextBoolean()) {
                    x[i] = x[i].negate();
                }
            }
        }
        a = ArrowLayout.ofDecimal128(arena, x, null);
        b = ArrowLayout.ofDecimal128(arena, y, null);
        ra = new BigDecimal[N];
        rb = new BigDecimal[N];
        ra2 = new BigDecimal[N];
        rb2 = new BigDecimal[N];
        for (int i = 0; i < N; i++) {
            ra[i] = new BigDecimal(x[i], SCALE);
            rb[i] = new BigDecimal(y[i], SCALE);
            ra2[i] = new BigDecimal(x[i], 2);
            rb2[i] = new BigDecimal(y[i], 2);
        }
        scalar = x[N / 2];
        scalarDec = new BigDecimal(scalar, SCALE);
        out = ArrowLayout.allocateData(arena, VecType.DECIMAL128, N);
        overflow = ArrowLayout.allocateBitmap(arena, N);
        zero = ArrowLayout.allocateBitmap(arena, N);
        sel = ArrowLayout.allocateBitmap(arena, N);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    // ---- + (same scale: the limbs, no rescale) --------------------------------------------------------

    @Benchmark
    public MemorySegment addKernel() {
        Bitmap.fill(overflow, N, false);
        WideDecimalKernels.addSub(WideDecimalKernels.Operand.of(a, 0), WideDecimalKernels.Operand.of(b, 0), false, SCALE,
                SCALE, PRECISION, N, out, overflow);
        return out;
    }

    @Benchmark
    public long addReference() {
        long s = 0;
        for (int i = 0; i < N; i++) {
            BigDecimal r = ra[i].add(rb[i]);
            // Spark's toPrecision: the precision check the kernel also runs.
            if (r.precision() > PRECISION) {
                s++;
            } else {
                s += r.unscaledValue().bitLength();
            }
        }
        return sink = s;
    }

    // ---- * (exact scale 20, rounded half up to 10: Spark's decimal(38,10) * decimal(38,10) -> (38,6) is
    // capped; we keep the declared scale here so both magnitudes exercise the same path) -------------------

    @Benchmark
    public MemorySegment mulKernel() {
        Bitmap.fill(overflow, N, false);
        WideDecimalKernels.mul(WideDecimalKernels.Operand.of(a, 0), WideDecimalKernels.Operand.of(b, 0), 2 * SCALE,
                SCALE, PRECISION, N, out, overflow);
        return out;
    }

    @Benchmark
    public long mulReference() {
        long s = 0;
        for (int i = 0; i < N; i++) {
            BigDecimal r = ra[i].multiply(rb[i]).setScale(SCALE, RoundingMode.HALF_UP);
            if (r.precision() > PRECISION) {
                s++;
            } else {
                s += r.unscaledValue().bitLength();
            }
        }
        return sink = s;
    }

    // ---- * where the result keeps the exact scale (decimal(20,2) * decimal(20,2) -> decimal(38,4) in Spark): the
    // limb fast path -- a 128-bit product via multiplyHigh when both operands fit a long. The same unscaled values
    // read at scale 2. On the wide magnitude every product leaves 38 digits, so both sides only flag overflow. ----

    @Benchmark
    public MemorySegment mulExactScaleKernel() {
        Bitmap.fill(overflow, N, false);
        WideDecimalKernels.mul(WideDecimalKernels.Operand.of(a, 0), WideDecimalKernels.Operand.of(b, 0), 4, 4,
                PRECISION, N, out, overflow);
        return out;
    }

    @Benchmark
    public long mulExactScaleReference() {
        long s = 0;
        for (int i = 0; i < N; i++) {
            BigDecimal r = ra2[i].multiply(rb2[i]);
            if (r.precision() > PRECISION) {
                s++;
            } else {
                s += r.unscaledValue().bitLength();
            }
        }
        return sink = s;
    }

    // ---- / (BigDecimal on both sides) -------------------------------------------------------------------

    @Benchmark
    public MemorySegment divKernel() {
        Bitmap.fill(overflow, N, false);
        Bitmap.fill(zero, N, false);
        WideDecimalKernels.divide(
                WideDecimalKernels.Operand.of(a, 0),
                SCALE,
                WideDecimalKernels.Operand.of(b, 0),
                SCALE,
                SCALE,
                PRECISION,
                N,
                out,
                overflow,
                zero);
        return out;
    }

    @Benchmark
    public long divReference() {
        long s = 0;
        for (int i = 0; i < N; i++) {
            BigDecimal r = ra[i].divide(rb[i], 38, RoundingMode.HALF_UP).setScale(SCALE, RoundingMode.HALF_UP);
            if (r.precision() > PRECISION) {
                s++;
            } else {
                s += r.unscaledValue().bitLength();
            }
        }
        return sink = s;
    }

    // ---- compare (column < literal, column < column) -> selection bitmap ---------------------------------

    @Benchmark
    public MemorySegment compareScalarKernel() {
        CompareKernels.compareScalar(a, scalar, CompareOp.LT, sel);
        return sel;
    }

    @Benchmark
    public long compareScalarReference() {
        long s = 0;
        for (int i = 0; i < N; i++) {
            if (ra[i].compareTo(scalarDec) < 0) {
                s++;
            }
        }
        return sink = s;
    }

    @Benchmark
    public MemorySegment compareColumnKernel() {
        CompareKernels.compare(a, b, CompareOp.LT, sel);
        return sel;
    }

    @Benchmark
    public long compareColumnReference() {
        long s = 0;
        for (int i = 0; i < N; i++) {
            if (ra[i].compareTo(rb[i]) < 0) {
                s++;
            }
        }
        return sink = s;
    }
}
