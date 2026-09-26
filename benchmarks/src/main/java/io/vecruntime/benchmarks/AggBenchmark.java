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
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.vecruntime.kernels.AggKernels;
import io.vecruntime.kernels.ArrowLayout;
import io.vecruntime.kernels.VectorBuffers;
import io.vecruntime.kernels.reference.ScalarReference;
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

/** Ungrouped reductions: sum/min/max over doubles with varying null density. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(AggBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class AggBenchmark {

    static final int N = 8192;

    /** Fraction of null rows: 0 (fast path), sparse, dense. */
    @Param({"0.0", "0.01", "0.3"})
    double nullFraction;

    Arena arena;
    VectorBuffers doubles;
    VectorBuffers longs;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(5);
        boolean[] nulls = nullFraction == 0.0 ? null : new boolean[N];
        double[] d = new double[N];
        long[] l = new long[N];
        for (int i = 0; i < N; i++) {
            d[i] = rnd.nextDouble() * 1000;
            l[i] = rnd.nextLong(1_000_000);
            if (nulls != null) {
                nulls[i] = rnd.nextDouble() < nullFraction;
            }
        }
        doubles = ArrowLayout.ofDoubles(arena, d, nulls);
        longs = ArrowLayout.ofLongs(arena, l, nulls);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public double sumDouble_simd() {
        return AggKernels.sumDouble(doubles);
    }

    @Benchmark
    public double sumDouble_reference() {
        return ScalarReference.sumDouble(doubles);
    }

    @Benchmark
    public long sumLong_simd() {
        return AggKernels.sumLong(longs);
    }

    @Benchmark
    public long sumLong_reference() {
        return ScalarReference.sumLong(longs);
    }

    @Benchmark
    public double minDouble_simd() {
        return AggKernels.minDouble(doubles);
    }

    @Benchmark
    public double minDouble_reference() {
        return ScalarReference.minDouble(doubles);
    }
}
