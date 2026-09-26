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
package io.sparkvector.benchmarks;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.CompactKernels;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.kernels.reference.ScalarReference;
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

/** Compaction of a column by a selection bitmap at several selectivities. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(CompactBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class CompactBenchmark {

    static final int N = 8192;

    @Param({"INT32", "FLOAT64"})
    String type;

    /** Fraction of selected rows. */
    @Param({"0.02", "0.5", "0.98"})
    double selectivity;

    @Param({"false", "true"})
    boolean withNulls;

    Arena arena;
    VectorBuffers in;
    MemorySegment selection;
    int outCount;
    MemorySegment outData;
    MemorySegment outValidity;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(2);
        boolean[] nulls = null;
        if (withNulls) {
            nulls = new boolean[N];
            for (int i = 0; i < N; i++) {
                nulls[i] = rnd.nextInt(10) == 0;
            }
        }
        if (type.equals("INT32")) {
            int[] x = new int[N];
            for (int i = 0; i < N; i++) {
                x[i] = rnd.nextInt();
            }
            in = ArrowLayout.ofInts(arena, x, nulls);
        } else {
            double[] x = new double[N];
            for (int i = 0; i < N; i++) {
                x[i] = rnd.nextDouble();
            }
            in = ArrowLayout.ofDoubles(arena, x, nulls);
        }
        selection = ArrowLayout.allocateBitmap(arena, N);
        for (int i = 0; i < N; i++) {
            if (rnd.nextDouble() < selectivity) {
                Bitmap.set(selection, i);
            }
        }
        outCount = CompactKernels.selectedCount(selection, N);
        VecType t = in.type();
        outData = ArrowLayout.allocateData(arena, t, Math.max(outCount, 1));
        outValidity = withNulls ? ArrowLayout.allocateBitmap(arena, Math.max(outCount, 1)) : null;
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public MemorySegment compact_simd() {
        CompactKernels.compactFixed(in, selection, outCount, outData, outValidity);
        return outData;
    }

    @Benchmark
    public MemorySegment compact_reference() {
        ScalarReference.compactFixed(in, selection, outData, outValidity);
        return outData;
    }
}
