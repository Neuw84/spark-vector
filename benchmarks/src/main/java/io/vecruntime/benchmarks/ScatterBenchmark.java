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
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.vecruntime.kernels.GatherKernels;
import io.vecruntime.kernels.PartitionKernels;
import io.vecruntime.kernels.ScatterKernels;
import io.vecruntime.kernels.VecType;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * The shuffle writer's staged flush (#20): partitioning one staged column of
 * {@code rows} values over {@code partitions} partitions, by a gather through
 * the partition order (the flush before #20) against a scatter to each row's
 * slot. Both include computing their index array. Time per staged column.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class ScatterBenchmark {

    @Param({"1048576", "4194304"})
    int rows;

    @Param({"300"})
    int partitions;

    @Param({"INT64", "INT32"})
    VecType type;

    Arena arena;
    MemorySegment in;
    MemorySegment out;
    int[] ids;
    int[] starts;
    int[] index;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        int width = type == VecType.INT32 ? 4 : 8;
        in = arena.allocate((long) rows * width, 8);
        out = arena.allocate((long) rows * width, 8);
        Random r = new Random(20);
        for (long b = 0; b < in.byteSize(); b += 8) {
            in.set(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED,
                    b, r.nextLong());
        }
        ids = new int[rows];
        for (int i = 0; i < rows; i++) {
            ids[i] = r.nextInt(partitions);
        }
        starts = new int[partitions + 1];
        index = new int[rows];
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public MemorySegment gatherThroughOrder() {
        PartitionKernels.partitionOrder(ids, rows, partitions, starts, index);
        GatherKernels.gatherFixedPlain(type, in, index, 0, rows, out);
        return out;
    }

    @Benchmark
    public MemorySegment scatterToSlots() {
        PartitionKernels.partitionDestinations(ids, rows, partitions, starts, index);
        ScatterKernels.scatterFixed(type, in, rows, index, out);
        return out;
    }
}
