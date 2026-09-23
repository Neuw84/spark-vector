package io.sparkvector.benchmarks;

import java.lang.foreign.Arena;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.GroupAssignment;
import io.sparkvector.kernels.GroupedAccumulators;
import io.sparkvector.kernels.VectorBuffers;
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
 * Grouped double sum over a batch, mask path (per-group masked SIMD reductions)
 * versus scatter path, for a few groups (TPC-H Q1 has 4).
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(GroupedAggBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class GroupedAggBenchmark {

    static final int N = 4096;

    @Param({"1", "4", "16"})
    int groups;

    Arena arena;
    VectorBuffers values;
    int[] ids;
    GroupAssignment masked;
    GroupAssignment scattered;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(3);
        double[] d = new double[N];
        ids = new int[N];
        for (int i = 0; i < N; i++) {
            d[i] = rnd.nextDouble() * 100;
            ids[i] = rnd.nextInt(groups);
        }
        values = ArrowLayout.ofDoubles(arena, d, null);
        masked = GroupAssignment.of(ids, N, groups, arena, true);
        scattered = GroupAssignment.of(ids, N, groups, arena, false);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public double sumMaskPath() {
        GroupedAccumulators.DoubleSum acc = new GroupedAccumulators.DoubleSum();
        acc.update(values, masked);
        return acc.sum(0);
    }

    @Benchmark
    public double sumScatterPath() {
        GroupedAccumulators.DoubleSum acc = new GroupedAccumulators.DoubleSum();
        acc.update(values, scattered);
        return acc.sum(0);
    }
}
