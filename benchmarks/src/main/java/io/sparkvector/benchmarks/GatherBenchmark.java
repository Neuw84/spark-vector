package io.sparkvector.benchmarks;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.ByteCopy;
import io.sparkvector.kernels.GatherKernels;
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
 * Gathering a plain UTF8 column by random indices, the join output path: the
 * kernel (short strings moved as long pairs through {@link ByteCopy}) against
 * the same loop with one {@link MemorySegment#copy} per string.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(GatherBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class GatherBenchmark {
    static final int N = 4096;
    static final int SOURCE_ROWS = 65536;

    /** Length of every string. */
    @Param({"6", "12", "25", "64"})
    int length;

    Arena arena;
    VectorBuffers in;
    int[] idx;
    MemorySegment outOffsets;
    MemorySegment outData;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(7);
        String[] values = new String[SOURCE_ROWS];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < SOURCE_ROWS; i++) {
            sb.setLength(0);
            for (int k = 0; k < length; k++) {
                sb.append((char) ('a' + rnd.nextInt(26)));
            }
            values[i] = sb.toString();
        }
        in = ArrowLayout.ofStrings(arena, values);
        idx = new int[N];
        for (int i = 0; i < N; i++) {
            idx[i] = rnd.nextInt(SOURCE_ROWS);
        }
        outOffsets = ArrowLayout.allocateOffsets(arena, N);
        outData = ArrowLayout.allocateBytes(arena, (long) N * length);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public int gather_kernel() {
        GatherKernels.gatherUtf8(in, idx, 0, N, outOffsets, outData,
                null);
        return outOffsets.get(VectorBuffers.LE_INT, (long) N << 2);
    }

    @Benchmark
    public int gather_segmentCopyPerString() {
        MemorySegment off = in.offsets();
        MemorySegment data = in.data();
        int pos = 0;
        for (int o = 0; o < N; o++) {
            int i = idx[o];
            outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
            int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
            int len = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - start;
            MemorySegment.copy(data, ValueLayout.JAVA_BYTE, start, outData, ValueLayout.JAVA_BYTE, pos,
                    len);
            pos += len;
        }
        outOffsets.set(VectorBuffers.LE_INT, (long) N << 2, pos);
        return pos;
    }
}
