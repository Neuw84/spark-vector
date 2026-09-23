package io.sparkvector.benchmarks;

import java.lang.foreign.Arena;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.GroupKeyTable;
import io.sparkvector.kernels.VecType;
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
 * Group-id assignment over one plain (non-dictionary) UTF8 key column, the
 * shape Comet's native scan hands over: `keyBytes` is the value length (8 fits
 * the packed fast path, 24 is a nation or region name), `distinct` the
 * cardinality (25 nations, 10k a wide key that still fits the plain dictionary
 * cap, 200k one that overflows it and takes the hashing path).
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(GroupKeyTableBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class GroupKeyTableBenchmark {

    static final int N = 4096;

    /**
     * Batches per table: the dictionary is built on the first and reused on the
     * rest.
     */
    static final int BATCHES = 64;

    @Param({"8", "24"})
    int keyBytes;

    @Param({"25", "10000", "200000"})
    int distinct;

    Arena arena;
    VectorBuffers[][] batches;
    int[] ids;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        Random rnd = new Random(7);
        String[] values = new String[distinct];
        for (int v = 0; v < distinct; v++) {
            StringBuilder sb = new StringBuilder(keyBytes);
            // Distinct values share their prefix (like "UNITED KINGDOM" / "UNITED STATES") and differ
            // in the trailing digits, so a byte compare has to read to the end.
            String tail = Integer.toString(v);
            while (sb.length() < keyBytes - tail.length()) {
                sb.append('k');
            }
            sb.append(tail);
            values[v] = sb.toString();
        }
        batches = new VectorBuffers[BATCHES][];
        for (int b = 0; b < BATCHES; b++) {
            String[] col = new String[N];
            for (int i = 0; i < N; i++) {
                col[i] = values[rnd.nextInt(distinct)];
            }
            batches[b] = new VectorBuffers[] {ArrowLayout.ofStrings(arena, col)};
        }
        ids = new int[N];
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    /**
     * Assigns all batches into a fresh table; the reported rate is rows per
     * millisecond.
     */
    @Benchmark
    @OperationsPerInvocation(N * BATCHES)
    public int assignPlainStrings() {
        GroupKeyTable table = new GroupKeyTable(new VecType[] {VecType.UTF8});
        int groups = 0;
        for (VectorBuffers[] batch : batches) {
            groups = table.assign(batch, N, ids);
        }
        return groups;
    }
}
