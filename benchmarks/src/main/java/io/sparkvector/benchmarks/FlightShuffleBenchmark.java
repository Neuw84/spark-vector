package io.sparkvector.benchmarks;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.spark.sql.vector.bench.FlightBench;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.openjdk.jmh.annotations.AuxCounters;
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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

/**
 * The shuffle transport end to end, in one JVM: two Arrow Flight servers (two
 * "executors") serving real map files, a reduce task fetching its partition
 * range from both -- see {@link FlightBench}. One operation is one reduce task.
 * The aux counters (rows, batches, bytes per iteration) turn the time into
 * rows/s and MB/s, and rows / batches is the mean batch handed to the operators
 * (how well the reader coalesces small blocks, #411/#416).
 *
 * <ul>
 *   <li>{@code reduceTask}: the real path -- one {@code FlightBlockStream} per
 *       server, batches decoded.
 *   <li>{@code reduceTasksConcurrent}: the same under eight reducer threads
 *       against the same two servers, each on its own partition: where the
 *       server pool, chunk size and gRPC flow control contend.
 *   <li>{@code rawTransport}: the same tickets, bytes consumed undecoded -- the
 *       transport alone, so {@code reduceTask - rawTransport} is the decode
 *       side.
 *   <li>{@code localRead}: the same blocks from the files with no Flight -- the
 *       floor.
 * </ul>
 *
 * Knobs: {@code partitions} sets the block size (250k rows per map at 1000
 * partitions is ~250-row blocks, the #416 regime), {@code rangeWidth} how many
 * consecutive partitions a task reads (AQE coalescing), {@code serverThreads}
 * the Flight executor pool, {@code chunkBytes} the bytes per gRPC message,
 * {@code strings} which string columns ride along. The reader's coalescing
 * threshold is the system property {@code
 * sparkvector.shuffle.reader.coalesceRows}, swept per fork with {@code
 * -jvmArgsAppend}. The default matrix is small on purpose; sweep with {@code
 * -p}.
 *
 * <pre>
 * java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
 *      -jar benchmarks/target/benchmarks.jar FlightShuffle -p partitions=1000 -p strings=mixed -rf json
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                    "--add-opens=java.base/java.nio=ALL-UNNAMED", "-Darrow.memory.debug.allocator=false"})
@State(Scope.Benchmark)
public class FlightShuffleBenchmark {

    @Param({"200", "1000"})
    public int partitions;

    @Param({"1", "5"})
    public int rangeWidth;

    @Param({"zstd"})
    public String compression;

    @Param({"8"})
    public int serverThreads;

    @Param({"4194304"})
    public int chunkBytes;

    @Param({"mixed"})
    public String strings;

    /**
     * Map tasks per server and rows per map: 16 x 250k x 2 servers = 8M rows,
     * ~0.5 GB of data.
     */
    @Param({"16"})
    public int maps;

    @Param({"250000"})
    public int rowsPerMap;

    private FlightBench.Fixture fixture;
    private final AtomicInteger nextReduce = new AtomicInteger();

    /**
     * Per-iteration totals beside the time: rows and batches give the mean rows
     * per batch handed to the operators (derive it as rows / batches -- a
     * derived counter would be summed across threads), bytes the transport
     * volume.
     */
    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class Counters {
        public long rows;
        public long batches;
        public long bytes;
    }

    @Setup(Level.Trial)
    public void setUp() {
        fixture = new FlightBench.Fixture(partitions, maps, rowsPerMap, strings, compression, serverThreads,
                chunkBytes);
        fixture.verify();
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        fixture.close();
    }

    /**
     * The next partition range: every task reads a different one, so the page
     * cache favours none.
     */
    private int reduce() {
        int slots = Math.max(1, partitions / rangeWidth);
        return (nextReduce.getAndIncrement() % slots) * rangeWidth;
    }

    /** Reads one value per column so the decode is not dead code. */
    private static long touch(ColumnarBatch b) {
        long h = 0;
        int last = b.numRows() - 1;
        if (last < 0) {
            return 0;
        }
        for (int c = 0; c < b.numCols(); c++) {
            if (!b.column(c).isNullAt(last)) {
                switch (b.column(c)
                         .dataType()
                         .typeName()) {
                    case "long":
                        h += b.column(c).getLong(last);
                        break;
                    case "integer":
                        h += b.column(c).getInt(last);
                        break;
                    case "double":
                        h += Double.doubleToLongBits(b.column(c)
                                .getDouble(last));
                        break;
                    default:
                        h += b.column(c)
                              .getUTF8String(last)
                              .numBytes();
                }
            }
        }
        return h;
    }

    @Benchmark
    public long reduceTask(Counters counters) {
        final long[] h = {0};
        long rows = fixture.reduceTask(reduce(), rangeWidth,
                b -> {
                    counters.batches++;
                    h[0] += touch(b);
                });
        counters.rows += rows;
        return h[0] + rows;
    }

    @Benchmark
    @Threads(8)
    public long reduceTasksConcurrent(Counters counters) {
        return reduceTask(counters);
    }

    @Benchmark
    public long rawTransport(Counters counters) {
        long bytes = fixture.rawTransport(reduce(), rangeWidth);
        counters.bytes += bytes;
        return bytes;
    }

    @Benchmark
    public long localRead(Counters counters) {
        final long[] h = {0};
        long rows = fixture.localRead(reduce(), rangeWidth,
                b -> {
                    counters.batches++;
                    h[0] += touch(b);
                });
        counters.rows += rows;
        return h[0] + rows;
    }
}
