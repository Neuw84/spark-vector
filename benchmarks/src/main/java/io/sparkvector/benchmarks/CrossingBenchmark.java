package io.sparkvector.benchmarks;

import java.lang.foreign.Arena;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.sparkvector.spark.adapter.ColumnVectorAdapters;
import io.sparkvector.spark.arrow.VectorAllocators;
import io.sparkvector.spark.arrow.VectorArrowColumnVector;
import io.sparkvector.spark.arrow.VectorDecimalColumnVector;
import io.sparkvector.spark.arrow.VectorDictionaryColumnVector;
import io.sparkvector.spark.comet.CometBatchBridge;
import io.sparkvector.spark.comet.CometVectorAdapter;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
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
import org.openjdk.jmh.infra.Blackhole;

/**
 * The crossing cost between our batches and Comet's (#279), both directions, so
 * a candidate swap of one operator for Comet's can be judged as "operator delta
 * minus twice the crossing". Into Comet: {@link CometBatchBridge#convert} --
 * our Arrow vectors exported through the C Data interface (buffers pointed at,
 * retained; a dictionary decoded; an INT64 decimal widened to 128 bits) and
 * imported by Comet's {@code ArrowImporter}, then the imported vectors closed
 * (the release callback). Back: {@link ColumnVectorAdapters#adapt} over the
 * Comet vectors that conversion produced, through the registered {@link
 * CometVectorAdapter} (zero-copy but for large strings and, before #257,
 * 128-bit decimals). The unit is microseconds per batch; the docs quote ns per
 * row per column.
 *
 * <pre>
 * java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
 *   -cp benchmarks/target/benchmarks.jar:$COMET_JAR org.openjdk.jmh.Main CrossingBenchmark \
 *   -p rows=8192 -p columns=8 -p lane=INT64 -wi 3 -i 5 -w 1 -r 1 -f 1
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
            "-Xmx4g",
            // The opens Arrow's memory module and Spark need on JDK 25, as the run scripts pass them.
            "--sun-misc-unsafe-memory-access=allow",
            "-XX:+IgnoreUnrecognizedVMOptions",
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            "--add-opens=java.base/java.io=ALL-UNNAMED",
            "--add-opens=java.base/java.net=ALL-UNNAMED",
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/java.util=ALL-UNNAMED",
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
            "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
        })
@State(Scope.Thread)
public class CrossingBenchmark {

    @Param({"4096", "8192"})
    int rows;

    @Param({"4", "8", "16"})
    int columns;

    /**
     * INT64 (no validity buffer), INT64_NULLS (10% nulls), FLOAT64, UTF8 (12-40
     * bytes), UTF8_DICT (dictionary of 1000 values, decoded when crossing into
     * Comet), DECIMAL64 (decimal(12,2) on the INT64 lane, widened to 128 bits
     * when crossing into Comet).
     */
    @Param({"INT64", "INT64_NULLS", "FLOAT64", "UTF8", "UTF8_DICT", "DECIMAL64"})
    String lane;

    BufferAllocator allocator;
    ColumnarBatch ours;
    String[] names;
    DataType[] types;
    CometBatchBridge bridge;

    /** Comet's vectors over our buffers, converted once for the way back. */
    ColumnarBatch theirs;

    @Setup(Level.Trial)
    public void setup() {
        bridge = CometBatchBridge.tryCreate();
        if (bridge == null) {
            throw new IllegalStateException("Comet is not on the classpath: add the Comet jar to -cp");
        }
        if (!CometVectorAdapter.tryRegister()) {
            throw new IllegalStateException("Comet's vector API is not the expected one");
        }
        allocator = VectorAllocators.newChild("crossing-benchmark");
        Random rnd = new Random(42);
        ColumnVector[] cols = new ColumnVector[columns];
        names = new String[columns];
        types = new DataType[columns];
        for (int c = 0; c < columns; c++) {
            names[c] = "c" + c;
            cols[c] = column(rnd, names[c]);
            types[c] = cols[c].dataType();
        }
        ours = new ColumnarBatch(cols, rows);
        theirs = bridge.convert(ours, names, types);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        theirs.close();
        ours.close();
        bridge.releaseOutstanding();
        allocator.close();
    }

    private ColumnVector column(Random rnd, String name) {
        switch (lane) {
            case "INT64", "INT64_NULLS" -> {
                BigIntVector v = new BigIntVector(name, allocator);
                v.allocateNew(rows);
                boolean nulls = lane.endsWith("_NULLS");
                for (int i = 0; i < rows; i++) {
                    if (nulls && rnd.nextInt(10) == 0) {
                        v.setNull(i);
                    } else {
                        v.set(i, rnd.nextLong());
                    }
                }
                v.setValueCount(rows);
                return new VectorArrowColumnVector(v);
            }
            case "FLOAT64" -> {
                Float8Vector v = new Float8Vector(name, allocator);
                v.allocateNew(rows);
                for (int i = 0; i < rows; i++) {
                    v.set(i, rnd.nextDouble() * 1e6);
                }
                v.setValueCount(rows);
                return new VectorArrowColumnVector(v);
            }
            case "UTF8" -> {
                VarCharVector v = new VarCharVector(name, allocator);
                v.allocateNew(rows * 40L, rows);
                for (int i = 0; i < rows; i++) {
                    v.set(i, randomString(rnd, 12 + rnd.nextInt(29)).getBytes(StandardCharsets.UTF_8));
                }
                v.setValueCount(rows);
                return new VectorArrowColumnVector(v);
            }
            case "UTF8_DICT" -> {
                VarCharVector dict = new VarCharVector(name + "-dict", allocator);
                dict.allocateNew(1000 * 24L, 1000);
                for (int d = 0; d < 1000; d++) {
                    dict.set(d, ("value-" + Long.toHexString(rnd.nextLong())).getBytes(StandardCharsets.UTF_8));
                }
                dict.setValueCount(1000);
                IntVector idx = new IntVector(name, allocator);
                idx.allocateNew(rows);
                for (int i = 0; i < rows; i++) {
                    idx.set(i, rnd.nextInt(1000));
                }
                idx.setValueCount(rows);
                return new VectorDictionaryColumnVector(idx, dict);
            }
            case "DECIMAL64" -> {
                BigIntVector v = new BigIntVector(name, allocator);
                v.allocateNew(rows);
                for (int i = 0; i < rows; i++) {
                    v.set(i, rnd.nextLong() % 1_000_000_000_000L);
                }
                v.setValueCount(rows);
                return new VectorDecimalColumnVector(v, DataTypes.createDecimalType(12, 2));
            }
            default -> throw new IllegalArgumentException(lane);
        }
    }

    private static String randomString(Random rnd, int length) {
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            sb.append((char) ('a' + rnd.nextInt(26)));
        }
        return sb.toString();
    }

    /**
     * Ours into Comet: export through C Data, Comet's import, then the imported
     * vectors released.
     */
    @Benchmark
    public void intoComet(Blackhole bh) {
        ColumnarBatch converted = bridge.convert(ours, names, types);
        bh.consume(converted.numRows());
        converted.close();
        bridge.releaseOutstanding();
    }

    /** Comet's vectors read as our buffers, one adapter call per column. */
    @Benchmark
    public void backFromComet(Blackhole bh) {
        try (Arena scratch = Arena.ofConfined()) {
            for (int c = 0; c < columns; c++) {
                bh.consume(ColumnVectorAdapters.adapt(theirs.column(c), rows, scratch));
            }
        }
    }
}
