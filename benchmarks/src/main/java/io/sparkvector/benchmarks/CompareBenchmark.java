package io.sparkvector.benchmarks;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.CompareKernels;
import io.sparkvector.kernels.CompareOp;
import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.kernels.reference.ScalarReference;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import java.util.concurrent.TimeUnit;
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

/** Column-vs-scalar and column-vs-column comparisons, SIMD kernels against the scalar reference. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@OperationsPerInvocation(CompareBenchmark.N)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class CompareBenchmark {

  /** Rows per batch; throughput is reported per element. */
  static final int N = 8192;

  @Param({"INT32", "INT64", "FLOAT64"})
  String type;

  Arena arena;
  VectorBuffers a;
  VectorBuffers b;
  MemorySegment out;
  Number scalar;

  @Setup(Level.Trial)
  public void setup() {
    arena = Arena.ofShared();
    Random rnd = new Random(1);
    switch (type) {
      case "INT32" -> {
        int[] x = new int[N];
        int[] y = new int[N];
        for (int i = 0; i < N; i++) {
          x[i] = rnd.nextInt(1000);
          y[i] = rnd.nextInt(1000);
        }
        a = ArrowLayout.ofInts(arena, x, null);
        b = ArrowLayout.ofInts(arena, y, null);
        scalar = 500;
      }
      case "INT64" -> {
        long[] x = new long[N];
        long[] y = new long[N];
        for (int i = 0; i < N; i++) {
          x[i] = rnd.nextLong(1000);
          y[i] = rnd.nextLong(1000);
        }
        a = ArrowLayout.ofLongs(arena, x, null);
        b = ArrowLayout.ofLongs(arena, y, null);
        scalar = 500L;
      }
      default -> {
        double[] x = new double[N];
        double[] y = new double[N];
        for (int i = 0; i < N; i++) {
          x[i] = rnd.nextDouble() * 1000;
          y[i] = rnd.nextDouble() * 1000;
        }
        a = ArrowLayout.ofDoubles(arena, x, null);
        b = ArrowLayout.ofDoubles(arena, y, null);
        scalar = 500.0;
      }
    }
    out = ArrowLayout.allocateBitmap(arena, N);
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    arena.close();
  }

  @Benchmark
  public MemorySegment scalarLessThan_simd() {
    CompareKernels.compareScalar(a, scalar, CompareOp.LT, out);
    return out;
  }

  @Benchmark
  public MemorySegment scalarLessThan_reference() {
    ScalarReference.compareScalar(a, scalar, CompareOp.LT, out);
    return out;
  }

  @Benchmark
  public MemorySegment columnLessThan_simd() {
    CompareKernels.compare(a, b, CompareOp.LT, out);
    return out;
  }

  @Benchmark
  public MemorySegment columnLessThan_reference() {
    ScalarReference.compare(a, b, CompareOp.LT, out);
    return out;
  }
}
