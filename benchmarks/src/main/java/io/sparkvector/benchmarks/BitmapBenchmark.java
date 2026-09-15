package io.sparkvector.benchmarks;

import io.sparkvector.kernels.Bitmap;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.concurrent.TimeUnit;
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

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class BitmapBenchmark {

  @Param({"4096", "65536"})
  int numBits;

  Arena arena;
  MemorySegment bitmap;

  @Setup(Level.Trial)
  public void setup() {
    arena = Arena.ofShared();
    bitmap = Bitmap.allocate(arena, numBits);
    Random rnd = new Random(7);
    for (long i = 0; i < bitmap.byteSize(); i++) {
      bitmap.set(ValueLayout.JAVA_BYTE, i, (byte) rnd.nextInt(256));
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    arena.close();
  }

  @Benchmark
  public int popcountWords() {
    return Bitmap.popcount(bitmap, numBits);
  }

  @Benchmark
  public int popcountScalarBits() {
    int c = 0;
    for (int i = 0; i < numBits; i++) {
      if (Bitmap.isSet(bitmap, i)) {
        c++;
      }
    }
    return c;
  }
}
