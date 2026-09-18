package io.sparkvector.benchmarks;

import io.sparkvector.kernels.ArrowLayout;
import io.sparkvector.kernels.SegmentVectorBuffers;
import io.sparkvector.kernels.SortKernels;
import io.sparkvector.kernels.VectorBuffers;
import java.lang.foreign.Arena;
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

/**
 * The index sort (#285): the radix kernel against the previous one ({@link LegacySortKernels}, one
 * {@code Arrays.sort(long[])} per 32-bit key pass) over the key types, key counts, null shares and
 * input orders the issue lists. The unit is milliseconds per sort of {@code rows} rows; the number
 * the docs quote is ns/row at 10M random INT64 rows.
 *
 * <pre>
 * java --add-modules=jdk.incubator.vector -jar benchmarks/target/benchmarks.jar SortBenchmark \
 *   -p rows=1000000 -p key=INT64 -p order=random -wi 2 -i 3 -w 1 -r 1 -f 1
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xmx8g"})
@State(Scope.Thread)
public class SortBenchmark {

  @Param({"1000000", "10000000"})
  int rows;

  /** INT32, INT64, FLOAT64, UTF8_SHORT (up to 8 bytes), UTF8_LONG (12-40 bytes), UTF8_DICT (dictionary-encoded, 1000 values). */
  @Param({"INT32", "INT64", "FLOAT64", "UTF8_SHORT", "UTF8_LONG", "UTF8_DICT"})
  String key;

  /** One key, or three (the named key twice around a low-cardinality int). */
  @Param({"1", "3"})
  int keys;

  /** Share of null rows. */
  @Param({"0", "0.1"})
  double nulls;

  /** random, presorted, reverse, lowcard (100 distinct values). */
  @Param({"random", "presorted", "reverse", "lowcard"})
  String order;

  Arena arena;
  VectorBuffers[] keyColumns;
  boolean[] ascending;
  boolean[] nullsFirst;

  @Setup(Level.Trial)
  public void setup() {
    arena = Arena.ofShared();
    Random rnd = new Random(7);
    boolean[] nullFlags = null;
    if (nulls > 0) {
      nullFlags = new boolean[rows];
      for (int i = 0; i < rows; i++) {
        nullFlags[i] = rnd.nextDouble() < nulls;
      }
    }
    // The value's rank in the requested order, from which every key type is derived.
    long[] shape = new long[rows];
    for (int i = 0; i < rows; i++) {
      shape[i] =
          switch (order) {
            case "presorted" -> i;
            case "reverse" -> rows - 1 - i;
            case "lowcard" -> rnd.nextInt(100);
            default -> rnd.nextLong();
          };
    }
    VectorBuffers main = column(key, shape, nullFlags, rnd);
    if (keys == 1) {
      keyColumns = new VectorBuffers[] {main};
      ascending = new boolean[] {true};
      nullsFirst = new boolean[] {true};
    } else {
      int[] small = new int[rows];
      for (int i = 0; i < rows; i++) {
        small[i] = rnd.nextInt(4);
      }
      keyColumns = new VectorBuffers[] {main, ArrowLayout.ofInts(arena, small, null), column(key, shape, nullFlags, rnd)};
      ascending = new boolean[] {true, false, true};
      nullsFirst = new boolean[] {true, false, true};
    }
  }

  private VectorBuffers column(String type, long[] shape, boolean[] nullFlags, Random rnd) {
    switch (type) {
      case "INT32" -> {
        int[] v = new int[rows];
        for (int i = 0; i < rows; i++) {
          v[i] = (int) shape[i];
        }
        return ArrowLayout.ofInts(arena, v, nullFlags);
      }
      case "INT64" -> {
        return ArrowLayout.ofLongs(arena, shape, nullFlags);
      }
      case "FLOAT64" -> {
        double[] v = new double[rows];
        for (int i = 0; i < rows; i++) {
          v[i] = shape[i] * 1.5;
        }
        return ArrowLayout.ofDoubles(arena, v, nullFlags);
      }
      case "UTF8_SHORT" -> {
        return ArrowLayout.ofStrings(arena, strings(shape, nullFlags, 8));
      }
      case "UTF8_LONG" -> {
        return ArrowLayout.ofStrings(arena, strings(shape, nullFlags, 40));
      }
      case "UTF8_DICT" -> {
        String[] dict = new String[1000];
        for (int d = 0; d < dict.length; d++) {
          dict[d] = "value-" + Long.toHexString(rnd.nextLong());
        }
        int[] ids = new int[rows];
        for (int i = 0; i < rows; i++) {
          ids[i] = (int) Math.floorMod(shape[i], (long) dict.length);
        }
        SegmentVectorBuffers idx = ArrowLayout.ofInts(arena, ids, nullFlags);
        return SegmentVectorBuffers.dictionaryUtf8(rows, idx.validity(), idx.data(), ArrowLayout.ofStrings(arena, dict));
      }
      default -> throw new IllegalArgumentException(type);
    }
  }

  /** Strings ordered like the shape value: its hex digits, zero-padded, with a tail to reach {@code length}. */
  private static String[] strings(long[] shape, boolean[] nullFlags, int length) {
    String[] v = new String[shape.length];
    for (int i = 0; i < shape.length; i++) {
      if (nullFlags != null && nullFlags[i]) {
        continue;
      }
      String hex = Long.toHexString(shape[i]);
      String s = length <= 8 ? hex.substring(0, Math.min(hex.length(), length)) : hex;
      if (length > 8) {
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < length) {
          sb.append('x');
        }
        s = sb.toString();
      }
      v[i] = s;
    }
    return v;
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    arena.close();
  }

  @Benchmark
  public int[] radix() {
    return SortKernels.sortIndices(keyColumns, ascending, nullsFirst, rows);
  }

  @Benchmark
  public int[] legacy() {
    return LegacySortKernels.sortIndices(keyColumns, ascending, nullsFirst, rows);
  }
}
