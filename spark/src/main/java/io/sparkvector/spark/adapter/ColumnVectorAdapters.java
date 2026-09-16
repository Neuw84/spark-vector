package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.spark.arrow.ArrowVectorBuffers;
import io.sparkvector.spark.arrow.BorrowedColumnVector;
import io.sparkvector.spark.arrow.VectorArrowColumnVector;
import io.sparkvector.spark.arrow.VectorDecimalColumnVector;
import io.sparkvector.spark.arrow.VectorDictionaryColumnVector;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.apache.spark.sql.vectorized.ColumnVector;

/**
 * Turns any Spark {@link ColumnVector} into {@link VectorBuffers}.
 *
 * <p>Zero-copy adapters are tried first (our own Arrow-backed vectors, plus any registered
 * adapter such as the Comet one, which is loaded reflectively so this jar has no Comet
 * dependency). Anything else is copied into the per-batch {@code Arena}.
 */
public final class ColumnVectorAdapters {

  /**
   * Zero-copy adapter for a foreign columnar vector class. Returns {@code null} to decline. The
   * scratch arena lives as long as the batch is being evaluated; adapters that need small side
   * buffers (a validity bitmap derived from a foreign nullability structure, a decoded dictionary)
   * allocate them there and still hand the column's own data buffers over without a copy.
   */
  @FunctionalInterface
  public interface Adapter {
    VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch);
  }

  private static final List<Adapter> ADAPTERS = new CopyOnWriteArrayList<>();

  static {
    // Comet's and Iceberg's scan vectors are read zero-copy when their jars are on the classpath
    // (executor side); registration is a no-op otherwise.
    io.sparkvector.spark.comet.CometVectorAdapter.tryRegister();
    io.sparkvector.spark.iceberg.IcebergVectorAdapter.tryRegister();
  }

  private ColumnVectorAdapters() {}

  public static void register(Adapter adapter) {
    ADAPTERS.add(adapter);
  }

  public static VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch) {
    if (cv instanceof BorrowedColumnVector b) {
      return adapt(b.inner(), numRows, scratch);
    }
    if (cv instanceof VectorArrowColumnVector v) {
      return ArrowVectorBuffers.forRead(v.getValueVector());
    }
    if (cv instanceof VectorDictionaryColumnVector d) {
      return d.buffers();
    }
    if (cv instanceof VectorDecimalColumnVector d) {
      return ArrowVectorBuffers.forRead(d.vector());
    }
    for (Adapter adapter : ADAPTERS) {
      VectorBuffers vb = adapter.adapt(cv, numRows, scratch);
      if (vb != null) {
        return vb;
      }
    }
    return SparkColumnVectorBuffers.copy(cv, numRows, scratch);
  }

  /** True if {@link #adapt} would not need to copy. */
  public static boolean isZeroCopy(ColumnVector cv, int numRows) {
    if (cv instanceof BorrowedColumnVector b) {
      return isZeroCopy(b.inner(), numRows);
    }
    if (cv instanceof VectorArrowColumnVector || cv instanceof VectorDictionaryColumnVector
        || cv instanceof VectorDecimalColumnVector) {
      return true;
    }
    try (Arena scratch = Arena.ofConfined()) {
      for (Adapter adapter : ADAPTERS) {
        if (adapter.adapt(cv, numRows, scratch) != null) {
          return true;
        }
      }
    }
    return false;
  }
}
