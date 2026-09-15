package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.VectorBuffers;
import io.sparkvector.spark.arrow.ArrowVectorBuffers;
import io.sparkvector.spark.arrow.VectorArrowColumnVector;
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

  /** Zero-copy adapter for a foreign columnar vector class. Returns {@code null} to decline. */
  @FunctionalInterface
  public interface Adapter {
    VectorBuffers adapt(ColumnVector cv, int numRows);
  }

  private static final List<Adapter> ADAPTERS = new CopyOnWriteArrayList<>();

  static {
    // Comet's scan vectors are read zero-copy when Comet is on the classpath (executor side).
    io.sparkvector.spark.comet.CometVectorAdapter.tryRegister();
  }

  private ColumnVectorAdapters() {}

  public static void register(Adapter adapter) {
    ADAPTERS.add(adapter);
  }

  public static VectorBuffers adapt(ColumnVector cv, int numRows, Arena scratch) {
    if (cv instanceof VectorArrowColumnVector v) {
      return ArrowVectorBuffers.forRead(v.getValueVector());
    }
    for (Adapter adapter : ADAPTERS) {
      VectorBuffers vb = adapter.adapt(cv, numRows);
      if (vb != null) {
        return vb;
      }
    }
    return SparkColumnVectorBuffers.copy(cv, numRows, scratch);
  }

  /** True if {@link #adapt} would not need to copy. */
  public static boolean isZeroCopy(ColumnVector cv, int numRows) {
    if (cv instanceof VectorArrowColumnVector) {
      return true;
    }
    for (Adapter adapter : ADAPTERS) {
      if (adapter.adapt(cv, numRows) != null) {
        return true;
      }
    }
    return false;
  }
}
