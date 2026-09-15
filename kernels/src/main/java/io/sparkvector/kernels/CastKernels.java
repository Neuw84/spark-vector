package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Widening numeric casts (INT32 to INT64/FLOAT64, INT64 to FLOAT64). These never fail or lose
 * integer precision beyond what Spark itself does (long to double rounds), so they are valid in
 * both legacy and ANSI mode. Validity is unchanged and is shared by the caller.
 *
 * <p>Widening halves the lane count, so the int source is loaded with a species of half the
 * preferred bit size and converted with {@code convertShape} into the full-size destination.
 */
public final class CastKernels {

  static final VectorSpecies<Long> L = LongVector.SPECIES_PREFERRED;
  static final VectorSpecies<Double> D = DoubleVector.SPECIES_PREFERRED;
  /** Int species with as many lanes as {@link #D} / {@link #L}. */
  static final VectorSpecies<Integer> IH =
      VectorSpecies.of(int.class, VectorShape.forBitSize(D.vectorBitSize() / 2));
  static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

  private CastKernels() {}

  /** Casts {@code a} to {@code target}, writing values into {@code out}. */
  public static void cast(VectorBuffers a, VecType target, MemorySegment out) {
    int n = a.length();
    switch (a.type()) {
      case INT32 -> {
        switch (target) {
          case INT64 -> i32ToI64(a.data(), n, out);
          case FLOAT64 -> i32ToF64(a.data(), n, out);
          default -> throw unsupported(a.type(), target);
        }
      }
      case INT64 -> {
        if (target != VecType.FLOAT64) {
          throw unsupported(a.type(), target);
        }
        i64ToF64(a.data(), n, out);
      }
      default -> throw unsupported(a.type(), target);
    }
  }

  public static boolean isSupported(VecType from, VecType to) {
    return (from == VecType.INT32 && (to == VecType.INT64 || to == VecType.FLOAT64))
        || (from == VecType.INT64 && to == VecType.FLOAT64);
  }

  private static IllegalArgumentException unsupported(VecType from, VecType to) {
    return new IllegalArgumentException("unsupported cast " + from + " -> " + to);
  }

  static void i32ToI64(MemorySegment a, int n, MemorySegment out) {
    int lanes = L.length(), i = 0;
    for (; i + lanes <= n; i += lanes) {
      IntVector.fromMemorySegment(IH, a, (long) i << 2, LE)
          .convertShape(VectorOperators.I2L, L, 0)
          .reinterpretAsLongs()
          .intoMemorySegment(out, (long) i << 3, LE);
    }
    if (i < n) {
      VectorMask<Integer> mi = IH.indexInRange(i, n);
      VectorMask<Long> ml = L.indexInRange(i, n);
      IntVector.fromMemorySegment(IH, a, (long) i << 2, LE, mi)
          .convertShape(VectorOperators.I2L, L, 0)
          .reinterpretAsLongs()
          .intoMemorySegment(out, (long) i << 3, LE, ml);
    }
  }

  static void i32ToF64(MemorySegment a, int n, MemorySegment out) {
    int lanes = D.length(), i = 0;
    for (; i + lanes <= n; i += lanes) {
      IntVector.fromMemorySegment(IH, a, (long) i << 2, LE)
          .convertShape(VectorOperators.I2D, D, 0)
          .reinterpretAsDoubles()
          .intoMemorySegment(out, (long) i << 3, LE);
    }
    if (i < n) {
      VectorMask<Integer> mi = IH.indexInRange(i, n);
      VectorMask<Double> md = D.indexInRange(i, n);
      IntVector.fromMemorySegment(IH, a, (long) i << 2, LE, mi)
          .convertShape(VectorOperators.I2D, D, 0)
          .reinterpretAsDoubles()
          .intoMemorySegment(out, (long) i << 3, LE, md);
    }
  }

  static void i64ToF64(MemorySegment a, int n, MemorySegment out) {
    int lanes = D.length(), i = 0;
    for (; i + lanes <= n; i += lanes) {
      LongVector.fromMemorySegment(L, a, (long) i << 3, LE)
          .convertShape(VectorOperators.L2D, D, 0)
          .reinterpretAsDoubles()
          .intoMemorySegment(out, (long) i << 3, LE);
    }
    if (i < n) {
      VectorMask<Long> ml = L.indexInRange(i, n);
      VectorMask<Double> md = D.indexInRange(i, n);
      LongVector.fromMemorySegment(L, a, (long) i << 3, LE, ml)
          .convertShape(VectorOperators.L2D, D, 0)
          .reinterpretAsDoubles()
          .intoMemorySegment(out, (long) i << 3, LE, md);
    }
  }
}
