package io.sparkvector.kernels;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * The vector species every kernel uses. By default the platform's preferred (widest native) shape:
 * 128 bits on NEON, 256 on AVX2, 512 on AVX-512. {@code -Dsparkvector.vectorBits=128|256|512}
 * forces a shape; wider-than-native shapes run emulated (slowly) but exercise the lane counts of
 * the other platforms, which is how the AVX-512 code paths are tested on a laptop.
 */
public final class Species {

  public static final VectorShape SHAPE = shape();

  public static final VectorSpecies<Integer> I = VectorSpecies.of(int.class, SHAPE);
  public static final VectorSpecies<Long> L = VectorSpecies.of(long.class, SHAPE);
  public static final VectorSpecies<Double> D = VectorSpecies.of(double.class, SHAPE);

  /** Int species with as many lanes as {@link #L} / {@link #D}, for widening conversions. */
  public static final VectorSpecies<Integer> IH =
      VectorSpecies.of(int.class, VectorShape.forBitSize(SHAPE.vectorBitSize() / 2));

  /** Lanes of {@link #D}: 2 on NEON, 4 on AVX2, 8 on AVX-512. */
  public static final int DOUBLE_LANES = D.length();

  private Species() {}

  private static VectorShape shape() {
    int bits = Integer.getInteger("sparkvector.vectorBits", 0);
    if (bits == 0) {
      return DoubleVector.SPECIES_PREFERRED.vectorShape();
    }
    if (bits != 128 && bits != 256 && bits != 512) {
      throw new IllegalArgumentException("sparkvector.vectorBits must be 128, 256 or 512, got " + bits);
    }
    return VectorShape.forBitSize(bits);
  }

  /** Sanity: the three species agree on width. */
  static {
    if (I.vectorBitSize() != L.vectorBitSize() || L.vectorBitSize() != D.vectorBitSize()) {
      throw new IllegalStateException("species width mismatch");
    }
    // Reference the classes so the incubator module is loaded eagerly with a clear error.
    IntVector.zero(I);
    LongVector.zero(L);
  }
}
