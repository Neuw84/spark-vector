package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Set;
import jdk.incubator.vector.VectorMask;
import org.junit.jupiter.api.Test;

/** The platform probe (#283) and the equivalence of the two mask constructions it chooses between. */
class PlatformTest {

  @Test
  void probeNamesAKnownPlatformAndItsConstantsAgree() {
    Set<String> known = Set.of(Platform.NEON, Platform.SVE, Platform.AVX2, Platform.AVX512, "unknown");
    assertTrue(known.contains(Platform.NAME), Platform.NAME);
    boolean predicated = Platform.NAME.equals(Platform.AVX512) || Platform.NAME.equals(Platform.SVE);
    assertEquals(predicated, Platform.MASK_REGISTERS);
    assertEquals(predicated, Platform.NATIVE_COMPRESS);
    String arch = System.getProperty("os.arch", "");
    if (System.getProperty("sparkvector.platform") == null && (arch.equals("amd64") || arch.equals("x86_64"))) {
      // HotSpot on x86-64 always has some AVX level; the probe must not fall through to "unknown".
      assertTrue(Platform.NAME.equals(Platform.AVX2) || Platform.NAME.equals(Platform.AVX512), Platform.NAME);
      assertTrue(Platform.MAX_VECTOR_BYTES >= 16, "MaxVectorSize " + Platform.MAX_VECTOR_BYTES);
    }
  }

  @Test
  void bothMaskFormsAgreeOnEveryLanePattern() {
    int longLanes = Species.L.length();
    for (long bits = 0; bits < (1L << longLanes); bits++) {
      VectorMask<Long> expected = VectorMask.fromLong(Species.L, bits);
      assertEquals(expected.toLong(), AggKernels.maskLBroadcast(bits).toLong(), "long lanes " + bits);
      assertEquals(expected.toLong(), AggKernels.maskL(bits).toLong(), "maskL " + bits);
      assertEquals(bits, AggKernels.maskD(bits).toLong(), "maskD " + bits);
    }
    int intLanes = Species.I.length();
    long step = intLanes > 8 ? 4099 : 1; // every pattern for 8 lanes, a dense sample of the 65536 for 16
    for (long bits = 0; bits < (1L << intLanes); bits += step) {
      assertEquals(bits, AggKernels.maskIBroadcast(bits).toLong(), "int lanes " + bits);
      assertEquals(bits, AggKernels.maskI(bits).toLong(), "maskI " + bits);
    }
    int halfLanes = Species.IH.length();
    for (long bits = 0; bits < (1L << halfLanes); bits++) {
      assertEquals(bits, AggKernels.maskIHBroadcast(bits).toLong(), "half int lanes " + bits);
      assertEquals(bits, AggKernels.maskIH(bits).toLong(), "maskIH " + bits);
    }
  }
}
