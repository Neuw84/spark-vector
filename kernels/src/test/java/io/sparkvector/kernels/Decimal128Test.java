package io.sparkvector.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigInteger;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** The limb layout round-trips through big-endian bytes and BigInteger for every awkward value. */
class Decimal128Test {

  private static final BigInteger[] EXTREMES = {
    BigInteger.ZERO, BigInteger.ONE, BigInteger.ONE.negate(),
    BigInteger.valueOf(Long.MAX_VALUE), BigInteger.valueOf(Long.MIN_VALUE),
    BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE), BigInteger.valueOf(Long.MIN_VALUE).subtract(BigInteger.ONE),
    TestData.MAX_DECIMAL38, TestData.MAX_DECIMAL38.negate(),
    BigInteger.ONE.shiftLeft(127).negate(), BigInteger.ONE.shiftLeft(127).subtract(BigInteger.ONE),
    BigInteger.valueOf(-128), BigInteger.valueOf(127), BigInteger.valueOf(255), BigInteger.valueOf(-256)
  };

  @Test
  void bigEndianBytesOfAnyLengthBecomeTheSameLimbsAsBigInteger() {
    Random rnd = new Random(257);
    for (int k = 0; k < 20000; k++) {
      BigInteger v = k < EXTREMES.length ? EXTREMES[k] : TestData.randomDecimal128(rnd);
      byte[] be = v.toByteArray(); // minimal two's complement, 1..16 bytes
      assertTrue(be.length <= 16, v.toString());
      long hi = Decimal128.hiFromBigEndian(be, 0, be.length);
      long lo = Decimal128.loFromBigEndian(be, 0, be.length);
      assertEquals(Decimal128.hiOf(v), hi, "hi of " + v);
      assertEquals(Decimal128.loOf(v), lo, "lo of " + v);
      assertEquals(v, Decimal128.toBigInteger(hi, lo), "round trip of " + v);
      // The 16-byte form, as Parquet FIXED_LEN_BYTE_ARRAY(16) stores it, reads the same.
      byte[] wide = Decimal128.toBigEndian(hi, lo);
      assertEquals(v, new BigInteger(wide), "16-byte form of " + v);
      assertEquals(hi, Decimal128.hiFromBigEndian(wide, 0, 16));
      assertEquals(lo, Decimal128.loFromBigEndian(wide, 0, 16));
      // And at an offset inside a larger array, shorter than 16.
      byte[] padded = new byte[be.length + 5];
      System.arraycopy(be, 0, padded, 3, be.length);
      assertEquals(hi, Decimal128.hiFromBigEndian(padded, 3, be.length));
      assertEquals(lo, Decimal128.loFromBigEndian(padded, 3, be.length));
    }
  }

  @Test
  void layoutIsLittleEndianLimbsAndTheReferenceWritesTheSameBytes() {
    Random rnd = new Random(258);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment ours = ArrowLayout.allocateData(arena, VecType.DECIMAL128, 64);
      MemorySegment ref = ArrowLayout.allocateData(arena, VecType.DECIMAL128, 64);
      for (int i = 0; i < 64; i++) {
        BigInteger v = i < EXTREMES.length ? EXTREMES[i] : TestData.randomDecimal128(rnd);
        Decimal128.set(ours, i, Decimal128.hiOf(v), Decimal128.loOf(v));
        io.sparkvector.kernels.reference.ScalarReference.setDecimal128(ref, i, v);
        assertEquals(v, Decimal128.toBigInteger(Decimal128.hi(ours, i), Decimal128.lo(ours, i)));
        // Little-endian: byte 0 is the least significant byte of the low limb.
        assertEquals((byte) v.longValue(), ours.get(java.lang.foreign.ValueLayout.JAVA_BYTE, (long) i << 4), "byte 0 of " + v);
        assertEquals((byte) (v.shiftRight(120).longValue()), ours.get(java.lang.foreign.ValueLayout.JAVA_BYTE, ((long) i << 4) + 15), "byte 15 of " + v);
      }
      assertEquals(0, ours.asSlice(0, 64 * 16).mismatch(ref.asSlice(0, 64 * 16)) == -1 ? 0 : 1, "reference layout differs");
    }
  }

  @Test
  void compareOrdersLikeBigIntegerAndHashIsConsistent() {
    Random rnd = new Random(259);
    for (int k = 0; k < 20000; k++) {
      BigInteger a = k < EXTREMES.length ? EXTREMES[k] : TestData.randomDecimal128(rnd);
      BigInteger b = rnd.nextInt(4) == 0 ? a : TestData.randomDecimal128(rnd);
      int expected = Integer.signum(a.compareTo(b));
      int actual = Integer.signum(Decimal128.compare(Decimal128.hiOf(a), Decimal128.loOf(a), Decimal128.hiOf(b), Decimal128.loOf(b)));
      assertEquals(expected, actual, a + " vs " + b);
      if (a.equals(b)) {
        assertEquals(Decimal128.hash(Decimal128.hiOf(a), Decimal128.loOf(a)), Decimal128.hash(Decimal128.hiOf(b), Decimal128.loOf(b)));
      }
    }
  }
}
