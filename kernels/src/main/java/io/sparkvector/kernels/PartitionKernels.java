package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.math.BigInteger;

/**
 * Partition ids for the columnar shuffle (#288). The hash partitioning is bit-identical to Spark's
 * {@code Pmod(Murmur3Hash(keys, 42), numPartitions)}: every key column mixes into the running hash
 * in key order with Spark's {@code Murmur3_x86_32} routines (a null key leaves the hash unchanged),
 * the seed is Spark's 42, and the partition is the non-negative remainder. AQE's exchange reuse and
 * the co-partitioning Spark assumes between our exchange and its own depend on this equality, which
 * {@code VectorShuffleSuite} checks against Spark's expression over every lane type.
 *
 * <p>The per-type rules are Spark's {@code HashExpression.computeHash}: int and date as
 * {@code hashInt}; long, timestamp and a decimal of at most 18 digits (its unscaled value) as
 * {@code hashLong}; a double as the bits of {@code normalize(v)} ({@code -0.0} is {@code 0.0}, a
 * NaN is the canonical NaN); a boolean as {@code hashInt(1 / 0)}; a string as
 * {@code hashUnsafeBytes} over its UTF-8 bytes (Spark's non-standard tail, one byte at a time, each
 * sign-extended); a wider decimal as {@code hashUnsafeBytes} over the big-endian two's-complement
 * bytes of its unscaled value.
 */
public final class PartitionKernels {

  private PartitionKernels() {}

  /** Spark's seed for {@code Murmur3Hash} ({@code HashPartitioning.partitionIdExpression}). */
  public static final int SPARK_SEED = 42;

  /** How Spark hashes the values a lane carries; the lane alone does not say (long vs decimal). */
  public enum KeyKind {
    /** IntegerType, DateType: {@code hashInt}. */
    INT,
    /** LongType, TimestampType(NTZ), DecimalType with precision <= 18: {@code hashLong}. */
    LONG,
    /** DoubleType: {@code hashLong(doubleToLongBits(normalize(v)))}. */
    DOUBLE,
    /** BooleanType: {@code hashInt(v ? 1 : 0)}. */
    BOOL,
    /** StringType: {@code hashUnsafeBytes(utf8)}. */
    UTF8,
    /** DecimalType with precision > 18: {@code hashUnsafeBytes(unscaled.toByteArray())}. */
    DECIMAL128
  }

  private static final int C1 = 0xcc9e2d51;
  private static final int C2 = 0x1b873593;

  // --- Spark's Murmur3_x86_32, kept verbatim in shape so the equality is auditable line by line.

  static int mixK1(int k1) {
    k1 *= C1;
    k1 = Integer.rotateLeft(k1, 15);
    k1 *= C2;
    return k1;
  }

  static int mixH1(int h1, int k1) {
    h1 ^= k1;
    h1 = Integer.rotateLeft(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
    return h1;
  }

  static int fmix(int h1, int length) {
    h1 ^= length;
    h1 ^= h1 >>> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >>> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >>> 16;
    return h1;
  }

  /** {@code Murmur3_x86_32.hashInt}. */
  public static int hashInt(int input, int seed) {
    int k1 = mixK1(input);
    int h1 = mixH1(seed, k1);
    return fmix(h1, 4);
  }

  /** {@code Murmur3_x86_32.hashLong}: the low word, then the high word. */
  public static int hashLong(long input, int seed) {
    int low = (int) input;
    int high = (int) (input >>> 32);
    int k1 = mixK1(low);
    int h1 = mixH1(seed, k1);
    k1 = mixK1(high);
    h1 = mixH1(h1, k1);
    return fmix(h1, 8);
  }

  /**
   * {@code Murmur3_x86_32.hashUnsafeBytes}: little-endian 4-byte words, then the tail one signed byte
   * at a time (Spark's non-standard tail, which {@code Murmur3Hash} uses for strings and binary).
   */
  public static int hashUnsafeBytes(MemorySegment base, long offset, int lengthInBytes, int seed) {
    int lengthAligned = lengthInBytes - lengthInBytes % 4;
    int h1 = seed;
    for (int i = 0; i < lengthAligned; i += 4) {
      int halfWord = base.get(VectorBuffers.LE_INT, offset + i);
      int k1 = mixK1(halfWord);
      h1 = mixH1(h1, k1);
    }
    for (int i = lengthAligned; i < lengthInBytes; i++) {
      int halfWord = base.get(ValueLayout.JAVA_BYTE, offset + i);
      int k1 = mixK1(halfWord);
      h1 = mixH1(h1, k1);
    }
    return fmix(h1, lengthInBytes);
  }

  /** {@link #hashUnsafeBytes(MemorySegment, long, int, int)} over a heap array. */
  public static int hashUnsafeBytes(byte[] bytes, int seed) {
    return hashUnsafeBytes(MemorySegment.ofArray(bytes), 0, bytes.length, seed);
  }

  /** Spark's {@code normalize} for doubles before hashing: {@code -0.0} to {@code 0.0}, one NaN. */
  public static long doubleBits(double v) {
    if (v == 0.0d) {
      return 0L;
    }
    if (Double.isNaN(v)) {
      return Double.doubleToLongBits(Double.NaN);
    }
    return Double.doubleToLongBits(v);
  }

  // --- Column-wise mixing.

  /** Fills {@code hashes[0..n)} with the seed. */
  public static void init(int[] hashes, int n, int seed) {
    java.util.Arrays.fill(hashes, 0, n, seed);
  }

  /**
   * Mixes one key column into the running hashes: {@code hashes[i] = hashX(value_i, hashes[i])} for
   * every valid row; a null row leaves its hash as it was, which is Spark's rule.
   */
  public static void mixColumn(VectorBuffers col, KeyKind kind, int[] hashes, int n) {
    MemorySegment validity = col.validity();
    switch (kind) {
      case INT -> {
        MemorySegment d = col.data();
        for (int i = 0; i < n; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            hashes[i] = hashInt(d.get(VectorBuffers.LE_INT, (long) i << 2), hashes[i]);
          }
        }
      }
      case LONG -> {
        MemorySegment d = col.data();
        for (int i = 0; i < n; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            hashes[i] = hashLong(d.get(VectorBuffers.LE_LONG, (long) i << 3), hashes[i]);
          }
        }
      }
      case DOUBLE -> {
        MemorySegment d = col.data();
        for (int i = 0; i < n; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            hashes[i] = hashLong(doubleBits(d.get(VectorBuffers.LE_DOUBLE, (long) i << 3)), hashes[i]);
          }
        }
      }
      case BOOL -> {
        for (int i = 0; i < n; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            hashes[i] = hashInt(col.getBoolean(i) ? 1 : 0, hashes[i]);
          }
        }
      }
      case UTF8 -> mixUtf8(col, hashes, n, validity);
      case DECIMAL128 -> {
        MemorySegment d = col.data();
        for (int i = 0; i < n; i++) {
          if (validity == null || Bitmap.isSet(validity, i)) {
            BigInteger unscaled = Decimal128.toBigInteger(Decimal128.hi(d, i), Decimal128.lo(d, i));
            hashes[i] = hashUnsafeBytes(unscaled.toByteArray(), hashes[i]);
          }
        }
      }
    }
  }

  private static void mixUtf8(VectorBuffers col, int[] hashes, int n, MemorySegment validity) {
    VectorBuffers dict = col.dictionary();
    if (dict != null) {
      // Hash each dictionary entry once with the seed folded in per row? No: the seed differs per row
      // (it is the running hash), so the entry is hashed per row; the win is the offsets/data locality.
      MemorySegment ids = col.data();
      MemorySegment off = dict.offsets();
      MemorySegment data = dict.data();
      for (int i = 0; i < n; i++) {
        if (validity == null || Bitmap.isSet(validity, i)) {
          int id = ids.get(VectorBuffers.LE_INT, (long) i << 2);
          int start = off.get(VectorBuffers.LE_INT, (long) id << 2);
          int end = off.get(VectorBuffers.LE_INT, (long) (id + 1) << 2);
          hashes[i] = hashUnsafeBytes(data, start, end - start, hashes[i]);
        }
      }
      return;
    }
    MemorySegment off = col.offsets();
    MemorySegment data = col.data();
    for (int i = 0; i < n; i++) {
      if (validity == null || Bitmap.isSet(validity, i)) {
        int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
        int end = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2);
        hashes[i] = hashUnsafeBytes(data, start, end - start, hashes[i]);
      }
    }
  }

  /** Spark's {@code Pmod(hash, numPartitions)}: the non-negative remainder. */
  public static int pmod(int hash, int numPartitions) {
    int r = hash % numPartitions;
    return r < 0 ? r + numPartitions : r;
  }

  /**
   * Partition id per row for {@code HashPartitioning(keys, numPartitions)}: {@code ids[i] =
   * Pmod(Murmur3Hash(keys_i, 42), numPartitions)}. {@code hashes} is scratch of at least {@code n}.
   */
  public static void hashPartitionIds(
      VectorBuffers[] keys, KeyKind[] kinds, int n, int numPartitions, int[] hashes, int[] ids) {
    init(hashes, n, SPARK_SEED);
    for (int k = 0; k < keys.length; k++) {
      mixColumn(keys[k], kinds[k], hashes, n);
    }
    for (int i = 0; i < n; i++) {
      ids[i] = pmod(hashes[i], numPartitions);
    }
  }

  /**
   * Round-robin ids continuing from {@code start} (Spark's {@code RoundRobinPartitioning} starts each
   * task at a random partition and increments per row); returns the next start.
   */
  public static int roundRobinIds(int n, int numPartitions, int start, int[] ids) {
    int p = start;
    for (int i = 0; i < n; i++) {
      ids[i] = p;
      p++;
      if (p == numPartitions) {
        p = 0;
      }
    }
    return p;
  }

  /** One bitmap per partition from the id vector: {@code masks[p]} has bit {@code i} for {@code ids[i] == p}. */
  public static void partitionMasks(int[] ids, int n, MemorySegment[] masks, int[] counts) {
    java.util.Arrays.fill(counts, 0);
    for (int i = 0; i < n; i++) {
      int p = ids[i];
      Bitmap.set(masks[p], i);
      counts[p]++;
    }
  }
}
