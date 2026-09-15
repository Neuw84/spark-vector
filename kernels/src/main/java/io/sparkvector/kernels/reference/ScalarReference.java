package io.sparkvector.kernels.reference;

import io.sparkvector.kernels.Bitmap;
import io.sparkvector.kernels.CompareOp;
import io.sparkvector.kernels.VecType;
import io.sparkvector.kernels.VectorBuffers;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Straightforward scalar implementations of every kernel. They define the expected semantics and
 * are used as oracles by the tests and as baselines by the JMH benchmarks. Nothing here is tuned.
 */
public final class ScalarReference {

  private ScalarReference() {}

  // ---------------------------------------------------------------- comparisons

  /** Element-wise compare; writes one result bit per element (null lanes get arbitrary bits). */
  public static void compareScalar(VectorBuffers a, Number scalar, CompareOp op, MemorySegment out) {
    int n = a.length();
    switch (a.type()) {
      case INT32 -> {
        int s = scalar.intValue();
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(out, i, op.test(Integer.compare(a.getInt(i), s)));
        }
      }
      case INT64 -> {
        long s = scalar.longValue();
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(out, i, op.test(Long.compare(a.getLong(i), s)));
        }
      }
      case FLOAT64 -> {
        double s = scalar.doubleValue();
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(out, i, op.test(a.getDouble(i), s));
        }
      }
      default -> throw new IllegalArgumentException("unsupported " + a.type());
    }
  }

  public static void compare(VectorBuffers a, VectorBuffers b, CompareOp op, MemorySegment out) {
    int n = a.length();
    switch (a.type()) {
      case INT32 -> {
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(out, i, op.test(Integer.compare(a.getInt(i), b.getInt(i))));
        }
      }
      case INT64 -> {
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(out, i, op.test(Long.compare(a.getLong(i), b.getLong(i))));
        }
      }
      case FLOAT64 -> {
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(out, i, op.test(a.getDouble(i), b.getDouble(i)));
        }
      }
      default -> throw new IllegalArgumentException("unsupported " + a.type());
    }
  }

  // ---------------------------------------------------------------- bitmaps

  public static void and(MemorySegment a, MemorySegment b, MemorySegment out, int n) {
    for (int i = 0; i < n; i++) {
      Bitmap.setTo(out, i, Bitmap.isSet(a, i) && Bitmap.isSet(b, i));
    }
  }

  public static void or(MemorySegment a, MemorySegment b, MemorySegment out, int n) {
    for (int i = 0; i < n; i++) {
      Bitmap.setTo(out, i, Bitmap.isSet(a, i) || Bitmap.isSet(b, i));
    }
  }

  public static void not(MemorySegment a, MemorySegment out, int n) {
    for (int i = 0; i < n; i++) {
      Bitmap.setTo(out, i, !Bitmap.isSet(a, i));
    }
  }

  /** Bits set where the element is true and not null. */
  public static void selection(MemorySegment bits, MemorySegment validity, MemorySegment out, int n) {
    for (int i = 0; i < n; i++) {
      Bitmap.setTo(out, i, Bitmap.isSet(bits, i) && (validity == null || Bitmap.isSet(validity, i)));
    }
  }

  /** Three-valued AND: null if either side is null unless the other side is false. */
  public static void kleeneAnd(
      MemorySegment aBits, MemorySegment aValid,
      MemorySegment bBits, MemorySegment bValid,
      MemorySegment outBits, MemorySegment outValid, int n) {
    for (int i = 0; i < n; i++) {
      Boolean a = value(aBits, aValid, i);
      Boolean b = value(bBits, bValid, i);
      Boolean r;
      if (Boolean.FALSE.equals(a) || Boolean.FALSE.equals(b)) {
        r = Boolean.FALSE;
      } else if (a == null || b == null) {
        r = null;
      } else {
        r = Boolean.TRUE;
      }
      Bitmap.setTo(outValid, i, r != null);
      Bitmap.setTo(outBits, i, Boolean.TRUE.equals(r));
    }
  }

  /** Three-valued OR: null if either side is null unless the other side is true. */
  public static void kleeneOr(
      MemorySegment aBits, MemorySegment aValid,
      MemorySegment bBits, MemorySegment bValid,
      MemorySegment outBits, MemorySegment outValid, int n) {
    for (int i = 0; i < n; i++) {
      Boolean a = value(aBits, aValid, i);
      Boolean b = value(bBits, bValid, i);
      Boolean r;
      if (Boolean.TRUE.equals(a) || Boolean.TRUE.equals(b)) {
        r = Boolean.TRUE;
      } else if (a == null || b == null) {
        r = null;
      } else {
        r = Boolean.FALSE;
      }
      Bitmap.setTo(outValid, i, r != null);
      Bitmap.setTo(outBits, i, Boolean.TRUE.equals(r));
    }
  }

  private static Boolean value(MemorySegment bits, MemorySegment valid, int i) {
    if (valid != null && !Bitmap.isSet(valid, i)) {
      return null;
    }
    return Bitmap.isSet(bits, i);
  }

  // ---------------------------------------------------------------- compaction

  /** Number of selected elements. */
  public static int selectedCount(MemorySegment selection, int n) {
    return Bitmap.popcount(selection, n);
  }

  /**
   * Copies selected elements of a fixed-width or BOOL column into the output buffers. Output
   * validity may be {@code null} when the input has no nulls.
   */
  public static void compactFixed(
      VectorBuffers in, MemorySegment selection, MemorySegment outData, MemorySegment outValidity) {
    int n = in.length();
    int o = 0;
    for (int i = 0; i < n; i++) {
      if (!Bitmap.isSet(selection, i)) {
        continue;
      }
      switch (in.type()) {
        case INT32 -> outData.set(VectorBuffers.LE_INT, (long) o << 2, in.getInt(i));
        case INT64 -> outData.set(VectorBuffers.LE_LONG, (long) o << 3, in.getLong(i));
        case FLOAT64 -> outData.set(VectorBuffers.LE_DOUBLE, (long) o << 3, in.getDouble(i));
        case BOOL -> Bitmap.setTo(outData, o, in.getBoolean(i));
        default -> throw new IllegalArgumentException("not fixed width: " + in.type());
      }
      if (outValidity != null) {
        Bitmap.setTo(outValidity, o, !in.isNull(i));
      }
      o++;
    }
  }

  /** Total UTF-8 bytes of the selected, non-null elements. */
  public static long selectedUtf8Bytes(VectorBuffers in, MemorySegment selection) {
    if (in.type() != VecType.UTF8 || in.isDictionaryEncoded()) {
      throw new IllegalArgumentException("expected plain UTF8");
    }
    long total = 0;
    MemorySegment off = in.offsets();
    for (int i = 0; i < in.length(); i++) {
      if (Bitmap.isSet(selection, i) && !in.isNull(i)) {
        total += off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) i << 2);
      }
    }
    return total;
  }

  public static void compactUtf8(
      VectorBuffers in, MemorySegment selection,
      MemorySegment outOffsets, MemorySegment outData, MemorySegment outValidity) {
    int n = in.length();
    int o = 0;
    int pos = 0;
    MemorySegment off = in.offsets();
    for (int i = 0; i < n; i++) {
      if (!Bitmap.isSet(selection, i)) {
        continue;
      }
      outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
      boolean valid = !in.isNull(i);
      if (valid) {
        int start = off.get(VectorBuffers.LE_INT, (long) i << 2);
        int end = off.get(VectorBuffers.LE_INT, (long) (i + 1) << 2);
        MemorySegment.copy(in.data(), ValueLayout.JAVA_BYTE, start, outData, ValueLayout.JAVA_BYTE, pos, end - start);
        pos += end - start;
      }
      if (outValidity != null) {
        Bitmap.setTo(outValidity, o, valid);
      }
      o++;
    }
    outOffsets.set(VectorBuffers.LE_INT, (long) o << 2, pos);
  }
}
