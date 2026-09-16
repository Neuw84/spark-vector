package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Row selection between several same-typed columns, the kernel behind {@code CASE WHEN}, {@code IF}
 * and {@code COALESCE}: for each row the first branch whose {@code wins} bit is set supplies the
 * value, else the {@code otherwise} column does, else the row is null. Branch masks are expected to
 * be disjoint (the caller carries "already decided" forward so a later branch only wins where no
 * earlier one did); a branch column may be {@code null} to mean a {@code NULL} literal.
 *
 * <p>The blend is a scalar pass over the rows: the decision per row is a handful of bit tests and
 * one copy, and unlike the compare and arithmetic kernels there is no per-lane arithmetic for SIMD
 * to speed up. A {@code VectorMask.blend} formulation for the fixed-width types is the obvious next
 * step if a profile ever shows this loop.
 */
public final class SelectKernels {

  private SelectKernels() {}

  /**
   * Selects per row among {@code branches} (a column and its winning-rows bitmap each) and
   * {@code otherwise} (may be {@code null}: no ELSE), producing a new column of {@code type} in
   * {@code arena}. Rows outside {@code active} (may be {@code null}: all rows) are produced as null,
   * since nothing downstream reads them.
   */
  public static SegmentVectorBuffers select(
      VecType type,
      int n,
      MemorySegment[] wins,
      VectorBuffers[] branches,
      VectorBuffers otherwise,
      MemorySegment active,
      Arena arena) {
    if (wins.length != branches.length) {
      throw new IllegalArgumentException("one mask per branch");
    }
    MemorySegment validity = ArrowLayout.allocateBitmap(arena, n);
    if (type == VecType.UTF8) {
      return selectUtf8(n, wins, branches, otherwise, active, arena, validity);
    }
    MemorySegment data =
        type == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, n) : ArrowLayout.allocateData(arena, type, n);
    for (int i = 0; i < n; i++) {
      VectorBuffers src = source(i, wins, branches, otherwise, active);
      if (src == null || src.isNull(i)) {
        continue; // validity bit stays clear; data stays zero
      }
      Bitmap.set(validity, i);
      switch (type) {
        case INT32 -> data.setAtIndex(VectorBuffers.LE_INT, i, src.getInt(i));
        case INT64 -> data.setAtIndex(VectorBuffers.LE_LONG, i, src.getLong(i));
        case FLOAT64 -> data.setAtIndex(VectorBuffers.LE_DOUBLE, i, src.getDouble(i));
        case BOOL -> Bitmap.setTo(data, i, src.getBoolean(i));
        default -> throw new IllegalArgumentException("unsupported type " + type);
      }
    }
    return SegmentVectorBuffers.fixedWidth(type, n, validity, data);
  }

  /** The column row {@code i} takes its value from, or {@code null} for a null result. */
  private static VectorBuffers source(
      int i, MemorySegment[] wins, VectorBuffers[] branches, VectorBuffers otherwise, MemorySegment active) {
    if (active != null && !Bitmap.isSet(active, i)) {
      return null;
    }
    for (int k = 0; k < wins.length; k++) {
      if (Bitmap.isSet(wins[k], i)) {
        return branches[k];
      }
    }
    return otherwise;
  }

  private static SegmentVectorBuffers selectUtf8(
      int n,
      MemorySegment[] wins,
      VectorBuffers[] branches,
      VectorBuffers otherwise,
      MemorySegment active,
      Arena arena,
      MemorySegment validity) {
    // Two passes over the same decisions: sizes first, then one contiguous copy.
    long total = 0;
    for (int i = 0; i < n; i++) {
      VectorBuffers src = source(i, wins, branches, otherwise, active);
      if (src != null && !src.isNull(i)) {
        total += utf8Length(src, i);
      }
    }
    MemorySegment offsets = ArrowLayout.allocateOffsets(arena, n);
    MemorySegment data = ArrowLayout.allocateBytes(arena, total);
    int pos = 0;
    offsets.setAtIndex(VectorBuffers.LE_INT, 0, 0);
    for (int i = 0; i < n; i++) {
      VectorBuffers src = source(i, wins, branches, otherwise, active);
      if (src != null && !src.isNull(i)) {
        Bitmap.set(validity, i);
        pos += copyUtf8(src, i, data, pos);
      }
      offsets.setAtIndex(VectorBuffers.LE_INT, i + 1, pos);
    }
    return SegmentVectorBuffers.utf8(n, validity, offsets, data);
  }

  /** Byte length of row {@code i}, through the dictionary when the column is encoded. */
  private static int utf8Length(VectorBuffers src, int i) {
    VectorBuffers plain = src.isDictionaryEncoded() ? src.dictionary() : src;
    int row = src.isDictionaryEncoded() ? src.data().getAtIndex(VectorBuffers.LE_INT, i) : i;
    MemorySegment off = plain.offsets();
    return off.getAtIndex(VectorBuffers.LE_INT, row + 1) - off.getAtIndex(VectorBuffers.LE_INT, row);
  }

  private static int copyUtf8(VectorBuffers src, int i, MemorySegment out, int pos) {
    VectorBuffers plain = src.isDictionaryEncoded() ? src.dictionary() : src;
    int row = src.isDictionaryEncoded() ? src.data().getAtIndex(VectorBuffers.LE_INT, i) : i;
    MemorySegment off = plain.offsets();
    int start = off.getAtIndex(VectorBuffers.LE_INT, row);
    int len = off.getAtIndex(VectorBuffers.LE_INT, row + 1) - start;
    MemorySegment.copy(plain.data(), ValueLayout.JAVA_BYTE, start, out, ValueLayout.JAVA_BYTE, pos, len);
    return len;
  }
}
