package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Copies columns out of a producer's batch into memory an operator owns, and joins those copies
 * into one column. Blocking operators (the sort) need this because Spark's columnar contract lets
 * the producer reuse a batch as soon as the next one is requested.
 *
 * <p>Dictionary-encoded strings are decoded when concatenated: every chunk may carry a different
 * dictionary (one per Parquet row group), and a sort key or output needs a single encoding.
 */
public final class ChunkKernels {

  private ChunkKernels() {}

  /**
   * A copy of {@code in} restricted to the rows set in {@code selection} ({@code null} for all
   * rows), allocated from {@code arena}. {@code count} is the number of selected rows.
   */
  public static VectorBuffers materialize(VectorBuffers in, MemorySegment selection, int count, Arena arena) {
    int n = in.length();
    boolean nulls = in.hasNulls();
    MemorySegment validity = nulls ? ArrowLayout.allocateBitmap(arena, count) : null;
    if (in.type() == VecType.UTF8 && !in.isDictionaryEncoded()) {
      MemorySegment offsets = ArrowLayout.allocateOffsets(arena, count);
      if (selection == null) {
        int end = in.offsets().get(VectorBuffers.LE_INT, (long) n << 2);
        int start = in.offsets().get(VectorBuffers.LE_INT, 0);
        MemorySegment data = ArrowLayout.allocateBytes(arena, end - start);
        copyOffsetsRebased(in.offsets(), n, start, offsets);
        MemorySegment.copy(in.data(), start, data, 0, end - start);
        if (nulls) {
          BitmapKernels.copy(in.validity(), validity, n);
        }
        return SegmentVectorBuffers.utf8(count, validity, offsets, data);
      }
      long bytes = CompactKernels.selectedUtf8Bytes(in, selection);
      MemorySegment data = ArrowLayout.allocateBytes(arena, bytes);
      CompactKernels.compactUtf8(in, selection, count, offsets, data, validity);
      return SegmentVectorBuffers.utf8(count, validity, offsets, data);
    }
    VecType physical = in.isDictionaryEncoded() ? VecType.INT32 : in.type();
    MemorySegment data =
        physical == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, count) : ArrowLayout.allocateData(arena, physical, count);
    if (selection == null) {
      if (physical == VecType.BOOL) {
        BitmapKernels.copy(in.data(), data, n);
      } else {
        MemorySegment.copy(in.data(), 0, data, 0, (long) n * physical.byteWidth());
      }
      if (nulls) {
        BitmapKernels.copy(in.validity(), validity, n);
      }
    } else {
      CompactKernels.compactFixed(in, selection, count, data, validity);
    }
    if (in.isDictionaryEncoded()) {
      // The dictionary is small and belongs to the producer: copy it as well.
      return SegmentVectorBuffers.dictionaryUtf8(count, validity, data, materialize(in.dictionary(), null, in.dictionary().length(), arena));
    }
    return SegmentVectorBuffers.fixedWidth(in.type(), count, validity, data);
  }

  /**
   * One column holding the rows of every chunk in order. Chunks must share the logical type;
   * dictionary-encoded chunks are decoded to plain UTF8.
   */
  public static VectorBuffers concat(VectorBuffers[] chunks, int total, Arena arena) {
    if (chunks.length == 0) {
      throw new IllegalArgumentException("no chunks");
    }
    VecType type = chunks[0].type();
    boolean nulls = false;
    for (VectorBuffers c : chunks) {
      nulls |= c.hasNulls();
    }
    MemorySegment validity = nulls ? ArrowLayout.allocateBitmap(arena, total) : null;
    if (nulls) {
      Bitmap.fill(validity, total, true);
    }
    if (type == VecType.UTF8) {
      return concatUtf8(chunks, total, validity, arena);
    }
    MemorySegment data =
        type == VecType.BOOL ? ArrowLayout.allocateBitmap(arena, total) : ArrowLayout.allocateData(arena, type, total);
    int pos = 0;
    for (VectorBuffers c : chunks) {
      int n = c.length();
      if (type == VecType.BOOL) {
        for (int i = 0; i < n; i++) {
          Bitmap.setTo(data, pos + i, Bitmap.isSet(c.data(), i));
        }
      } else {
        MemorySegment.copy(c.data(), 0, data, (long) pos * type.byteWidth(), (long) n * type.byteWidth());
      }
      copyValidity(c, validity, pos);
      pos += n;
    }
    return SegmentVectorBuffers.fixedWidth(type, total, validity, data);
  }

  private static VectorBuffers concatUtf8(VectorBuffers[] chunks, int total, MemorySegment validity, Arena arena) {
    long bytes = 0;
    for (VectorBuffers c : chunks) {
      bytes += utf8Bytes(c);
    }
    MemorySegment offsets = ArrowLayout.allocateOffsets(arena, total);
    MemorySegment data = ArrowLayout.allocateBytes(arena, bytes);
    int pos = 0;
    int written = 0;
    offsets.set(VectorBuffers.LE_INT, 0, 0);
    for (VectorBuffers c : chunks) {
      int n = c.length();
      if (c.isDictionaryEncoded()) {
        VectorBuffers dict = c.dictionary();
        for (int i = 0; i < n; i++) {
          if (!c.isNull(i)) {
            int idx = c.getInt(i);
            if (!dict.isNull(idx)) {
              int s = dict.offsets().get(VectorBuffers.LE_INT, (long) idx << 2);
              int e = dict.offsets().get(VectorBuffers.LE_INT, (long) (idx + 1) << 2);
              MemorySegment.copy(dict.data(), s, data, written, e - s);
              written += e - s;
            }
          }
          offsets.set(VectorBuffers.LE_INT, (long) (pos + i + 1) << 2, written);
        }
      } else {
        int start = c.offsets().get(VectorBuffers.LE_INT, 0);
        int end = c.offsets().get(VectorBuffers.LE_INT, (long) n << 2);
        MemorySegment.copy(c.data(), start, data, written, end - start);
        for (int i = 1; i <= n; i++) {
          int o = c.offsets().get(VectorBuffers.LE_INT, (long) i << 2);
          offsets.set(VectorBuffers.LE_INT, (long) (pos + i) << 2, written + (o - start));
        }
        written += end - start;
      }
      copyValidity(c, validity, pos);
      pos += n;
    }
    return SegmentVectorBuffers.utf8(total, validity, offsets, data);
  }

  /** Bytes a chunk contributes once decoded to plain UTF8. */
  private static long utf8Bytes(VectorBuffers c) {
    int n = c.length();
    if (!c.isDictionaryEncoded()) {
      return c.offsets().get(VectorBuffers.LE_INT, (long) n << 2) - c.offsets().get(VectorBuffers.LE_INT, 0);
    }
    VectorBuffers dict = c.dictionary();
    long bytes = 0;
    for (int i = 0; i < n; i++) {
      if (!c.isNull(i)) {
        int idx = c.getInt(i);
        if (!dict.isNull(idx)) {
          bytes += dict.offsets().get(VectorBuffers.LE_INT, (long) (idx + 1) << 2)
              - dict.offsets().get(VectorBuffers.LE_INT, (long) idx << 2);
        }
      }
    }
    return bytes;
  }

  private static void copyValidity(VectorBuffers c, MemorySegment validity, int pos) {
    if (validity == null || !c.hasNulls()) {
      return; // already all ones
    }
    int n = c.length();
    for (int i = 0; i < n; i++) {
      if (!Bitmap.isSet(c.validity(), i)) {
        Bitmap.clear(validity, pos + i);
      }
    }
  }

  private static void copyOffsetsRebased(MemorySegment in, int n, int base, MemorySegment out) {
    if (base == 0) {
      MemorySegment.copy(in, 0, out, 0, ((long) n + 1) << 2);
      return;
    }
    for (int i = 0; i <= n; i++) {
      out.set(VectorBuffers.LE_INT, (long) i << 2, in.get(VectorBuffers.LE_INT, (long) i << 2) - base);
    }
  }
}
