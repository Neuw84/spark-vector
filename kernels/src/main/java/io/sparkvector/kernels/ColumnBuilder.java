package io.sparkvector.kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Accumulates the rows of many batches of one column into a single Arrow-layout
 * column in native memory, for blocking operators (sort, hash-join build side)
 * that must see a whole partition. Batches are appended whole or through a
 * selection bitmap; dictionary-encoded strings are decoded on the way in, since
 * the dictionaries of different batches are unrelated.
 *
 * <p>Buffers grow by doubling inside the given {@link Arena}; a shared arena is
 * expected, so the abandoned smaller buffers are released with everything else
 * when the operator closes.
 */
public final class ColumnBuilder {

    private final Arena arena;
    private final VecType type;
    private int length;
    private MemorySegment data;
    private MemorySegment validity; // allocated lazily on the first null
    private MemorySegment offsets; // UTF8 only
    private long bytesUsed; // UTF8 data bytes

    public ColumnBuilder(Arena arena, VecType type, int expectedRows) {
        this.arena = arena;
        this.type = type;
        int cap = Math.max(expectedRows, 1024);
        if (type == VecType.UTF8) {
            offsets = ArrowLayout.allocateOffsets(arena, cap);
            data = ArrowLayout.allocateBytes(arena, (long) cap * 8);
        } else if (type == VecType.BOOL) {
            data = ArrowLayout.allocateBitmap(arena, cap);
        } else {
            data = ArrowLayout.allocateData(arena, type, cap);
        }
    }

    public VecType type() {
        return type;
    }

    public int length() {
        return length;
    }

    /** Appends the whole batch. */
    public void append(VectorBuffers in) {
        append(in, null, in.length());
    }

    /**
     * Appends the rows of {@code in} set in {@code selection} ({@code null} for
     * all rows).
     */
    public void append(VectorBuffers in, MemorySegment selection, int count) {
        if (in.type() != type) {
            throw new IllegalArgumentException("appending " + in.type() + " to a " + type + " column");
        }
        if (count == 0) {
            return;
        }
        int start = length;
        ensureRows(start + count);
        boolean needValidity = in.hasNulls();
        if (needValidity && validity == null) {
            validity = ArrowLayout.allocateBitmap(arena, capacityRows());
            Bitmap.fill(validity, start, true);
        }
        switch (type) {
            case UTF8 -> appendUtf8(in, selection, count, start);
            case BOOL -> {
                if (selection == null) {
                    for (int i = 0; i < count; i++) {
                        Bitmap.setTo(data, start + i, in.getBoolean(i));
                    }
                } else {
                    int o = start;
                    for (int w = 0, words = Bitmap.wordsFor(in.length());
                         w < words;
                         w++) {
                        long bits = Bitmap.wordAt(selection, w, in.length());
                        while (bits != 0L) {
                            int i = (w << 6) + Long.numberOfTrailingZeros(bits);
                            bits &= bits - 1;
                            Bitmap.setTo(data, o++, in.getBoolean(i));
                        }
                    }
                }
            }
            default -> {
                int width = type.byteWidth();
                if (selection == null) {
                    MemorySegment.copy(in.data(), 0, data, (long) start * width,
                            (long) count * width);
                } else {
                    // compactFixed writes from the start of the given segment: hand it the tail (the layouts
                    // are unaligned, and start * width is a multiple of the element width anyway).
                    // Validity is appended bit by bit below (the start bit is not byte aligned), so the
                    // kernel's own validity output goes to scratch.
                    MemorySegment scratch = in.hasNulls() ? scratchBitmap(count) : null;
                    CompactKernels.compactFixed(in, selection, count, data.asSlice((long) start * width), scratch);
                }
            }
        }
        appendValidity(in, selection, count, start);
        length = start + count;
    }

    private void appendValidity(VectorBuffers in, MemorySegment selection, int count,
            int start) {
        if (validity == null) {
            return;
        }
        if (!in.hasNulls()) {
            for (int i = 0; i < count; i++) {
                Bitmap.set(validity, start + i);
            }
            return;
        }
        MemorySegment v = in.validity();
        if (selection == null) {
            for (int i = 0; i < count; i++) {
                Bitmap.setTo(validity, start + i, Bitmap.isSet(v, i));
            }
        } else {
            int o = start;
            for (int w = 0, words = Bitmap.wordsFor(in.length());
                 w < words;
                 w++) {
                long bits = Bitmap.wordAt(selection, w, in.length());
                while (bits != 0L) {
                    int i = (w << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    Bitmap.setTo(validity, o++, Bitmap.isSet(v, i));
                }
            }
        }
    }

    private void appendUtf8(VectorBuffers in, MemorySegment selection, int count,
                            int start) {
        if (selection == null && !in.isDictionaryEncoded()) {
            // A plain Arrow string vector's values are one contiguous range of its data buffer, so the
            // whole batch is one copy and the offsets move by a constant (#394: one MemorySegment.copy
            // per value -- with its bounds, alignment and liveness checks -- was 29% of an executor's
            // time in the sort stage of q67 at 1 TB). Nulls need nothing: a null value has zero length.
            int n = in.length();
            MemorySegment off = in.offsets();
            int first = off.get(VectorBuffers.LE_INT, 0L);
            int last = off.get(VectorBuffers.LE_INT, (long) n << 2);
            long bytes = last - first;
            ensureBytes(bytesUsed + bytes);
            MemorySegment.copy(in.data(), ValueLayout.JAVA_BYTE, first, data, ValueLayout.JAVA_BYTE,
                    bytesUsed, bytes);
            int delta = (int) bytesUsed - first;
            for (int i = 0; i <= n; i++) {
                offsets.set(VectorBuffers.LE_INT, (long) (start + i) << 2, off.get(VectorBuffers.LE_INT, (long) i << 2) + delta);
            }
            bytesUsed += bytes;
            return;
        }
        long bytes;
        if (in.isDictionaryEncoded()) {
            bytes = 0;
            VectorBuffers dict = in.dictionary();
            int n = in.length();
            for (int i = 0; i < n; i++) {
                if ((selection == null || Bitmap.isSet(selection, i)) && !in.isNull(i)) {
                    bytes += utf8Length(dict, in.getInt(i));
                }
            }
        } else if (selection == null) {
            bytes = in.offsets().get(VectorBuffers.LE_INT, (long) in.length() << 2);
        } else {
            bytes = CompactKernels.selectedUtf8Bytes(in, selection);
        }
        ensureBytes(bytesUsed + bytes);
        int o = start;
        long pos = bytesUsed;
        int n = in.length();
        for (int i = 0; i < n; i++) {
            if (selection != null && !Bitmap.isSet(selection, i)) {
                continue;
            }
            offsets.set(VectorBuffers.LE_INT, (long) o << 2, (int) pos);
            if (!in.isNull(i)) {
                MemorySegment src;
                int srcStart;
                int len;
                if (in.isDictionaryEncoded()) {
                    VectorBuffers dict = in.dictionary();
                    int k = in.getInt(i);
                    src = dict.data();
                    srcStart = dict.offsets().get(VectorBuffers.LE_INT, (long) k << 2);
                    len = utf8Length(dict, k);
                } else {
                    src = in.data();
                    srcStart = in.offsets().get(VectorBuffers.LE_INT, (long) i << 2);
                    len = in.offsets().get(VectorBuffers.LE_INT, (long) (i + 1) << 2) - srcStart;
                }
                MemorySegment.copy(src, ValueLayout.JAVA_BYTE, srcStart, data, ValueLayout.JAVA_BYTE, pos,
                        len);
                pos += len;
            }
            o++;
        }
        offsets.set(VectorBuffers.LE_INT, (long) o << 2, (int) pos);
        bytesUsed = pos;
    }

    private static int utf8Length(VectorBuffers dict, int k) {
        MemorySegment off = dict.offsets();
        return off.get(VectorBuffers.LE_INT, (long) (k + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) k << 2);
    }

    /** The accumulated column; valid until the arena is closed or the next append. */
    public VectorBuffers view() {
        if (type == VecType.UTF8) {
            return SegmentVectorBuffers.utf8(length, validity, offsets, data);
        }
        return SegmentVectorBuffers.fixedWidth(type, length, validity, data);
    }

    // ------------------------------------------------------------------ growth

    private MemorySegment scratch;

    private MemorySegment scratchBitmap(int bits) {
        if (scratch == null || scratch.byteSize() < Bitmap.bytesFor(bits)) {
            scratch = ArrowLayout.allocateBitmap(arena, Math.max(bits, 8192));
        }
        return scratch;
    }

    private int capacityRows() {
        return switch (type) {
            case UTF8 -> (int) Math.min(Integer.MAX_VALUE, (offsets.byteSize() >>> 2) - 1);
            case BOOL -> (int) Math.min(Integer.MAX_VALUE, data.byteSize() << 3);
            default -> (int) Math.min(Integer.MAX_VALUE, data.byteSize() / type.byteWidth());
        };
    }

    private void ensureRows(int rows) {
        int cap = capacityRows();
        if (rows <= cap) {
            return;
        }
        int newCap = Math.max(rows, cap * 2);
        switch (type) {
            case UTF8 -> {
                MemorySegment grown = ArrowLayout.allocateOffsets(arena, newCap);
                MemorySegment.copy(offsets, 0, grown, 0, ((long) length + 1) << 2);
                offsets = grown;
            }
            case BOOL -> {
                MemorySegment grown = ArrowLayout.allocateBitmap(arena, newCap);
                MemorySegment.copy(data, 0, grown, 0, Bitmap.bytesFor(length));
                data = grown;
            }
            default -> {
                MemorySegment grown = ArrowLayout.allocateData(arena, type, newCap);
                MemorySegment.copy(data, 0, grown, 0, (long) length * type.byteWidth());
                data = grown;
            }
        }
        if (validity != null) {
            MemorySegment grown = ArrowLayout.allocateBitmap(arena, newCap);
            MemorySegment.copy(validity, 0, grown, 0, Bitmap.bytesFor(length));
            validity = grown;
        }
    }

    private void ensureBytes(long bytes) {
        if (bytes <= data.byteSize()) {
            return;
        }
        MemorySegment grown = ArrowLayout.allocateBytes(arena, Math.max(bytes, data.byteSize() * 2));
        MemorySegment.copy(data, 0, grown, 0, bytesUsed);
        data = grown;
    }
}
