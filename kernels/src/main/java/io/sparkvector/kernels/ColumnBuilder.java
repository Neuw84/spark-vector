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
 * <p>Buffers grow by doubling. In the given {@link Arena} (the constructor) the
 * abandoned smaller buffers stay allocated until the arena closes -- a shared
 * arena is expected there, sized by the caller -- so a column that grew to
 * {@code C} bytes holds {@code 2C}. An {@link #owning owning} builder keeps
 * each buffer in an arena of its own and releases a grown-out buffer at once,
 * so it holds its capacity and no more; the caller closes it (#416: the build
 * side of a hash join held four times its budget through the doubling's
 * leftovers).
 */
public final class ColumnBuilder implements AutoCloseable {

    private static final int DATA = 0;
    private static final int OFFSETS = 1;
    private static final int VALIDITY = 2;
    private static final int SCRATCH = 3;

    private final Arena arena; // the caller's, or null for an owning builder
    private final Arena[] owned; // per buffer, an owning builder's arenas; null otherwise
    private final VecType type;
    private int length;
    private MemorySegment data;
    private MemorySegment validity; // allocated lazily on the first null
    private MemorySegment offsets; // UTF8 only
    private long bytesUsed; // UTF8 data bytes

    public ColumnBuilder(Arena arena, VecType type, int expectedRows) {
        this(arena, false, type, expectedRows);
    }

    /**
     * A builder that owns its memory: a buffer it outgrows is released when the
     * grown one is in place, and {@link #close()} releases the rest. Its
     * {@link #view()} is valid until the next append or {@code close()}.
     */
    public static ColumnBuilder owning(VecType type, int expectedRows) {
        return new ColumnBuilder(null, true, type, expectedRows);
    }

    private ColumnBuilder(Arena arena, boolean owning, VecType type,
                          int expectedRows) {
        this.arena = arena;
        this.owned = owning ? new Arena[4] : null;
        this.type = type;
        int cap = Math.max(expectedRows, 1024);
        if (type == VecType.UTF8) {
            Arena o = freshArena();
            offsets = ArrowLayout.allocateOffsets(o, cap);
            install(OFFSETS, o);
            Arena d = freshArena();
            data = ArrowLayout.allocateBytes(d, (long) cap * 8);
            install(DATA, d);
        } else if (type == VecType.BOOL) {
            Arena d = freshArena();
            data = ArrowLayout.allocateBitmap(d, cap);
            install(DATA, d);
        } else {
            Arena d = freshArena();
            data = ArrowLayout.allocateData(d, type, cap);
            install(DATA, d);
        }
    }

    /**
     * The arena a new (or grown) buffer goes into: the caller's, or a fresh one
     * of this builder's.
     */
    private Arena freshArena() {
        return owned == null ? arena : Arena.ofShared();
    }

    /**
     * Makes {@code fresh} the arena of buffer {@code slot}, releasing the one it
     * grew out of (its bytes were copied over before this call).
     */
    private void install(int slot, Arena fresh) {
        if (owned == null) {
            return;
        }
        Arena old = owned[slot];
        owned[slot] = fresh;
        if (old != null) {
            old.close();
        }
    }

    /**
     * Releases an owning builder's memory (the view included); a no-op for a
     * builder over a caller's arena.
     */
    @Override
    public void close() {
        if (owned == null) {
            return;
        }
        for (int slot = 0; slot < owned.length; slot++) {
            if (owned[slot] != null) {
                owned[slot].close();
                owned[slot] = null;
            }
        }
        data = null;
        offsets = null;
        validity = null;
        scratch = null;
    }

    /**
     * The bytes this builder's buffers occupy now: their capacity, not the rows
     * in them.
     */
    public long allocatedBytes() {
        long bytes = data.byteSize();
        if (offsets != null) {
            bytes += offsets.byteSize();
        }
        if (validity != null) {
            bytes += validity.byteSize();
        }
        return bytes;
    }

    /**
     * The bytes this builder's buffers would occupy after appending the
     * {@code count} rows of {@code in} set in {@code selection} ({@code null}
     * for all rows): {@link #allocatedBytes()} as it would read then, the
     * growth rule and the lazily allocated validity bitmap included. A caller
     * with a memory budget asks before appending, so a batch that would double
     * the capacity past the budget is never appended.
     */
    public long bytesAfterAppend(VectorBuffers in, MemorySegment selection, int count) {
        if (count == 0) {
            return allocatedBytes();
        }
        int rows = length + count;
        int cap = capacityRows();
        int newCap = rows <= cap ? cap : Math.max(rows, cap * 2);
        long bytes;
        switch (type) {
            case UTF8 -> {
                bytes = ArrowLayout.padded(((long) newCap + 1) << 2);
                long dataBytes = bytesUsed + utf8Bytes(in, selection);
                bytes += dataBytes <= data.byteSize() ? data.byteSize() : ArrowLayout.padded(Math.max(dataBytes, data.byteSize() * 2));
            }
            case BOOL -> bytes = ArrowLayout.padded(Bitmap.bytesFor(newCap));
            default -> bytes = ArrowLayout.padded((long) newCap * type.byteWidth());
        }
        if (validity != null || in.hasNulls()) {
            bytes += ArrowLayout.padded(Bitmap.bytesFor(newCap));
        }
        return bytes;
    }

    /**
     * The UTF8 bytes an append of the selected rows of {@code in} adds to the
     * data buffer.
     */
    private static long utf8Bytes(VectorBuffers in, MemorySegment selection) {
        if (in.isDictionaryEncoded()) {
            long bytes = 0;
            VectorBuffers dict = in.dictionary();
            int n = in.length();
            for (int i = 0; i < n; i++) {
                if ((selection == null || Bitmap.isSet(selection, i)) && !in.isNull(i)) {
                    bytes += utf8Length(dict, in.getInt(i));
                }
            }
            return bytes;
        }
        if (selection == null) {
            MemorySegment off = in.offsets();
            return (long) off.get(VectorBuffers.LE_INT, (long) in.length() << 2) - off.get(VectorBuffers.LE_INT, 0L);
        }
        return CompactKernels.selectedUtf8Bytes(in, selection);
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
            Arena v = freshArena();
            validity = ArrowLayout.allocateBitmap(v, capacityRows());
            install(VALIDITY, v);
            Bitmap.fill(validity, start, true);
        }
        switch (type) {
            case UTF8 -> appendUtf8(in, selection, start);
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

    private void appendUtf8(VectorBuffers in, MemorySegment selection, int start) {
        if (selection == null && !in.isDictionaryEncoded()) {
            // A plain Arrow string vector's values are one contiguous range of its data buffer, so the
            // whole batch is one copy and the offsets move by a constant (#394: one MemorySegment.copy
            // per value -- with its bounds, alignment and liveness checks -- was 29% of an executor's
            // time in the sort stage of q67 at 1 TB). Nulls need nothing: a null value has zero length.
            int n = in.length();
            MemorySegment off = in.offsets();
            int first = off.get(VectorBuffers.LE_INT, 0L);
            int last = off.get(VectorBuffers.LE_INT, (long) n << 2);
            long bytes = (long) last - first;
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
            Arena s = freshArena();
            scratch = ArrowLayout.allocateBitmap(s, Math.max(bits, 8192));
            install(SCRATCH, s);
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
                Arena o = freshArena();
                MemorySegment grown = ArrowLayout.allocateOffsets(o, newCap);
                MemorySegment.copy(offsets, 0, grown, 0, ((long) length + 1) << 2);
                offsets = grown;
                install(OFFSETS, o);
            }
            case BOOL -> {
                Arena d = freshArena();
                MemorySegment grown = ArrowLayout.allocateBitmap(d, newCap);
                MemorySegment.copy(data, 0, grown, 0, Bitmap.bytesFor(length));
                data = grown;
                install(DATA, d);
            }
            default -> {
                Arena d = freshArena();
                MemorySegment grown = ArrowLayout.allocateData(d, type, newCap);
                MemorySegment.copy(data, 0, grown, 0, (long) length * type.byteWidth());
                data = grown;
                install(DATA, d);
            }
        }
        if (validity != null) {
            Arena v = freshArena();
            MemorySegment grown = ArrowLayout.allocateBitmap(v, newCap);
            MemorySegment.copy(validity, 0, grown, 0, Bitmap.bytesFor(length));
            validity = grown;
            install(VALIDITY, v);
        }
    }

    private void ensureBytes(long bytes) {
        if (bytes <= data.byteSize()) {
            return;
        }
        Arena d = freshArena();
        MemorySegment grown = ArrowLayout.allocateBytes(d, Math.max(bytes, data.byteSize() * 2));
        MemorySegment.copy(data, 0, grown, 0, bytesUsed);
        data = grown;
        install(DATA, d);
    }
}
