package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Distinct UTF8 values of any length, stored contiguously so the dictionary
 * reads as a UTF8 VectorBuffers: the group table's per-key dictionary and the
 * shuffle writer's staging dictionary (#377). Each entry is keyed by its length
 * and a 64-bit fingerprint (see PACKED_KEY_BYTES).
 */
public final class StringDictionary {

    /**
     * Values of up to this many bytes are keyed by their packed bytes (so
     * equality is a long compare); longer values by a 64-bit hash plus a byte
     * compare.
     */
    public static final int PACKED_KEY_BYTES = 8;

    private long[] bits = new long[64];
    private int[] lens = new int[64];
    private int[] ids = new int[64];
    private int mask = 63;
    private int size;
    private byte[] data = new byte[256];
    private int used;
    private int[] offsets = new int[33];
    private MemorySegment dataSegment = MemorySegment.ofArray(data);
    private MemorySegment offsetSegment = MemorySegment.ofArray(offsets);

    public int size() {
        return size;
    }

    /**
     * Direct index for single-byte values (TPC-H's flag columns), bypassing the
     * probe.
     */
    private final int[] singleByte = new int[256];

    public StringDictionary() {
        Arrays.fill(ids, -1);
        Arrays.fill(singleByte, -1);
    }

    private static final VarHandle LONG_LE = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /**
     * The 64-bit key of a value: its bytes packed little-endian when they fit
     * in a long, otherwise a multiply-xorshift mix over the bytes taken eight
     * at a time (a collision between two long values is caught by the byte
     * compare).
     */
    public static long fingerprint(byte[] src, int from, int len) {
        if (len <= PACKED_KEY_BYTES) {
            long packed = 0L;
            for (int b = len - 1; b >= 0; b--) {
                packed = (packed << 8) | (src[from + b] & 0xFFL);
            }
            return packed;
        }
        long h = 0x9E3779B97F4A7C15L ^ len;
        int end = from + len;
        int p = from;
        for (; p + 8 <= end; p += 8) {
            h = (h ^ (long) LONG_LE.get(src, p)) * 0xBF58476D1CE4E5B9L;
            h ^= h >>> 31;
        }
        if (p < end) {
            long tail = 0L;
            for (int b = end - 1; b >= p; b--) {
                tail = (tail << 8) | (src[b] & 0xFFL);
            }
            h = (h ^ tail) * 0x94D049BB133111EBL;
            h ^= h >>> 29;
        }
        return h;
    }

    /**
     * The id of the value, inserting it when absent if {@code insert}; -1 when
     * absent otherwise.
     */
    public int indexOf(long packed, int len, byte[] src,
                       int from, boolean insert) {
        if (len == 1) {
            int b = (int) packed; // 0..255
            int id = singleByte[b];
            if (id < 0) {
                id = probe(packed, len, src, from, insert);
                if (id >= 0) {
                    singleByte[b] = id;
                }
            }
            return id;
        }
        return probe(packed, len, src, from, insert);
    }

    /**
     * Per-caller byte scratch for {@link #indexOf(MemorySegment, long, int,
     * boolean, Scratch)}.
     */
    public static final class Scratch {
        byte[] bytes = new byte[64];
    }

    /**
     * {@link #indexOf} over a value that lives in a segment (an input
     * dictionary's entry).
     */
    public int indexOf(MemorySegment data, long start, int len,
                       boolean insert, Scratch s) {
        if (s.bytes.length < len) {
            s.bytes = new byte[Math.max(len, s.bytes.length * 2)];
        }
        MemorySegment.copy(data, ValueLayout.JAVA_BYTE, start, s.bytes, 0, len);
        return indexOf(fingerprint(s.bytes, 0, len), len, s.bytes, 0, insert);
    }

    /**
     * Forgets every value: ids start again from 0 (the shuffle writer, once
     * every pending row is flushed).
     */
    public void clear() {
        Arrays.fill(ids, -1);
        Arrays.fill(singleByte, -1);
        size = 0;
        used = 0;
    }

    private int probe(long packed, int len, byte[] src,
                      int from, boolean insert) {
        int pos = HashKernels.finish(HashKernels.mix32(HashKernels.fold(packed), len)) & mask;
        while (true) {
            int id = ids[pos];
            if (id < 0) {
                return insert ? insert(packed, len, src, from, pos) : -1;
            }
            if (bits[id] == packed
                    && lens[id] == len
                    && (len <= PACKED_KEY_BYTES || sameBytes(id, src, from, len))) {
                return id;
            }
            pos = (pos + 1) & mask;
        }
    }

    private boolean sameBytes(int id, byte[] src, int from,
            int len) {
        int off = offsets[id];
        return Arrays.equals(data, off, off + len, src, from,
                from + len);
    }

    private int insert(long packed, int len, byte[] src,
                       int from, int pos) {
        int id = size;
        if (id == bits.length) {
            bits = Arrays.copyOf(bits, id * 2);
            lens = Arrays.copyOf(lens, id * 2);
        }
        if (id + 1 >= offsets.length) {
            offsets = Arrays.copyOf(offsets, offsets.length * 2);
            offsetSegment = MemorySegment.ofArray(offsets);
        }
        if (used + len > data.length) {
            data = Arrays.copyOf(data,
                    Math.max(data.length * 2, used + len));
            dataSegment = MemorySegment.ofArray(data);
        }
        bits[id] = packed;
        lens[id] = len;
        System.arraycopy(src, from, data, used, len);
        used += len;
        offsets[id + 1] = used;
        ids[pos] = id;
        size++;
        if (size * 2 > ids.length) {
            rehash();
        }
        return id;
    }

    private void rehash() {
        int[] newIds = new int[ids.length * 2];
        Arrays.fill(newIds, -1);
        int newMask = newIds.length - 1;
        for (int id = 0; id < size; id++) {
            int pos = HashKernels.finish(HashKernels.mix32(HashKernels.fold(bits[id]), lens[id])) & newMask;
            while (newIds[pos] >= 0) {
                pos = (pos + 1) & newMask;
            }
            newIds[pos] = id;
        }
        ids = newIds;
        mask = newMask;
    }

    /**
     * The dictionary as a UTF8 column; valid until the next {@link #indexOf}
     * that inserts.
     */
    public VectorBuffers view() {
        return SegmentVectorBuffers.utf8(size, null, offsetSegment, dataSegment);
    }

    public int offset(int id) {
        return offsets[id];
    }

    public int length(int id) {
        return offsets[id + 1] - offsets[id];
    }

    public byte[] bytes() {
        return data;
    }

    /** Bytes of the distinct values held. */
    public long valueBytes() {
        return used;
    }

    public long memoryBytes() {
        return 8L * bits.length
                + 4L * lens.length
                + 4L * ids.length
                + data.length
                + 4L * offsets.length
                + 4L * singleByte.length;
    }
}
