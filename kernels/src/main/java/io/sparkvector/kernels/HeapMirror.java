package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;

/**
 * A fixed-width column mirrored into Java arrays, for gathers that touch a row
 * per candidate pair.
 *
 * <p>A hash join over a many-to-many key evaluates its residual condition on
 * 10^9 candidate pairs (TPC-DS q72), and every pair costs one element read per
 * condition column plus a validity bit. Those reads through {@link
 * MemorySegment} accessors compile to virtual calls when the receiver profile
 * mixes native and heap segments -- the JIT's inlining log says {@code no
 * static binding} on the segment's offset lookup -- at tens of nanoseconds
 * each. An {@code int[]} read is a bounds check. So a column the condition
 * reads is copied into arrays once (one bulk copy per build table, or per
 * streamed batch) and the pair-wise loop runs over the arrays; the gathered
 * chunk is then copied into the Arrow output in one bulk move (#332).
 *
 * <p>INT32, INT64 and FLOAT64 only (a double is a long here), plain encoding: a
 * dictionary-encoded or variable-width column is gathered the usual way.
 */
public final class HeapMirror {
    public final VecType type;
    public final int length;

    /** INT32 values, or null. */
    public final int[] ints;

    /** INT64 / FLOAT64 raw bits, or null. */
    public final long[] longs;

    /**
     * Validity as 64-row words (LSB first, Arrow's order), or null when every
     * row is valid.
     */
    public final long[] validity;

    private HeapMirror(VecType type, int length, int[] ints,
                       long[] longs, long[] validity) {
        this.type = type;
        this.length = length;
        this.ints = ints;
        this.longs = longs;
        this.validity = validity;
    }

    /**
     * A mirror over arrays already laid out by the caller ({@link
     * RangeResidual#cluster}).
     */
    static HeapMirror clustered(VecType type, int length, int[] ints,
            long[] longs, long[] validity) {
        return new HeapMirror(type, length, ints, longs, validity);
    }

    /** Whether {@code in} is a column this mirrors: plain INT32, INT64 or FLOAT64. */
    public static boolean mirrors(VectorBuffers in) {
        if (in.isDictionaryEncoded()) {
            return false;
        }
        VecType t = in.type();
        return t == VecType.INT32 || t == VecType.INT64 || t == VecType.FLOAT64;
    }

    /**
     * Copies the column's data (and validity, when it has nulls) into arrays:
     * one bulk move each.
     */
    public static HeapMirror of(VectorBuffers in) {
        int n = in.length();
        VecType t = in.type();
        int[] ints = null;
        long[] longs = null;
        if (t == VecType.INT32) {
            ints = new int[n];
            MemorySegment.copy(in.data(), VectorBuffers.LE_INT, 0L, ints, 0,
                    n);
        } else {
            longs = new long[n];
            MemorySegment.copy(in.data(), VectorBuffers.LE_LONG, 0L, longs, 0,
                    n);
        }
        long[] validity = null;
        if (in.hasNulls()) {
            int words = Bitmap.wordsFor(n);
            validity = new long[words];
            for (int w = 0; w < words; w++) {
                validity[w] = Bitmap.wordAt(in.validity(), w, n);
            }
        }
        return new HeapMirror(t, n, ints, longs, validity);
    }

    /** Whether row {@code i} is valid. */
    public boolean isValid(int i) {
        return validity == null || ((validity[i >>> 6] >>> (i & 63)) & 1L) != 0L;
    }

    /**
     * Gathers rows {@code idx[from..to)} (a negative index pads a null / zero
     * row) into the Arrow buffers: the values go through a heap scratch array
     * and one bulk copy; {@code outValidity} may be null when the caller knows
     * every gathered row is valid.
     */
    public void gather(int[] idx, int from, int to,
                       MemorySegment outData, MemorySegment outValidity, GatherScratch scratch) {
        int count = to - from;
        if (ints != null) {
            int[] out = scratch.ints(count);
            for (int o = 0; o < count; o++) {
                int i = idx[from + o];
                out[o] = i < 0 ? 0 : ints[i];
            }
            MemorySegment.copy(out, 0, outData, VectorBuffers.LE_INT, 0L, count);
        } else {
            long[] out = scratch.longs(count);
            for (int o = 0; o < count; o++) {
                int i = idx[from + o];
                out[o] = i < 0 ? 0L : longs[i];
            }
            MemorySegment.copy(out, 0, outData, VectorBuffers.LE_LONG, 0L, count);
        }
        if (outValidity != null) {
            int words = Bitmap.wordsFor(count);
            long[] bits = scratch.longs(words);
            for (int w = 0; w < words; w++) {
                int limit = Math.min(64, count - (w << 6));
                long word = 0L;
                int base = from + (w << 6);
                for (int j = 0; j < limit; j++) {
                    int i = idx[base + j];
                    if (i >= 0 && isValid(i)) {
                        word |= 1L << j;
                    }
                }
                bits[w] = word;
            }
            for (int w = 0; w < words; w++) {
                Bitmap.setWord(outValidity, w, count, bits[w]);
            }
        }
    }

    /**
     * Reusable heap arrays for {@link #gather}: one per iterator, grown on
     * demand.
     */
    public static final class GatherScratch {
        private int[] ints = new int[0];
        private long[] longs = new long[0];

        int[] ints(int n) {
            if (ints.length < n) {
                ints = new int[Math.max(n, ints.length * 2)];
            }
            return ints;
        }

        long[] longs(int n) {
            if (longs.length < n) {
                longs = new long[Math.max(n, longs.length * 2)];
            }
            return longs;
        }
    }
}
