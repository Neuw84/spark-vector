package io.sparkvector.benchmarks;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

import io.sparkvector.kernels.Decimal128;
import io.sparkvector.kernels.VectorBuffers;

/**
 * The index sort before #285, kept only as the baseline {@link SortBenchmark}
 * measures the radix kernel against: one {@code Arrays.sort(long[])} per 32-bit
 * key pass. Not used by the plugin. Its documentation follows.
 *
 * <p>Columnar sort: a permutation of row indices ordered by one or more key
 * columns; {@link GatherKernels} applies it to the output columns.
 *
 * <p>The sort is least-significant-key-first over order-preserving 32-bit key
 * passes. Each pass packs {@code (key, current position)} into a {@code long}
 * and sorts the array with {@link Arrays#sort(long[])}: the position in the low
 * bits makes every pass stable, so processing keys from the last to the first
 * yields the lexicographic order. Values are mapped to unsigned 32-bit keys
 * that compare like the value does:
 *
 * <ul>
 *   <li>int32 and booleans: one pass;
 *   <li>int64 and doubles: two passes (low then high half) over the
 *       sign-normalised bits, doubles first put into total order the way Spark
 *       compares them ({@code -0.0 == 0.0}, NaN above everything, NaN equal to
 *       NaN);
 *   <li>UTF8: strings up to 8 bytes long take three passes (length, then the
 *       two halves of the zero-padded big-endian prefix, which orders bytes
 *       unsigned like Spark's {@code UTF8String}); longer strings are ranked
 *       once by a stable merge sort with a byte comparator and the rank takes
 *       one pass;
 *   <li>nulls: one final pass per key on the null flag, placing them first or
 *       last.
 * </ul>
 *
 * <p>Descending order inverts the key bits. Everything is primitive-array work
 * the JIT vectorises where it can; the Vector API has no sort primitive and
 * this is not one.
 */
final class LegacySortKernels {

    private LegacySortKernels() {}

    /**
     * Row indices of the {@code n}-row columns in sorted order. {@code keys[c]}
     * is compared in the direction {@code ascending[c]} and nulls placed
     * according to {@code nullsFirst[c]}; the first key is the most
     * significant. Stable: rows equal on every key keep their input order.
     */
    public static int[] sortIndices(VectorBuffers[] keys, boolean[] ascending, boolean[] nullsFirst,
            int n) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        if (n < 2) {
            return order;
        }
        long[] packed = new long[n];
        int[] keyScratch = new int[n];
        for (int c = keys.length - 1; c >= 0; c--) {
            VectorBuffers k = keys[c];
            boolean desc = !ascending[c];
            switch (k.type()) {
                // Null rows get a constant key in every value pass so that they keep their relative
                // order (the reference sort is stable and so must this be); the null pass below then moves
                // them to the front or the back as a block.
                case INT32 -> {
                    for (int i = 0; i < n; i++) {
                        int row = order[i];
                        keyScratch[i] = k.isNull(row) ? 0 : flip(k.getInt(row) ^ Integer.MIN_VALUE, desc);
                    }
                    order = pass(order, keyScratch, packed, n);
                }
                case BOOL -> {
                    for (int i = 0; i < n; i++) {
                        int row = order[i];
                        keyScratch[i] = k.isNull(row) ? 0 : flip(k.getBoolean(row) ? 1 : 0, desc);
                    }
                    order = pass(order, keyScratch, packed, n);
                }
                case INT64 -> order = passes64(order, k, false, desc, packed, keyScratch,
                        n);
                case FLOAT64 -> order = passes64(order, k, true, desc, packed, keyScratch,
                        n);
                case UTF8 -> order = passesUtf8(order, k, desc, packed, keyScratch, n);
                case DECIMAL128 -> order = passes128(order, k, desc, packed, keyScratch, n);
                default -> throw new IllegalArgumentException("unsupported sort key type " + k.type());
            }
            if (k.hasNulls()) {
                int nullKey = nullsFirst[c] ? 0 : 1;
                for (int i = 0; i < n; i++) {
                    keyScratch[i] = k.isNull(order[i]) ? nullKey : 1 - nullKey;
                }
                order = pass(order, keyScratch, packed, n);
            }
        }
        return order;
    }

    private static int flip(int key, boolean desc) {
        return desc ? ~key : key;
    }

    /**
     * One stable pass: reorders {@code order} by the unsigned 32-bit {@code
     * keys} (given in the current order), ties keeping their current relative
     * position.
     */
    private static int[] pass(int[] order, int[] keys, long[] packed,
            int n) {
        for (int i = 0; i < n; i++) {
            // Unsigned key into signed long order: flip the top bit. Position in the low 32 bits.
            packed[i] = ((long) (keys[i] ^ Integer.MIN_VALUE) << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed, 0, n);
        int[] next = new int[n];
        for (int j = 0; j < n; j++) {
            next[j] = order[(int) packed[j]];
        }
        return next;
    }

    /**
     * Two passes over the sign-normalised 64-bit value: low half first, then
     * high half.
     */
    private static int[] passes64(
            int[] order,
            VectorBuffers k,
            boolean isDouble,
            boolean desc,
            long[] packed,
            int[] keyScratch,
            int n) {
        long[] normalised = new long[n];
        for (int i = 0; i < n; i++) {
            int row = order[i];
            if (k.isNull(row)) {
                normalised[i] = 0L;
                continue;
            }
            long v = isDouble ? doubleKey(k.getDouble(row)) : (k.getLong(row) ^ Long.MIN_VALUE);
            normalised[i] = desc ? ~v : v;
        }
        for (int i = 0; i < n; i++) {
            keyScratch[i] = (int) normalised[i];
        }
        int[] afterLow = pass(order, keyScratch, packed, n);
        // The low pass permuted the rows; carry the high halves along through the same permutation.
        for (int j = 0; j < n; j++) {
            keyScratch[j] = (int) (normalised[(int) packed[j]] >>> 32);
        }
        return pass(afterLow, keyScratch, packed, n);
    }

    /**
     * Four passes over a 128-bit value: the low limb is unsigned as it is, the
     * high limb is sign-normalised like an int64; low limb halves first, then
     * the high limb's. Between passes the limb arrays follow the permutation
     * the pass produced.
     */
    private static int[] passes128(int[] order, VectorBuffers k, boolean desc,
            long[] packed, int[] keyScratch, int n) {
        long[] lo = new long[n];
        long[] hi = new long[n];
        MemorySegment data = k.data();
        for (int i = 0; i < n; i++) {
            int row = order[i];
            if (k.isNull(row)) {
                continue; // both limbs stay 0: a constant key keeps null rows in their relative order
            }
            long l = Decimal128.lo(data, row);
            long h = Decimal128.hi(data, row) ^ Long.MIN_VALUE;
            lo[i] = desc ? ~l : l;
            hi[i] = desc ? ~h : h;
        }
        for (int i = 0; i < n; i++) {
            keyScratch[i] = (int) lo[i];
        }
        order = pass(order, keyScratch, packed, n);
        long[] lo2 = new long[n];
        long[] hi2 = new long[n];
        for (int j = 0; j < n; j++) {
            int from = (int) packed[j];
            lo2[j] = lo[from];
            hi2[j] = hi[from];
            keyScratch[j] = (int) (lo[from] >>> 32);
        }
        order = pass(order, keyScratch, packed, n);
        for (int j = 0; j < n; j++) {
            int from = (int) packed[j];
            hi[j] = hi2[from];
            keyScratch[j] = (int) hi2[from];
        }
        order = pass(order, keyScratch, packed, n);
        for (int j = 0; j < n; j++) {
            keyScratch[j] = (int) (hi[(int) packed[j]] >>> 32);
        }
        return pass(order, keyScratch, packed, n);
    }

    /**
     * Total order on doubles as an unsigned 64-bit key, matching Spark's
     * comparison: {@code -0.0} and {@code 0.0} are equal, NaN is greater than
     * every other value and equal to itself.
     */
    static long doubleKey(double d) {
        if (d == 0.0) {
            d = 0.0; // folds -0.0
        }
        long bits = Double.doubleToLongBits(d); // canonical NaN
        // Positive doubles (bits >= 0) keep their order once the sign bit is flipped; negative ones
        // are inverted so that more-negative values come first.
        return bits >= 0 ? bits ^ Long.MIN_VALUE : ~bits;
    }

    private static final int SHORT_STRING = 8;

    private static int[] passesUtf8(int[] order, VectorBuffers k, boolean desc,
            long[] packed, int[] keyScratch, int n) {
        if (maxLength(k) <= SHORT_STRING) {
            return shortStringPasses(order, k, desc, packed, keyScratch, n);
        }
        int[] rank = ranks(k, n);
        for (int i = 0; i < n; i++) {
            keyScratch[i] = flip(rank[order[i]], desc);
        }
        return pass(order, keyScratch, packed, n);
    }

    /**
     * Length pass, then the two halves of the zero-padded 8-byte prefix as
     * unsigned keys.
     */
    private static int[] shortStringPasses(int[] order, VectorBuffers k, boolean desc,
            long[] packed, int[] keyScratch, int n) {
        long[] prefix = new long[n];
        int[] len = new int[n];
        for (int i = 0; i < n; i++) {
            int row = order[i];
            if (k.isNull(row)) {
                prefix[i] = 0L;
                len[i] = 0;
            } else {
                byte[] b = k.getUtf8Bytes(row);
                long p = 0L;
                for (int j = 0; j < b.length; j++) {
                    p |= (b[j] & 0xFFL) << (56 - 8 * j);
                }
                prefix[i] = desc ? ~p : p;
                len[i] = flip(b.length, desc);
            }
        }
        // Least significant first: among equal prefixes the shorter string is a prefix of the longer
        // and sorts first.
        int[] afterLen = pass(order, len, packed, n);
        long[] carried = new long[n];
        for (int j = 0; j < n; j++) {
            carried[j] = prefix[(int) packed[j]];
        }
        for (int j = 0; j < n; j++) {
            keyScratch[j] = (int) carried[j];
        }
        int[] afterLow = pass(afterLen, keyScratch, packed, n);
        for (int j = 0; j < n; j++) {
            keyScratch[j] = (int) (carried[(int) packed[j]] >>> 32);
        }
        return pass(afterLow, keyScratch, packed, n);
    }

    private static int maxLength(VectorBuffers k) {
        if (k.isDictionaryEncoded()) {
            return maxLength(k.dictionary());
        }
        MemorySegment off = k.offsets();
        int n = k.length();
        int max = 0;
        int prev = off.get(VectorBuffers.LE_INT, 0);
        for (int i = 1; i <= n; i++) {
            int o = off.get(VectorBuffers.LE_INT, (long) i << 2);
            max = Math.max(max, o - prev);
            prev = o;
        }
        return max;
    }

    /**
     * Dense rank of every row's string (nulls get 0, which the null pass
     * separates out anyway): rows are merge-sorted by bytes, then equal
     * neighbours share a rank.
     */
    static int[] ranks(VectorBuffers k, int n) {
        if (k.isDictionaryEncoded()) {
            VectorBuffers dict = k.dictionary();
            int[] dictRank = ranks(dict, dict.length());
            int[] rank = new int[n];
            for (int i = 0; i < n; i++) {
                rank[i] = k.isNull(i) ? 0 : dictRank[k.getInt(i)];
            }
            return rank;
        }
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        mergeSort(idx, new int[n], 0, n, k);
        int[] rank = new int[n];
        int r = 0;
        for (int j = 0; j < n; j++) {
            if (j > 0 && compareUtf8(k, idx[j - 1], idx[j]) != 0) {
                r++;
            }
            rank[idx[j]] = k.isNull(idx[j]) ? 0 : r + 1;
        }
        return rank;
    }

    private static void mergeSort(int[] a, int[] tmp, int from,
            int to, VectorBuffers k) {
        if (to - from < 2) {
            return;
        }
        int mid = (from + to) >>> 1;
        mergeSort(a, tmp, from, mid, k);
        mergeSort(a, tmp, mid, to, k);
        int i = from;
        int j = mid;
        int o = from;
        while (i < mid && j < to) {
            tmp[o++] = compareUtf8(k, a[i], a[j]) <= 0 ? a[i++] : a[j++];
        }
        while (i < mid) {
            tmp[o++] = a[i++];
        }
        while (j < to) {
            tmp[o++] = a[j++];
        }
        System.arraycopy(tmp, from, a, from, to - from);
    }

    /** Unsigned lexicographic byte comparison, nulls before everything. */
    static int compareUtf8(VectorBuffers k, int a, int b) {
        boolean na = k.isNull(a);
        boolean nb = k.isNull(b);
        if (na || nb) {
            return Boolean.compare(!na, !nb);
        }
        MemorySegment off = k.offsets();
        MemorySegment data = k.data();
        int sa = off.get(VectorBuffers.LE_INT, (long) a << 2);
        int ea = off.get(VectorBuffers.LE_INT, (long) (a + 1) << 2);
        int sb = off.get(VectorBuffers.LE_INT, (long) b << 2);
        int eb = off.get(VectorBuffers.LE_INT, (long) (b + 1) << 2);
        int la = ea - sa;
        int lb = eb - sb;
        long mismatch = MemorySegment.mismatch(data, sa, ea, data, sb, eb);
        if (mismatch == -1) {
            return 0;
        }
        if (mismatch >= Math.min(la, lb)) {
            return Integer.compare(la, lb);
        }
        return Integer.compare(data.get(ValueLayout.JAVA_BYTE, sa + mismatch) & 0xFF, data.get(ValueLayout.JAVA_BYTE, sb + mismatch) & 0xFF);
    }
}
