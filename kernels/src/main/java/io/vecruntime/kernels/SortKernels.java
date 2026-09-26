/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Columnar sort: a permutation of row indices ordered by one or more key
 * columns; {@link GatherKernels} applies it to the output columns.
 *
 * <p>The sort is least-significant-key-first over order-preserving unsigned key
 * passes. Each pass is a stable LSD radix sort of the current positions by a
 * 32- or 64-bit key: one counting sort per 8-bit digit (four or eight), each
 * O(n), a digit whose histogram has a single bucket skipped (the high digits of
 * small ints, dates and dense ranks, most of the time) and a pass whose keys
 * are already in order skipped whole (presorted and constant inputs).
 * Processing keys from the last to the first yields the lexicographic order.
 * Values are mapped to unsigned keys that compare like the value does:
 *
 * <ul>
 *   <li>int32 and booleans: one 32-bit pass;
 *   <li>int64 and doubles: one 64-bit pass over the sign-normalised bits,
 *       doubles first put into total order the way Spark compares them ({@code
 *       -0.0 == 0.0}, NaN above everything, NaN equal to NaN);
 *   <li>UTF8: strings up to 8 bytes long take a length pass and a 64-bit pass
 *       over the zero-padded big-endian prefix (which orders bytes unsigned
 *       like Spark's {@code UTF8String}); longer strings are ranked once by a
 *       stable merge sort with a byte comparator and the rank takes one pass;
 *   <li>decimal128: four 32-bit passes over the two limbs;
 *   <li>nulls: one final pass per key on the null flag, placing them first or
 *       last.
 * </ul>
 *
 * <p>Descending order inverts the key bits. The Vector API has no sort
 * primitive: the lanes work on the key normalisation (sign flips, the double
 * total order, the descending inversion, all over the gathered key array); the
 * histograms and the scatter are scalar, as is the gather of the keys through
 * the current permutation. Scratch is two {@code int[n]} position buffers, a
 * key buffer of each width and one {@code int[n]} output buffer, allocated once
 * per sort (#285).
 */
public final class SortKernels {

    private SortKernels() {}

    private static final VectorSpecies<Integer> I = Species.I;
    private static final VectorSpecies<Long> L = Species.L;
    private static final VectorSpecies<Double> D = Species.D;

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
        Passes p = new Passes(n);
        int[] key = p.key;
        for (int c = keys.length - 1; c >= 0; c--) {
            VectorBuffers k = keys[c];
            boolean desc = !ascending[c];
            switch (k.type()) {
                // Null rows get a constant key in every value pass so that they keep their relative
                // order (the reference sort is stable and so must this be); the null pass below then moves
                // them to the front or the back as a block.
                case INT32 -> {
                    int[] values = new int[n];
                    MemorySegment.copy(k.data(), VectorBuffers.LE_INT, 0, values, 0,
                            n);
                    long[] valid = validityWords(k, n);
                    if (valid == null) {
                        for (int i = 0; i < n; i++) {
                            key[i] = values[order[i]];
                        }
                    } else {
                        for (int i = 0; i < n; i++) {
                            int row = order[i];
                            key[i] = isNull(valid, row) ? Integer.MIN_VALUE : values[row];
                        }
                    }
                    normalise32(key, desc, n);
                    order = p.pass(order);
                }
                case BOOL -> {
                    for (int i = 0; i < n; i++) {
                        int row = order[i];
                        key[i] = k.isNull(row) ? 0 : flip(k.getBoolean(row) ? 1 : 0, desc);
                    }
                    order = p.pass(order);
                }
                case INT64 -> order = passes64(order, k, false, desc, p, n);
                case FLOAT64 -> order = passes64(order, k, true, desc, p, n);
                case UTF8 -> order = passesUtf8(order, k, desc, p, n);
                case DECIMAL128 -> order = passes128(order, k, desc, p, n);
                default -> throw new IllegalArgumentException("unsupported sort key type " + k.type());
            }
            if (k.hasNulls()) {
                int nullKey = nullsFirst[c] ? 0 : 1;
                long[] valid = validityWords(k, n);
                for (int i = 0; i < n; i++) {
                    key[i] = isNull(valid, order[i]) ? nullKey : 1 - nullKey;
                }
                order = p.pass(order);
            }
        }
        return order;
    }

    private static int flip(int key, boolean desc) {
        return desc ? ~key : key;
    }

    /**
     * The validity bitmap as words (a set bit is a valid row), or null when
     * every row is valid.
     */
    private static long[] validityWords(VectorBuffers k, int n) {
        MemorySegment v = k.validity();
        if (v == null) {
            return null;
        }
        long[] words = new long[(n + 63) >>> 6];
        long bytes = Math.min(v.byteSize(), (long) words.length << 3);
        MemorySegment.copy(v, ValueLayout.JAVA_BYTE, 0, MemorySegment.ofArray(words), ValueLayout.JAVA_BYTE,
                0, bytes);
        return words;
    }

    private static boolean isNull(long[] valid, int row) {
        return valid != null && (valid[row >>> 6] & (1L << row)) == 0;
    }

    /**
     * The scratch of one sort and the stable pass over it. After {@link #pass}
     * {@code src[j]} is the position, before the pass, of the row now at {@code
     * j}: callers carry the other halves of a multi-pass key along through it.
     */
    static final class Passes {
        private static final int DIGITS = 4;
        private static final int DIGITS64 = 8;

        final int[] key;
        int[] src;
        private int[] posA;
        private int[] posB;
        private int[] keyB;
        private long[] key64B;
        private int[] out;
        private final int[][] count = new int[DIGITS64][256];
        private final int n;

        Passes(int n) {
            this.n = n;
            key = new int[n];
            posA = new int[n];
            posB = new int[n];
            keyB = new int[n];
            out = new int[n];
            src = posA;
        }

        /**
         * One stable pass: reorders {@code order} by the unsigned 32-bit {@code
         * key} (given in the current order), ties keeping their current
         * relative position. Returns the new order, whose array is one of the
         * pass's buffers; the argument becomes a buffer of the pass.
         */
        int[] pass(int[] order) {
            int[] k = key;
            boolean sorted = true;
            for (int i = 1;
                 i < n && sorted;
                 i++) {
                sorted = Integer.compareUnsigned(k[i - 1], k[i]) <= 0;
            }
            if (sorted) {
                return identity(order); // already in order (equal keys included): the pass is a no-op
            }
            for (int d = 0; d < DIGITS; d++) {
                java.util.Arrays.fill(count[d], 0);
            }
            for (int i = 0; i < n; i++) {
                int v = k[i];
                count[0][v & 0xFF]++;
                count[1][(v >>> 8) & 0xFF]++;
                count[2][(v >>> 16) & 0xFF]++;
                count[3][v >>> 24]++;
            }
            boolean first = true;
            int[] a = posA;
            int[] b = posB;
            // The keys travel with the positions through every scatter, so each digit reads its keys
            // sequentially; only the scatter's writes are out of order.
            int[] ka = k;
            int[] kb = keyB;
            for (int d = 0; d < DIGITS; d++) {
                int[] c = count[d];
                if (singleBucket(c)) {
                    continue;
                }
                prefixSums(c);
                int shift = d << 3;
                if (first) {
                    for (int i = 0; i < n; i++) {
                        int kv = k[i];
                        int dst = c[(kv >>> shift) & 0xFF]++;
                        b[dst] = i;
                        kb[dst] = kv;
                    }
                    first = false;
                } else {
                    for (int i = 0; i < n; i++) {
                        int kv = ka[i];
                        int dst = c[(kv >>> shift) & 0xFF]++;
                        b[dst] = a[i];
                        kb[dst] = kv;
                    }
                }
                int[] t = a;
                a = b;
                b = t;
                int[] kt = ka;
                ka = kb;
                kb = kt; // the key array itself is scratch from the second digit on: callers refill it per pass
            }
            return finish(order, a, b);
        }

        /**
         * One stable pass by an unsigned 64-bit key: eight digits over the long
         * directly, so a 64-bit value needs no carrying of its high half
         * through a first pass's permutation. {@code key64} is scratch
         * afterwards.
         */
        int[] pass64(int[] order, long[] key64) {
            boolean sorted = true;
            for (int i = 1;
                 i < n && sorted;
                 i++) {
                sorted = Long.compareUnsigned(key64[i - 1], key64[i]) <= 0;
            }
            if (sorted) {
                return identity(order);
            }
            for (int d = 0; d < DIGITS64; d++) {
                java.util.Arrays.fill(count[d], 0);
            }
            for (int i = 0; i < n; i++) {
                long v = key64[i];
                for (int d = 0; d < DIGITS64; d++) {
                    count[d][(int) ((v >>> (d << 3)) & 0xFF)]++;
                }
            }
            if (key64B == null) {
                key64B = new long[n];
            }
            boolean first = true;
            int[] a = posA;
            int[] b = posB;
            long[] ka = key64;
            long[] kb = key64B;
            for (int d = 0; d < DIGITS64; d++) {
                int[] c = count[d];
                if (singleBucket(c)) {
                    continue;
                }
                prefixSums(c);
                int shift = d << 3;
                if (first) {
                    for (int i = 0; i < n; i++) {
                        long kv = key64[i];
                        int dst = c[(int) ((kv >>> shift) & 0xFF)]++;
                        b[dst] = i;
                        kb[dst] = kv;
                    }
                    first = false;
                } else {
                    for (int i = 0; i < n; i++) {
                        long kv = ka[i];
                        int dst = c[(int) ((kv >>> shift) & 0xFF)]++;
                        b[dst] = a[i];
                        kb[dst] = kv;
                    }
                }
                int[] t = a;
                a = b;
                b = t;
                long[] kt = ka;
                ka = kb;
                kb = kt;
            }
            return finish(order, a, b);
        }

        private int[] identity(int[] order) {
            int[] a = posA;
            for (int i = 0; i < n; i++) {
                a[i] = i;
            }
            src = a;
            return order;
        }

        private int[] finish(int[] order, int[] a, int[] b) {
            posA = a;
            posB = b;
            src = a;
            int[] next = out;
            for (int j = 0; j < n; j++) {
                next[j] = order[a[j]];
            }
            out = order;
            return next;
        }

        private static void prefixSums(int[] c) {
            int sum = 0;
            for (int bucket = 0; bucket < c.length; bucket++) {
                int t = c[bucket];
                c[bucket] = sum;
                sum += t;
            }
        }

        private boolean singleBucket(int[] c) {
            for (int bucket = 0; bucket < c.length; bucket++) {
                if (c[bucket] == n) {
                    return true;
                }
                if (c[bucket] != 0) {
                    return false;
                }
            }
            return false;
        }
    }

    /**
     * Sign-normalises the gathered int32 keys (null rows hold MIN_VALUE, so
     * they map to 0).
     */
    private static void normalise32(int[] key, boolean desc, int n) {
        int i = 0;
        int bound = I.loopBound(n);
        IntVector min = IntVector.broadcast(I, Integer.MIN_VALUE);
        for (; i < bound; i += I.length()) {
            IntVector v = IntVector.fromArray(I, key, i).lanewise(VectorOperators.XOR, min);
            if (desc) {
                v = v.lanewise(VectorOperators.NOT);
            }
            v.intoArray(key, i);
        }
        for (; i < n; i++) {
            key[i] = flip(key[i] ^ Integer.MIN_VALUE, desc);
        }
    }

    /**
     * Sign-normalises gathered int64 keys in place: {@code v ^ MIN_VALUE},
     * inverted when descending. Null rows must hold {@code MIN_VALUE} so that
     * they map to the constant 0.
     */
    private static void normalise64(long[] key, boolean desc, int n) {
        int i = 0;
        int bound = L.loopBound(n);
        LongVector min = LongVector.broadcast(L, Long.MIN_VALUE);
        for (; i < bound; i += L.length()) {
            LongVector v = LongVector.fromArray(L, key, i).lanewise(VectorOperators.XOR, min);
            if (desc) {
                v = v.lanewise(VectorOperators.NOT);
            }
            v.intoArray(key, i);
        }
        for (; i < n; i++) {
            long v = key[i] ^ Long.MIN_VALUE;
            key[i] = desc ? ~v : v;
        }
    }

    /**
     * Maps gathered doubles (as raw bits in {@code key}) to Spark's total order
     * in place; see {@link #doubleKey}. Null rows must hold the bits of {@code
     * -0.0}, which fold to 0.0 and then to the constant key of a positive zero
     * -- the same constant for every null row.
     */
    private static void normaliseDouble(long[] key, boolean desc, int n) {
        int i = 0;
        int bound = L.loopBound(n);
        LongVector min = LongVector.broadcast(L, Long.MIN_VALUE);
        LongVector nan = LongVector.broadcast(L, Double.doubleToLongBits(Double.NaN));
        DoubleVector zero = DoubleVector.zero(D);
        for (; i < bound; i += L.length()) {
            LongVector bits = LongVector.fromArray(L, key, i);
            DoubleVector d = bits.reinterpretAsDoubles();
            VectorMask<Double> isZero = d.compare(VectorOperators.EQ, zero); // -0.0 == 0.0
            VectorMask<Long> isNaN = d.compare(VectorOperators.NE, d).cast(L);
            bits = d.blend(zero, isZero)
                    .reinterpretAsLongs()
                    .blend(nan, isNaN);
            VectorMask<Long> negative = bits.compare(VectorOperators.LT, 0L);
            LongVector v = bits.lanewise(VectorOperators.XOR, min).blend(bits.lanewise(VectorOperators.NOT), negative);
            if (desc) {
                v = v.lanewise(VectorOperators.NOT);
            }
            v.intoArray(key, i);
        }
        for (; i < n; i++) {
            long v = doubleKey(Double.longBitsToDouble(key[i]));
            key[i] = desc ? ~v : v;
        }
    }

    /** One 64-bit pass over the sign-normalised value. */
    private static int[] passes64(int[] order, VectorBuffers k, boolean isDouble,
            boolean desc, Passes p, int n) {
        // The column's 64-bit words in row order (a double's raw bits read as a long), then gathered
        // through the current permutation from the array rather than through the segment per row.
        long[] values = new long[n];
        MemorySegment.copy(k.data(), VectorBuffers.LE_LONG, 0, values, 0,
                n);
        long[] valid = validityWords(k, n);
        long[] normalised = new long[n];
        long nullBits = isDouble ? Double.doubleToRawLongBits(-0.0) : Long.MIN_VALUE;
        if (valid == null) {
            for (int i = 0; i < n; i++) {
                normalised[i] = values[order[i]];
            }
        } else {
            for (int i = 0; i < n; i++) {
                int row = order[i];
                normalised[i] = isNull(valid, row) ? nullBits : values[row];
            }
        }
        if (isDouble) {
            normaliseDouble(normalised, desc, n);
        } else {
            normalise64(normalised, desc, n);
        }
        return p.pass64(order, normalised);
    }

    /**
     * Four passes over a 128-bit value: the low limb is unsigned as it is, the
     * high limb is sign-normalised like an int64; low limb halves first, then
     * the high limb's. Between passes the limb arrays follow the permutation
     * the pass produced.
     */
    private static int[] passes128(int[] order, VectorBuffers k, boolean desc,
            Passes p, int n) {
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
        int[] key = p.key;
        for (int i = 0; i < n; i++) {
            key[i] = (int) lo[i];
        }
        order = p.pass(order);
        long[] lo2 = new long[n];
        long[] hi2 = new long[n];
        int[] src = p.src;
        for (int j = 0; j < n; j++) {
            int from = src[j];
            lo2[j] = lo[from];
            hi2[j] = hi[from];
            key[j] = (int) (lo[from] >>> 32);
        }
        order = p.pass(order);
        src = p.src;
        for (int j = 0; j < n; j++) {
            int from = src[j];
            hi[j] = hi2[from];
            key[j] = (int) hi2[from];
        }
        order = p.pass(order);
        src = p.src;
        for (int j = 0; j < n; j++) {
            key[j] = (int) (hi[src[j]] >>> 32);
        }
        return p.pass(order);
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
    private static final ValueLayout.OfLong BE_LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);

    /**
     * Strings up to this length sort by 8-byte chunks; longer ones are ranked
     * by a merge sort.
     */
    private static final int CHUNKED_STRING = 64;

    private static int[] passesUtf8(int[] order, VectorBuffers k, boolean desc,
            Passes p, int n) {
        int maxLen = maxLength(k);
        if (maxLen <= SHORT_STRING) {
            return shortStringPasses(order, k, desc, p, n);
        }
        // A dictionary-encoded key ranks its dictionary (a few entries) and takes one pass over the codes;
        // the chunked passes are for plain columns, where the rank would merge-sort every row.
        if (maxLen <= CHUNKED_STRING && !k.isDictionaryEncoded()) {
            return chunkedStringPasses(order, k, desc, p, n,
                    (maxLen + SHORT_STRING - 1) / SHORT_STRING);
        }
        int[] rank = ranks(k, n);
        int[] key = p.key;
        for (int i = 0; i < n; i++) {
            key[i] = flip(rank[order[i]], desc);
        }
        return p.pass(order);
    }

    /**
     * Length pass, then one 64-bit pass over the zero-padded big-endian 8-byte
     * prefix.
     */
    private static int[] shortStringPasses(int[] order, VectorBuffers k, boolean desc,
            Passes p, int n) {
        long[] prefix = new long[n];
        int[] key = p.key;
        VectorBuffers values = k.isDictionaryEncoded() ? k.dictionary() : k;
        MemorySegment off = values.offsets();
        MemorySegment data = values.data();
        for (int i = 0; i < n; i++) {
            int row = order[i];
            if (k.isNull(row)) {
                prefix[i] = 0L;
                key[i] = 0;
            } else {
                int v = k.isDictionaryEncoded() ? k.getInt(row) : row;
                int start = off.get(VectorBuffers.LE_INT, (long) v << 2);
                int len = off.get(VectorBuffers.LE_INT, (long) (v + 1) << 2) - start;
                long pre;
                if (len == SHORT_STRING) {
                    pre = data.get(BE_LONG, start);
                } else {
                    pre = 0L;
                    for (int j = 0; j < len; j++) {
                        pre |= (data.get(ValueLayout.JAVA_BYTE, start + j) & 0xFFL) << (56 - 8 * j);
                    }
                }
                prefix[i] = desc ? ~pre : pre;
                key[i] = flip(len, desc);
            }
        }
        // Least significant first: among equal prefixes the shorter string is a prefix of the longer
        // and sorts first.
        int[] afterLen = p.pass(order);
        long[] carried = new long[n];
        int[] src = p.src;
        for (int j = 0; j < n; j++) {
            carried[j] = prefix[src[j]];
        }
        return p.pass64(afterLen, carried);
    }

    /**
     * Strings of up to {@link #CHUNKED_STRING} bytes (#377): a length pass,
     * then one 64-bit pass per 8-byte chunk from the last chunk to the first
     * (least significant first), each over the zero-padded big-endian chunk of
     * every row. A pass whose digits are all equal is skipped by {@code
     * pass64}, so a partition whose strings are all the same -- q67's window,
     * partitioned by the sort's first key -- costs one scan per chunk; the
     * merge-sort rank it replaced compared every pair on the way, n log n byte
     * comparisons for an order that was already known.
     */
    private static int[] chunkedStringPasses(int[] order, VectorBuffers k, boolean desc,
            Passes p, int n, int chunks) {
        int[] key = p.key;
        VectorBuffers values = k.isDictionaryEncoded() ? k.dictionary() : k;
        MemorySegment off = values.offsets();
        MemorySegment data = values.data();
        for (int i = 0; i < n; i++) {
            int row = order[i];
            if (k.isNull(row)) {
                key[i] = 0;
            } else {
                int v = k.isDictionaryEncoded() ? k.getInt(row) : row;
                int len = off.get(VectorBuffers.LE_INT, (long) (v + 1) << 2) - off.get(VectorBuffers.LE_INT, (long) v << 2);
                key[i] = flip(len, desc);
            }
        }
        order = p.pass(order);
        long[] chunk = new long[n];
        for (int c = chunks - 1; c >= 0; c--) {
            int from = c * SHORT_STRING;
            for (int i = 0; i < n; i++) {
                int row = order[i];
                long pre = 0L;
                if (!k.isNull(row)) {
                    int v = k.isDictionaryEncoded() ? k.getInt(row) : row;
                    int start = off.get(VectorBuffers.LE_INT, (long) v << 2);
                    int len = off.get(VectorBuffers.LE_INT, (long) (v + 1) << 2) - start;
                    if (len >= from + SHORT_STRING) {
                        pre = data.get(BE_LONG, start + from);
                    } else {
                        for (int j = from; j < len; j++) {
                            pre |= (data.get(ValueLayout.JAVA_BYTE, start + j) & 0xFFL) << (56 - 8 * (j - from));
                        }
                    }
                    if (desc) {
                        pre = ~pre;
                    }
                }
                chunk[i] = pre;
            }
            order = p.pass64(order, chunk);
        }
        return order;
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
        return Integer.signum(StringCompareKernels.compareBytes(data, sa, ea, data, sb, eb));
    }
}
