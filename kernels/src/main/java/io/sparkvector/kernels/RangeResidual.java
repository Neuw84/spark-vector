package io.sparkvector.kernels;

/**
 * The residual condition of an equi-join, `buildLane OP streamedValue` (either
 * side plus a literal offset), tested over one key's build rows when those rows
 * are stored contiguously -- the join clusters its build table by key, so a
 * probe row's candidates are the positions {@code [from, to)} of a clustered
 * lane rather than a chain of pointer hops (q72: 10^10 candidate pairs, a few
 * percent kept). The scan is one loop over an {@code int[]} or {@code long[]}
 * with the comparison folded into a three-bit mask over {@code
 * Integer.compare}, so there is no branch per operator and no call per pair;
 * the survivors' positions are appended to {@code out}.
 *
 * <p>Semantics match Spark's non-ANSI comparison of two lanes: a null on either
 * side fails the pair (a null streamed value produces no pairs, which the
 * caller handles), and the offsets are added in the lane's own width, wrapping
 * as Spark's {@code Add} and {@code DateAdd} do.
 */
public final class RangeResidual {
    private RangeResidual() {}

    /**
     * The three-bit mask of an operator over {@code Integer.compare(a, b) + 1}:
     * bit 0 = a &lt; b, bit 1 = a == b, bit 2 = a &gt; b.
     */
    public static int mask(CompareOp op) {
        switch (op) {
            case LT:
                return 0b001;
            case LE:
                return 0b011;
            case EQ:
                return 0b010;
            case NE:
                return 0b101;
            case GT:
                return 0b100;
            case GE:
                return 0b110;
            default:
                throw new IllegalArgumentException(op.toString());
        }
    }

    /**
     * Appends to {@code out[count..]} every position {@code p} in {@code [from,
     * to)} whose lane value (plus {@code laneOffset}, wrapping) compares to
     * {@code value} as {@code mask} allows and is valid under {@code validity}
     * (null = all valid; bit {@code p} of the clustered validity words). {@code
     * out} must have room for {@code to - from} more entries. Returns the new
     * count.
     */
    public static int scanInts(
            int[] lane,
            long[] validity,
            int from,
            int to,
            int laneOffset,
            int value,
            int mask,
            int[] out,
            int count) {
        if (validity == null) {
            for (int p = from; p < to; p++) {
                int c = threeWay(lane[p] + laneOffset, value);
                if (((mask >>> c) & 1) != 0) {
                    out[count++] = p;
                }
            }
        } else {
            for (int p = from; p < to; p++) {
                int c = threeWay(lane[p] + laneOffset, value);
                if (((mask >>> c) & 1) != 0 && ((validity[p >>> 6] >>> (p & 63)) & 1L) != 0L) {
                    out[count++] = p;
                }
            }
        }
        return count;
    }

    /** The {@code long} lane twin of {@link #scanInts}. */
    public static int scanLongs(
            long[] lane,
            long[] validity,
            int from,
            int to,
            long laneOffset,
            long value,
            int mask,
            int[] out,
            int count) {
        if (validity == null) {
            for (int p = from; p < to; p++) {
                int c = threeWay(lane[p] + laneOffset, value);
                if (((mask >>> c) & 1) != 0) {
                    out[count++] = p;
                }
            }
        } else {
            for (int p = from; p < to; p++) {
                int c = threeWay(lane[p] + laneOffset, value);
                if (((mask >>> c) & 1) != 0 && ((validity[p >>> 6] >>> (p & 63)) & 1L) != 0L) {
                    out[count++] = p;
                }
            }
        }
        return count;
    }

    /**
     * Gathers a mirrored lane into key-clustered order: {@code lane[p] =
     * mirror.ints[rows[p]]} (or the long twin), with the validity bits
     * clustered the same way, or a null validity when the mirror has no nulls.
     * The result is what {@link #scanInts} / {@link #scanLongs} scan.
     */
    public static HeapMirror cluster(HeapMirror mirror, int[] rows) {
        int n = rows.length;
        int[] ints = null;
        long[] longs = null;
        if (mirror.ints != null) {
            ints = new int[n];
            for (int p = 0; p < n; p++) {
                ints[p] = mirror.ints[rows[p]];
            }
        } else {
            longs = new long[n];
            for (int p = 0; p < n; p++) {
                longs[p] = mirror.longs[rows[p]];
            }
        }
        long[] validity = null;
        if (mirror.validity != null) {
            validity = new long[Bitmap.wordsFor(n)];
            for (int p = 0; p < n; p++) {
                if (mirror.isValid(rows[p])) {
                    validity[p >>> 6] |= 1L << (p & 63);
                }
            }
        }
        return HeapMirror.clustered(mirror.type, n, ints, longs, validity);
    }

    /**
     * 0, 1 or 2 for {@code a < b}, {@code a == b}, {@code a > b}: the bit of
     * the operator mask to test.
     */
    private static int threeWay(int a, int b) {
        return a < b
                ? 0
                : (a == b ? 1 : 2);
    }

    private static int threeWay(long a, long b) {
        return a < b
                ? 0
                : (a == b ? 1 : 2);
    }
}
