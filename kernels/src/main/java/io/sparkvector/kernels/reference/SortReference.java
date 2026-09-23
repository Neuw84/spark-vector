package io.sparkvector.kernels.reference;

import java.util.Arrays;
import java.util.Comparator;

import io.sparkvector.kernels.VectorBuffers;

/**
 * Oracle for {@link io.sparkvector.kernels.SortKernels}: a stable comparator
 * sort over boxed row indices implementing Spark's ordering rules directly
 * (null ordering per key, {@code -0.0 == 0.0}, NaN greatest, unsigned byte
 * order for strings).
 */
public final class SortReference {

    private SortReference() {}

    public static int[] sortIndices(VectorBuffers[] keys, boolean[] ascending, boolean[] nullsFirst,
            int n) {
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Comparator<Integer> cmp = (a, b) -> {
            for (int c = 0; c < keys.length; c++) {
                int r = compareKey(keys[c], a, b, !ascending[c], nullsFirst[c]);
                if (r != 0) {
                    return r;
                }
            }
            return 0;
        };
        Arrays.sort(idx, cmp); // stable (TimSort)
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    public static int compareKey(VectorBuffers k, int a, int b,
            boolean desc, boolean nullsFirst) {
        boolean na = k.isNull(a);
        boolean nb = k.isNull(b);
        if (na && nb) {
            return 0;
        }
        if (na) {
            return nullsFirst ? -1 : 1;
        }
        if (nb) {
            return nullsFirst ? 1 : -1;
        }
        int r = switch (k.type()) {
            case INT32 -> Integer.compare(k.getInt(a), k.getInt(b));
            case INT64 -> Long.compare(k.getLong(a), k.getLong(b));
            case FLOAT64 -> compareDoubles(k.getDouble(a), k.getDouble(b));
            case BOOL -> Boolean.compare(k.getBoolean(a), k.getBoolean(b));
            case UTF8 -> compareBytes(k.getUtf8Bytes(a), k.getUtf8Bytes(b));
            case DECIMAL128 -> k.getDecimal128(a).compareTo(k.getDecimal128(b));
            default -> throw new IllegalArgumentException("unsupported " + k.type());
        };
        return desc ? -r : r;
    }

    /** Spark's SQLOrderingUtil.compareDoubles. */
    public static int compareDoubles(double x, double y) {
        return x == y ? 0 : Double.compare(x, y);
    }

    /** Spark's UTF8String.compareTo: unsigned bytes, then length. */
    public static int compareBytes(byte[] a, byte[] b) {
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            int r = (a[i] & 0xFF) - (b[i] & 0xFF);
            if (r != 0) {
                return r;
            }
        }
        return a.length - b.length;
    }
}
