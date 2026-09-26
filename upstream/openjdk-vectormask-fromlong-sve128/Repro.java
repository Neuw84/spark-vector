import java.util.Random;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * VectorMask.fromLong vs a broadcast-AND-compare mask, for a masked long sum over a validity bitmap
 * (the shape of spark-vector's AggKernels with nulls). On AArch64 SVE at MaxVectorSize=16 (Graviton4,
 * Neoverse V2) fromLong is not intrinsified and the fromLong variant runs at 0.5-0.8x of the other.
 *
 *   javac --add-modules jdk.incubator.vector Repro.java
 *   java --add-modules jdk.incubator.vector Repro                       # timings
 *   java --add-modules jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+PrintIntrinsics Repro \
 *     | grep -i -E "fromBitsCoerced|fromLong"                          # intrinsic decisions
 *
 * Arguments: [rows (default 8M)] [null percent (default 30)] [rounds (default 15)].
 */
public final class Repro {
    static final VectorSpecies<Long> L = LongVector.SPECIES_PREFERRED;
    static final int LANES = L.length();
    static final long LANE_MASK = LANES == 64 ? -1L : (1L << LANES) - 1;
    static final LongVector LANE_BITS = LongVector.fromArray(L, laneBits(), 0);

    static long[] laneBits() {
        long[] b = new long[LANES];
        for (int i = 0; i < LANES; i++) b[i] = 1L << i;
        return b;
    }

    /** The lanes' validity bits, from a Spark/Arrow-style LSB-first bitmap. */
    static long bits(long[] validity, int row) {
        long word = validity[row >>> 6] >>> (row & 63);
        return word & LANE_MASK;
    }

    static long sumFromLong(long[] values, long[] validity, int n) {
        LongVector acc = LongVector.zero(L);
        int i = 0;
        for (; i <= n - LANES; i += LANES) {
            VectorMask<Long> m = VectorMask.fromLong(L, bits(validity, i));
            acc = acc.add(LongVector.fromArray(L, values, i), m);
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail(values, validity, i, n);
    }

    static long sumBroadcast(long[] values, long[] validity, int n) {
        LongVector acc = LongVector.zero(L);
        int i = 0;
        for (; i <= n - LANES; i += LANES) {
            VectorMask<Long> m = LongVector.broadcast(L, bits(validity, i)).and(LANE_BITS).compare(VectorOperators.NE, 0L);
            acc = acc.add(LongVector.fromArray(L, values, i), m);
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail(values, validity, i, n);
    }

    static long tail(long[] values, long[] validity, int from, int n) {
        long s = 0;
        for (int i = from; i < n; i++) if ((validity[i >>> 6] >>> (i & 63) & 1) != 0) s += values[i];
        return s;
    }

    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 8 << 20;
        int nullPct = args.length > 1 ? Integer.parseInt(args[1]) : 30;
        int rounds = args.length > 2 ? Integer.parseInt(args[2]) : 15;
        Random r = new Random(253);
        long[] values = new long[n];
        long[] validity = new long[(n + 63) >>> 6];
        for (int i = 0; i < n; i++) {
            values[i] = r.nextInt(1_000_000);
            if (r.nextInt(100) >= nullPct) validity[i >>> 6] |= 1L << (i & 63);
        }
        long a = sumFromLong(values, validity, n), b = sumBroadcast(values, validity, n);
        if (a != b) throw new AssertionError("results differ: " + a + " vs " + b);
        System.out.printf("species %s (%d lanes), %d rows, %d%% nulls%n", L, LANES, n, nullPct);
        long bestF = Long.MAX_VALUE, bestB = Long.MAX_VALUE, sink = 0;
        for (int k = 0; k < rounds; k++) {
            long t0 = System.nanoTime();
            sink += sumFromLong(values, validity, n);
            long t1 = System.nanoTime();
            sink += sumBroadcast(values, validity, n);
            long t2 = System.nanoTime();
            if (k >= rounds / 3) { // after warm-up
                bestF = Math.min(bestF, t1 - t0);
                bestB = Math.min(bestB, t2 - t1);
            }
        }
        System.out.printf("fromLong            %8.2f ms (best)%n", bestF / 1e6);
        System.out.printf("broadcast-and-cmp   %8.2f ms (best)%n", bestB / 1e6);
        System.out.printf("fromLong / broadcast = %.2f (sink %d)%n", (double) bestF / bestB, sink);
    }
}
