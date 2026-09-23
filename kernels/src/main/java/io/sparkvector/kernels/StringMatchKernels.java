package io.sparkvector.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * {@code startswith}, {@code endswith} and {@code contains} against a constant
 * pattern -- what Spark's {@code LikeSimplification} turns {@code LIKE
 * 'PROMO%'}, {@code LIKE '%x'} and {@code LIKE '%green%'} into. Byte-level, as
 * {@code UTF8String} is under the default {@code UTF8_BINARY} collation: a
 * UTF-8 pattern can only match at character boundaries, so byte matching is
 * character matching. The empty pattern matches every string.
 *
 * <p>A dictionary-encoded column is matched once per dictionary entry and the
 * verdicts gathered through the indices ({@link StringCompareKernels#gather}).
 * A plain column is matched row by row: prefix and suffix are one {@link
 * MemorySegment#mismatch} over a fixed range; contains scans for the pattern's
 * first byte and confirms candidates with {@code mismatch}. Blocks with no
 * active row are skipped.
 */
public final class StringMatchKernels {
    /** Which part of the string the pattern must occupy. */
    public enum Kind {
        PREFIX,
        SUFFIX,
        CONTAINS
    }

    private StringMatchKernels() {}

    /**
     * One result bit per row; null lanes get arbitrary bits, the caller shares
     * the validity.
     */
    public static void match(Kind kind, VectorBuffers a, byte[] pattern,
            MemorySegment active, MemorySegment out) {
        if (a.type() != VecType.UTF8) {
            throw new IllegalArgumentException("expected UTF8, got " + a.type());
        }
        int n = a.length();
        MemorySegment pat = MemorySegment.ofArray(pattern);
        VectorBuffers dict = a.dictionary();
        if (dict != null) {
            int m = dict.length();
            boolean[] verdict = new boolean[m];
            MemorySegment doff = dict.offsets();
            MemorySegment ddata = dict.data();
            for (int j = 0; j < m; j++) {
                int start = doff.get(VectorBuffers.LE_INT, (long) j << 2);
                int end = doff.get(VectorBuffers.LE_INT, (long) (j + 1) << 2);
                verdict[j] = matches(kind, ddata, start, end, pat, pattern);
            }
            StringCompareKernels.gather(a.data(), n, verdict, active, out);
            return;
        }
        MemorySegment off = a.offsets();
        MemorySegment data = a.data();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base);
            long word = 0L;
            int start = off.get(VectorBuffers.LE_INT, (long) base << 2);
            for (int k = 0; k < limit; k++) {
                int end = off.get(VectorBuffers.LE_INT, (long) (base + k + 1) << 2);
                if (matches(kind, data, start, end, pat, pattern)) {
                    word |= 1L << k;
                }
                start = end;
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /**
     * A {@code LIKE} pattern with several wildcards, {@code
     * [prefix%]tok1%tok2[%...][%suffix]}, as a multi-token matcher (#264): the
     * prefix must occupy the start, each token is found left to right with the
     * search resuming after the previous match (leftmost matches leave the most
     * room, so greedy is exact), and the suffix must occupy the end without
     * overlapping the last match. Empty tokens ({@code %%}) match trivially and
     * should be dropped by the caller. No {@code _} and no escapes: those stay
     * with Spark. One result bit per row, null lanes arbitrary, as {@link
     * #match}.
     */
    public static void matchTokens(VectorBuffers a, byte[] prefix, byte[][] tokens,
            byte[] suffix, MemorySegment active, MemorySegment out) {
        if (a.type() != VecType.UTF8) {
            throw new IllegalArgumentException("expected UTF8, got " + a.type());
        }
        int n = a.length();
        MemorySegment[] segs = new MemorySegment[tokens.length];
        for (int t = 0; t < tokens.length; t++) {
            segs[t] = MemorySegment.ofArray(tokens[t]);
        }
        MemorySegment pre = MemorySegment.ofArray(prefix);
        MemorySegment suf = MemorySegment.ofArray(suffix);
        VectorBuffers dict = a.dictionary();
        if (dict != null) {
            int m = dict.length();
            boolean[] verdict = new boolean[m];
            MemorySegment doff = dict.offsets();
            MemorySegment ddata = dict.data();
            for (int j = 0; j < m; j++) {
                int start = doff.get(VectorBuffers.LE_INT, (long) j << 2);
                int end = doff.get(VectorBuffers.LE_INT, (long) (j + 1) << 2);
                verdict[j] = matchesTokens(ddata, start, end, pre, prefix, segs,
                        tokens, suf, suffix);
            }
            StringCompareKernels.gather(a.data(), n, verdict, active, out);
            return;
        }
        MemorySegment off = a.offsets();
        MemorySegment data = a.data();
        for (int w = 0, words = Bitmap.wordsFor(n);
             w < words;
             w++) {
            if (active != null && Bitmap.wordAt(active, w, n) == 0L) {
                Bitmap.setWord(out, w, n, 0L);
                continue;
            }
            int base = w << 6, limit = Math.min(64, n - base);
            long word = 0L;
            int start = off.get(VectorBuffers.LE_INT, (long) base << 2);
            for (int k = 0; k < limit; k++) {
                int end = off.get(VectorBuffers.LE_INT, (long) (base + k + 1) << 2);
                if (matchesTokens(data, start, end, pre, prefix, segs,
                        tokens, suf, suffix)) {
                    word |= 1L << k;
                }
                start = end;
            }
            Bitmap.setWord(out, w, n, word);
        }
    }

    /**
     * Whether {@code s[start, end)} matches {@code prefix%tok...%suffix}; see
     * {@link #matchTokens}.
     */
    public static boolean matchesTokens(
            MemorySegment s,
            long start,
            long end,
            MemorySegment pre,
            byte[] prefix,
            MemorySegment[] segs,
            byte[][] tokens,
            MemorySegment suf,
            byte[] suffix) {
        long len = end - start;
        if (prefix.length + suffix.length > len) {
            return false;
        }
        if (prefix.length > 0 && MemorySegment.mismatch(s, start, start + prefix.length, pre, 0,
                prefix.length)
                >= 0) {
            return false;
        }
        long limit = end - suffix.length; // the suffix's region is off limits to the tokens
        if (suffix.length > 0 && MemorySegment.mismatch(s, limit, end, suf, 0, suffix.length) >= 0) {
            return false;
        }
        long pos = start + prefix.length;
        for (int t = 0; t < tokens.length; t++) {
            byte[] tok = tokens[t];
            int plen = tok.length;
            if (plen == 0) {
                continue;
            }
            long found = -1;
            byte first = tok[0];
            long last = limit - plen; // last position a match can start at
            for (long p = pos; p <= last; p++) {
                if (s.get(ValueLayout.JAVA_BYTE, p) == first && (plen == 1 || MemorySegment.mismatch(s, p + 1, p + plen, segs[t],
                        1, plen)
                        < 0)) {
                    found = p;
                    break;
                }
            }
            if (found < 0) {
                return false;
            }
            pos = found + plen;
        }
        return true;
    }

    /** Whether {@code s[start, end)} matches the pattern under {@code kind}. */
    public static boolean matches(Kind kind, MemorySegment s, long start,
            long end, MemorySegment pat, byte[] pattern) {
        int plen = pattern.length;
        long len = end - start;
        if (plen > len) {
            return false;
        }
        if (plen == 0) {
            return true;
        }
        switch (kind) {
            case PREFIX:
                return MemorySegment.mismatch(s, start, start + plen, pat, 0,
                        plen)
                        < 0;
            case SUFFIX:
                return MemorySegment.mismatch(s, end - plen, end, pat, 0,
                        plen)
                        < 0;
            default:
                byte first = pattern[0];
                long last = end - plen; // last position a match can start at
                for (long p = start; p <= last; p++) {
                    if (s.get(ValueLayout.JAVA_BYTE, p) == first && (plen == 1 || MemorySegment.mismatch(s, p + 1, p + plen, pat,
                            1, plen)
                            < 0)) {
                        return true;
                    }
                }
                return false;
        }
    }
}
