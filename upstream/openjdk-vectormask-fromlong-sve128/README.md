# OpenJDK: `VectorMask.fromLong` is not intrinsified on AArch64 SVE at 128 bits (Graviton4)

Found under #253 (the kernels on Graviton4), 2026-09-25. The draft report is `OPENJDK-ISSUE.md`, and
`Repro.java` is a JDK-only reproducer.

## What we saw

On 9 x m8g.4xlarge (Neoverse V2, SVE2, 128-bit vectors), JDK 25 (Corretto 25.0.4.1), the kernels' JMH
suite was run in three modes:

| mode | JIT codegen | the kernels' mask construction |
|---|---|---|
| `sve` | SVE (the JVM default) | `VectorMask.fromLong` (`Platform.MASK_REGISTERS`) |
| `sve-neon` | SVE | broadcast-AND-compare (`-Dsparkvector.platform=neon`) |
| `neon` | NEON (`-XX:UseSVE=0`) | broadcast-AND-compare |

The mask-heavy kernels run slower with `fromLong` than with the broadcast-AND-compare mask, under the
same SVE codegen (`sve` / `sve-neon`, 2 forks x 5 iterations, small errors):

| kernel | 1 % nulls | 30 % nulls |
|---|---|---|
| `minDouble_simd` | 0.77x | 0.50x |
| `sumLong_simd` | 0.78x | 0.62x |
| `sumDouble_simd` | 0.76x | 0.93x |
| `sumMaskPath` (4 groups) | 0.72x | |

With `-XX:+PrintIntrinsics`, the `sve` mode inlines `jdk.incubator.vector.VectorMask::lambda$fromLong$0`,
the Java fallback of `VectorSupport.fromBitsCoerced`, four times; the NEON modes never do. So every mask
built from a validity word goes through the Java fallback: a loop that sets lanes from bits and loads
the mask from a boolean array.

On x86 with AVX-512 (the development host, 8 long lanes) the same reproducer intrinsifies `fromBitsCoerced` ("late inline
succeeded") and both mask forms run at the same speed (`fromLong / broadcast = 1.00`).

## Why it happens (reading of the JDK 25 sources, to confirm on the node)

`src/hotspot/cpu/aarch64/aarch64_vector.ad`, `Matcher::match_rule_supported_vector`:

```
case Op_VectorLongToMask:
  if (vlen > 64 || !VM_Version::supports_svebitperm()) {
    return false;
  }
```

`VectorMask.fromLong` becomes `VectorLongToMask` only on CPUs that report the SVE2 BitPerm extension
(`svebitperm`, for `BDEP`); everywhere else, NEON included, it falls back to Java. The cause on
Graviton4 is one of two:

1. **The node does not report `svebitperm`.** Then the JDK behaves as designed, and the report asks for a
   fallback lowering that doesn't need BitPerm. A broadcast of the long, an AND with the lane bits
   and a compare is 3 instructions on NEON and SVE alike, and it's what the kernels now do by hand.
2. **The node reports `svebitperm` and the intrinsic still fails.** Then it's a bug in the SVE
   `VectorLongToMask` path at 128 bits (vlen 2 for long lanes, 4 for int), and the report is a bug report.

The first step before filing is to find out which (see *Before filing* in `OPENJDK-ISSUE.md`).

## What spark-vector does meanwhile

- #253's decision 1 split `Platform.MASK_REGISTERS` from `NATIVE_COMPRESS`. SVE keeps native `compress`
  (1.36x on 50 % selectivity) and builds masks with broadcast-AND-compare (`AggKernels.maskLBroadcast`
  and siblings); AVX-512 keeps `fromLong`.
- Re-measure on each JDK update: run `Repro.java` on a Graviton4 node. Once
  `fromLong / broadcast` is ~1.0 and `PrintIntrinsics` shows `fromBitsCoerced ... (intrinsic)`, set
  `MASK_REGISTERS` back for SVE.

## Files

- `Repro.java`: a masked long sum over a validity bitmap, with both mask forms, correctness-checked
  against each other, JDK only (no Maven).
- `OPENJDK-ISSUE.md`: the report text for bugs.openjdk.org (component `hotspot/compiler`, or
  `core-libs/jdk.incubator.vector`), or the panama-dev mailing list.
