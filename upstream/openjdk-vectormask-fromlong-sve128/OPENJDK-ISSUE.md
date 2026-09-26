# Upstream report: `VectorMask.fromLong` slow on AArch64 SVE, 128-bit vectors (Neoverse V2)

OpenJDK takes bug reports at https://bugs.openjdk.org (JBS). Without a JBS account, send reports
through https://bugreport.java.com or to the panama-dev mailing list
(panama-dev@openjdk.org, the Vector API's list). Try the list first: someone there can say quickly
whether this is known.

## Before filing

1. Done 2026-09-26: the CPU flags and `Repro.java` on a Graviton4 node. The output is below.
2. Still to do: run it on the latest JDK EA build (jdk.java.net) and say whether it reproduces.
3. Still to do: search JBS for `VectorLongToMask SVE` / `fromLong aarch64` so this doesn't duplicate an open
   issue.
4. Optional, and it makes the report sharper: find the failing call site with
   `-XX:+PrintCompilation -XX:+PrintInlining`, or split `Repro` so that only the `fromLong` loop is compiled.

---

**Title:** C2: `VectorMask.fromLong` is 1.6x slower than broadcast+and+compare on AArch64 SVE with 128-bit vectors (Neoverse V2, svebitperm present)

**Component:** hotspot/compiler (Vector API intrinsics)

**Affects:** JDK 25 (Corretto 25.0.4.1, build 25.0.4.1+8-LTS). Latest EA: _to check_.

**Environment:** AWS m8g.4xlarge (Graviton4, Arm Neoverse V2, CPU part 0xd4f), Linux arm64,
`UseSVE=2` and `MaxVectorSize=16` (both defaults). CPU features include `sve sve2 svebitperm`:

```
Features : fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 flagm2 frint svei8mm svebf16 i8mm bf16 dgh rng bti
```

**Description**

A masked long sum over a validity bitmap builds one mask per vector of rows. Built with
`VectorMask.fromLong(species, bits)`, it runs 1.62x slower than the same loop with the mask built as
`LongVector.broadcast(S, bits).and(laneBits).compare(NE, 0)`. The CPU reports SVE2 BitPerm, which
`aarch64_vector.ad` requires for `Op_VectorLongToMask`, so `fromLong` is expected to be at least
as fast.

With `-XX:+PrintIntrinsics`, most `VectorSupport::fromBitsCoerced` calls are intrinsified ("late
inline succeeded"), but one or two are `failed to inline (intrinsic)`, and
`jdk.incubator.vector.VectorMask::lambda$fromLong$0`, the Java fallback, is inlined as hot. So either
some `fromLong` call sites miss the intrinsic, or the SVE lowering of `VectorLongToMask` at vlen 2
costs more than a broadcast + AND + compare.

In a columnar query engine's aggregation kernels (masked `min`/`sum` with nulls), JMH on the same
machine and JDK measured `fromLong` at 0.50-0.78x of the broadcast mask. We now build masks by hand
on SVE.

With `-XX:UseSVE=0` (NEON, where `VectorLongToMask` is not supported by design), `fromLong` is 13x
slower than the broadcast mask. A NEON lowering such as broadcast + AND + `cmtst` would remove the
cliff.

On x86 AVX-512 the same reproducer intrinsifies `fromBitsCoerced`, and both forms run at the same speed.

**Reproducer:** `Repro.java` (attached), JDK only:

```
javac --add-modules jdk.incubator.vector Repro.java
java --add-modules jdk.incubator.vector Repro 8388608 30 15
java --add-modules jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+PrintIntrinsics Repro 2097152 30 6 | grep -E "fromBitsCoerced|lambda.fromLong"
```

Output on Graviton4, JDK 25.0.4.1:

```
species Species[long, 2, S_128_BIT] (2 lanes), 8388608 rows, 30% nulls
fromLong                6.03 ms (best)
broadcast-and-cmp       3.73 ms (best)
fromLong / broadcast = 1.62

-XX:UseSVE=0:
fromLong               82.64 ms (best)
broadcast-and-cmp       6.05 ms (best)
fromLong / broadcast = 13.66

PrintIntrinsics (counts):
  7 @ 19   jdk.internal.vm.vector.VectorSupport::fromBitsCoerced (35 bytes)   (intrinsic)   late inline succeeded
  3 @ 25   jdk.internal.vm.vector.VectorSupport::fromBitsCoerced (35 bytes)   (intrinsic)   late inline succeeded
  2 @ 26   jdk.internal.vm.vector.VectorSupport::fromBitsCoerced (35 bytes)   (intrinsic)   late inline succeeded
  3 @ 39   jdk.internal.vm.vector.VectorSupport::fromBitsCoerced (35 bytes)   (intrinsic)   late inline succeeded
  1 @ 39   jdk.internal.vm.vector.VectorSupport::fromBitsCoerced (35 bytes)   failed to inline: failed to inline (intrinsic)
  2 @ 5    jdk.incubator.vector.VectorMask::lambda$fromLong$0 (108 bytes)   inline (hot)
```

Output on x86 AVX-512 (same JDK), for comparison:

```
species Species[long, 8, S_512_BIT] (8 lanes), 8388608 rows, 30% nulls
fromLong                3.58 ms (best)
broadcast-and-cmp       3.59 ms (best)
fromLong / broadcast = 1.00
```

**Expected:** on an SVE2 BitPerm CPU, `fromLong` is compiled to predicate instructions and runs no
slower than the broadcast-AND-compare mask.

**Workaround:** build the mask as `LongVector.broadcast(S, bits).and(laneBits).compare(NE, 0)`.
