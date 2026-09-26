# Upstream report: `VectorMask.fromLong` not intrinsified on AArch64 SVE, 128-bit vectors (Neoverse V2)

OpenJDK takes bug reports at https://bugs.openjdk.org (JBS). Without a JBS account, send reports
through https://bugreport.java.com or to the panama-dev mailing list
(panama-dev@openjdk.org, the Vector API's list). Try the list first: someone there can say quickly
whether this is known or by design.

## Before filing

1. On a Graviton4 node (m8g), record the CPU features: `grep -m1 Features /proc/cpuinfo`.
   Note whether `svebitperm` is present, and add the line to the report.
2. Run `Repro.java` there on the same JDK: the timings, plus the `-XX:+PrintIntrinsics` lines for
   `fromBitsCoerced` and `fromLong`. Paste both into the report in place of the placeholders below.
3. Run it again on the latest JDK EA build (jdk.java.net), and say whether it reproduces.
4. Search JBS for `VectorLongToMask SVE` / `fromLong aarch64` so this doesn't duplicate an open issue.
   If `svebitperm` is missing (case 1 in the README), rewrite the title as an enhancement: "Lower
   `VectorLongToMask` without SVE2 BitPerm (and on NEON)".

---

**Title:** C2: `VectorMask.fromLong` falls back to Java on AArch64 SVE with 128-bit vectors (Neoverse V2)

**Component:** hotspot/compiler (Vector API intrinsics); also `core-libs/jdk.incubator.vector`

**Affects:** JDK 25 (Corretto 25.0.4.1, build 25.0.4.1+8-LTS). Latest EA: _to check_.

**Environment:** AWS m8g.4xlarge (Graviton4, Arm Neoverse V2), Linux arm64, `UseSVE=2` (the default),
`MaxVectorSize=16`. CPU features: _`/proc/cpuinfo` Features line here_.

**Description**

On this CPU, `VectorMask.fromLong(species, bits)` is not intrinsified. With `-XX:+PrintIntrinsics`,
C2 inlines the fallback lambda `jdk.incubator.vector.VectorMask::lambda$fromLong$0` of
`VectorSupport.fromBitsCoerced` rather than emitting a `VectorLongToMask` node. A mask built with a
broadcast, an AND with the lane bits and a compare is compiled to SVE instructions, and runs
1.3-2x faster in the same loops.

In a columnar query engine's aggregation kernels (masked `min` / `sum` over values with a validity
bitmap, one `fromLong` per vector of rows), JMH on JDK 25, same SVE codegen, `fromLong` against
broadcast-AND-compare:

| kernel | 1 % nulls | 30 % nulls |
|---|---|---|
| masked min, double | 0.77x | 0.50x |
| masked sum, long | 0.78x | 0.62x |
| masked sum, double | 0.76x | 0.93x |
| grouped masked sum, 4 groups | 0.72x | |

The same code on x86 AVX-512 intrinsifies `fromBitsCoerced` and both forms run at the same speed.

`aarch64_vector.ad` (`Matcher::match_rule_supported_vector`) only supports `Op_VectorLongToMask` when
`VM_Version::supports_svebitperm()`. _Either_: this CPU reports `svebitperm` and the intrinsic still
fails at vlen 2/4 (a bug); _or_ it doesn't, and then every AArch64 CPU without SVE2 BitPerm, NEON
included, pays the Java fallback. There, a broadcast + AND + compare (or, on SVE, `index`/`lsl` +
`and` + `cmpne`) lowering would keep `fromLong` fast.

**Reproducer:** `Repro.java` (attached), JDK only:

```
javac --add-modules jdk.incubator.vector Repro.java
java --add-modules jdk.incubator.vector Repro
java --add-modules jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+PrintIntrinsics Repro | grep -E "fromBitsCoerced|fromLong"
```

Output on Graviton4: _paste here_.

Output on x86 AVX-512 (same JDK), for comparison:

```
species Species[long, 8, S_512_BIT] (8 lanes), 8388608 rows, 30% nulls
fromLong                3.58 ms (best)
broadcast-and-cmp       3.59 ms (best)
fromLong / broadcast = 1.00
... jdk.internal.vm.vector.VectorSupport::fromBitsCoerced (35 bytes)   (intrinsic)   late inline succeeded
```

**Expected:** `fromLong` compiled to vector/predicate instructions, and no slower than the
broadcast-AND-compare mask.

**Workaround:** build the mask as `LongVector.broadcast(S, bits).and(laneBits).compare(NE, 0)`.
