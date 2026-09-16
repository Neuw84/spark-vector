# AGENTS.md

Working notes for agents and contributors changing spark-vector. This file records the design
decisions the code embodies, why they were taken, and how a change must be validated before it is
considered done. `README.md` is the user-facing description, `docs/results.md` the measurements,
`docs/comet.md` the Comet integration; this file is the contract behind them.

## 1. What this project is

A Spark SQL plugin that runs `Filter`, `Project` and `HashAggregate` (Partial and Final) over
Arrow-layout columnar batches with the Java Vector API (`jdk.incubator.vector`), in the style of
Apache DataFusion Comet but entirely on the JVM. It reads batches from Spark's vectorized Parquet
reader or from Comet's native scan, and emits unshaded Arrow vectors that Spark's own
`ColumnarToRowExec` consumes. With Comet configured, it can also feed Comet's native shuffle.

Non-goals, deliberately: no native code, no JNI, no bundled Parquet reader (yet), no shading of
Arrow, no dependence on Comet at compile time.

Fixed versions: Spark 4.1.x, Scala 2.13, JDK 25 (`/opt/homebrew/opt/openjdk@25` on the dev
machine), Arrow 18.3.0 (the version Spark bundles), Comet 1.0.0 built from source for the
Comet-backed tests and benchmarks. Maven builds everything.

## 2. Layout and commands

| Module | Language | Contents |
|---|---|---|
| `kernels/` | Java 25 | `VectorBuffers` (Arrow-layout `MemorySegment`s), `VecType`, `Species`, the SIMD kernels (compare, bitmap, compact, arith, cast, agg, hash, grouped accumulators, group key table) and `reference/ScalarReference`, the scalar oracle the tests compare against |
| `spark/` | Scala 2.13 + Java | plugin, session extension, `VectorColumnarRule`, expression compiler, the three operators, Arrow output, input adapters (Spark on-heap, Arrow, Comet), the Comet bridge, the Vector Acceleration UI tab |
| `benchmarks/` | Java + Scala | JMH kernel microbenchmarks and the TPC-H Q1/Q6 runner with its markdown/HTML report |

Commands that are known to work (always unset `JAVA_TOOL_OPTIONS` first; the IDE sets one that
breaks Spark's JVM options):

```bash
unset JAVA_TOOL_OPTIONS; export JAVA_HOME=/opt/homebrew/opt/openjdk@25
mvn -B -q clean install                    # kernels + Spark suites, Comet suites skipped (~3 min)
mvn -B -q -Pcomet clean install            # also the Comet-backed suites (needs the Comet jar in ~/.m2)
mvn -pl kernels test -Dvector.jvm.args="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -Dsparkvector.vectorBits=512"
mvn -pl spark install -Dsuites=io.sparkvector.spark.VectorAggregateSuite   # one suite
benchmarks/scripts/gen-tpch.sh 1           # DuckDB-generated lineitem; 10 for SF10 (2.1 GB, gitignored)
benchmarks/scripts/run-tpch.sh benchmarks/data/sf10 spark,vector,comet-scan,comet-scan-vector,comet-scan-vector-shuffle,comet --iterations 7 --warmup 5
benchmarks/scripts/run-tpch.sh --report    # rewrite benchmarks/results/results.{md,html} from the jsonl files
```

Maven output is large: redirect to a log file and grep it. Full builds take about three minutes;
run them in the background and poll.

## 3. Design decisions

Each item is a decision the code depends on. If you change one, update this section and the tests
that pin it.

### 3.1 Planning: a Comet-style columnar rule with explicit fallbacks

- Operators are replaced by `VectorExecRule` (registered through `injectColumnar`, in
  `preColumnarTransitions`), bottom-up. An operator is converted only if its child already
  produces columnar batches of supported types (vectorized Parquet scan, Comet scan, or another
  spark-vector operator) and every expression compiles.
- Everything that is not converted gets a reason attached as a `VectorFallback.Tag` on the
  original operator. There is no silent fallback: tests assert on the reason text
  (`checkFallback(..., reasonContains = ...)`), the UI shows it, and
  `spark.vector.explainFallback.enabled` prints it with `EXPLAIN`.
- Supported types are exactly `VecType`: BOOL, INT32 (also DateType), INT64 (also TimestampType),
  FLOAT64, UTF8. Decimals are not supported; the TPC-H data is generated with decimals as doubles
  for that reason. Adding a type means: `VecType` + `TypeMapping` + every kernel switch + the
  adapters + `ArrowOutput` + tests at all three vector widths.
- Spark 4 defaults to ANSI mode. Double arithmetic is bit-identical in both modes, so it is
  compiled; integer arithmetic in ANSI mode needs overflow checks the kernels do not do, so it
  falls back with a reason. ANSI division by zero is raised only for rows that are active (survive
  earlier conjuncts / the selection), matching Spark's short-circuit semantics.
- Expressions compile to a small `VectorExpr` tree (`ColumnRef`, `LiteralExpr`, `CompareExpr`,
  `And/Or/Not`, `IsNull/IsNotNull`, `ArithExpr`, `CastExpr`, `NegateExpr`). Anything else is a
  `Left(reason)`. Do not add an expression without a kernel, a scalar reference and a Spark
  comparison test.

### 3.2 Memory model: Arrow layout in native `MemorySegment`s

- `VectorBuffers` is the only view kernels have of a column: validity bitmap, data, offsets (UTF8),
  optional dictionary. Buffers are Arrow-layout so they can be handed to Arrow and Comet without
  conversion.
- Batches live in native memory allocated from a per-batch `Arena`. Reading Spark's on-heap
  `double[]` in place as a heap segment was measured at half the speed (the Vector API's heap
  segment path is not intrinsified as well), so Spark's `OnHeapColumnVector` batches are copied
  once into native memory per operator chain. Comet's and Arrow's off-heap buffers are wrapped
  zero-copy with `MemorySegment.ofAddress(...).reinterpret(size)`.
- Dictionary-encoded strings stay dictionary encoded end to end: the Spark adapter forwards
  Parquet dictionary indices, the filter compacts int32 indices and copies the small dictionary,
  `VectorDictionaryColumnVector` lets Spark's row conversion read them, and the aggregate hashes
  the dictionary rather than every row. Decoding strings row by row three times was the first
  big profiling finding (see `docs/results.md`); do not reintroduce it.
- Batch lifecycle follows Spark's columnar contract: the producer may reuse or release a batch when
  the next one is requested. Operators finish with a batch before pulling the next; the aggregate
  copies what it keeps.

### 3.3 SIMD: one species, constant operators, platform-aware paths

- `Species` is the single source of vector shape: the platform's preferred width by default (128
  bits on NEON, 256 on AVX2, 512 on AVX-512), forced with `-Dsparkvector.vectorBits=128|256|512`.
  Kernels never pick their own species.
- Three Vector API lessons learned by JMH, all now baked into the kernels and easy to undo by
  accident:
  1. `lanewise(op, ...)` / `compare(op, ...)` are only intrinsified when `op` is a compile-time
     constant. Never pass a `VectorOperators` value through a variable or a parameter; switch on
     the operation and call the constant form.
  2. `compress` is not native on NEON. Compaction uses a shuffle table there, and a real `compress`
     only on 512-bit species (16-lane int).
  3. Two 64-bit lanes are not worth a shuffle: compacting doubles by `rearrange` lost to a scalar
     walk over set bits on NEON. Full selection words are bulk copied; partial words take the scalar
     walk when the species has two lanes, the 256-entry shuffle table on 8-lane species.
- Grouped aggregation decides between one masked reduction per group and a scatter into
  accumulators by `sparkvector.agg.maskPathMaxGroups` (1 on <=4-lane species, 8 on wider; the 8 is
  a guess from lane count, not a measurement). The scatter rotates over
  `sparkvector.agg.interleave` accumulator copies (default 4: +40% at TPC-H Q1's 4 groups) and
  therefore sums doubles in a different order than Spark; `interleave=1` restores Spark's exact
  rounding.
- AVX2 and AVX-512 paths exist and are executed emulated (`vectorBits=256|512`) by the kernel test
  suite on the Apple M3 development machine. They have never been measured on real hardware. Two
  things to re-measure there before trusting defaults: `VectorMask.fromLong` is a single `kmov` on
  AVX-512 so the broadcast-AND-compare mask construction chosen for NEON may be the slower option,
  and the 8-group masked-reduction threshold.
- Popcount and bitmap bookkeeping are `Long.bitCount` over 64-bit words and never show in profiles.

### 3.4 Selection vectors between our operators

- A filter feeding another spark-vector operator forwards a `SelectedColumnarBatch` (the input
  batch plus a selection bitmap) instead of compacting, when at least
  `sparkvector.selection.minFraction` (0.5) of the rows survive; below that it compacts, because
  downstream operators then walk far fewer rows (Q6 at 2% selectivity forwarded made the project
  and aggregate walk all 6M rows). `spark.vector.exec.selection.enabled` turns forwarding off.
- Later conjuncts are evaluated only for 64-row blocks with an undecided row (`EvalContext.active`
  and the `active` overloads of `CompareKernels`). With survivors spread uniformly this rarely
  skips anything (TPC-H Q6); it is there for clustered data.
- Columns a project forwards unchanged are borrowed (`BorrowedColumnVector`), not copied.

### 3.5 Aggregation

- Partial and Final aggregates are both ours. Final consumes Spark's row shuffle through a
  `RowToColumnarExec` or, with Comet, Comet's columnar shuffle directly; merge functions
  (`CountMergeAgg`, `AverageMergeAgg`, sum/min/max over buffers) reuse the accumulators; result
  expressions are compiled with `evaluateExpression` substituted. `FILTER` clauses apply at
  Partial only. `spark.vector.exec.aggregate.final.enabled` turns the Final conversion off.
- `GroupKeyTable` memoises group ids per combination of dictionary indices when every key is
  dictionary encoded and the product of dictionary sizes is small. Plain UTF8 keys whose values
  fit in 8 bytes are dictionary encoded on the fly against a per-column dictionary kept across
  batches (single-byte values through a 256-entry direct table) so the same path applies; longer
  plain strings take hash-and-compare per row. This exists because Comet's native scan delivers
  plain strings where Spark's reader keeps dictionaries, and without it the zero-copy
  configuration lost to the copying one at SF10.
- Floating-point sums differ from Spark's in the last bits (lane-parallel and interleaved
  accumulation reorder additions). Tests compare doubles with a tolerance (`1e-9` relative in
  `VectorQuerySuite`), benchmark checksums use 10 significant digits. Never assert bit equality
  on a double sum.

### 3.6 The Arrow compatibility layer (designed to be replaced natively)

Everything that crosses a boundary goes through one of three seams. New sources or sinks must plug
into these rather than adding special cases to operators.

- Input: `ColumnVectorAdapters.adapt(ColumnVector, numRows, arena)` turns any Spark
  `ColumnVector` into `VectorBuffers`. Zero-copy adapters are tried first (our own
  `VectorArrowColumnVector`/`BorrowedColumnVector`, then anything registered through
  `ColumnVectorAdapters.register(Adapter)`, which is how the Comet adapter joins without a
  compile-time dependency); anything else is copied by `SparkColumnVectorBuffers`. A future
  Parquet reader of our own that writes Arrow memory directly would be one more `Adapter`, or
  better, would emit `VectorArrowColumnVector`s and need none.
- Output: `ArrowOutput` writes result columns as unshaded Arrow 18.3.0 vectors (`ArrowSegments`,
  `VectorAllocators`) wrapped in Spark's `ArrowColumnVector`, so `ColumnarToRowExec`, Spark's
  Arrow-based Python/R paths and any Arrow consumer work unchanged. The Arrow version must stay
  the one Spark bundles; we do not ship or shade Arrow.
- Between our operators: `SelectedColumnarBatch` (3.4). Operators check for it explicitly; a
  foreign consumer never sees one because the rule only forwards selections into our own operators
  (`markSelectionProducers`).

### 3.7 The Comet compatibility layer (designed to be replaced natively)

- No compile-time dependency on Comet. `CometVectorAdapter`, `CometBatchBridge` and
  `VectorToCometExec`/`CometShuffle` resolve Comet classes reflectively and register only if Comet
  is on the classpath. The `-Pcomet` profile adds the jar for tests; the benchmark script adds it
  when `COMET_JAR` is set.
- Comet's shaded Arrow (`org.apache.comet.shaded.arrow.*`) and Spark's unshaded Arrow are the same
  version and cannot share a class. Rejected approaches, do not retry them: putting `arrow-c-data`
  on our classpath (Comet keeps `org.apache.arrow.c.*` unshaded but with shaded signatures, so the
  classes collide), maven-shade relocation (Comet's JNI looks classes up by literal name), and a
  bulk-copy bridge (replaced by the zero-copy one).
- What works is sharing memory: `ArrowCData` writes Arrow C Data Interface structs (80-byte
  `ArrowArray`, 72-byte `ArrowSchema`) with the FFM API, with `Linker.upcallStub` release
  callbacks and a live-export registry, and Comet's own `ArrowImporter` imports them into a
  `CometVector` over our buffers. `CometShuffleSuite` asserts every export is released.
- The rule rewrites a `ShuffleExchangeExec` (or Comet's row-based columnar shuffle) above a
  `VectorExec` into Comet's native shuffle over `VectorToCometExec`, for hash, single and
  round-robin partitioning only (range partitioning over a non-native child makes Comet sample
  twice). Requires `spark.shuffle.manager=...CometShuffleManager` and
  `spark.comet.exec.shuffle.enabled=true`; `spark.vector.comet.shuffle.enabled` turns the rewrite
  off.
- Not combined with Comet: Comet's Final aggregate (needs Comet's own partial buffers), Comet's
  Sort, and native blocks (our operators are not `CometNativeExec`s). Comet's `LargeVarCharVector`
  falls back to the copying adapter. Dictionary strings are decoded when crossing into Comet.
- The seams a native replacement would fill: `Adapter` for the scan (replace `CometVectorAdapter`
  with our own reader's vectors), a columnar shuffle exchange of our own where `CometShuffle`
  builds Comet's (this would also remove the `ColumnarToRow`/`RowToColumnar` pair around Spark's
  row shuffle in the non-Comet configuration), and a vectorized Sort. Keep those boundaries where
  they are.

### 3.8 The Vector Acceleration UI tab

- Attached from the driver plugin; lives under `org.apache.spark.sql.vector.ui` because
  `SparkUITab`, `WebUIPage` and `UIUtils` are `private[spark]`. It never influences execution: every
  listener callback and the attachment itself are wrapped so a UI failure cannot fail a query or
  application start.
- Engine classification is by operator identity (`VectorExec`, a class in Comet's packages, a
  columnar leaf, a transition), not by tags we set; fallback reasons come from `VectorFallback`.
  A plan is "fully accelerated" when no operator runs on plain Spark; scans and row/columnar
  transitions are plumbing and count for neither side.
- Spark 4 ships Bootstrap 4 and jQuery 3.5: use Spark's own `collapseTable` from `webui.js` and
  its CSS classes, not Bootstrap 5 `data-bs-*` attributes (they silently do nothing). The DAG is
  rendered client side with the d3/dagre-d3/graphlib-dot bundles Spark already serves.

## 4. Validation: what "done" means

A change is not done until all of the following that apply have run green, locally, on JDK 25.

1. Kernel correctness against the scalar oracle. Every kernel has a scalar twin in
   `ScalarReference`; `kernels/src/test` compares them on random data with nulls, selections and
   awkward lengths (tails shorter than a vector, batches not multiple of 64). Run the kernel suite
   at 128, 256 and 512 bits (`-Dsparkvector.vectorBits=...`); the wider ones are emulated but they
   are the only coverage the AVX2/AVX-512 paths have.
2. Spark SQL comparison. Operator and expression behaviour is validated by running the same SQL
   twice on the same session with `spark.vector.enabled` toggled and comparing rows
   (`VectorQuerySuite.checkVectorized`, tolerance `1e-9` for doubles), while asserting the expected
   spark-vector operators are in the final (post-AQE) plan. Unsupported cases are validated the same
   way with `checkFallback`, which asserts the Spark operator stayed and the recorded reason
   contains the expected text. Suites: `VectorFilterSuite`, `VectorProjectSuite`,
   `VectorAggregateSuite`, plus adapter/Arrow suites and `SparkOnJdkSmokeSuite` (Spark itself works
   on this JDK with these flags).
3. Comet integration. `CometScanSuite` and `CometShuffleSuite` are tagged `CometTest` and run only
   with `-Pcomet`. They cover zero-copy scan adaptation, dictionary strings from Comet, the shuffle
   rewrite for each partitioning, and that every C Data export is released. They need the Comet jar
   built from source (`docs/comet.md` explains why, on macOS).
4. UI. `PlanAccelerationSuite` pins the classification rules without a session (stand-ins in
   `org.apache.spark.sql.comet` stand for Comet operators); `VectorAccelerationUiSuite` binds a
   real Spark UI, runs converted queries and fetches both pages over HTTP.
5. End-to-end results. `TpchRunner` computes a checksum of every configuration's result rows to 10
   significant digits; the report states whether all configurations agree. A benchmark run where
   the checksums differ is a correctness bug, not a performance result.
6. Performance claims need evidence: a JMH number for a kernel change (`benchmarks` module,
   `-wi 2 -i 3 -w 1 -r 1 -f 1` is the convention in `docs/results.md`) or a TPC-H median plus a
   JFR profile for an operator change. "It should be faster" is not evidence; several intuitive
   changes in this project's history were slower (see the list of reversed assumptions in
   `docs/results.md`).
7. Unexpected results are profiled with Java Flight Recorder before they are explained. When a
   number is worse than expected, or better in a way you cannot account for, do not write a
   hypothesis into the docs or the code: record the run and read the profile first. The
   procedure:

   ```bash
   JVM_EXTRA="-XX:StartFlightRecording=filename=/tmp/x.jfr,settings=profile,dumponexit=true" \
   RESULTS_DIR=/tmp/profiling \
   benchmarks/scripts/run-tpch.sh benchmarks/data/sf10 <config> --queries q1 --warmup 2 --iterations 5
   $JAVA_HOME/bin/jfr view hot-methods /tmp/x.jfr
   $JAVA_HOME/bin/jfr print --events jdk.ExecutionSample --stack-depth 12 /tmp/x.jfr
   ```

   `RESULTS_DIR` keeps the profiling rows out of the report; `JVM_EXTRA` applies to the benchmark
   JVMs only (the report JVM would otherwise overwrite the recording). Read the `[tpch]`
   per-operator kernel times first, then the hot-method list, then the callers of any JDK-internal
   frame near the top (`MemorySessionImpl.checkValidStateRaw`, `checkBounds`,
   `isAlignedForElement` mean a `MemorySegment` access that the JIT did not hoist or inline: a
   megamorphic call site, a segment from a different session per call, or `MemorySegment.mismatch`
   on tiny ranges). Compare two configurations by recording both. The SF10 `comet-scan-vector`
   regression (3.5) is the worked example: the first written explanation (thread competition,
   batch size) was wrong, and the profile showed the real cause in one look. Only when the
   profile is understood does the fix, the doc entry and the rerun follow, in that order.

Current counts: 78 kernel tests, 76 Spark tests (64 without the Comet profile). If a change lowers
either number, explain why in the commit.

## 5. Benchmarking protocol

- One JVM per configuration, `local[8]`, 8 GB heap, `spark.sql.shuffle.partitions=8`; SF1 with 10
  warm-up and 10 measured runs, SF10 with 5 and 7. Configurations: `spark`, `vector`, `comet-scan`,
  `comet-scan-vector`, `comet-scan-vector-shuffle`, `comet`.
- Results are appended to `benchmarks/results/<config>.jsonl` (committed) and the report takes the
  latest measurement per (dataset, config, query). Outliers stay in the files with older
  timestamps; note discarded runs in `docs/results.md`.
- Run on a quiet machine. A video call or a full build minutes earlier moved medians by up to 2x on
  the development laptop; a `spark` Q1 median far from the documented one (1123 ms at SF10) means
  the environment, not the code. Check `uptime` and the top CPU consumers before trusting a run.
- Update `docs/results.md` tables from the report and keep the earlier phase tables for history.
  Speedups are always relative to plain Spark on the same dataset and session.

## 6. Conventions

- Scala for planner rules, operators and expression compilation; Java for kernels and anything
  touching `MemorySegment` in a hot loop. No native code.
- Spark-facing configuration is `spark.vector.*` (read from `SQLConf`, or from `SparkConf` for the
  UI keys the plugin needs before a session exists); kernel tuning knobs are JVM system properties
  `sparkvector.*` (`vectorBits`, `agg.interleave`, `agg.maskPathMaxGroups`, `selection.minFraction`)
  because kernels have no Spark dependency. Document every new key in `README.md`.
- Commit messages follow Conventional Commits with an explanatory body; commit early, never push
  from an agent session. Build before committing.
- Inclusive terminology throughout (allowlist/denylist, primary/replica).
- Keep the running doc set in sync: `README.md` (usage, keys, lessons), `docs/results.md`
  (numbers and what they mean), `docs/comet.md` (integration and limitations), this file (design
  and validation).

## 7. Known gaps

- Group keys longer than 8 bytes arriving as plain strings are hashed and compared per row.
- Selective predicates with scattered survivors (TPC-H Q6) lose to Spark's codegen over Spark's
  scan (0.84x at SF10): the on-heap copy plus full-column evaluation of a 1.9% predicate. Over
  Comet's scan the copy is gone and the plugin beats Spark (1.09x) but not Comet's scan under
  Spark's codegen (1.20x).
- Without Comet, the shuffle is Spark's row shuffle with a `ColumnarToRowExec` above the partial
  aggregate and a `RowToColumnarExec` below the Final. A columnar shuffle of our own is the
  natural next seam to fill (3.7).
- The final `Sort` is Spark's.
- AVX2/AVX-512 paths are tested emulated, never measured on real hardware.
- `TpchRunner --keep-alive` leaves the session and the Spark UI up for inspection; the demo JVM's
  Jetty resets some parallel static-resource fetches under load, so reload the page if the tab's
  toggles do not react (jQuery failed to load).
