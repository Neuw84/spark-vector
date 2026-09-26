---
layout: default
title: Comet as the scan
---

# Using Comet as the scan

vecruntime accelerates Filter, Project and HashAggregate. It does not read Parquet itself: it
consumes whatever columnar batches sit below it. Two sources work out of the box:

| Source | How batches are read | Notes |
|---|---|---|
| Spark's vectorized Parquet reader (`FileSourceScanExec`, `Batched: true`) | copied once into Arrow-layout buffers per batch | zero configuration; the copy shows up in `time in vecruntime kernels` |
| Apache DataFusion Comet scan (`CometScanExec` / `CometBatchScanExec`) | zero copy: the Arrow buffers Comet's native reader produced are wrapped as `MemorySegment`s | dictionary-encoded strings stay encoded, so `GROUP BY` string keys hash the dictionary once per batch |

Comet is used in *scan-only* mode: its native Parquet-to-Arrow reader replaces Spark's, its native
operators stay off, and vecruntime's JVM SIMD operators run above the scan. Optionally Comet's
native shuffle carries the partial aggregates too (see below). Comet 1.0 only ships
the fully native DataFusion scan (`CometNativeScanExec`), which needs `spark.comet.exec.enabled=true`
and off-heap memory; "scan-only" therefore means enabling exec and switching every Comet operator
off individually (`io.sparkvector.benchmarks.TpchRunner.CometScanOnly` lists the full set).

## Configuration

```
--conf spark.plugins=org.apache.spark.CometPlugin,io.sparkvector.spark.VectorPlugin
--conf spark.comet.enabled=true
--conf spark.comet.scan.enabled=true
--conf spark.comet.exec.enabled=true
--conf spark.comet.exec.shuffle.enabled=false
--conf spark.comet.exec.project.enabled=false
--conf spark.comet.exec.filter.enabled=false
--conf spark.comet.exec.aggregate.enabled=false
--conf spark.comet.exec.sort.enabled=false        # ... and so on for the other spark.comet.exec.<op>.enabled keys
--conf spark.memory.offHeap.enabled=true
--conf spark.memory.offHeap.size=2g
--conf spark.driver.extraJavaOptions="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED"
--conf spark.executor.extraJavaOptions="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED"
--jars comet-spark-spark4.1_2.13-1.0.0.jar,spark-vector-spark_2.13-0.0.1.jar
```

List Comet's plugin first: session extensions run in registration order, and vecruntime's
planner rule needs to see `CometScanExec` already in place. Nothing else is Comet-specific: the rule
treats any child with `supportsColumnar = true` and supported column types as an input, and the
Comet vector adapter (`io.sparkvector.spark.comet.CometVectorAdapter`) registers itself on first
use if Comet's classes are on the classpath. There is no compile-time dependency on Comet; the
adapter binds to `org.apache.comet.vector.CometVector`, `CometDictionaryVector` and the shaded
Arrow classes reflectively, so the same jar works with or without Comet.

## Comet's native shuffle

Add to the configuration above:

```
--conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager
--conf spark.comet.exec.shuffle.enabled=true
```

Comet's planner will not hand a Comet shuffle to a columnar child it does not recognise, and its
row-based "columnar" shuffle would convert our batches to rows and back. vecruntime's rule
therefore rewrites any exchange (Spark's, or the one Comet chose) sitting on one of its operators
into Comet's *native* shuffle over `VectorToCometExec`, which is the one piece of glue: each column
of our batch is exported through the Arrow C Data Interface and imported by Comet's Arrow.

The export is written with the FFM API rather than `arrow-c-data`. Comet bundles that module with
`org.apache.arrow.c.*` left unshaded (its JNI library resolves those class names literally) but
with parameter types from the shaded Arrow, so a second `arrow-c-data` on the classpath would
collide and a relocated copy would break the JNI lookups. Two C structs, a couple of format strings
and an upcall stub for the release callback need none of that. Comet's `ArrowImporter` (reached
reflectively) wraps our buffers without copying; when it releases the imported vector the callback
drops our references. `ArrowCData.liveExports()` counts outstanding exports and is checked by the
tests. The shuffle output on the reading side is Comet vectors, which our Final aggregate reads
zero copy through the existing adapter.

Hash, single-partition, round-robin and range partitioning are rewritten. For range partitioning
(the `ORDER BY` above a Final aggregate) Comet computes the bounds with Spark's `RangePartitioner`
over a sampling pass of the child, which is exactly what Spark's own exchange does, so the child runs
twice in either case; the sort above the shuffle then becomes `VectorSortExec` because Comet's
shuffle output is columnar. It follows Comet's own switch
(`spark.comet.shuffle.native.partitioning.range.enabled`, default on) and can be turned off alone
with `spark.vector.comet.shuffle.range.enabled=false`. Comet's sampling pass never closes the
vectors it imports, so the bridge releases whatever is still outstanding when the task completes.
Decimals cross the bridge widened to Arrow's 128-bit decimal layout. In the other direction a wide
decimal column (`decimal(p > 18)`) of a Comet batch is Arrow `Decimal128` already, the DECIMAL128
lane's own layout, and is wrapped in place like the other fixed-width types (#257); 32-bit decimals
still take the copy path.

## Comet on macOS (Apple Silicon)

The Comet jars on Maven Central bundle native libraries for Linux only. On macOS build Comet from
source once (Rust toolchain, `protoc` and JDK 17 needed for the build; the resulting jar runs on
JDK 25 with vecruntime):

```bash
brew install protobuf
git clone --branch 1.0.0 --depth 1 https://github.com/apache/datafusion-comet.git
cd datafusion-comet
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
(cd native && RUSTFLAGS="-Ctarget-cpu=native" cargo build --release)
./mvnw install -Prelease -DskipTests -Pspark-4.1 -Dmaven.gitcommitid.skip=true
```

`./mvnw install` puts `org.apache.datafusion:comet-spark-spark4.1_2.13:1.0.0` into `~/.m2`, which
is what the `comet` Maven profile of this project resolves. The Rust build takes 10-30 minutes.

## Running the Comet-backed tests

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@25   # or wherever JDK 25 lives
mvn -Pcomet -pl spark verify -Dsuites=io.sparkvector.spark.comet.CometScanSuite,io.sparkvector.spark.comet.CometShuffleSuite
```

Without `-Pcomet` the suites are excluded by their `CometTest` tag and the rest of the build has no
Comet dependency. The Iceberg-over-Comet suite (`CometIcebergSuite`) carries both the `CometTest`
and the `IcebergTest` tag and runs with `-Pcomet,iceberg`; see [iceberg.md](iceberg.html).

## Mixed chains: Comet's operators above ours (#280)

Comet below ours needs nothing new: a Comet native block is a columnar child like any other, and
`CometVectorAdapter` reads its vectors zero-copy. Comet *above* ours cannot come from Comet's own
rule, which runs first and never sees our operators, so our rule builds it (`mixedChains`, behind
`spark.vector.comet.mixed.enabled`, default `false`): a Spark operator that was left to Spark and
whose children are all ours is offered to Comet through the **sink leaf** -- Comet's
`CometSinkPlaceHolder` over a one-child `CometUnionExec` over our `VectorToCometExec` -- and Comet's
`CometExecRule` is applied to that subtree; the result is kept only when Comet planned the operator
natively. The pieces are Comet's own classes, constructed reflectively (`CometMixedBridge`): the Scan
proto comes from Comet's sink serde (`CometExchangeSink.convert`, with Comet's type checks), the union
is the pass-through Comet's input walk recognises (`foreachUntilCometInput` lists Comet's own JVM
operators, never a foreign node -- a placeholder directly over our export node planned but had no
input at execution), and our batches reach native as `CometVector`s through Comet's
`ColumnarBatchArrowReader`, a hand-over of the Arrow buffers without a copy. In the final plan Comet's
block pass unwraps the placeholder, so a mixed plan reads `CometProject` / `CometUnion` /
`VectorToComet` / ours.

Rejected as the leaf: Comet's `spark.comet.sparkToColumnar` transition. It wraps leaf nodes only
(`shouldApplySparkToColumnar`, "TODO: consider converting other intermediate operators") and its
`SparkColumnarArrowReader` copies a Spark columnar batch value by value through `ArrowWriter`.

Boundaries the pass keeps. An aggregate half changes engine only when Comet's own predicate
(`QueryPlanSerde.allAggsSupportMixedExecution`, asked through the bridge) says every function's
intermediate buffer is laid out the same way by Spark and by Comet -- in Comet 1.0 that is `min`,
`max`, the bit aggregates, a non-decimal `avg` and a non-decimal, non-TRY `sum`; `count` is *not* on
the list, nor decimal sums or averages -- so a `GROUP BY` with `count(*)` keeps its pair whole and the
plan says why (`mixed: aggregate halves cannot be split across engines (intermediate buffer formats
differ)`); with a shareable pair Comet's partial runs above our chain and the final above the shuffle
is whoever plans it. Exchanges are not offered. A selection is compacted by the export itself;
dictionaries are decoded and INT64 decimals widened as for the shuffle (#279 measured both). When
Comet declines an operator the plan records its reasons (`mixed: Comet declined -- ...`, read through
Comet's `ExtendedExplainInfo`), and a column type Comet's sink refuses is a reason too. The split
between the engines is otherwise the two per-operator toggles: an operator ours refused or has
switched off (`spark.vector.exec.<op>.enabled=false`) with Comet's `spark.comet.exec.<op>.enabled=true`
goes to Comet. The acceleration view classifies the one-child union over `VectorToComet` as the
bridge, so a mixed plan shows both engines and counts as accelerated without the hand-off inflating
either. `CometMixedChainSuite` (`-Pcomet`) pins Comet's projection, collect limit, expand and union
above our chains, Comet's partial aggregate above our filter for a shareable pair and the refusal for
`count`, ours above Comet's filter, today's plan with the key off, and the view.

The pass walks bottom-up, so a parent above an operator it has just given to Comet is offered too
-- Comet's own rule ran before ours and never saw a native child there -- and a broadcast exchange
is looked through, since Comet converts one only together with the join above it. Joins and final
aggregates are then reached through the exchange: a shuffle over one of our operators is Comet's
native shuffle over `VectorToComet` (the section above), and Comet's JVM shuffle that its rule had
planned over a block this pass later converted becomes Comet's native shuffle over that block.
`CometMixedShuffleSuite` (Comet's shuffle manager on) pins Comet's hash join above its shuffles
above two of our filters, Comet's final aggregate above its shuffle above its partial above our
filter, and Comet's broadcast join with its broadcast over our chain -- the last with adaptive
execution off: under it the broadcast stage is planned and run before the join's stage, with Spark's
exchange over the child as Comet's rule saw it, so Comet's broadcast is out of reach and the join
stays Spark's above our rows. A failure inside Comet's block above the leaf (an ANSI division by
zero in Comet's projection) still releases every export -- the task-completion listener the shuffle
bridge already had. What the seam cannot reach in Comet 1.0: a window (Comet has no window operator)
and a sort-merge join without Comet's shuffle. Which operators *should* go to Comet is the
allowlist below.

### The allowlist: `spark.vector.comet.preferComet` (#281)

The pass offers Comet only the operator kinds this key names -- comma-separated, each optionally
qualified by a predicate the planner reads off the plan: `project:wideDecimal` (an input or output
column of `decimal(p > 18)`), `filter:strings` (a string input), `sort:estimatedRows>1000000` (the
logical estimate, rows or bytes over the row width); `all` names every kind. Kinds: `filter`,
`project`, `sort`, `sortMergeJoin`, `hashJoin`, `broadcastHashJoin`, `window`, `expand`, `union`,
`limit`. `aggregate` is refused with a warning -- a pair cannot be split; under `all` an aggregate
half is offered and Comet's buffer rule decides. Comet's own rule runs before ours and already owns
whatever sits on its scan, so the list is about the operators above our chains.

A listed operator our rule could take is left to the pass instead, with the reason `delegated to
Comet (spark.vector.comet.preferComet)` (shown on Comet's operator once it runs there); if Comet
declines it -- its own fallback, the sink's type rule, the aggregate pair -- ours converts it after
all, so a requested swap never ends on Spark's operator (`CometPreferCometSuite`). The two escape
hatches: `spark.vector.comet.mixed.enabled=false` is today's plan, and an empty list under it allows
mixed plans but requests none.

An entry is added only when all three hold on the SF10 matrices of #279 (`docs/results.md`): Comet's
operator time is lower than ours in the queries the operator dominates by more than twice the
crossing cost for its widths; the swap regresses no TPC-H or TPC-DS query against the better pure
configuration (`comet-scan-vector-shuffle`, `comet`) beyond noise -- the harness's `hybrid`
configuration and its report section check that query by query; and the reason is understood from a
profile. The entry names the commit it was measured at.

#### The decision table (TPC-H SF10, the #279 protocol, Comet 1.0.0, measured at the commit that closes #281)

Each candidate ran as `hybrid` with only its Comet toggle on. Protocol note: the #279 baselines and
the first joins run kept Spark's shuffle files in `/tmp`, a RAM-backed tmpfs on the host; every later
run keeps them on disk and sets Comet's off-heap pool, which alone makes q5 7.4 s instead of 2.9 s
and q7 3.7 s instead of 3.6 s, so those candidates are judged against the unswapped plan rerun under
the same conditions (`comet-scan-vector-shuffle`, disk: 48.4 s over 22 queries; the RAM-backed run
42.9 s; `comet` 39.0 s). "vs unswapped" is that like-for-like baseline; "vs better pure" is the
issue's literal rule 2, which no entry can meet on this host because `comet` beats the unswapped
plan on 17 of 22 queries regardless. Every run had a 13 GB memory cap the unswapped plan fits under.

| entry | where it fired | operator time, ours -> Comet's | wall clock vs unswapped | vs better pure | rule 1 | rule 2 | rule 3 | verdict |
|---|---|---|---|---|---|---|---|---|
| `hashJoin,broadcastHashJoin` | 4-5 queries (q3 q12 q18 q19 q20): adaptive execution's runtime broadcast joins whose inputs are both Comet shuffle stages -- Comet's own conversion once its toggle is on; a static broadcast above our chain never crosses under adaptive execution (#280) | q3 568 -> 157 ms on 1.46M rows, q12 742 -> 334, q19 43 -> 25 (2-3.6x) | q12 1.26-1.30x, q3 1.10-1.13x, q19 1.07x; the rerun after the fixes lost q7 (0.88x) and q21 exceeded the memory cap the unswapped plan fits under | 18 of 20 slower, the pure gap | met: the crossing is per batch, negligible against 400 ms | met in the first run; the rerun's q7 and q21 say no | the #279 q18 profile: our per-row `GroupKeyTable` probe against DataFusion's hash join; q21's memory unprofiled | the join is the operator that pays; the entry as a whole is **not shipped** until q21's memory is understood |
| `sort` | every local sort above our chains | -- | q3 1.5x slower where it ran; the run died on q5 | -- | not met | not met | a blocking Comet consumer above our chain holds every exported batch until it finishes, so memory grows with the input (12.7 GB resident at SF10, then the kernel's OOM killer) | **not eligible** |
| `filter,project` | Comet's filter and projection on its scan in every query (Comet's rule) and above our chains | our join times within noise (q5 SHJ 2987 vs 2187 ms: the export and import around a projection between two joins on 9.1M rows); Comet's projections ~15 ms | 13 queries faster (q19 1.72x, q6 1.70x, q14 1.43x, q15 1.30x, q13 1.29x, q20 1.26x, q12 1.21x, q4 1.17x, q5 1.16x, q3 1.15x, q21 1.10x), q8 0.86x and q11 0.86x slower; 44.2 s vs 48.4 s (-8.7%) | 14 of 22 slower | the filter's wins are Comet's scan-side filter over dictionary-encoded columns beating ours (#14's `decodeDictionary`, 20% of Q6's samples) -- a cost on our side; a projection between two of our operators has no own cost and costs its two crossings (q8, q11) | two losses: narrow | #14's profile for the wins; q8/q11 are the crossings | **not as one entry**: drop `project`, see `filter` |
| `filter` (alone) | Comet's filter on its scan in every query; above our chains almost never | filter ~20 ms either way; join times unchanged | 13 queries faster (q19 1.74x, q14 1.45x, q15 1.29x, q20 1.29x, q12 1.26x, q6 1.23x, q13 1.22x, q2 1.19x, q7 1.12x, q3 1.09x, ...), q11 0.94x and q1 0.95x (noise); 37.6 s vs 40.9 s over 20 queries (-8%); **q21 exceeded the 13 GB cap** the unswapped plan fits under | 17 of 20 slower | met for Comet's scan-side filter, but as Comet's own conversion, not the seam's; the reason is ours to fix (#14) | no wall-clock loss beyond noise; q21's memory is a regression | #14's profile; q21 unprofiled | **the best available win (-8%) and the maintainer's call**: enabling Comet's filter is `spark.comet.exec.filter.enabled`, the allowlist entry adds little; fix #14 first, understand q21's memory |
| `project:wideDecimal`, `filter:wideDecimal` | never fired: the decimal schema of the harness is decimal(12,2)/(15,2), narrow by the predicate's definition, and no wide-decimal dataset exists in the protocol | -- | -- | -- | not measurable here; #258's JMH plus the crossing (~3 ns per row per column at 1024-row batches) says Comet's operator time -- unmeasured -- decides | -- | -- | **not measured**; needs a wide-decimal dataset |
| `aggregate` | refused by the key; under `all` Comet's buffer rule decides the half (#280) | -- | -- | -- | -- | -- | -- | not eligible (the pair) |

The default stays **empty**: no entry met the three rules as written. Two things are worth the
maintainer's attention: Comet's join is 2-3.6x faster than ours where it fires (a DataFusion hash
join against our per-row probe -- the #288 and #12 levers), and Comet's filter on its scan beats ours
by the #14 dictionary decode (-8% over TPC-H) -- both are costs to remove on our side before an
allowlist entry would be the right answer; and two candidates pushed q21 past a memory cap the
unswapped plan fits under, which needs a profile before any entry that touches it ships.

#### What the study found and fixed on the way

- **Wide decimals crossed the seam wrong.** The C Data export widened every decimal from a 64-bit
  lane; a 19-38 digit decimal is a DECIMAL128 lane already, so its two limbs became two rows and
  Comet read zeros and neighbours' values -- TPC-H q11, q15, q17, q18 at SF1 decimals returned wrong
  results whenever a delegated Comet operator read a sum or average our side had computed. Fixed in
  `ArrowCData.export` (the 128-bit lane crosses as is, zero-copy from an Arrow vector); the shuffle
  bridge had refused wide decimals all along and now carries them too. Regression cases in
  `CometMixedChainSuite` and `CometShuffleSuite`.
- **The pass-through union asserted on a chain rooted in a join.** Comet 1.0's `CometUnionExec`
  reads its partitioning through `originalPlan.withNewChildren(children)`; the leaf now names the
  export node (one child) as the original plan (TPC-H q2).
- **A delegated child is columnar.** Our operators refused a delegated (still Spark's) child as
  "not columnar" and the chain broke above the swap -- Spark's row join and aggregate above a
  columnar-to-row transition (q5). A delegated child counts as columnar: it ends on Comet or on ours.
- **Comet's operators need the off-heap pool.** The `hybrid` configuration sets
  `spark.memory.offHeap.enabled` and a size, as the pure `comet` one does; without it Comet's sort
  grew its native allocation until the kernel killed the JVM.

## Per-operator attribution against Comet

The benchmark harness attributes time per operator for both engines (#279): our operators through
their `time` metric, Comet's native operators through DataFusion's `elapsed_compute` and
`output_rows` on each `Comet*Exec` node (nanoseconds, whatever the metric's description says). Each
results row carries the list as `operatorTimes`; `--report` prints an operator matrix -- milliseconds
per operator kind under each configuration, and per query the kinds both engines ran with the delta.
Comet's fallback reasons are read through its `ExtendedExplainInfo` and listed as `Comet: ...` beside
ours, so an operator that one side left to Spark is never mistaken for a comparison. The times exclude
the wait on children and the crossing between the engines; the crossing is measured on its own.

**The measured crossing cost** (`CrossingBenchmark`, the table in `docs/results.md`, "Hybrid planning
study"): into Comet, a fixed-width or plain-string column is a pointer hand-over of about 2.7-3 µs per
column regardless of rows (0.3-0.7 ns per row per column at 4096-8192 rows); back is zero-copy at
0.1-0.25 µs per column. A dictionary-encoded string column is decoded on the way in, 20-25 ns per row
per column; an INT64-lane decimal is widened to 128 bits going in (2.5-3 ns per row per column) and
narrowed back (0.4-0.5 ns). A swap of one operator for Comet's pays this twice, so it must beat ours
by more than twice the crossing of the columns it touches.

## Limitations

- Comet's `LargeVarCharVector` (64-bit offsets) is not adapted zero-copy; such columns fall back to
  the copying adapter.
- Batch lifecycle follows Spark's columnar contract: Comet may reuse or release a batch as soon as
  the next one is requested, so vecruntime operators finish with a batch (or copy what they
  keep, as the aggregate does) before pulling the next.
- Comet's native operators other than the scan and the shuffle are not combined with ours: a Comet
  Final aggregate would need Comet's own partial buffers (its `missingCometProducer` guard), and our
  operators are not `CometNativeExec`s, so Comet cannot inline them into a native block.
- Dictionary-encoded strings are decoded when crossing into Comet (its stream reader decodes them
  anyway); everything else crosses as-is.
