# Using Comet as the scan

spark-vector accelerates Filter, Project and HashAggregate. It does not read Parquet itself: it
consumes whatever columnar batches sit below it. Two sources work out of the box:

| Source | How batches are read | Notes |
|---|---|---|
| Spark's vectorized Parquet reader (`FileSourceScanExec`, `Batched: true`) | copied once into Arrow-layout buffers per batch | zero configuration; the copy shows up in `time in spark-vector kernels` |
| Apache DataFusion Comet scan (`CometScanExec` / `CometBatchScanExec`) | zero copy: the Arrow buffers Comet's native reader produced are wrapped as `MemorySegment`s | dictionary-encoded strings stay encoded, so `GROUP BY` string keys hash the dictionary once per batch |

Comet is used in *scan-only* mode: its native Parquet-to-Arrow reader replaces Spark's, its native
operators stay off, and spark-vector's JVM SIMD operators run above the scan. Optionally Comet's
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
--jars comet-spark-spark4.1_2.13-1.0.0.jar,spark-vector-spark_2.13-0.1.0-SNAPSHOT.jar
```

List Comet's plugin first: session extensions run in registration order, and spark-vector's
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
row-based "columnar" shuffle would convert our batches to rows and back. spark-vector's rule
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
JDK 25 with spark-vector):

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
and the `IcebergTest` tag and runs with `-Pcomet,iceberg`; see [iceberg.md](iceberg.md).

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
profile. The entry names the commit it was measured at. Decision table: none yet -- the default is
empty until the study of #281 lands its rows here.

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
  the next one is requested, so spark-vector operators finish with a batch (or copy what they
  keep, as the aggregate does) before pulling the next.
- Comet's native operators other than the scan and the shuffle are not combined with ours: a Comet
  Final aggregate would need Comet's own partial buffers (its `missingCometProducer` guard), and our
  operators are not `CometNativeExec`s, so Comet cannot inline them into a native block.
- Dictionary-encoded strings are decoded when crossing into Comet (its stream reader decodes them
  anyway); everything else crosses as-is.
