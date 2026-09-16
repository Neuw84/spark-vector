# spark-vector

A Spark SQL plugin that executes Filter, Project and HashAggregate (Partial and Final) on
Arrow-layout batches with the Java Vector API (`jdk.incubator.vector`). It follows the
architecture of [Apache DataFusion Comet](https://github.com/apache/datafusion-comet), but stays
entirely on the JVM: no native library, no JNI, no serialization boundary. Unsupported operators,
expressions or types fall back to Spark with a recorded reason. With Comet installed it can sit
between Comet's native Parquet scan and Comet's native shuffle, both reached zero copy.

- Spark 4.1.x, Scala 2.13, JDK 25 (the Vector API is still an incubator module)
- Input: Spark's vectorized Parquet reader (copied into Arrow layout once per batch, keeping
  dictionary-encoded strings as dictionary indices) or Comet's native Parquet reader in scan-only
  mode (read zero-copy). See [docs/comet.md](docs/comet.md).
- Output: unshaded Arrow 18.3.0 vectors (the version Spark bundles) wrapped in Spark's
  `ArrowColumnVector`, so Spark's own `ColumnarToRowExec` consumes them unchanged.

## Layout

| Module | Language | Contents |
|---|---|---|
| `kernels/` | Java 25 | `VectorBuffers` (Arrow-layout `MemorySegment`s), SIMD kernels: compare, bitmap logic, compaction, arithmetic, casts, reductions, group hashing, grouped accumulators; scalar references used as test oracles |
| `spark/` | Scala 2.13 + Java | `VectorPlugin`, session extension, `VectorColumnarRule`, expression compiler, `VectorFilterExec` / `VectorProjectExec` / `VectorHashAggregateExec`, Arrow output, input adapters (Spark vectors, Arrow, Comet), the Vector Acceleration UI tab |
| `benchmarks/` | Java + Scala | JMH kernel microbenchmarks and the TPC-H Q1/Q6 runner |

## Building

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@25      # any JDK 25
mvn verify                                          # kernels + Spark suites (Comet suites skipped)
mvn -Pcomet verify                                  # also runs the Comet-backed suites (see docs/comet.md)
```

The plugin jar is `spark/target/spark-vector-spark_2.13-<version>.jar` (kernels shaded in, nothing
else). Spark and Arrow are `provided`.

## Running with spark-submit

```bash
spark-submit \
  --conf spark.plugins=io.sparkvector.spark.VectorPlugin \
  --conf spark.driver.extraJavaOptions="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED" \
  --conf spark.executor.extraJavaOptions="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED" \
  --jars spark-vector-spark_2.13-0.1.0-SNAPSHOT.jar \
  ...
```

`spark.plugins` registers the session extension automatically; alternatively set
`spark.sql.extensions=io.sparkvector.spark.VectorSparkSessionExtensions`.

Configuration keys (all default to `true` except the last):

| Key | Meaning |
|---|---|
| `spark.vector.enabled` | main switch |
| `spark.vector.exec.filter.enabled` | convert `FilterExec` |
| `spark.vector.exec.project.enabled` | convert `ProjectExec` |
| `spark.vector.exec.aggregate.enabled` | convert `HashAggregateExec` |
| `spark.vector.exec.aggregate.final.enabled` | also convert Final-mode aggregates (their input is the shuffle) |
| `spark.vector.exec.selection.enabled` | pass selection bitmaps between our operators instead of compacting |
| `spark.vector.comet.shuffle.enabled` | feed Comet's native shuffle from our operators when Comet's shuffle is configured |
| `spark.vector.ui.enabled` | attach the Vector Acceleration tab to the Spark UI (default `true`) |
| `spark.vector.ui.retainedExecutions` | queries kept by that tab (default `100`) |
| `spark.vector.explainFallback.enabled` | log why each operator was left to Spark (default `false`) |

JVM system properties for the kernels: `sparkvector.vectorBits=128|256|512` forces a vector shape
(the default is the platform's preferred one), `sparkvector.agg.interleave=1|2|4` sets how many
accumulator copies the grouped aggregation rotates through (default 4; 1 reproduces Spark's
floating-point rounding exactly), `sparkvector.selection.minFraction` (default 0.5) is the
surviving fraction below which a filter compacts instead of forwarding a selection.

### JDK 25 and Spark 4.1

Spark 4.1 officially supports JDK 17 and 21. Running it on 25 needs two things beyond the usual
`--add-opens` set that Spark's launcher already passes:

- Hadoop 3.4.2, which Spark 4.1.3 bundles, calls `Subject.getSubject` and fails on JDK 24+
  ([HADOOP-19212](https://issues.apache.org/jira/browse/HADOOP-19212)). Replace
  `hadoop-client-api` and `hadoop-client-runtime` in `$SPARK_HOME/jars` with 3.4.3 (drop-in shaded
  jars; this project's tests do the same through Maven).
- `--sun-misc-unsafe-memory-access=allow` silences the deprecation warnings from Spark's and Arrow's
  use of `Unsafe`.

## How it works

```
Driver                                           Executor (per batch)
------                                           --------------------
FilterExec        -> VectorFilterExec            ColumnarBatch -> VectorBuffers (adapter)
ProjectExec       -> VectorProjectExec           kernels over MemorySegment (compare, arith, reduce)
HashAggregateExec -> VectorHashAggregateExec     Arrow vectors -> ArrowColumnVector -> next operator
(Partial and Final)                              or a selection bitmap over the child's columns
ShuffleExchange   -> CometShuffleExchange        Arrow C Data export -> CometVector (zero copy)
(with Comet)         over VectorToComet
```

`VectorColumnarRule` runs in Spark's `preColumnarTransitions`, bottom-up. An operator is converted
when its child is already columnar with supported types (a vectorized Parquet scan, a Comet scan, or
another spark-vector operator) and every expression compiles to the kernel IR. Otherwise the reason
is stored as a tree-node tag; `VectorFallback.reasons(plan)` lists them.

### The Vector Acceleration tab

`spark.plugins` also attaches a **Vector Acceleration** tab to the Spark UI (disable with
`spark.vector.ui.enabled=false`). It lists every SQL execution and, per execution, draws the final
physical plan as a DAG with each operator coloured by the engine that runs it:

| Colour | Engine | Counts as accelerated |
|---|---|---|
| green | our SIMD kernels (`VectorFilter`, `VectorProject`, `VectorHashAggregate`) | yes |
| blue | Comet's native operators (any class under `org.apache.spark.sql.comet` / `org.apache.comet`) | yes |
| purple | `VectorToComet`, the zero-copy hand-off to Comet's shuffle | yes |
| teal | a Spark scan that already emits batches, i.e. the vectorized Parquet reader | plumbing |
| grey, dashed | left to Spark | **no** |
| yellow | `ColumnarToRow` / `RowToColumnar` / `AQEShuffleRead` | plumbing |

A **Fully Accelerated** badge is shown when no operator is grey: every operator runs on the kernels
or on Comet, with only supported columnar sources and unavoidable row transitions around them. A
vectorized scan and a transition are plumbing rather than missed operators, so they do not block the
badge; a Spark exchange or sort does, because those are operators we do not implement (without
Comet's shuffle, any query with a stage boundary is therefore only partly accelerated).

Grey operators that the planner rule *tried* to convert are listed under "Why operators were not
accelerated" with the reason it recorded — the same information `spark.vector.explainFallback.enabled`
logs, and it reads as a cascade, since one uncompilable expression makes every operator above it
non-columnar:

```
Filter          unsupported expression StartsWith: startswith(lineitem.l_comment, 'cmt1')
Project         child Filter is not columnar
HashAggregate   child Project is not columnar
```

Node classification comes from the operator's identity, not from a tag: a `VectorExec` is ours, a
class in Comet's packages is Comet's, and anything we declined to convert is the original Spark class
carrying the fallback tag. The tab prefers the final plan objects (captured from
`SparkListenerSQLExecutionEnd`) and falls back to matching node names on the listener event's
serialised plan for queries still running, which it labels *approximate*.

Between two of our operators a filter (or projection) does not compact: when at least half of the
rows survive it forwards the child's columns with a selection bitmap (`SelectedColumnarBatch`), and
the consumer folds the bitmap into its validity masks or group assignment. Inside a predicate,
`AND`/`OR` evaluate their right operand only where the left one leaves the row undecided, so the
compare kernels skip 64-row blocks with no live row and ANSI errors are only raised for rows Spark
would have evaluated too. Compaction happens once, at the boundary to Spark.

Semantics follow Spark, including the corners: NaN-safe double ordering (`NaN = NaN`, NaN sorts
last), three-valued `AND`/`OR`, `WHERE` treating null as false, and ANSI mode (Spark 4's default):
double division by zero raises `DIVIDE_BY_ZERO`, while integer arithmetic under ANSI (which needs
overflow checks) falls back.

### Aggregation

The partial aggregate emits exactly Spark's buffer schema (`sum`, `count`, `min`, `max`,
`(sum, count)` for `avg`), so any exchange and Final aggregate run unchanged. Our own Final
aggregate merges those buffers per group (Spark's or ours) and evaluates the result expressions
with each aggregate's `evaluateExpression` substituted (`sum / count` for `avg`) through the
projection kernels; over Spark's row shuffle its input arrives through `RowToColumnarExec`, over
Comet's shuffle it is read zero copy. Without grouping keys, one buffer row per partition. With keys, a hash table assigns dense group ids across the
task's batches. When every key is a dictionary-encoded string (the usual case for low-cardinality
Parquet columns) the ids are memoised per combination of dictionary indices, so a batch probes the
table at most once per distinct key tuple. Rows are then scattered into per-group accumulators;
the alternative, one masked SIMD reduction per group, only wins for one group on 128-bit vectors
(`sparkvector.agg.maskPathMaxGroups` sets the cut-over, default 1 for ≤4 lanes and 8 above). The
scatter rotates over four independent accumulator copies so consecutive rows of the same group do
not serialise on one `sum[g] += x` chain (+40% at 4 groups); the price is that double sums are
rounded in a different order than Spark's sequential loop (12th significant digit on TPC-H Q1).

### Comet shuffle

Comet's planner only gives a Comet shuffle to children it recognises, but at run time its native
shuffle writer accepts any columnar child whose batches hold `CometVector`s. `VectorToCometExec`
exports each of our columns through the Arrow C Data Interface, written directly with the FFM API
(two C structs and an upcall release stub; no `arrow-c-data`, no JNI on our side, which matters
because Comet ships that library's classes under their original names with shaded signatures), and
Comet's `ArrowImporter` wraps the same memory in a shaded vector, releasing ours when it is done.
The rule replaces a Spark exchange, or Comet's row-based columnar exchange, above one of our
operators with the native `CometShuffleExchangeExec` over the bridge (hash, single and round-robin
partitioning; range partitioning is left to Spark). Requires `spark.shuffle.manager` set to
Comet's shuffle manager and `spark.comet.exec.shuffle.enabled=true`; see [docs/comet.md](docs/comet.md).

### Supported today

| Area | Supported | Falls back |
|---|---|---|
| Types | Int, Long, Double, Date, Timestamp, Boolean, String (strings pass through and serve as group keys) | Float, Short/Byte, Decimal, Binary, nested |
| Predicates | `=`, `!=`, `<`, `<=`, `>`, `>=` on numeric/date columns vs literal or column; `AND`/`OR`/`NOT`; `IS [NOT] NULL`; boolean columns | string comparisons, `LIKE`, `IN`, functions |
| Arithmetic | `+ - *` on Int/Long/Double, `/` on Double, unary minus, widening casts | ANSI integer arithmetic, `%`, decimals, other casts |
| Aggregates | `sum`, `count`, `min`, `max`, `avg` in Partial and Final mode; keys of Int/Long/Boolean/String/Date | `DISTINCT`, `FILTER` (Partial), double keys, PartialMerge/Complete modes, other functions |

## Benchmarks

Kernel microbenchmarks (JMH):

```bash
mvn -DskipTests package
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
     -jar benchmarks/target/benchmarks.jar "Compare|Compact|Agg"
```

The kernel tests can run with the lane counts of other platforms, emulated (slowly) on any machine,
which is how the AVX2 and AVX-512 code paths (`compress`, 256-entry shuffle tables, 8-lane masks)
are kept honest on a laptop:

```bash
mvn -pl kernels test -Dvector.jvm.args="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -Dsparkvector.vectorBits=512"
```

TPC-H Q1 and Q6 (decimals replaced by doubles, generated with DuckDB):

```bash
brew install duckdb
benchmarks/scripts/gen-tpch.sh 1                      # benchmarks/data/sf1/lineitem  (6M rows, 207 MB)
benchmarks/scripts/gen-tpch.sh 10                     # benchmarks/data/sf10/lineitem (60M rows, 2.1 GB)
mvn -DskipTests install
export JAVA_HOME=/opt/homebrew/opt/openjdk@25
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1    # spark + vector
COMET_JAR=/path/to/comet-spark-spark4.1_2.13-1.0.0.jar \
benchmarks/scripts/run-tpch.sh benchmarks/data/sf10   # + comet-scan, comet-scan-vector, comet-scan-vector-shuffle, comet
```

Each configuration runs in its own JVM and appends its measurements to
`benchmarks/results/<config>.jsonl`. The runner then rewrites two reports from every `.jsonl` file,
one section per dataset (`sf1`, `sf10`, ...): `benchmarks/results/results.md` and a self-contained
`benchmarks/results/results.html` with bar charts (median, p90 whisker, speedup against plain
Spark), the operators found in each final plan and a checksum proving all configurations returned
the same rows (to 10 significant digits). Regenerate them without benchmarking with
`benchmarks/scripts/run-tpch.sh --report`. See [docs/results.md](docs/results.md) for numbers
measured on an Apple M3 Pro; at SF10, Q1 runs 1.58x faster than Spark over Spark's own scan and
1.86x over Comet's scan (Comet end to end: 1.59x), while the highly selective Q6 stays at 0.84x
over Spark's scan.

### Vector API lessons

Three things silently turned SIMD code into something slower than a scalar loop on this machine
(Apple M3, NEON, 128-bit vectors), and only JMH revealed them:

1. `Vector.compare(op, ...)` / `lanewise(op, ...)` is only intrinsified when `op` is a compile-time
   constant. Every kernel loop spells its operator out; the others are complements or swapped
   operands.
2. `Vector.compress(mask)` is native only on AVX-512/SVE. For species of up to 8 lanes we index a
   precomputed `VectorShuffle` table by the selection bits and use `rearrange`.
3. `VectorMask.fromLong` is slow on NEON. Masks are built with a broadcast-AND-compare against lane
   bit constants, and fully valid 64-row blocks skip masks entirely.

Three more came out of profiling TPC-H rather than microbenchmarks (see
[docs/results.md](docs/results.md)):

4. Two 64-bit lanes are not worth a shuffle. Compacting doubles by `rearrange` on NEON lost to a
   scalar walk over the selection bits; 64-bit compaction uses the scalar walk when the species
   has two lanes, and every kernel bulk-copies 64-row blocks whose selection word is all ones.
5. Heap segments are slow inputs. Wrapping Spark's `double[]` with `MemorySegment.ofArray` instead
   of copying it into native memory doubled the filter's time; one copy into native memory wins.
6. Per-group masked reductions only pay off for one or two groups on 128-bit vectors; grouped
   aggregation scatters into per-group accumulators otherwise.

## Not in scope (yet)

- A columnar shuffle of our own. Without Comet, every stage boundary goes through Spark's row
  shuffle (`ColumnarToRowExec` above, `RowToColumnarExec` below); a `ShuffleExchangeLike` with a
  serializer that dumps the Arrow buffers would remove both.
- A Parquet-to-Arrow reader of our own; Comet's reader covers the zero-copy case.
- Decimal arithmetic (`Decimal(p<=18)` as long lanes), range-partitioned Comet shuffles, Sort and
  joins, running Spark's SQL test suite Comet-style.
