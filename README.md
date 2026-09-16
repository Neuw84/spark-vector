# spark-vector

A Spark SQL plugin that executes Filter, Project, HashAggregate (Partial and Final), Sort and
hash joins on Arrow-layout batches with the Java Vector API (`jdk.incubator.vector`). It follows the
architecture of [Apache DataFusion Comet](https://github.com/apache/datafusion-comet), but stays
entirely on the JVM: no native library, no JNI, no serialization boundary. Unsupported operators,
expressions or types fall back to Spark with a recorded reason. With Comet installed it can sit
between Comet's native Parquet scan and Comet's native shuffle, both reached zero copy.

- Spark 4.1.x, Scala 2.13, JDK 25 (the Vector API is still an incubator module)
- Input: Spark's vectorized Parquet reader (copied into Arrow layout once per batch, keeping
  dictionary-encoded strings as dictionary indices), Comet's native Parquet and Iceberg readers in
  scan-only mode (read zero-copy), or Iceberg's own JVM vectorized reader (read zero-copy, with
  merge-on-read deletes turned into a selection). See [docs/comet.md](docs/comet.md) and
  [docs/iceberg.md](docs/iceberg.md). Which operators convert, under what conditions and with
  which fallback reasons: [docs/operators.md](docs/operators.md); which expressions compile, on which
  types, and why the others fall back: [docs/expressions.md](docs/expressions.md).
- Output: unshaded Arrow 18.3.0 vectors (the version Spark bundles) wrapped in Spark's
  `ArrowColumnVector`, so Spark's own `ColumnarToRowExec` consumes them unchanged.

## Layout

| Module | Language | Contents |
|---|---|---|
| `kernels/` | Java 25 | `VectorBuffers` (Arrow-layout `MemorySegment`s), SIMD kernels: compare, bitmap logic, compaction, arithmetic, decimal rescaling and division, casts, reductions (plain and overflow-checked), group hashing and key table, grouped accumulators, sort, gather, column builder; scalar references used as test oracles |
| `spark/` | Scala 2.13 + Java | `VectorPlugin`, session extension, `VectorColumnarRule`, expression compiler, `VectorFilterExec` / `VectorProjectExec` / `VectorHashAggregateExec` / `VectorSortExec` / `VectorBroadcastHashJoinExec` / `VectorShuffledHashJoinExec`, Arrow output, input adapters (Spark vectors, Arrow, Comet, Iceberg), the Vector Acceleration UI tab |
| `benchmarks/` | Java + Scala | JMH kernel microbenchmarks and the TPC-H runner (all 22 queries) |
| `spark-sql-tests/` | Scala 2.13 | Spark's own SQL golden-file suite run with the plugin (profile `spark-sql-tests`, on demand only; see below) |

## Building

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@25      # any JDK 25
mvn verify                                          # kernels + Spark suites (Comet suites skipped)
mvn -Pcomet verify                                  # also runs the Comet-backed suites (see docs/comet.md)
mvn -Piceberg verify                                # also runs the Iceberg suites (see docs/iceberg.md)
mvn -Pcomet,iceberg verify                          # everything, including Comet's native Iceberg scan
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
| `spark.vector.exec.sort.enabled` | convert `SortExec` over a columnar child (in memory, no spill) |
| `spark.vector.exec.broadcastHashJoin.enabled` | convert `BroadcastHashJoinExec` when the streamed side is columnar (the build side stays Spark's broadcast) |
| `spark.vector.exec.shuffledHashJoin.enabled` | convert `ShuffledHashJoinExec` (both inputs are exchanges; Spark's row shuffle is converted below us) |
| `spark.vector.comet.shuffle.range.enabled` | also hand range-partitioned exchanges (global `ORDER BY`) to Comet's native shuffle |
| `spark.vector.exec.selection.enabled` | pass selection bitmaps between our operators instead of compacting |
| `spark.vector.comet.shuffle.enabled` | feed Comet's native shuffle from our operators when Comet's shuffle is configured |
| `spark.vector.ui.enabled` | attach the Vector Acceleration tab to the Spark UI (default `true`) |
| `spark.vector.ui.retainedExecutions` | queries kept by that tab (default `100`) |
| `spark.vector.explainFallback.enabled` | log why each operator was left to Spark (default `false`) |

JVM system properties for the kernels: `sparkvector.vectorBits=128|256|512` forces a vector shape
(the default is the platform's preferred one), `sparkvector.agg.interleave=1|2|4` sets how many
accumulator copies the grouped aggregation rotates through (default 4; 1 reproduces Spark's
floating-point rounding exactly), `sparkvector.selection.minFraction` (default 0.5) is the
surviving fraction below which a filter compacts instead of forwarding a selection, and
`sparkvector.agg.plainDictMaxEntries` (default 512) is the number of distinct values above which a
plain (non-dictionary) string group key stops being dictionary-encoded on the fly and is hashed and
compared per row instead.

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
`spark.vector.ui.enabled=false`). It lists every SQL execution and how much of it was accelerated:

![The Vector Acceleration tab listing TPC-H Q1 executions, each 89% accelerated](images/vector-ui.png)

Per execution, it draws the final physical plan as a DAG with each operator coloured by the engine
that runs it. This is TPC-H Q1 over Comet's scan with Comet's shuffle between our Partial and Final
aggregates, taken before the columnar sort existed: the final `Sort` was the one operator left to
Spark, which is why the query shows 89% rather than the badge (with `VectorSortExec` it is fully
accelerated):

![The plan of one Q1 execution: Comet scan, Vector filter, project and aggregates, the bridge into Comet's shuffle, and Spark's Sort](images/query-accel-details.png)

The colours:

| Colour | Engine | Counts as accelerated |
|---|---|---|
| green | our SIMD kernels (`VectorFilter`, `VectorProject`, `VectorHashAggregate`) | yes |
| blue | Comet's native operators (any class under `org.apache.spark.sql.comet` / `org.apache.comet`) | yes |
| purple | `VectorToComet`, the zero-copy hand-off to Comet's shuffle | yes |
| teal | a Spark scan that already emits batches, i.e. the vectorized Parquet reader | plumbing |
| grey, dashed | left to Spark | **no** |
| yellow | `ColumnarToRow` / `RowToColumnar`, the only nodes that convert between rows and batches | plumbing |
| light grey | `AQEShuffleRead` (adaptive execution's shuffle reader) and `ReusedExchange`/`ReusedSubquery`; they hand over whatever the exchange wrote, columnar when it is | plumbing |

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

One conversion the colours alone would hide: Comet's JVM shuffle (`CometColumnarExchange`, used when
its native writer is not) reads its child through `execute()`, so over one of our operators it turns
batches into rows and re-encodes them as Arrow. The node stays blue (it is Comet's), but its tooltip
says so. Comet's native shuffle (`CometExchange`), which the plugin selects for every partitioning
it can bridge, including range partitioning for global sorts, takes the batches directly.

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
operators with the native `CometShuffleExchangeExec` over the bridge, for hash, single, round-robin
and range partitioning (the last one when Comet's own
`spark.comet.shuffle.native.partitioning.range.enabled` is on: Comet samples the child for the
bounds with Spark's `RangePartitioner`, which is what Spark's exchange does too). Requires `spark.shuffle.manager` set to
Comet's shuffle manager and `spark.comet.exec.shuffle.enabled=true`; see [docs/comet.md](docs/comet.md).

### Supported today

| Area | Supported | Falls back |
|---|---|---|
| Types | Int, Long, Double, Date, Timestamp, Boolean, String (strings pass through and serve as group keys), Decimal of at most 18 digits (unscaled long lanes) | Float, Short/Byte, Decimal above 18 digits, Binary, nested |
| Predicates | `=`, `!=`, `<`, `<=`, `>`, `>=` on numeric/date/decimal columns vs literal or column; `AND`/`OR`/`NOT`; `IS [NOT] NULL`; boolean columns | string comparisons, `LIKE`, `IN`, functions |
| Arithmetic | `+ - *` on Int/Long/Double/Decimal, `/` on Double and Decimal (Spark's half-up rounding; null or ANSI error past the precision), unary minus, widening casts, casts between decimals, integers and doubles, literal columns | ANSI integer arithmetic, `%`, decimal results above 18 digits, other casts |
| Aggregates | `sum` (ANSI bigint sums overflow-checked), `count`, `min`, `max`, `avg` in Partial and Final mode; `sum`/`avg` of decimals up to 8/11 digits through Spark's own rewrite to long/double sums; keys of Int/Long/Boolean/String/Date/Decimal | `DISTINCT`, `FILTER` (Partial), double keys, PartialMerge/Complete modes, wider decimal sums, other functions |
| Sort | `SORT BY`/`ORDER BY` over a columnar child, every supported type as key, in memory | sorts over Spark's row shuffle (kept by Spark), spilling |
| Joins | broadcast and shuffled hash joins: inner, left/right/full outer, left semi, left anti, each with an optional non-equi condition; keys of Int/Long/Boolean/String/Date/Decimal | sort-merge joins, existence and null-aware anti joins, double keys |

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

TPC-H, all 22 queries over the eight tables (decimals replaced by doubles, generated with DuckDB):

```bash
brew install duckdb
benchmarks/scripts/gen-tpch.sh 1                      # benchmarks/data/sf1/<table>  (lineitem: 6M rows, 207 MB)
benchmarks/scripts/gen-tpch.sh 10                     # benchmarks/data/sf10/<table> (lineitem: 60M rows, 2.1 GB)
benchmarks/scripts/gen-tpch.sh 1 benchmarks/data --decimals   # benchmarks/data/sf1-decimal: real DECIMAL(15,2) columns
mvn -DskipTests install
export JAVA_HOME=/opt/homebrew/opt/openjdk@25
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1    # spark + vector, q1..q22 (--queries q1,q6 for a subset)
COMET_JAR=/path/to/comet-spark-spark4.1_2.13-1.0.0.jar \
benchmarks/scripts/run-tpch.sh benchmarks/data/sf10   # + comet-scan, comet-scan-vector, comet-scan-vector-shuffle, comet
```

Each configuration runs in its own JVM and appends its measurements to
`benchmarks/results/<config>.jsonl`. The runner then rewrites two reports from every `.jsonl` file,
one section per dataset (`sf1`, `sf10`, ...): `benchmarks/results/results.md` and a self-contained
`benchmarks/results/results.html` with bar charts (median, p90 whisker, speedup against plain
Spark), an accelerated-operators table per query (operators executed by our kernels or Comet over
the operators that count, the same classification the Vector Acceleration UI tab uses, so each closed
compatibility gap shows up as a query moving), the operators found in each final plan with the
reason for every one the planner declined to convert, and a checksum proving all configurations
returned the same rows (to 10 significant digits). Regenerate them
without benchmarking with
`benchmarks/scripts/run-tpch.sh --report`. See [docs/results.md](docs/results.md) for numbers
measured on an Apple M3 Pro; at SF10, Q1 runs 1.64x faster than Spark over Spark's own scan and
1.84x over Comet's scan and shuffle (Comet end to end: 1.62x), while the highly selective Q6 stays
at 0.86x over Spark's scan.

When a result is not what you expected, profile before theorising. Java Flight Recorder attaches
to a benchmark JVM with one environment variable, and `RESULTS_DIR` keeps the profiling run out of
the report:

```bash
JVM_EXTRA="-XX:StartFlightRecording=filename=/tmp/q1.jfr,settings=profile,dumponexit=true" \
RESULTS_DIR=/tmp/profiling \
benchmarks/scripts/run-tpch.sh benchmarks/data/sf10 comet-scan-vector --queries q1 --warmup 2 --iterations 5
$JAVA_HOME/bin/jfr view hot-methods /tmp/q1.jfr
$JAVA_HOME/bin/jfr print --events jdk.ExecutionSample --stack-depth 12 /tmp/q1.jfr   # callers of a hot frame
```

Every performance finding in this project came out of such a recording rather than out of the
median alone: the masked-reduction aggregate, triple string decoding, the two-lane shuffle table,
and most recently a zero-copy configuration that was slower than a copying one because Comet's scan
delivers plain strings where Spark's delivers dictionaries (`MemorySegment.mismatch` set-up was a
third of the aggregate). The `[tpch]` lines the runner prints with per-operator kernel time are the
first thing to read; the JFR hot-method list and the callers of any JDK-internal frame near the top
(`checkValidStateRaw`, `checkBounds`) are the second.

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

### Sort

`SortExec` over a columnar child becomes `VectorSortExec`: every batch of the partition is copied into
operator-owned native memory (Spark lets the producer reuse a batch once the next one is requested),
joined into one column per attribute, and a permutation is computed key by key, least significant
first. Each pass maps the key to an order-preserving unsigned 32-bit value (`int32` and booleans in
one pass, `int64` and doubles in two, strings of at most 8 bytes in three over their zero-padded
big-endian prefix, longer strings through a rank from one stable merge sort) and sorts `(key,
position)` packed into a `long` with `Arrays.sort`, so every pass is stable and the passes compose.
Nulls take one final pass per key. The output is gathered through the permutation into Arrow vectors
in batches of 4096 rows. Double ordering is Spark's (`-0.0 = 0.0`, NaN greatest and equal to itself),
strings compare as unsigned bytes like `UTF8String`.

Two deliberate limits: the sort is in memory only (no spill; turn it off with
`spark.vector.exec.sort.enabled=false` for partitions that need one), and it is planned only over a
columnar child. A global `ORDER BY` over Spark's row shuffle keeps `SortExec`: converting rows to
columns just to sort them gains nothing. Over Comet's columnar shuffle (or one of our operators, for
`SORT BY`) the sort is ours, which makes TPC-H Q1 with Comet's scan and shuffle fully accelerated.

### Decimals

A `Decimal(p, s)` with `p <= 18` travels through the kernels as its unscaled value in long lanes,
which is how Spark's own vectors and the Parquet reader hold small decimals (ints for `p <= 9`,
widened once on the way in). Comparisons are long comparisons: Spark's analyzer already casts both
sides to one type. Additions and subtractions rescale the operands to the result scale,
multiplications multiply the unscaled values; Spark's result precision always has room for them, so
none can overflow. Division follows Spark exactly: one long division with a half-up correction when
the scaled dividend fits (with a divisor below 10^18 Spark's two roundings cannot disagree with
it), `BigDecimal` otherwise; a quotient past the result precision is null in legacy mode and an
error in ANSI mode, like a cast that does not fit. Results wider than 18 digits (`Decimal(12,2) *
Decimal(12,2)` is `Decimal(25,4)`) fall back with a reason. Output decimals are `BigIntVector`s
behind `VectorDecimalColumnVector`, which gives Spark's row conversion `getDecimal` over the lanes;
into Comet they are widened to the 128-bit C Data layout.

Spark's optimizer rewrites `sum` over a decimal of up to 8 digits into `MakeDecimal(sum(UnscaledValue(x)))`,
an ANSI `sum(bigint)`, and `avg` over one of up to 11 digits into a double average; both are compiled,
which is also why bigint sums are now overflow-checked in ANSI mode (`AggKernels.sumLongExact`, a
sign-trick overflow lane carried alongside the accumulator) instead of falling back. Wider decimal
sums keep their `(sum, isEmpty)` buffer of more than 18 digits and stay with Spark.

### Joins

`BroadcastHashJoinExec` becomes `VectorBroadcastHashJoinExec` when the streamed side is columnar.
The build side is left exactly as Spark planned it, a `BroadcastExchangeExec` producing a
`HashedRelation`: each task reads its rows once into columns and builds a `GroupKeyTable` over the
keys (with a lookup-only probe), so no exchange of our own is needed and a Spark join over the same
broadcast keeps working. `ShuffledHashJoinExec` becomes `VectorShuffledHashJoinExec`; like the
Final aggregate it accepts exchanges as inputs, so it runs over Spark's row shuffle (a
`RowToColumnarExec` on each side) as well as over Comet's. Probing evaluates the streamed keys with
the kernels, looks every row up in one pass, expands the match chains into two index arrays and
gathers the output with `GatherKernels` (`-1` pads the unmatched rows of outer joins); semi and anti
joins are a selection over the streamed batch, compacted once; an inner join's non-equi condition is
evaluated on the joined batch and the failing rows compacted away. Semi, anti and outer joins with a
condition gather the candidate pairs of each streamed row first, evaluate the condition over them
and only then decide what the row becomes (kept or dropped, its passing pairs or one padded row);
a full outer join remembers which build rows were paired and emits the rest after the last
streamed batch. Null keys never match. Double
keys are refused because Spark compares them after NaN/zero normalisation and the key table by bits.
Sort-merge joins are not converted; with `spark.sql.join.preferSortMergeJoin=false` or a
`SHUFFLE_HASH` hint Spark plans the hash join instead.

### Spark's SQL test suite

Comet validates itself by running Spark's own SQL golden-file tests with its extension injected;
so can this plugin. The `spark-sql-tests` module (only built under the `spark-sql-tests` profile,
never by `mvn verify`: the whole suite takes about 15 minutes) subclasses `SQLQueryTestSuite` from
the `spark-sql` tests jar, unpacks its `sql-tests/` golden files and `test-data/`, sets
`spark.sql.extensions` to ours and requires every golden result to match whether an operator was
converted or not. Run it on demand, whole or by a regex over test-case names:

```bash
benchmarks/scripts/run-spark-sql-tests.sh                 # everything
benchmarks/scripts/run-spark-sql-tests.sh '^(group-by|join|decimal)'
SQL_TESTS_EXCLUDE='^$' benchmarks/scripts/run-spark-sql-tests.sh   # include the excluded files too
```

A few files are excluded by default (`VectorSQLQueryTestSuite.defaultExclude`): `explain*.sql`,
whose golden output is Spark's own physical plan, the DataSketches files (`hll`, `kllquantiles`,
`thetasketch`), whose library refuses to start on any JDK newer than 21, and `udtf/udtf.sql`, which
needs `pyspark` installed (the Python UDF variants skip themselves without it and count as ignored).
Everything else passes: 642 test cases, 111 ignored, with 1806 of the 33764 query executions running
at least one spark-vector operator (the last line of the run reports these counts). Its first run
found a bare literal projection (`SELECT 1 FROM ... HAVING max(id) > 0`) that compiled but could not
be materialised, which is exactly the kind of gap it exists to catch. The test JVM runs with
`-Dspark.testing=true` (the golden files assume Spark's test-mode defaults, such as the TIME type)
and `-XX:-OmitStackTraceInFastThrow`: after enough ANSI overflows in one JVM, HotSpot's preallocated
`ArithmeticException` carries no message and Spark's error formatting fails on the null (in Spark's
own operators, not ours).

## Not in scope (yet)

- A columnar shuffle of our own. Without Comet, every stage boundary goes through Spark's row
  shuffle (`ColumnarToRowExec` above, `RowToColumnarExec` below); a `ShuffleExchangeLike` with a
  serializer that dumps the Arrow buffers would remove both.
- A Parquet-to-Arrow reader of our own; Comet's reader covers the zero-copy case.
- A columnar broadcast exchange of our own (the build side of a broadcast join is read from Spark's
  `HashedRelation` once per task), sort-merge joins, a spilling sort or join, decimals wider than 18
  digits (two 64-bit lanes per value would be the next step), and TPC-H on real decimals: the
  benchmark data keeps decimals as doubles because `Decimal(12,2)` arithmetic exceeds 18 digits.
