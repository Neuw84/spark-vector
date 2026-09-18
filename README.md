# spark-vector

A Spark SQL plugin that executes Filter, Project, HashAggregate (all four modes), Sort and
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
| `spark/` | Scala 2.13 + Java | `VectorPlugin`, session extension, `VectorColumnarRule`, expression compiler, `VectorFilterExec` / `VectorProjectExec` / `VectorHashAggregateExec` / `VectorSortExec` / `VectorTakeOrderedAndProjectExec` / `VectorLocalLimitExec` / `VectorGlobalLimitExec` / `VectorCollectLimitExec` / `VectorUnionExec` / `VectorCoalesceExec` / `VectorExpandExec` / `VectorBroadcastHashJoinExec` / `VectorShuffledHashJoinExec`, Arrow output, input adapters (Spark vectors, Arrow, Comet, Iceberg), the Vector Acceleration UI tab |
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
| `spark.vector.exec.mergeRows.enabled` | convert `MergeRowsExec`, the row-level operator of a `MERGE INTO` (#21), when the join below it is ours |
| `spark.vector.exec.project.enabled` | convert `ProjectExec` |
| `spark.vector.exec.aggregate.enabled` | convert `HashAggregateExec` |
| `spark.vector.exec.aggregate.final.enabled` | also convert Final-mode aggregates (their input is the shuffle) |
| `spark.vector.exec.sort.enabled` | convert `SortExec` over a columnar child (in memory, no spill) |
| `spark.vector.exec.takeOrdered.enabled` | convert `TakeOrderedAndProjectExec` (`ORDER BY ... LIMIT`) over a columnar child; the per-partition top-N is columnar, the final merge of at most `limit` rows per partition goes through Spark's single-partition shuffle |
| `spark.vector.exec.limit.enabled` | convert `LocalLimitExec` / `GlobalLimitExec` / `CollectLimitExec` over a columnar child (no offset); batches pass through until the boundary, the collect limit's final take goes through Spark's single-partition shuffle |
| `spark.vector.exec.union.enabled` | convert `UnionExec` when at least one child is columnar (row children go through Spark's `RowToColumnarExec`); keep it on with Spark 4.1.3, whose own columnar union concatenates co-partitioned children it reports as partition-aligned |
| `spark.vector.exec.coalesce.enabled` | convert `CoalesceExec` over a columnar child (no shuffle, batches forwarded) |
| `spark.vector.exec.window.enabled` | convert `WindowExec` for `row_number`, `rank`, `dense_rank` and whole-partition aggregates over any child (a row sort below is converted by Spark's transitions) |
| `spark.vector.exec.generate.enabled` | convert `GenerateExec` with `explode`/`posexplode` (and the `_outer` forms) over an array column: the other columns gathered through a repeat index built from the array lengths, the elements copied once per array from Spark's array vector |
| `spark.vector.exec.sample.enabled` | convert `SampleExec` without replacement over a columnar child: Spark's own Bernoulli sequence per partition as a selection bitmap, so a seed returns Spark's rows |
| `spark.vector.exec.localTableScan.enabled` | convert `LocalTableScanExec` (`VALUES`, local relations) into one batch per partition; **off by default** -- nothing to accelerate, it only lets small-table tests run our operators |
| `spark.vector.exec.expand.enabled` | convert `ExpandExec` (`ROLLUP` / `CUBE` / `GROUPING SETS`, the `count(distinct)` rewrite) over a columnar child: one borrowed-column batch per grouping set, no data copy |
| `spark.vector.exec.broadcastHashJoin.enabled` | convert `BroadcastHashJoinExec` when the streamed side is columnar or an exchange (the build side stays Spark's broadcast) |
| `spark.vector.exec.broadcastNestedLoopJoin.enabled` | convert `BroadcastNestedLoopJoinExec` (non-equi joins) when the streamed side is columnar or an exchange; inner/cross, semi/anti/existence and outer joins with the streamed side preserved |
| `spark.vector.join.maxBuildSize` | largest build side (bytes or a size string) the hash-style joins convert for -- they hold it in memory per task; default 1 GiB, or `spark.memory.offHeap.size / spark.executor.cores` when off-heap is configured; larger estimates stay with Spark, unknown estimates convert |
| `spark.vector.exec.shuffledHashJoin.enabled` | convert `ShuffledHashJoinExec` (both inputs are exchanges; Spark's row shuffle is converted below us) |
| `spark.vector.exec.sortMergeJoin.enabled` | **off by default**; re-express `SortMergeJoinExec` as our shuffled hash join when the smaller side's statistics fit `spark.vector.join.maxBuildSize` and no parent relies on the merge's ordering (#10). Opt-in because tie order under `ORDER BY` and unordered `LIMIT` picks can differ from Spark's order-preserving merge. Comet's equivalent replacement, `spark.comet.exec.forceShuffledHashJoin`, is also off by default (experimental) -- but Comet does not need it: it executes the merge join natively (`spark.comet.exec.sortMergeJoin.enabled`, on by default), while we have no merge join, so the rewrite is our only way to accelerate these joins. The benchmark configurations set it to `true` for that reason |
| `spark.vector.comet.shuffle.range.enabled` | also hand range-partitioned exchanges (global `ORDER BY`) to Comet's native shuffle |
| `spark.vector.exec.strictFloatingPoint` | **on by default**: double `sum`/`avg` round exactly like Spark (one accumulator per group, rows added in order). `false` uses lane-parallel and interleaved partial sums that differ from Spark's in the last bits (about 7% of aggregate kernel time, 2.5% of TPC-H Q1) and can make an equality between two double sums fail (TPC-H Q15 returns no rows). Comet's `spark.comet.exec.strictFloatingPoint` is the analogous switch with the opposite default (`false`) and mechanism (`true` makes Comet fall back to Spark for such operations; we compute the strict result in our kernels). The benchmark configurations run with `false`, matching Comet's default |
| `spark.vector.exec.selection.enabled` | pass selection bitmaps between our operators instead of compacting |
| `spark.vector.comet.shuffle.enabled` | feed Comet's native shuffle from our operators when Comet's shuffle is configured |
| `spark.vector.ui.enabled` | attach the Vector Acceleration tab to the Spark UI (default `true`) |
| `spark.vector.ui.retainedExecutions` | queries kept by that tab (default `100`) |
| `spark.vector.explainFallback.enabled` | log why each operator was left to Spark (default `false`) |

JVM system properties for the kernels: `sparkvector.vectorBits=128|256|512` forces a vector shape
(the default is the platform's preferred one), `sparkvector.agg.interleave=1|2|4` sets how many
accumulator copies the grouped aggregation rotates through when `spark.vector.exec.strictFloatingPoint`
is off (default 4; strict mode always uses one), `sparkvector.selection.minFraction` (default 0.5) is the
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
Filter          unsupported expression RLike: lineitem.l_comment RLIKE 'special.*requests'
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
| Types | Int, Long, Double, Date, Timestamp, Boolean, String (strings pass through and serve as group keys), Decimal of at most 18 digits (unscaled long lanes); Decimal of 19 to 38 digits as a two-limb DECIMAL128 lane that filters compact and projections forward (#257; arithmetic and keys follow in #258/#259); columns of any other type pass through a filter or projection untouched as Spark's vectors | Float, Short/Byte, Binary, nested -- as computed values or keys; Decimal above 18 digits as a computed value or key (yet) |
| Strings | `instr`/`locate`/`position`, `replace`, `translate`, `substring_index`, `split_part`, `find_in_set` (one byte-search primitive); `upper`/`lower`/`initcap` (ASCII in lanes, the rest through Spark's own case mapping), `trim`/`ltrim`/`rtrim`/`btrim` with a literal trim set; `concat`, `concat_ws`, `elt` (lengths summed across the inputs, one buffer); `length`/`len`/`char_length`, `octet_length`, `bit_length`, `ascii`, `chr`/`char` (measured once per dictionary entry); `substring`/`substr`, `left`, `right`, `lpad`, `rpad`, `repeat`, `space`, `overlay` with Spark's code-point semantics (a two-pass UTF8 writer: lengths and source ranges, then one sized buffer); literals or lanes as arguments; a per-batch output cap declines a runaway `repeat`/`space` | collated columns; column trim sets and `translate` tables; regular expressions; binary and array subjects |
| Nested columns | struct, array and map columns pass through filters and projections as Spark's own vectors; struct fields (`st.a`, `st.c.d`) are read from the struct vector's children, with the struct's nulls | `arr[i]`, `map[key]`, building a struct/array/map, the array/map/lambda function families (#50) |
| Hashes | `hash` (Spark's exact Murmur3 chain), `xxhash64`, `md5`, `sha1`, `sha2`, `crc32` (per-row digests inside the operator) | binary columns; non-literal `sha2` bit lengths |
| Predicates | `=`, `!=`, `<`, `<=`, `>`, `>=` on numeric/date/decimal/string columns vs literal or column (strings in `UTF8_BINARY` order, once per dictionary entry on dictionary pages); `IN (literals)`; `LIKE` against a literal with any number of `%` wildcards (`'p%'`, `'%p%'`, `'%a%b%'`: a multi-token matcher) and `startswith` / `endswith` / `contains`; `InSet` (long `IN` lists, one binary search per row); scalar subquery results as literals (incl. Spark's merged struct-valued ones); Spark's runtime bloom-filter probe (`might_contain(filter, xxhash64(key))`, through Spark's own `BloomFilter`); `xxhash64`; `<=>`; `isnan`; `BETWEEN`; `AND`/`OR`/`NOT`; `IS [NOT] NULL`; boolean columns and their comparisons | `LIKE` with `_` or an escape character, `rlike`, an `IN` set holding `NULL`, functions |
| Arithmetic | `+ - *` on Int/Long/Double/Decimal (integers overflow-checked in ANSI mode, raising Spark's error only for rows that survive earlier filters), `/` on Double and Decimal (Spark's half-up rounding; null or ANSI error past the precision), wide decimals (`decimal(p > 18)`, #257 / #258) as two `long` limbs: comparisons, `IN`, `+ - * /`, `abs`, negation and casts to and from the lane with Spark's rescale and overflow semantics, the exact `BigInteger` path for a row whose intermediate leaves 128 bits, unary minus (ANSI-checked on integers), casts: widening, narrowing (Spark's truncation and saturation; ANSI `CAST_OVERFLOW` on active rows), decimals, booleans both ways, date <-> timestamp under a fixed offset, number/boolean/date/timestamp -> string and string -> number/boolean/date/timestamp with Spark's own parsers and formatters per row (ANSI `CAST_INVALID_INPUT` on active rows); `abs`, `sign`, the transcendental and trigonometric family (`exp`/`log*`/`pow`/`sqrt`/`cbrt`, trig, hyperbolic, `atan2`, `hypot`, `degrees`/`radians` -- bit-identical to Spark: the kernel makes exactly Spark's `Math`/`StrictMath` call per lane), `%` / `pmod` / `div` on Int/Long/Double (zero divisors raise in ANSI mode, null otherwise, active rows only), `try_add` / `try_subtract` / `try_multiply` / `try_divide` / `try_mod` / `try_cast` (the ANSI masks null the row instead of raising), `try_sum` (the whole group nulled on overflow, as Spark) and `try_avg`, `greatest` / `least`, `nanvl`, `ceil` / `floor` / `rint` / `round` / `bround` with Spark's exact definitions (incl. negative scales and decimals on the unscaled value), bitwise `& | ^ ~`, the three shifts and `bit_count` on Int/Long, widening casts, casts between decimals, integers and doubles, literal columns | `try_*` on decimals, `try_to_number` / `try_to_binary`, `%` / `div` on decimals, `round`-family functions and a string source for a cast over a wide decimal, casts to binary, intervals, nested types and the lane-less tinyint/smallint/float |
| Dates | `year`, `month`, `dayofmonth`, `dayofyear`, `quarter`, `dayofweek`, `weekday`, `extract`, `trunc(date, year/quarter/month/week)`, `date_add`, `date_sub`, `datediff`, `last_day`, `add_months`, `next_day`, `weekofyear`, `make_date`, `unix_date`/`date_from_unix_date`, `timestamp_seconds`/`millis`/`micros` and `unix_seconds`/`millis`/`micros`; `date_format`/`from_unixtime` with literal patterns (Spark's own formatter per row); `cast(timestamp AS date)`, `hour`, `minute`, `second`, `months_between`, `date_trunc`, `unix_timestamp`/`to_unix_timestamp` under a UTC or fixed-offset session zone | zone-dependent arithmetic under a zone with rules (DST) -- see the Not planned table in `docs/expressions.md` |
| Conditionals | `CASE WHEN ... [ELSE] END`, `IF`, `COALESCE`, `NVL`, `NVL2`, `NULLIF`, `IFNULL` over any supported type, with `NULL`, numeric, string and boolean literal branches; a null condition counts as false, later branches are evaluated only where earlier ones did not match | conditionals producing wide decimals or nested types |
| Aggregates | `sum` (ANSI bigint sums overflow-checked), `count`, `count_if`, `min`, `max` (incl. booleans and strings), `avg`, `first`/`last`, `bool_and`/`bool_or`, `bit_and`/`bit_or`/`bit_xor`, `max_by`/`min_by`, the statistical family (`stddev`/`variance` pop and samp, `skewness`, `kurtosis`, `covar_*`, `corr`, `regr_*`; Spark's Welford update and merge, agreement to a relative tolerance) in every aggregate mode (`Partial`, `PartialMerge`, `Final`, `Complete`), with `FILTER` clauses, `DISTINCT` (Spark's rewrites, incl. TPC-H Q16's `count(distinct)`) and keys-only aggregates (`SELECT DISTINCT`, `UNION`); `sum`/`avg` of decimals up to 8/11 digits through Spark's own rewrite to long/double sums, wider ones through 128-bit accumulators emitting Spark's own wide buffers (a decimal `avg`'s result is Spark's own division over the merged buffer); keys of Int/Long/Boolean/String/Date/Decimal; Spark's `SortAggregateExec` for string buffers converted too | double keys, `stddev`/`variance`, `ObjectHashAggregateExec` functions (`collect_*`, `percentile_*`) |
| Sort | `SORT BY`/`ORDER BY` over a columnar child, every supported type as key, in memory | sorts over Spark's row shuffle (kept by Spark), spilling |
| Generate | `explode`, `posexplode`, `explode_outer`, `posexplode_outer` over an array column or a struct field of one (elements of any lane type; nulls, empty arrays and arrays longer than a batch) | `inline`, `stack`, `json_tuple`, generators over maps, user-defined generators, computed arrays |
| Windows | `row_number`, `rank`, `dense_rank` over `PARTITION BY ... ORDER BY ...` (one walk over the sorted input; a partition longer than a batch is one partition); whole-partition `sum`/`avg`/`count`/`min`/`max` and the rest of the aggregate family over non-decimal inputs (the default frame without `ORDER BY`, or `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`); running `sum`/`avg`/`count`/`min`/`max` (`ROWS` or `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, the latter the default with `ORDER BY`); the child may be Spark's row sort; the per-partition top-k Spark inserts under a `rank <= k` filter (`WindowGroupLimitExec`, both modes) ; `lag`/`lead` (literal offset and default), `first_value`/`last_value`/`nth_value` over the same frames ; `percent_rank`/`cume_dist`/`ntile`; sliding `sum`/`avg`/`count`/`min`/`max` over `ROWS BETWEEN a AND b` | `RANGE` frames with value offsets, decimal window aggregates (#28), `IGNORE NULLS` (#58) |
| Joins | broadcast and shuffled hash joins: inner, left/right/full outer, left semi, left anti (including the null-aware anti join Spark plans for `NOT IN (subquery)` over nullable columns), existence (`EXISTS` as a value), each with an optional non-equi condition; keys of Int/Long/Boolean/String/Date/Decimal | sort-merge joins, existence and null-aware anti joins, double keys |

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
benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1   # Iceberg merge-on-read variants of lineitem (docs/iceberg.md)
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1 spark,vector --iceberg benchmarks/data/iceberg --variant sf1.pos_10 --queries q1,q6,probe-count,probe-sum,probe-group
```

TPC-DS, the 99 queries (103 with the a/b variants) over the 24 tables, keeping the real
`DECIMAL(7,2)` and `DATE` columns; the query text is Spark's own (`tpcds/q*.sql` from the spark-sql
tests jar, the files its plan-stability suite runs), so each query's plan is the one Spark's optimizer
is tested against:

```bash
benchmarks/scripts/gen-tpcds.sh 1                     # benchmarks/data/tpcds-sf1/<table> (store_sales: 2.9M rows)
benchmarks/scripts/run-tpcds.sh benchmarks/data/tpcds-sf1                      # spark + vector, q1..q99 -> benchmarks/results/tpcds
benchmarks/scripts/run-tpcds.sh benchmarks/data/tpcds-sf1 spark,vector --queries q10,q35,q45
benchmarks/scripts/run-tpcds.sh --report
```

On a cluster the same runners take the session `spark-submit` built (`--cluster`), the tables from
a base URI or a catalog (`--tables s3a://bucket/tpcds/sf1000/parquet`, `--tables catalog:db`), and
write one `.jsonl` per run to any Hadoop file system (`--out s3a://...`), each row carrying Spark's
stage metrics for the query (executor time, GC, shuffle bytes, spill, peak memory) beside the plan
and the operator counts; `benchmarks/scripts/submit-cluster.sh <config> <tables> <dataset> <out>`
turns a configuration into the `spark-submit` line, and `run-tpcds.sh --cluster-report <out>` writes
the report in the layout of the data-on-EKS Comet benchmark (summary, speedup distribution,
regressions with their stage evidence, per-query table, environment) -- see `benchmarks/k8s/README.md`.

Each configuration runs in its own JVM and appends its measurements to
`benchmarks/results/<config>.jsonl` (`benchmarks/results/tpcds/<config>.jsonl` for TPC-DS). The runner then rewrites two reports from every `.jsonl` file,
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

Or as one command: `benchmarks/scripts/profile-query.sh benchmarks/data/sf10 comet-scan-vector q1`
runs, records and summarises (`benchmarks/scripts/jfr-summary.sh <file.jfr>` summarises any
recording: hot methods, the callers of the JDK-internal `MemorySegment` frames, the plugin's own
frames by self time, allocation, GC, waits and native methods -- text you can paste into an issue).

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
7. A lane nobody computes on still pays. The 128-bit decimal lane has no SIMD path (two-limb scalar
   loops throughout), yet carrying it through every operator moved TPC-DS from 65% to 73% of operators
   ours: the wins are the chains that stayed columnar around a wide column -- a `decimal(27,2)` running
   total no longer sends the window, the join on it and the aggregate above back to rows.

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

`ORDER BY ... LIMIT n` (`TakeOrderedAndProjectExec`) over a columnar child becomes
`VectorTakeOrderedAndProjectExec`, which runs the same per-partition sort with a limit and gathers only
the first `n` rows of each partition -- the part that touches every row stays columnar. Those at most
`n` rows per partition then go as rows through Spark's own single-partition shuffle, where Spark's
ordering takes the final top `n`, Spark's projection applies the select list and the result is
materialised as one columnar batch: the merge sees at most `n x partitions` rows. `OFFSET` falls back.
This is the operator TPC-H Q2, Q3, Q10, Q18 and Q21 end in; `spark.vector.exec.takeOrdered.enabled`
turns it off.

Plain `LIMIT n` is the same idea without the sort: `VectorLocalLimitExec` and `VectorGlobalLimitExec` let
whole batches through until the boundary and compact only the batch that crosses it (no further batch
is pulled from the child), and `VectorCollectLimitExec` -- the operator a query ending in `LIMIT` plans
to -- does that cut per partition and then takes the first `n` rows through Spark's single-partition
shuffle. `spark.vector.exec.limit.enabled` turns the three off; `OFFSET` falls back.

Two structural operators keep a columnar chain whole without computing anything. `VectorUnionExec`
concatenates its children's batches and is columnar as soon as one child is -- Spark's own union only is
when every child is, so a `VALUES` side or a row shuffle used to drop the whole union to rows; Spark's
transitions convert such a child through `RowToColumnarExec` below us. `VectorCoalesceExec` forwards the
child's batches through a shuffle-free `coalesce(n)`. `spark.vector.exec.union.enabled` and
`spark.vector.exec.coalesce.enabled` turn them off. `VectorSampleExec` (`TABLESAMPLE`, `df.sample` without
replacement) is a selection producer like the filter: it runs Spark's own Bernoulli sampler per partition
over the live rows, so the same seed returns exactly Spark's rows, and forwards the bitmap. `VectorLocalTableScanExec`
turns a `VALUES` relation into batches; it is off by default (`spark.vector.exec.localTableScan.enabled`)
because there is nothing to accelerate -- it only removes the row-to-columnar transition for small-table tests.

Window functions start with the ranking layer (#58): `VectorWindowExec` computes `row_number`, `rank`
and `dense_rank` in one walk over the input Spark already sorted by partition and order keys -- a new
partition where a partition key changes, a new peer group where an order key changes, counters carried
across batches -- and forwards every input column. The child may be Spark's row sort: `Window` sits
above `Sort` above an exchange, and without a columnar shuffle that sort stays Spark's, so Spark
converts rows to columns below us and the filter on the rank and everything above it run columnar.
Whole-partition aggregates (`sum(x) OVER (PARTITION BY k)`: the default frame without `ORDER BY`) reuse
the grouped aggregate functions with each partition as a group -- the value is computed exactly as the
Final aggregate computes it -- and hold a partition's rows in memory until it ends, since the value is
known only then. A filter `rank <= k` on a ranking window makes Spark plan a per-partition top-k
(`WindowGroupLimitExec`) below the window in two modes -- before the shuffle over whatever produced the
rows, and after the sort -- and both are ours: the same ranking walk, keeping a row while its rank is at
most k, so the operators feeding the shuffle stay columnar. Running frames (`sum(x) OVER (PARTITION BY k
ORDER BY d)`, whose default frame is `RANGE ... CURRENT ROW`, and the `ROWS` form) make each peer group
or row a group and combine its buffers with the running buffers before it -- sums and counts add,
`min`/`max` compare -- so the same result expression yields the running value. The offset functions
(`lag`, `lead`, `first_value`, `last_value`, `nth_value`) are one row of the partition each, read from
the held rows across batch boundaries; `percent_rank`, `cume_dist` and `ntile`, which need the partition
size, take the same path, as do sliding frames (`sum(x) OVER (... ROWS BETWEEN 2 PRECEDING AND 1
FOLLOWING)`), re-aggregated per row over the frame's rows in order exactly as Spark's sliding frames are.
Decimal window aggregates run over the 128-bit lane for whole-partition and running frames (#259); a sliding frame over a decimal and `RANGE` frames with value offsets are refused.

`ROLLUP`, `CUBE` and `GROUPING SETS` (and the rewrite Spark applies to `count(distinct)`) go through
`ExpandExec`, which duplicates every row once per grouping set with the unused keys nulled and a
grouping id appended. `VectorExpandExec` does that without copying a byte: for each grouping set the
output batch borrows the retained columns of the input batch, nulled keys are all-invalid constant
columns and the grouping id is a constant column, so an `n`-set expand emits `n` batches per input
batch and the aggregate above it -- the expensive part -- stays ours. The input batch is held until
its last projection has been consumed. `spark.vector.exec.expand.enabled` turns it off.

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
Decimal(12,2)` is `Decimal(25,4)`) fall back with a reason -- except directly under a decimal `sum`,
or `avg`, where products, sums and differences (and their nestings) are computed speculatively in 64 bits,
checked per row with `Math.multiplyHigh` and a sign-trick overflow test, and the rare row that overflows
is added to the 128-bit accumulator exactly (#26): TPC-H's `sum(l_extendedprice * (1 - l_discount))` and
`sum(x * (1 - d) * (1 + t))` stay ours. Output decimals are `BigIntVector`s
behind `VectorDecimalColumnVector`, which gives Spark's row conversion `getDecimal` over the lanes;
into Comet they are widened to the 128-bit C Data layout.

Spark's optimizer rewrites `sum` over a decimal of up to 8 digits into `MakeDecimal(sum(UnscaledValue(x)))`,
an ANSI `sum(bigint)`, and `avg` over one of up to 11 digits into a double average; both are compiled,
which is also why bigint sums are now overflow-checked in ANSI mode (`AggKernels.sumLongExact`, a
sign-trick overflow lane carried alongside the accumulator) instead of falling back. A wider decimal
sum (`Decimal(12,2)` and up) is accumulated in 128 bits per group on the Partial side and emitted as
Spark's `(sum, isEmpty)` buffer with the `sum` a wide Arrow decimal column; the Final merges those
buffers with Spark's `isEmpty` rules and applies its overflow check at emission (an error in ANSI
mode, null otherwise), so a wide decimal sum runs on our operators in both stages.

### Joins

`BroadcastHashJoinExec` becomes `VectorBroadcastHashJoinExec` when the streamed side is columnar or
an exchange (adaptive execution re-plans a shuffled join as a broadcast join over the bare shuffle
read; Spark converts it below us as for the shuffled hash join),
and `BroadcastNestedLoopJoinExec` (a join with no equi-keys) becomes `VectorBroadcastNestedLoopJoinExec`:
the same iterator with every broadcast row a candidate, the streamed rows chunked to a fixed pair
budget so the condition is evaluated over gathered pairs and the product is never materialised.
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
Sort-merge joins -- Spark's default for large equi joins -- are re-expressed as the shuffled hash
join when `spark.vector.exec.sortMergeJoin.enabled` is set (off by default): the smaller side by the
AQE stages' statistics becomes the per-task build table, it must fit `spark.vector.join.maxBuildSize`,
the sorts Spark placed for the merge are dropped, and a merge join whose ordering a parent relies on
(a window over the join key, a merge join above on the same key) stays Spark's. Same rows; tied rows
under `ORDER BY` and unordered `LIMIT`s can come out in another order than Spark's order-preserving
merge, so it is opt-in. Without it, `spark.sql.join.preferSortMergeJoin=false` or a `SHUFFLE_HASH`
hint make Spark plan the hash join directly.

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
SQL_TESTS_UPDATE_BASELINE=true benchmarks/scripts/run-spark-sql-tests.sh   # full run; rewrite the coverage floor
```

A few files are excluded by default (`VectorSQLQueryTestSuite.defaultExclude`): `explain*.sql`,
whose golden output is Spark's own physical plan, the DataSketches files (`hll`, `kllquantiles`,
`thetasketch`), whose library refuses to start on any JDK newer than 21, and `udtf/udtf.sql`, which
needs `pyspark` installed (the Python UDF variants skip themselves without it and count as ignored).
Everything else passes: 642 test cases, 111 ignored, with 2972 of the 33856 query executions running
at least one spark-vector operator. Passing is the low bar -- a file passes just as well when every
operator falls back -- so the run also prints a per-test-case table (executions, executions that ran
one of our operators, operators) split into the 147 cases that run our operators and the 437 that never
can (analyzer-only cases, DDL, files with no supported operator), and a full run compares every case
with the checked-in floor `spark-sql-tests/src/test/resources/vector-sql-coverage.tsv`: a case that
lost accelerated executions fails the suite, naming the case, because a fallback introduced by a planner
change is otherwise invisible; cases above the floor are listed, and `SQL_TESTS_UPDATE_BASELINE=true`
records them. The golden files pin Spark's summation order for doubles, so the suite's JVM runs with
`-Dsparkvector.agg.interleave=1`, the mode in which our double sums add in Spark's order (the default
rotates accumulators and can differ in the last digits; `docs/results.md`). The suite earns its keep:
its first run found a bare literal projection (`SELECT 1 FROM ... HAVING max(id) > 0`) that compiled
but could not be materialised, and the run that introduced the table found four more -- `nanvl`
evaluating its second argument eagerly (`nanvl(c, 1/c)` raised where Spark keeps `c`), `count(DISTINCT
3, 2)` evaluating a literal as a column, a collated string accepted as a lane (Spark's own
`RowToColumnarExec` cannot convert one), and a cross join whose build side was pruned to no columns
returning nothing. The test JVM runs with
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
  `HashedRelation` once per task), a merge join of our own (the opt-in rewrite above is a hash join), a spilling sort or join, decimals wider than 18
  digits (two 64-bit lanes per value would be the next step), and TPC-H on real decimals: the
  benchmark data keeps decimals as doubles because `Decimal(12,2)` arithmetic exceeds 18 digits.
