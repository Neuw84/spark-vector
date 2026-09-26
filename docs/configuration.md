---
layout: default
title: Configuration reference
---

# Configuration reference

Every `spark.vector.*` key is a session-level Spark SQL configuration: set it with `--conf` on
`spark-submit`, in `spark-defaults.conf`, or at run time with `spark.conf.set(...)`. The planner
reads them from the session's `SQLConf` when it plans a query, so a change applies to the next query
(the two UI keys and the shuffle keys are read from the `SparkConf` at start-up, see the notes in
their tables). The defaults were set from measurements on TPC-DS at 1 TB and TPC-H at SF10 -- see
`docs/results.md` for the runs behind each threshold. Size-valued keys accept Spark's size strings
(`512m`, `1g`) as well as byte counts. Where a `spark.vector.*` key is read is named in
`spark/src/main/scala/io/sparkvector/spark/VectorConf.scala` (the planner and operator keys),
`spark/src/main/scala/org/apache/spark/sql/vector/AggregateSpill.scala` (the aggregate spill keys) and
the `shuffle` module (the columnar shuffle keys).

## Main switch and diagnostics

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.enabled` | `true` | boolean | Main switch: `false` leaves every operator to Spark and the plugin only attaches its UI tab. |
| `spark.vector.explainFallback.enabled` | `false` | boolean | Logs why each operator was left to Spark (the same reasons the Vector Acceleration tab shows); turn it on when a plan is not as accelerated as expected. |
| `spark.vector.ui.enabled` | `true` | boolean | Attaches the Vector Acceleration tab to the Spark UI; read from the `SparkConf` by the driver plugin before any session exists, so it must be set at start-up. |
| `spark.vector.ui.retainedExecutions` | `100` | positive int | How many SQL executions the Vector Acceleration tab keeps; also read from the `SparkConf` at start-up. |

## Operator switches

Each converts one Spark operator into its vecruntime counterpart when the conditions in
`docs/operators.md` hold; set one to `false` to keep Spark's operator for that kind while the rest
of the plan stays columnar.

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.exec.filter.enabled` | `true` | boolean | Convert `FilterExec`. |
| `spark.vector.exec.project.enabled` | `true` | boolean | Convert `ProjectExec`. |
| `spark.vector.exec.aggregate.enabled` | `true` | boolean | Convert `HashAggregateExec` (and the `SortAggregateExec` Spark plans for string buffers). |
| `spark.vector.exec.aggregate.final.enabled` | `true` | boolean | Also convert the merging modes (`Final`, `PartialMerge`), whose input is a shuffle; `false` keeps Spark's Final aggregate over our partial one. |
| `spark.vector.exec.aggregate.rollupRewrite.enabled` | `true` | boolean | A partial aggregate over our `Expand` (`ROLLUP`, `CUBE`, `GROUPING SETS`) aggregates the finest grouping once and rolls the partials up instead of hashing every input row once per grouping set (#383); `false` disables the rewrite. |
| `spark.vector.exec.mergeRows.enabled` | `true` | boolean | Convert Iceberg's `MergeRowsExec` (the row-level operator of a `MERGE INTO`, #21) when the join below it is ours. |
| `spark.vector.exec.sort.enabled` | `true` | boolean | Convert `SortExec` over a columnar child; a global `ORDER BY` over Spark's row shuffle stays Spark's regardless. |
| `spark.vector.exec.takeOrdered.enabled` | `true` | boolean | Convert `TakeOrderedAndProjectExec` (`ORDER BY ... LIMIT n`) over a columnar child; the per-partition top-n is columnar, the final merge goes through Spark's single-partition shuffle. |
| `spark.vector.exec.limit.enabled` | `true` | boolean | Convert `LocalLimitExec` / `GlobalLimitExec` / `CollectLimitExec` over a columnar child; `OFFSET` falls back. |
| `spark.vector.exec.union.enabled` | `true` | boolean | Convert `UnionExec` when at least one child is columnar (row children go through `RowToColumnarExec`); keep it on with Spark 4.1.3, whose own columnar union concatenates co-partitioned children (#128). |
| `spark.vector.exec.coalesce.enabled` | `true` | boolean | Convert `CoalesceExec` (a shuffle-free `coalesce(n)`) over a columnar child. |
| `spark.vector.exec.expand.enabled` | `true` | boolean | Convert `ExpandExec` (grouping sets, the `count(distinct)` rewrite) over a columnar child: one borrowed-column batch per projection, no data copy. |
| `spark.vector.exec.sample.enabled` | `true` | boolean | Convert `SampleExec` without replacement over a columnar child, running Spark's own Bernoulli sampler per partition so a seed returns Spark's rows. |
| `spark.vector.exec.generate.enabled` | `true` | boolean | Convert `GenerateExec` for `explode` / `posexplode` (and the `_outer` forms) over an array column or a struct field of one. |
| `spark.vector.exec.window.enabled` | `true` | boolean | Convert `WindowExec` (ranking functions, whole-partition, running and sliding aggregates, the offset functions) and the `WindowGroupLimitExec` Spark plans below a `rank <= k` filter; the child may be Spark's row sort. |
| `spark.vector.exec.localTableScan.enabled` | `false` | boolean | Convert `LocalTableScanExec` (`VALUES`, small local relations) into one batch per partition; off because there is nothing to accelerate, it only lets small-table tests run our operators. |
| `spark.vector.exec.broadcastHashJoin.enabled` | `true` | boolean | Convert `BroadcastHashJoinExec` when the streamed side is columnar or an exchange; the build side stays Spark's broadcast `HashedRelation`. |
| `spark.vector.exec.broadcastNestedLoopJoin.enabled` | `true` | boolean | Convert `BroadcastNestedLoopJoinExec` (joins without equi-keys) when the streamed side is columnar or an exchange. |
| `spark.vector.exec.shuffledHashJoin.enabled` | `true` | boolean | Convert `ShuffledHashJoinExec`; over Spark's row shuffle both inputs go through `RowToColumnarExec`. |
| `spark.vector.exec.sortMergeJoin.mode` | `auto` | `off`, `hash`, `merge`, `auto` | What becomes of `SortMergeJoinExec` (#286, #287): `off` leaves it to Spark, `hash` re-expresses it as our shuffled hash join, `merge` plans our order-preserving merge join over Spark's sorted inputs, `auto` decides per join -- the merge join where a parent relies on the ordering, where the row order can reach a limit or a sort, or where the hash rewrite is not allowed (no statistics, both sides past `spark.vector.join.hashMaxBuildSize`, a skew join), the hash rewrite where a side's statistics fit. Since #416 nothing is left to Spark's own operator under `auto`. |
| `spark.vector.exec.sortMergeJoin.enabled` | unset | boolean | Compatibility alias for the mode: `false` reads as `mode=off`, anything else as `auto`; ignored when `spark.vector.exec.sortMergeJoin.mode` is set. Prefer the mode key. |

## Execution semantics

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.exec.strictFloatingPoint` | `true` | boolean | Double `sum` / `avg` round exactly like Spark's (one accumulator per group, rows added in order); `false` uses lane-parallel and interleaved partial sums that differ in the last bits, buying back about 7% of aggregate kernel time (2.5% of TPC-H Q1 at SF10) but able to make an equality between two double sums fail (TPC-H Q15). The benchmark configurations run with `false`, Comet's default. |
| `spark.vector.exec.selection.enabled` | `true` | boolean | Pass a selection bitmap between our operators instead of compacting each filtered batch (compaction still happens below `sparkvector.selection.minFraction` survivors and at the boundary to Spark); a debugging switch. |
| `spark.vector.agg.dictionaryKeys` | `true` | boolean | The grouped aggregate emits its UTF8 keys dictionary-encoded, with ids over the group table's own dictionary (#377), so the shuffle writer stages ids and the final aggregate maps dictionary entries rather than rows; `false` emits plain strings. |

## Memory budgets and spilling

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.agg.spillThreshold` | `1g` | size, `0` = never | Hard cap on one grouped aggregate table; below it the operator acquires its real footprint from Spark's task memory manager and acts on a refusal (#363, #367). Past the budget a buffer-emitting mode emits its table and starts over, a merging mode spills into hash buckets; `0` keeps everything in memory. |
| `spark.vector.agg.spillBuckets` | `16` | int (at least 2) | Buckets a merging (Final) aggregate spills its table into and merges one at a time; each bucket is merged in memory, so the buckets, not the input, must fit. |
| `spark.vector.agg.passThroughRatio` | `1.5` | double, `0` = never | A partial aggregate whose full table reduced its input by less than this factor stops aggregating and passes each batch on to the exchange (#376); `0` keeps aggregating whatever the ratio. |
| `spark.vector.sort.runRows` | `1048576` | positive int | Rows per sorted run (#285): the sort orders each run as the partition arrives and k-way merges the runs on output, bounding its JVM scratch to the run rather than the partition. |
| `spark.vector.sort.spillBytes` | `1g` (1 GiB) | size, `0` = off | The sort's memory budget per task (#416): sealed runs past it are written to local disk in sorted order and merged from there, which is what lets the merge join take inputs of any size. Measured alone at 1 TB in the full run's shape, the 32 MB of #451 cost q14a 30% and q4 50% against 1 GiB because spilled runs share the node disk with the shuffle files, hence the 1 GiB default. |
| `spark.vector.join.maxBuildSize` | 1 GiB, or `spark.memory.offHeap.size / spark.executor.cores` when off-heap is enabled | size | The largest build side, by statistics, the two broadcast joins convert for (#86); they hold the relation in memory per task for the whole query. The shuffled hash join is no longer gated by it (#416). |
| `spark.vector.join.spillBytes` | `256m` | size, `0` = never split | Build bytes a shuffled hash join holds in memory before it splits both sides into buckets on disk (#416, the grace hash join). At 1 TB on the eight join-heaviest TPC-DS queries 256 MB and 1 GiB were the same within the band and neither spilled, so the smaller one leaves the headroom to the sort and the operators beside the join. |
| `spark.vector.join.spillBuckets` | `32` | int | Buckets a split shuffled hash join writes each side into and joins one at a time (#416); `1` or less turns the split off. |
| `spark.vector.join.hashMaxBuildSize` | `spillBuckets x spillBytes` (8 GiB) | size, `0` = no cap | The most a sort-merge join's build side may weigh per task, by statistics, for `mode=auto` to make it the hash join: within `spark.vector.join.spillBytes` it builds in memory, within this cap it splits into buckets once, past it -- or without an estimate -- the merge join over the spilling sort takes it (#416). |

## Scan

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.scan.prefetch` | `0` (off) | int 0-8 | Depth of the prefetching converter's queue (#403, lever 2): `1` or `2` inserts `VectorPrefetchScanExec` between a Spark vectorized file scan (Parquet, Iceberg's `BatchScanExec`; not a Comet scan) and the first operator of ours above it, whose helper thread pulls and converts the reader's next batch while the task thread works on the previous one. Memory grows by that many converted batches per task; larger values are accepted and capped at 8. |

## Columnar shuffle (the `spark-vector-shuffle` jar)

`spark.vector.shuffle.enabled` is a session key; the rest are read from the `SparkConf` by the
shuffle manager, the writer and the Flight server at start-up, so set them on `spark-submit`. None
of them has an effect unless `spark.shuffle.manager` names `VectorShuffleManager` (see the last
section) and the shuffle jar is on the classpath.

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.shuffle.enabled` | `true` | boolean | Replace a `ShuffleExchangeExec` above one of our operators with `VectorShuffleExchangeExec` (Arrow IPC record batches per reduce partition, #288) for hash, round-robin, single and range partitioning; Comet's native shuffle takes precedence where it is configured. |
| `spark.vector.shuffle.backend` | `flight` | `flight`, `block`, or a class name | How a reducer fetches a remote map output: `flight` opens one Arrow Flight `DoGet` per remote executor carrying all of its blocks for the reducer, `block` uses Spark's block transfer over the same files (use it where TLS is required, since the Flight server has none), or the class name of a `VectorShuffleBackend` from another jar (where a push-based service would plug in). |
| `spark.vector.shuffle.compression` | `zstd` | `zstd`, `lz4`, `none` | Body compression of the shuffle's record batches; `lz4` is Arrow's pure-Java codec and an order of magnitude slower, `none` writes about 1.8x Spark's lz4 bytes. |
| `spark.vector.shuffle.batchRows` | `8192` | int | Rows a map task holds per reduce partition before writing them as one record batch. |
| `spark.vector.shuffle.batchBytes` | `1m` | size | Bytes a map task holds per reduce partition before writing them as one record batch (whichever of rows or bytes fills first). |
| `spark.vector.shuffle.bufferBytes` | `64m` | size | Cap on what one map task holds across all reduce partitions before flushing, measured on the writer's allocator rather than estimated (#340). |
| `spark.vector.shuffle.flushBytes` | `1m` | size | Serialised bytes a map task keeps in memory per reduce partition before spilling that partition's stream to a temporary file merged into the data file at commit. |
| `spark.vector.shuffle.writer.memoryLimit` | `1g` | size | The hard limit of one map task's Arrow allocator; the writer flushes long before it, this is the backstop behind `bufferBytes`, and hitting it fails the task with a serializable memory error. |
| `spark.vector.shuffle.writer.dictionaryMaxRatio` | `0.5` | double 0-1 | A string column is dictionary-encoded on the wire only when distinct values / rows in the record batch is at most this (#356); `1` always encodes, `0` never. |
| `spark.vector.shuffle.rebalance.rowSizing` | `true` | boolean | Session key. A rebalance exchange (the one a data source such as Iceberg asks for ahead of its write) hands AQE partition sizes proportional to rows instead of our compressed bytes, whose bytes per row vary several-fold between kinds of rows (#485). Reducers still fetch by the real sizes. |
| `spark.vector.shuffle.rebalance.advisoryScaling` | `true` | boolean | Session key. A rebalance's advisory partition size is sized for Spark's shuffle (Iceberg: target file size x Spark's expected shuffle compression); once the map stage has written, it is scaled by our uncompressed bytes per row over an estimate of Spark's `UnsafeRow` bytes for the same rows, so AQE's pieces hold about as many rows as Spark's (#20: the CDC MERGE's write stage took 40-46 s at Iceberg's 384 MiB and Spark's ~24 s at a hand-set 128 MiB). A size set with `spark.sql.iceberg.advisory-partition-size` is kept as given, in our bytes; one set as an Iceberg write option or table property is scaled like the default -- set this key to `false` to keep it exact. |
| `spark.vector.shuffle.aqe.mapSizeScaling` | `true` | boolean | Session key; `false` restores AQE on our real bytes. Every exchange of ours other than a rebalance reports its map output sizes to AQE times Spark's estimated on-disk bytes per row over ours (`UnsafeRow` estimate with measured string bytes, over `sparkCompressionRatio`, bounded to [1/16, 16]), so coalescing, skew detection and join groups see about Spark's bytes for the same rows. Only the statistics change; reducers fetch the real bytes (#511: our shuffle is 1.6-4.3x smaller than Spark's at TPC-DS 1 TB, so AQE packed up to 2x Spark's rows into a task -- q67's final aggregate ran 150 tasks where Spark's ran 300, and spilled). |
| `spark.vector.shuffle.aqe.sparkCompressionRatio` | `0` | double | The compression `mapSizeScaling` expects of Spark's shuffle (uncompressed `UnsafeRow` bytes over bytes on disk). `0` or negative, the default, assumes Spark's shuffle compresses as well as ours and takes the uncompressed ratio. On TPC-DS 1 TB (Graviton4, advisory 128m) that gave 1,835.6 s against 1,986.4 s unscaled; `2.5` (Spark's measured 2.57) gave 1,973.8 s, because the larger factor stopped AQE merging the short queries' small partitions. |
| `spark.vector.shuffle.flight.bindHost` | the executor's block manager host | host name | The address the executor's Flight server binds to; change it when the executor's advertised host is not the one it can bind. |
| `spark.vector.shuffle.flight.threads` | `max(4, available processors)` | int | Serving threads of the executor's Flight server. |

## Comet integration

These matter only with Comet's jar on the classpath; see `docs/comet.md`.

| Key | Default | Type / values | What it does |
|---|---|---|---|
| `spark.vector.comet.shuffle.enabled` | `true` | boolean | Rewrite a Spark exchange (or Comet's row-based columnar exchange) above one of our operators into Comet's native shuffle over the zero-copy `VectorToCometExec` bridge; requires Comet's shuffle manager and `spark.comet.exec.shuffle.enabled=true`. |
| `spark.vector.comet.shuffle.range.enabled` | `true` | boolean | Also hand range-partitioned exchanges (global `ORDER BY`) to Comet's native shuffle, which then samples the child with Spark's `RangePartitioner` as Spark's own exchange does; gated as well by Comet's `spark.comet.shuffle.native.partitioning.range.enabled`. |
| `spark.vector.comet.mixed.enabled` | `false` | boolean | Mixed chains (#280): a Spark operator left to Spark whose children are ours is offered to Comet's native operator through the sink leaf, for the operator kinds `spark.vector.comet.preferComet` names. |
| `spark.vector.comet.preferComet` | empty | comma-separated operator kinds, optionally qualified | The allowlist of the mixed pass (#281): kinds (`filter`, `project`, `sort`, `sortMergeJoin`, `hashJoin`, `broadcastHashJoin`, `window`, `expand`, `union`, `limit`, `all`) optionally qualified (`project:wideDecimal`, `filter:strings`, `sort:estimatedRows>1000000`); a listed operator is offered to Comet first and ours steps aside, one Comet declines is ours after all. Empty means mixed plans allowed, none requested; the #281 study found no entry that meets `docs/comet.md`'s three-part rule. |

## Non-`spark.vector` keys the plugin needs

- **JVM options on driver and executors.** The kernels use the incubating Vector API and the
  FFM API, so both JVMs need `--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED`
  in `spark.driver.extraJavaOptions` and `spark.executor.extraJavaOptions`
  (`--sun-misc-unsafe-memory-access=allow` silences the `Unsafe` deprecation warnings from Spark
  and Arrow on JDK 25). Every batch our operators produce, the shuffle's record batches and the
  Flight server's buffers are Arrow direct memory, bounded by `-XX:MaxDirectMemorySize`, which
  defaults to the heap size: set it to the executor's memory overhead less what the JVM itself needs
  (the 1 TB campaign ran a 30 GB heap and 20 GB of overhead per 13-core executor, the cluster
  manifests setting the bound to the overhead less 2 GB; see "Memory tuning" in `README.md`).
- **Registering the plugin.** `spark.plugins=io.sparkvector.spark.VectorPlugin` registers the
  session extension and attaches the UI tab; alternatively
  `spark.sql.extensions=io.sparkvector.spark.VectorSparkSessionExtensions` injects the planner
  rule alone.
- **The columnar shuffle manager.**
  `spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager` (from the
  `spark-vector-shuffle` jar) is what makes `spark.vector.shuffle.enabled` take effect; the manager
  serves our dependencies with the Arrow IPC writer and reader and delegates every other shuffle to
  Spark's sort shuffle. With `spark.authenticate` on, the Flight server requires Spark's shuffle
  secret as a bearer token; under `spark.ssl.rpc.enabled` it refuses to start, so use
  `spark.vector.shuffle.backend=block` there.
- **Comet's shuffle instead.** To feed Comet's native shuffle from our operators set
  `spark.shuffle.manager` to Comet's `CometShuffleManager` and `spark.comet.exec.shuffle.enabled=true`.
- **Kernel tuning knobs** are JVM system properties, not Spark confs, because the kernels have no
  Spark dependency: `sparkvector.vectorBits`, `sparkvector.platform`, `sparkvector.agg.interleave`,
  `sparkvector.agg.maskPathMaxGroups`, `sparkvector.selection.minFraction`,
  `sparkvector.agg.plainDictMaxEntries`; `README.md` describes each.
