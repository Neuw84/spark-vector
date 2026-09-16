# Spark operator support

One row per Spark physical operator: whether spark-vector converts it, what has to hold for the
conversion to happen, the `spark.vector.*` key that turns it off, and the exact fallback reason the
planner records when it does not convert. Modelled on Comet's
[Spark Operator Support](https://datafusion.apache.org/comet/user-guide/latest/operators.html) page;
the companion for expressions is `docs/expressions.md` (#30).

**How to read a plan.** `VectorExecRule` (`VectorColumnarRule.scala`) walks the physical plan bottom-up
and replaces an operator only when (a) its input already produces columnar batches of supported types
-- a vectorized Parquet scan, a Comet scan, an Iceberg scan through the adapter, or another
spark-vector operator -- and (b) every expression it carries compiles. Anything else is left as
Spark's operator with the reason attached as a `VectorFallback` tag. There is no silent fallback: the
Vector Acceleration UI tab colours every node by the engine that runs it and shows the reason on
hover, `spark.vector.explainFallback.enabled=true` logs each reason at INFO, the TPC-H runner records
them per query, and the test suites assert on the strings (`checkFallback(..., reasonContains)`). The
strings in the **Fallback reasons** column are therefore a contract: change one and the suites and
this page change with it.

**Supported types** are exactly the `VecType` lanes: `boolean`; `int` and `date` (INT32); `bigint`,
`timestamp` and `decimal(p <= 18)` as the unscaled value (INT64); `double` (FLOAT64); `string` (UTF8).
An operator whose input or output carries any other type -- `decimal(p > 18)`, `float`, `short`,
`byte`, `binary`, arrays, maps, structs -- records `unsupported column type <type> for <column>` (input)
or `unsupported output type <type> for <name>` (a projection or aggregate result), whatever else is
true of it. This is the single most common reason in real plans (see the decimal measurement in
[docs/results.md](results.md)).

`spark.vector.enabled=false` turns the whole rule off. Every converted operator has its own key,
listed below; all default to `true`.

## Supported

| Spark operator | spark-vector operator | Requirements | Config key | Fallback reasons |
|---|---|---|---|---|
| `FilterExec` | `VectorFilterExec` | Columnar child of supported types; the condition compiles (see the expression matrix). Between two spark-vector operators the filter forwards a selection bitmap instead of compacting (`spark.vector.exec.selection.enabled`). | `spark.vector.exec.filter.enabled` | `child <op> is not columnar`; `unsupported column type ...`; any expression-compiler reason, e.g. `unsupported expression Like: ...`, `unsupported literal type string` |
| `ProjectExec` | `VectorProjectExec` | Columnar child; every projected expression compiles and has a supported result type. A projection may be a bare literal (`SELECT 1 FROM ...`). | `spark.vector.exec.project.enabled` | `child <op> is not columnar`; `<expr>: <compiler reason>`; `<expr>: unsupported output type <type> for <name>` (several failures are joined with `; `) |
| `HashAggregateExec` (Partial) | `VectorHashAggregateExec` | Columnar child. Functions: `count`, `sum`, `min`, `max`, `avg` over numeric lanes (`sum`/`avg` buffers must stay within 18 digits); grouping keys of type int/bigint/boolean/string/date/timestamp/decimal(<=18) and never a literal; no `DISTINCT` and no `FILTER` clause. Result expressions must be plain attributes at Partial. | `spark.vector.exec.aggregate.enabled` | `child <op> is not columnar`; `unsupported column type ...`; `distinct aggregates not supported`; `aggregate mode <mode> not supported (Partial and Final only)`; `aggregates with FILTER not supported`; `aggregation modes <modes> not supported`; `aggregate without functions (distinct-style)`; `unsupported aggregate function <Class>: <expr>`; `sum buffer <type> exceeds 18 digits`; `avg buffer <type> exceeds 18 digits`; `sum over <type> producing <type> not supported`; `avg producing <type> not supported`; `count with several arguments not supported`; `min/max over <type> not supported`; `aggregate over <type> not supported`; `aggregate over a literal`; `grouping key type <type> not supported`; `literal grouping key`; `result expression <expr> is not a plain attribute`; `literal result <expr>`; `unsupported result type ...`; `unsupported output type ...` |
| `HashAggregateExec` (Final) | `VectorHashAggregateExec` | Reads an exchange, so only the **types** of its input matter: Spark inserts `RowToColumnarExec` under us over its row shuffle, and Comet's columnar shuffle is read directly. Merge functions reuse the accumulators; result expressions are compiled with `evaluateExpression` substituted. | `spark.vector.exec.aggregate.enabled`, `spark.vector.exec.aggregate.final.enabled` | As above, plus `Final aggregation disabled by configuration`; `merging sum buffers of <type> not supported`; `sum with a <n>-column buffer not supported` |
| `SortExec` | `VectorSortExec` | Columnar child **only** -- over Spark's row shuffle the sort stays Spark's (converting rows to sort them gains nothing), so a global `ORDER BY` is ours only over Comet's shuffle or a local `SORT BY` above our operators. Every sort key compiles, is not a literal and has a supported type. In memory, no spill (#12). | `spark.vector.exec.sort.enabled` | `child <op> is not columnar`; `sort without keys`; `<key>: literal sort key`; `<key>: sort key type <type> not supported`; `<key>: <compiler reason>` |
| `BroadcastHashJoinExec` | `VectorBroadcastHashJoinExec` | Streamed side columnar; build side is Spark's own `BroadcastExchangeExec`/`HashedRelation` (kept so a Spark join over the same broadcast still works; every task re-reads it into columns, #11) and only its types matter. Equi-join keys of type int/bigint/boolean/string/date/timestamp/decimal(<=18), never a literal, never `double`. Join types: inner, left outer / left semi / left anti with build right, right outer with build left, full outer; each with an optional non-equi condition. Null keys never match. | `spark.vector.exec.broadcastHashJoin.enabled` | `child <op> is not columnar`; `unsupported column type ...`; `null-aware anti join not supported`; `join without equi-join keys`; `join key type <type> not supported`; `literal join key`; `join type <type> with build side <side> not supported`; any compiler reason for the condition |
| `ShuffledHashJoinExec` | `VectorShuffledHashJoinExec` | Both inputs are exchanges (or their AQE stages / `AQEShuffleRead`), accepted on types alone like the Final aggregate; the build side of each partition is drained into columns first. Same keys, join types and conditions as the broadcast join. Spark plans it only with `spark.sql.join.preferSortMergeJoin=false` or a `SHUFFLE_HASH` hint (#10). | `spark.vector.exec.shuffledHashJoin.enabled` | As the broadcast join, plus `skew join not supported` |
| `ShuffleExchangeExec` / Comet's JVM columnar shuffle above a spark-vector operator | Comet native shuffle over `VectorToCometExec` | Comet on the classpath, `spark.shuffle.manager=...CometShuffleManager`, `spark.comet.exec.shuffle.enabled=true`; hash, single, round-robin and range partitioning (range also needs Comet's `spark.comet.shuffle.native.partitioning.range.enabled` and samples the child twice, like Spark); every column and range key of a type the bridge carries (decimals cross widened to 128-bit). Without Comet the exchange stays Spark's row shuffle, with `ColumnarToRowExec` above our operator and `RowToColumnarExec` below the consumer. | `spark.vector.comet.shuffle.enabled`, `spark.vector.comet.shuffle.range.enabled` | none recorded -- an exchange that is not bridged is simply left as Spark's and shows as `Spark` in the UI |

Leaf scans are **inputs, not conversions**: a `FileSourceScanExec` whose vectorized Parquet reader
emits `ColumnarBatch`es (#61), Comet's native scan, and a `BatchScanExec` over Iceberg through
`IcebergVectorAdapter` (or any other DSv2 source whose `ColumnVector`s the adapters copy, #62) all
count as columnar children. The UI shows them as "Spark columnar scan", which neither counts as
accelerated nor as a missed conversion. Row/columnar transitions (`ColumnarToRowExec`,
`RowToColumnarExec`) and AQE plumbing (`AQEShuffleReadExec`, `ReusedExchangeExec`) are shown the
same way.

## Planned

Tracked issues, in the order they unblock TPC-H:

| Spark operator | Issue | What is missing |
|---|---|---|
| `TakeOrderedAndProjectExec` (`ORDER BY ... LIMIT`) | #6 | A columnar top-N; today every such query ends in a Spark operator over rows |
| `SortMergeJoinExec` | #10 | Not converted; Spark's default for large equi-joins |
| `HashAggregateExec` in `PartialMerge` / `Complete` mode, `count(distinct)`, `ExpandExec` | #7, #56 | Only `Partial` and `Final` modes are accepted; `Expand` is not converted |
| `GlobalLimitExec`, `LocalLimitExec`, `CollectLimitExec` | #53 | Not converted |
| `UnionExec`, `CoalesceExec` | #54 | Not converted |
| `InMemoryTableScanExec` (cached tables) | #55 | Not accepted as a columnar input |
| `ObjectHashAggregateExec`, `SortAggregateExec` | #57 | Not converted |
| `WindowExec`, `WindowGroupLimitExec` | #58 | Not converted |
| `GenerateExec` (`explode`, `posexplode`) | #59 | Not converted |
| `BroadcastNestedLoopJoinExec` | #60 | Not converted |
| `FileSourceScanExec` -- widen and document what is accepted | #61 | Which readers/types count as columnar input |
| `BatchScanExec` -- DSv2 sources beyond Iceberg | #62 | Verification per source (Delta, Hudi, built-in DSv2); unknown sources already work, copied once per batch |
| `SampleExec`, `LocalTableScanExec` | #63 | Not converted |
| `DataWritingCommandExec` -- Parquet from Arrow batches | #64 | Not converted |
| Arrow-based Python UDF operators | #65 | Not converted |
| Spill for `VectorSortExec` and the hash joins | #12 | Both hold the partition / build side in memory |
| Columnar broadcast exchange | #11 | The build side is re-read into columns by every task |

## Not planned

Recorded here so that an absence is a decision rather than an omission (#66). These are open to
revisiting on demand, not permanent exclusions.

| Family | Reason |
|---|---|
| Structured Streaming operators (`StateStoreSaveExec`, `StateStoreRestoreExec`, `StreamingSymmetricHashJoinExec`, `FlatMapGroupsWithStateExec`, ...) | This project targets batch execution. The state store is row-oriented and its on-disk format is a compatibility surface across Spark versions and checkpoints. A streaming query still benefits where a micro-batch contains supported batch operators. |
| `CartesianProductExec` | Rare, and expensive for reasons a columnar engine does not change. A cross join with a broadcast side and a filter is a different plan -- see `BroadcastNestedLoopJoinExec` (#60). |
| `RangeExec` | A niche leaf that generates rows; `RowToColumnarExec` above it already starts a columnar chain. Cheap enough to reconsider, since `range()` appears in many tests and examples. |
| Pickled (non-Arrow) Python UDFs (`BatchEvalPythonExec`) | The data has to become Python objects row by row; there is no columnar path. Arrow-based UDFs are the ones worth accelerating (#65). |
| Command and DDL operators | Nothing to accelerate. |

## Keeping this page honest

Every operator issue names this file as part of its definition of done: the row, the requirements,
the config key and the fallback strings land in the same commit as the operator. Until the matrix is
generated from `PlanAcceleration` (the UI's node classification, which is the natural source), the
check is the suites: a fallback string that changes without its row changing fails a
`checkFallback` assertion before it reaches a user.
