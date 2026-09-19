# AGENTS.md

Working notes for agents and contributors changing spark-vector. This file records the design
decisions the code embodies, why they were taken, and how a change must be validated before it is
considered done. `README.md` is the user-facing description, `docs/results.md` the measurements,
`docs/comet.md` the Comet integration; this file is the contract behind them.

## 1. What this project is

A Spark SQL plugin that runs `Filter`, `Project`, `HashAggregate` (all four modes), `Sort` and
the two hash joins over
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
| `kernels/` | Java 25 | `VectorBuffers` (Arrow-layout `MemorySegment`s), `VecType`, `Species`, the SIMD kernels (compare, bitmap, compact, arith, decimal, cast, agg incl. overflow-checked sums, hash, grouped accumulators, group key table with lookup), the sort, gather and column-builder kernels, and `reference/` (`ScalarReference`, `SortReference`), the scalar oracles the tests compare against |
| `spark-sql-tests/` | Scala 2.13 | Spark's `SQLQueryTestSuite` with the extension injected; profile `spark-sql-tests` only, run by `benchmarks/scripts/run-spark-sql-tests.sh` |
| `spark/` | Scala 2.13 + Java | plugin, session extension, `VectorColumnarRule`, expression compiler, the four operators, Arrow output, input adapters (Spark on-heap, Arrow, Comet), the Comet bridge, the Vector Acceleration UI tab |
| `benchmarks/` | Java + Scala | JMH kernel microbenchmarks and the TPC-H / TPC-DS runners (`TpchQueries`: all 22 queries, `TpchRunner`; `TpcdsRunner`: the 103 queries from Spark's `tpcds/q*.sql` test resources, sharing `TpchRunner`'s engine through `TpchRunner.Suite`) with their markdown/HTML reports and per-query accelerated-operator counts |

Commands that are known to work (always unset `JAVA_TOOL_OPTIONS` first; the IDE sets one that
breaks Spark's JVM options):

```bash
unset JAVA_TOOL_OPTIONS; export JAVA_HOME=/opt/homebrew/opt/openjdk@25
mvn -B -q clean install                    # kernels + Spark suites, Comet suites skipped (~3 min)
mvn -B -q -Pcomet clean install            # also the Comet-backed suites (needs the Comet jar in ~/.m2)
mvn -B -q -Pcomet,iceberg clean install    # plus the Iceberg suites (Iceberg 1.11 runtime from Maven Central)
mvn -pl kernels test -Dvector.jvm.args="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -Dsparkvector.vectorBits=512"
mvn -pl spark install -Dsuites=io.sparkvector.spark.VectorAggregateSuite   # one suite
benchmarks/scripts/gen-tpch.sh 1           # DuckDB-generated eight tables, decimals as doubles; 10 for SF10 (gitignored)
benchmarks/scripts/gen-tpch.sh 1 benchmarks/data --decimals   # same tables with real DECIMAL(15,2), into sf1-decimal
benchmarks/scripts/run-tpch.sh benchmarks/data/sf10 spark,vector,comet-scan,comet-scan-vector,comet-scan-vector-shuffle,comet --iterations 7 --warmup 5
benchmarks/scripts/run-tpch.sh --report    # rewrite benchmarks/results/results.{md,html} from the jsonl files
benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1   # Iceberg merge-on-read variants of lineitem (#260), namespace sf1
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1 spark,vector --iceberg benchmarks/data/iceberg --variant sf1.pos_10 --queries q1,q6,probe-count,probe-sum,probe-group
benchmarks/scripts/gen-tpcds.sh 1          # DuckDB dsdgen, 24 tables with real DECIMAL(7,2)/DATE columns, into benchmarks/data/tpcds-sf1
benchmarks/scripts/run-tpcds.sh benchmarks/data/tpcds-sf1 spark,vector --queries q10,q35,q45   # results under benchmarks/results/tpcds
benchmarks/scripts/profile-query.sh benchmarks/data/sf10 vector q6   # one query under a JFR recording + jfr-summary.sh (section 4.7)
benchmarks/scripts/submit-cluster.sh vector s3a://bucket/tpcds/sf1000/parquet sf1000-parquet s3a://bucket/results/sf1000-parquet   # cluster run (#246; DRY_RUN=1 prints the spark-submit line)
benchmarks/scripts/run-tpcds.sh --cluster-report s3a://bucket/results/sf1000-parquet   # data-on-EKS-style report over a cluster run's rows
benchmarks/scripts/run-tpcds.sh --report   # rewrite benchmarks/results/tpcds/results.{md,html}
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
- Supported types are exactly `VecType`: BOOL, INT32 (also DateType), INT64 (also TimestampType and
  `Decimal(p <= 18)` as the unscaled value), FLOAT64, UTF8, and DECIMAL128 for `Decimal(p > 18)` --
  two little-endian `long` limbs per value (`kernels/Decimal128.java`), Arrow's `Decimal128` layout,
  a lane that deliberately has no SIMD path (#28: the emulated lane-pair variant was rejected). The
  DECIMAL128 lane is carried (adapted, compacted, gathered, appended, emitted: #257) and computed on
  by the wide-decimal kernels (#258: `CompareKernels` DECIMAL128, `WideDecimalKernels` for `+ - * /`,
  `WideDecimalCastKernels` for casts, `abs` and negation -- two `long` limbs with an exact `BigInteger`
  path for a row whose intermediate leaves 128 bits; JMH numbers in `docs/results.md`). The general
  switches still tell the two apart: `TypeMapping.isSupported` (every kernel computes on it) is false
  for wide decimals while `TypeMapping.hasLane` is true; operators that merely move a column (filter
  compaction, projection pass-through, the sort's keys and payloads) ask `hasLane`, the hash aggregate
  reads the lane as grouping keys and as `sum` / `avg` / `min` / `max` / `count` / `first` / `last` inputs and
  computes arithmetic over wide sums in its result projection, the column movers (take-ordered,
  limits, sample, expand, union, coalesce), the hash joins (keys and payloads) and the window
  (partition and order keys, whole-partition and running frames) carry it (#259), and the expression
  compiler routes a wide operand to the wide kernels explicitly (`isWideDecimal`). What still asks
  `isSupported` computes on the lane rather than carrying it: the conditional (`CASE WHEN` with a wide
  result), the `round` family, `%`, a scalar subquery of a wide type, a sliding window frame over a
  decimal -- the TPC-DS reasons that remain name exactly those (docs/results.md). The TPC-H data is
  still generated with decimals as doubles (`Decimal(12,2) * Decimal(12,2)` is 25 digits).
  Adding a type means: `VecType` + `TypeMapping` + every kernel switch + the adapters +
  `ArrowOutput` + tests at all three vector widths.
- Decimals ride on INT64 with the scale kept in the Spark type (`DecimalArithExpr`,
  `DecimalCastExpr` (also serving `CheckOverflow`, which Spark 4.1 no longer leaves in batch plans), `UnscaledValueExpr`, `MakeDecimalExpr` in `expr/DecimalExprs.scala`;
  `DecimalKernels` for rescaling, range checks and Spark-exact division). Spark's result types for
  `+ - *` leave room for every result so only `/` and casts check ranges. Output decimals are
  `VectorDecimalColumnVector` over a `BigIntVector` (Arrow's own `DecimalVector` is 128-bit and
  read through `BigDecimal` by Spark). Remember that the optimizer's `DecimalAggregates` turns
  `sum(decimal <= 8 digits)` into `MakeDecimal(sum(UnscaledValue(x)))`, an ANSI bigint sum, and
  `avg(decimal <= 11 digits)` into a double average: that is the path narrow decimal aggregates take;
  wider ones go through `WideDecimalSumAgg` / `WideDecimalAvgAgg` and their merge counterparts.
- Spark 4 defaults to ANSI mode. Double arithmetic is bit-identical in both modes, so it is
  compiled; integer `+ - *` and negation are computed wrapping and then checked with an overflow
  lane mask (`OverflowKernels`: the sign trick for add/subtract, the exact product for multiply,
  `MIN_VALUE` for negation), raising Spark's `ARITHMETIC_OVERFLOW` with `Math.*Exact`'s message and
  the `try_*` hint, and the `try_*` forms (`nullOnOverflow` on `ArithExpr` / `NarrowCastExpr`) null those rows instead; `try_sum` (`TrySumLongAgg` / `TrySumLongMergeAgg`, Spark's `(sum, isEmpty)` buffer) poisons the whole group on its first overflow; ANSI `sum(bigint)` is overflow-checked the same way (`AggKernels.sumLongExact`
  carries a sign-trick overflow lane; `GroupedAccumulators.LongSum(checked)` uses `Math.addExact`).
  ANSI errors (overflow, division by zero, decimal overflow) are raised only for rows that are active
  (survive earlier conjuncts / the selection), matching Spark's short-circuit semantics.
- Expressions compile to a small `VectorExpr` tree (`ColumnRef`, `LiteralExpr`, `CompareExpr`
  (numbers via `CompareKernels`, strings via `StringCompareKernels` in UTF8_BINARY order), `InExpr`
  (one equality pass per literal), `StringMatchExpr` (`startswith`/`endswith`/`contains`, i.e. the
  `LIKE` shapes `LikeSimplification` rewrites, via `StringMatchKernels`), the hashes in `HashExprs.scala` (`Murmur3HashExpr` over Spark's `Murmur3_x86_32` steps sharing `XxHash64Expr`'s per-type kinds; `DigestExpr` per-row digests through `StringConcatKernels.fromRows`, the row-writer escape hatch), the search family in `StringSearchExprs.scala` (`LocateExpr`, `ReplaceExpr`, `TranslateExpr`, `SubstringIndexExpr`, `SplitPartExpr`, `FindInSetExpr` over `StringSearchKernels.find`, Spark's `UTF8String` algorithms byte for byte; `split_part` is matched on Spark's `ElementAt(StringSplitSQL(...))` rewrite), the case and trim nodes in `StringCaseExprs.scala` (`CaseMapExpr`: ASCII rows in `StringCaseKernels`, flagged rows through `CollationSupport` with the session's ICU choice; `TrimExpr` over a literal code-point set), the several-input writers in `StringConcatExprs.scala` (`ConcatExpr`, `ConcatWsExpr`, `EltExpr` over `StringConcatKernels.Part` lanes-or-literals; `elt`'s ANSI raise checks only active rows), the string measures in `StringLengthExprs.scala` (`StringMeasureExpr` for length/octet_length/bit_length/ascii, once per dictionary entry via `StringLengthKernels.perEntry`, and `ChrExpr`), the string writers in
  `StringSliceExprs.scala` (`SubstringExpr`, `PadExpr`, `RepeatExpr`, `SpaceExpr`, `OverlayExpr` over
  `StringSliceKernels` -- the first kernel that produces new UTF8 data: two passes, per-row lengths and
  source ranges then one data buffer sized from their prefix sum; Spark's `UTF8String` code-point rules
  and first-byte width table; dictionary input read through the dictionary, plain output; a 1 GiB
  per-batch cap and a 2^20 literal bound decline runaway counts; reuse `substringRange`/`numChars`/
  `byteOffsetOfChar` for the remaining string issues), the datetime arithmetic in `DateArithExprs.scala` (`RelabelExpr`, `EpochScaleExpr`, `DateScalarExpr` for last_day/add_months/next_day/weekofyear, `MonthsBetweenExpr`, `MakeDateExpr` over `DateKernels`' month cores -- `lengthOfMonth`, `lastDay`, `addMonths`, `nextDay`, `weekOfYear`, `monthsBetween`, `makeDate`), the pattern functions in `DateFormatExprs.scala` (`FormatInstantExpr` running Spark's own `TimestampFormatter` per row through `SparkFormatters`, `UnixTimestampExpr`, `TruncTimestampExpr`), the date nodes in `DateExprs.scala`
  (`DateFieldExpr`/`DateTruncExpr` over `DateKernels`' civil-from-days arithmetic, `TimestampToDateExpr`/`TimeFieldExpr`
  under a fixed-offset session zone only -- zones with rules fall back), `MonotonicIdExpr`
  (`monotonically_increasing_id()`: partition prefix from the task context plus a per-task counter
  carried across batches; a forwarded selection numbers only the selected rows), the math nodes in
  `MathExprs.scala` (`AbsExpr`, `SignumExpr`, `DivideLikeExpr` for `%`/`pmod`/`div` with the same
  divisor-zero tail as `ArithExpr` -- `REMAINDER_BY_ZERO` for the remainders, `DIVIDE_BY_ZERO` for `div`
  -- `PickExpr` for `greatest`/`least`, `NanvlExpr`; all over `MathKernels`), the transcendental
  family in `TranscendentalExprs.scala` (`UnaryMathExpr`/`BinaryMathExpr` over `TranscendentalKernels`:
  one scalar call per lane that is exactly Spark's -- `Math` or `StrictMath` per function, see the
  enum -- so results are bit-identical and tests use tolerance 0; the log family is null at or below
  its asymptote by Spark's `!(x <= a)` test, `log(base, x)` null for a non-positive operand; the
  Vector API's 1-2 ulp operators are deliberately not used), the rounding nodes in
  `PredicateExprs.scala` (`NullSafeEqExpr`, `IsNaNExpr`, `BoolCompareExpr`/`BoolCompareScalarExpr` on
  packed words, `InSetExpr` with a sorted-key binary search; over `PredicateKernels`),
  `InjectedExprs.scala` (`SubqueryLiteralExpr` for a scalar subquery -- or any reference-free
  expression over one -- read as a literal at execution: Spark evaluates it before the operator runs
  and the result rides in the node; `XxHash64Expr` with Spark's `XXH64` steps per type; `BloomProbeExpr`
  probing Spark's runtime bloom filter through Spark's `BloomFilter`; note that a merging aggregate
  whose function instance was rewritten between stages -- `ReusedSubquery` in its input -- carries fresh
  buffer exprIds, so `VectorAggregates` binds buffers by Spark's position when the ids do not match),
  `BitExprs.scala` (`BitBinaryExpr` for `& | ^` and the shifts, `BitNotExpr`, `BitCountExpr` over
  `BitKernels`; Java's semantics, `bit_count` on the value widened to a long like Spark's),
  `RoundExprs.scala` (`CeilFloorExpr`, `RintExpr`, `RoundExpr` for `round`/`bround`/two-argument
  `ceil`/`floor` over `RoundKernels`; doubles take Spark's BigDecimal-of-toString path, decimals round
  the unscaled value), `And/Or/Not`, `IsNull/IsNotNull`, `ArithExpr`, `CastExpr` (widening) and the cast slice in `CastExprs.scala` (`NarrowCastExpr`, `ToBooleanExpr`, `FromBooleanExpr`, `StringToBooleanExpr`, `DateToTimestampExpr`, `ToStringExpr` over `CastKernels.narrow/toBool/fromBool/daysToMicros`; `StringToNumberExpr`, `DateTimeToStringExpr`, `StringToDateTimeExpr` running Spark's own parsers and formatters per row through `SparkCasts`), `NegateExpr`, the decimal nodes, and
  `CaseWhenExpr` for `CASE WHEN`/`IF`/`COALESCE`: per-branch win masks carried forward as `active`,
  a null condition counts as false, `SelectKernels.select` blends -- literal branches, string and
  boolean ones included, are materialised as constant columns).
  Anything else is a `Left(reason)`. Do not add an expression without a kernel, a scalar
  reference and a Spark comparison test. A `LiteralExpr` may be a projection's whole expression
  (`SELECT 1 FROM ...`); `ArrowOutput.constant` materialises it.

### 3.2 Memory model: Arrow layout in native `MemorySegment`s

- `VectorBuffers` is the only view kernels have of a column: validity bitmap, data, offsets (UTF8),
  optional dictionary. Buffers are Arrow-layout so they can be handed to Arrow and Comet without
  conversion.
- Batches live in native memory allocated from a per-batch `Arena`. Reading Spark's on-heap
  `double[]` in place as a heap segment was measured at half the speed (the Vector API's heap
  segment path is not intrinsified as well), so Spark's `OnHeapColumnVector` batches are copied
  once into native memory per operator chain. Comet's and Arrow's off-heap buffers are wrapped
  zero-copy with `MemorySegment.ofAddress(...).reinterpret(size)`, and so are the fixed-width
  columns of Spark's own `OffHeapColumnVector` (`spark.sql.columnVector.offheap.enabled=true`,
  no dictionary) -- `SparkColumnVectorBuffers.wrappedOffHeapColumns()` counts them; strings and
  dictionaries still take the copy. `docs/operators.md` "Scan compatibility" is the matrix. A cached
  table is a columnar input only when Spark's `DefaultCachedBatchSerializer` says so, and it decides on
  the cached relation's *whole* schema: boolean/byte/short/int/long/float/double only, so a string or
  date column anywhere in the cache makes `InMemoryTableScanExec` a row scan whatever is projected
  (#55 level 2, an Arrow `CachedBatchSerializer`, would lift that).
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
  therefore sums doubles in a different order than Spark; strict floating point (below) forces one
  copy for double sums.
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

- All four aggregate modes are ours: update vs merge is decided per aggregate expression (`Partial` / `Complete` read the input, `PartialMerge` / `Final` merge buffers), buffers vs results per operator (`VectorAggregatePlanner.emitsResults`; a mix is refused). The merge modes consume Spark's row shuffle through a
  `RowToColumnarExec` or, with Comet, Comet's columnar shuffle directly; merge functions
  (`CountMergeAgg`, `AverageMergeAgg`, sum/min/max over buffers) reuse the accumulators; result
  expressions are compiled with `evaluateExpression` substituted, keyed on the operator's
  `aggregateAttributes` (Spark's distinct rewrite gives the Final distinct expression a fresh
  `resultId`). `FILTER` clauses apply in the update modes through `FilteredAgg` (a narrowed
  selection per function; grouped, a copy of the assignment with the other ids cleared).
  `DISTINCT` is a marker only; keys-only aggregates emit their keys. Beyond the kernel-backed
  `count`/`sum`/`min`/`max`/`avg`, the scalar per-row accumulators in `ExtraAggregates.scala`
  (`Rows` loop + `GroupValues`): `min`/`max` over booleans and strings (`OrderedMinMaxAgg`, so
  `bool_and`/`bool_or`/`every`/`any`/`some`), `first`/`last` with and without `ignoreNulls`
  (`FirstAgg`/`FirstMergeAgg`, `LastAgg`), `bit_and`/`bit_or`/`bit_xor` (`BitAgg`), `max_by`/`min_by`
  (`MaxMinByAgg`, ties take the later row like Spark's predicate); the statistical family in
  `StatAggregates.scala` (`MomentsAgg` with a `Kind` per Spark state: `CentralMomentAgg` incl. m3/m4,
  `Covariance`, `PearsonCorrelation`, the regr_slope/intercept pair; update = Spark's row-by-row Welford
  step in partition order, merge = Spark's `mergeExpressions`, buffers emitted as doubles so the Final
  compiles Spark's own result expression -- `sqrt` was added to the compiler for it; tests compare at a
  relative tolerance); `count(a, b)` counts rows with every argument non-null (`CountAllAgg`,
  regr_count's replacement); `count_if` arrives as `count` over a
  rewritten boolean. A string buffer makes Spark plan `SortAggregateExec`: the rule builds the same
  hash operator from its fields, drops the sort Spark placed below when it is exactly the required one,
  and keeps the output ordering of a result-emitting stage with a `VectorSortExec` above.
  `spark.vector.exec.aggregate.final.enabled` turns the Final conversion off.
- `GroupKeyTable` memoises group ids per combination of dictionary indices when every key is
  dictionary encoded and the product of dictionary sizes is small. Plain UTF8 keys are dictionary
  encoded on the fly against a per-column dictionary kept across batches so the same path applies:
  values of up to 8 bytes are keyed by their packed bytes (single-byte values through a 256-entry
  direct table), longer ones by a 64-bit fingerprint over 8-byte words confirmed with a byte
  compare. A column whose dictionary grows past `sparkvector.agg.plainDictMaxEntries` (default 512)
  makes the table stop encoding plain strings for good and hash and compare per row: the memoised
  path's per-batch reset and miss rate grow with the combinations, and `GroupKeyTableBenchmark`
  puts the crossover at a few hundred distinct values (2x slower by 4000, for 8- and 24-byte keys
  alike). This exists because Comet's native scan delivers plain strings where Spark's reader keeps
  dictionaries, and without it the zero-copy configuration lost to the copying one at SF10.
- Floating-point rounding is a configuration choice, like Comet's `spark.comet.exec.strictFloatingPoint`.
  `spark.vector.exec.strictFloatingPoint` (default `true`) makes every double `sum`/`avg`, Partial and
  Final, grouped or not, bit-identical to Spark's: `GroupedAccumulators.DoubleSum(strict = true)` keeps
  one accumulator per group and `AggKernels.sumDoubleSequential(v, start)` continues the running sum
  row by row (adding a per-batch sum to the running one would round differently; `AggKernelsTest`
  and `GroupedAggregationTest` pin bit equality against a sequential reference, `VectorAggregateSuite`
  against Spark with tolerance 0). `false` restores the lane-parallel and interleaved sums (7% of
  aggregate kernel time, 2.5% of TPC-H Q1 at SF10) that differ from Spark's in the last bits; TPC-H
  Q15 then returns no rows because a double sum is compared for equality against the maximum of the
  same sums computed by Spark's subquery. The benchmark configurations run with `false`
  (`TpchRunner.VectorFast`), Comet's default, so the Q15 checksum mismatch in the reports is
  expected. Tests compare other doubles with a tolerance (`1e-9` relative in `VectorQuerySuite`),
  benchmark checksums use 10 significant digits; assert bit equality on a double sum only under
  strict mode.

### 3.6 Sort

- `VectorSortExec` replaces `SortExec` only over a columnar child (Comet's columnar shuffle, or one
  of our operators for a local `SORT BY`). Over Spark's row shuffle the rule leaves `SortExec` with
  the reason "child ... is not columnar": converting rows to columns to sort them gains nothing.
  `spark.vector.exec.sort.enabled` turns it off.
- The partition is sorted in runs (#285): every `spark.vector.sort.runRows` rows (default 1M; the
  issue names it `sparkvector.sort.runRows`) the iterator seals the builders into a run -- its
  columns, a dictionary string column decoded once, its keys and the permutation -- so the sort's
  JVM scratch is bounded by the run, not the partition. One run emits through its permutation as
  before; several are k-way merged by `kernels/RunMerge` (a loser tree over one cursor per run, one
  compare per level per row; fixed-width keys compared through per-run normalised `long[]` arrays in
  sorted order, strings and decimals through the columns; ties by run index then position, so the
  merge is stable) and gathered run by run through `ArrowOutput.gatherRuns`. The widened leaf: after
  a winner is chosen, a binary search over its normalised keys finds how many of its next rows still
  precede the runner-up's (equal ones too when the tie rule favours it) and they leave as one block
  without a replay each -- tried after a wide block or every 64th row, single fixed-width key, no
  nulls in the winner. Under a limit each run keeps only its first `n` rows for the merge (top-N by
  run). Measured (`SortBenchmark.runs8`, eight runs and the merge against one sort, INT64, 10M rows):
  random 1010 vs 685 ms (the merge costs ~30 ns/row and the loser tree did not beat the heap, so
  the cost is not the compares), presorted 98 vs 112 ms and low cardinality 258 vs 322 ms (the
  blocks). Runs are the memory bound and the shape #85's spill plugs into; on random keys they cost
  time, on ordered or repetitive keys they save it.
- `VectorTakeOrderedAndProjectExec` replaces `TakeOrderedAndProjectExec` (`ORDER BY ... LIMIT n`)
  over a columnar child. Per partition it is the sort iterator with a `limit` (the partition is
  still fully sorted, only the first `n` rows are gathered); those at most `n` rows per partition
  go as `UnsafeRow`s through Spark's own single-partition shuffle (`ShuffleExchangeExec.
  prepareShuffleDependency` + `ShuffledRowRDD`, the same as Spark's operator), Spark's
  `LazilyGeneratedOrdering` takes the final top `n`, Spark's `UnsafeProjection` applies the select
  list (copy each projected row -- the projection reuses one buffer) and the rows become one
  columnar batch of `OnHeapColumnVector`s for the `ColumnarToRowExec` above. `OFFSET` falls back.
  `spark.vector.exec.takeOrdered.enabled` turns it off.
- `VectorLocalLimitExec` / `VectorGlobalLimitExec` / `VectorCollectLimitExec` share `VectorLimitIterator`
  (a `VectorBatchIterator` with the `exhausted` hook: whole batches pass through, the boundary batch is
  compacted to its first surviving live rows through `ArrowOutput.compact` -- a forwarded selection is
  honoured -- and the child is not pulled again once the limit is reached). The collect limit's final
  take reuses `VectorRowStages` (UnsafeRow copies -> single-partition shuffle -> one on-heap batch),
  the same tail as the top-N operator. `spark.vector.exec.limit.enabled`; `OFFSET` falls back.
- `VectorUnionExec` / `VectorCoalesceExec` are `VectorPassThrough` operators: they forward children's
  batches unchanged, so the rule does not mark a filter below them as a selection producer (a forwarded
  selection would reach whatever sits above). The union keeps Spark's partitioning contract:
  `outputPartitioning` is what `UnionExec` would report for the same children, and when that is a known
  partitioning the children's i-th partitions are read together (`SQLPartitioningAwareUnionRDD`) --
  `EnsureRequirements` ran before the replacement and planned no shuffle above the union on the strength
  of that claim (TPC-DS q33/q56/q60 returned one row per channel for a key before this, #128). Spark
  4.1.3's own `UnionExec` has the concatenating columnar path, so ours must not be disabled over
  co-partitioned columnar children.
  The union is columnar whatever its children are, provided one is: Spark's transitions insert
  `RowToColumnarExec` under the row children. Spark's own `UnionExec` is columnar only when every
  child is, and the UI classifies it as a Spark operator either way.
- `VectorWindowExec` (#58, layer 1) computes `row_number` / `rank` / `dense_rank` in `VectorWindowIterator`: per
  live row, `sameKeys` boxes the partition and order key lanes (`Rows.box`, null-safe equality -- `RankLike`
  compares with `<=>`) against the previous row's, resets the counters on a new partition, bumps `rank` /
  `denseRank` on a new peer group; state carries across batches. Requires exactly `WindowExec`'s
  distribution and ordering (`requiredChildDistribution` / `requiredChildOrdering`), and the rule accepts
  ANY child on types (`typeReason`) because the sort below is Spark's without a columnar shuffle -- Spark
  inserts `RowToColumnarExec`. `VectorWindowPlanner.rankKind` refuses everything else with a reason naming
  the function; double keys are refused like join keys. `VectorWindowGroupLimitExec` (layer 4) replaces
  `WindowGroupLimitExec` (Spark's top-k below the window, inserted for `rank <= k` filters with k up to
  `spark.sql.optimizer.windowGroupLimitThreshold`, Partial below the exchange and Final above the sort) with the
  same walk in `VectorWindowGroupLimitIterator`, keeping a row while its ranking value is at most k -- tighter
  than Spark's, which also passes the first row of the next peer group; both are pre-filters, the window above
  computes the real ranks. Partial sits over our operators, so the chain stays columnar up to the shuffle.
  Layer 2, `VectorWindowAggregateIterator`: whole-partition frames (`SpecifiedWindowFrame(_, UnboundedPreceding,
  UnboundedFollowing)`, Complete mode, no FILTER) reuse `VectorAggregates.compile` + `GroupedAggState` with
  the partition ordinal as the group id (`GroupAssignment.of(ids, n, numGroups, arena, selection)`, ids
  cumulative across batches), `AggBufferColumns.column` (shared with the hash aggregate) to lay a run of
  partitions' buffers out as a batch, `compileFinalResults(Nil, aggs, resultAttrs, resultAttrs)` for the
  values and `ArrowOutput.gather` to spread them over the rows. Batches are held as copies
  (`ArrowOutput.copy` / `compact`) until their last row's partition has ended. Decimal aggregates are
  refused: the Complete-mode decimal sum buffer is wide and `compileFinalResults`'s wide-sum shortcut
  assumes a merge emitted the result -- gate on #28. The two modes never mix in one operator (a ranking
  function needs `ORDER BY`, whose default frame is `RANGE ... CURRENT ROW`). Layer 2b, running frames:
  `VectorWindowPlanner.frameKind` (WholePartition / RunningRows / RunningRange); the iterator opens a group per
  row or per peer group (`KeyTracker(orderKeys)`) and marks partition starts in a `BitSet`; `prefixOf(from, to)`
  computes each group's running buffers from the previous group's (`lastPrefix`, `prefixDone`; a group
  straddling two held batches is the one earlier group a later batch re-reads) with `prefixCombiners` --
  `addLong` (ANSI `Math.addExact` -> `VectorErrors.arithmeticOverflow`), `addDouble`, `extreme` via
  `Comparable` -- and `AggBufferColumns.values` lays them out for `compileFinalResults`. Functions without a
  slot-wise prefix (stats, bit_*) are refused; an operator mixing two frame kinds is refused.
  Layer 3, `VectorWindowOffsetIterator`: `VectorWindowPlanner.offsetWindows` maps `lag`/`lead`
  (`FrameLessOffsetWindowFunction`: `offset()` is already signed -- `Lag.offset = -inputOffset` -- and must fold
  to an int; `default()` must fold; both via `eval(EmptyRow)`, decimals boxed as their unscaled long),
  `first_value`/`last_value` (`First`/`Last` AggregateExpressions -- routed here, not to the prefix family)
  and `nth_value` to `OffsetFunction(kind, inputOrdinal, dt, offset, default, frame)`; inputs must be child
  columns (Spark projects complex inputs below the window). Rows are held as copies with (partition
  ordinal, position, peer ordinal) per row and `partitionLength` / `peerEnd` filled as groups end; the
  target row is addressed globally (`firstGlobal + r + (target - pos)`) and found in `released` or `held`.
  Output columns are BORROWED from the held copies: a released batch stays in `released` until no later
  batch can address its partitions, then in `retired` until the consumer has moved past its output
  (`consumed`), and only then is closed -- do not shortcut this, `lag` reads earlier batches after they
  were emitted. `IGNORE NULLS` and an offset function beside an aggregate in one operator are refused.
  Layer 1b rides the same iterator: `RowNumberAt`/`RankAt`/`DenseRankAt`/`PercentRankAt`/`CumeDistAt`/`NTileAt`
  kinds in `rankingValue` from `pos`, `peerStart(peer)`, `peerEnd(peer)`, `partitionFirstPeer(part)` and the
  partition length (`percent_rank` = peerStart / (n - 1), `cume_dist` = (peerEnd + 1) / n, `ntile` = Spark's
  padded-bucket walk from NTile's update expressions); an all-ranking operator still takes the streaming
  `VectorWindowIterator`, the held path only when the partition size or an offset function is involved.
  Layer 2c, sliding frames, same iterator: `VectorWindowPlanner.slidingAggregate` maps sum/count/avg/min/max
  over `SpecifiedWindowFrame(RowFrame, lo, hi)` with literal bounds (and the RANGE unbounded/current-row
  bounds, so a running aggregate beside a sliding one shares the operator) to `SlidingSum..SlidingMax` with
  `frameLo`/`frameHi` (`UnboundedLo` / `UnboundedHi` / `PeerEndHi` sentinels) and `inputType` (a sum over ints
  reads ints, yields longs). `slidingValue`: a frame starting at the partition advances a per-function
  `runningStates` (rows added in order as the end moves -- Spark's UnboundedPrecedingWindowFunctionFrame);
  any other frame re-aggregates its rows in order per row (Spark's SlidingWindowFunctionFrame does the same,
  so double sums are bit-identical and the O(n x frame) cost is Spark's). `aggregateReason` defers ROWS
  frames with literal bounds to `slidingAggregate` for the reason text. `RANGE ... n PRECEDING` needs
  order-key value comparisons and is refused.
- `VectorSampleExec` (no replacement) is a selection producer like the filter, marked by the rule the same
  way: per partition it seeds Spark's own `BernoulliCellSampler` with `seed + partitionIndex` and draws once
  per *live* row in order -- the row path and codegen of `SampleExec` do exactly that, so the rows match
  Spark's for a seed; a forwarded input selection means rows the filter dropped draw nothing, as in Spark.
  `VectorLocalTableScanExec` (off by default, `spark.vector.exec.localTableScan.enabled`) is a leaf that
  writes each partition of a local relation into one on-heap batch through `VectorRowStages.toBatch`; tests
  over `VALUES` must exclude the optimizer's `ConvertToLocalRelation` or no operator survives above it.
- `VectorExpandExec` (also `VectorPassThrough`) emits one output batch per projection per input batch:
  `ColumnRef` slots are `BorrowedColumnVector`s of the input, `NULL` literals `ArrowOutput.nulls`, other
  literals `ArrowOutput.constant`; the planner refuses any other slot shape. Literal slots bypass the
  expression compiler (which refuses `NULL` and boolean operands) and become `LiteralExpr` directly. The
  iterator holds the input batch until its last projection has been emitted and released, and only then
  asks the child for the next one; a normalized (row-id-mapped) input is compacted instead of borrowed.
- Blocking and in memory: the partition's batches are appended to one `ColumnBuilder` per column
  in an operator-owned shared `Arena` (applying any forwarded selection; dictionary strings are
  decoded because every chunk may carry a different dictionary), sorted, and gathered out in
  4096-row batches through `GatherKernels` (shared with the joins; `-1` indices pad outer joins). There is no spill; that is documented
  and the reason the config key exists.
- `SortKernels.sortIndices` is an LSD sort over order-preserving unsigned key passes; each pass is a
  stable LSD *radix* sort of the current positions by a 32- or 64-bit key -- one counting sort per
  8-bit digit, a digit whose histogram is a single bucket skipped, a pass whose keys are already in
  order skipped whole -- with the keys carried alongside the positions through every scatter (#285;
  before it each pass was an `Arrays.sort` of `(key, position)` packed into a `long`). Null rows get
  a constant value key and a separate null pass. int32/bool: one 32-bit pass; int64/double: one
  64-bit pass; strings <= 8 bytes: a length pass and a 64-bit pass over the zero-padded big-endian
  prefix (unsigned byte order); longer strings: a rank from a stable merge sort; decimal128: four
  32-bit passes. Doubles use Spark's total order (`-0.0 == 0.0`, NaN greatest, NaN == NaN), strings
  `UTF8String`'s. What is lane-parallel: the key normalisation over the gathered key array (sign
  flips, the descending inversion, the double total order through masks). What is not: the
  histograms, the scatters and the gathers through the permutation -- at 1M random 64-bit keys the
  scatters' random writes cost what the comparison sort did (`SortBenchmark`, docs/results.md); the
  wins are 32-bit keys, low cardinality, dictionaries, short strings and multi-key sorts, and the
  scratch is bounded by the run once the sort is chunked (#285 slice 2).
- Oracle: `reference/SortReference` (stable comparator sort with Spark's rules). `SortKernelsTest`
  compares permutations for every type, direction, null ordering, multi-key combination, and the
  special doubles; `VectorSortSuite` compares against `SortExec` per partition and positionally on
  the key columns (ties in non-key columns may differ in order and are not compared).

### 3.6b Joins

- `VectorBroadcastHashJoinExec` replaces `BroadcastHashJoinExec` when the streamed side is columnar or
  an exchange (`exchangeInputReason`, like the shuffled hash join: AQE's runtime broadcast conversion
  leaves the streamed side as a bare `AQEShuffleRead`, which refused eleven TPC-DS queries' chains, #98);
  the build side is deliberately left as Spark's `BroadcastExchangeExec`/`HashedRelation`. Each task
  reads the relation's rows once into columns (`HashedRelationAccess` in
  `org.apache.spark.sql.execution.vector`, the relation being `private[execution]`;
  `valuesWithKeyIndex` for unsafe maps, `keys().flatMap(get)` for `LongHashedRelation`, because
  `keys()` of an unsafe map repeats a key once per row) and builds a `GroupKeyTable` over the key
  expressions (`GroupKeyTable.lookup` probes without inserting). No exchange of our own means a
  Spark join over the same broadcast still works; a columnar broadcast exchange is a listed gap.
- `VectorBroadcastNestedLoopJoinExec` replaces `BroadcastNestedLoopJoinExec` (no equi-keys) with the
  same `VectorHashJoinIterator`: `spec.streamedKeys.isEmpty` switches the candidate step from the key
  table chain to "every build row" (`firstCandidate` / `nextCandidate`), and `probe` walks the streamed
  rows in chunks of `PairBudget / build.numRows` so the pairs in flight stay bounded; semi/anti/existence
  decide from `rowMatched` after all chunks, outer joins emit per chunk. The build side is Spark's
  `IdentityBroadcastMode` array of rows. Full outer and outer-with-the-broadcast-side-preserved are
  refused (a matched bitmap over a broadcast shared by every task).
- Every replacement operator must report `outputPartitioning` exactly as the Spark operator it replaces
  -- *through its output aliases*. `VectorHashAggregateExec` and `VectorProjectExec` mix in Spark's
  `PartitioningPreservingUnaryExecNode` (`outputExpressions` = result / project list) so `GROUP BY
  d_year` output as `year` is partitioned by `year`; a partitioning naming an attribute the operator does
  not output is one a union above cannot match, and both unions (ours and Spark's) then concatenate at
  execution -- TPC-DS q66 returned every row twice (#162). `VectorUnionExec` is planned whatever the
  column types for the same reason: left to Spark over columnar children it runs the concatenating
  columnar `UnionExec`. Run the TPC-DS harness (`run-tpcds.sh`, SF1 in scratch, ~5 minutes) after any
  change to an operator's output contract.
- There is a merge join (#286, section 3.6b): `VectorSortMergeJoinExec` under
  `spark.vector.exec.sortMergeJoin.mode=merge`, or under `auto` (#287) where the pre-pass chooses it. The
  modes: `off` | `hash` (the rewrite below) | `merge` | `auto`; the boolean flag reads as `auto`; the
  default is `off` (the flip to `auto` is the maintainer's: the issue conditions it on the golden files
  under `auto`, on memory accounted through #12 and on SF10 not being slower than `hash`). The `auto`
  rule, in `markSortMergeJoins`: a join whose ordering a parent relies on (`orderingNeeded`) takes the
  merge join; so does one whose row order is *visible* -- below a `LocalLimit` / `GlobalLimit` /
  `CollectLimit` / `TakeOrderedAndProject` / `Sort`, or a range-partitioned exchange (a global sort's,
  which is all a stage sees of the sort above it under AQE), until an aggregate or another exchange ends
  it -- because the hash rewrite's tie order shows there (the three golden files); otherwise the hash
  rewrite when `sortMergeEligibility` finds a side whose statistics fit the budget, the merge join when
  it does not (no statistics, both sides large, a skew join). The choice is the `SortMergeChoice` tag,
  its reason the `SortMergeWhy` tag both operators print (`Sort-merge join as hash join: right side fits
  spark.vector.join.maxBuildSize by statistics` / `as merge join: the row order reaches a limit or a
  sort` / `ordering relied on by the parent` / `no size statistics ...`). A hash join below a merge join
  (its sorts stripped) gets a `VectorSortExec` back (`resortedMerge`). Measured at TPC-H SF10 (#279):
  q21's chain of two merge joins over lineitem (four-row runs) ran 44.9 s against Spark's own
  sort-merge join at 16.8 s and three of our hash joins (under Comet's shuffle) at 6.5 s, so `auto` is
  slower than `hash` at scale and the merge join must not be chosen for large inputs until it walks
  runs without a run object. A hash rewrite whose inputs are
  refused (Spark's operators below, q97) does NOT fall through to the merge join: measured, the merge
  join over Spark's row inputs on q97 ran 2960 ms against 385 ms for Spark's own (per-run bookkeeping
  over unique keys, #286); the join stays Spark's until the merge join walks runs without a run object. Spark's contract is kept (clustered distribution, both children
  sorted by the keys ascending, the preserved side's ordering out), so the sorts below stay and no
  pre-pass, build side or statistics are involved: the rule's `merge` case plans it directly
  (`VectorJoinPlanner.planMergeJoin`: keys and condition compile, every column a lane, any SMJ join
  type, skew joins included). `VectorSortMergeJoinIterator` streams its first input and buffers the
  second run by run (`RunKernels.boundaries` over the sorted keys -- the lane-parallel part; a run at a
  batch edge continues into the next batch while `RunMerge.compareKeys` says the key holds; a run
  inside one batch borrows the batch's copy, a run at the edge is copied into its own *confined* arena --
  a shared arena's close is a handshake with every thread, and paid per run it made q17 a hundred
  times slower; the matched bitmap of an outer type is allocated on the first match), walks the
  streamed batch run by run (one copy per batch), and emits equal runs' cross products in Spark's order
  -- streamed row, then the buffered rows; under a condition each streamed row's survivors then its pad
  -- in 8192-pair chunks through `ArrowOutput.gather` (`-1` pads). Single rows (pads, right-only rows,
  a condition's survivors) go through an ordered 8192-row buffer flushed when full, before a direct
  chunk gather and before a source they reference is released -- a full outer join over unique keys
  emitted one Arrow batch per row without it (q51 3.9 s -> 1.1 s). The shape that stays slow is a merge
  where every row of both large sides is its own run (q97: 8x the hash rewrite); the remedy is a
  per-batch cursor without a run object. Readings in docs/results.md. A right outer join runs the iterator with the sides
  swapped (Spark streams the preserved side, so its output is in right order) and the gather lays the
  columns back in left ++ right order. Null keys never match. Tested positionally against Spark
  (`VectorSortMergeJoinSuite`): the hash join suites compare row sets, this one row order.
- `SortMergeJoinExec` is re-expressed as `VectorShuffledHashJoinExec` under the opt-in
  `spark.vector.exec.sortMergeJoin.enabled` (#10): `VectorJoinPlanner.sortMergeBuildSide` picks the smaller
  side by `estimatedBuildSize` (both sides must have statistics -- without AQE they do not -- and the
  smaller must fit the budget), the rule strips the two required sorts (`sortMergeInputs`), and a
  top-down pre-pass (`markSortMergeJoins`) keeps a merge join whose ordering an ancestor relies on. The
  pass decides every merge join (tag `VectorExecRule.SortMergeDecision`: a build side, or the reason it
  stays) from a bottom-up, memoised eligibility (`sortMergeEligibility`) that judges inputs by what they
  will be after the transform -- an exchange, a converting merge join (so same-key chains convert whole,
  #102), a project or filter over one, any other operator the rule replaces when its output types are
  lanes -- and the transform builds from that decision instead of re-deriving it (our operators carry no
  size estimate). The judgement is optimistic and self-healing: when a join stays after all, `resorted`
  puts a `VectorSortExec` over a converted child that no longer offers the ordering, so a wrong guess
  costs a sort, never a row (`VectorJoinSuite` has the shape: an aggregate input that does not convert
  above a chain that did). Tie order
  under `ORDER BY` and unordered `LIMIT` picks differ from Spark's order-preserving merge (three
  `subquery/in-subquery` golden files); `SQL_TESTS_JVM_ARGS` runs the golden suite under the flag.
- All three hash-style joins refuse a build side whose estimate exceeds `spark.vector.join.maxBuildSize`
  (`VectorJoinPlanner.buildSizeReason`; the estimate is the AQE stage's `computeStats` for a
  materialised stage, else the logical link's `stats.sizeInBytes`; `Long.MaxValue` or no link =
  unknown = convert). The runtime half of #86 (fail with a message when the build actually exceeds
  the budget) needs #12's memory accounting.
- `VectorShuffledHashJoinExec` replaces `ShuffledHashJoinExec` and, like the Final aggregate, accepts
  exchanges (or their AQE stages) as inputs on types alone: Spark inserts `RowToColumnarExec` under
  us for its row shuffle. `ClusteredDistribution` on both sides, `PartitioningCollection` out.
- Supported: inner, left/right/full outer, left semi, left anti, existence (`ExistenceJoin(exists)`:
  the semi join's probe, every streamed row emitted plus a BOOL column for `exists`, false where
  nothing matched), each with an optional non-equi
  condition. An inner join evaluates it on the joined batch and compacts the failing rows; semi,
  anti and outer joins gather every candidate pair of a streamed row as the joined row (the
  condition is compiled against `left ++ right`, not the operator's output), evaluate the condition
  over the pairs and decide per row afterwards -- kept/dropped, or its passing pairs / one `-1`
  padded row -- so a row whose candidates all fail is padded exactly like one with no candidate.
  Full outer (shuffled hash join only -- Spark never broadcasts one and our per-task trailing pass
  would duplicate the unmatched build rows; the broadcast planner refuses it with a reason) keeps a
  per-build-row matched flag and emits the unmatched build rows, streamed side
  null (`ArrowOutput.nulls`), after the input is exhausted; build rows with a null key are never in
  the table and so always come out there. A null-aware anti join (`NOT IN` over nullable columns) is the broadcast path plus two singleton relations (empty: keep all; a null build key: keep none) and, otherwise, the null-key streamed rows dropped. Refused with a reason: skew
  joins, double keys (Spark normalises NaN/-0.0 before comparing, the key table compares bits).
  Sort-merge joins are not converted.
- Both joins are `VectorBinaryExec`; `VectorPlan` is the base the rule, selection marking, Comet
  bridging and the UI classify on.

### 3.7 The Arrow compatibility layer (designed to be replaced natively)

Everything that crosses a boundary goes through one of three seams. New sources or sinks must plug
into these rather than adding special cases to operators.

- Input: `ColumnVectorAdapters.adapt(ColumnVector, numRows, arena)` turns any Spark
  `ColumnVector` into `VectorBuffers`. Zero-copy adapters are tried first (our own
  `VectorArrowColumnVector`/`BorrowedColumnVector`, then anything registered through
  `ColumnVectorAdapters.register(Adapter)`, which is how the Comet and Iceberg adapters join
  without a compile-time dependency); anything else is copied by `SparkColumnVectorBuffers`.
  `ColumnVectorAdapters.adaptedColumns()` / `copiedColumns()` count the two outcomes (test-visible in
  local mode); `VectorUnknownSourceSuite` pins the copy fallback with a test-only DSv2 source whose
  vectors no adapter knows (`test/UnknownColumnarSource.scala`). An
  adapter receives the batch's scratch arena for small derived buffers (Iceberg keeps nulls in a
  byte-per-row holder rather than an Arrow validity buffer, and hands strings over as Parquet
  dictionary indices) while the data buffers stay in place. A future Parquet reader of our own
  that writes Arrow memory directly would be one more `Adapter`, or better, would emit
  `VectorArrowColumnVector`s and need none.
- Foreign batch shapes are normalized where batches enter our operators
  (`InputBatches.normalize`, called from `VectorBatchIterator.hasNext` and
  `EvalContexts.withBatch`). Iceberg's JVM reader applies merge-on-read deletes by wrapping every
  column in a `ColumnVectorWithFilter` over a shared row-id mapping and reporting the live count as
  the batch size; normalization unwraps that into a `SelectedColumnarBatch` over the physical rows
  (3.4), so the mapping costs one bitmap per batch instead of an indirection per access. Any
  consumer that sizes scratch by the input must take the physical count from `ctx.numRows`, not
  from `batch.numRows()` (the grouped aggregate got this wrong once and silently dropped rows).
  See `docs/iceberg.md`.
- Output: `ArrowOutput` writes result columns as unshaded Arrow 18.3.0 vectors (`ArrowSegments`,
  `VectorAllocators`) wrapped in Spark's `ArrowColumnVector`, so `ColumnarToRowExec`, Spark's
  Arrow-based Python/R paths and any Arrow consumer work unchanged. The Arrow version must stay
  the one Spark bundles; we do not ship or shade Arrow.
- Between our operators: `SelectedColumnarBatch` (3.4). Operators check for it explicitly; a
  foreign consumer never sees one because the rule only forwards selections into our own operators
  (`markSelectionProducers`).

### 3.8 The Comet compatibility layer (designed to be replaced natively)

- No compile-time dependency on Comet. `CometVectorAdapter`, `CometBatchBridge` and
  `VectorToCometExec`/`CometShuffle` resolve Comet classes reflectively and register only if Comet
  is on the classpath. The `-Pcomet` profile adds the jar for tests; the benchmark script adds it
  when `COMET_JAR` is set.
- Comet's shaded Arrow (`org.apache.comet.shaded.arrow.*`) and Spark's unshaded Arrow are the same
  version and cannot share a class. Rejected approaches, do not retry them: putting `arrow-c-data`
  on our classpath (Comet keeps `org.apache.arrow.c.*` unshaded but with shaded signatures, so the
  classes collide), maven-shade relocation (Comet's JNI looks classes up by literal name), and a
  bulk-copy bridge (replaced by the zero-copy one).
- Mixed chains (#280): Comet above ours is our rule's doing (`mixedChains`, `spark.vector.comet.mixed.enabled`,
  default off) -- the sink leaf `CometSinkPlaceHolder(scanOp, chain, CometUnionExec(chain, output,
  Seq(VectorToCometExec(chain))))` built reflectively in `CometMixedBridge`, then Comet's `CometExecRule`
  applied to the parent subtree. Rejected, do not retry: `sparkToColumnar` (leaf-only, copies) and a
  placeholder directly over our export node (Comet's input walk ignores foreign nodes: `None.get` in
  `buildNativeContext`). Aggregate pairs are not offered yet (the pair must move as a unit).
- The hybrid planning study (#279, `docs/results.md`) is the evidence for #280/#281: Comet's
  operator wins by more than twice the crossing on the many-group hash aggregate (but the wall clock
  there is Spark's row shuffle, #288), the wide-decimal reduction and the broadcast-join probe; ours
  stays on windows (Comet has none), narrow decimals, few-group aggregates and the shuffled hash join;
  the plain filter is 1-3x behind DataFusion's even over Arrow input, which #282-#284 must explain
  before an allowlist is written.
- The crossing cost is a known number (#279, `CrossingBenchmark`; `docs/results.md`, "Hybrid planning
  study"): into Comet ~2.7-3 µs per column, rows aside, for fixed-width and plain-string lanes (a
  pointer hand-over); back zero-copy at 0.1-0.25 µs per column; a dictionary string column decoded on
  the way in at 20-25 ns per row per column; an INT64-lane decimal widened in at 2.5-3 ns and
  narrowed back at 0.4-0.5 ns per row per column. A per-operator swap pays it twice. Run it with the
  Comet jar first on the classpath: `java --add-modules=jdk.incubator.vector
  --enable-native-access=ALL-UNNAMED -cp <comet jar>:benchmarks/target/classes:$(cat
  benchmarks/target/classpath.txt) org.openjdk.jmh.Main CrossingBenchmark` (the shaded
  `benchmarks.jar` lacks the provided Spark and Arrow classes; `classpath.txt` is what the run scripts
  build with `dependency:build-classpath`).
- What works is sharing memory: `ArrowCData` writes Arrow C Data Interface structs (80-byte
  `ArrowArray`, 72-byte `ArrowSchema`) with the FFM API, with `Linker.upcallStub` release
  callbacks and a live-export registry, and Comet's own `ArrowImporter` imports them into a
  `CometVector` over our buffers. `CometShuffleSuite` asserts every export is released.
- The rule rewrites a `ShuffleExchangeExec` (or Comet's row-based columnar shuffle) above a
  `VectorPlan` into Comet's native shuffle over `VectorToCometExec`, for hash, single, round-robin
  and range partitioning. Range partitioning makes Comet sample the child through Spark's
  `RangePartitioner`, exactly as Spark's own exchange does (the child runs twice either way); it is
  gated by Comet's `spark.comet.shuffle.native.partitioning.range.enabled` and our
  `spark.vector.comet.shuffle.range.enabled`. That sampling pass reads the bridged batches through
  `rowIterator()` and never closes the imported vectors, so the bridge remembers its exports and
  `releaseOutstanding()`s them on task completion (registered after the child iterators, run
  before them). Requires `spark.shuffle.manager=...CometShuffleManager` and
  `spark.comet.exec.shuffle.enabled=true`; `spark.vector.comet.shuffle.enabled` turns the rewrite
  off. Decimals cross the bridge widened to the 128-bit `d:p,s` layout.
- Not combined with Comet: Comet's Final aggregate (needs Comet's own partial buffers) and native
  blocks (our operators are not `CometNativeExec`s). Comet's `LargeVarCharVector`
  falls back to the copying adapter. Dictionary strings are decoded when crossing into Comet.
- The seams a native replacement would fill: `Adapter` for the scan (replace `CometVectorAdapter`
  with our own reader's vectors), a columnar shuffle exchange of our own where `CometShuffle`
  builds Comet's (this would also remove the `ColumnarToRow`/`RowToColumnar` pair around Spark's
  row shuffle in the non-Comet configuration, and let the global sort be ours without Comet). Keep
  those boundaries where they are.

### 3.9 The Vector Acceleration UI tab

- Attached from the driver plugin; lives under `org.apache.spark.sql.vector.ui` because
  `SparkUITab`, `WebUIPage` and `UIUtils` are `private[spark]`. It never influences execution: every
  listener callback and the attachment itself are wrapped so a UI failure cannot fail a query or
  application start.
- Engine classification is by operator identity (`VectorExec`, a class in Comet's packages, a
  columnar leaf, a transition), not by tags we set; fallback reasons come from `VectorFallback`.
  A plan is "fully accelerated" when no operator runs on plain Spark; scans, row/columnar
  transitions and AQE's shuffle reader/reuse markers are plumbing and count for neither side.
  Only `ColumnarToRow`/`RowToColumnar` are transitions: `AQEShuffleRead` hands over whatever the
  exchange wrote and must not be coloured as a conversion. The one conversion the colours cannot
  show is inside Comet's JVM shuffle (`CometColumnarExchange`), which reads its child with
  `execute()` (rows); over one of our operators the UI says so in the node's tooltip. Prefer
  Comet's native shuffle for every partitioning the bridge supports, range partitioning included.
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
   `VectorAggregateSuite`, `VectorSortSuite`, `VectorDecimalSuite` (exact comparison, no double
   tolerance), `VectorJoinSuite`, plus adapter/Arrow suites and `SparkOnJdkSmokeSuite` (Spark
   itself works on this JDK with these flags).
   Spark's own SQL golden-file suite runs the same idea at scale, on demand only:
   `benchmarks/scripts/run-spark-sql-tests.sh [regex]` (profile `spark-sql-tests`, about 15
   minutes for everything; 642 cases pass, 111 Python UDF variants are ignored without pyspark).
   It prints a per-case coverage table and, on a full run, fails when a case runs fewer of our
   operators than `spark-sql-tests/src/test/resources/vector-sql-coverage.tsv` records
   (`SQL_TESTS_UPDATE_BASELINE=true` rewrites the floor; #17). Its JVM sets
   `sparkvector.agg.interleave=1` so double sums add in Spark's order and match the golden digits.
   Run it after any planner or expression change: after a day of merges it found four bugs the
   hand-written suites had not (lazy `nanvl`, `count` over literal arguments, collated strings as
   lanes, a build side pruned to zero columns).
   `VectorSQLQueryTestSuite.defaultExclude` skips `explain*.sql` (golden plans are Spark's), the
   DataSketches files (their memory library rejects JDK > 21) and `udtf/udtf.sql` (needs pyspark);
   `SQL_TESTS_EXCLUDE='^$'` runs them anyway. The test JVM needs `-Dspark.testing=true` (Spark's
   test-mode defaults, such as the TIME type, and the STANDARD error format the golden files
   assume) and `-XX:-OmitStackTraceInFastThrow` (a hot ANSI overflow otherwise becomes a
   message-less exception that Spark's error formatting cannot render). Run at least the files
   touching a change (`group-by`, `join`, `decimal`, `order-by`) before calling an operator or
   expression done, and the whole suite before a release; its first run found a gap
   (`SELECT 1 FROM ...`) the comparison suites had not.
3. Comet integration. `CometScanSuite` and `CometShuffleSuite` are tagged `CometTest` and run only
   with `-Pcomet`. They cover zero-copy scan adaptation, dictionary strings from Comet, the shuffle
   rewrite for each partitioning, and that every C Data export is released. They need the Comet jar
   built from source (`docs/comet.md` explains why, on macOS).
   Iceberg integration. `IcebergScanSuite` (tag `IcebergTest`, `-Piceberg`) and `CometIcebergSuite`
   (both tags, `-Pcomet,iceberg`) run the same merge-on-read battery from `IcebergMorSuiteBase`:
   positional deletes, deletion vectors (v3), equality deletes written with the Iceberg Java API,
   a merge-on-read lineitem for Q1/Q6, and a `MERGE INTO` over a heavily mutated table run with the
   plugin on and off whose results must match row for row. The JVM suite also asserts through
   adapter counters that batches were normalized and columns (including dictionary strings) adapted
   in place rather than copied.
4. UI. `PlanAccelerationSuite` pins the classification rules without a session (stand-ins in
   `org.apache.spark.sql.comet` stand for Comet operators); `VectorAccelerationUiSuite` binds a
   real Spark UI, runs converted queries and fetches both pages over HTTP.
5. End-to-end results. `TpchRunner` (and `TpcdsRunner`, which shares its engine) computes a
   checksum of every configuration's result rows to 10 significant digits; the report states whether
   all configurations agree. A benchmark run where the checksums differ is a correctness bug, not a
   performance result. The per-query acceleration column (`k/n` operators ours) is what the TPC-DS
   per-query issues are closed against.
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

   Two scripts make that procedure one command each (#251):

   ```bash
   benchmarks/scripts/profile-query.sh benchmarks/data/sf10 vector q6 --iterations 3 --warmup 2   # run + record + summarise
   benchmarks/scripts/profile-query.sh benchmarks/data/tpcds-sf1 vector q72 --tpcds --conf spark.vector.exec.sortMergeJoin.enabled=true
   benchmarks/scripts/jfr-summary.sh /tmp/profiling/sf10/q6-vector.jfr --top 20                     # any recording, as text for an issue
   benchmarks/scripts/profile-query.sh - vector q6 --flags-only   # the recording flags for a cluster run's executor/driver options
   ```

   `profile-query.sh` runs one query in one configuration under a `settings=profile` recording
   (rows into a scratch `RESULTS_DIR`, never the report), prints the `[tpch]`/`[tpcds]` kernel times,
   and writes `<query>-<config>.jfr` plus `.summary.txt` next to it. `jfr-summary.sh` prints, in the
   reading order above: `jfr view hot-methods`; the callers of the JDK-internal `MemorySegment` and
   `Buffer.checkIndex` frames (a stack walk over `jdk.ExecutionSample`, so the hot JDK frame is
   attributed to the kernel or reader that called it); the plugin's own frames by self time (the
   first `io.sparkvector` frame of each stack -- which kernel or expression owns the samples, and how
   much of the JVM's time is not ours at all); then allocation sites, GC pauses, latencies by type
   and native methods (Comet's JVM side). The cluster half of the regression protocol -- submitting
   the single-query application with `--flags-only`'s options and pulling the recordings back --
   attaches to the cluster runner of #246 when it lands; the summary script reads those recordings
   unchanged.

Current counts: 176 kernel tests, 283 Spark tests with the Comet and Iceberg profiles (259 with
Iceberg alone; the Comet suites contribute 24, `CometMixedChainSuite` 4 of them). If a change lowers either number, explain why in the commit.

## 5. Benchmarking protocol

- One JVM per configuration, `local[8]`, 8 GB heap, `spark.sql.shuffle.partitions=8`; SF1 with 10
  warm-up and 10 measured runs, SF10 with 5 and 7. Configurations: `spark`, `vector`, `comet-scan`,
  `comet-scan-vector`, `comet-scan-vector-shuffle`, `comet`.
- Results are appended to `benchmarks/results/<config>.jsonl` (committed) and the report takes the
  latest measurement per (dataset, config, query). Outliers stay in the files with older
  timestamps; note discarded runs in `docs/results.md`.
- Run on a quiet machine. A video call or a full build minutes earlier moved medians by up to 2x on
  the development laptop; a `spark` Q1 median far from the documented one (1123 ms at SF10) means
  the environment, not the code. Check `uptime` and the top CPU consumers before trusting a run,
  and make sure no other agent or test job (`run-spark-sql-tests.sh`, a Maven build) is using the
  machine for the whole run: one that started mid-run moved `spark` Q1 from 1106 to 1451 ms.
- Update `docs/results.md` tables from the report and keep the earlier phase tables for history.
  Speedups are always relative to plain Spark on the same dataset and session.
- Per-operator attribution (#279): every results row carries `operatorTimes` -- one entry per plan
  node that ours or Comet's native engine ran: class, engine, output rows, milliseconds summed over
  tasks. Ours is the operator's `time` metric (kernel time, nanoseconds in the metric); Comet's is
  DataFusion's `elapsed_compute` (the operator's own compute, input waits excluded -- the metric's
  description says milliseconds, the value is nanoseconds: a 21 ms filter reads 21200470) and
  `output_rows`. Spark's operators have no per-operator time and are absent, so a kind missing under
  one engine is that engine's fallback, listed beside it: Comet's reasons come from its
  `ExtendedExplainInfo.getFallbackReasons` (reflection, `Comet: ...` entries in `fallbacks`) when the
  jar is on the classpath. `--report` prints the operator matrix (ms per kind per attributed
  configuration, then per query `ours vs theirs (delta)` for the kinds both ran). The two times are
  not the same clock: ours counts kernel work inside the operator, Comet's counts native compute; both
  exclude the wait on children, neither is wall clock, and neither includes the crossing between
  engines, which the crossing-cost benchmark measures on its own.
- Iceberg merge-on-read (#260): `gen-iceberg-mor.sh <tpch-dir> [namespace]` builds the `lineitem`
  variants (`plain`, `pos_<pct>[_clustered]`, `pos_upd_<pct>`, `eq_<pct>`, `dv_*`) into a local
  Hadoop catalog with Spark alone, and writes `README-<namespace>.md` with live rows, delete files and
  snapshot ids -- the oracle. Run every configuration over a variant with
  `run-tpch.sh <tpch-dir> <configs> --iceberg <warehouse> --variant <namespace>.<variant>` and the
  probes `probe-count,probe-sum,probe-group` beside `q1,q6`; checksums must agree across
  configurations for every variant, and `[tpch] scan=... merge-on-read live/physical=...` says which
  reader ran and how many rows the deletes removed. The report groups the variants under "Iceberg
  merge-on-read" with the speedup versus `spark` on the variant and versus `plain` for the same
  configuration. `docs/iceberg.md` has the variant table.

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
  (numbers and what they mean), `docs/comet.md` (integration and limitations),
  `docs/operators.md` (the per-operator support matrix: requirements, config keys, fallback
  strings -- every operator change updates its row in the same commit), `docs/expressions.md` (the
  per-expression matrix: lane types and fallback reasons -- every expression change updates its
  row in the same commit), this file (design and validation).

## 7. Known gaps

- Iceberg merge-on-read reads (#261): the delete cost is a fixed per-batch price paid inside Iceberg's
  reader (`buildRowIdMapping`, the per-task position index) by both engines, so our margin over Spark
  shrinks on deleted tables rather than growing; equality deletes are evaluated row-at-a-time by the
  reader and are the one shape where `vector` loses the pure-merge probe. The lever would be taking
  the position-filtered batch and applying the equality-delete set as our own anti-join, which Iceberg
  1.11's reader does not expose. Q1 under `vector` has shown a transient 20x (two or three consecutive
  8 s iterations inside one JVM, then recovery; the partial aggregate's kernel time balloons) in two of
  sixteen JVMs of the v2 run and two more in the v3 run (four of five on the update/merge shape: 18 data files, two small), never under JFR or `-Xlog:deoptimization` -- a JIT signature; catch it by running the sweep command itself with `-XX:+PrintCompilation` and keeping the output of a JVM that shows the 8 s iterations. On v3 the delete cost is `buildRowIdMapping` inside Iceberg's reader (19 % of the pure-merge probe for us, 15 % for Spark); a direct bitmap from the deletion vector would need the reader to hand out the index instead of wrapping the vectors -- an upstream option, not an operator change.

- Iceberg `MERGE INTO`: columnar end to end since #273 -- `MergeRows` has its columnar operator
  (`VectorMergeRowsExec`, #21; differential tests against Spark's operator in `VectorMergeRowsSuite`)
  and the merge's join is ours: `VectorHashJoinExec` passes a lane-less payload column (the struct
  `_partition` beside `_file` / `_pos` / `_spec_id`) through on its **streamed** side as a
  `RemappedColumnVector` over the streamed batch with the probe row ids, null-padded for unmatched
  build rows; the target is that streamed side. A lane-less column on the **build** side is still
  refused (`unsupported column type struct<...>`): build rows are laid out in lanes and a row store for
  them is not written. The write stays Spark's. The merge-on-read suite asserts the operators ran
  (`PlanUtils.allNodes` descends into `CommandResultExec` for that).
- Pass-through of columns without a lane (#19): `VectorColumnarRule.forwardingInputReason` (filter,
  project) requires only a columnar child; `VectorProjectExec.isPassThrough` (a bare
  `AttributeReference` or an `Alias` of one) is never compiled and becomes a `ColumnRef` of the
  column's own type. Dense batches borrow Spark's vector (`BorrowedColumnVector` delegates every getter,
  `getStruct` reads through `getChild`); a selection is applied to such a column with
  `RemappedColumnVector` (row ids of the selection, shared by all foreign columns of the batch, children
  remapped too) instead of `ArrowOutput.compact`, in both iterators; the lanes stay lazy, so a foreign
  column never reaches `ColumnVectorAdapters`. Everything above still judges by `typeReason`, and the
  Comet bridge's `bridgeable` already refuses a child output the C Data interface cannot carry.
- Struct field access (#50): `ExpressionCompiler.structPath` resolves `GetStructField` chains down to
  the input ordinal and the field ordinals; `StructFieldExpr` (spark/expr/NestedExprs.scala) walks
  `ctx.column(ordinal).getChild(...)` to the leaf and adapts it with `ColumnVectorAdapters.adapt` (the
  leaf is a Spark vector -- on-heap copy, off-heap view, foreign generic copy; a `RemappedColumnVector`
  child is remapped too), then ANDs each ancestor's non-null bitmap into the validity
  (`StructFieldExpr.withValidity` rebuilds a `SegmentVectorBuffers` of the lane's shape). A struct-typed
  field as a value, `arr[i]`, `map[key]` and `arr.field` are refused with reasons naming #50; nested
  results and the array/map/lambda families are recorded as not planned in `docs/expressions.md`.
  A struct field whose type has no lane, projected as a value, is a pass-through too
  (`VectorProjectExec.isPassThrough` accepts a `GetStructField` chain of such a type ->
  `NestedColumnRef` -> `NestedFieldColumnVector.of(ctx.column, path, numRows)`, a view with the
  ancestors' nulls folded in; remapped like any foreign column under a selection). `SizeExpr` and
  `NestedValidityExpr` read the array/map length and the nulls of such a column from Spark's vector.
- Generate (#59): `VectorGenerateExec` (spark/src/main/scala/org/apache/spark/sql/vector/VectorGenerateExec.scala)
  replaces `GenerateExec` for `Explode`/`PosExplode` over a nested column path
  (`ExpressionCompiler.nestedColumnPath`) with a lane element type. The iterator builds `rowIdx`/`elemIdx`
  from the array lengths (a null/empty array -> one `-1` row under `outer`), gathers lanes with
  `ArrowOutput.gather(ctx.input(ord), rowIdx)`, views foreign columns through `RemappedColumnVector`,
  and writes the elements (one `getArray` per source row, boxed per element into
  `AggBufferColumns.values`) and the iota position. Spark's `InferFiltersFromGenerate` puts
  `size(arr) > 0 AND isnotnull(arr)` below a non-outer explode and `NestedColumnAliasing` projects
  `st.inner` as `_extract_inner` -- both compile now, so the whole chain stays columnar. A
  `VectorPlan` parent turns the filter below into a selection producer, so the iterator honours
  `ctx.selection`.
- Speculative narrow decimals (#26, slice 1): `ExpressionCompiler.speculativeDecimalMultiply` recognises
  a `Multiply` of two decimal(<=18) operands whose declared result is wider than 18 digits (scale exactly
  `s1 + s2`, precision <= 38, not TRY) and builds `SpeculativeDecimalMulExpr` (DecimalExprs.scala); only
  `VectorAggregates`' Sum-over-decimal case asks for it. `evalChecked` computes the INT64 products with a
  `Math.multiplyHigh` check per row, clears the lane's validity for overflowing rows and returns their
  exact `BigInteger` products; `WideDecimalSumAgg.Escalation` adds those per group beside the
  `WideLongSum` accumulator (ungrouped: only selected rows; grouped: `GroupAssignment.ids` >= 0), and
  `bufferValue` sums both parts. `eval` on the speculative expression throws -- it must never be a lane
  for another consumer. `SpeculativeDecimals.escalatedRows()` is the (global) escalation counter the
  tests read. Slice 2: an operand may be a `SpeculativeDecimalMulExpr` itself (`operand()` in the
  compiler recurses into a wide `Multiply`); `evalChecked` reads each operand through an `Operand`
  (literal / lane / speculative child with a cursor over its ascending escalated rows), multiplies
  exactly where a child escalated, and applies Spark's range check for a capped declared precision
  (`|v| >= 10^p` -> null in legacy, `VectorErrors.decimalPrecisionOverflow` in ANSI). Spark 4.1's
  `Multiply` carries a `NumericEvalContext`, not an `EvalMode` -- read `m.evalContext.evalMode`, a
  pattern-bound third field compares unequal to every `EvalMode` value. Slice 3: the wide decimal
  `avg` (`WideDecimalAvgAgg` / `WideDecimalAvgMergeAgg`, Spark's `(sum: Decimal(p+10), count)` buffer;
  `Escalation` is shared with the sum, so `avg(a * b)` is speculative too). Its `Final` result is not
  re-derived: `DecimalAvgResult` binds `Average.evaluateExpression` to the two buffer slots and evaluates
  it per group (`DecimalDivideWithOverflowCheck`: 39-digit half-up quotient, `toPrecision` to
  `Decimal(p+4, s+4)`, `ARITHMETIC_OVERFLOW` on a null sum) -- exact by construction, one eval per group
  like Spark's own Final. The function emits the result in the sum slot under the *result* type
  (`VectorAggFunction.emittedTypes`; the exec builds its result-mode batch from those, not from the
  declared buffer types) and `VectorAggregatePlanner.wideResult` forwards it like the sum's. Spark 4.1's
  `Average` carries `evalMode` directly (unlike `Sum`/`Multiply`). Overflow convention: a partial or a
  grouped total past `Decimal(p+10)` is null (Spark's `UnsafeRow` write nulls it, then
  `DecimalAddNoOverflowCheck` keeps it null), but an UNGROUPED `Final` divides the exact total -- Spark's
  generated ungrouped Final keeps the sum in a local nothing re-checks -- so `avg(big * big)` over the test
  table is a value ungrouped and null grouped, in both engines. `Complete` mode (streaming planners only)
  puts the result in the sum slot for both the wide sum and avg. Slice 4: `SpeculativeDecimalAddExpr`
  (`+`/`-` whose declared scale is `max(s1, s2)`; both operands rescaled in the lane with a `multiplyHigh`
  check, added with the sign trick, escalated exactly otherwise); the multiply and add share
  `SpeculativeOperand` and `Escalations`, both implement `SpeculativeDecimalExpr`, and the compiler entry is
  `speculativeDecimalArithmetic` (any nesting of `* + -` over lanes, literals and speculative children). An
  uncapped `+`/`-` can never exceed its declared precision; Spark's cap lowers the scale (and rounds) only
  past 32 integer digits, and that shape is refused. Test-fixture arithmetic to remember: `big` is ~9e14 so
  `big * big` is 8.1e29 -- a `sum` over ~12 such rows already overflows a `decimal(38,6)` buffer, so a
  fallback probe on that shape must bound its rows. Not yet: wide products or sums as values (needs a
  128-bit output column on escalated batches) -- that is #28's lane.
- Comet 1.0 reads Iceberg v3 tables (deletion vectors) through the JVM reader; the Iceberg adapter
  covers that path, but it is a copy of the validity bits and a per-batch dictionary decode, not a
  native read.
- Do not reach for `spark.comet.parquet.rowFilterPushdown.enabled` to close the Q6 gap: measured
  2x slower for Comet itself and for us on uniformly spread survivors (`docs/results.md`). Comet's
  default format-level pruning already reaches our configurations.
- Selective predicates with scattered survivors (TPC-H Q6) lose to Spark's codegen over Spark's
  scan (0.86x at SF10): the on-heap copy plus full-column evaluation of a 1.9% predicate. The
  off-heap wrap (#61) does NOT help Q6 -- measured 0.92x vs 0.94x on-heap (#14): Parquet writers
  dictionary-encode `l_shipdate`, `l_discount` and `l_quantity`, the wrap skips dictionary columns,
  and the adapter's dictionary decode is 20% of the JVM's samples against ~5% for all filter kernels;
  the lever is to evaluate comparisons over the dictionary table and gather bits by id. Over
  Comet's scan the copy is gone and the plugin beats Spark (1.12x) but not Comet's scan under
  Spark's codegen (1.17x).
- Without Comet, the shuffle is Spark's row shuffle with a `ColumnarToRowExec` above the partial
  aggregate and a `RowToColumnarExec` below the Final. A columnar shuffle of our own is the
  natural next seam to fill (3.7).
- The sort does not spill, and without Comet the global sort sits above Spark's row shuffle and
  stays Spark's. Joins do not spill either (the build side is held in memory per task).
- The build side of a broadcast join is Spark's `HashedRelation`, re-read into columns by every
  task; a columnar broadcast exchange would read it once. Sort-merge joins are not converted.
- Decimals wider than 18 digits fall back, including the TPC-H price arithmetic. The one exception
  is the `sum` buffer of a decimal of more than 8 digits: the Partial side accumulates it in 128 bits
  (`GroupedAccumulators.WideLongSum`, scalar two-word adds -- an accumulator is one value per group,
  not per row) and emits Spark's `(sum: Decimal(p+10, s), isEmpty)` buffer through a wide Arrow
  `DecimalVector` (`ArrowOutput.newVector`/`decimalColumn`); the merge modes (`WideDecimalSumMergeAgg`)
  read that buffer row by row from the batch's own column (`EvalContext.column`, `getDecimal` -- a merge
  sees one row per partition per group) into an exact per-group total with Spark's `isEmpty` rules, and
  `Final` emits `If(isEmpty, null, CheckOverflowInSum(sum))` ready-made, which `compileFinalResults`
  recognises and forwards. The planner accepts a wide decimal input only as that buffer of a merging
  aggregate (`VectorAggregatePlanner.wideSumBuffers`) and a wide output only as that buffer or result.
- AVX2/AVX-512 paths are tested emulated, never measured on real hardware.
- `TpchRunner --keep-alive` leaves the session and the Spark UI up for inspection; the demo JVM's
  Jetty resets some parallel static-resource fetches under load, so reload the page if the tab's
  toggles do not react (jQuery failed to load).
