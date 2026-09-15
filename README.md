# spark-vector

A Spark SQL plugin that executes Filter, Project and HashAggregate on Arrow-layout batches with
the Java Vector API (`jdk.incubator.vector`). It follows the architecture of
[Apache DataFusion Comet](https://github.com/apache/datafusion-comet), but stays entirely on the
JVM: no native library, no JNI, no serialization boundary. Unsupported operators, expressions or
types fall back to Spark with a recorded reason.

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
| `spark/` | Scala 2.13 + Java | `VectorPlugin`, session extension, `VectorColumnarRule`, expression compiler, `VectorFilterExec` / `VectorProjectExec` / `VectorHashAggregateExec`, Arrow output, input adapters (Spark vectors, Arrow, Comet) |
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
| `spark.vector.exec.aggregate.enabled` | convert Partial `HashAggregateExec` |
| `spark.vector.explainFallback.enabled` | log why each operator was left to Spark (default `false`) |

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
(Partial only; Final stays in Spark)
```

`VectorColumnarRule` runs in Spark's `preColumnarTransitions`, bottom-up. An operator is converted
when its child is already columnar with supported types (a vectorized Parquet scan, a Comet scan, or
another spark-vector operator) and every expression compiles to the kernel IR. Otherwise the reason
is stored as a tree-node tag; `VectorFallback.reasons(plan)` lists them.

Semantics follow Spark, including the corners: NaN-safe double ordering (`NaN = NaN`, NaN sorts
last), three-valued `AND`/`OR`, `WHERE` treating null as false, and ANSI mode (Spark 4's default):
double division by zero raises `DIVIDE_BY_ZERO`, while integer arithmetic under ANSI (which needs
overflow checks) falls back.

### Aggregation

The partial aggregate emits exactly Spark's buffer schema (`sum`, `count`, `min`, `max`,
`(sum, count)` for `avg`), so Spark's exchange and Final aggregate run unchanged. Without grouping
keys, one buffer row per partition. With keys, a hash table assigns dense group ids across the
task's batches. When every key is a dictionary-encoded string (the usual case for low-cardinality
Parquet columns) the ids are memoised per combination of dictionary indices, so a batch probes the
table at most once per distinct key tuple. Rows are then scattered into per-group accumulators;
the alternative, one masked SIMD reduction per group, only wins for one group on 128-bit vectors
(`sparkvector.agg.maskPathMaxGroups` sets the cut-over, default 1 for ≤4 lanes and 8 above).

### Supported today

| Area | Supported | Falls back |
|---|---|---|
| Types | Int, Long, Double, Date, Timestamp, Boolean, String (strings pass through and serve as group keys) | Float, Short/Byte, Decimal, Binary, nested |
| Predicates | `=`, `!=`, `<`, `<=`, `>`, `>=` on numeric/date columns vs literal or column; `AND`/`OR`/`NOT`; `IS [NOT] NULL`; boolean columns | string comparisons, `LIKE`, `IN`, functions |
| Arithmetic | `+ - *` on Int/Long/Double, `/` on Double, unary minus, widening casts | ANSI integer arithmetic, `%`, decimals, other casts |
| Aggregates | `sum`, `count`, `min`, `max`, `avg`; keys of Int/Long/Boolean/String/Date | `DISTINCT`, `FILTER`, double keys, Final mode, other functions |

## Benchmarks

Kernel microbenchmarks (JMH):

```bash
mvn -DskipTests package
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
     -jar benchmarks/target/benchmarks.jar "Compare|Compact|Agg"
```

TPC-H Q1 and Q6 (decimals replaced by doubles, generated with DuckDB):

```bash
brew install duckdb
benchmarks/scripts/gen-tpch.sh 1                      # benchmarks/data/sf1/lineitem
mvn -DskipTests install
export JAVA_HOME=/opt/homebrew/opt/openjdk@25
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1    # spark + vector
COMET_JAR=/path/to/comet-spark-spark4.1_2.13-1.0.0.jar \
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1    # + comet-scan, comet-scan-vector, comet
```

Each configuration runs in its own JVM; the report (`benchmarks/results/results.md`) gives median
times, speedups against plain Spark, the operators found in each final plan and a checksum proving
all configurations returned the same rows. See [docs/results.md](docs/results.md) for numbers
measured on an Apple M3 Pro.

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

- Columnar shuffle. Every stage boundary still goes through Spark's row shuffle via
  `ColumnarToRowExec`. Two paths are open: Comet's shuffle, reached by bridging our unshaded Arrow
  vectors to Comet's shaded ones zero-copy through the Arrow C Data Interface (both are Arrow
  18.3.0), or our own `ShuffleExchangeLike` with a serializer that dumps the Arrow buffers.
- A Parquet-to-Arrow reader of our own; Comet's reader covers the zero-copy case.
- Decimal arithmetic (`Decimal(p<=18)` as long lanes), lazy selection vectors instead of compaction,
  Sort and joins, running Spark's SQL test suite Comet-style.
