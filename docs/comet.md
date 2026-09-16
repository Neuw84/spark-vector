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

Only hash, single-partition and round-robin partitioning are rewritten. Range partitioning (the
`ORDER BY` above a Final aggregate) stays with whichever shuffle Comet picked, because a native
Comet shuffle over a non-native child samples the child a second time.

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
Comet dependency.

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
