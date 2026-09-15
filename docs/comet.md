# Using Comet as the scan

spark-vector accelerates Filter, Project and HashAggregate. It does not read Parquet itself: it
consumes whatever columnar batches sit below it. Two sources work out of the box:

| Source | How batches are read | Notes |
|---|---|---|
| Spark's vectorized Parquet reader (`FileSourceScanExec`, `Batched: true`) | copied once into Arrow-layout buffers per batch | zero configuration; the copy shows up in `time in spark-vector kernels` |
| Apache DataFusion Comet scan (`CometScanExec` / `CometBatchScanExec`) | zero copy: the Arrow buffers Comet's native reader produced are wrapped as `MemorySegment`s | dictionary-encoded strings stay encoded, so `GROUP BY` string keys hash the dictionary once per batch |

Comet is used in *scan-only* mode: its native Parquet-to-Arrow reader replaces Spark's, its native
operators stay off, and spark-vector's JVM SIMD operators run above the scan.

## Configuration

```
--conf spark.plugins=org.apache.spark.CometPlugin,io.sparkvector.spark.VectorPlugin
--conf spark.comet.enabled=true
--conf spark.comet.scan.enabled=true
--conf spark.comet.exec.enabled=false
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
mvn -Pcomet -pl spark verify -Dsuites=io.sparkvector.spark.comet.CometScanSuite
```

Without `-Pcomet` the suite is excluded by its `CometTest` tag and the rest of the build has no
Comet dependency.

## Limitations

- Comet's `LargeVarCharVector` (64-bit offsets) is not adapted zero-copy; such columns fall back to
  the copying adapter.
- Batch lifecycle follows Spark's columnar contract: Comet may reuse or release a batch as soon as
  the next one is requested, so spark-vector operators finish with a batch (or copy what they
  keep, as the aggregate does) before pulling the next.
- Comet's own shuffle and native operators are not used. A later phase can bridge spark-vector's
  unshaded Arrow output to Comet's shaded vectors zero-copy through the Arrow C Data Interface
  (both are Arrow 18.3.0) to reuse Comet's columnar shuffle.
