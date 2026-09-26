---
layout: default
title: Overview
description: A vectorized execution runtime for Apache Spark using Java
---

# vecruntime

A Spark SQL plugin that runs Filter, Project, HashAggregate, Sort, Window and the hash joins on
Arrow-layout batches with the Java Vector API -- on the JVM, no native code -- with its own columnar
shuffle over Arrow Flight. Source, releases and the README:
[github.com/vecruntime/vecruntime](https://github.com/vecruntime/vecruntime).

vecruntime accelerates Spark SQL workloads by executing core operators directly on Arrow-layout
columnar batches using the Java Vector API (`jdk.incubator.vector`), bringing SIMD-optimized
execution to the JVM without native libraries, JNI, or serialization boundaries. Unsupported
operators, expressions, and types transparently fall back to Spark, always with a recorded reason.

## Key characteristics

- **JVM-native:** no native runtime, JNI, or external execution engine.
- **SIMD-accelerated:** uses the Java Vector API for hardware-vectorized execution.
- **Columnar by design:** operators consume and produce Arrow-layout batches.
- **Spark-compatible:** unsupported operators, expressions, and types fall back to Spark.
- **Vectorized operators:** Filter, Project, HashAggregate, Sort, Window, and hash joins.
- **Zero-copy integration:** interoperates with columnar native components such as Apache
  DataFusion Comet without serialization between execution stages.
- **Incremental adoption:** operators can be accelerated individually while the rest of the Spark
  plan continues to execute normally.

## Getting started

> vecruntime was previously named spark-vector. The configuration keys (`spark.vector.*`), the `sparkvector.*` JVM system properties, and the shuffle manager class (`spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager`) are **unchanged**. What changed: the plugin class `io.sparkvector.spark.VectorPlugin` → `io.vecruntime.spark.VectorPlugin`; the Java/Scala packages `io.sparkvector.*` → `io.vecruntime.*`; and the Maven coordinates — groupId `io.sparkvector` → `io.github.vecruntime`, artifacts `spark-vector-*` → `vecruntime-*` (e.g. `spark-vector-spark_2.13` → `vecruntime-spark_2.13`).

Add the plugin jar to an existing Spark job -- no code changes:

```bash
spark-submit \
  --conf spark.plugins=io.vecruntime.spark.VectorPlugin \
  --conf spark.driver.extraJavaOptions="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED" \
  --conf spark.executor.extraJavaOptions="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED" \
  --jars vecruntime-spark_2.13-0.0.1.jar \
  ...
```

`spark.plugins` registers the session extension automatically; alternatively set
`spark.sql.extensions=io.vecruntime.spark.VectorSparkSessionExtensions`. Every release, with the
plugin jar, the columnar shuffle jar and a `SHA256SUMS` file, is on the
[releases page](https://github.com/vecruntime/vecruntime/releases); the same artifacts are
published to a Maven repository served from the repository's `maven-repo` branch. See the
[README](https://github.com/vecruntime/vecruntime#getting-the-jars) for Maven coordinates and
`--packages` usage, and the [Configuration reference](configuration.html) for every `spark.vector.*`
key.

## Status

Version 0.0.1, a preview release under the Apache License 2.0. The plugin runs the whole of TPC-DS
(103 queries) and TPC-H (22) with every operator accelerated and returns Spark's results. On the
1 TB TPC-DS Parquet dataset on EKS, vecruntime finished the 103 queries in 2,557 s against
Spark's 3,309 s (23% less runtime, faster on 82 of 103 queries), within 2% of Apache DataFusion
Comet. The per-query tables, configurations, and every study behind the numbers are in the
[results notebook](results.html).

## Benchmarks

- [Apache Spark vs vecruntime vs DataFusion Comet on TPC-DS 1 TB](benchmarks/tpcds-1tb.html) --
  103 queries on Amazon EKS, the three engines on identical hardware and data, per-query charts and tables.
- [Apache Spark vs vecruntime on TPC-DS 1 TB, AWS Graviton4](benchmarks/tpcds-1tb-graviton.html) --
  the same run on arm64 nodes (Neoverse V2, SVE2), set against the x86 run.
- [Iceberg merge-on-read: vecruntime vs Apache Spark](benchmarks/iceberg-mor.html) -- the v2
  delete-file and v3 deletion-vector merge cost at TPC-H SF1, against OSS Spark.
- [results.md](results.html) -- the lab notebook: every run, configuration and study behind the numbers.

## Reference

- [Configuration](configuration.html) -- every `spark.vector.*` key with its default.
- [Operators](operators.html), [Expressions](expressions.html) -- what converts, under which
  conditions, and why the rest falls back.
- [Comet as the scan](comet.html), [Apache Iceberg](iceberg.html),
  [The Flight shuffle](flight-shuffle.html).
