# Changelog

All notable changes to spark-vector. The format follows [Keep a Changelog](https://keepachangelog.com/);
the project uses [semantic versioning](https://semver.org/) once it reaches 1.0 -- until then a minor
version may change configuration keys or defaults, always noted here.

## 0.0.1 -- 2026-09-24

The first preview release: the plugin as measured on the 1 TB TPC-DS campaign
(`docs/results.md`), under the Apache License 2.0.

### What it does

- Filter, Project, HashAggregate (all four modes, with spilling), Sort (spilling runs), Window,
  Expand, Generate, Union, Limit, Sample, Coalesce and the hash joins (broadcast and shuffled, the
  shuffled one a grace hash join that spills past `spark.vector.join.spillBytes`), sort-merge joins
  re-planned as hash or order-preserving merge joins -- on Arrow-layout batches with the Java Vector
  API, on the JVM, no native code. The operator and expression coverage, with what falls back and why,
  is in `docs/operators.md` and `docs/expressions.md`; the summary table is in the README.
- A columnar shuffle (`spark-vector-shuffle`): Arrow IPC record batches per reduce partition,
  zstd-compressed, served between executors over Arrow Flight or through Spark's block transfer
  (`docs/flight-shuffle.md`).
- Input from Spark's vectorized Parquet reader, from Comet's native Parquet and Iceberg scans in
  scan-only mode (zero copy), and from Iceberg's vectorized reader with merge-on-read deletes
  (v2 positional and equality deletes, v3 deletion vectors) as a selection (`docs/iceberg.md`,
  `docs/comet.md`).
- A Vector Acceleration tab in the Spark UI: per query, which operators converted and why the
  others did not.

### Measured

TPC-DS at 1 TB on EKS, eight 13-core executors with 50 GB each, 103 queries, all accelerated,
results equal to Spark's (q65 ties aside): 2557 s against Spark 4.1.3's 3309 (23% less, faster on 82
of 103) and Comet 1.0's 2514. TPC-H (22 queries) and the Iceberg merge-on-read paths verified against
Spark at SF1 and SF10. Every threshold default (`docs/configuration.md`) was set from these runs.

### Requirements

Spark 4.1.x, Scala 2.13, JDK 25 with `--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED`
on the driver and the executors; Hadoop 3.4.3 client jars in place of Spark's bundled 3.4.2 on JDK 25.
Comet 1.0 and Iceberg 1.11 optional. See "Requirements and known limitations" in the README.

### Known limitations

`ObjectHashAggregateExec` functions, cached tables, nested-type accessors and constructors, Python
UDFs and the Parquet write path fall back to Spark; the window operator does not spill; the Flight
shuffle has no TLS (use `spark.vector.shuffle.backend=block` under `spark.ssl.rpc.enabled`); measured
on x86-64 (AVX-512, AVX2) and Apple silicon (NEON), not yet on Graviton.
