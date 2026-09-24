---
title: spark-vector
---

# spark-vector

A Spark SQL plugin that runs Filter, Project, HashAggregate, Sort, Window and the hash joins on
Arrow-layout batches with the Java Vector API -- on the JVM, no native code -- with its own columnar
shuffle over Arrow Flight. Source, releases and the README: [github.com/Neuw84/spark-vector](https://github.com/Neuw84/spark-vector).

## Benchmarks

- [Apache Spark vs spark-vector vs DataFusion Comet on TPC-DS 1 TB](benchmarks/tpcds-1tb.html) --
  103 queries on Amazon EKS, the three engines on identical hardware and data, per-query charts and tables.
- [Iceberg merge-on-read: spark-vector vs Apache Spark](benchmarks/iceberg-mor.html) -- the v2 delete-file and v3 deletion-vector merge cost at TPC-H SF1, against OSS Spark.
- [results.md](results.md) -- the lab notebook: every run, configuration and study behind the numbers.

## Reference

- [Configuration](configuration.md) -- every `spark.vector.*` key with its default.
- [Operators](operators.md), [Expressions](expressions.md) -- what converts, under which conditions, and why the rest falls back.
- [Comet](comet.md), [Iceberg](iceberg.md), [The Flight shuffle](flight-shuffle.md).
