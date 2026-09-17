# Cluster runs on Kubernetes (#246)

The cluster half of the benchmark runners: one Spark application per (configuration, dataset) pair,
reading S3 and writing one `.jsonl` row per query to S3, then `run-tpcds.sh --cluster-report <out>`
for the report in the layout of the
[data-on-EKS Comet benchmark](https://awslabs.github.io/data-on-eks/docs/benchmarks/spark-datafusion-comet-benchmark).

**Status: the runner, the submit script and the report are exercised locally (a `local[N]` session,
`file://` tables and output); nothing here has run against a cluster yet.** The manifest below is a
template with the settings the runner needs; the image is the maintainer's to build (Spark 4.1.x on
JDK 25 with `benchmarks.jar`, the Comet 1.0 Spark 4.1 jar and `iceberg-spark-runtime-4.1_2.13`; the
reference image is Spark 3.5 on JDK 17 and cannot be reused).

## What a run needs

- `benchmarks.jar` (`mvn -Piceberg -pl benchmarks -am -DskipTests package`) on the image or on S3;
  the main class is `io.sparkvector.benchmarks.TpcdsRunner` (`TpchRunner` for TPC-H), invoked with
  `--cluster --config <cfg> --tables <base URI | catalog:<ns>> --dataset <label> --out <URI>`.
- The engine configuration as `spark.*` properties -- `benchmarks/scripts/submit-cluster.sh` emits
  them for `spark`, `vector`, `comet-scan-vector-shuffle` and `comet` (`DRY_RUN=1` prints the full
  `spark-submit` line to copy into a manifest). A plugin cannot join a running context, so the runner
  only checks the session against the configuration it is labelled with and warns on disagreement.
- The JVM flags of the local harness on driver and executors (`--add-modules=jdk.incubator.vector`,
  `--enable-native-access`, the `--add-opens` set): `submit-cluster.sh` puts them in
  `spark.driver.extraJavaOptions` / `spark.executor.extraJavaOptions`.
- Query texts: the classpath (the spark-sql tests jar) or `--queries-dir s3://.../tpcds/queries`
  holding `q1.sql` ... `q99.sql`, `q14a.sql` ... for an image without it.
- For the JFR-first protocol (#251): `profile-query.sh --flags-only` prints the recording options;
  the recording lands in the executor's `/tmp`, so mount an `emptyDir` there and copy it out in a
  `preStop` hook or a task-completion listener.

## Row shape

The cluster rows are the local runners' rows plus: `stages`, `executorRunTimeMs`, `gcTimeMs`,
`shuffleReadBytes`, `shuffleWriteBytes`, `spillBytes`, `peakExecutionMemory` (sums over the query's
stages from `SparkListenerStageCompleted`, attributed through a job group per run), and
`sparkVersion`, `executors`, `engineConf` for the environment block. Object stores have no append, so
each run writes `<config>[-label]-<timestamp>.jsonl`; the report reads every file of the directory
and takes the latest row per (configuration, query).

## Report

`run-tpcds.sh --cluster-report s3://bucket/results/sf1000-parquet` (or `TpcdsRunner --cluster-report
<dir>`; reading S3 needs the Hadoop S3 connector on the classpath, as on the cluster image) writes
`cluster-results.md` beside the rows: summary (total time, speedup, % less runtime), the speedup
distribution in the reference's five buckets, top improvements and every regression, the analysis
table with the stage evidence pre-filled (the cause is written from the profile), the per-query table
with the accelerated-operator ratio, and the environment. Queries whose checksums differ between
configurations are excluded from the totals and listed as correctness bugs.

## Template

`spark-application.yaml` is a spark-operator `SparkApplication` for one configuration; substitute the
placeholders (`IMAGE`, `S3_BUCKET`, `DATASET`, `CONFIG`) and add the engine `sparkConf` entries from
`submit-cluster.sh`'s dry run.
