# Cluster runs on Kubernetes (#246)

The cluster half of the benchmark runners: one Spark application per (configuration, dataset) pair,
reading S3 and writing one `.jsonl` row per query to S3, then `run-tpcds.sh --cluster-report <out>`
for the report in the layout of the
[data-on-EKS Comet benchmark](https://awslabs.github.io/data-on-eks/docs/benchmarks/spark-datafusion-comet-benchmark).

**Status (#247): runs on the `sfi-iceberg-bench` EKS cluster.** The image is built *on the cluster*
by `kaniko-build.yaml` from `Dockerfile` (Spark 4.1.3 on JDK 25; `benchmarks.jar` compiled from this
repository inside the build; Comet 1.0, the Iceberg 1.11 runtime, `hadoop-aws` 3.4.3 with the AWS SDK
v2 bundle, `spark-hadoop-cloud` for the S3A magic committer, Spark's `spark-sql` tests jar for the
query texts and `TPCDSSchema`, tpcds-kit's `dsdgen`; Spark's bundled Hadoop 3.4.2 client jars swapped
for 3.4.3 because their `Subject.getSubject` fails on JDK 24+). `tpcds-gen.yaml` /
`tpcds-gen-xl.yaml` generate the TPC-DS tables with `TpcdsGenRunner` (dsdgen on the executors, ZSTD
Parquet in Spark's TPC-DS schema, the fact tables partitioned by their date key); SF100 came out as 24
tables, 12,096 objects, 27 GB, exact row counts, in about 30 minutes on four 4-core executors.
`render-run.sh` renders the `SparkApplication` of one benchmark configuration from
`submit-cluster.sh`'s own settings.

Cluster facts the manifests assume: the spark-operator watches namespace `bench` only; service account
`sfi-engine` there carries the IRSA role with S3 read/write on `sfi-iceberg-wh-378683551918`; the
`bench` node group (4 x m5.2xlarge) is shared with the kafka and flink operators, so about 5 cores and
21 GiB per node are free; the `bench-xl` group (m5.4xlarge, label `workload=spark-xl`) has no
autoscaler and is scaled by hand (`aws eks update-nodegroup-config ... --scaling-config
minSize=0,maxSize=12,desiredSize=N`) -- back to 0 when idle. The repository is private: the build job
reads a fine-grained GitHub token (Contents: read) from the Secret `github-read`.

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
