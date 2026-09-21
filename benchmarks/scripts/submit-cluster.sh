#!/usr/bin/env bash
# Submits ONE (configuration, dataset) pair of the cluster runner (#246) as a Spark application: the
# engine configuration becomes spark-submit --conf pairs (a plugin cannot join a running context, so
# the runner takes the session it is given and only checks it), the JVM flags are the ones the local
# harness uses, and the rows land under --out on any Hadoop file system.
#
#   benchmarks/scripts/submit-cluster.sh <config> <tables> <dataset> <out> [runner args...]
#
#   config    spark | vector | comet-scan-vector-shuffle | comet   (TpchRunner.Configs; comet* need COMET_JAR)
#   tables    s3://bucket/tpcds/sf1000/parquet  (a directory per table)  or  catalog:<namespace>  (Iceberg)
#   dataset   label the report groups by, e.g. sf1000-parquet, sf1000-iceberg, sf1000-iceberg-mor
#   out       s3://bucket/results/sf1000-parquet  (one <config>-<timestamp>.jsonl per run)
#   runner args, e.g. --queries q1,q6 --iterations 1 --warmup 0 --queries-dir s3://bucket/tpcds/queries
#
# Environment:
#   SPARK_SUBMIT   spark-submit binary (default: spark-submit on PATH)
#   MASTER         e.g. k8s://https://<api-server> (default: whatever spark-submit's defaults give)
#   SUBMIT_ARGS    extra spark-submit arguments (--deploy-mode, --conf spark.kubernetes.*, executors, image)
#   BENCH_JAR      benchmarks.jar (default: benchmarks/target/benchmarks.jar, `mvn -pl benchmarks -am -DskipTests package`)
#   COMET_JAR      the Comet Spark 4.1 jar, added with --jars for the comet* configurations
#   SUITE          tpcds (default) or tpch
#   OFFHEAP        off-heap size for the comet configurations (default 32g, the reference's setting)
#
# Print the configuration without submitting: DRY_RUN=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:?usage: submit-cluster.sh <config> <tables> <dataset> <out> [runner args...]}"
TABLES="${2:?tables base URI or catalog:<namespace>}"
DATASET="${3:?dataset label}"
OUT="${4:?output URI}"
shift 4
SUITE="${SUITE:-tpcds}"
case "$SUITE" in
  tpcds) MAIN=io.sparkvector.benchmarks.TpcdsRunner ;;
  tpch) MAIN=io.sparkvector.benchmarks.TpchRunner ;;
  *) echo "SUITE must be tpcds or tpch" >&2; exit 2 ;;
esac
JAR="${BENCH_JAR:-$ROOT/benchmarks/target/benchmarks.jar}"
OFFHEAP="${OFFHEAP:-32g}"

# The engine configurations, mirroring TpchRunner.Configs (kept in step by hand; the runner warns
# when the session disagrees with the configuration it is labelled with).
VECTOR=(--conf spark.plugins=io.sparkvector.spark.VectorPlugin
        --conf spark.vector.exec.strictFloatingPoint=false
        --conf spark.vector.exec.sortMergeJoin.enabled=true
        --conf spark.sql.parquet.enableVectorizedReader=true
        --conf spark.sql.columnVector.offheap.enabled=true) # #403: fixed-width lanes wrapped in place
COMET_SCAN_ONLY=(--conf spark.comet.enabled=true --conf spark.comet.scan.enabled=true --conf spark.comet.exec.enabled=true
        --conf spark.comet.exec.shuffle.enabled=false
        --conf spark.comet.exec.project.enabled=false --conf spark.comet.exec.filter.enabled=false
        --conf spark.comet.exec.aggregate.enabled=false --conf spark.comet.exec.sort.enabled=false
        --conf spark.comet.exec.localLimit.enabled=false --conf spark.comet.exec.globalLimit.enabled=false
        --conf spark.comet.exec.takeOrderedAndProject.enabled=false --conf spark.comet.exec.hashJoin.enabled=false
        --conf spark.comet.exec.sortMergeJoin.enabled=false --conf spark.comet.exec.broadcastHashJoin.enabled=false
        --conf spark.comet.exec.broadcastExchange.enabled=false --conf spark.comet.exec.expand.enabled=false
        --conf spark.comet.exec.union.enabled=false --conf spark.comet.exec.window.enabled=false
        --conf spark.comet.exec.coalesce.enabled=false --conf spark.comet.exec.collectLimit.enabled=false
        --conf spark.comet.exec.explode.enabled=false --conf spark.comet.exec.sample.enabled=false
        --conf spark.memory.offHeap.enabled=true --conf "spark.memory.offHeap.size=$OFFHEAP")
case "$CONFIG" in
  spark) ENGINE=() ;;
  vector) ENGINE=("${VECTOR[@]}") ;;
  comet-scan-vector-shuffle)
    ENGINE=("${VECTOR[@]}" --conf spark.plugins=org.apache.spark.CometPlugin,io.sparkvector.spark.VectorPlugin
            --conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager
            "${COMET_SCAN_ONLY[@]}" --conf spark.comet.exec.shuffle.enabled=true) ;;
  comet)
    ENGINE=(--conf spark.plugins=org.apache.spark.CometPlugin --conf spark.comet.enabled=true --conf spark.comet.scan.enabled=true
            --conf spark.comet.exec.enabled=true --conf spark.comet.exec.shuffle.enabled=true --conf spark.comet.exec.shuffle.mode=auto
            --conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager
            --conf spark.comet.explainFallback.enabled=true --conf spark.comet.cast.allowIncompatible=true
            --conf spark.memory.offHeap.enabled=true --conf "spark.memory.offHeap.size=$OFFHEAP") ;;
  *) echo "unknown config $CONFIG (spark, vector, comet-scan-vector-shuffle, comet)" >&2; exit 2 ;;
esac
JARS=()
case "$CONFIG" in comet*) JARS=(--jars "${COMET_JAR:?set COMET_JAR for the comet configurations}") ;; esac

# The JVM flags the local harness uses, for driver and executors alike.
JVM_FLAGS="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow \
--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED \
--add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED \
--add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED \
--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED \
--add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED \
-Djdk.reflect.useDirectMethodHandle=false -Dio.netty.tryReflectionSetAccessible=true"

MASTER_ARGS=()
if [ -n "${MASTER:-}" ]; then MASTER_ARGS=(--master "$MASTER"); fi
EXTRA=()
if [ -n "${SUBMIT_ARGS:-}" ]; then read -ra EXTRA <<< "$SUBMIT_ARGS"; fi

CMD=("${SPARK_SUBMIT:-spark-submit}" "${MASTER_ARGS[@]}" --class "$MAIN" --name "spark-vector-$SUITE-$CONFIG-$DATASET"
     --conf "spark.driver.extraJavaOptions=$JVM_FLAGS" --conf "spark.executor.extraJavaOptions=$JVM_FLAGS"
     --conf spark.sql.adaptive.enabled=true
     "${ENGINE[@]}" ${JARS[@]+"${JARS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
     "$JAR" --cluster --config "$CONFIG" --tables "$TABLES" --dataset "$DATASET" --out "$OUT" --iterations 1 --warmup 0 "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
