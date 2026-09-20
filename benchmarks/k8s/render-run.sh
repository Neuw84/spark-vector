#!/usr/bin/env bash
# Renders the spark-operator SparkApplication for ONE (configuration, dataset) run of the cluster
# runner (#246/#248), taking the engine's spark.* settings from `submit-cluster.sh` DRY_RUN=1 so the
# manifests and the spark-submit path cannot drift. Prints YAML; pipe into kubectl.
#
#   benchmarks/k8s/render-run.sh <config> <tables> <dataset> <out> <image> [runner args...] | kubectl apply -f -
#
#   config   spark | vector | vector-shuffle | vector-shuffle-strict | comet-scan-vector-ourshuffle | hybrid | comet
#   tables   s3a://bucket/tpcds/sf1000/parquet  or  catalog:<namespace>
#   dataset  label the report groups by (sf1000-parquet)
#   out      s3a://bucket/results/sf1000-parquet
#   image    the cluster image (benchmarks/k8s/Dockerfile), e.g. <account>.dkr.ecr.<region>.amazonaws.com/spark-vector/spark:<tag>
#
# Environment (defaults for the sfi-iceberg-bench cluster's bench-xl group, m5.4xlarge):
#   NAMESPACE (bench) SERVICE_ACCOUNT (sfi-engine) SUITE (tpcds | tpch)
#   EXECUTORS (8) EXEC_CORES (14) EXEC_MEM (40g) EXEC_OVERHEAD (10g) DRIVER_CORES (2) DRIVER_MEM (8g)
#   NODE_SELECTOR (workload=spark-xl; empty for none) OFFHEAP (32g, the comet configurations)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:?config}"; TABLES="${2:?tables}"; DATASET="${3:?dataset}"; OUT="${4:?out}"; IMAGE="${5:?image}"; shift 5
NAMESPACE="${NAMESPACE:-bench}"; SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-sfi-engine}"; SUITE="${SUITE:-tpcds}"
EXECUTORS="${EXECUTORS:-8}"; EXEC_CORES="${EXEC_CORES:-14}"; EXEC_MEM="${EXEC_MEM:-40g}"; EXEC_OVERHEAD="${EXEC_OVERHEAD:-10g}"
DRIVER_CORES="${DRIVER_CORES:-2}"; DRIVER_MEM="${DRIVER_MEM:-8g}"; NODE_SELECTOR="${NODE_SELECTOR-workload=spark-xl}"
MAIN=io.sparkvector.benchmarks.TpcdsRunner; [ "$SUITE" = tpch ] && MAIN=io.sparkvector.benchmarks.TpchRunner

# The engine's --conf pairs, from the submit script's dry run (the Comet jar is on the image: no --jars).
eval "SUBMIT=($(DRY_RUN=1 BENCH_JAR=/opt/spark/jars/benchmarks.jar SUITE="$SUITE" OFFHEAP="${OFFHEAP:-32g}" \
  "$ROOT/benchmarks/scripts/submit-cluster.sh" "$CONFIG" "$TABLES" "$DATASET" "$OUT"))"
CONFS=()
for ((i = 0; i < ${#SUBMIT[@]}; i++)); do
  if [ "${SUBMIT[$i]}" = "--conf" ]; then CONFS+=("${SUBMIT[$((i + 1))]}"); fi
done

NAME="spark-vector-$SUITE-$CONFIG-$DATASET"
NAME="${NAME//[^a-z0-9-]/-}"
cat <<EOF
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: $NAME
  namespace: $NAMESPACE
  labels:
    app: spark-vector-bench
    config: $CONFIG
    dataset: $DATASET
spec:
  type: Scala
  mode: cluster
  image: $IMAGE
  imagePullPolicy: IfNotPresent
  mainClass: $MAIN
  mainApplicationFile: local:///opt/spark/jars/benchmarks.jar
  arguments:
    - "--cluster"
    - "--config"
    - "$CONFIG"
    - "--tables"
    - "$TABLES"
    - "--dataset"
    - "$DATASET"
    - "--out"
    - "$OUT"
    - "--iterations"
    - "1"
    - "--warmup"
    - "0"
EOF
for a in "$@"; do echo "    - \"$a\""; done
cat <<EOF
  sparkVersion: "4.1.3"
  sparkConf:
    spark.kubernetes.executor.deleteOnTermination: "true"
    spark.kubernetes.authenticate.executor.serviceAccountName: "$SERVICE_ACCOUNT"
    # A query that kills executors (native memory past the container limit) must not end the whole run:
    # the runner records the failure and moves on; Spark's default gives up after 16 executor losses.
    spark.executor.maxNumFailures: "200"
    spark.eventLog.enabled: "true"
    spark.eventLog.dir: "s3a://sfi-iceberg-wh-378683551918/spark-events"
    spark.hadoop.fs.s3a.connection.maximum: "200"
    spark.hadoop.fs.s3a.aws.credentials.provider: "software.amazon.awssdk.auth.credentials.WebIdentityTokenFileCredentialsProvider"
EOF
# Last setting of a key wins, as it does for spark-submit (the strict configuration overrides VECTOR's).
declare -A SEEN=(); ORDER=()
for kv in "${CONFS[@]}"; do
  k="${kv%%=*}"; v="${kv#*=}"
  if [ -z "${SEEN[$k]+x}" ]; then ORDER+=("$k"); fi
  SEEN[$k]="$v"
done
for k in "${ORDER[@]}"; do
  v="${SEEN[$k]}"
  v="${v//\"/\\\"}"   # YAML: the value quoted, embedded double quotes escaped
  echo "    $k: \"$v\""
done
cat <<EOF
  restartPolicy:
    type: Never
  driver:
    cores: $DRIVER_CORES
    memory: "$DRIVER_MEM"
    serviceAccount: $SERVICE_ACCOUNT
    labels:
      app: spark-vector-bench
EOF
if [ -n "$NODE_SELECTOR" ]; then
  echo "    nodeSelector:"; echo "      ${NODE_SELECTOR%%=*}: \"${NODE_SELECTOR#*=}\""
fi
cat <<EOF
  executor:
    instances: $EXECUTORS
    cores: $EXEC_CORES
    memory: "$EXEC_MEM"
    memoryOverhead: "$EXEC_OVERHEAD"
    labels:
      app: spark-vector-bench
    serviceAccount: $SERVICE_ACCOUNT
    volumeMounts:
      - name: tmp
        mountPath: /tmp
EOF
if [ -n "$NODE_SELECTOR" ]; then
  echo "    nodeSelector:"; echo "      ${NODE_SELECTOR%%=*}: \"${NODE_SELECTOR#*=}\""
fi
cat <<EOF
  volumes:
    - name: tmp
      emptyDir: {}
EOF
