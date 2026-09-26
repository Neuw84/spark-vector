#!/usr/bin/env bash
# Renders the spark-operator SparkApplication for ONE (configuration, dataset) run of the cluster
# runner (#246/#248), taking the engine's spark.* settings from `submit-cluster.sh` DRY_RUN=1 so the
# manifests and the spark-submit path cannot drift. Prints YAML; pipe into kubectl.
#
#   benchmarks/k8s/render-run.sh <config> <tables> <dataset> <out> <image> [runner args...] | kubectl apply -f -
#
#   config   spark | vector | vector-shuffle | vector-shuffle-strict | comet-scan | comet-scan-vector-ourshuffle | hybrid | comet
#   tables   s3a://bucket/tpcds/sf1000/parquet  or  catalog:<namespace>
#   dataset  label the report groups by (sf1000-parquet)
#   out      s3a://bucket/results/sf1000-parquet
#   image    the cluster image (benchmarks/k8s/Dockerfile), e.g. <account>.dkr.ecr.<region>.amazonaws.com/spark-vector/spark:<tag>
#
# Environment (defaults for the sfi-iceberg-bench cluster's bench-xl group, m5.4xlarge):
#   NAMESPACE (bench) SERVICE_ACCOUNT (sfi-engine) SUITE (tpcds | tpch)
#   EXECUTORS (8) EXEC_CORES (14) EXEC_MEM (40g) EXEC_OVERHEAD (10g) DRIVER_CORES (2) DRIVER_MEM (8g)
#   NODE_SELECTOR (workload=spark-xl; empty for none) OFFHEAP (32g, the comet configurations)
#   EXEC_JAVA_OPTS (extra executor JVM options, e.g. a JFR recording: -XX:StartFlightRecording=duration=300s,filename=/tmp/exec.jfr,settings=profile)
#   KEEP_EXECUTORS (unset; set to 1 to keep dead executor pods for their logs)
#   DIRECT_MEM (EXEC_OVERHEAD minus 2g; the executors' -XX:MaxDirectMemorySize)
#   AOT_CACHE (0; 1 = the executors start from the image's AOT cache, trained on the cluster (#416):
#             an init container fetches s3://<bucket>/aot/<image tag>/executor.aot (AOT_BUCKET, default
#             the results bucket) to /aot and the JVM gets -XX:AOTCache=/aot/executor.aot. No object
#             yet -- the image's training run has not happened -- means no file and a JVM warning, the
#             run proceeds without a cache. Off by default since the cache measured +22 % on the heavy
#             queries at 1 TB (q14a +58 %) for a cold-start gain on the short ones -- docs/results.md.)
#   AOT_RECORD (unset; 1 = the training run: the executors record their configuration with
#             -XX:AOTMode=record to a per-node hostPath directory, /mnt/vecruntime-aot/<image tag>,
#             which benchmarks/k8s/aot/train-cluster.sh then assembles into the cache and uploads)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:?config}"; TABLES="${2:?tables}"; DATASET="${3:?dataset}"; OUT="${4:?out}"; IMAGE="${5:?image}"; shift 5
NAMESPACE="${NAMESPACE:-bench}"; SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-sfi-engine}"; SUITE="${SUITE:-tpcds}"
EXECUTORS="${EXECUTORS:-8}"; EXEC_CORES="${EXEC_CORES:-14}"; EXEC_MEM="${EXEC_MEM:-40g}"; EXEC_OVERHEAD="${EXEC_OVERHEAD:-10g}"
DRIVER_CORES="${DRIVER_CORES:-2}"; DRIVER_MEM="${DRIVER_MEM:-8g}"; NODE_SELECTOR="${NODE_SELECTOR-workload=spark-xl}"
MAIN=io.vecruntime.benchmarks.TpcdsRunner; [ "$SUITE" = tpch ] && MAIN=io.vecruntime.benchmarks.TpchRunner
DIRECT_MEM="${DIRECT_MEM:-$(( ${EXEC_OVERHEAD%g} - 2 ))g}"

# The engine's --conf pairs, from the submit script's dry run (the Comet jar is on the image: no --jars).
eval "SUBMIT=($(DRY_RUN=1 BENCH_JAR=/opt/spark/jars/benchmarks.jar SUITE="$SUITE" OFFHEAP="${OFFHEAP:-32g}" \
  "$ROOT/benchmarks/scripts/submit-cluster.sh" "$CONFIG" "$TABLES" "$DATASET" "$OUT"))"
CONFS=()
for ((i = 0; i < ${#SUBMIT[@]}; i++)); do
  if [ "${SUBMIT[$i]}" = "--conf" ]; then CONFS+=("${SUBMIT[$((i + 1))]}"); fi
done

NAME="vecruntime-$SUITE-$CONFIG-$DATASET"
NAME="${NAME//[^a-z0-9-]/-}"
cat <<EOF
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: $NAME
  namespace: $NAMESPACE
  labels:
    app: vecruntime-bench
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
    # KEEP_EXECUTORS=1 leaves finished/dead executor pods in place, so a dying executor's own log survives.
    spark.kubernetes.executor.deleteOnTermination: "$([ -n "${KEEP_EXECUTORS:-}" ] && echo false || echo true)"
    spark.kubernetes.authenticate.executor.serviceAccountName: "$SERVICE_ACCOUNT"
    # A query that kills executors (native memory past the container limit) must not end the whole run:
    # the runner records the failure and moves on; Spark's default gives up after 16 executor losses.
    spark.executor.maxNumFailures: "200"
    # Spark 4.1 on JDK 24+: SerializationDebugger's initialiser fails (SPARK-55679, fixed in 4.2.0), which
    # turns a non-serializable task failure into an executor death; off, the plain exception is reported.
    # See upstream/spark-55679-serialization-debugger-jdk25/.
    spark.serializer.extraDebugInfo: "false"
    # Shuffle files go when the driver's ContextCleaner sees the ShuffleDependency collected, which waits for
    # a GC: over a 100-query run with the default 30 min the files of finished queries filled a node's 20 GB
    # root disk and the kubelet evicted the executor (SF100 v4: q95-q99 lost to disk pressure).
    spark.cleaner.periodicGC.interval: "2min"
    # Arrow's Netty allocator is bounded by the JVM's direct-memory limit, which defaults to the heap size:
    # give it the overhead instead (DIRECT_MEM; default EXEC_OVERHEAD less 2g), or our kernels' and the
    # shuffle's buffers hit a 20 GiB wall inside a 50 GiB container.
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
# The executor's JVM options are one key: the engine's module flags, the direct-memory bound and any
# EXEC_JAVA_OPTS appended -- a second `spark.executor.extraJavaOptions` line would silently win over
# the first in the YAML map (it did: the direct-memory bound never reached the executors before this).
EXEC_OPTS="${SEEN[spark.executor.extraJavaOptions]:-}"
EXEC_OPTS="${EXEC_OPTS:+$EXEC_OPTS }-XX:MaxDirectMemorySize=$DIRECT_MEM${EXEC_JAVA_OPTS:+ $EXEC_JAVA_OPTS}"
# The executor's AOT cache (#416), trained on the cluster by real executors. A training run
# (AOT_RECORD=1) records each executor's configuration to the node; a normal run fetches the assembled
# cache for this image tag from S3 in an init container and points the JVM at it. A missing object,
# or a cache whose class path or module options differ from the JVM's, is a warning and a run without
# a cache -- AOTMode stays auto, so nothing fails.
IMAGE_TAG="${IMAGE##*:}"
AOT_BUCKET="${AOT_BUCKET:-${OUT#s3a://}}"; AOT_BUCKET="${AOT_BUCKET%%/*}"
AOT_MODE=""
if [ "${AOT_RECORD:-0}" = "1" ]; then
  AOT_MODE=record
  EXEC_OPTS="$EXEC_OPTS -XX:AOTMode=record -XX:AOTConfiguration=/aot/executor.aotconf -Xlog:aot=info:file=/aot/record.log"
elif [ "${AOT_CACHE:-0}" = "1" ]; then
  AOT_MODE=fetch
  EXEC_OPTS="$EXEC_OPTS -XX:AOTCache=/aot/executor.aot"
fi
if [ -z "${SEEN[spark.executor.extraJavaOptions]+x}" ]; then ORDER+=("spark.executor.extraJavaOptions"); fi
SEEN[spark.executor.extraJavaOptions]="$EXEC_OPTS"
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
      app: vecruntime-bench
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
      app: vecruntime-bench
    serviceAccount: $SERVICE_ACCOUNT
    volumeMounts:
      - name: tmp
        mountPath: /tmp
EOF
if [ -n "$AOT_MODE" ]; then
  echo "      - name: aot"; echo "        mountPath: /aot"
fi
if [ "$AOT_MODE" = fetch ]; then
  # The cache for this image tag, if the cluster has trained one; the executor's service account
  # (IRSA) is the init container's too. Nothing to fetch is not a failure: the JVM runs without it.
  cat <<EOF
    initContainers:
      - name: aot-fetch
        image: public.ecr.aws/aws-cli/aws-cli:latest
        command: ["sh", "-c", "aws s3 cp s3://$AOT_BUCKET/aot/$IMAGE_TAG/executor.aot /aot/executor.aot || echo 'no AOT cache for $IMAGE_TAG'"]
        volumeMounts:
          - name: aot
            mountPath: /aot
EOF
elif [ "$AOT_MODE" = record ]; then
  # The node directory is created root-owned by the kubelet and the executor runs as the image's
  # user: a JVM in record mode that cannot open its configuration file exits at start, so a root
  # init container opens the directory first.
  cat <<EOF
    initContainers:
      - name: aot-prep
        image: public.ecr.aws/aws-cli/aws-cli:latest
        command: ["sh", "-c", "chmod 1777 /aot"]
        securityContext:
          runAsUser: 0
        volumeMounts:
          - name: aot
            mountPath: /aot
EOF
fi
if [ -n "$NODE_SELECTOR" ]; then
  echo "    nodeSelector:"; echo "      ${NODE_SELECTOR%%=*}: \"${NODE_SELECTOR#*=}\""
fi
cat <<EOF
  volumes:
    - name: tmp
      emptyDir: {}
EOF
case "$AOT_MODE" in
  fetch)  printf '    - name: aot\n      emptyDir: {}\n' ;;
  record) printf '    - name: aot\n      hostPath:\n        path: /mnt/vecruntime-aot/%s\n        type: DirectoryOrCreate\n' "$IMAGE_TAG" ;;
esac
