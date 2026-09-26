#!/bin/bash
# Trains the executor's JDK AOT cache on the cluster (#416) and publishes it for the image tag, so
# every later run of that image starts its executors from it (render-run.sh fetches
# s3://<bucket>/aot/<image tag>/executor.aot in an init container).
#
#   benchmarks/k8s/aot/train-cluster.sh <tables> <dataset> <out> <image> [runner args...]
#
#   tables/dataset/out/image  as for render-run.sh; `out` names the results bucket the cache goes to
#   runner args              the training set, default --queries q18,q67,q22,q4 --iterations 1 --warmup 0
#
# Three steps, about ten minutes of cluster time on the bench-xl group:
#   1. a recording run: the vector-shuffle configuration over the training set with the executors in
#      -XX:AOTMode=record (render-run.sh AOT_RECORD=1), each writing its configuration to the node's
#      /mnt/vecruntime-aot/<tag>/executor.aotconf. Real executors: the S3A/parquet client, the Flight
#      transport over the network and the executor backend are in the recording -- a local training run
#      at image build cannot reach them, and measured flat at 1 TB where this took -30% off a cold q18.
#   2. a DaemonSet on the same nodes: on every node that holds a recording, assemble.sh turns it into
#      the cache (the executor JVM options and class path are fixed by the image, so any node's serves
#      all) and an aws-cli container uploads it to s3://<bucket>/aot/<tag>/executor.aot -- the first
#      upload wins, the others find the object and skip.
#   3. the DaemonSet is removed; the linked-class count is printed as the check.
# Sizing variables (EXECUTORS, EXEC_CORES, EXEC_MEM, EXEC_OVERHEAD, NODE_SELECTOR, ...) pass through to
# render-run.sh; one executor per node keeps one recording per node.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TABLES="${1:?tables}"; DATASET="${2:?dataset}"; OUT="${3:?out}"; IMAGE="${4:?image}"; shift 4
[ $# -gt 0 ] || set -- --queries q18,q67,q22,q4 --iterations 1 --warmup 0
NAMESPACE="${NAMESPACE:-bench}"; SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-sfi-engine}"; NODE_SELECTOR="${NODE_SELECTOR-workload=spark-xl}"
TAG="${IMAGE##*:}"
BUCKET="${AOT_BUCKET:-${OUT#s3a://}}"; BUCKET="${BUCKET%%/*}"
DEST="s3://$BUCKET/aot/$TAG/executor.aot"
HOSTDIR="/mnt/vecruntime-aot/$TAG"
DS="aot-assemble-${TAG//[^a-z0-9-]/-}"

echo "[aot] recording run of $IMAGE: $*"
AOT_RECORD=1 CONFIGS="${CONFIGS:-vector-shuffle}" "$HERE/../run-matrix.sh" "$TABLES" "$DATASET" "${OUT%/}/aot-train-$TAG" "$IMAGE" "$@"

echo "[aot] assembling on the nodes and uploading to $DEST"
kubectl -n "$NAMESPACE" delete daemonset "$DS" --ignore-not-found --wait=true >/dev/null
{
cat <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: $DS
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      app: $DS
  template:
    metadata:
      labels:
        app: $DS
    spec:
      serviceAccountName: $SERVICE_ACCOUNT
EOF
if [ -n "$NODE_SELECTOR" ]; then
  echo "      nodeSelector:"; echo "        ${NODE_SELECTOR%%=*}: \"${NODE_SELECTOR#*=}\""
fi
cat <<EOF
      volumes:
        - name: aot
          hostPath:
            path: $HOSTDIR
            type: DirectoryOrCreate
      containers:
        - name: assemble
          image: $IMAGE
          command: ["bash", "-c", "if [ -f /aot/executor.aotconf ]; then rm -f /aot/executor.aot.ok; /opt/spark/aot/assemble.sh /aot/executor.aotconf /aot/executor.aot && touch /aot/executor.aot.ok; echo ASSEMBLE_EXIT=\$?; else echo NO_RECORDING; fi; sleep infinity"]
          volumeMounts:
            - name: aot
              mountPath: /aot
        - name: upload
          image: public.ecr.aws/aws-cli/aws-cli:latest
          command: ["sh", "-c", "for i in \$(seq 1 120); do [ -f /aot/executor.aot.ok ] && break; sleep 5; done; if [ -f /aot/executor.aot.ok ]; then if aws s3api head-object --bucket $BUCKET --key aot/$TAG/executor.aot >/dev/null 2>&1; then echo UPLOAD_SKIPPED; else aws s3 cp /aot/executor.aot $DEST; echo UPLOAD_EXIT=\$?; fi; else echo NO_CACHE; fi; sleep infinity"]
          volumeMounts:
            - name: aot
              mountPath: /aot
EOF
} | kubectl -n "$NAMESPACE" apply -f - >/dev/null

done_line=""
for i in $(seq 1 120); do
  sleep 10
  for p in $(kubectl -n "$NAMESPACE" get pods -l "app=$DS" --no-headers 2>/dev/null | awk '$3=="Running"{print $1}'); do
    if kubectl -n "$NAMESPACE" logs "$p" -c upload 2>/dev/null | grep -q "UPLOAD_EXIT=0"; then done_line="$p"; break 2; fi
  done
done
if [ -z "$done_line" ]; then
  echo "[aot] no node uploaded a cache within 20 minutes; assemble logs:"
  for p in $(kubectl -n "$NAMESPACE" get pods -l "app=$DS" --no-headers 2>/dev/null | awk '{print $1}'); do
    echo "--- $p"; kubectl -n "$NAMESPACE" logs "$p" -c assemble 2>/dev/null | tail -3
  done
  kubectl -n "$NAMESPACE" delete daemonset "$DS" --ignore-not-found >/dev/null
  exit 1
fi
kubectl -n "$NAMESPACE" logs "$done_line" -c assemble 2>/dev/null | grep -a "AOT cache:" | tail -1
kubectl -n "$NAMESPACE" delete daemonset "$DS" --ignore-not-found >/dev/null
echo "[aot] published $DEST"
