#!/bin/bash
# The build-time training run (#416): AotTraining under the executor's JVM options and class path with
# -XX:AOTCacheOutput (one step: records the configuration and assembles the cache in a child JVM). It
# is the image's smoke test of the runtime -- the build fails when the cache holds no AOT-linked class,
# the incubator-module trap the jre stage exists to avoid.
# The cache the executors run with comes from the cluster (train-cluster.sh, assemble.sh): a local run
# cannot exercise the S3A/parquet client, the Flight transport over the network or the executor
# backend, and the build-time cache measured flat at 1 TB where the in-cluster one took -30% off a
# cold q18.
#
#   train.sh <output .aot> [--rounds N] [--rows N]
set -euo pipefail
OUT="${1:?output .aot path}"; shift
. "$(dirname "$0")/aot-env.sh"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
LOG="${OUT%.aot}.log"
"$JAVA" "${AOT_BASE_OPTS[@]}" \
  -Xmx"${TRAIN_HEAP:-4g}" -XX:MaxDirectMemorySize=2g -Dlog4j2.level=warn -Dspark.log.level=WARN \
  -XX:AOTCacheOutput="$OUT" -Xlog:aot=info:file="$LOG" \
  -cp "$CP" io.vecruntime.benchmarks.AotTraining "$@"

LINKED=$(grep -a "aot-linked" "$LOG" | grep -a "instance classes" | sed 's/.*aot-linked = *\([0-9]*\).*/\1/' | head -1)
echo "AOT cache: $(du -h "$OUT" | cut -f1), aot-linked instance classes = ${LINKED:-0}"
[ "${LINKED:-0}" -gt 0 ]
