#!/usr/bin/env bash
# Runs the benchmark configurations one after another on the cluster (#248): renders each with
# render-run.sh, applies it, waits for the SparkApplication to complete or fail, and moves on. The rows
# land under <out>/<config>-<timestamp>.jsonl; `run-tpcds.sh --cluster-report <out>` builds the table.
#
#   benchmarks/k8s/run-matrix.sh <tables> <dataset> <out> <image> [runner args...]
#
# CONFIGS (env, space-separated) selects the configurations; default: the six of docs/results.md plus
# the strict one. The render-run.sh sizing variables pass through.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TABLES="${1:?tables}"; DATASET="${2:?dataset}"; OUT="${3:?out}"; IMAGE="${4:?image}"; shift 4
NAMESPACE="${NAMESPACE:-bench}"
CONFIGS="${CONFIGS:-spark vector-shuffle vector-shuffle-strict comet-scan-vector-ourshuffle comet hybrid}"
SUITE="${SUITE:-tpcds}"

for cfg in $CONFIGS; do
  name="spark-vector-$SUITE-$cfg-$DATASET"; name="${name//[^a-z0-9-]/-}"
  kubectl -n "$NAMESPACE" delete sparkapplication "$name" --ignore-not-found >/dev/null
  "$HERE/render-run.sh" "$cfg" "$TABLES" "$DATASET" "$OUT" "$IMAGE" "$@" | kubectl apply -f - >/dev/null
  start=$(date +%s)
  echo "[matrix] $cfg started $(date -u +%H:%M:%S)"
  while true; do
    state=$(kubectl -n "$NAMESPACE" get sparkapplication "$name" -o jsonpath='{.status.applicationState.state}' 2>/dev/null || true)
    case "$state" in
      COMPLETED|FAILED|SUBMISSION_FAILED|FAILING|UNKNOWN) break ;;
    esac
    sleep 30
  done
  echo "[matrix] $cfg $state after $(( $(date +%s) - start )) s"
  if [ "$state" != COMPLETED ]; then
    kubectl -n "$NAMESPACE" get sparkapplication "$name" -o jsonpath='{.status.applicationState.errorMessage}{"\n"}' | cut -c1-400
  fi
done
