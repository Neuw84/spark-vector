#!/usr/bin/env bash
# Profiles ONE query in ONE configuration with Java Flight Recorder and summarises the recording --
# AGENTS.md section 4.7 as a single command. Local mode runs the TPC-H or TPC-DS harness in this
# machine's JVM with the recording flags; the rows go to a scratch RESULTS_DIR so they never enter the
# committed report.
#
#   benchmarks/scripts/profile-query.sh <data-dir> <config> <query> [options] [extra runner args...]
#
#   data-dir     the dataset directory (see gen-tpch.sh / gen-tpcds.sh); its basename names the dataset
#   config       one of spark, vector, comet-scan, comet-scan-vector, comet-scan-vector-shuffle, comet
#   query        q1 .. q22 (TPC-H) or q1 .. q99 / q14a ... (TPC-DS, with --tpcds)
#
#   --tpcds            use the TPC-DS harness (run-tpcds.sh) instead of TPC-H
#   --iterations N     measured runs (default 3, the isolation rerun of the regression protocol)
#   --warmup N         warm-up runs (default 2)
#   --out DIR          where the .jfr, the summary and the runner rows go (default: $TMPDIR or /tmp,
#                      under profiling/<dataset>/)
#   --flags-only       print the recording flags for a cluster run's spark.executor.extraJavaOptions
#                      (and driver) and exit -- the recording lands in the executor's /tmp; copy it back
#                      with the pod's preStop hook or a task-completion listener (issue #251, step 2)
#   anything else      passed to the runner (e.g. --conf spark.sql.columnVector.offheap.enabled=true --label offheap)
#
# Environment: JAVA_HOME (JDK 25), COMET_JAR for the comet* configurations, THREADS / JVM_MEM as the
# harness takes them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
usage() { echo "usage: profile-query.sh <data-dir> <config> <query> [--tpcds] [--iterations N] [--warmup N] [--out DIR] [--flags-only] [runner args...]" >&2; exit 2; }
DATA="${1:-}"; CONFIG="${2:-}"; QUERY="${3:-}"
[ -n "$DATA" ] && [ -n "$CONFIG" ] && [ -n "$QUERY" ] || usage
shift 3
HARNESS=run-tpch.sh
ITER=3
WARM=2
OUT=""
FLAGS_ONLY=0
PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tpcds) HARNESS=run-tpcds.sh; shift ;;
    --iterations) ITER="$2"; shift 2 ;;
    --warmup) WARM="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --flags-only) FLAGS_ONLY=1; shift ;;
    *) PASS+=("$1"); shift ;;
  esac
done
DATASET="$(basename "$DATA")"
STEM="${QUERY}-${CONFIG}"
if [ "$FLAGS_ONLY" = 1 ]; then
  # Executors write to their own /tmp (mount an emptyDir there on Kubernetes); one file per JVM.
  echo "spark.executor.extraJavaOptions=-XX:StartFlightRecording=filename=/tmp/${STEM}.jfr,settings=profile,dumponexit=true"
  echo "spark.driver.extraJavaOptions=-XX:StartFlightRecording=filename=/tmp/${STEM}-driver.jfr,settings=profile,dumponexit=true"
  exit 0
fi
: "${JAVA_HOME:?set JAVA_HOME to a JDK 25}"
[ -d "$DATA" ] || { echo "no such data directory: $DATA" >&2; exit 2; }
OUT="${OUT:-${TMPDIR:-/tmp}/profiling/$DATASET}"
mkdir -p "$OUT/results"
JFR_FILE="$OUT/$STEM.jfr"
SUMMARY="$OUT/$STEM.summary.txt"
LOG="$OUT/$STEM.log"
rm -f "$JFR_FILE"

echo "=== profiling $QUERY / $CONFIG over $DATASET: $WARM warm-up + $ITER measured, recording to $JFR_FILE"
# JVM_EXTRA reaches the benchmark JVM only (the harness starts a separate JVM for the report, which
# would otherwise overwrite the recording); RESULTS_DIR keeps the rows out of benchmarks/results.
JVM_EXTRA="-XX:StartFlightRecording=filename=$JFR_FILE,settings=profile,dumponexit=true" \
RESULTS_DIR="$OUT/results" \
  "$ROOT/benchmarks/scripts/$HARNESS" "$DATA" "$CONFIG" --queries "$QUERY" --iterations "$ITER" --warmup "$WARM" ${PASS[@]+"${PASS[@]}"} 2>&1 | tee "$LOG" \
  | grep -E '^\[tpc(h|ds)\]' || true
[ -f "$JFR_FILE" ] || { echo "no recording was written; see $LOG" >&2; exit 1; }

echo
echo "=== summary -> $SUMMARY"
"$ROOT/benchmarks/scripts/jfr-summary.sh" "$JFR_FILE" > "$SUMMARY"
# Read in the order section 4.7 gives: kernel times (above), then the hot methods and their callers.
sed -n '1,/^== 3\./p' "$SUMMARY"
echo "(allocation, GC, latencies and native methods follow in $SUMMARY)"
