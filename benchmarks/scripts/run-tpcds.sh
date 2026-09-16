#!/usr/bin/env bash
# Runs the TPC-DS queries under several Spark configurations, one JVM each, then writes the markdown
# and HTML reports from every measurement recorded so far. Same arguments and environment as
# run-tpch.sh; results go to benchmarks/results/tpcds (override with RESULTS_DIR).
#
#   benchmarks/scripts/run-tpcds.sh <data-dir> [configs] [extra TpcdsRunner args...]
#   benchmarks/scripts/run-tpcds.sh benchmarks/data/tpcds-sf1 spark,vector --queries q10,q35,q45
#   benchmarks/scripts/run-tpcds.sh --report
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export RUNNER=io.sparkvector.benchmarks.TpcdsRunner
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmarks/results/tpcds}"
exec "$ROOT/benchmarks/scripts/run-tpch.sh" "$@"
