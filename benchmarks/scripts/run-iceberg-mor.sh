#!/usr/bin/env bash
# Iceberg merge-on-read on the cluster: generate the v2 and v3 warehouses from lineitem on S3, then
# run the CDC merge benchmark for `spark` and `vector` over each variant, applying the manifests one
# SparkApplication at a time (one alone on the cluster, like run-matrix.sh). The generated warehouse
# and the change batch live on S3; each run's JSONL is written to the driver's local disk, and this
# script copies it from the driver pod to s3://.../results/iceberg-mor-cdc/ when the run completes.
#
# Usage:
#   run-iceberg-mor.sh <s3_bucket> <tpch_sf_prefix> <image> [v2|v3|both] [variants]
# e.g.
#   run-iceberg-mor.sh sfi-iceberg-wh-378683551918 tpch/sf480 REGISTRY/spark:tag both
#
# The variant list defaults to the eight positional/deletion-vector shapes plus the two equality
# shapes (v2 only); v3 has no equality variant. lineitem at the requested scale must already exist
# under s3a://<bucket>/<tpch_sf_prefix>/lineitem (generate it with the TPC-H generator first).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUCKET="${1:?s3 bucket}"; DATA_PREFIX="${2:?data prefix holding <base-table>/, e.g. tpcds/sf1000/parquet}"; IMAGE="${3:?image}"
WHICH="${4:-both}"
BASE_TABLE="${BASE_TABLE:-lineitem}"  # the base table under s3a://<bucket>/<data_prefix>/<base_table>/
NS_BENCH=bench
FILES="${FILES:-300}"                 # data files the base table is written as (deletes span all)
V2_VARIANTS="${5:-plain,pos_2,pos_10,pos_30,pos_10_clustered,pos_30_clustered,pos_upd_1,pos_upd_5,eq_2,eq_10}"
V3_VARIANTS="${5:-plain,dv_2,dv_10,dv_30,dv_10_clustered,dv_30_clustered,dv_upd_1,dv_upd_5}"

apply_and_wait() { # <name> <rendered-yaml-file>
  local name="$1" file="$2"
  kubectl -n "$NS_BENCH" delete sparkapplication "$name" --ignore-not-found >/dev/null 2>&1 || true
  sleep 5
  kubectl -n "$NS_BENCH" apply -f "$file" >/dev/null
  echo "=== $name submitted $(date -u +%H:%M:%S)"
  while :; do
    local state
    state="$(kubectl -n "$NS_BENCH" get sparkapplication "$name" -o jsonpath='{.status.applicationState.state}' 2>/dev/null || true)"
    case "$state" in
      COMPLETED) echo "=== $name COMPLETED $(date -u +%H:%M:%S)"; return 0 ;;
      FAILED|SUBMISSION_FAILED) echo "=== $name $state $(date -u +%H:%M:%S)"; kubectl -n "$NS_BENCH" logs "${name}-driver" 2>/dev/null | tail -30; return 1 ;;
    esac
    sleep 20
  done
}

gen() { # <namespace> <variants>
  local ns="$1" variants="$2" name="spark-vector-mor-gen-$1" f
  f="$(mktemp)"
  sed -e "s|IMAGE|${IMAGE}|g" -e "s|S3_BUCKET|${BUCKET}|g" -e "s|DATA_PREFIX|${DATA_PREFIX}|g" \
      -e "s|NAMESPACE|${ns}|g" -e "s|VARIANTS|${variants}|g" -e "s|FILES|${FILES}|g" -e "s|BASE_TABLE|${BASE_TABLE}|g" \
      "$HERE/iceberg-mor-gen.yaml" > "$f"
  apply_and_wait "$name" "$f"; rm -f "$f"
}

cdc() { # <config> <namespace> <variant>
  local config="$1" ns="$2" variant="$3" table="${ns}.${variant}"
  local name="spark-vector-mor-cdc-${config}-${ns}-${variant}" f
  f="$(mktemp)"
  # The manifest name pattern is CONFIG-TABLE; render TABLE as ns.variant and give the SparkApplication a k8s-safe name.
  sed -e "s|IMAGE|${IMAGE}|g" -e "s|S3_BUCKET|${BUCKET}|g" -e "s|NAMESPACE|${ns}|g" -e "s|BASE_TABLE|${BASE_TABLE}|g" \
      -e "s|spark-vector-mor-cdc-CONFIG-TABLE|${name}|" -e "s|\"CONFIG\"|\"${config}\"|" -e "s|\"TABLE\"|\"${table}\"|" \
      "$HERE/iceberg-mor-cdc.yaml" > "$f"
  apply_and_wait "$name" "$f" || { rm -f "$f"; return 1; }
  # Copy the driver's local JSONL to S3 (one file per config, appended across variants in-JVM; here per run).
  kubectl -n "$NS_BENCH" cp "${name}-driver:/opt/spark/work-dir/cdc-results/cdc-${config}.jsonl" "/tmp/cdc-${config}-${ns}-${variant}.jsonl" 2>/dev/null \
    && aws s3 cp "/tmp/cdc-${config}-${ns}-${variant}.jsonl" "s3://${BUCKET}/results/iceberg-mor-cdc/cdc-${config}-${ns}-${variant}.jsonl" >/dev/null \
    && echo "    results -> s3://${BUCKET}/results/iceberg-mor-cdc/cdc-${config}-${ns}-${variant}.jsonl" \
    || echo "    WARNING: could not copy results JSONL for ${name}"
  rm -f "$f"
}

run_ns() { # <namespace> <variants>
  local ns="$1" variants="$2"
  gen "$ns" "$variants"
  local IFS=,
  for v in $variants; do
    cdc spark  "$ns" "$v"
    cdc vector "$ns" "$v"
  done
}

[ "$WHICH" = "v2" ] || [ "$WHICH" = "both" ] && run_ns v2 "$V2_VARIANTS"
[ "$WHICH" = "v3" ] || [ "$WHICH" = "both" ] && run_ns v3 "$V3_VARIANTS"
echo "=== iceberg-mor done $(date -u +%H:%M:%S); results under s3://${BUCKET}/results/iceberg-mor-cdc/"
