#!/usr/bin/env bash
# The Iceberg v2 merge-on-read CDC benchmark: reads plus a timed MERGE INTO of a ~quarter-of-the-
# live-rows change batch, per configuration, the table rolled back to the pinned snapshot between
# runs (see CdcMergeRunner). One JVM per configuration, then the report.
#
#   benchmarks/scripts/run-cdc-merge.sh <warehouse> <namespace.table> [configs] [extra args]
#   benchmarks/scripts/run-cdc-merge.sh benchmarks/data/iceberg sf25.pos_20 spark,vector
#   benchmarks/scripts/run-cdc-merge.sh benchmarks/data/iceberg sf25.pos_20 spark,vector --iterations 3
#   benchmarks/scripts/run-cdc-merge.sh --report                    # only rewrite the reports
#
# The table comes from gen-iceberg-mor.sh (a pos_<pct> variant is the CDC shape). RESULTS_DIR
# overrides where the jsonl rows and the report land (default benchmarks/results).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${RESULTS_DIR:-$ROOT/benchmarks/results}"
JAVA="${JAVA_HOME:?set JAVA_HOME to a JDK 25}/bin/java"
JVM_MEM="${JVM_MEM:-8g}"
THREADS="${THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"
CP_FILE="$ROOT/benchmarks/target/classpath.txt"
if [ ! -f "$CP_FILE" ] || [ "$ROOT/benchmarks/pom.xml" -nt "$CP_FILE" ]; then
  (cd "$ROOT" && mvn -q -B -pl benchmarks dependency:build-classpath -Dmdep.outputFile="$CP_FILE" >/dev/null)
fi
CP="$ROOT/benchmarks/target/classes:$(cat "$CP_FILE")"
# COMET_JAR (as in run-tpch.sh) enables the Comet-backed configurations.
if [ -n "${COMET_JAR:-}" ]; then CP="$COMET_JAR:$CP"; fi
JVM_OPTS=(
  -Xmx"$JVM_MEM" -XX:+UseG1GC
  --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
  --sun-misc-unsafe-memory-access=allow -XX:+IgnoreUnrecognizedVMOptions
  --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED
  --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED
  --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED
  --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
  --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED
  --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED
  --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED
  ${JVM_EXTRA:-}
  -Djdk.reflect.useDirectMethodHandle=false -Dio.netty.tryReflectionSetAccessible=true
  -Dlog4j2.level=warn -Dspark.log.level=WARN
)
if [ "${1:-}" = "--report" ]; then
  "$JAVA" "${JVM_OPTS[@]}" -cp "$CP" io.vecruntime.benchmarks.CdcMergeRunner --report "$OUT"
  exit 0
fi
WAREHOUSE="${1:?usage: run-cdc-merge.sh <warehouse> <namespace.table> [configs] [extra args] | --report}"
TABLE="${2:?usage: run-cdc-merge.sh <warehouse> <namespace.table> [configs] [extra args]}"
CONFIGS="${3:-spark,vector}"
shift 3 || shift $#
for cfg in ${CONFIGS//,/ }; do
  case "$cfg" in
    comet*|hybrid) if [ -z "${COMET_JAR:-}" ]; then echo "skipping $cfg: COMET_JAR not set"; continue; fi ;;
  esac
  echo "[cdc] === $cfg ==="
  "$JAVA" "${JVM_OPTS[@]}" -cp "$CP" io.vecruntime.benchmarks.CdcMergeRunner \
    --config "$cfg" --iceberg "$WAREHOUSE" --table "$TABLE" --out "$OUT" --threads "$THREADS" "$@"
done
"$JAVA" "${JVM_OPTS[@]}" -cp "$CP" io.vecruntime.benchmarks.CdcMergeRunner --report "$OUT"
