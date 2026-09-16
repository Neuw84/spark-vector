#!/usr/bin/env bash
# Runs TPC-H Q1/Q6 under several Spark configurations, one JVM each, then writes the markdown and
# HTML reports (benchmarks/results/results.{md,html}) from every measurement recorded so far.
#
#   benchmarks/scripts/run-tpch.sh <data-dir> [configs] [extra TpchRunner args...]
#   benchmarks/scripts/run-tpch.sh --report          # only rewrite the reports
#
#   data-dir  directory containing lineitem/ (see gen-tpch.sh); its basename (sf1, sf10) names
#             the dataset section in the reports
#   configs   comma-separated subset of: spark,vector,comet-scan,comet-scan-vector,comet-scan-vector-shuffle,comet
#             (default: spark,vector plus the Comet configs when COMET_JAR is set)
#
# Environment:
#   JAVA_HOME   JDK 25
#   COMET_JAR   path to comet-spark-spark4.1_2.13-<version>.jar (enables the Comet configurations)
#   JVM_MEM     heap for the local Spark JVM (default 6g)
#   THREADS     local[N] parallelism (default: all cores)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${1:?usage: run-tpch.sh <data-dir> [configs] [extra args] | run-tpch.sh --report}"
shift
CONFIGS="${1:-}"
if [ -n "${CONFIGS}" ]; then shift; fi
if [ "$DATA" = "--report" ]; then
  CONFIGS=""
elif [ -z "$CONFIGS" ]; then
  CONFIGS="spark,vector"
  if [ -n "${COMET_JAR:-}" ]; then CONFIGS="$CONFIGS,comet-scan,comet-scan-vector,comet-scan-vector-shuffle,comet"; fi
fi

JAVA="${JAVA_HOME:?set JAVA_HOME to a JDK 25}/bin/java"
JVM_MEM="${JVM_MEM:-6g}"
THREADS="${THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"
OUT="$ROOT/benchmarks/results"
mkdir -p "$OUT"

# Classpath: benchmark classes + every dependency (Spark is 'provided', so this includes it).
CP_FILE="$ROOT/benchmarks/target/classpath.txt"
if [ ! -f "$CP_FILE" ] || [ "$ROOT/benchmarks/pom.xml" -nt "$CP_FILE" ]; then
  (cd "$ROOT" && mvn -q -B -pl benchmarks dependency:build-classpath -Dmdep.outputFile="$CP_FILE" >/dev/null)
fi
CP="$ROOT/benchmarks/target/classes:$(cat "$CP_FILE")"
if [ -n "${COMET_JAR:-}" ]; then CP="$COMET_JAR:$CP"; fi

# Same JVM options a spark-submit deployment would use.
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
  -Djdk.reflect.useDirectMethodHandle=false -Dio.netty.tryReflectionSetAccessible=true
  -Dlog4j2.level=warn -Dspark.log.level=WARN
)

LIST=()
if [ -n "$CONFIGS" ]; then IFS=',' read -ra LIST <<< "$CONFIGS"; fi
for cfg in ${LIST[@]+"${LIST[@]}"}; do
  case "$cfg" in
    comet*) if [ -z "${COMET_JAR:-}" ]; then echo "skipping $cfg: COMET_JAR not set"; continue; fi ;;
  esac
  echo "=== $cfg"
  "$JAVA" "${JVM_OPTS[@]}" -cp "$CP" io.sparkvector.benchmarks.TpchRunner \
    --config "$cfg" --data "$DATA" --threads "$THREADS" --out "$OUT" "$@"
done

"$JAVA" "${JVM_OPTS[@]}" -cp "$CP" io.sparkvector.benchmarks.TpchRunner --report "$OUT"
