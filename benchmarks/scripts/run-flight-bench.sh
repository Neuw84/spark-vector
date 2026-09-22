#!/usr/bin/env bash
# The shuffle transport benchmark (FlightShuffleBenchmark): two Arrow Flight servers and a reduce task
# in one JVM, over map files the real writer produced. Spark is a 'provided' dependency of the
# benchmarks module, so unlike the kernel microbenchmarks this one cannot run from benchmarks.jar
# alone: the classpath is built as run-tpch.sh builds it.
#
#   benchmarks/scripts/run-flight-bench.sh                       # the default matrix, JSON to benchmarks/results
#   benchmarks/scripts/run-flight-bench.sh -p partitions=1000 -p strings=high -p chunkBytes=1048576,4194304,16777216
#   COALESCE_ROWS=4096 benchmarks/scripts/run-flight-bench.sh    # the reader's coalescing threshold (a JVM property)
#
# An A/B is two runs on two checkouts and a diff of the JSON files (jmh -rf json). Host noise on a
# shared box is several percent: compare against a measured band, not a fixed number.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JAVA="${JAVA_HOME:?set JAVA_HOME to a JDK 25}/bin/java"
OUT="${RESULTS_DIR:-$ROOT/benchmarks/results}"
mkdir -p "$OUT"

CP_FILE="$ROOT/benchmarks/target/classpath.txt"
if [ ! -f "$CP_FILE" ] || [ "$ROOT/benchmarks/pom.xml" -nt "$CP_FILE" ]; then
  (cd "$ROOT" && mvn -q -B -pl benchmarks dependency:build-classpath -Dmdep.outputFile="$CP_FILE" >/dev/null)
fi
CP="$ROOT/benchmarks/target/classes:$(cat "$CP_FILE")"

JVM_OPTS=(
  -Xmx"${JVM_MEM:-4g}" -XX:+UseG1GC
  --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
  --sun-misc-unsafe-memory-access=allow -XX:+IgnoreUnrecognizedVMOptions
  --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED
  --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED
  -Dio.netty.tryReflectionSetAccessible=true -Dlog4j2.level=warn
)
# The forked benchmark JVMs get the same options; the reader's coalescing threshold rides along as a property.
FORK_OPTS="${JVM_OPTS[*]}"
if [ -n "${COALESCE_ROWS:-}" ]; then FORK_OPTS="$FORK_OPTS -Dsparkvector.shuffle.reader.coalesceRows=$COALESCE_ROWS"; fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec "$JAVA" "${JVM_OPTS[@]}" -cp "$CP" org.openjdk.jmh.Main FlightShuffle \
  -jvmArgs "$FORK_OPTS" -rf json -rff "$OUT/flight-bench-$STAMP.json" "$@"
