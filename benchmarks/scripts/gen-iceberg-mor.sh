#!/usr/bin/env bash
# Builds the Iceberg merge-on-read variants of TPC-H lineitem for the MoR harness (#260) from the
# Parquet tables gen-tpch.sh produced: one Iceberg table per variant in a local Hadoop catalog, plus
# a README with what each holds (live rows, delete files, delete rows per data file, snapshot id).
# Spark alone writes them -- the plugin is not loaded -- so the README's counts are the oracle.
#
#   benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1                 # every variant into benchmarks/data/iceberg, namespace sf1
#   benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf10 sf10           # namespace sf10
#   benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1 sf1 --variants plain,pos_10,dv_10
#   ICEBERG_WAREHOUSE=/data/iceberg benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1
#
# Variants: plain; pos_<pct> / pos_<pct>_clustered (v2 positional deletes, scattered by order key or
# whole shipdate ranges); pos_upd_<pct> (pos_10 plus an UPDATE and a MERGE INTO); eq_<pct> (v2
# equality delete files on l_orderkey, written through the Iceberg Java API); dv_<pct>,
# dv_<pct>_clustered, dv_upd_<pct> (v3 deletion vectors). Then run them with
#   benchmarks/scripts/run-tpch.sh benchmarks/data/sf1 spark,vector --iceberg benchmarks/data/iceberg --variant sf1.pos_10 --queries q1,q6,probe-count,probe-sum,probe-group
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${1:?usage: gen-iceberg-mor.sh <tpch-parquet-dir> [namespace] [--variants a,b,c] [--files N]}"
shift
NS="${1:-}"
if [ -n "$NS" ] && [[ "$NS" != --* ]]; then shift; else NS="$(basename "$DATA")"; fi
WAREHOUSE="${ICEBERG_WAREHOUSE:-$ROOT/benchmarks/data/iceberg}"
JAVA="${JAVA_HOME:?set JAVA_HOME to a JDK 25}/bin/java"
JVM_MEM="${JVM_MEM:-6g}"
THREADS="${THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"
CP_FILE="$ROOT/benchmarks/target/classpath.txt"
if [ ! -f "$CP_FILE" ] || [ "$ROOT/benchmarks/pom.xml" -nt "$CP_FILE" ]; then
  (cd "$ROOT" && mvn -q -B -pl benchmarks dependency:build-classpath -Dmdep.outputFile="$CP_FILE" >/dev/null)
fi
CP="$ROOT/benchmarks/target/classes:$(cat "$CP_FILE")"
# The same JVM flags as run-tpch.sh: Spark 4 on JDK 25 needs the opens, the kernels the vector module.
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
mkdir -p "$WAREHOUSE"
"$JAVA" "${JVM_OPTS[@]}" -cp "$CP" io.vecruntime.benchmarks.IcebergMorGenerator \
  --data "$DATA" --warehouse "$WAREHOUSE" --namespace "$NS" --threads "$THREADS" "$@"
