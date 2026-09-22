#!/bin/bash
# Produces the executor's AOT cache at image build (#416): a training run of AotTraining under the
# JVM options and class path the executor will start with, one-step (-XX:AOTCacheOutput records the
# configuration and assembles the cache in a child JVM). The JVM only uses a cache whose class path
# and module options match its own, so this script mirrors what the executor gets:
#   - Spark's launcher module options (JavaModuleOptions.defaultModuleOptions(), the same list the
#     launcher prepends to every driver and executor JVM), read from the Spark on the class path;
#   - the flags submit-cluster.sh sets through spark.executor.extraJavaOptions (JVM_FLAGS there);
#   - the class path entrypoint.sh gives an executor in this image: the jars directory, wildcard,
#     as SPARK_CLASSPATH and again as SPARK_DIST_CLASSPATH (see the sed in the Dockerfile).
# Heap and system properties are not part of the match and are free.
#
#   train.sh <output .aot> [--rounds N] [--rows N]
set -euo pipefail
OUT="${1:?output .aot path}"; shift
SPARK_HOME="${SPARK_HOME:-/opt/spark}"
JAVA="${JAVA_HOME:-/opt/java/openjdk}/bin/java"
CP="$SPARK_HOME/jars/*:$SPARK_HOME/jars/*"

mapfile -t MODULE_OPTS < <("$JAVA" -cp "$CP" io.sparkvector.benchmarks.AotTraining --module-options)
# Keep in step with JVM_FLAGS in benchmarks/scripts/submit-cluster.sh.
JVM_FLAGS=(--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow
  --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED
  --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED
  --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED
  --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED
  --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED
  -Djdk.reflect.useDirectMethodHandle=false -Dio.netty.tryReflectionSetAccessible=true)

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
LOG="${OUT%.aot}.log"
"$JAVA" -Djava.net.preferIPv6Addresses=false -XX:+IgnoreUnrecognizedVMOptions "${MODULE_OPTS[@]}" "${JVM_FLAGS[@]}" \
  -Xmx"${TRAIN_HEAP:-4g}" -XX:MaxDirectMemorySize=2g -Dlog4j2.level=warn -Dspark.log.level=WARN \
  -XX:AOTCacheOutput="$OUT" -Xlog:aot=info:file="$LOG" \
  -cp "$CP" io.sparkvector.benchmarks.AotTraining "$@"

# The build fails if the cache holds no AOT-linked class: that is the incubator-module trap the jre
# stage exists to avoid (the boot layer is not archived when jdk.incubator.vector is an incubator).
LINKED=$(grep -a "aot-linked" "$LOG" | grep -a "instance classes" | sed 's/.*aot-linked = *\([0-9]*\).*/\1/' | head -1)
echo "AOT cache: $(du -h "$OUT" | cut -f1), aot-linked instance classes = ${LINKED:-0}"
[ "${LINKED:-0}" -gt 0 ]
