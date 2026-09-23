#!/bin/bash
# The executor's JVM options and class path for the JDK AOT cache (#416), shared by train.sh (the
# build-time smoke test) and assemble.sh (the in-cluster cache). The JVM only uses a cache whose class
# path and module options match its own, so this mirrors what the executor gets:
#   - Spark's launcher module options (JavaModuleOptions.defaultModuleOptions(), the same list the
#     launcher prepends to every driver and executor JVM), read from the Spark on the class path;
#   - the flags submit-cluster.sh sets through spark.executor.extraJavaOptions (JVM_FLAGS there);
#   - the class path entrypoint.sh gives an executor in this image: the explicit sorted jar list in
#     /opt/spark/aot/classpath, as SPARK_CLASSPATH and again as SPARK_DIST_CLASSPATH (the sed in the
#     Dockerfile). Not the directory wildcard: the JVM expands that in directory order, which is not
#     the same on the build's filesystem and the container's overlay, and the cache checks the order.
# Heap and system properties are not part of the match and are free.
#
# Sets: JAVA, CP, MODULE_OPTS (array), JVM_FLAGS (array), AOT_BASE_OPTS (array: the options every
# executor JVM in this image starts with).
SPARK_HOME="${SPARK_HOME:-/opt/spark}"
JAVA="${JAVA_HOME:-/opt/java/openjdk}/bin/java"
LIST="$(cat "${CLASSPATH_FILE:-$SPARK_HOME/aot/classpath}")"
CP="$LIST:$LIST"

mapfile -t MODULE_OPTS < <("$JAVA" -cp "$CP" io.sparkvector.benchmarks.AotTraining --module-options)
# Keep in step with JVM_FLAGS in benchmarks/scripts/submit-cluster.sh.
JVM_FLAGS=(--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow
  --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED
  --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED
  --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED
  --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED
  --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED
  -Djdk.reflect.useDirectMethodHandle=false -Dio.netty.tryReflectionSetAccessible=true)
AOT_BASE_OPTS=(-Djava.net.preferIPv6Addresses=false -XX:+IgnoreUnrecognizedVMOptions "${MODULE_OPTS[@]}" "${JVM_FLAGS[@]}")
