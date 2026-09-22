# The cluster image (#246/#247): Spark 4.1.x on JDK 25 -- the kernels are compiled for release 25 with
# the incubator Vector API, so the official Spark images (JDK 17/21) cannot be used -- carrying
# benchmarks.jar built from this repository, the Comet Spark 4.1 jar, the Iceberg Spark 4.1 runtime,
# the S3A connector with its AWS SDK bundle, Spark's spark-sql tests jar (the TPC-DS query texts and
# TPCDSSchema the runners read from the classpath) and tpcds-kit's dsdgen for TpcdsGenRunner.
#
# Everything is fetched from the Apache archive, Maven Central and GitHub: the build needs no file
# from the machine that starts it, so it runs on the cluster itself (benchmarks/k8s/kaniko-build.yaml)
# with this repository as the build context.
#
#   docker build -f benchmarks/k8s/Dockerfile -t <registry>/spark-vector/spark:<tag> .
#
# The layout is Spark's own (/opt/spark, /opt/entrypoint.sh, user 185) so the spark-operator and
# spark-submit on Kubernetes drive it exactly like the official image.

ARG SPARK_VERSION=4.1.3
ARG HADOOP_VERSION=3.4.3
ARG AWS_SDK_V2_VERSION=2.35.4
ARG COMET_VERSION=1.0.0
ARG ICEBERG_VERSION=1.11.0

# ---- build: our jars from this repository, dsdgen from tpcds-kit ------------------------------------
FROM eclipse-temurin:25-jdk AS build
RUN apt-get update && apt-get install -y --no-install-recommends maven git make gcc libc6-dev bison flex ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . /src
# The shaded benchmarks.jar carries io.sparkvector:* (kernels, spark, shuffle, benchmarks); Spark, Comet
# and Iceberg are provided by the image's jars directory.
RUN mvn -B -q -Pcomet,iceberg -DskipTests -pl benchmarks -am install
# The Flight shuffle's runtime jars: the shaded benchmarks.jar carries only io.sparkvector:*, and Spark's
# distribution has Arrow and guava but no flight-core, gRPC or protobuf-java (the Netty classes are Spark's).
RUN mvn -B -q -pl shuffle dependency:copy-dependencies -DincludeScope=runtime -DoutputDirectory=/src/flight-jars \
      -DincludeArtifactIds=flight-core,grpc-api,grpc-core,grpc-context,grpc-netty,grpc-protobuf,grpc-protobuf-lite,grpc-stub,grpc-util,perfmark-api,proto-google-common-protos,protobuf-java,protobuf-java-util,animal-sniffer-annotations,error_prone_annotations,annotations,j2objc-annotations,jspecify \
 && ls /src/flight-jars
RUN git clone --depth 1 https://github.com/databricks/tpcds-kit /opt/tpcds-kit \
    && make -C /opt/tpcds-kit/tools OS=LINUX LINUX_CFLAGS="-O3 -std=gnu89 -Wno-implicit-int -Wno-implicit-function-declaration -Wno-return-mismatch" -j"$(nproc)" >/dev/null

# ---- runtime -----------------------------------------------------------------------------------------
FROM eclipse-temurin:25-jre
ARG SPARK_VERSION
ARG HADOOP_VERSION
ARG AWS_SDK_V2_VERSION
ARG COMET_VERSION
ARG ICEBERG_VERSION

# The packages Spark's own Kubernetes Dockerfile installs (tini for PID 1, gosu for the entrypoint's
# user switch, krb5/nss for the Hadoop client, procps/net-tools for diagnostics).
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash tini libc6 libpam-modules krb5-user libnss3 procps net-tools gosu curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid=185 spark && useradd --system --uid=185 --gid=spark spark

RUN curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
      | tar -xz -C /opt \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop3" /opt/spark \
    && cp /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt/entrypoint.sh \
    && cp /opt/spark/kubernetes/dockerfiles/spark/decom.sh /opt/decom.sh \
    && chmod a+x /opt/entrypoint.sh /opt/decom.sh \
    && rm -rf /opt/spark/examples /opt/spark/data \
    && cd /opt/spark/jars \
    && rm -f hadoop-client-api-3.4.*.jar hadoop-client-runtime-3.4.*.jar \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-api/${HADOOP_VERSION}/hadoop-client-api-${HADOOP_VERSION}.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-runtime/${HADOOP_VERSION}/hadoop-client-runtime-${HADOOP_VERSION}.jar" \
    && ls hadoop-client-*.jar
# (The Hadoop swap above: Spark 4.1.3 bundles Hadoop 3.4.2, whose UserGroupInformation calls
# Subject.getSubject and fails on JDK 24+ -- HADOOP-19212, fixed in 3.4.3 -- so the shaded client
# jars are replaced, as the root pom does for the local harness; in the extraction's own layer so the
# removal cannot be lost to a snapshot that misses deletions.)

# The jars the runners need beyond the distribution: S3A (hadoop-aws + the AWS SDK v2 bundle it is
# built against), the S3A magic committer binding (spark-hadoop-cloud), Comet, the Iceberg runtime,
# and Spark's spark-sql tests jar.
RUN cd /opt/spark/jars \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/${AWS_SDK_V2_VERSION}/bundle-${AWS_SDK_V2_VERSION}.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/datafusion/comet-spark-spark4.1_2.13/${COMET_VERSION}/comet-spark-spark4.1_2.13-${COMET_VERSION}.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-4.1_2.13/${ICEBERG_VERSION}/iceberg-spark-runtime-4.1_2.13-${ICEBERG_VERSION}.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/${ICEBERG_VERSION}/iceberg-aws-bundle-${ICEBERG_VERSION}.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/spark/spark-sql_2.13/${SPARK_VERSION}/spark-sql_2.13-${SPARK_VERSION}-tests.jar" \
    && curl -fsSLO "https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.13/${SPARK_VERSION}/spark-hadoop-cloud_2.13-${SPARK_VERSION}.jar"

COPY --from=build /src/benchmarks/target/benchmarks.jar /opt/spark/jars/benchmarks.jar
COPY --from=build /src/flight-jars/*.jar /opt/spark/jars/
COPY --from=build /opt/tpcds-kit/tools/dsdgen /opt/tpcds-kit/tools/tpcds.idx /opt/tpcds-kit/tools/

ENV SPARK_HOME=/opt/spark
WORKDIR /opt/spark/work-dir
RUN chmod g+w /opt/spark/work-dir && chown -R spark:spark /opt/spark/work-dir /opt/tpcds-kit \
    && chmod a+rx /opt/tpcds-kit/tools/dsdgen
# The executor's class path from entrypoint.sh carries its conf directory (on Kubernetes a mounted
# ConfigMap with spark.properties), its working directory (where the executor fetches the job's jars)
# and empty entries (the working directory again), and the JDK's AOT cache / CDS refuses a non-empty
# directory on the class path (#416). The executor reads its properties through --properties-file
# and the job's jars through its own class loader: its class path is the jars directory alone. The
# driver's class path is left as the distribution builds it.
RUN sed -i '/^  executor)$/{n;s|^    shift 1$|    shift 1\n    SPARK_CLASSPATH="$SPARK_HOME/jars/*"; SPARK_DIST_CLASSPATH="$SPARK_HOME/jars/*"|}' /opt/entrypoint.sh \
    && grep -q 'SPARK_CLASSPATH="$SPARK_HOME/jars/\*"; SPARK_DIST_CLASSPATH' /opt/entrypoint.sh
USER spark
ENTRYPOINT ["/opt/entrypoint.sh"]
