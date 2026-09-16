#!/usr/bin/env bash
# Runs Apache Spark's SQL golden-file test suite (SQLQueryTestSuite, from the spark-sql tests jar)
# with the spark-vector plugin enabled. This is deliberately not part of `mvn verify`: the whole
# suite takes on the order of an hour. Run it on demand, optionally on a subset:
#
#   benchmarks/scripts/run-spark-sql-tests.sh                 # everything
#   benchmarks/scripts/run-spark-sql-tests.sh 'group-by.*'    # test cases whose name matches the regex
#   benchmarks/scripts/run-spark-sql-tests.sh '^(join|decimal)'
#
# Requires JAVA_HOME pointing at JDK 25 and the plugin installed in ~/.m2 (`mvn -DskipTests install`).
# Results: spark-sql-tests/target/scalatest-reports/SparkSqlTests.txt; the last line of the run
# reports how many executions ran at least one spark-vector operator.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FILTER="${1:-.*}"
unset JAVA_TOOL_OPTIONS
: "${JAVA_HOME:?set JAVA_HOME to a JDK 25}"
cd "$ROOT"
mvn -B -Pspark-sql-tests -pl spark-sql-tests -am -DskipTests install -q
mvn -B -Pspark-sql-tests -pl spark-sql-tests -Dsuites=io.sparkvector.spark.sqltests.VectorSQLQueryTestSuite "-DsqlTests.filter=$FILTER" test
