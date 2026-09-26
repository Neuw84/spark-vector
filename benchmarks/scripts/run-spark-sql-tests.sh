#!/usr/bin/env bash
# Runs Apache Spark's SQL golden-file test suite (SQLQueryTestSuite, from the spark-sql tests jar)
# with the spark-vector plugin enabled. This is deliberately not part of `mvn verify`: the whole
# suite takes about ten minutes. Run it on demand, optionally on a subset:
#
#   benchmarks/scripts/run-spark-sql-tests.sh                 # everything
#   benchmarks/scripts/run-spark-sql-tests.sh 'group-by.*'    # test cases whose name matches the regex
#   benchmarks/scripts/run-spark-sql-tests.sh '^(join|decimal)'
#   SQL_TESTS_EXCLUDE='^$' benchmarks/scripts/run-spark-sql-tests.sh   # also the excluded files
#   SQL_TESTS_UPDATE_BASELINE=true benchmarks/scripts/run-spark-sql-tests.sh   # full run; rewrite the coverage floor
#   SQL_TESTS_JVM_ARGS='-Dspark.vector.exec.sortMergeJoin.enabled=true' benchmarks/scripts/run-spark-sql-tests.sh 'join'   # a non-default configuration
#
# Excluded by default (VectorSQLQueryTestSuite.defaultExclude): explain*.sql, whose golden output
# is Spark's own physical plan, and the DataSketches files (hll, kllquantiles, thetasketch), whose
# library does not start on JDK 25.
#
# Requires JAVA_HOME pointing at JDK 25 and the plugin installed in ~/.m2 (`mvn -DskipTests install`).
# Results: spark-sql-tests/target/scalatest-reports/SparkSqlTests.txt; the last line of the run
# reports how many executions ran at least one spark-vector operator, per test case; a full run compares the
# per-case counts with src/test/resources/vector-sql-coverage.tsv and fails when a case lost acceleration (#17).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FILTER="${1:-.*}"
EXCLUDE="${SQL_TESTS_EXCLUDE:-}"
unset JAVA_TOOL_OPTIONS
: "${JAVA_HOME:?set JAVA_HOME to a JDK 25}"
cd "$ROOT"
# -Piceberg: the spark module's test sources include the Iceberg suites, which only compile with that profile.
mvn -B -Pspark-sql-tests -Piceberg -pl spark-sql-tests -am -DskipTests install -q
mvn -B -Pspark-sql-tests -pl spark-sql-tests -Dsuites=io.vecruntime.spark.sqltests.VectorSQLQueryTestSuite "-DsqlTests.filter=$FILTER" "-DsqlTests.exclude=$EXCLUDE" "-DsqlTests.jvmArgs=${SQL_TESTS_JVM_ARGS:-}" "-DsqlTests.updateBaseline=${SQL_TESTS_UPDATE_BASELINE:-false}" test
