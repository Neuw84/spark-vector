#!/usr/bin/env bash
# Generates TPC-H lineitem as Parquet with decimals replaced by doubles, using DuckDB's tpch
# extension (brew install duckdb). Only lineitem is needed for Q1 and Q6.
#
#   benchmarks/scripts/gen-tpch.sh <scale-factor> [output-dir]
#
# Example: benchmarks/scripts/gen-tpch.sh 1 benchmarks/data
set -euo pipefail

SF="${1:?usage: gen-tpch.sh <scale-factor> [output-dir]}"
OUT="${2:-$(cd "$(dirname "$0")/.." && pwd)/data}"
DIR="$OUT/sf$SF"
mkdir -p "$DIR"

if ! command -v duckdb >/dev/null; then
  echo "duckdb not found (brew install duckdb)" >&2
  exit 1
fi

echo "Generating TPC-H SF=$SF into $DIR"
duckdb <<SQL
INSTALL tpch; LOAD tpch;
CALL dbgen(sf=$SF);
COPY (
  SELECT
    l_orderkey, l_partkey, l_suppkey, CAST(l_linenumber AS INTEGER) AS l_linenumber,
    CAST(l_quantity AS DOUBLE) AS l_quantity,
    CAST(l_extendedprice AS DOUBLE) AS l_extendedprice,
    CAST(l_discount AS DOUBLE) AS l_discount,
    CAST(l_tax AS DOUBLE) AS l_tax,
    l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate,
    l_shipinstruct, l_shipmode, l_comment
  FROM lineitem
) TO '$DIR/lineitem' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
SQL

echo "Done:"
du -sh "$DIR/lineitem"
