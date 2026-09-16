#!/usr/bin/env bash
# Generates the eight TPC-H tables as Parquet, one directory per table, using DuckDB's tpch
# extension (brew install duckdb). By default decimal columns are written as doubles: decimals wider
# than 18 digits are not accelerated yet, and TPC-H's price arithmetic overflows 18 digits (see the
# issue tracker). With --decimals they keep DuckDB's native TPC-H types (DECIMAL(15,2)) and land in
# a parallel sf<N>-decimal directory, so both schemas can be measured against each other. The
# queries in TpchQueries run unchanged against either.
#
#   benchmarks/scripts/gen-tpch.sh <scale-factor> [output-dir] [--decimals]
#
# Example: benchmarks/scripts/gen-tpch.sh 1 benchmarks/data              # benchmarks/data/sf1
#          benchmarks/scripts/gen-tpch.sh 1 benchmarks/data --decimals   # benchmarks/data/sf1-decimal
set -euo pipefail

SF="${1:?usage: gen-tpch.sh <scale-factor> [output-dir] [--decimals]}"
shift
DECIMALS=0
OUT=""
for arg in "$@"; do
  case "$arg" in
    --decimals) DECIMALS=1 ;;
    *) OUT="$arg" ;;
  esac
done
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
if [ "$DECIMALS" = 1 ]; then
  DIR="$OUT/sf$SF-decimal"
  # DuckDB's dbgen types: DECIMAL(15,2) for every money and quantity column.
  NUM() { echo "$1"; }
else
  DIR="$OUT/sf$SF"
  NUM() { echo "CAST($1 AS DOUBLE)"; }
fi
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
    $(NUM l_quantity) AS l_quantity,
    $(NUM l_extendedprice) AS l_extendedprice,
    $(NUM l_discount) AS l_discount,
    $(NUM l_tax) AS l_tax,
    l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate,
    l_shipinstruct, l_shipmode, l_comment
  FROM lineitem
) TO '$DIR/lineitem' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    o_orderkey, o_custkey, o_orderstatus,
    $(NUM o_totalprice) AS o_totalprice,
    o_orderdate, o_orderpriority, o_clerk, CAST(o_shippriority AS INTEGER) AS o_shippriority, o_comment
  FROM orders
) TO '$DIR/orders' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    c_custkey, c_name, c_address, c_nationkey, c_phone,
    $(NUM c_acctbal) AS c_acctbal,
    c_mktsegment, c_comment
  FROM customer
) TO '$DIR/customer' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    p_partkey, p_name, p_mfgr, p_brand, p_type, CAST(p_size AS INTEGER) AS p_size, p_container,
    $(NUM p_retailprice) AS p_retailprice,
    p_comment
  FROM part
) TO '$DIR/part' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    ps_partkey, ps_suppkey, CAST(ps_availqty AS INTEGER) AS ps_availqty,
    $(NUM ps_supplycost) AS ps_supplycost,
    ps_comment
  FROM partsupp
) TO '$DIR/partsupp' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    s_suppkey, s_name, s_address, s_nationkey, s_phone,
    $(NUM s_acctbal) AS s_acctbal,
    s_comment
  FROM supplier
) TO '$DIR/supplier' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (SELECT * FROM nation) TO '$DIR/nation' (FORMAT PARQUET, COMPRESSION SNAPPY, PER_THREAD_OUTPUT true);
COPY (SELECT * FROM region) TO '$DIR/region' (FORMAT PARQUET, COMPRESSION SNAPPY, PER_THREAD_OUTPUT true);
SQL

echo "Done:"
du -sh "$DIR"/*
