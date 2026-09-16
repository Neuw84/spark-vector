#!/usr/bin/env bash
# Generates the eight TPC-H tables as Parquet, one directory per table, using DuckDB's tpch
# extension (brew install duckdb). Decimal columns are written as doubles: decimals wider than 18
# digits are not accelerated yet, and TPC-H's price arithmetic overflows 18 digits (see the issue
# tracker); the queries in TpchQueries are written against this schema.
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
COPY (
  SELECT
    o_orderkey, o_custkey, o_orderstatus,
    CAST(o_totalprice AS DOUBLE) AS o_totalprice,
    o_orderdate, o_orderpriority, o_clerk, CAST(o_shippriority AS INTEGER) AS o_shippriority, o_comment
  FROM orders
) TO '$DIR/orders' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    c_custkey, c_name, c_address, c_nationkey, c_phone,
    CAST(c_acctbal AS DOUBLE) AS c_acctbal,
    c_mktsegment, c_comment
  FROM customer
) TO '$DIR/customer' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    p_partkey, p_name, p_mfgr, p_brand, p_type, CAST(p_size AS INTEGER) AS p_size, p_container,
    CAST(p_retailprice AS DOUBLE) AS p_retailprice,
    p_comment
  FROM part
) TO '$DIR/part' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    ps_partkey, ps_suppkey, CAST(ps_availqty AS INTEGER) AS ps_availqty,
    CAST(ps_supplycost AS DOUBLE) AS ps_supplycost,
    ps_comment
  FROM partsupp
) TO '$DIR/partsupp' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (
  SELECT
    s_suppkey, s_name, s_address, s_nationkey, s_phone,
    CAST(s_acctbal AS DOUBLE) AS s_acctbal,
    s_comment
  FROM supplier
) TO '$DIR/supplier' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);
COPY (SELECT * FROM nation) TO '$DIR/nation' (FORMAT PARQUET, COMPRESSION SNAPPY, PER_THREAD_OUTPUT true);
COPY (SELECT * FROM region) TO '$DIR/region' (FORMAT PARQUET, COMPRESSION SNAPPY, PER_THREAD_OUTPUT true);
SQL

echo "Done:"
du -sh "$DIR"/*
