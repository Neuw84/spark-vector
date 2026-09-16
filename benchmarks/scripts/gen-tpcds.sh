#!/usr/bin/env bash
# Generates the 24 TPC-DS tables as Parquet, one directory per table, using DuckDB's tpcds
# extension (brew install duckdb). The schema is Spark's own TPCDSSchema (the one the approved plans
# quoted by the per-query issues were produced with): money columns DECIMAL(7,2), which fit the
# plugin's INT64 lanes (sums and products over them are where the wide-decimal issues show up, by
# design), dates DATE, identifiers and counts INT -- DuckDB's dsdgen emits them as BIGINT, so they
# are cast on the way out -- and customer.c_last_review_date under Spark's name (dsdgen appends _sk).
# The seven fact tables are written per thread so Spark parallelises their scans; each dimension is
# a single file.
#
#   benchmarks/scripts/gen-tpcds.sh <scale-factor> [output-dir]
#
# Example: benchmarks/scripts/gen-tpcds.sh 1     # benchmarks/data/tpcds-sf1/<table>/*.parquet (~1.2 GB)
#          benchmarks/scripts/gen-tpcds.sh 10    # benchmarks/data/tpcds-sf10 (~12 GB; dsdgen takes a few minutes)
set -euo pipefail

SF="${1:?usage: gen-tpcds.sh <scale-factor> [output-dir]}"
OUT="${2:-$(cd "$(dirname "$0")/.." && pwd)/data}"
DIR="$OUT/tpcds-sf$SF"
mkdir -p "$DIR"

if ! command -v duckdb >/dev/null; then
  echo "duckdb not found (brew install duckdb)" >&2
  exit 1
fi

FACTS="store_sales store_returns catalog_sales catalog_returns web_sales web_returns inventory"
DIMS="call_center catalog_page customer customer_address customer_demographics date_dim household_demographics
      income_band item promotion reason ship_mode store time_dim warehouse web_page web_site"

# The SELECT list of a table with Spark's types: dsdgen's schema (an empty catalog is enough) read
# once, BIGINT columns cast to INTEGER, c_last_review_date_sk renamed.
SCHEMA="$(duckdb -csv -noheader -c "INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf=0);
  SELECT table_name, column_name, data_type FROM information_schema.columns ORDER BY table_name, ordinal_position;")"
columns() {
  echo "$SCHEMA" | awk -F, -v t="$1" '$1 == t {
    col = $2; expr = $2;
    if ($3 == "BIGINT") expr = "CAST(" $2 " AS INTEGER)";
    if (col == "c_last_review_date_sk") col = "c_last_review_date";
    printf "%s%s AS %s", (n++ ? ", " : ""), expr, col }'
}

echo "Generating TPC-DS SF=$SF into $DIR"
{
  echo "INSTALL tpcds; LOAD tpcds;"
  echo "CALL dsdgen(sf=$SF);"
  for t in $FACTS; do
    echo "COPY (SELECT $(columns "$t") FROM $t) TO '$DIR/$t' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576, PER_THREAD_OUTPUT true);"
  done
  for t in $DIMS; do
    mkdir -p "$DIR/$t"
    echo "COPY (SELECT $(columns "$t") FROM $t) TO '$DIR/$t/$t.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 1048576);"
  done
} | duckdb

echo "Done:"
du -sh "$DIR"/*
