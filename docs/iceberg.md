# Apache Iceberg merge-on-read tables

spark-vector operators run over Iceberg tables through either of the two columnar readers Spark
can plan for them. Neither needs planner changes: the rule accepts any columnar leaf with supported
types, and the Iceberg-specific work is in the input adapter seam.

| Reader | Plan node | Where deletes are merged | How our operators see the batch |
|---|---|---|---|
| Comet's native Iceberg scan (iceberg-rust) | `CometIcebergNativeScanExec` | inside the reader: positional deletes and deletion vectors become a Parquet `RowSelection`, equality deletes a row filter | plain Comet batches, adapted zero-copy by `CometVectorAdapter` |
| Iceberg's JVM vectorized reader | `BatchScanExec` | after decoding: every column is wrapped in a `ColumnVectorWithFilter` that remaps row ids through a shared `int[] rowIdMapping`; the batch reports the live count | unwrapped by `IcebergVectorAdapter.normalize` into a `SelectedColumnarBatch` over the physical rows, with the mapping turned into a selection bitmap; the Arrow buffers are read in place |

Iceberg encodes updates as delete plus insert, so on the read side merge-on-read is deletes only;
there is no positional update to scatter.

## Comet's native scan

Comet 1.0 enables the native Iceberg reader by default (`spark.comet.scan.icebergNative.enabled`).
It reads format versions 1 and 2 natively, including mixed positional and equality deletes, and
falls back to Iceberg's JVM reader for format version 3 tables (deletion vectors); newer Comet
versions read v3 natively. Row-level operations (`DELETE`, `UPDATE`, `MERGE INTO`) read the target
through the JVM reader as well, because they need the `_file`, `_pos` and `_partition` metadata
columns.

## The JVM reader adapter

`IcebergVectorAdapter` (registered reflectively, like the Comet one, so the plugin has no Iceberg
dependency) handles `IcebergArrowColumnVector`:

- Data and offset buffers are wrapped as `MemorySegment`s over Iceberg's shaded Arrow buffers.
- Iceberg disables Arrow's null checking (`arrow.enable_null_check_for_get=false`) and therefore
  does not maintain Arrow validity buffers; nulls live in a byte-per-row `NullabilityHolder`. The
  adapter builds the validity bitmap from that array (`ArrowLayout.validityFromNullBytes`) into the
  batch's scratch arena; the data buffers are still not copied.
- Dictionary-encoded strings arrive as an `IntVector` of indices plus the Parquet row-group
  dictionary. The indices are wrapped in place and the dictionary is decoded once per batch into the
  scratch arena, so the aggregate's dictionary-key paths apply. Dictionaries larger than the batch
  are declined to the copying path, which decodes per row through Iceberg's accessor.
- Anything else (constant partition columns, dictionary-encoded non-string columns, 64-bit offset
  strings) is declined and copied; the result is always correct, only slower.

Row-id-mapped batches are normalized where batches enter our operators (`VectorBatchIterator` for
filter and project, `EvalContexts.withBatch` for aggregate and sort). The wrapper owns only its
selection bitmap; Iceberg keeps owning the vectors and may reuse them for the next batch, as with
any columnar producer.

## Tests

```bash
mvn -Piceberg -pl spark verify -Dsuites=io.sparkvector.spark.iceberg.IcebergScanSuite
mvn -Pcomet,iceberg -pl spark verify -Dsuites=io.sparkvector.spark.iceberg.CometIcebergSuite
```

Both suites share `IcebergMorSuiteBase` and run the same queries; only the expected scan node
differs. The tables (`IcebergTables`) are written through Spark's Iceberg catalog into a temporary
Hadoop warehouse:

- `t_pos` (v2): `DELETE` producing positional delete files, then an `UPDATE`;
- `t_dv` (v3): the same, encoded as deletion vectors;
- `t_eq` (v2): positional deletes plus an equality delete file on the key column written with the
  Iceberg Java API (`GenericAppenderFactory.newEqDeleteWriter`, committed with `newRowDelta`), the
  shape a streaming CDC writer produces;
- `lineitem` (v2): the TPC-H shaped table with about 2% of the rows deleted, for Q1 and Q6;
- `t_merge_on` / `t_merge_off`: six further DELETE/UPDATE rounds on top of `t_pos`, then a
  `MERGE INTO` with updates, deletes and inserts from an incoming batch, run with the plugin on and
  off; the two tables must be identical afterwards and the merged table is queried with our
  operators, whose batches now carry the merge's position deletes on top of the earlier ones.

Every query is compared against Spark with the plugin disabled (`checkVectorized`), and the suites
assert the expected scan node sits directly under our operators. `IcebergScanSuite` also asserts,
through counters on the adapter, that merge-on-read batches were normalized and that fixed-width,
plain string and dictionary string columns were adapted rather than copied.

## Limitations

- The `MERGE INTO` itself is not accelerated: its row-level operator (`MergeRows`, #21) and the write
  are Spark's. The project above the target scan -- `monotonically_increasing_id()` (compiled since
  #18) beside the struct-typed `_partition` metadata column -- is no longer refused for the struct
  since #19, which passes such columns through as Spark's vectors. The reads before and after the
  merge are accelerated.
- The per-batch dictionary decode is not cached across the batches of a row group.
- Comet 1.0 reads v3 tables through the JVM reader; the adapter path above applies to them.

## Measuring merge-on-read reads: the local harness (#260)

The claim the adapter path makes -- that a delete bitmap consumed in 64-row blocks beats Spark's
`int[] rowIdMapping` indirection per column access per row -- is measured on TPC-H `lineitem` tables
carrying the delete shapes a lakehouse table has between compactions. Everything runs on a laptop
at SF1 (an evening at 5 warm-up and 10 measured iterations per cell); the 1 TB variant is #247/#249.

```
benchmarks/scripts/gen-tpch.sh 1                                 # the Parquet lineitem the variants are built from
benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1        # every variant into benchmarks/data/iceberg, namespace sf1
benchmarks/scripts/gen-iceberg-mor.sh benchmarks/data/sf1 sf1 --variants plain,pos_10,dv_10
benchmarks/scripts/run-tpch.sh benchmarks/data/sf1 spark,vector \
  --iceberg benchmarks/data/iceberg --variant sf1.pos_10 \
  --queries q1,q6,probe-count,probe-sum,probe-group --warmup 5 --iterations 10
benchmarks/scripts/run-tpch.sh --report                          # per-variant sections plus the "Iceberg merge-on-read" tables
```

`gen-iceberg-mor.sh` runs `IcebergMorGenerator` with Spark alone (no plugin): one table per variant
in a local Hadoop catalog (`local.<namespace>.<variant>`), the source rows written as 16 data files
so every delete spans all of them, then mutated with merge-on-read modes. The variants:

| variant | format | mutation |
|---|---|---|
| `plain` | v2 | none: the reader's own cost |
| `pos_<pct>` | v2 | positional deletes of `<pct>` % of the rows, scattered: picked by a hash of `l_orderkey`, so every 64-row block loses a few rows |
| `pos_<pct>_clustered` | v2 | positional deletes of whole `l_shipdate` ranges from the start of the seven years: entire blocks go, which is what `EvalContext`'s active-block skipping is for |
| `pos_upd_<pct>` | v2 | `pos_10`, then an `UPDATE` of a further `<pct>` % (deletes of the old images plus small new data files) and one `MERGE INTO` that deletes, updates and inserts: the heavily mutated shape |
| `eq_<pct>` | v2 | equality delete files on `l_orderkey` (files of at most 20000 keys, about `<pct>` % of the rows), written through the Iceberg Java API the way a CDC sink does -- Spark never writes them |
| `dv_<pct>`, `dv_<pct>_clustered`, `dv_upd_<pct>` | v3 | the `pos_*` mutations encoded as deletion vectors in Puffin files |

Next to the warehouse, `README-<namespace>.md` records what each table holds -- live rows
(`count(*)` through Spark's row path, the oracle every configuration is compared to), data and delete
files with their formats, delete rows and delete rows per data file, the snapshot id -- and the SQL
that produced it. The runner reads the current snapshot and prints its id; pin it with
`VERSION AS OF` if a table is mutated again.

`run-tpch.sh --iceberg <warehouse> --variant <namespace>.<variant>` points the `lineitem` view at that
table (the other TPC-H tables stay Parquet) and labels the dataset `iceberg:<namespace>.<variant>`,
so every configuration -- `spark`, `vector`, `comet-scan-vector-shuffle`, `comet` -- runs unchanged
over it. Besides Q1 and Q6, three full-scan probes isolate the delete cost from the operator cost:
`probe-count` (`count(*)`), `probe-sum` (`sum(l_extendedprice)`: one column, no predicate, the pure
merge cost) and `probe-group` (`count(*) GROUP BY l_returnflag`: a dictionary key through the
selection); Q6 is the selective predicate on top of a selection. Per query the `[tpch]` line and the
`.jsonl` record carry the scan operators of the plan (`BatchScanExec` for the JVM reader,
`CometIcebergNativeScanExec` for Comet's) and, from counters on the adapter, the rows the normalized
merge-on-read batches read (physical) against the rows the deletes left (live) during the last
measured run -- local mode only, the counters live in the executor JVM. One thing the scan tag
catches: on a table without delete files Iceberg answers `probe-count` from its manifests (the plan
is a `LocalTableScanExec`, no reader runs), so `plain`'s count is a metadata lookup for every
engine and only the deleted variants measure a scan there. The report's "Iceberg
merge-on-read" section lists, per query, every variant against every configuration with the speedup
versus `spark` on the same variant and versus the same configuration on `plain` (what the deletes
cost that engine), plus the live/physical share. The numbers themselves are #261 (v2) and #262 (v3).

Wide decimals (`decimal(p > 18)`) from either reader become a DECIMAL128 lane (#257). Iceberg's
reader keeps them as a `FixedSizeBinaryVector` of big-endian bytes -- as many per value as the
precision needs, twelve for `decimal(27,2)` -- which the adapter converts limb by limb (a dictionary
form is decoded once per batch); Comet's native scan hands over Arrow `Decimal128` in place.
