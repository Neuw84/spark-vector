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
