---
layout: default
title: Columnar v3 deletion-vector writer (design)
---

# Columnar Iceberg v3 deletion-vector writer — design (#20, option B)

This is the design and placement decision for the columnar deletion-vector (DV) writer, and the
record of slice 1 (the profile split that gates the work). The writer itself lands in slices; this
document is the map, and `spark.vector.iceberg.dvWriter.enabled` (default off) is the switch it will
hang from.

## What it does

On a **format-version 3** merge-on-read table whose delete file format is PUFFIN
(`SparkPositionDeltaWrite.Context.useDVs()`), the DELETE half of `DELETE` / `UPDATE` / `MERGE INTO`
is written columnarly:

- our Arrow batches already carry `_file` and `_pos` as lanes, and the REBALANCE exchange Spark
  plans for the delta write already clusters rows by `_spec_id`, `_partition`, `_file`;
- from runs of equal `_file` we fill one `PositionDeleteIndex` (a `RoaringPositionBitmap`) per data
  file and make one `BaseDVFileWriter.delete(path, index, spec, partition)` call per file;
- `close()` merges any previous DV for that file itself (`loadPreviousDeletes` + `positions.merge`),
  serialises each bitmap into one Puffin blob, and returns a `DeleteWriteResult`;
- the commit stays **Iceberg's own** `RowDelta` (its `DeltaWrite` / `DeltaBatchWrite`) — we do not
  reimplement commit semantics, snapshot summaries, or previous-DV bookkeeping.

Inserts and the insert half of updates stay on Iceberg's data writer. **v2 tables and anything
unsupported decline to Spark's `WriteDeltaExec` unchanged**, with a printed fallback reason, like
every other rule in the plugin.

## Slice 1 — the profile split (done)

The design's gain estimate was a range because the split of the write stage into delete-writing vs
data-writing was unverified. `DvWriteProfileSuite` (spark module, `-Piceberg`) pins it locally:
Spark's own row writer, plugin off, a format-version-3 merge-on-read table of 400,000 rows, the two
halves isolated on the same table shape.

| phase | what it writes | task-time |
|---|---|---|
| D — pure `DELETE` (`i % 3 = 0`) | deletion vectors only (32 DV files, **0 data files**) | 2,188 ms |
| I — pure `INSERT` (append, same order of magnitude of rows) | Parquet data files only | 1,659 ms |
| M — the real CDC `MERGE` (delete + update + insert) | both halves | 1,432 ms |

**Delete-writer share of the isolated write halves: D / (D + I) = 56.9 %.**

Reading of the number, honestly:

- The delete half is a **majority** of the write work on this shape, not a negligible share, so
  option B clears its own stop condition ("if the delete writer is a small share, stop and prefer
  the AQE packing fix / a columnar data writer instead"). We build.
- This local 56.9 % is consistent with, not the same as, the cluster's "~27 % of MERGE JFR samples":
  that 27 % is the position-delta writer as a share of the **whole** MERGE (scan + join + write),
  while 56.9 % is the delete half as a share of the **write stage's two halves only**.
- The MERGE phase's absolute task-time (1,432 ms) is *below* either isolated half because its delete
  set (only the `d2 < 0` matched keys) is far smaller than phase D's `i % 3 = 0`. The D/I ratio is
  the defensible per-half attribution; the MERGE row is context, not a third data point to compare
  head-to-head.
- Local, 4 shuffle partitions, one JVM: this pins the *share*, not a wall-clock speedup. The owner's
  cluster benchmark on a v3 DV variant is what will turn the share into a MERGE-level number; the
  design's honest headline stays low-to-mid-teens percent on the MERGE, v3 tables only.

## Placement (§2.E of the scope): the injection seam

Two facts decide it, both verified against the jars on the classpath (Spark 4.1.3, Iceberg 1.11.0):

1. **`WriteDeltaExec` is a `V2CommandExec`, not a columnar operator.** It is executed eagerly
   (`executeCollect` / `run()`), and `injectColumnar` / `ColumnarRule.preColumnarTransitions` never
   visits it — the columnar transition pass runs over query plans, not the command tree. So the
   plugin's existing `VectorColumnarRule` seam, which is where `MergeRowsExec` is intercepted, is the
   **wrong** seam for the write. Its per-row work lives in a `WritingSparkTask` that pulls
   `InternalRow`s and calls Iceberg's `DeltaWriter.delete(row)` / `insert(row)` per row.

2. **The DV writer types are public.** `org.apache.iceberg.deletes.BaseDVFileWriter`,
   `DVFileWriter`, and `PositionDeleteIndex` (with `delete(pos)`, `delete(posLo, posHi)`, `merge`,
   `serialize`, `deserialize`) are all **public** API. Only the assembly of `DeleteWriteResult` into
   `SparkPositionDeltaWrite`'s package-private `DeltaTaskCommit`, and the writer factory wiring, are
   package-private.

Therefore the seam is a **planner strategy** injected with
`SparkSessionExtensions.injectPlannerStrategy`, matching the logical `WriteDelta`
(`org.apache.spark.sql.catalyst.plans.logical.WriteDelta`, a `RowLevelWrite`) when the target uses
DVs and the write is DV-eligible; it produces a `VectorWriteDeltaExec` command whose `child` is the
(columnar) plan producing the delta rows. When the table is v2, or the shape is unsupported, the
strategy returns `Nil` and Spark's own `DataSourceV2Strategy` plans the ordinary `WriteDeltaExec` —
the fallback is Spark's real writer, untouched.

**Module: an optional `iceberg-bridge`, compiled against Iceberg, like the optional shuffle module.**
The plugin core reaches Iceberg only reflectively (`IcebergVectorAdapter` resolves every Iceberg
class by name and the plugin has no compile-time Iceberg dependency). The DV writer needs to *call*
Iceberg's writer factories and, for the commit result, reach one or two package-private members; a
separate module compiled against `iceberg-spark-runtime` (loaded only when Iceberg is on the
classpath) keeps that dependency out of the core and is far more testable than a second large block
of reflection. This mirrors the shuffle module (`spark-vector-shuffle`, kept apart so the core never
carries gRPC) and the reader adapter's reflective registration. Reflection was the alternative
(§2.E); it is rejected here because the write path touches enough Iceberg surface
(`BaseDVFileWriter`, `OutputFileFactory`, `PartitionSpec`/`StructLike` routing, the commit-result
assembly) that a compiled module is the more robust and testable of the two, at the cost of one more
optional artifact.

## Slices (option B path)

1. **Profile split** — *done, this document + `DvWriteProfileSuite`.*
2. **`iceberg-bridge` module skeleton** — optional module compiled against Iceberg; handles to
   `BaseDVFileWriter`, `PositionDeleteIndex`, `OutputFileFactory`, and the commit-result assembly.
   Unit test: build an empty DV, commit via `RowDelta`, read back with Spark.
3. **Columnar DV build** — from a batch's `_file`/`_pos` lanes, per run of equal `_file`, fill a
   `PositionDeleteIndex`. Differential test: our bitmap equals Iceberg's per-row-built index.
4. **`VectorWriteDeltaExec` (v3, delete-only first)** — the planner strategy, gated on `useDVs()`;
   inserts still via Iceberg's appender. Declines to Spark's writer for v2.
5. **Insert/update path via Iceberg's appender** — data rows through Iceberg's existing appender so
   full MERGE works on v3; verify commit and row lineage.

Not in option B: a columnar Parquet **data** writer, and v2 position-delete files (option A).

## Configuration

`spark.vector.iceberg.dvWriter.enabled` (session `SQLConf`, boolean, default `false`). Off while the
writer is landed in slices; flips to `true` once every Iceberg merge-on-read suite is byte-identical
with it on and off. See `docs/configuration.md`.
