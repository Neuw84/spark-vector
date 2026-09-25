# Upstream report: cache the merged equality-delete set on executors

Iceberg tracks work in GitHub issues (https://github.com/apache/iceberg/issues) and takes PRs against
`main`; a Core change like this is titled `Core: ...`. The text below is the issue, and the PR
description follows from it. Numbers are filled in from the measurement in `README.md` before filing.

---

**Title:** Core: equality-delete sets are rebuilt in every scan task even with the executor cache on

**Affects:** 1.11.0, `main` (checked 2026-09-25). Any engine going through `BaseDeleteLoader`; measured on Spark 4.1.

**Description**

`DeleteFilter.applyEqDeletes()` calls `DeleteLoader.loadEqualityDeletes(deleteFiles, projection)` once per scan task (per group of equality field ids). `BaseDeleteLoader.loadEqualityDeletes` creates a new `StructLikeSet` and inserts every row of every applicable delete file into it. The Spark executor cache (`spark.sql.iceberg.executor-cache.*`, apache/iceberg#8755) caches each equality delete file's rows, keyed by the file location, but not the merged set. So on a table whose equality deletes span the key range -- the usual shape for Flink upserts and CDC sinks, where every data file is covered by the same delete files -- every task of a scan re-inserts the same rows into a fresh hash set of boxed rows.

With millions of equality-delete rows that rebuild dominates the scan. On a Spark 4.1 `MERGE INTO` over a v2 table with 432 M live rows, 64 data files and 616 delete files (5 % of rows removed by equality deletes on one `long` column), JFR on all executors shows ~40 % of the MERGE's wall time in equality-delete application, with `HashMap.putVal` (23.9 % of all samples), `HashMap.resize` (9.4 %) and `Iterators.addAll` (7.6 %) at the top of the stack and the probe (`HashMap.getNode`, 5.5 %) far below.

**Proposal**

Cache the merged set through the loader's existing `canCache`/`getOrLoad` hooks, keyed by the projection and the sorted delete file locations. Tasks with exactly the same delete files share one set; any difference in the file list (sequence numbers exclude deletes from newer data files) gives a different key. When the merged set is cached, its files are read directly instead of through their per-file entries (a cache load should not start other cache loads, and the merged set supersedes them); when it is not cacheable, behaviour is unchanged. The size handed to the cache is the sum of the per-file estimates, the convention `estimateEqDeletesSize` already uses, so the existing `max-entry-size` / `max-total-size` limits decide as they do today. `StructLikeSet.contains` is safe for concurrent readers (`ThreadLocal` wrapper), and the set is not mutated after it is built.

**Results** (same cluster, 8 × m5.4xlarge; same table and change batch; cold one-shot `MERGE INTO` of 42.0 M updates, 11.0 M deletes and 4.6 M inserts; Spark 4.1.3; `spark.sql.iceberg.executor-cache.max-entry-size` / `max-total-size` raised to 2 / 4 GiB so the merged set fits, on both runs): **162.1 s on 1.11.0, 74.6 s with the change (2.17×)**. The two scan stages that apply the equality deletes drop from 118 s and 96 s to 10 s and 32 s; the join and write stages are unchanged. Both runs leave the same 425,665,996 rows.

**Open question for reviewers:** the per-file size estimate (`recordCount × estimated record size`) is well below the heap a `StructLikeSet` actually takes (wrappers and `HashMap` nodes). With the merged set cached that gap matters more, since one entry now holds the whole set. The opposite problem shows too: with the default limits (64 MB per entry, 128 MB total) this table's merged set (~11 M keys, ~88 MB by the estimate) would not be cached at all, so the gain above needed the limits raised. A more realistic per-row overhead in the estimate, a separate limit for merged sets, or larger defaults may be preferable; happy to follow the maintainers' preference.
