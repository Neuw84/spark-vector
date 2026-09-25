# Iceberg: the merged equality-delete set is rebuilt in every scan task (#20)

## What we saw

The CDC `MERGE INTO` from #249 on the EKS cluster (8 executors × 14 cores, Spark 4.1.3, Iceberg
1.11.0): a ~20 GB `store_sales` sample as a v2 table, 432 M live rows in 64 data files, 20 % of the
rows deleted by position deletes and 5 % by equality deletes on `ss_ticket_number` (616 delete files
in all), and a 10 % change batch. One MERGE takes 172 s under spark-vector and 165 s under OSS Spark.

JFR on all 8 executors: about **40 % of the MERGE** (~70 of 178 s) is Iceberg applying the equality
deletes, and most of that is *building* the delete set, not probing it. Top-of-stack samples in the
scan stages:

| frame | share of all samples |
|---|---|
| `java.util.HashMap.putVal` | 23.9 % |
| `java.util.HashMap.resize` | 9.4 % |
| `...collect.Iterators.addAll` | 7.6 % |
| `java.util.HashMap.getNode` (the probe) | 5.5 % |
| `JavaHashes.hashCode(CharSequence)`, `StructLikeSet.add` | 4.6 %, 4.5 % |

OSS Spark reads through the same reader, so it pays the same cost.

## The mechanism

1. Every scan task builds its own `SparkDeleteFilter` (`BaseReader`), and its first batch calls
   `DeleteFilter.applyEqDeletes()` → `BaseDeleteLoader.loadEqualityDeletes(files, projection)` once
   per group of equality field ids.
2. `loadEqualityDeletes` creates a **new** `StructLikeSet` and `Iterables.addAll`s every row of every
   applicable delete file into it.
3. The executor cache (`CachingDeleteLoader`, `spark.sql.iceberg.executor-cache.*`) caches each
   delete file's rows individually -- `getOrReadEqDeletes` keys on the file location -- but **not the
   merged set**. So every task re-inserts every equality-delete row into a fresh boxed hash set.

A table with equality deletes spread over the whole key range (the usual shape: Flink upserts, CDC
sinks) gives every data file the same delete files, so every task of the scan rebuilds an identical
set. With millions of delete rows, the rebuild is most of the task.

The code is the same on Iceberg `main` (checked 2026-09-25), and no open PR touches it.

## The change

`iceberg-eqdelete-set-cache.patch` (against `apache-iceberg-1.11.0`, applies cleanly; the sources
are also here as files):

- `BaseDeleteLoader.loadEqualityDeletes` caches the **merged** set through the loader's existing
  `canCache` / `getOrLoad` hooks, under a key naming the projection and the **sorted** delete file
  locations, so any two tasks with exactly the same delete files share one set. Tasks whose delete
  files differ in any file (sequence numbers exclude some deletes from some data files) get their
  own set.
- When the merged set is cached, its files are read directly rather than through their per-file
  entries: a cache load must not start other cache loads, and the merged set supersedes them.
  When it is not cacheable (too large for `max-entry-size`, or caching off), the behaviour is
  exactly today's: per-file entries, a set per task.
- The size given to the cache is the sum of the per-file estimates (`recordCount × record size`),
  the convention `estimateEqDeletesSize` already uses. Like it, this is far below the real heap of a
  `StructLikeSet` (boxed wrappers, `HashMap` nodes); the Spark executor cache's defaults (64 MB per
  entry, 128 MB total) then decide what is cached, as they do today.
- `StructLikeSet.contains` keeps its wrapper in a `ThreadLocal`, so concurrent tasks can probe the
  shared set; nothing writes to it after it is built.

`TestBaseDeleteLoaderEqualityDeleteCache` (Iceberg's data module, its own fixtures): same files in
any order share one set; a different file set or different equality fields get their own; no
caching builds a set per call; a merged set too large for the cache falls back to per-file entries.

```
mvn -q test                          # 5 tests, pass on the patched class
mvn -q test -Dmaven.main.skip=true   # the same tests on stock 1.11.0: 3 of 5 fail, as they should
```

## Measurement

`Dockerfile.test-image` builds a cluster image identical to the one the numbers above were taken
with, except for the patched class in Iceberg's runtime jar. The measurement runs the same cold,
MERGE-only one-shot as #249 for OSS Spark and spark-vector, stock and patched, with
`spark.sql.iceberg.executor-cache.max-entry-size` / `max-total-size` raised so the merged set fits.

Results (2026-09-25; cold MERGE-only one-shot, 8 × m5.4xlarge, the #249 table and change batch):

| engine | stock Iceberg 1.11.0 | patched | speedup |
|---|---|---|---|
| OSS Spark 4.1.3 | 162.1 s | **74.6 s** | **2.17×** |
| spark-vector (columnar shuffle) | 172.2 s | 133.6 s | 1.29× |

Every run leaves 425,665,996 rows. Per stage (Spark stock → patched), the two scan stages that apply
the equality deletes go from 118 / 96 s to 10 / 32 s; the join (≈22 s) and the write (≈19 s) are
unchanged. The table's checksums were not captured by these runs (the image predates the runner's
own results upload); the unit test and the local SF1 runs (identical checksums, stock vs patched,
both engines) cover correctness.

spark-vector gains less because the change exposes costs of its own that the equality-delete work
used to hide (#20): its scan-side stages take 62 / 47 s against Spark's 10 / 32 s, and AQE coalesces
its write stage to 20 tasks where Spark's has 44 (47 s against 18 s).

`ICEBERG-ISSUE.md` is the text for the upstream issue and PR description.
