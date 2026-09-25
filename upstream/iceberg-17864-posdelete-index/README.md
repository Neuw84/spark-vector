# Iceberg: backport of apache/iceberg#17864 onto 1.11.0 (#20)

Position-delete indexes no longer hash and look up the data-file path for every delete row. This is fixed upstream by [apache/iceberg#17864](https://github.com/apache/iceberg/pull/17864) ("Core: Avoid per-row path rehashing in toPositionIndexes", merged to `main` on 2026-09-05; not in a release yet, since 1.11.0 predates it). This folder backports that change onto 1.11.0, for our measurement image, and records what it is worth on our workload. **No upstream issue is filed**, because the change is already on Iceberg `main`.

## What we saw

This is the same CDC `MERGE INTO` as `../iceberg-eqdelete-set-cache` (#249, #20):

- **Cluster and versions:** EKS, 8 executors × 13 cores, Spark 4.1.3, Iceberg 1.11.0.
- **Table:** a v2 `store_sales` sample with 432 M live rows in 64 data files. 20 % of the rows are deleted by position deletes and 5 % by equality deletes, 616 delete files in all.
- **Engine and patch:** spark-vector, with that folder's equality-delete set cache applied.

JFR on all 8 executors over the target scan stage (256 tasks reading all 432 M rows, 84,138 samples):

| share of the stage | where |
|---|---|
| 20.5 % | loading the position deletes (`BaseDeleteLoader.readPosDeletes` → `Deletes.toPositionIndexes`) |
| 14.0 % | Parquet read and decode |
| 11.0 % | delete filtering per batch |

Inside the position-delete load, 86 % of the samples are in `toPositionIndexes`. Of the load:

- **`CharSequenceMap.get`:** 61 %.
- **`JavaHashes.hashCode(CharSequence)`:** 38 %, part of the lookup.

OSS Spark reads through the same loader.

## The mechanism, and #17864

- **The stock loop:** in 1.11.0, `Deletes.toPositionIndexes` (run once per position delete file, its result cached per executor by file location) calls `indexes.computeIfAbsent(filePath, ...)` for every row. That wraps the path, hashes it character by character (about 150 characters for an S3 path), looks it up and compares it again.
- **Why the lookup is redundant:** the spec requires position delete files to be sorted by `file_path`, then `pos`. Consecutive rows almost always name the same data file: this benchmark's file has 64 path changes in 1.92 M rows.
- **What #17864 does:** it keeps the last path (a `toString()` copy) and its index, compares the next row's path with `String.contentEquals`, which is `String.equals` for `String` paths, and looks up only when the path changes. A path that returns later finds its existing index through the map, as before.
- **Scope:** only v2 position delete files take this path. Deletion vectors (v3) go through `BaseDeleteLoader.readDV`, one bitmap per data file with no per-row path, and are untouched.

`iceberg-17864-backport.patch` is #17864's `toPositionIndexes` body applied to `apache-iceberg-1.11.0`'s `Deletes.java`, taken verbatim from Iceberg `main`; it applies cleanly. `orig/` holds the stock file.

`TestDeletesPositionIndexes` (ours) checks the result against an index built independently of `Deletes`:

- sorted paths;
- every row a different path;
- a path that returns after others;
- paths of equal length that differ in one character.

```
mvn -q test                          # 4 tests on the backported class
mvn -q test -Dmaven.main.skip=true   # the same 4 on stock 1.11.0: same results, as they should be
```

## Measurement

`PositionIndexesBenchmark` (JMH 1.37; 3 forks × 5 warm-up × 10 measured iterations of 2 s):

- **The loop:** Iceberg 1.11.0's loop, copied verbatim, against the backported method.
- **The data:** 64 data files × 30,000 deletes, 144-character paths, a fresh `String` per row as the Parquet reader produces them.

```
mvn -q test-compile dependency:build-classpath -Dmdep.outputFile=cp.txt -Dmdep.includeScope=test
java -cp target/test-classes:target/classes:$(cat cp.txt) org.openjdk.jmh.Main PositionIndexesBenchmark
```

| order of the rows | stock 1.11.0 | #17864 backport |
|---|---|---|
| sorted by path, as the spec requires | 249.3 ± 37.4 ns / row | **35.9 ± 3.3 ns / row (6.9×)** |
| interleaved, every row a different path | 260.8 ± 2.7 ns / row | 319.5 ± 6.9 ns / row (1.23× slower) |

The stock sorted figure was noisy in this run (one fork slower); an earlier run of the same stock loop gave 209.8 ± 1.1 ns / row, which puts the speedup at 5.8×.

The interleaved case is a file that breaks the spec's order. There every row pays the failed comparison and the copy on top of the unchanged lookup. It is here to bound the cost.

`Dockerfile.test-image` builds the cluster image with **both** Iceberg patches in the runtime jar: this backport and our equality-delete set cache (`../iceberg-eqdelete-set-cache`, apache/iceberg#18257). The runs for #20 use that image from here on, for OSS Spark and spark-vector alike.
