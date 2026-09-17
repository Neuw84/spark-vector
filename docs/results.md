# Benchmark results

Measured on an Apple M3 Pro (11 cores: 5 performance + 6 efficiency, 18 GB), macOS 27, OpenJDK
25.0.4, Spark 4.1.3, Arrow 18.3.0, Comet 1.0.0 built from source. NEON gives the Vector API
128-bit vectors: 4 int lanes, 2 long/double lanes. Numbers on an AVX-512 machine would look
different (8 double lanes per operation, native `compress`), and the kernels take those paths
automatically; they have not been measured there.

Every number below is reproducible with the commands in the README; the JMH text report and the
`benchmarks/results/*.jsonl` files are the raw sources.

## Kernel microbenchmarks (JMH)

`-wi 2 -i 3 -w 1 -r 1 -f 1`, throughput in elements per millisecond, one thread, batches of 8192
(4096 for grouped aggregation). `reference` is the scalar loop in
`io.sparkvector.kernels.reference.ScalarReference`, the same code the unit tests use as oracle.

### Compare (column `<` literal, column `<` column) → selection bitmap

| type | scalar `<` reference | scalar `<` SIMD | speedup | column `<` reference | column `<` SIMD | speedup |
|---|---:|---:|---:|---:|---:|---:|
| INT32 | 535k | 4683k | 8.7x | 532k | 3598k | 6.8x |
| INT64 | 529k | 2294k | 4.3x | 538k | 1678k | 3.1x |
| FLOAT64 | 563k | 2256k | 4.0x | 525k | 1569k | 3.0x |

The int32 numbers double the 64-bit ones because 128-bit vectors hold twice as many lanes; the
kernels are memory-bound at this point.

### Compaction (gather selected rows into a dense output)

| type | selectivity | nulls | reference | SIMD | speedup |
|---|---:|---|---:|---:|---:|
| INT32 | 2% | no | 2434k | 22897k | 9.4x |
| INT32 | 50% | no | 966k | 1957k | 2.0x |
| INT32 | 98% | no | 1720k | 3009k | 1.7x |
| INT32 | 50% | yes | 333k | 1304k | 3.9x |
| FLOAT64 | 2% | no | 2330k | 25282k | 10.9x |
| FLOAT64 | 50% | no | 1054k | 2430k | 2.3x |
| FLOAT64 | 98% | no | 1694k | 1691k | 1.0x |
| FLOAT64 | 50% | yes | 330k | 1659k | 5.0x |

Two lessons are baked into these numbers. Selection words that are all ones (the common case for a
98% filter) are bulk-copied, which is why 98% is faster than 50% for both types. For 64-bit
elements the two-lane `rearrange` on NEON was slower than a scalar walk over the set bits, so that
is what the kernel does when the species has two lanes; the doubles row at 98% is now a memcpy race
that both sides tie. The int32 kernel learnt the same lesson later (phase 4): a word with at most
16 selected rows is walked bit by bit instead of shuffled lane group by lane group, which took the
2% row from 8445k to 22897k (10%: 3592k to 11209k, 25%: 2386k to 3230k, 50%: unchanged). A Q6
profile found `compactInt32` on the one date column costing three times the compaction of the
three double columns; that is where this came from.

### Ungrouped reductions

| kernel | null fraction | reference | SIMD | speedup |
|---|---:|---:|---:|---:|
| sum(double) | 0 | 1361k | 3010k | 2.2x |
| sum(double) | 1% | 1496k | 2331k | 1.6x |
| sum(double) | 30% | 1896k | 1813k | 1.0x |
| min(double) | 0 | 1696k | 3667k | 2.2x |
| min(double) | 1% | 929k | 2322k | 2.5x |
| sum(long) | 0 | 3701k | 3677k | 1.0x |

Summing longs gains nothing: C2 already vectorizes the scalar loop. Summing doubles does, because
Java forbids reassociating floating-point additions in the scalar loop while the kernel keeps two
lanes of partial sums (this also means the kernel's rounding differs from Spark's in the last bits,
which is why results are compared to four decimals). Once a third of the rows are null the masked
path costs as much as it saves on NEON.

### Grouped aggregation: masked reductions versus scatter

| groups | one masked reduction per group | scatter into accumulators |
|---:|---:|---:|
| 1 | 2667k | 1268k |
| 4 | 539k | 845k |
| 16 | 252k | 1370k |

This benchmark decided the design. The first version ran a masked SIMD reduction per group for up
to 64 groups, which is 1.6x slower than the scatter at TPC-H Q1's 4 groups and 5x slower at 16: each
group re-reads the whole batch, and with 2 lanes a masked reduction is barely faster than a scalar
loop. The threshold is now 1 group on ≤4-lane species (8 on wider ones, untested).

The scatter itself is a dependent `sum[g] += x` chain whenever consecutive rows hit the same group.
Rotating over `sparkvector.agg.interleave` independent accumulator copies (default 4):

| groups | 1 copy | 2 copies | 4 copies |
|---:|---:|---:|---:|
| 1 | 1340k | 1095k | 1181k |
| 4 | 832k | 959k | 1166k |
| 16 | 1291k | 1143k | 1203k |

+40% at the target case, -10% where the chain was not the bottleneck (one group: every row hits
the same slot anyway; 16 groups: the load-store forwarding stalls are already rare). Copies are
summed on read, so double sums round in a different order than Spark's sequential loop; on TPC-H
Q1 the results differ from Spark's in the 12th significant digit. `spark.vector.exec.strictFloatingPoint`
(default on, like Comet's `spark.comet.exec.strictFloatingPoint` but with the opposite default) uses
one copy and sequential reductions for double sums and reproduces Spark's rounding bit for bit, at
7% of aggregate kernel time (2.5% of Q1 at SF10); the benchmarks set it to `false`.

### Bitmap popcount

`Long.bitCount` per 64-bit word versus testing each bit, on a 4096-bit bitmap: 57272 vs 451
bitmaps per ms (127x). Included as a sanity check that bitmap bookkeeping never shows up in
profiles, which held in every JFR recording taken for this document.

### Group-key assignment over plain strings: on-the-fly dictionary for any length, with a cap

`GroupKeyTableBenchmark`, 64 batches of 4096 plain (non-dictionary) UTF8 keys into one
`GroupKeyTable`, rows per ms. Measured on an x86 host (Xeon 8488C, AVX-512, shared, so ±10-20%
noise) rather than the M3 the rest of this document uses; the before/after ratio is the point.
"Before" encodes only values of ≤8 bytes and bails on a batch holding a longer one; "after" encodes
any length (64-bit fingerprint plus byte compare) and stops encoding a column for good once its
dictionary passes `sparkvector.agg.plainDictMaxEntries` (default 512).

| distinct values | 8-byte keys, before | after | 24-byte keys, before | after |
|---:|---:|---:|---:|---:|
| 25 | 50417 | 50494 | 39106 | 54089 |
| 100 | 60128 | 59028 | 39260 | 47365 |
| 1000 | 32403 | 35347 | 32883 | 32604 |
| 4000 | 16865 | 31361 | 29871 | 28617 |
| 10000 | 13804 | 27528 | 25621 | 26321 |
| 200000 | 3604 | 15013 | 13228 | 12990 |

Two things the numbers decided. The dictionary is worth having for long keys only at low
cardinality (+38% at 25 distinct, the `n_name` / `l_returnflag` shape; +21% at 100), because the
memoised path resets a memo of one slot per combination on every batch and misses more of them as
combinations approach the batch size: with the earlier 65536 cap the same change was 2x *slower*
at 4000-10000 distinct 24-byte keys. And the cap is a win in its own right for the 8-byte keys the
old code already encoded: with it, 4000 and 10000 distinct keys run 1.9-2x faster and 200k
distinct 4x, because they leave the memo path for hash-and-compare instead of growing a
dictionary nobody benefits from. 512 sits at the crossover, where both key lengths are at parity
with hash-and-compare.

## TPC-H Q1 and Q6, scale factors 1 and 10

`lineitem` generated by DuckDB (decimals as doubles; SF1: 6,001,215 rows, 207 MB of Parquet in 11
files of one row group; SF10: 59,986,052 rows, 2.1 GB, 11 files, 65 row groups), one local JVM per
configuration with
`local[8]`, 8 GB heap, `spark.sql.shuffle.partitions=8`. SF1: 10 warm-up and 10 measured runs per
query; SF10: 5 and 7. Median wall-clock time of `collect()` in milliseconds, speedup against plain
Spark in parentheses. All configurations returned identical results to 10 significant digits (4
decimals in the phase 1 table, before the interleaved accumulators). `benchmarks/results/results.html`
is the same data as bar charts with p90 whiskers, regenerated by every benchmark run.

### Phase 4: columnar Sort, native range-partitioned shuffle (both scales, one session)

`VectorSortExec` replaces the final `Sort`, and the range-partitioned exchange above the Final
aggregate goes through Comet's native shuffle like the hash-partitioned one. With Comet's scan and
shuffle the Q1 plan is now `CometNativeScan`, our filter, project and both aggregates, two native
exchanges fed through the bridge, our sort, and one `ColumnarToRow` for `collect()`: every operator
is accelerated. The decimal and join work that landed in the same build is not exercised by these
two queries.

| dataset | query | spark | vector | comet-scan | comet-scan-vector | comet-scan-vector-shuffle | comet |
|---|---|---:|---:|---:|---:|---:|---:|
| sf10 | q1 | 1105.7 (1.00x) | 674.6 (1.64x) | 809.8 (1.37x) | 623.2 (1.77x) | 599.5 (1.84x) | 684.1 (1.62x) |
| sf10 | q6 | 245.6 (1.00x) | 286.8 (0.86x) | 210.4 (1.17x) | 220.1 (1.12x) | 227.4 (1.08x) | 179.4 (1.37x) |
| sf1 | q1 | 205.2 (1.00x) | 158.7 (1.29x) | 166.5 (1.23x) | 132.0 (1.55x) | 135.4 (1.52x) | 148.0 (1.39x) |
| sf1 | q6 | 62.9 (1.00x) | 82.6 (0.76x) | 57.6 (1.09x) | 59.6 (1.06x) | 60.1 (1.05x) | 53.5 (1.18x) |

Against phase 3 (same machine, same day) every SF10 median is within 4%, including the
configurations whose plans did not change (`spark` 1123 to 1106 ms, `comet` 707 to 684 ms), so
the sort and the native range exchange are a plan-shape improvement, not a speed one: at SF10 that
exchange carries four rows per partition and the sort orders four rows. Their kernel time in the
`[tpch]` output is under a millisecond. The gain they do bring is that nothing in the pipeline
converts our batches to rows any more before the final `collect()`; on a query whose sort input is
large, that is where it would show.

### Phase 3: on-the-fly dictionaries for plain string keys (both scales, one session)

| dataset | query | spark | vector | comet-scan | comet-scan-vector | comet-scan-vector-shuffle | comet |
|---|---|---:|---:|---:|---:|---:|---:|
| sf10 | q1 | 1122.9 (1.00x) | 711.0 (1.58x) | 833.2 (1.35x) | 603.1 (1.86x) | 596.6 (1.88x) | 706.5 (1.59x) |
| sf10 | q6 | 242.9 (1.00x) | 288.6 (0.84x) | 203.0 (1.20x) | 222.8 (1.09x) | 235.1 (1.03x) | 175.5 (1.38x) |
| sf1 | q1 | 216.0 (1.00x) | 158.9 (1.36x) | 166.6 (1.30x) | 126.5 (1.71x) | 135.4 (1.60x) | 151.7 (1.42x) |
| sf1 | q6 | 64.3 (1.00x) | 84.7 (0.76x) | 64.3 (1.00x) | 59.8 (1.08x) | 62.0 (1.04x) | 54.4 (1.18x) |

### Scale factor 10, phase 2 code (first SF10 run, before the dictionary fix)

| query | spark | vector | comet-scan | comet-scan-vector | comet-scan-vector-shuffle | comet |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 1200.0 (1.00x) | 742.0 (1.62x) | 919.6 (1.30x) | 802.6 (1.50x) | 807.7 (1.49x) | 709.6 (1.69x) |
| q6 | 269.1 (1.00x) | 331.6 (0.81x) | 217.7 (1.24x) | 242.4 (1.11x) | 243.8 (1.10x) | 188.1 (1.43x) |

### Scale factor 1, phase 2: selection vectors, Final aggregate, Comet shuffle

| query | spark | vector | comet-scan | comet-scan-vector | comet-scan-vector-shuffle | comet |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 212.8 (1.00x) | 178.6 (1.19x) | 168.9 (1.26x) | 164.2 (1.30x) | 157.9 (1.35x) | 152.9 (1.39x) |
| q6 | 63.3 (1.00x) | 97.3 (0.65x) | 60.4 (1.05x) | 68.2 (0.93x) | 71.6 (0.88x) | 54.8 (1.16x) |

### Phase 1: Partial aggregate only, compaction between operators

| query | spark | vector | comet-scan | comet-scan-vector | comet |
|---|---:|---:|---:|---:|---:|
| q1 | 219.4 (1.00x) | 185.5 (1.18x) | 185.3 (1.18x) | 181.0 (1.21x) | 158.3 (1.39x) |
| q6 | 71.5 (1.00x) | 79.5 (0.90x) | 63.5 (1.13x) | 67.1 (1.07x) | 59.2 (1.21x) |

Configurations:

- `spark`: Spark 4.1.3 unchanged, whole-stage code generation.
- `vector`: this plugin over Spark's vectorized Parquet reader (batches copied into native
  Arrow-layout memory once per operator chain); in phase 2 both aggregate stages are ours, with
  Spark's row shuffle and a `RowToColumnarExec` in between.
- `comet-scan`: Comet's native DataFusion Parquet scan, every Comet operator disabled, Spark codegen
  for the rest.
- `comet-scan-vector`: Comet's scan read zero-copy by this plugin's operators.
- `comet-scan-vector-shuffle` (phase 2): as above, plus Comet's native shuffle between our partial
  and Final aggregates, fed through the Arrow C Data bridge. Up to phase 3 the final `Sort` and the
  row conversion above our Final aggregate were Spark's; since phase 4 the sort is ours and the
  range-partitioned exchange below it is native too, so the only Spark node left in Q1 is the
  `ColumnarToRow` that `collect()` needs.
- `comet`: Comet end to end (scan, filter, project, both aggregates, columnar shuffle).

Run-to-run noise is about ±10% on the medians, and more for the 60 ms queries (the machine is a
laptop with efficiency cores and no core pinning; the phase 2 table was measured after two hours of
benchmarking and a back-to-back `spark`/`vector` re-run of Q6 gave 86 vs 105 ms, 0.82x). Differences
below that are not meaningful. The phase 3 rows were measured in one session on a machine with
nothing else running; two earlier attempts with a video call in the background produced medians
up to 2x off (and one `vector` JVM whose Q6 ran at 755 ms, three times its usual) and were
discarded, so the numbers in `benchmarks/results/*.jsonl` include those outliers with earlier
timestamps. The phase 4 rows likewise: a first attempt overlapped another Spark test job on the
same machine and put `spark` Q1 at 1451 ms; it was discarded and the run repeated once the
machine was idle. Check the `spark` baseline against its previous value before reading any other
column.

### What the numbers say

Scale factor 10 is the more informative table. At SF1 a query is 60-200 ms of wall clock, of which
a fixed share is job scheduling, task launch and the 8-partition shuffle, and that share does not
shrink with the plugin; at SF10 the kernels are the bulk of the time and the picture changes:

- `vector` over Spark's own scan is 1.58x on Q1 at SF10 against 1.36x at SF1 (1.19x in the phase
  2 session). The plugin's operators do the same work per row at both scales, so this is the fixed
  overhead being diluted; 1.6x is the honest number for "Spark's Parquet reader plus JVM SIMD
  operators versus Spark's generated code" on this query.
- Over Comet's scan the same operators are 1.86x, and with Comet's shuffle 1.88x: faster than
  Comet end to end (1.59x) by 15%. Comet's scan is the better scan (`comet-scan` alone is 1.35x),
  the zero-copy import makes the filter 6x cheaper than over Spark's on-heap batches, and the rest
  is the aggregate, where the JVM kernels hold their own against DataFusion's. At SF1 the same
  ordering holds (1.71x versus 1.42x).
- The first SF10 run had `comet-scan-vector` at 1.50x, slower than `vector` at 1.62x, while it
  was faster at SF1. A JFR recording of that configuration explained it: Comet's native scan hands
  over `l_returnflag` and `l_linestatus` as plain Arrow strings, where Spark's reader keeps them
  dictionary encoded. Our grouped aggregate memoises group ids per combination of dictionary
  indices, so over Spark's scan it never compared a string; over Comet's it hashed and compared
  every one of the 60M rows, and a third of the aggregate's time was `MemorySegment.mismatch`
  set-up for one-byte strings (the filter, meanwhile, was 6x cheaper on Comet's batches: 87 ms of
  kernel time against 555 ms, the on-heap copy). The fix is in `GroupKeyTable`: plain UTF8 keys
  whose values fit in 8 bytes are dictionary encoded on the fly against a per-column dictionary
  kept across batches (single-byte values through a 256-entry direct table), and the memoised
  path applies. Aggregate kernel time summed over the 8 tasks went from 5368 ms to 3843 ms, and
  the same change (plus walking selection words instead of testing the bitmap per row) took
  `vector`'s from 4636 ms to 3959 ms. The phase 3 table is the rerun: `comet-scan-vector` went
  from 803 to 603 ms.
- Comet's native shuffle makes no measurable difference on Q1 at either scale (1.88x versus
  1.86x): the shuffle carries 4 groups per partition. It is there for the plan shape, not for the
  bytes.
- Q6 stays a loss for `vector` (0.84x, the same ratio the SF1 back-to-back re-run gave). Fed by
  Comet's scan our operators beat plain Spark (1.09x) but not Comet's scan under Spark's codegen
  (1.20x): the copy out of Spark's on-heap vectors is gone, the full-column evaluation of a 1.9%
  predicate is not. Comet's own filter is 1.38x. The reasons are in the next paragraphs.

The SF1 table reads as follows.

Q1 is the case the plugin was built for: a filter that keeps 98.6% of the rows, four derived double
columns, and a grouped aggregate with 8 aggregate functions over 4 groups. Over Spark's own scan
the JVM SIMD path is 1.19x faster than Spark's generated code. Over Comet's scan, with Comet's
native shuffle carrying our partial buffers and our Final aggregate reading Comet's shuffle output,
it is 1.35x, within 3% of the fully native Comet pipeline (1.39x). Per-operator kernel time summed
over the 8 tasks in the `vector` configuration was 33 ms in the filter (down from 163 ms in phase 1:
it no longer compacts, it forwards a selection bitmap) and 382 ms in the aggregate (up from 207 ms:
it now reads every physical row through validity masks, and inherits the compaction's memory
traffic), for a query that is 7 ms faster overall. The Parquet scan is most of the remaining time
and is identical in `spark` and `vector`.

Q6 is the case it was not built for, and it loses. The predicate touches three columns and keeps
1.9% of the rows. Spark's generated loop evaluates the `l_shipdate` range first and skips the other
two comparisons for 85% of the rows; the vectorized filter evaluates the comparisons over full
columns. Phase 2 added block skipping (later conjuncts are only evaluated for 64-row blocks with an
undecided row) but with survivors scattered uniformly through the file almost every block still has
one, so the kernel time stayed at 95 ms (12 ms of wall clock over 8 threads). Over Comet's zero-copy
scan the same operators cost 43 ms of kernel time, which puts a number on the batch adaptation from
Spark's `OnHeapColumnVector`: for a selective filter the copy is worth more than the compares. The
selection policy compacts when fewer than half the rows survive, so Q6's aggregate sees dense
batches; an earlier run that forwarded the 2% selection made the projection and aggregate walk all
6M rows (37 ms of kernel time between them instead of 2 ms).

Q6 at SF10 against Comet end to end (phase 4, 267 versus 210 ms over the same native scan), from
JFR recordings of `vector`, `comet-scan-vector`, `comet` and `comet-scan`:

- The plan shapes are the same. Comet's `CometColumnarToRowExec` is the row conversion of the one
  result row for `collect()`, and ours has it in the same place; both pipelines are columnar to
  the top.
- Comet's native scan pushes the scan's `dataFilters` into DataFusion whenever
  `spark.sql.parquet.filterPushdown` is on, regardless of which operator sits above it, so our
  configuration gets the same row-group, page-index and bloom-filter pruning as Comet's. None of it
  helps here: `l_shipdate` is uniformly spread, so every row group and page has survivors.
- Comet's second level, `spark.comet.parquet.rowFilterPushdown.enabled` (DataFusion's
  `pushdown_filters`, late materialisation), is off by default and measured as a loss for
  everyone on this query: our filter's kernel time drops to 12 ms because the scan hands it 2% of
  the rows, but the scan takes twice as long (`comet-scan-vector` 267 to 563 ms, `comet` 210 to
  562 ms, `comet-scan` 252 to 604 ms). With survivors on every page the reader's second pass over
  the projected columns costs more than decoding them once.
- What remains is per-row filter throughput: three compares over 60M rows and the compaction of
  the survivors. The compaction half was the int32 kernel above (filter kernel time 472 to 365 ms
  summed over tasks); the compare half is the Vector API at two double lanes against DataFusion's
  native loops at the same NEON width. Spark's codegen over the same scan (`comet-scan`, 252 ms)
  sits between the two, evaluating `l_shipdate` first and skipping the rest for 85% of rows.

Things that were true before profiling and are not any more:

1. The first end-to-end run had `vector` at 0.61x on Q1. The grouped aggregate was running masked
   reductions per group (see the JMH table above); the fix was the threshold change.
2. Strings were being decoded from Parquet dictionaries row by row three times: once by Spark's
   `getUTF8String`, once when the filter compacted them into plain Arrow `VarCharVector`s, and once
   more when the aggregate hashed and compared every row. Dictionary indices now flow through the
   adapter, the filter (which compacts int32 indices and copies the small dictionary) and a
   `VectorDictionaryColumnVector` that Spark's row conversion can still read, and the aggregate
   memoises group ids per combination of dictionary indices. That halved the aggregate's time.
3. Compacting doubles through a two-lane shuffle table cost more than the filter's comparisons
   (274 of 653 samples in one profile were the vector stores). Full selection words are now a bulk
   copy and partial words a scalar walk.
4. (Phase 2) Compaction between our own operators. A filter feeding our aggregate now forwards a
   selection bitmap; the aggregate folds it into its validity masks and group assignment. Q1's
   filter went from 163 ms to 33 ms of kernel time; the rows are read once, by the aggregate.
5. (Phase 2) Spark's Final aggregate and row shuffle after our partial aggregate. The Final is ours
   now, and with Comet configured the shuffle is Comet's native one, fed zero copy through the Arrow
   C Data Interface. That closed most of the gap to Comet on Q1 (1.21x to 1.35x versus Comet's
   1.39x); the rest is Comet's native Sort and its row conversion.
6. (SF10) The dictionary memoisation in the grouped aggregate only applied when the scan delivered
   dictionaries. Comet's native scan does not, so at SF10 the zero-copy configuration lost to the
   copying one until short plain strings were dictionary encoded inside the aggregate.

### The Comet bridge

Comet's shaded Arrow and Spark's unshaded Arrow are the same version and cannot share a class.
What they share is memory: `VectorToCometExec` writes an `ArrowArray` and `ArrowSchema` struct per
column with the FFM API, hands the addresses to Comet's `ArrowImporter` (reflectively; Comet keeps
`org.apache.arrow.c.*` unshaded but with shaded signatures, so `arrow-c-data` cannot be on our
classpath and a relocated copy would break its JNI lookups) and gets a `CometVector` over our
buffers back. An upcall stub receives Comet's release call and drops our references; the test
suite checks that every export is released. The bridge carries 4 groups x 8 partitions in Q1, so
its cost does not show in the table; its value is that Comet's writer, reader and native Final
aggregate stage boundaries are available to a JVM operator.

### Where the remaining time is

- Adaptation of Spark's on-heap vectors. Reading Spark's `double[]` in place as a heap
  `MemorySegment` was tried and made the kernels twice as slow (the Vector API's heap-segment
  path is not intrinsified as well as the native one), so the batch is copied into native memory
  once per operator chain; Q6 shows that copy is about half of the filter's cost -- and on Q6 the
  copy is a *dictionary decode*, since Parquet writers dictionary-encode its predicate columns (see
  "Q6 revisited" below: 20% of the JVM's samples, four times the comparisons). Comet's scan
  avoids it, and so would a Parquet reader of our own that writes Arrow memory directly.
- Selective predicates with scattered survivors. Block skipping needs whole 64-row blocks to be
  dead; Q6's survivors are spread over 98% of the blocks. Evaluating the second conjunct only on
  the surviving rows (a gather, or a compaction of the operands) is the remaining option, and on
  2-lane species the gather is a scalar loop. Comet's late-materialising reader is not the answer
  either: measured 2x slower for Comet itself on this data (see the Q6 notes above).
- The grouped aggregate reads every physical row of a selected batch through validity masks. With
  98% selectivity that is the right trade; a middle ground (10-50% survivors) would want the
  aggregate to walk the selection bits instead of the validity words, which the scatter loops
  already do when nulls are present.
- Without Comet, the shuffle is Spark's row shuffle, with a `ColumnarToRowExec` above the partial
  aggregate and a `RowToColumnarExec` below the Final. A columnar shuffle of our own would remove
  both; Comet's is the shortcut taken here.
- The `ColumnarToRow` above the final operator is the one Spark node left in Q1 with Comet's scan
  and shuffle; it exists because `collect()` wants rows. Without Comet, the global sort still sits
  above Spark's row shuffle and stays Spark's (the rule only converts a sort over a columnar
  child), so the columnar shuffle above is what would make that configuration fully columnar too.
- Group keys longer than 8 bytes that arrive as plain strings still take the hash-and-compare
  path per row; a wider packed key (two longs) or hashing the batch's distinct offsets first
  would extend the on-the-fly dictionary to them.

## Real TPC-H decimals versus doubles, scale factor 1

The benchmark data has always had prices, discounts and quantities as doubles (`gen-tpch.sh`),
because TPC-H's price arithmetic leaves 18 digits. `gen-tpch.sh 1 <dir> --decimals` now writes the
same tables with DuckDB's native `DECIMAL(15,2)` into `sf1-decimal`, and the runner records every
operator the planner rule declined to convert with its reason (`fallbacks` in the `.jsonl`, listed
under each query in the reports). This is what the 22 queries look like over both schemas, `spark`
against `vector`, medians of 3 runs after 1 warm-up, `local[8]`, 6 GB heap.

Hardware caveat: an 8-core x86 host (Xeon 8488C, AVX-512) shared with other work, not the M3 the
rest of this document uses, so the absolute medians are noisy (±10%) and the double-schema speedups
below do not match the M3 tables above. Read the decimal columns against the double columns of the
same row, and read the accelerated-operator counts and the reason list, which are deterministic.

| query | doubles: spark | vector (speedup) | accel. | decimals: spark | vector (speedup) | accel. |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 472 | 361 (1.31x) | 4/7 | 1939 | 1991 (0.97x) | 2/7 |
| q2 | 475 | 490 (0.97x) | 17/44 | 538 | 456 (1.18x) | 17/44 |
| q3 | 538 | 547 (0.98x) | 6/16 | 540 | 598 (0.90x) | 5/16 |
| q4 | 496 | 511 (0.97x) | 5/15 | 464 | 456 (1.02x) | 5/15 |
| q5 | 794 | 884 (0.90x) | 9/31 | 822 | 955 (0.86x) | 8/31 |
| q6 | 130 | 160 (0.81x) | 4/5 | 160 | 149 (1.07x) | 2/5 |
| q7 | 581 | 610 (0.95x) | 7/29 | 582 | 653 (0.89x) | 6/29 |
| q8 | 426 | 390 (1.09x) | 11/39 | 399 | 441 (0.90x) | 10/39 |
| q9 | 907 | 936 (0.97x) | 10/30 | 941 | 935 (1.01x) | 9/30 |
| q10 | 496 | 512 (0.97x) | 4/25 | 672 | 1276 (0.53x) | 6/22 |
| q11 | 246 | 215 (1.14x) | 8/16 | 258 | 280 (0.92x) | 6/16 |
| q12 | 348 | 313 (1.11x) | 2/13 | 332 | 320 (1.04x) | 2/13 |
| q13 | 608 | 625 (0.97x) | 3/15 | 584 | 591 (0.99x) | 3/15 |
| q14 | 231 | 434 (0.53x) | 6/9 | 234 | 387 (0.60x) | 5/9 |
| q15 | 310 | 316 (0.98x) | 0/1 | 406 | 404 (1.00x) | 3/11 |
| q16 | 263 | 279 (0.94x) | 1/17 | 262 | 264 (0.99x) | 1/17 |
| q17 | 509 | 550 (0.93x) | 11/18 | 923 | 896 (1.03x) | 4/18 |
| q18 | 969 | 994 (0.97x) | 13/29 | 1273 | 1337 (0.95x) | 3/29 |
| q19 | 220 | 185 (1.19x) | 1/9 | 186 | 240 (0.77x) | 0/9 |
| q20 | 242 | 281 (0.86x) | 11/32 | 264 | 271 (0.97x) | 6/32 |
| q21 | 1466 | 1467 (1.00x) | 6/30 | 1418 | 1513 (0.94x) | 6/30 |
| q22 | 491 | 532 (0.92x) | 1/9 | 498 | 491 (1.02x) | 0/9 |

Doubles: 21 of 22 checksums identical between `spark` and `vector`. Decimals: 22 of 22. The one
double mismatch is Q15, and it is a finding in its own right: the query joins on
`total_revenue = (select max(total_revenue) ...)`, an equality between two separately computed
double sums. Spark sums each partition in one order and the two aggregations agree bit for bit;
our interleaved accumulators (`sparkvector.agg.interleave=4`) can sum the same rows in a different
order in the two aggregations, the last bits differ, the equality finds nothing and AQE replaces the
join with an `EmptyRelation` -- `vector` returns 0 rows where Spark returns 1. Over decimals the
sums are exact and both engines agree. This is why `spark.vector.exec.strictFloatingPoint` exists
and defaults to on: it restores Spark's order (Q15 returns its row), and the benchmarks turn it off
to measure the fast sums, which is also what Comet's default does, so the Q15 mismatch stays in the
reports by design.

### What the reason list says

Every decimal-only fallback is on the aggregate, and it comes in two shapes:

- **The `sum` buffer widens past 18 digits even for a plain column.** `sum(l_quantity)` over
  `DECIMAL(15,2)` has a `DECIMAL(25,2)` buffer (Spark adds 10 digits), so Q1, Q17, Q18, Q20 and Q22
  lose their aggregate on `sum` or `avg` of an unmodified column: `sum buffer decimal(25,2) exceeds
  18 digits`, `avg buffer decimal(25,2) exceeds 18 digits`. This is #27's case and it is the one
  that unlocks the most: it is the only decimal reason in Q17, Q18, Q20 and Q22.
- **The multiply's declared result type is `DECIMAL(38,4)` and the sum over it needs a wider buffer
  still.** `l_extendedprice * (1 - l_discount)` is `DECIMAL(38,4)` (`(15,2) * (17,2)` capped at 38),
  so `unsupported column type decimal(38,4) for sum` appears in ten queries (Q3, Q5, Q6, Q7, Q8, Q9,
  Q10, Q14, Q15, Q19) and `decimal(38,6)` / `decimal(36,2)` in Q1 and Q11. The multiply itself never
  shows up as a `Project` fallback: Spark folds it into the aggregate's input expression, so the
  aggregate is the operator that gives up. Keeping these products on INT64 lanes when the values
  fit (#26) only helps if the `sum` buffer over them can be wide too, i.e. #26 depends on #27; #27
  alone already unlocks the plain-column sums.
- **Nothing in TPC-H needs a genuinely wide declared input (#28):** every base column is `(15,2)`.
  The division in Q1's `avg` does not appear either -- `avg` fails earlier, on its buffer.

**Update (#26, slice 1).** With the product under a decimal `sum` computed speculatively in 64 bits
and its overflowing rows added exactly to the 128-bit sum, the `unsupported column type decimal(38,4)
for sum` reason is gone from all ten queries. On the same SF1 decimal schema (`local[8]`, one iteration,
no warm-up, the shared x86 host, so timings are indicative only) the accelerated-operator counts read
q3 11/16, q5 11/31, q6 4/5, q7 9/29, q8 14/39, q9 12/30, q10 9/22, q11 10/16, q12 7/13, q15 5/11,
q19 7/9, q20 17/32 -- against 5, 8, 2, 6, 10, 9, 6, 6, 2, 3, 0 and 6 in the table above (part of the
rise is the operators landed since that run: windows, nested columns, the broadcast join over an
exchange). 22 of 22 checksums identical. What still holds decimals back: Q1's nested product
`sum((l_extendedprice * (1 - l_discount)) * (1 + l_tax))` (`decimal(38,6)`, a wide *operand* -- the next
slice), the `avg` buffers (`decimal(25,2)`, the wide `avg` buffer), and the operators *above* a wide
sum (`TakeOrderedAndProject` / `Filter` over `revenue`, `sum(l_quantity)`: a wide result column as an
input, #28). Slice 2 of #26 (nested products) then removed Q1's `decimal(38,6)` reason as well: its
aggregate then waited only on the `avg` buffers (`avg buffer decimal(25,2) exceeds 18 digits`). Slice 3
(the wide `avg`: Spark's `(sum: decimal(p+10), count)` buffer on the same 128-bit accumulator, the
result Spark's own division evaluated over the merged buffer) removed that reason from every query
that carried it: Q1 is at 4/7 accelerated operators (from 2/7 -- both aggregate stages ours; what
remains is the global sort over the shuffle, by design), the same count as on the double schema, and
Q17 (`0.2 * avg(l_quantity)` in the correlated subquery) at 7/18 from 4/18 -- its aggregate stages
convert, the multiplication *above* the wide average is still refused (`decimal result decimal(21,7)
exceeds 18 digits`, a wide result as an input, #28). 22 of 22 checksums identical. No decimal `sum`
or `avg` buffer reason is left on TPC-H; the decimal list is now only wide results consumed above
their aggregate (`TakeOrderedAndProject` over `revenue`, `sum(x) / 7.0`, `sum(a) / sum(b)`,
`0.5 * sum(l_quantity)`) and the `+`/`-` rescale shapes of #26's next slice.

The rest of the decimal-only list is the cascade: `Filter: child HashAggregate is not columnar`,
`BroadcastHashJoin: child Filter is not columnar`, `Project`/`Sort: child ... is not columnar` --
operators that would have been ours had the aggregate below them stayed columnar (Q11, Q15, Q17,
Q18, Q20). Fixing the aggregate takes them back for free.

Two performance notes rather than conclusions, given the host: Spark itself is 4x slower on Q1 with
real decimals (472 -> 1939 ms; its decimal `sum` is a `BigDecimal` path), so the gap a wide
accumulator would open is larger than the double numbers suggest; and Q10 with decimals is the one
query where `vector` is markedly slower than Spark (0.53x) while still accelerating 6 of 22
operators -- worth a profile before #27 lands, since it is the shape that will run more of our code
afterwards.

## Q6 revisited: the copy is a dictionary decode (#14)

The Q6 analysis above blames two costs, and #61 built a lever for the first one: with
`spark.sql.columnVector.offheap.enabled=true` the scan's fixed-width columns are handed to the
kernels as views of Spark's native memory instead of being copied. Measured at SF10 under the
protocol (one JVM per configuration, `local[8]`, 8 GB heap, 5 warm-up and 7 measured runs, a quiet
shared x86 host -- so the absolute medians are not the M3 numbers above, only the ratios and the
profile are the point):

| configuration | spark | vector |
|---|---:|---:|
| on-heap column vectors (default) | 531.6 ms (1.00x) | 563.9 ms (0.94x) |
| `spark.sql.columnVector.offheap.enabled=true` | 530.4 ms (1.00x) | 575.0 ms (0.92x) |

The filter's kernel time is 1100 ms on-heap and 1128 ms off-heap (summed over 8 threads): the wrap
changes nothing, because it never engages. The Parquet files DuckDB writes (and any writer with the
default dictionary threshold) dictionary-encode all three predicate columns -- `l_shipdate`,
`l_discount` and `l_quantity` are `PLAIN_DICTIONARY` in every column chunk; only `l_extendedprice` is
`PLAIN` -- and the adapter decodes a dictionary-encoded column on either heap
(`SparkColumnVectorBuffers.decodeDictionary`: copy the ids, mask the nulls, scan for the largest id,
decode the table, gather, then copy into the arena). A JFR profile of the `vector` run puts that
decode at **20.5% of all JVM samples**, against about 5% for every filter kernel together (the three
comparisons, the bitmap work and the compaction) and 12% in Spark's own `VectorizedRleValuesReader`,
which both engines pay. So the "copy" of the first cost is really the decode of a dictionary Spark
kept, and it costs four times the comparisons it feeds.

What that suggests, for the owner (measured here, not built): a dictionary-encoded predicate column
should not be decoded at all -- evaluate the comparison over the dictionary *table* (2,526 dates,
11 discounts, 50 quantities over the whole SF10 table, fewer per chunk) into one bit per id and gather the bits by id, which
turns three full-column compares plus three decodes into three table compares plus three id gathers;
the selection is then known before any value column is materialised, and only the survivors of
`l_extendedprice` and `l_discount` (1.9%) need decoding for the product -- the issue's "fuse the copy
with the compaction", in the form the data actually takes. A cheaper interim step is to gather
directly into the arena segment instead of a heap array followed by a copy (one of the decode's five
passes). `spark.sql.columnVector.offheap.enabled` is not a Q6 lever and is not recommended for it.

## TPC-DS coverage, scale factor 1

TPC-DS (#90) is the compatibility target after TPC-H: 24 tables, 103 queries (Spark's own
`tpcds/q*.sql`), and the parts of Spark this plugin does not convert yet. `gen-tpcds.sh` writes the
tables with Spark's `TPCDSSchema` types -- `DECIMAL(7,2)` money, `DATE`, `INT` identifiers and counts
-- so the plans are the approved ones the per-query issues quote; `run-tpcds.sh` runs the same
measurement, checksum and acceleration code as TPC-H. The first run of all 103 queries was made on a
shared x86 build host (8 threads, JDK 25, `--warmup 0 --iterations 1`), which is a **coverage and
correctness** reading, not a timing one: no time from it is reported here.

Acceleration column (`spark,vector`, SF1, `spark.sql.shuffle.partitions=8`):

| operators run by our kernels | queries |
|---|---:|
| 100% | 0 |
| 75% or more | 25 |
| 50% or more | 56 |
| 25% or more | 20 |
| under 25% | 2 |

2785 of the 4734 operators that count across the 103 plans are ours (59%). No query is fully
accelerated at SF1, and the reasons are the ones the per-query issues list, in this order of how many
queries each touches once the "child X is not columnar" cascades are followed to their root:
Spark's global `Sort` above its row shuffle (37 queries, by design without Comet's shuffle), decimal
`sum`/`avg` results wider than 18 digits (#27/#87: 27 queries -- 24 end in a Spark aggregate over a
`decimal(27,2)` `sum`, others have a `sum` or `avg` buffer past 18 digits inside), sort-merge joins
(#10: 19 at SF1), window functions (#58: 15), `substr` (#39: 10), scalar subqueries (#48: 5),
broadcast nested loop joins (#60: 3), a TINYINT cast in `ROLLUP` plans (3), `stddev_samp` (#46: 2)
and `upper` (2). The best plans are the star-join aggregates: q22 and q96 at 15/19, q7 and q26
at 19/24, q3/q42/q43/q52/q55 at 12/15, where the remaining Spark operators are the final sort and the
`TakeOrderedAndProject` above it.

Refreshed after the wide decimal sum landed in both stages (#27, #87): 2844 of 4735 operators ours
(60%), 27 queries at 75% or more, still none fully accelerated; the decimal `sum`/`avg` refusals fell
from 27 queries to 9 -- those sum a *wide input* (`decimal(27,2)` totals of a `UNION ALL` of channel
sums) or average a decimal, which need a wide lane rather than a wide accumulator (#28). All 103
queries agree with Spark to 10 significant digits.

Refreshed again after the string, datetime, cast and `try_*` families, nested loop joins, sampling
and the literal work (#37--#49, #52, #60, #63): **2956 of 4735 operators ours (62%)**, 40 queries at
75% or more, 47 at 50% or more, and the first fully accelerated query, q9 (every operator ours; its
plan has no global sort). Sort-merge joins are named in 19 queries. With the opt-in rewrite of
sort-merge joins into the shuffled hash join (#10, `spark.vector.exec.sortMergeJoin.enabled=true`):
**3051 of 4682 operators ours (65%)**, 44 queries at 75% or more; 12 queries change -- ten gain
(q8, q11, q14a, q14b, q25, q29, q31, q54, q72 and q78, q72 from 16/51 to 37/49, q25 and q29 from 14/38
to 27/36) and two lose a few (q38, q92: adaptive execution re-plans a join as a broadcast join whose
streamed side is then the bare row shuffle read, which our broadcast join refuses -- accepting an
exchange as the streamed input, as the shuffled join does, would recover them -- done since, #98).
Eleven queries still carried a sort-merge join then; with chains of merge joins on the same key
converting whole (#102: q10 21/45 to 31/41, q35 20/44 to 30/40, q69 21/44 to 30/40, q95 21/43 to
30/40) and the broadcast join streaming from a shuffle read, the flag gives **3179 of 4668 operators
ours (68%)**, 50 queries at 75% or more, all 103 checksums equal. Seven queries still carry a merge
join, each for a reason outside the join itself: a `decimal(24,7)` or `decimal(19,2)` input the kernels
have no lane for (q1, q30, q81, q64), a `decimal(27,2)` join column (q51), a `Window` input (q44), an
aggregate that does not convert (q97). The per-query TPC-DS issues name the sort-merge join as their
blocker some sixty times; the flag is how to see which of those it lifts.

Refreshed after the broadcast join learned to stream from a bare shuffle read (#98; the shape adaptive
execution leaves when it re-plans a shuffled join as a broadcast join at runtime): **3042 of 4736
operators ours (64%)** by default, 41 queries at 75% or more, 14 queries up (q6 17/33 to 24/33, q23a
43/82 to 57/82, q23b 58/118 to 77/118, q14a 178/357 to 199/357), none down, all 103 checksums equal.

Refreshed after the first window layer (#58: `row_number`, `rank`, `dense_rank`): **3065 of 4734
operators ours (65%)** by default, q44 14/41 to 22/41, q47 and q57 36/70 to 42/70, q70 22/40 to 25/40,
all checksums equal. Fourteen queries still carry a `Window`: seven for a whole-partition `avg` or
`sum` (q12, q20, q47, q53, q57, q63, q89, q98), five for a wide decimal column in the window's input
(q36, q49, q51, q67 -- #28), and two for a `TINYINT` grouping-id column (q70, q86 -- the kernels have no
8- or 16-bit lane; widening those to INT32 at the adapter would be cheap). The second window layer
(whole-partition aggregates) does not move these numbers: every TPC-DS window aggregate is over a
decimal (`avg(sum(ss_sales_price))`, `sum(itemrevenue)`), whose buffer is Spark's `Decimal(p + 10)` --
the 128-bit lane of #28 gates all eight. The fourth layer, the per-partition top-k Spark plans under a
`rank <= k` filter (`WindowGroupLimitExec`, Partial before the shuffle and Final after the sort), brings
the tally to **3070 of 4736 (65%)**: q44 22/41 to 26/41 (its four group limits, which sat directly on our
aggregates and forced them back to rows), q70 25/40 to 26/40; the group limits still Spark's are the
ones over a wide decimal (q67) or the `TINYINT` grouping id (q70, q86), for the same reasons as their windows.
The later window layers (running frames, offset functions, `percent_rank`/`cume_dist`/`ntile`, sliding frames)
complete the operator's scope without moving the TPC-DS tally: the suite's remaining window fallbacks are all
decimal (#28) or the `TINYINT` grouping id; q51's sliding `sum` is over a decimal too.

**Correctness: q66 returned every row twice (#162), now fixed.** The two channel aggregates of
q66 are ours and emit a `decimal(28,2)` sum (#87); the union above refused that type and stayed Spark's,
whose columnar `UnionExec` concatenates (the #128 upstream bug), so the aggregate planned without a
shuffle on the union's partitioning saw every warehouse once per channel. Two fixes: the union is
planned whatever its children's types -- it forwards batches and reads nothing -- and our aggregate
reports its output partitioning through its result aliases as Spark's `HashAggregateExec` does
(q66 groups by `d_year` and outputs it as `year`; a partitioning naming an attribute the operator does
not output is one a union cannot match, and the union then falls back to concatenation at execution).
All 103 queries agree with Spark again.

**Correctness: three checksums differed, now fixed (#128).** q33, q56 and q60 -- the same template,
three sales channels aggregated per item and combined with `UNION ALL` -- returned 100 rows under both
configurations but with different sums (for q33, `i_manufact_id` 1000 totalled 8756.30 under Spark
and 525.60 under the plugin). The cause was the union's partitioning contract: every channel's Final
aggregate is hash-partitioned by the key, so Spark's `UnionExec` reports that partitioning and
`EnsureRequirements` plans no shuffle between the union and the aggregate above it -- the union must
then keep the children's i-th partitions together. `VectorUnionExec` reported `UnknownPartitioning`
(too late: the shuffle was already gone) and concatenated the children's RDDs, so every key came out
once per channel with that channel's sum, and `ORDER BY total_sales LIMIT 100` picked the small
per-channel sums. The union now reports what Spark's would and reads co-partitioned children through
`SQLPartitioningAwareUnionRDD`; the bisect that pointed at the Final aggregate was misread -- with
Spark's Final the union's children were Spark's, whose union is partition-aware. Spark 4.1.3's own
columnar `UnionExec` has the same concatenation (fixed upstream later), which is why disabling ours
did not help. All 103 queries now agree to 10 significant digits.

## Phase 5: q9/q14/q17/q18 against Comet -- profiles and what they changed

Profiling the four SF1 queries where `comet-scan-vector-shuffle` trailed `comet` the most led to
three changes (plus `spark.vector.exec.strictFloatingPoint`, documented with the aggregation
sections above).

**Why q14 was slow: eight tasks rebuilding the same broadcast table.** The q14 JFR profile put 47%
of all samples under `VectorBroadcastHashJoinExec`'s build path -- `BuildTable.fromRelation`,
`ArrowLayout.ofStrings` (decoding `UTF8String`s to `java.lang.String` and back),
`GroupKeyTable.insert` and `rehash` -- against 4% for Comet's whole native execution. Every task
read Spark's 200k-row broadcast `HashedRelation` into columns and hashed it into a key table, so
the same work ran once per task rather than once. Now the table is built once per executor and
shared read-only by every task of every join over that relation (`BuildTable.sharedFromRelation`,
keyed weakly on the relation object, arena freed by a `Cleaner` when the broadcast is dropped).
Strings go straight from `UTF8String` bytes to Arrow memory, the build table skips the on-the-fly
string dictionaries (`GroupKeyTable(types, false)`: join keys are mostly distinct, and an immutable
table is what makes concurrent probing safe), and each probe passes its own hash scratch.

**Double grouping and join keys** now compile: Spark's optimizer wraps them in
`NormalizeNaNAndZero`, which is a real kernel pass (canonical NaN, `-0.0` to `0.0`), after which the
key tables' bit comparison agrees with Spark's equality. This removed q18's Partial-aggregate
fallback (`grouping key type double not supported` on `o_totalprice`) and the whole non-columnar
cascade above it: q18 in `comet-scan-vector-shuffle` goes from 29/39 operators accelerated to 31/35.

**Short-string copies** (`ByteCopy`): gathering q9's 320k joined rows spent 13% of samples in
`MemorySegment.copy` set-up (session and bounds checks) for 10-25-byte strings. Runs of <= 16 bytes
are now moved as two overlapping unaligned longs in the UTF8 gather and compact kernels. JMH
(`GatherBenchmark`, 4096-row gathers from a 64k-row column, M3, noisy machine): 6-byte strings 414k
vs 189k rows/us against the per-string `MemorySegment.copy` loop (2.2x); 12 and 25 bytes within
noise of each other.

The benchmark configurations also turn the opt-in sort-merge join rewrite on
(`TpchRunner.VectorFast`): Comet accelerates those joins too -- natively, as merge joins; its own
hash-join replacement, `spark.comet.exec.forceShuffledHashJoin`, is likewise off by default -- and
the rewrite is our only way to. q9 goes to 38/42
operators accelerated (no `SortMergeJoin` row round trips left) and q18 to 31/35. The medians of the
verification run are not quotable -- the machine carried a load average of 6-7 and `spark`'s own
medians rose 20-50% over the phase-4 session -- but the checksums and operator counts are
deterministic: q14/q17/q18 agree with Spark everywhere, and q9 differs from `spark` in the last ulp
of 3 of 175 rows because the hash join feeds the double sums in a different row order than the merge
join did (strict floating point reproduces Spark's rounding for the same row order, not across
different plans).

## AVX2 / AVX-512

Not measured: this document is written from an Apple M3. The kernels select the platform's
preferred vector shape at start-up and the code paths for 4 and 8 double lanes exist (a real
`compress` for 16-lane int compaction, 256-entry shuffle tables for 8-lane 64-bit types, the
8-group masked-reduction cut-over), but they were only ever executed emulated:
`-Dsparkvector.vectorBits=256` and `512` force those shapes on any machine, and the kernel test
suite passes at all three widths. Two things worth re-measuring on AVX-512 before trusting the
defaults: `VectorMask.fromLong` is a single `kmov` there, so the broadcast-AND-compare mask
construction chosen for NEON may be the slower option; and the masked-reduction threshold of 8
groups is a guess from lane count, not a measurement.
