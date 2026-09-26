---
layout: default
title: Benchmark results
---

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

### Wide decimals: the two-limb DECIMAL128 kernels versus `BigDecimal` per row (#258)

`WideDecimalBenchmark`, `decimal(38,10)` columns of 8192 rows, one thread, same flags as above.
`reference` is a loop over pre-built `BigDecimal` objects doing what Spark's row path does per row
(the operation, then the `precision` check of `toPrecision`); it flatters Spark, whose row path also
has to materialise a `Decimal` from the unsafe row first, which the lane never does. Two magnitudes:
`long` -- unscaled values of 18 digits (TPC-H-sized amounts declared wide, the common case), where the
kernels stay on the limbs; `wide` -- unscaled values of 30 to 37 digits.

| operation | magnitude | reference | kernel | ratio |
|---|---|---:|---:|---:|
| `+` (same scale) | long | 90k | 439k | 4.9x |
| `+` (same scale) | wide | 75k | 447k | 6.0x |
| `*` keeping the exact scale (`decimal(20,2) * decimal(20,2)` -> `decimal(38,4)`) | long | 34k | 398k | 11.8x |
| `*` keeping the exact scale | wide | 28k | 403k | 14.6x (every row overflows on both sides) |
| `*` with Spark's capped scale (`decimal(38,10) * decimal(38,10)` -> the half-up rescale of a 20-digit-scale product) | long | 10.3k | 7.8k | 0.75x |
| `*` with Spark's capped scale | wide | 10.0k | 6.4k | 0.64x |
| `/` | long | 4.1k | 3.3k | 0.80x |
| `/` | wide | 4.1k | 3.1k | 0.77x |
| `<` literal -> bitmap | long | 1590k | 1258k | 0.79x |
| `<` literal -> bitmap | wide | 1285k | 675k | 0.53x |
| `<` column -> bitmap | long | 722k | 881k | 1.22x |
| `<` column -> bitmap | wide | 729k | 316k | 0.43x |

What the numbers say:

- Where the kernel stays on the limbs it wins by the margin the design promised: five to six times
  on `+`, twelve times on a `*` whose result keeps the exact scale (`multiplyHigh` for the 128-bit
  product, one precision check). That is the shape TPC-H-like amounts declared wide take.
- Where a row leaves the limbs the kernel is a `BigDecimal` loop with columnar overhead, and it
  shows: a `*` whose result Spark caps below the exact scale -- every `decimal(38,s)` product, since
  Spark's `adjustPrecisionScale` turns `(38,10) * (38,10)` into `(38,6)` -- needs the half-up rescale
  of a 77-digit product and takes the exact path for every row, as does every `/` (Spark's own is
  `BigDecimal.divide(38, HALF_UP)` then `toPrecision`). Building the `BigInteger` from the limbs and
  writing the result back costs 20 to 35 percent against the pre-built objects. The values are
  right by construction; making those two shapes fast means a 192-bit rescale on the limbs and a
  two-limb long division, which is the follow-up the issue leaves open.
- The comparison kernel is at parity with the reference on values that fit a long and behind it on
  wide ones. It is the scalar word loop #28 asked for (no species), so nothing hides the per-row
  cost: two limb loads, a signed then an unsigned compare and a data-dependent branch into the
  bitmap word, where the reference compares two objects already in cache. The lane still pays for
  itself where it matters -- a wide predicate no longer forces the whole operator to Spark's row
  path -- but the compare loop is a candidate for a branch-free formulation.

Run it with `"WideDecimal"` as the benchmark filter.

### The index sort: radix passes versus one comparison sort per pass (#285)

`SortBenchmark`, one thread, `-wi 1 -i 2 -w 1 -r 1`, x86 build host (8 threads, JDK 25), ms per sort
of the whole column; `legacy` is the kernel before #285 (one `Arrays.sort(long[])` of packed `(key,
position)` per 32-bit key pass, kept in the benchmarks module as the baseline), `radix` the LSD radix
passes over 8-bit digits with the uniform digits and the already-ordered passes skipped. One key, no
nulls, 1M rows:

| key | random | low cardinality (100 values) | presorted |
|---|---:|---:|---:|
| INT32 | 21.6 → 21.3 | 21.4 → 11.8 | 4.1 → 4.4 |
| INT64 | 44.5 → 51.0 | 29.7 → 21.1 | 9.0 → 6.5 |
| FLOAT64 | 51.4 → 47.7 | 26.6 → 23.4 | 23.2 → 6.6 |
| UTF8, up to 8 bytes | 64.5 → 35.7 | 55.1 → 30.9 | 60.4 → 29.2 |
| UTF8, 40 bytes (ranked) | 625 → 632 | 487 → 443 | 122 → 139 |
| UTF8, dictionary of 1000 | 21.5 → 15.5 | 21.8 → 14.4 | 23.7 → 16.9 |

10M random INT64 rows: one key 729 → 688 ms (73 → 69 ns/row), three keys (the long, a 4-value
int, the long again) 1798 → 1261 ms; with 10 % nulls one key 793 → 762 ms.

What the table says. A radix pass is O(n), but at these sizes its cost is the scatter's random
writes -- two per row per digit, eight digits for a random 64-bit key -- and that is what the
comparison sort cost too: random 64-bit keys are a draw (the JMH stack profile puts the scatter at
37 % of the benchmark thread, the scratch zeroing at 8 %, the gathers through the permutation at 5 %,
and half the runnable samples in the collector over ~30 MB of per-sort scratch that both kernels
allocate). The wins are where digits disappear: 32-bit keys, low-cardinality and dictionary keys
(the high digits are one bucket and are skipped, 1.4-1.8×), short strings (one 64-bit prefix pass
instead of three packed sorts, 1.8×), presorted input (the pass is a scan, on a par with the
comparison sort's run detection) and multi-key sorts (30 % at 10M, every later key a full pass in
both kernels but cheaper here). Long strings are the rank's merge sort in both kernels, 600 ns/row
at 1M -- the next lever there is not the pass. Three variants were measured before settling: 11-bit
digits (2048 buckets), positions-only scatters with the keys gathered per digit, and per-row segment
reads instead of one array copy of the column; none moved the random fixed-width number beyond
noise. The scratch -- five arrays of the partition's length -- is what the chunked runs of slice 2
bound to the run size.

Runs and the merge (`SortBenchmark.runs8`: the same rows sorted as eight runs, then `RunMerge` -- a
loser tree over the runs' cursors with the widened leaf, a block of the winner's rows below the
runner-up's key emitted in one step -- walked to the end; INT64, one key, ms):

| input | 1M, one sort | 1M, 8 runs + merge | 10M, one sort | 10M, 8 runs + merge |
|---|---:|---:|---:|---:|
| random | 50 | 85 | 685 | 1010 |
| presorted | 6.4 | 7.7 | 112 | 98 |
| low cardinality (100 values) | 18.7 | 20.1 | 322 | 258 |

On random keys the merge costs about 30 ns/row and chunking is not a speed-up (a loser tree in place
of the heap changed nothing, so the cost is not the compares but the per-row bookkeeping and the
gather through the runs' permutations): runs bound the sort's scratch to the run and are the shape a
spill plugs into. On presorted and low-cardinality keys the widened leaf turns the merge into block
moves and eight runs beat one sort at 10M. Under a limit each run keeps only its first *n* rows for
the merge (top-N by run), so `ORDER BY ... LIMIT n` over a large partition merges *n* rows per run.

TPC-DS SF1, the three queries with the large window sorts (`run-tpcds.sh --queries q67,q70,q86`, 3
iterations, 1 warm-up, medians): q67 1495 ms Spark / 1818 ms ours (20/26 operators), q70 496 / 552
(26/40), q86 188 / 213 (10/19). Our sort runs in two of the three plans (the window's local sort) and
the take-ordered in one; at SF1 these partitions are far below a run, so the numbers say nothing about
runs -- they are the window queries' SF1 reading on a shared build host with three iterations, in which the
plugin trails Spark by 10-20 %; a coverage reading, not a timing one, and not moved by this issue.

### The merge join (#286): order-preserving, against the hash rewrite and Spark

`VectorSortMergeJoinExec` under `spark.vector.exec.sortMergeJoin.mode=merge`, measured three ways on
the x86 build host (8 threads, JDK 25).

**Golden files.** Spark's SQL golden-file suite with the merge join: 642 passed, 0 failed. The three
`in-subquery` files that differ under the hash rewrite (`in-limit`, `in-order-by`, `in-set-operations`
-- tie order under `ORDER BY` and unordered `LIMIT`) pass unchanged: the merge join emits Spark's
order. And the coverage floor moved up in 27 files, the join files chief among them: `postgreSQL/join.sql`
385 → 634 executions on our operators, `in-joins` 138 → 288, `exists-joins-and-set-ops` 84 → 117,
`udf-join` 22 → 43, `using-join` 0 → 9, `inner-join` 3 → 6 (a few decimal files in the list are
#258/#259's gains; the floor predates them).

**TPC-DS SF1**, all 103 queries, one iteration after a warm-up, `spark.sql.shuffle.partitions=8`;
checksums identical to Spark in every mode:

| mode | operators ours | queries at 75 % or more | sum of medians |
|---|---:|---:|---:|
| `off` (Spark's sort-merge join) | 3204 / 4734 (67 %) | 52 | -- |
| `hash` (the #10 rewrite; the harness's default) | 3425 / 4655 (73 %) | 65 | 90.8 s |
| `merge` | 3442 / 4736 (72 %) | 57 | 92.0 s |

The merge join takes every sort-merge join in the 103 plans -- none is left to Spark -- and the seven
queries the hash rewrite had to leave (a `decimal(24,7)` or `decimal(27,2)` input, a window above, an
aggregate that did not convert: q1, q30, q44, q64, q78, q81, q97) gain operators; none loses. The
ratio reads a point lower than the hash rewrite's because the sorts Spark places below a merge join
stay in the plan and count, where the rewrite dropped them: 4736 operators against 4655 for the same
rows. On time the two are a wash overall: the merge is faster on 23 queries (q62 0.47×, q72 0.55×,
q43 0.73×, q55 0.76×) and slower where a join has hundreds of thousands of one-to-three-row runs.
The issue's three named queries: q4 Spark 2016 ms / off 2164 / hash 2108 / merge 2438, q11 1054 /
1846 / 1217 / 1185, q74 993 / 697 / 762 / 682.

**The slow shapes, and what they taught.** The first merge-mode run took 98 s on q17 (0.8 s under
the hash rewrite) and three times the hash rewrite's time on q10, q14 and q25: every buffered run
lived in its own *shared* arena, and closing a shared arena is a thread-local handshake with every
thread in the JVM -- hundreds of microseconds, paid per run, and q17 joins `store_sales` with
`store_returns` on three keys, a few hundred thousand tiny runs. Confined arenas brought q17 to 2.3 s.
Then the issue's own design -- a run inside one batch *borrows* the batch's copy, only a run reaching
a batch edge is copied -- brought it to 0.95 s. The two full outer joins over mostly unique keys (q51,
q97) had a third cost: every unmatched row left the operator as its own one-row Arrow batch; single
rows now go through an ordered 8192-row buffer, and q51 went from 3.9 s to 1.1 s. After the three,
with three iterations (`hash` in brackets): q17 856 ms (797), q25 904 (787), q29 970 (811), q51 1122
(1082), q64 3058 (1833), q97 2979 (385). q97 is the one that stays: a full outer join of two
aggregated, mostly unique-key sets of some 300k rows each with few matches, so every row on both
sides is a run and the per-run bookkeeping -- the run object, the compare through the columns, the
buffer append -- is the whole cost; the remedy is a per-batch cursor that walks runs as index pairs
without a run object, left for the planning issue (#287) to weigh against simply choosing the hash
join there.

**Reading.** At SF1 a side is nearly always small enough for a hash table, and the hash rewrite wins
or ties; the merge join is the accelerated option where the rewrite cannot go -- a parent relying on
the join's ordering, no statistics, two large sides -- and it converts every sort-merge join with
Spark's row order. Which one a given join gets is #287's decision.

### The planning mode `auto` (#287): the merge join where the order can show, the hash rewrite otherwise

`spark.vector.exec.sortMergeJoin.mode=auto` decides per sort-merge join in the pre-pass over the plan
(`markSortMergeJoins`). Three things send a join to the merge join: a parent that relies on its
ordering (a window over the same keys, a merge join above without a shuffle in between); an *order
that can show* -- the join sits below a `LIMIT`, a `TakeOrderedAndProject`, a `Sort`, or a
range-partitioned exchange (a global sort's, which is all an adaptive stage sees of the sort above it)
with no aggregate and no other exchange in between, so the rows a `LIMIT` picks or the tie order under
`ORDER BY` would differ under the hash rewrite; and a hash rewrite that is not allowed -- no runtime
statistics, both sides over `spark.vector.join.maxBuildSize`, a skew join -- Everything else takes the hash
rewrite; a hash rewrite whose inputs are refused (Spark's operators below) leaves the join to Spark --
the merge join could read those inputs, but on q97 (a full outer join of two 300k-row unique-key sides
over Spark's aggregates) it ran 2960 ms against 385 ms for Spark's own merge join, the per-run
bookkeeping of #286, so it does not take them until it walks runs without a run object. The plan prints the decision on the operator:
`Sort-merge join as hash join: right side fits spark.vector.join.maxBuildSize by statistics`, `... as
merge join: the row order reaches a limit or a sort`, `... ordering relied on by the parent`, `... no
size statistics for a build side`.

**Golden files.** The Spark SQL golden suite under `auto` (`benchmarks/scripts/run-spark-sql-tests.sh`
with `-Dspark.vector.exec.sortMergeJoin.mode=auto`): 642 passed, 0 failed, 111 ignored -- including
`subquery/in-subquery/in-limit.sql`, `in-order-by.sql` and `in-set-operations.sql`, the three files the
hash rewrite alone changes (its tie order). 3891 of 33856 executions ran at least one operator of ours,
8687 operators in total.

**TPC-DS SF1** (one warm-up, one measured iteration; the `spark` baseline of this run 70.5 s against
66.5 s in the run below, so this host was 6 % slower that day):

| mode | operators ours | queries ≥ 75 % ours | shuffled hash joins | merge joins | total (s) |
|---|---|---|---|---|---|
| `off` | 3204 / 4734 (67 %) | 52 | 0 | 0 | -- |
| `hash` | 3425 / 4655 (73 %) | 65 | 17 | 0 | 90.8 |
| `merge` | 3442 / 4736 (72 %) | 57 | 0 | 19 | 92.0 |
| `auto` | 3437 / 4662 (74 %) | 66 | 13 | 5 | 95.1 (≈ 89.7 at the other run's baseline) |

All 103 checksums equal Spark's in every mode. `auto` reads as `hash` on 98 queries and takes the merge
join where `hash` left the join to Spark: q1, q30, q44, q78 and q81, each with a parent relying on the
join's ordering, each gaining operators (q44 26 → 33 of 41). It is the best of the four on operators
ours and on queries above the 75 % floor, and its time is a wash against `hash` on a single iteration
(q62 halves through the merge join, 404 → 194 ms; q1 867 vs 681 ms; the rest within the noise of one
run). No query took the merge join for a visible order at SF1: TPC-DS puts an aggregate between every
join and its `ORDER BY ... LIMIT 100`.

**The default.** The issue conditions the flip from `off` to `auto` on three readings: the golden files
under `auto` (shown above, condition met); memory accounted through the task memory manager (#12, open,
a maintainer's decision); TPC-DS SF10 under `auto` no slower than `hash` (not measured here -- SF1 is a
wash, SF10 needs the data and a session). The flip itself is one default in `VectorConf.sortMergeJoinMode`;
this crew leaves it to the maintainer with the two readings it could take. The boolean flag
`spark.vector.exec.sortMergeJoin.enabled=true` reads as `auto` since #287, and the TPC-DS harness's
`vector` configuration runs under `auto`.

### The columnar shuffle (#288): TPC-H SF10, the row shuffle versus ours

`vector` is the plugin over Spark's row shuffle -- `ColumnarToRowExec` above every shuffled
operator, `RowToColumnarExec` below its consumer; `vector-shuffle` is the same plugin with
`spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager` and
`spark.vector.shuffle.enabled=true`, every exchange above our operators `VectorShuffleExchangeExec`
over Arrow IPC files (zstd bodies, 8192-row record batches, dictionary strings) with the Arrow Flight
data plane in place (all reads are local in this one-JVM run). One session per configuration,
`local[8]`, 8 GB heap, 8 shuffle partitions, 3 measured iterations after 1 warm-up (a study run in a
scratch results directory, not the committed report's 5 + 7 protocol), median in ms;
`accelerated` counts the plan's operators that are ours. Every query's checksum equals the row
shuffle's.

| Query | vector | vector-shuffle | ratio | accelerated (vector) | accelerated (ours) |
|---|---|---|---|---|---|
| Q1 | 1219 | 1154 | 0.95 | 4/7 | 7/7 |
| Q2 | 2047 | 1861 | 0.91 | 32/45 | 35/45 |
| Q3 | 4449 | 2699 | 0.61 | 12/17 | 16/17 |
| Q4 | 3199 | 1684 | 0.53 | 8/13 | 13/13 |
| Q5 | 6977 | 5885 | 0.84 | 20/30 | 27/30 |
| Q6 | 600 | 587 | 0.98 | 4/5 | 5/5 |
| Q7 | 7099 | 5662 | 0.80 | 18/28 | 25/28 |
| Q8 | 3091 | 2692 | 0.87 | 26/38 | 33/38 |
| Q9 | 5568 | 4881 | 0.88 | 19/29 | 26/29 |
| Q10 | 3127 | 3108 | 0.99 | 15/21 | 20/21 |
| Q11 | 849 | 805 | 0.95 | 0/1 | 0/1 |
| Q12 | 2397 | 1355 | 0.57 | 7/13 | 12/12 |
| Q13 | 2825 | 2149 | 0.76 | 8/13 | 13/13 |
| Q14 | 979 | 879 | 0.90 | 7/10 | 10/10 |
| Q15 | 2677 | 2274 | 0.85 | 8/11 | 10/11 |
| Q16 | 1241 | 876 | 0.71 | 11/18 | 17/18 |
| Q17 | 5446 | 4610 | 0.85 | 13/18 | 16/18 |
| Q18 | 8372 | 4674 | 0.56 | 20/29 | 27/29 |
| Q19 | 1206 | 1390 | 1.15 | 7/11 | 10/11 |
| Q20 | 1655 | 1903 | 1.15 | 21/32 | 27/32 |
| Q21 | 38904 | 10515 | 0.27 | 20/30 | 25/27 |
| Q22 | 1243 | 834 | 0.67 | 5/10 | 10/10 |
| **22 queries** | **105169** | **62477** | **0.59** | | |

Twenty of twenty-two are faster, the total is 0.59x, and 0.78x without Q21. The wins come from two
places. Where the exchange used to sit between two of our operators, the row conversion pair is gone
and the operator above reads Arrow batches straight off the wire (Q1 and Q6 are now whole-plan
accelerated, 7/7 and 5/5, though their shuffles are too small for it to show). Where the plan has
shuffled joins, the statistics changed them: with `dataSize` reported the way Spark does (the
uncompressed size), `spark.vector.exec.sortMergeJoin.mode=auto` now sees build sides that fit and
plans hash joins where the row shuffle's estimates made it keep the merge joins -- Q21's two
`VectorSortMergeJoinExec` (the #310 per-run cost, 271 s of task time in the q21 analysis) became
`VectorShuffledHashJoinExec`, 38.9 s to 10.5 s; Q3, Q4, Q12, Q18 and Q22 follow the same pattern
between 0.53x and 0.67x. Q19 and Q20 are 1.15x slower and are the honest residual, and it is not
the operators: their per-operator times are equal or lower under our shuffle (Q19's final aggregate
4.26 s vs 5.15 s, its filter 2.52 s vs 2.88 s). The difference is garbage collection -- Q20's GC time
is 1168 ms against 304 ms, Q19's 765 against 656 -- on queries whose shuffles are small (5-13 MB)
and whose wall time is short enough for it to show. Our path allocates on the heap where Spark's
does not: the writer serialises each partition's stream into a heap `ByteArrayOutputStream` (growth
copies, a copy per spill), the reader decodes a local file segment through an `InputStream` channel
(heap chunks copied into Arrow memory), and zstd stages through JNI buffers. The follow-up (the
PR after #320) removed the heap from the path without changing the format: local file segments are
read with positional reads straight into Arrow memory, streams go straight to their spill file up to
200 partitions, the shuffle writer adapts a batch once for the partition ids and the streams, and --
the largest share, 26% of Q20's samples -- the Parquet adapter decodes a dictionary column straight
into the arena instead of through three heap arrays per column per batch (its string twin reuses
per-thread scratch). Same probe, 3 iterations: Q20 1531 ms vs 1464 (1.05x, GC 320 vs 288 ms), Q19
1427 vs 1306 (1.09x, GC 709 vs 472 ms). The adapter change pays on every configuration, not only
the shuffle's.

What the measurement found, in the order the runs exposed it -- every one general, none visible
under `local[4]` unit tests:

1. The exchange never fed its `dataSize` metric, so AQE saw every one of our shuffle stages as 0
   bytes and turned every shuffled join into a broadcast join (first run: Q3 12.8 s vs 4.3 s, Q7
   22 s vs 7 s). Every map task now adds what it wrote.
2. Raw Arrow IPC wrote 1.8x the bytes of Spark's lz4-compressed rows (Q3: 1008 MB vs 551 MB).
   Record-batch bodies are zstd-compressed (`spark.vector.shuffle.compression`); Arrow's own lz4
   codec is commons-compress pure Java and an order of magnitude slower -- Q3 crawled for fourteen
   minutes under it -- so zstd is the default and lz4 is documented as the slow option. Q3 now
   writes 247 MB.
3. One record batch per (input batch, partition): 512 rows at 8 partitions, 20 at 200, and the
   per-message costs (Arrow object churn, metadata, a compression call per buffer) made shuffled
   joins 2x slower after fixes 1 and 2. The writer now holds each partition's compacted slices and
   writes 8192-row record batches with one merged dictionary (`spark.vector.shuffle.batchRows`,
   `batchBytes`, `bufferBytes`). Q3 went from 8.3 s to 4.0 s, below the row shuffle.
4. The task-level shuffle read metrics showed zero bytes: Spark's reader merges them in its
   completion iterator and the executor only merges on heartbeats, so ours merges them at task
   completion. A test now asserts the stage's bytes read equal the bytes written.
5. `dataSize` as the compressed file bytes made AQE broadcast sides three times the size it would
   for Spark (Q14: a 45 MB side broadcast, 2.4x slower). It is the uncompressed Arrow size now, as
   Spark's is its rows' pre-compression size.
6. Q14 stayed 2.1x slower: its shuffled hash join took 5.7 s instead of 285 ms because
   `ArrowOutput.gather` decoded a dictionary-encoded string column whole -- one string append per
   row -- before gathering. It gathers through the codes now; the join is back to 287 ms and Q14 to
   0.90x.

Not measured here: remote fetches (this run is one JVM; `FlightShuffleClusterSuite` covers the
cross-executor path functionally), TPC-DS, and a real cluster. Known gaps: no TLS on the Flight
server, one `DoGet` per map output block, and both backends assume executors that stay up -- a
push-based shuffle service is future work (AGENTS.md 3.10, 7).

### The merge join without a run object (#310): the walk as cursor state, pairs and conditions per 8192

#310 named the merge join's per-run object-and-call machinery as the reason it lost on short runs
(TPC-H `lineitem` on `l_orderkey`: ~4-row runs; a unique-key side: 1-row runs) and gave two readings:
TPC-DS q97 at 2960 ms against 385 ms for the hash rewrite, and SF10 q21 under `auto` at 44.9 s against
Spark's 16.8 s. Same machine and protocol as the sections above (`local[8]`, 8 GB, SF10 for q21 with
2 measured iterations after 1 warm-up, SF1 for q97 with 5 after 2, medians).

| shape | before | after | note |
|---|---|---|---|
| q97, `sortMergeJoin.mode=merge` (TPC-DS SF1) | 529 ms | 566 ms | already at parity with the hash rewrite (511 / 518 ms): the ordered row buffer of #286 had fixed the unique-key shape before this issue; the 2960 ms predates it |
| q21, `mode=auto`, Spark's shuffle (TPC-H SF10) | 38.9 s | 21.4 s | the two merge joins from 121 s + 123 s to 46 s + 46 s of task time; Spark's own SMJ 16.8 s on the issue's host |

What changed, in `VectorSortMergeJoinIterator`: the right run is cursor state -- one object per task,
a run inside a batch a view whose matched flags live in one bitmap per batch (the earlier per-run
bitmaps also accumulated in an arena for the whole task), a run at a batch edge keeping its own arena;
the single-key compare is hoisted; unconditional pairs go through the ordered 8192-row buffer instead
of one Arrow gather per run; and under a condition -- q21's joins carry `l2.l_suppkey <> l1.l_suppkey`
-- the pairs of short runs collect in a candidate buffer and are gathered and tested once per 8192
pairs, only the columns the condition reads gathered (two of q21's thirty), the rest placeholders the
lazily adapting context never touches. Emission order is Spark's: the candidates are flushed before
anything that must follow them, and a run never straddles a flush, so an outer join's pad is decided
inside it. `VectorSortMergeJoinSuite` (every join type with and without a condition, positional
against Spark) is unchanged and green.

What the remaining gap is: a JFR of q21 after the change puts the join's own machinery under a tenth
of the samples. The top frames are Spark's `UnsafeExternalSorter`, its spill writer and reader, and
`UnsafeRowWriter` -- the row sort Spark runs below the merge join (the exchange is Spark's row
shuffle here, so our sort cannot take it; 7.7 GB of spill in the stage) plus the `RowToColumnarExec`
above it. The operator's `time` includes pulling that input, which is why the joins still read 46 s
each. With #288's shuffle the sort below is ours and `auto` plans q21's joins as hash joins on the
real statistics (10.5 s); a trial of the merge join forced over our shuffle and our sort did not
finish one iteration in an hour and was parked for a profile under a hard timeout -- not a #310 item.
(#329 ran that profile on the finished #310 code: the pathology is gone -- q21 with `mode=merge` over
our shuffle reads 10.9 s warm, against Spark's merge join at 9.0 s, our hash rewrite at 8.3 s and
Comet's merge join at 11.0 s; the recording shows no quadratic step, the merge join's own frames at
about 5% of samples, our sort at 12% and the JDK's memory-segment bookkeeping -- alignment and
liveness checks, zeroing of fresh native allocations -- at about 30% of top-of-stack samples, which is
the sort's next work, not the join's.)

### `auto` by default (#311): the three conditions, measured

#287 left the planning mode `off` by default and named what a flip needs: the golden files under
`auto`, TPC-DS with every checksum equal to Spark's, and TPC-H SF10 under `auto` not slower than
without it. Two rules were added on the way and are part of the default (AGENTS 3.6b): the **size
gate** -- a merge join `auto` would choose is left to Spark unless both inputs have statistics and fit
`spark.vector.exec.sortMergeJoin.maxInputSize` (1 GiB; the shape that failed the third condition in
#287 was q21's merge joins over Spark's spilling row sort) -- and the **build-side rule** -- the hash
rewrite declines a build side larger than the streamed side (a semi or anti join may only build its
right side).

**Golden suite** (`run-spark-sql-tests.sh`, this branch, `auto` default): 642 passed, 0 failed, 111
ignored; 27 cases run above the floor and are not rewritten. Same as `off`.

**TPC-DS SF1** (3 iterations, 1 warmup -- a study run, the protocol's 10/10 was not needed for a
count; the accelerated-operator totals and the checksums are what the condition asks):

| configuration | operators accelerated | queries at 75%+ | checksums vs `spark` |
|---|---|---|---|
| `vector`, `auto` + gate, Spark's row shuffle | 3413 / 4662 (73%) | 62 / 103 | 103 equal |
| `vector`, `auto` without the gate (#287) | 3437 / 4662 (74%) | 66 / 103 | 103 equal |
| `vector-shuffle`, `auto` + gate, our columnar shuffle (#288) | 3772 / 4661 (81%) | 91 / 103 | 103 equal |

The gate costs 24 operators and four queries at 75% (the four merge joins it leaves to Spark: three
"inputs too large", one "no size statistics") -- the price of not repeating q21 at scale. Three of
those four were a false reading (#329): the sort above a self-joined aggregate carried the join's
product estimate, 10^16 bytes at SF1; with the estimate reading the query stage that has run, q1,
q30 and q81 keep their merge join (34/38, 40/46, 39/45 operators, checksums equal). Our shuffle is
worth far more than that: 37 of the ~90 fallback reasons over Spark's shuffle were a `Sort` or a
`TakeOrderedAndProject` over Spark's row exchange, and every one of them is gone under ours.

**What TPC-DS still leaves to Spark**, by root cause (the rest of the reason list is cascades --
"child X is not columnar" -- from these), with our shuffle on:

| root cause | reasons | where |
|---|---|---|
| a wide decimal as a *value* across operators: a scalar subquery result in a filter (6), `CASE WHEN` with a `decimal(p,s)` result in a broadcast join (2), `round` over a wide decimal (1) | 9 | the #258 follow-up |
| `TINYINT`: the `lochierarchy` window column (3), `CAST(... AS TINYINT)` over `spark_grouping_id` in grouping-set aggregates (4) | 7 | a `ByteType` / `ShortType` lane widening |
| a merge join left to Spark by the size gate | 3 | the rule above; a merge join over our sorted shuffle instead of Spark's row sort is the way back in |
| an aggregate whose result expression is an alias (`ss_customer_sk AS customer_sk is not a plain attribute`) | 2 | done in #328: the alias maps onto the key's lane; q97 21/23 (the two left are broadcast exchanges, #325), checksum equal |
| a `Union` with no columnar child | 1 | cascade of the above |

**TPC-H SF10** (2 iterations, 1 warmup, doubles schema; `vector` = our operators over Spark's row
shuffle, `vector-shuffle` = over our columnar shuffle; `off` = `--conf spark.vector.exec.sortMergeJoin.mode=off`,
Spark's own sort-merge join with our operators around it). Medians in ms, ratio = `auto` / `off`:

| query | `vector` off | `vector` auto | ratio | `vector-shuffle` off | `vector-shuffle` auto | ratio |
|---|---|---|---|---|---|---|
| q1 | 1122 | 1189 | 1.06 | 1122 | 1135 | 1.01 |
| q2 | 1608 | 1872 | 1.16 | 1595 | 1618 | 1.01 |
| q3 | 3887 | 4327 | 1.11 | 3343 | 2900 | 0.87 |
| q4 | 2675 | 3069 | 1.15 | 1495 | 1711 → 1508 | 1.14 → 1.01 |
| q5 | 8615 | 6763 | 0.79 | 7211 | 5893 | 0.82 |
| q6 | 482 | 576 | 1.20 (noise, same plan) | 500 | 540 | 1.08 (same plan) |
| q7 | 6003 | 6817 | 1.14 | 6186 | 5188 | 0.84 |
| q8 | 2161 | 2264 | 1.05 | 2329 | 2281 | 0.98 |
| q9 | 5401 | 4938 | 0.91 | 4550 | 4611 | 1.01 |
| q10 | 3065 | 3205 | 1.05 | 3289 | 3267 | 0.99 |
| q11 | 743 | 734 | 0.99 | 638 | 691 | 1.08 (same plan) |
| q12 | 2390 | 2465 | 1.03 | 1718 | 1332 | 0.78 |
| q13 | 2714 | 2893 | 1.07 | 2287 | 2306 | 1.01 |
| q14 | 925 | 924 | 1.00 | 950 | 1014 | 1.07 |
| q15 | 1798 | 1933 | 1.08 | 1849 | 1857 | 1.00 |
| q16 | 1195 | 1269 | 1.06 | 1125 | 982 | 0.87 |
| q17 | 5212 | 5561 | 1.07 | 4988 | 4613 | 0.92 |
| q18 | 8481 | 8302 | 0.98 | 6249 | 4881 | 0.78 |
| q19 | 1155 | 1127 | 0.98 | 1252 | 1332 | 1.06 |
| q20 | 1444 | 1341 | 0.93 | 1294 | 1324 | 1.02 |
| q21 | 18997 | 16941 | 0.89 | 9020 | 8674 | 0.96 |
| q22 | 1457 | 1255 | 0.86 | 1188 | 882 | 0.74 |
| **total** | 81529 | 79768 | **0.98** | 64175 | 59033 | **0.92** |

(The `vector-shuffle` q21/q22 `off` readings come from a separate run of those two queries; the first
run's unit was stopped by its memory cap at q21. The two q4 `auto` numbers are before and after the
build-side rule: 1711 with the hash rewrite building lineitem, 1508 with our merge join, 15 of 15
operators ours; Spark's merge join reads 1495. Our merge join *forced* on q4 read 1703 in another run
-- the 1.5 s query moves 10% between runs.)

What the tables say. Under our shuffle `auto` is a net win with no loser beyond noise: 0.92x overall,
q22 0.74x, q12 and q18 0.78x, q5 0.82x, q7 0.84x, q3 and q16 0.87x; q21 is 0.96x, the gate having left
its lineitem joins to Spark. Over Spark's row shuffle `auto` is 0.98x overall and q21 is at parity for
the same reason, but the hash rewrite's conversions on q2, q3, q4 and q7 run 11-16% *slower* than
Spark's merge join: there the rewrite's build side arrives as rows and is converted, and at SF10 that
conversion plus the per-task hash build cost more than Spark's merge over the inputs Spark has already
sorted. At SF1 (#287) the same rewrite won over Spark's shuffle, so this is a matter of scale, and it
is recorded here rather than gated on: a deployment without our shuffle manager that sees it can set
`spark.vector.exec.sortMergeJoin.mode=off` (or `merge`), and the shuffle manager is the configuration
this project is heading for.

### The five configurations (#311 follow-up): TPC-H SF10 and TPC-DS SF1

The maintainer's matrix after the `auto` default: pure Spark; Spark with our operators over our
columnar shuffle (`vector-shuffle`); Comet's native scan feeding our operators over our shuffle
(`comet-scan-vector-ourshuffle`, new in #324); native Comet; and hybrid (Comet's scan and shuffle,
ours in between, #279's empty allowlist). One JVM per configuration, 3 iterations after 1 warmup,
medians in ms, `JVM_MEM=8g`, 8 threads, our configurations with `spark.vector.exec.strictFloatingPoint=false`
(the runner's `VectorFast`, the counterpart of Comet's default), every checksum equal to Spark's in
every cell. In parentheses: the ratio to pure Spark and the operators run by our kernels or Comet's
over the plan's total. A first `vector-shuffle` run ended at q21 under a 13 GB memory cap (8 GB heap
plus the shuffle's off-heap Arrow); the numbers below are from a 20 GB cap.

**TPC-H SF10** (doubles schema):

| query | pure Spark | ours + our shuffle | Comet scan + ours + our shuffle | native Comet | hybrid |
|---|---|---|---|---|---|
| q1 | 2131 | 1166 (0.55x, 7/7) | 1008 (0.47x, 8/8) | 988 (0.46x, 9/9) | 1053 (0.49x, 10/10) |
| q2 | 1463 | 1627 (1.11x, 35/45) | 1379 (0.94x, 50/60) | 779 (0.53x, 55/55) | 1154 (0.79x, 59/65) |
| q3 | 4008 | 2708 (0.68x, 16/17) | 2594 (0.65x, 19/20) | 1845 (0.46x, 23/23) | 2019 (0.50x, 23/24) |
| q4 | 2896 | 1544 (0.53x, 15/15) | 1441 (0.50x, 17/17) | 1227 (0.42x, 18/18) | 1515 (0.52x, 21/21) |
| q5 | 9560 | 5827 (0.61x, 27/30) | 5404 (0.57x, 36/39) | 4365 (0.46x, 41/41) | 2895 (0.30x, 42/45) |
| q6 | 424 | 566 (1.34x, 5/5) | 434 (1.02x, 6/6) | 294 (0.69x, 7/7) | 456 (1.08x, 7/7) |
| q7 | 6023 | 5330 (0.89x, 25/28) | 4987 (0.83x, 34/37) | 3018 (0.50x, 39/39) | 3652 (0.61x, 40/43) |
| q8 | 1901 | 2378 (1.25x, 33/38) | 1554 (0.82x, 46/51) | 1015 (0.53x, 50/50) | 1496 (0.79x, 52/57) |
| q9 | 4942 | 4480 (0.91x, 26/29) | 3409 (0.69x, 35/38) | 2438 (0.49x, 40/40) | 2896 (0.59x, 41/44) |
| q10 | 2747 | 3535 (1.29x, 20/21) | 3223 (1.17x, 25/26) | 1539 (0.56x, 29/29) | 2112 (0.77x, 30/31) |
| q11 | 449 | 702 (1.56x, 0/1) | 518 (1.15x, 0/1) | 281 (0.63x, 0/1) | 525 (1.17x, 0/1) |
| q12 | 2210 | 1390 (0.63x, 12/12) | 1994 (0.90x, 14/15) | 1007 (0.46x, 16/16) | 1714 (0.78x, 18/19) |
| q13 | 4247 | 2307 (0.54x, 13/13) | 2085 (0.49x, 15/15) | 1324 (0.31x, 18/18) | 1997 (0.47x, 18/18) |
| q14 | 892 | 1046 (1.17x, 10/10) | 710 (0.80x, 12/12) | 461 (0.52x, 15/15) | 620 (0.69x, 15/15) |
| q15 | 1557 | 2070 (1.33x, 10/11) | 1217 (0.78x, 13/14) | 696 (0.45x, 13/13) | 1145 (0.74x, 15/16) |
| q16 | 1189 | 979 (0.82x, 17/18) | 987 (0.83x, 21/22) | 575 (0.48x, 24/24) | 774 (0.65x, 26/27) |
| q17 | 6850 | 4811 (0.70x, 16/18) | 4702 (0.69x, 20/22) | 2185 (0.32x, 22/22) | 4263 (0.62x, 23/25) |
| q18 | 10549 | 4932 (0.47x, 27/29) | 4990 (0.47x, 32/34) | 3425 (0.32x, 39/39) | 3754 (0.36x, 39/41) |
| q19 | 976 | 1271 (1.30x, 10/11) | 1384 (1.42x, 12/13) | 856 (0.88x, 14/14) | 1690 (1.73x, 15/16) |
| q20 | 1041 | 1452 (1.39x, 27/32) | 925 (0.89x, 36/41) | 510 (0.49x, 39/39) | 895 (0.86x, 41/46) |
| q21 | 19003 | 8338 (0.44x, 28/30) | 7414 (0.39x, 36/38) | 11023 (0.58x, 38/38) | 6589 (0.35x, 40/42) |
| q22 | 1409 | 1142 (0.81x, 12/12) | 1044 (0.74x, 14/14) | 756 (0.54x, 15/15) | 750 (0.53x, 17/17) |
| **total** | **86.5 s** | **59.6 s** (0.69x, 391/432) | **53.4 s** (0.62x, 501/543) | **40.6 s** (0.47x, 564/565) | **44.0 s** (0.51x, 592/630) |

Native Comet is fastest on 20 of 22 queries and overall (2.1x over Spark). Ours over our shuffle
is 1.45x over Spark, 1.62x with Comet's scan in front; the difference to Comet is the scan and the
short queries -- on q6, q11, q14, q19 and q20 we are slower than pure Spark and Comet is not, and the
Comet-scan column recovers most of it (the row-based Parquet reader feeding our operators is the
largest single cost we do not own). We win where the join is the work: q21 is 8.3 s over our shuffle
and 7.4 s with Comet's scan against Comet's 11.0 s, and hybrid takes q21 (6.6 s) and q5 (2.9 s
against 4.4 s) outright. Two rows need a footnote: **q11 returns no rows at SF10** in every
configuration -- the runner's text carries the SF1 fraction 0.0001 where the specification scales it
by 1/SF, adaptive execution collapses the empty `HAVING` into an `EmptyRelation` (hence 0/1), and the
row measures fixed cost only (#331); and **q19** under hybrid is the worst of the five because the
work is a string filter over dictionary columns where ours is 2x behind Comet's (#279's measurement,
the #14 decode) plus three `VectorToComet` crossings.

**TPC-DS SF1**, 103 queries:

| configuration | total | vs Spark | total without q72 | vs Spark | operators accelerated | queries at 75%+ | fully accelerated |
|---|---|---|---|---|---|---|---|
| pure Spark | 52.1 s | 1.00x | 49.1 s | 1.00x | 0 / 4718 (0%) | 0 | 0 |
| ours + our shuffle | 83.6 s | 1.61x | 57.5 s | 1.17x | 3771 / 4712 (80%) | 90 | 1 |
| Comet scan + ours + our shuffle | 78.5 s | 1.51x | 52.0 s | 1.06x | 5372 / 6313 (85%) | 99 | 1 |
| native Comet | 35.4 s | 0.68x | 29.6 s | 0.60x | 5705 / 5781 (99%) | 100 | 95 |
| hybrid | 75.7 s | 1.45x | 49.2 s | 1.00x | 5805 / 6652 (87%) | 100 | 1 |

At SF1 the suite is a coverage harness, not a speed benchmark: the queries average half a second, and
the fixed cost per task -- planning our operators, batch and shuffle-stream setup, the row/columnar
transitions around the operators still Spark's (about 640 of them the broadcast exchanges under our
broadcast joins, #325) -- outweighs the kernel work. Two things still read: **q72** (#332) is 26 s in
every configuration with our join, against Spark's 3.0 s -- the shuffled hash join re-expression of
catalog_sales x inventory on `item_sk` (many-to-many, on the order of 10^9 candidate pairs) gathers
both sides of every pair before the residual `inv_quantity_on_hand < cs_quantity`; take it out and
hybrid is level with Spark, Comet's scan in front of us costs 6%, our own scan path 17%. (#332 took
q72 from 26.1 s to **7.4 s** warm over our shuffle -- Comet's merge join reads 5.8 s, Spark 3.0 s --
in three steps: the condition evaluated over a gather of only its columns, output for the survivors
alone; heap mirrors for every fixed-width gather, because the per-element `MemorySegment.get` in a
pair-wise loop compiles to virtual calls -- the JIT's own log says `no static binding` on the
segment's offset lookup, the receiver profile being mixed across the kernels, and
`-XX:TypeProfileLevel=222` alone recovered 14% -- and the residual `lane OP lane` tested per pair on
the mirrors so a failing pair is never appended. The join now reads 28 s of task time for 10^9 pairs
and keeps 56 M; what is left of q72 is downstream of it over those rows. q37 1348 -> 732 ms, q82
1358 -> 1126; TPC-H q21 8.3 -> 7.9 s.) And native
Comet fully accelerates 95 of 103 queries where our best configuration fully accelerates one: the
broadcast exchange is in every other plan.

The heavy queries (pure Spark at 800 ms or more), where the data rather than the fixed cost decides:

| query | pure Spark | ours + our shuffle | Comet scan + ours + our shuffle | native Comet | hybrid |
|---|---|---|---|---|---|
| q72 | 3004 | 26150 (8.71x, 40/49) | 26530 (8.83x, 60/69) | 5820 (1.94x, 63/63) | 26522 (8.83x, 63/72) |
| q14a | 2269 | 1974 (0.87x, 245/357) | 1874 (0.83x, 371/483) | 1318 (0.58x, 415/415) | 1741 (0.77x, 446/510) |
| q23a | 1831 | 1196 (0.65x, 66/82) | 1112 (0.61x, 88/104) | 555 (0.30x, 97/97) | 957 (0.52x, 97/113) |
| q23b | 1662 | 1252 (0.75x, 92/118) | 1171 (0.70x, 122/148) | 704 (0.42x, 139/139) | 1135 (0.68x, 136/162) |
| q4 | 1650 | 2008 (1.22x, 66/90) | 1889 (1.15x, 96/120) | 1128 (0.68x, 109/109) | 1716 (1.04x, 102/126) |
| q14b | 1430 | 1633 (1.14x, 162/235) | 1493 (1.04x, 246/319) | 994 (0.70x, 274/274) | 1327 (0.93x, 296/337) |
| q67 | 1372 | 1486 (1.08x, 23/26) | 1288 (0.94x, 30/33) | 667 (0.49x, 27/32) | 1022 (0.74x, 32/35) |
| q10 | 1352 | 1762 (1.30x, 36/41) | 1679 (1.24x, 50/55) | 1370 (1.01x, 40/56) | 1740 (1.29x, 55/60) |
| q95 | 1348 | 770 (0.57x, 37/43) | 799 (0.59x, 52/58) | 422 (0.31x, 53/53) | 733 (0.54x, 56/62) |
| q64 | 1217 | 1538 (1.26x, 134/169) | 1602 (1.32x, 206/241) | 762 (0.63x, 210/210) | 1588 (1.30x, 213/248) |
| q69 | 1137 | 1584 (1.39x, 35/40) | 1481 (1.30x, 49/54) | 452 (0.40x, 53/53) | 1484 (1.30x, 54/59) |
| q35 | 1128 | 1256 (1.11x, 30/44) | 1132 (1.00x, 44/58) | 1130 (1.00x, 39/55) | 1121 (0.99x, 48/62) |
| q85 | 1118 | 2201 (1.97x, 27/34) | 2287 (2.05x, 42/49) | 468 (0.42x, 43/43) | 2307 (2.06x, 43/50) |
| q25 | 845 | 672 (0.80x, 30/36) | 561 (0.66x, 44/50) | 401 (0.47x, 47/47) | 534 (0.63x, 47/53) |
| q18 | 824 | 1083 (1.31x, 27/33) | 1067 (1.30x, 40/46) | 272 (0.33x, 41/41) | 1010 (1.23x, 41/47) |
| q11 | 824 | 1136 (1.38x, 46/60) | 1053 (1.28x, 66/80) | 665 (0.81x, 73/73) | 964 (1.17x, 70/84) |
| **total** | **23.0 s** | **47.7 s** (2.07x, 1096/1457) | **47.0 s** (2.04x, 1606/1967) | **17.1 s** (0.74x, 1723/1760) | **45.9 s** (1.99x, 1799/2080) |

Where the work is real we beat Spark -- q23a/b, q95, q25, q14a, q67 under hybrid at 0.52x-0.77x --
and lose on the chains of small stages (q85 2.06x, q10/q64/q69 1.3x). Against Comet, hybrid's worst
are q84 (5.3x), q85 (4.9x), q72 (4.6x), q82 (4.0x) and q37 (3.8x): q37 and q82 are the inventory join
shape of q72 at smaller volume, q84 and q85 chains of small joins where our per-task cost shows.

Every query:

| query | pure Spark | ours + our shuffle | Comet scan + ours + our shuffle | native Comet | hybrid |
|---|---|---|---|---|---|
| q1 | 479 | 442 (0.92x, 27/38) | 437 (0.91x, 37/48) | 358 (0.75x, 43/43) | 415 (0.87x, 41/52) |
| q2 | 516 | 550 (1.07x, 31/40) | 554 (1.07x, 43/52) | 533 (1.03x, 49/49) | 599 (1.16x, 49/55) |
| q3 | 167 | 196 (1.18x, 13/15) | 194 (1.17x, 18/20) | 155 (0.93x, 19/19) | 175 (1.05x, 19/21) |
| q4 | 1650 | 2008 (1.22x, 66/90) | 1889 (1.15x, 96/120) | 1128 (0.68x, 109/109) | 1716 (1.04x, 102/126) |
| q5 | 430 | 462 (1.07x, 53/60) | 457 (1.06x, 73/80) | 363 (0.84x, 74/74) | 397 (0.92x, 77/84) |
| q6 | 476 | 446 (0.94x, 28/33) | 443 (0.93x, 37/42) | 256 (0.54x, 40/40) | 415 (0.87x, 41/46) |
| q7 | 267 | 351 (1.31x, 20/24) | 295 (1.10x, 29/33) | 184 (0.69x, 30/30) | 248 (0.93x, 30/34) |
| q8 | 303 | 400 (1.32x, 30/38) | 422 (1.39x, 40/48) | 268 (0.88x, 43/43) | 489 (1.61x, 48/52) |
| q9 | 386 | 425 (1.10x, 2/2) | 321 (0.83x, 4/4) | 242 (0.63x, 3/4) | 340 (0.88x, 4/4) |
| q10 | 1352 | 1762 (1.30x, 36/41) | 1679 (1.24x, 50/55) | 1370 (1.01x, 40/56) | 1740 (1.29x, 55/60) |
| q11 | 824 | 1136 (1.38x, 46/60) | 1053 (1.28x, 66/80) | 665 (0.81x, 73/73) | 964 (1.17x, 70/84) |
| q12 | 163 | 196 (1.21x, 16/18) | 179 (1.10x, 21/23) | 122 (0.75x, 22/22) | 162 (0.99x, 23/25) |
| q13 | 387 | 390 (1.01x, 21/26) | 357 (0.92x, 32/37) | 204 (0.53x, 33/33) | 361 (0.93x, 33/38) |
| q14a | 2269 | 1974 (0.87x, 245/357) | 1874 (0.83x, 371/483) | 1318 (0.58x, 415/415) | 1741 (0.77x, 446/510) |
| q14b | 1430 | 1633 (1.14x, 162/235) | 1493 (1.04x, 246/319) | 994 (0.70x, 274/274) | 1327 (0.93x, 296/337) |
| q15 | 230 | 281 (1.22x, 15/18) | 244 (1.06x, 22/25) | 125 (0.54x, 23/23) | 253 (1.10x, 23/26) |
| q16 | 404 | 582 (1.44x, 22/27) | 602 (1.49x, 33/38) | 272 (0.67x, 34/34) | 594 (1.47x, 35/40) |
| q17 | 723 | 602 (0.83x, 0/1) | 524 (0.72x, 0/1) | 358 (0.50x, 46/46) | 481 (0.67x, 0/1) |
| q18 | 824 | 1083 (1.31x, 27/33) | 1067 (1.30x, 40/46) | 272 (0.33x, 41/41) | 1010 (1.23x, 41/47) |
| q19 | 171 | 253 (1.48x, 22/27) | 194 (1.13x, 33/38) | 114 (0.67x, 34/34) | 172 (1.01x, 34/39) |
| q20 | 206 | 189 (0.92x, 16/18) | 192 (0.93x, 21/23) | 114 (0.55x, 22/22) | 162 (0.79x, 23/25) |
| q21 | 226 | 406 (1.79x, 16/19) | 344 (1.52x, 23/26) | 146 (0.65x, 24/24) | 287 (1.27x, 24/27) |
| q22 | 688 | 934 (1.36x, 16/19) | 885 (1.29x, 23/26) | 299 (0.43x, 24/24) | 842 (1.22x, 24/27) |
| q23a | 1831 | 1196 (0.65x, 66/82) | 1112 (0.61x, 88/104) | 555 (0.30x, 97/97) | 957 (0.52x, 97/113) |
| q23b | 1662 | 1252 (0.75x, 92/118) | 1171 (0.70x, 122/148) | 704 (0.42x, 139/139) | 1135 (0.68x, 136/162) |
| q24a | 535 | 763 (1.43x, 23/29) | 620 (1.16x, 34/40) | 389 (0.73x, 36/36) | 627 (1.17x, 36/42) |
| q24b | 537 | 678 (1.26x, 23/29) | 599 (1.11x, 34/40) | 351 (0.65x, 36/36) | 560 (1.04x, 36/42) |
| q25 | 845 | 672 (0.80x, 30/36) | 561 (0.66x, 44/50) | 401 (0.47x, 47/47) | 534 (0.63x, 47/53) |
| q26 | 268 | 214 (0.80x, 20/24) | 173 (0.65x, 29/33) | 141 (0.53x, 30/30) | 189 (0.71x, 30/34) |
| q27 | 208 | 341 (1.64x, 18/24) | 238 (1.14x, 27/33) | 157 (0.75x, 30/30) | 271 (1.30x, 28/34) |
| q28 | 450 | 612 (1.36x, 53/58) | 413 (0.92x, 59/64) | 255 (0.57x, 65/65) | 408 (0.91x, 71/76) |
| q29 | 769 | 604 (0.79x, 30/36) | 622 (0.81x, 44/50) | 300 (0.39x, 46/46) | 529 (0.69x, 47/53) |
| q30 | 339 | 327 (0.97x, 33/46) | 304 (0.90x, 47/60) | 194 (0.57x, 53/53) | 340 (1.00x, 51/64) |
| q31 | 479 | 560 (1.17x, 65/88) | 412 (0.86x, 94/117) | 287 (0.60x, 106/106) | 392 (0.82x, 102/123) |
| q32 | 229 | 185 (0.81x, 24/28) | 126 (0.55x, 32/36) | 109 (0.48x, 34/34) | 122 (0.53x, 35/39) |
| q33 | 213 | 251 (1.18x, 58/70) | 206 (0.97x, 85/97) | 197 (0.93x, 86/86) | 196 (0.92x, 88/100) |
| q34 | 222 | 286 (1.29x, 22/26) | 232 (1.04x, 31/35) | 137 (0.62x, 32/32) | 242 (1.09x, 33/37) |
| q35 | 1128 | 1256 (1.11x, 30/44) | 1132 (1.00x, 44/58) | 1130 (1.00x, 39/55) | 1121 (0.99x, 48/62) |
| q36 | 229 | 276 (1.21x, 15/24) | 233 (1.02x, 22/31) | 157 (0.69x, 29/29) | 229 (1.00x, 24/32) |
| q37 | 406 | 596 (1.47x, 17/20) | 554 (1.37x, 24/27) | 155 (0.38x, 25/25) | 583 (1.44x, 25/28) |
| q38 | 361 | 493 (1.37x, 39/47) | 472 (1.31x, 54/63) | 203 (0.56x, 57/57) | 369 (1.02x, 58/66) |
| q39a | 565 | 826 (1.46x, 35/42) | 698 (1.24x, 49/56) | 278 (0.49x, 51/51) | 719 (1.27x, 52/59) |
| q39b | 472 | 796 (1.69x, 35/42) | 711 (1.51x, 49/56) | 302 (0.64x, 51/51) | 745 (1.58x, 52/59) |
| q40 | 254 | 293 (1.15x, 18/22) | 283 (1.11x, 27/31) | 138 (0.54x, 28/28) | 257 (1.01x, 28/32) |
| q41 | 119 | 97 (0.81x, 15/16) | 122 (1.02x, 17/18) | 130 (1.09x, 19/19) | 118 (0.98x, 19/20) |
| q42 | 111 | 130 (1.17x, 13/15) | 97 (0.88x, 18/20) | 84 (0.75x, 19/19) | 97 (0.87x, 19/21) |
| q43 | 173 | 184 (1.07x, 13/15) | 144 (0.84x, 18/20) | 92 (0.53x, 19/19) | 130 (0.75x, 19/21) |
| q44 | 287 | 279 (0.97x, 39/41) | 212 (0.74x, 45/47) | 184 (0.64x, 30/50) | 268 (0.94x, 49/51) |
| q45 | 151 | 191 (1.27x, 23/28) | 173 (1.15x, 34/39) | 131 (0.87x, 32/37) | 175 (1.16x, 35/40) |
| q46 | 226 | 323 (1.43x, 26/32) | 246 (1.09x, 39/45) | 184 (0.82x, 40/40) | 218 (0.96x, 40/46) |
| q47 | 743 | 1230 (1.66x, 59/70) | 1192 (1.60x, 80/91) | 425 (0.57x, 86/86) | 816 (1.10x, 86/97) |
| q48 | 426 | 356 (0.83x, 18/22) | 295 (0.69x, 27/31) | 165 (0.39x, 28/28) | 339 (0.80x, 28/32) |
| q49 | 313 | 291 (0.93x, 61/67) | 274 (0.87x, 76/82) | 259 (0.83x, 77/77) | 265 (0.84x, 82/88) |
| q50 | 474 | 488 (1.03x, 18/22) | 453 (0.96x, 27/31) | 177 (0.37x, 28/28) | 414 (0.87x, 28/32) |
| q51 | 640 | 772 (1.21x, 33/35) | 781 (1.22x, 39/41) | 473 (0.74x, 42/42) | 575 (0.90x, 46/48) |
| q52 | 102 | 126 (1.23x, 13/15) | 100 (0.97x, 18/20) | 89 (0.87x, 19/19) | 97 (0.95x, 19/21) |
| q53 | 167 | 178 (1.07x, 21/24) | 149 (0.89x, 28/31) | 118 (0.71x, 29/29) | 143 (0.86x, 30/33) |
| q54 | 577 | 404 (0.70x, 39/45) | 342 (0.59x, 54/60) | 239 (0.41x, 55/55) | 312 (0.54x, 58/64) |
| q55 | 109 | 143 (1.31x, 13/15) | 124 (1.14x, 18/20) | 78 (0.71x, 19/19) | 102 (0.93x, 19/21) |
| q56 | 225 | 251 (1.11x, 58/70) | 200 (0.89x, 85/97) | 172 (0.77x, 86/86) | 190 (0.85x, 88/100) |
| q57 | 472 | 616 (1.31x, 59/70) | 606 (1.28x, 80/91) | 281 (0.59x, 83/83) | 536 (1.14x, 86/97) |
| q58 | 260 | 384 (1.48x, 50/61) | 338 (1.30x, 71/82) | 242 (0.93x, 74/74) | 328 (1.26x, 74/85) |
| q59 | 349 | 410 (1.18x, 32/40) | 373 (1.07x, 46/54) | 248 (0.71x, 49/49) | 344 (0.98x, 51/58) |
| q60 | 232 | 283 (1.22x, 58/70) | 210 (0.90x, 85/97) | 168 (0.72x, 86/86) | 206 (0.88x, 88/100) |
| q61 | 240 | 445 (1.85x, 52/64) | 290 (1.20x, 76/88) | 176 (0.73x, 78/78) | 335 (1.39x, 78/90) |
| q62 | 128 | 378 (2.95x, 18/22) | 194 (1.52x, 27/31) | 100 (0.78x, 28/28) | 134 (1.04x, 28/32) |
| q63 | 168 | 182 (1.08x, 21/24) | 172 (1.02x, 28/31) | 120 (0.71x, 29/29) | 164 (0.97x, 30/33) |
| q64 | 1217 | 1538 (1.26x, 134/169) | 1602 (1.32x, 206/241) | 762 (0.63x, 210/210) | 1588 (1.30x, 213/248) |
| q65 | 410 | 414 (1.01x, 31/36) | 297 (0.72x, 41/46) | 227 (0.55x, 42/42) | 274 (0.67x, 45/50) |
| q66 | 349 | 328 (0.94x, 40/48) | 312 (0.89x, 58/66) | 490 (1.40x, 59/59) | 288 (0.83x, 60/68) |
| q67 | 1372 | 1486 (1.08x, 23/26) | 1288 (0.94x, 30/33) | 667 (0.49x, 27/32) | 1022 (0.74x, 32/35) |
| q68 | 200 | 300 (1.50x, 26/32) | 208 (1.04x, 39/45) | 138 (0.69x, 40/40) | 208 (1.04x, 40/46) |
| q69 | 1137 | 1584 (1.39x, 35/40) | 1481 (1.30x, 49/54) | 452 (0.40x, 53/53) | 1484 (1.30x, 54/59) |
| q70 | 332 | 468 (1.41x, 29/40) | 427 (1.29x, 38/49) | 252 (0.76x, 38/49) | 325 (0.98x, 42/52) |
| q71 | 207 | 260 (1.26x, 29/34) | 196 (0.94x, 42/47) | 148 (0.71x, 43/43) | 201 (0.97x, 44/49) |
| q72 | 3004 | 26150 (8.71x, 40/49) | 26530 (8.83x, 60/69) | 5820 (1.94x, 63/63) | 26522 (8.83x, 63/72) |
| q73 | 207 | 231 (1.12x, 21/25) | 202 (0.98x, 30/34) | 140 (0.68x, 30/30) | 223 (1.08x, 32/36) |
| q74 | 526 | 744 (1.41x, 45/59) | 672 (1.28x, 65/79) | 355 (0.67x, 72/72) | 610 (1.16x, 69/83) |
| q75 | 636 | 825 (1.30x, 85/104) | 750 (1.18x, 127/146) | 434 (0.68x, 129/129) | 694 (1.09x, 131/150) |
| q76 | 221 | 247 (1.12x, 26/32) | 186 (0.84x, 41/47) | 120 (0.54x, 42/42) | 166 (0.75x, 42/48) |
| q77 | 354 | 280 (0.79x, 72/85) | 238 (0.67x, 98/111) | 259 (0.73x, 102/102) | 231 (0.65x, 105/118) |
| q78 | 762 | 1015 (1.33x, 40/48) | 888 (1.17x, 55/63) | 414 (0.54x, 56/58) | 818 (1.07x, 58/66) |
| q79 | 176 | 292 (1.66x, 20/24) | 187 (1.06x, 29/33) | 122 (0.70x, 30/30) | 170 (0.97x, 30/34) |
| q80 | 507 | 659 (1.30x, 72/87) | 553 (1.09x, 105/120) | 345 (0.68x, 106/106) | 615 (1.21x, 109/124) |
| q81 | 210 | 237 (1.13x, 32/45) | 216 (1.03x, 46/59) | 133 (0.63x, 52/52) | 221 (1.05x, 50/63) |
| q82 | 640 | 1030 (1.61x, 17/20) | 994 (1.55x, 24/27) | 264 (0.41x, 25/25) | 1061 (1.66x, 25/28) |
| q83 | 182 | 209 (1.15x, 53/67) | 176 (0.96x, 80/94) | 170 (0.93x, 83/83) | 199 (1.09x, 83/97) |
| q84 | 659 | 947 (1.44x, 19/24) | 884 (1.34x, 31/36) | 177 (0.27x, 31/31) | 928 (1.41x, 31/36) |
| q85 | 1118 | 2201 (1.97x, 27/34) | 2287 (2.05x, 42/49) | 468 (0.42x, 43/43) | 2307 (2.06x, 43/50) |
| q86 | 136 | 165 (1.21x, 11/19) | 148 (1.09x, 16/24) | 91 (0.67x, 23/23) | 124 (0.91x, 18/25) |
| q87 | 394 | 446 (1.13x, 39/48) | 410 (1.04x, 54/62) | 201 (0.51x, 57/57) | 346 (0.88x, 60/68) |
| q88 | 405 | 747 (1.84x, 135/166) | 464 (1.15x, 191/222) | 312 (0.77x, 199/199) | 486 (1.20x, 199/230) |
| q89 | 188 | 221 (1.17x, 20/23) | 191 (1.01x, 27/30) | 134 (0.71x, 28/28) | 152 (0.81x, 29/32) |
| q90 | 80 | 139 (1.75x, 34/41) | 109 (1.37x, 48/55) | 82 (1.03x, 50/50) | 101 (1.27x, 50/57) |
| q91 | 233 | 255 (1.10x, 27/33) | 223 (0.96x, 40/46) | 138 (0.59x, 41/41) | 253 (1.09x, 42/48) |
| q92 | 224 | 134 (0.60x, 24/28) | 113 (0.50x, 32/36) | 90 (0.40x, 34/34) | 110 (0.49x, 35/39) |
| q93 | 446 | 350 (0.78x, 11/13) | 332 (0.74x, 16/18) | 129 (0.29x, 17/17) | 325 (0.73x, 17/19) |
| q94 | 218 | 380 (1.74x, 22/27) | 375 (1.72x, 33/38) | 177 (0.81x, 34/34) | 370 (1.70x, 35/40) |
| q95 | 1348 | 770 (0.57x, 37/43) | 799 (0.59x, 52/58) | 422 (0.31x, 53/53) | 733 (0.54x, 56/62) |
| q96 | 80 | 154 (1.93x, 16/19) | 95 (1.19x, 23/26) | 93 (1.16x, 24/24) | 82 (1.03x, 24/27) |
| q97 | 283 | 549 (1.94x, 15/25) | 503 (1.77x, 21/31) | 170 (0.60x, 30/30) | 319 (1.12x, 23/33) |
| q98 | 228 | 300 (1.31x, 18/20) | 277 (1.22x, 23/25) | 147 (0.64x, 24/24) | 206 (0.90x, 26/28) |
| q99 | 127 | 255 (2.00x, 18/22) | 217 (1.71x, 27/31) | 134 (1.05x, 28/28) | 222 (1.75x, 28/32) |
| **total** | **52.1 s** | **83.6 s** (1.61x, 3771/4712) | **78.5 s** (1.51x, 5372/6313) | **35.4 s** (0.68x, 5705/5781) | **75.7 s** (1.45x, 5805/6652) |

The priorities the two suites agree on: the hash join's residual condition (#332), the short-query
fixed cost (#331), the columnar broadcast exchange (#325); then q19's string filter and the merge join
at scale (#329).

### On the cluster (#247, #248): TPC-DS SF100 Parquet on S3, eight executors

TPC-DS SF100 (27 GB ZSTD Parquet on S3, generated on the cluster), `sfi-iceberg-bench` EKS, node group
`bench-xl` (m5.4xlarge), **eight executors of 13 cores and ~50 GiB each**, the memory placed where each
engine uses it (Spark and our shuffle: 20 GiB heap + 30 GiB overhead, the overhead less 2 GiB as the
JVM's direct-memory limit; Comet: 20 GiB heap + 24 GiB off-heap; the Comet-scan mix over our shuffle:
20 + 14 + 16), driver 4 GiB, one iteration per query, no warm-up. Image `benchmarks/k8s/Dockerfile`,
manifests from `render-run.sh`, report `TpcdsRunner --cluster-report` over the rows on S3.

**The first run (v2) found four defects; the fifth run has our shuffle at zero failures and every checksum
but one equal to Spark's.** The four, each with its own issue and fix:

1. **The Flight shuffle lost dictionary replacements (#338).** `FlightShuffle.Producer` started a stream
   with the block's first dictionaries and Arrow Flight writes dictionaries only at a stream's start,
   while our blocks carry one replacement dictionary per record batch: a remote block of several
   batches decoded against the first batch's dictionary -- `IndexOutOfBoundsException` in every string
   consumer (q4, q11, q23b, q24b, q38, q74, q87), and **silently wrong strings** where the index fit
   (the checksums of q1, q6, q23b, q24a, q24b, q38, q39a, q39b, q74, q87 disagreed with Spark's). The
   data plane now sends the block's IPC bytes as they are. Local reads never showed it.
2. **The shuffle writer's memory was not what it accounted (#340).** q67 took every executor down in
   four seconds: an Arrow `OutOfMemoryException` from the writer (one task's allocator at 1.14 GB
   against a 64 MB budget: the estimate ignored `setSafe`-grown capacity and the per-slice
   dictionaries; `ArrowStreamWriter` kept a copy of every dictionary it ever wrote), which Spark could
   not serialize, and on JDK 25 `SerializationDebugger`'s initializer then dies (SPARK-55679, fixed in
   4.2.0 only) -- exit 50. Now: flushes by the allocator's real allocation, one IPC stream per record
   batch, a limited writer allocator, and Arrow's OOM rethrown as a serializable `SparkException`.
   The new test also caught **arrow-java 18.3.0's zstd codec writing 8 bytes past its buffer**
   (GH-1116, fixed upstream for 20.0.0): the shuffle uses its own correctly-sized codec.
3. **The partitioner's hash loop deoptimized a hundred times per JVM (#343).** JFR on q79 at SF1: 37%
   of CPU in `PartitionKernels.mixColumn`, 101 `profile_predicate` traps on its null-row branch. One
   loop per key kind and validity shape, null rows blended by a mask: q79 at SF1 1984 -> 857 ms.
4. **Every partition slice of a dictionary-encoded string column carried the whole dictionary (#345).**
   The runner's `--explain` at SF100 showed q79's `customer` exchange at 1,623 MB against Spark's
   52 MB with identical plans and record counts: at 200 partitions a slice holds ~41 rows and copied a
   dictionary of thousands of names, and the flush concatenated hundreds of such copies per batch.
   That was the 15-25x shuffle-read gap and the whole 20x on q73/q79/q34/q68/q46/q30 (q79: 55.6 s ->
   6.3 s). A slice now carries only the entries it uses.
5. **The Flight reader fetched one block per round trip, in sequence (#347).** With the four above
   in, most queries were still 1.5-2x -- the small ones included -- at the *same* executor time as
   Spark's (q3: 62 s of executor time to Spark's 64, 2.75 s of wall to 1.39). Wall without CPU is
   waiting: a reduce task's input is a block per map task, and the backend opened a `DoGet` per block,
   one after another -- ~100 sequential gRPC stream setups per reduce task at SF100. Now one call per
   (executor, reducer) carrying all of that executor's blocks, the calls to every executor opened
   together: 0.58x -> 0.73x on the suite.

Two more things the cluster taught: a node's 20 GB root disk fills with the shuffle files of finished
queries (the driver's `ContextCleaner` removes them after a GC, every 30 min by default; the kubelet
evicted an executor at query 95 -- `spark.cleaner.periodicGC.interval=2min` in the manifests), and
hadoop-aws 3.4's default credential chain has no IRSA (`WebIdentityTokenFileCredentialsProvider` set).

**v8 (the five fixes and the writer rework: #349 plain slices encoded once, #351 per-partition
builders, #353 index-list gathers), all six configurations, 101 of 103 queries comparable, no failures
in any configuration:**

| configuration | suite wall (s) | failed | comparable total (s, 101 queries) | vs Spark |
|---|---:|---:|---:|---:|
| spark | 705 | 0 | 556.1 | 1.00x |
| vector-shuffle | 736 | 0 | 639.1 | 0.87x |
| vector-shuffle-strict | 736 | 0 | 639.3 | 0.87x |
| comet-scan-vector-ourshuffle | 674 | 0 | 575.5 | 0.97x |
| hybrid | 552 | 0 | 471.2 | 1.18x |
| comet | 520 | 0 | 441.6 | 1.26x |

`vector-shuffle-strict` -- bit-identical floating point against the benchmarks' fast default -- costs
**0.03%** (639.3 s against 639.1), noise. 
`comet-scan-vector-ourshuffle` is Comet's native Parquet scan feeding our operators over our shuffle
(its v2 reading: 0.78x with 3 failures; v7: 0.93x). The runs before: v2 (the first, 87 comparable) vector-shuffle
0.45x with 9 failures; v3 after #338: 0.33x, 2 failures; v4 after #340: 0.35x, 5 failures all one
evicted executor; v5 after #343 and #345: 0.58x, none; v6 after #347: 0.73x; v7 after #349 and #351: 0.87x, the mix
0.93x. hybrid and comet in v2 read 1.20x and 1.29x on the unfixed engine.

Where the numbers now live (v7/v8): under `vector-shuffle` 30 queries are faster than Spark (q97 1.96x,
q94 1.77x, q52, q82, q51, q42 1.5-1.6x, q29, q93 1.44x), 27 within 10%, 32 more than 20% slower --
q30 (4.35 -> 9.88 s) and q8 (2.71 -> 6.06) at the top, then q1, q39a/b, q46, q67, q72, q4 around 1.5-2x.
Under the Comet-scan mix 35 are faster (q97 2.25x, q94 1.90x, q82 1.86x, q42, q52, q41 1.75-1.84x),
27 within 10%, 26 more than 20% slower -- q22 (5.1 -> 14.3 s, the inventory rollup), q8, q39b, q46,
q30, q1. A JFR of q30 at SF10 after these changes has the shuffle write at a third of the query's CPU
(from 63% before #349) with the Parquet-to-Arrow adapter and the joins the rest; #353 (index-list
gathers, in main after this run: q30 at SF10 2377 -> 1963 ms) is not in this image. q65's checksum
differs from Spark's in every configuration but Spark (100 rows each: a tie in its `LIMIT` order, to
be confirmed); q64 returns 0 rows under Comet's scan (a Comet 1.0 issue). Per-query tables:
`results/sf100-parquet-v7/cluster-results.md` on the results bucket.
**1 TB (#248, `results/sf1000-parquet`, the same eight executors, 200 reduce partitions).** The
baselines ran first: spark, hybrid, comet, no failures. Our two configurations took three attempts:
the first filled the 20 GB node disks within minutes because our shuffle never deleted its map outputs
(#358); the second lost five executors to q78's aggregate over (item, customer), which our hash
aggregate kept entirely in memory (#363, the hash-partitioned spill; #364, a remote fetch error is
now Spark's `FetchFailedException`, so the scheduler recomputes the lost map outputs instead of
failing the query -- the cascade after q78 shrank from 17 queries to 7 while executors were replaced);
the third, with the spill's budget counting the accumulators at their interleaved, doubled capacity
and arbitrated by Spark's task memory manager (#367), ran all 103 with no failure and no executor
lost:

| 1 TB, 100 comparable queries | total (s) | vs Spark | failed |
|---|---:|---:|---:|
| spark | 3029.9 | 1.00x | 0 |
| vector-shuffle | 2809.4 | **1.08x** | 0 |
| comet-scan-vector-ourshuffle | 2541.7 | **1.19x** | 0 |
| hybrid | 2246.3 | 1.35x | 0 |
| comet | 2091.3 | 1.45x | 0 |

Excluded: q64, where Comet's scan returns no rows (Spark 12,185), and q65, where every accelerated
configuration -- pure Comet included -- returns 100 rows with a checksum different from Spark's
(the query orders by store name and item description with ties at this scale; not yet confirmed).
Our shuffle is faster than Spark at 1 TB on the whole suite, with the wins where the shuffle is the
work (q23b 286 -> 132 s, q5 47 -> 22, q29 23 -> 11, q97 33 -> 16, q23a 207 -> 124; q78 spills and
completes in 99 s against Spark's 114) and the losses where it is not: q8 6.8 -> 41 s (the widest,
a broadcast-side plan under study), q67 126 -> 178 (the rollup window), q72 27 -> 42, and a band of
small queries at 0.5-0.65x (q18, q19, q22, q99, q39b, q47, q57) whose exchanges are a few MB and
whose time is our per-stage overhead. The mix of Comet's scan with our operators and shuffle is
1.19x with its own regressions at q14b 95 -> 144 s and q72 27 -> 36.

**The losses, one root cause at a time (targeted 1 TB runs, each with a Spark baseline in the same
run, heap 30 g / overhead 20 g / direct 30 g, single iteration).** Each fix was measured at SF10
first and then on the cluster; the executor profiles that ranked the work came from JFR on all
eight executors, and one lesson of the series is that a recording's window must cover the stage
under study -- the first profiles covered the executors' first 150 s and could not see q67's sort
stage at all.

| query | Spark | before | after | fixes |
|---|---:|---:|---:|---|
| q8 | 6.9 s | 40.8 | 18.5 | string `IN` lists through a hash set (#371); the remaining broadcast join on a 2-character zip prefix is #378 |
| q67 | 114.4 | 178 | **100.2** | rollup as one chain (#383), writer dictionary without `ByteBuffer`s (#387), limb-based wide sum merge and geometric state growth (#388), group keys as one record per group (#377), and the sort's input materialised as one copy per column instead of one per value (#394) -- the last was 4470 s of task time |
| q22 | 8.7 | 14.6 | **6.4** | the rollup chain (#383) |
| q18 | 9.5-11.3 | 26.8 | 11-20 | memory split (#376); the rest varies run to run and is scan-side (#384) |
| q47 | 14.4-16.3 | 22.7 | 19.9-22.7 | scan-side (#384) |
| q57 | 7.5-8.1 | 11.5 | 10.2-11.3 | scan-side (#384) |
| q72 | 26-27 | 41-42 | 38-40 | the fused residual (#332) removes under 0.1% of the pairs at this scale; the remaining gap is six ~1.58 B-row joins each materialising its output, an architecture question |

q67 is faster than Spark for the first time. The small-query band is not the operators: with our
operators on Spark's own shuffle the gap is already there (q47 20.6 s, q57 10.8, q18 16.7), the
executors' task time is 30-60% above Spark's for the same scans, and their profile shows the task
threads parked in the S3 reader's `awaitData` for more than half of that time with a dozen samples
in our kernels -- a reader-side question, recorded on #384.

**Two full runs after the fixes, and what they taught about the measurement.** The suite again,
vector-shuffle only, on the code through #400 (v4, on-heap reader batches) and through #404 (v5,
`spark.sql.columnVector.offheap.enabled=true`), against the clean run's Spark results; 102
queries ran in all three:

| | Spark (run 3) | vector-shuffle, run 3 | vector-shuffle, v4 | vector-shuffle, v5 (off-heap) |
|---|---:|---:|---:|---:|
| total, 102 queries (s) | 3146.0 | 2925.4 | 2898.0 | 2931.2 |

The fixes show where they were aimed (q67 178 -> 103, q8 40.8 -> 6.6, q22 14.6 -> 6.5), but the
totals barely move, because a second set of queries went the other way: q9 85.6 -> 110, q88 133 ->
147, q94 44 -> 60, q90 28 -> 38, q51 19 -> 30, q2 39 -> 50. Those are not in the code. At SF10 the
code through #400 is equal or faster than the code before it on all six; on the cluster, the two
images run in adjacent windows each with its own Spark leg put the *older* code behind on every
one (q9 148 vs 109 s, q88 190 vs 149), while Spark itself moved 20-40% between the two windows
(q88 153 -> 128 s in ten minutes). The scan-bound queries at 1 TB are read-bound, and a single
iteration's time is the cluster's read throughput of that window as much as it is the engine's.
The rule that follows, applied to every targeted run since: **a loss at 1 TB is only a loss
against a Spark leg in the same window**; comparing with another day's baseline finds weather.

Off-heap reader batches were expected to help (SF10: q8 1.46 -> 1.14 s) and cost 1.1% instead,
the scan-heavy `store_sales` queries 5-10%. The reason was ours: the adapter's bulk validity and
dictionary paths (#398) read the reader's arrays only from `OnHeapColumnVector`, so off-heap
batches fell back to their per-row forms on every nullable or dictionary-encoded column (#407
fixed it: the native arrays are copied out once and take the same paths; SF10 q88 4.42 -> 3.79 s,
q28 4.08 -> 3.30). Paired 1 TB runs after it: q88 125.9 s against Spark's 125.6 in the same window
(1.40x behind in v5), q28 1.17x, q13 1.16x. The merge join under a top-N that q47 and q57 left to
Spark (#402: the second self-join's input is the first join, which has no stage of its own, and
the logical product read as 10^16 bytes) converts now -- SF10 q47 17.96 -> 8.68 s -- and did not
move the 1 TB time (22.3 vs 16.0), which places those two in the read-bound band with q18. The
join probe now compares plain integer keys from heap arrays mirrored once per batch instead of
through per-row segment reads (#409; q88 -9% at SF10); a per-column-chunk cache of the decoded
dictionary was tried on the same profile and measured slower twice (#408) -- the cost there is the
gather through the decoded table, not the decode.

**1000 shuffle partitions (v6).** The suite once more, both engines in one window, with
`spark.sql.shuffle.partitions=1000` and a 128 MB advisory partition size (the setting planned for
3 TB at 2000). Over the 100 comparable queries Spark took 2864.6 s and we took 3011.9 -- 1.05x
slower, where at 200 partitions we were 1.04x faster over the same queries. The partition count cost
Spark 4.6% and us 14.8%, and the difference is entirely the shuffle-heavy queries (q67 103 -> 142 s,
q18 10.8 -> 30.5, q95 93 -> 114, q75 70 -> 91, q4 106 -> 122, q84 20 -> 32, q35 20 -> 26); the
scan-bound ones moved with the window for both engines. Two notes on AQE first: it *does* coalesce,
but with `coalescePartitions.parallelismFirst=true` (Spark's default) only down to the core count,
using the advisory size as a cap -- the driver log reads `advisory 134217728, actual target 4317642`
-- and coalescing groups reducers, not the blocks the mappers wrote: every map task still writes
1000 blocks, and a coalesced task reads ten of them per map instead of two.

Why our shuffle pays for that and Spark's does not, in the order it was found (#411): a reduce task
opened one Flight stream per (executor, reduce partition) -- 70 streams per coalesced task instead
of 7 -- and the server served each (map, partition) block with its own index lookup and file open;
range tickets (#412: one stream per executor per task, one lookup per map for the task's whole
range) returned q84 to parity (31.7 -> 22.5 s, Spark 22.8) and changed nothing else. The rest
reproduces on one machine with no network at all (SF10, 1000 partitions: q35 7.6 s against Spark's
1.8, near parity at 200), and the profiles show not a hot frame but the per-record-batch machinery
paid five times as often at a fifth the bytes: on the writer a schema message, a dictionary batch
per string column, a record batch and an end marker per batch, three FlatBuffers builds and a
dozen Arrow allocations with their accounting; on the reader an `ArrowStreamReader` per block.
Writing the messages directly with the schema cached, and caching the Spark type parsed from each
field's metadata (#413), took single digits off. Spark's sort shuffle pays none of this per block:
its blocks are byte spans with no framing. The step that makes the partition count as irrelevant
for us as it is for Spark is one IPC stream per map output with the per-batch shape carried as a
flag rather than a schema per batch -- a file-layout change, put to the owner on #411. Until it
lands, 200-400 partitions is the right setting for 1 TB with this shuffle.

**The partition count, taken apart on q18 (#416).** With the per-block machinery reduced, q18 at
1000 partitions was still 26 s against Spark's 9.5, and each fix below was measured as a paired 1 TB
leg in one cluster window plus an SF10 A/B at 200 partitions (the local baseline moves 15% between
runs; anything smaller was confirmed twice). Staging the map output once and partitioning it at the
flush instead of holding builders per partition (#425: 21.8 -> 19.2 s); strings plain under 256 rows
and the reader coalescing small plain batches to 1024 rows (#426, -4%); the root cause of the
partition count itself (#427): our filter kept `IsNotNull`-guarded attributes nullable where Spark's
`FilterExec` does not, so the AQE-replanned broadcast join's required hash-relation mode differed
from the exchange's, `ValidateRequirements` failed, and `CoalesceShufflePartitions` was silently
dropped for the whole stage plan -- the two shuffled joins ran 1000 tasks over 1000-partition inputs
(18.4 -> 15.2 s, executor time -19%, shuffle bytes -34%). Then the map side: the wide-decimal average
merge state as two 64-bit limbs (#428), the parquet dictionary decoded once per column chunk instead
of once per 4096-row batch (#429, SF10 executor -13%), wide decimal sum/avg buffers written as lanes
instead of boxed values (#430, -13%, GC -37%), dictionary decimals of up to 18 digits through the bulk
path (#431). Dropped after measuring flat: bulk-copy coalescing in the reader, hoisting the merge
loop's lookups, and a flush that gathers once per column and cuts small partitions as slices.

What was left -- the reduce stages at 1000 partitions 7x slower per task than the same tasks at the
tail of the stage, Spark's 2.6x -- turned out to be warm-up. q18 run five times in one application:
the first iteration's rollup partial-aggregate stage takes 310 s of task run time and the final
aggregate 240, iterations two to five 52-68 and 38-57; Spark's own first iteration is 145 and 110, its
warm ones 48-58 and 35-45. Cold, we spend 1100 s of executor time against Spark's 690; warm, 330-400
against 270-340, and the wall clock is the same 5.6-5.8 s. A JFR over the first iteration's reduce
stages shows why: 1650 deoptimisations, and 14-38% of the samples with an interpreted top frame in a
handful of generic kernels (`gatherFixed`, `decodeDictionaryInto`, the string sort passes, the
wide-decimal average) as each stage brought a new combination of buffer type, lane type and call
shape. Every single-query 1 TB figure in this series is a cold first iteration (`run-matrix.sh`
runs one iteration, no warm-up); the campaign's single application amortises the warm-up unevenly
across its query list. None of the JVM's levers moved it at first: the JDK 25 AOT cache mapped but carried no
AOT-linked classes and no method profiles (the incubator module in the graph -- see the v7 passage
below), and compile thresholds and speculation knobs were flat across six legs. Making the
two hottest kernels branchless, each shape its own method (#432), halved their deoptimisations and
took 4-6% off the cold stage at no warm cost; the rest is first execution itself, and whether the
per-query legs should warm up first is a measurement decision, not an engine one.

The same profiles put q67's remaining cost (#377) in its rollup stage's shuffle write -- 2400 of
5100 s of task run time writing four string keys per grouping level, gathered per partition and
dictionary-encoded per block from scratch -- and not in the group table's compares: binding the
batch's key columns once per assign (#433, the build-side twin of #409's `ProbeKeys`) removed the
per-row interface calls and their segment checks, 3.5x on a gather microbenchmark, and moved q67 2%.
The strings the aggregate emits are the table's own distinct values; carrying them through the
writer as dictionary ids is the change that fits.

**v7: the suite at 1000 and at 300 partitions, both engines in one window each.** The image is
main after the #416 series and #433; the methodology is v5/v6's (one application per engine, one
iteration per query, no warm-up). At 1000 partitions, over 98 comparable queries, Spark took 3128.9 s
and we took 2578.0 -- 1.21x -- where v6 had us 1.05x behind at the same setting; query by query
against v6 (84 queries with a median in both) we improved 11.3% (2606.7 -> 2313.0 s) while Spark's
leg regressed 12.1%, so the honest reading of that pass is "we improved 11% and Spark had a bad
hour". At 300 partitions, over 102 comparable queries with both legs healthy, Spark took 3537.3 s
and we took 2936.3 -- 1.20x, 17% less time, 49 queries at least 10% faster (38 of them 20%), 45
within 10%, 8 slower. The partition count now moves us less than it moves Spark (ours 2606 -> 2562 s
from 1000 to 300 over the 99 queries in both passes, Spark 3161 -> 3039), the reverse of v6; Spark
itself is 8.6% slower at 300 than at 200 because 300 is above `spark.shuffle.sort.bypassMergeThreshold`
and its map side changes writer. The v6 losers moved as intended -- q67 142 -> 97 s (1.25x over Spark
at 1000 partitions, level at 300), q4 122 -> 77, q18 30.5 -> 9.2 (within 6% at 300), q84 32 -> 20,
q35 26 -> 13, q19 7.1 -> 3.4 -- and the ones that survive both settings are a short list: q99 (73%
behind at both), q36, q72, q47/q57 (the Sort fallback after `AQEShuffleRead`), q88, q30.

Spark's "bad hour" had a cause worth recording because it will return at 3 TB. In its 1000-partition
leg it lost q23b, q24a, q24b and q25: the kubelet evicted three executors for **ephemeral storage** --
the bench nodes have a 20 GB root volume, Spark's shuffle directory is an `emptyDir` on it, q23a and
q23b each write 83 GB of shuffle (10.4 GB per node) and the finished query's files stay two minutes,
so the two overlap and cross the kubelet's 10%-free threshold; the node sits under `DiskPressure` for
the five-minute transition period, every replacement executor is refused at admission, the next three
queries fail within a second on the dying executors, and q26-q28 run on five executors (q28 287 s
against 165 in v6). The same eviction happened in v6's Spark leg (q23b, q24a) and reproduced on demand
with the free-space curve sampled: all eight nodes within 0.2-1.0 GB of the threshold sixteen seconds
into q23b's map stage. Our leg never gets there because our shuffle for the same two queries is 38 GB
each, 46% of Spark's bytes. The fix is a larger root volume for the node group, not code.

**The AOT cache, second attempt (#416).** The earlier conclusion -- that `--add-modules` disables the
archived boot layer -- was wrong in its cause: JEP 483 allows `--add-modules`; what `ModuleBootstrap`
refuses is a configuration containing an *incubator* module, and `jdk.incubator.vector` is one until it
leaves incubation. A runtime linked from Corretto 25's jmods (Temurin stopped shipping jmods at 24)
with `jdk.incubator.vector` rebuilt without its `ModuleResolution` attribute and `java.base` without
its `ModuleHashes` attribute -- same JDK build, same JIT, only module metadata -- archives the boot
layer: the executor cache holds 18,970 AOT-linked classes (14,396 of them ours and Spark's) and the
JEP 515 method profiles, 190 MB. q18 at 1000 partitions, cold, alternating with the plain image in one
window: 15.9/16.2 s wall and 1155/1167 s of executor time without the cache, 10.7/11.0 s and 702/707 s
with it (-39%); the two reduce stages that carried the deoptimisation storm halve (296 -> 123 s,
238 -> 109 s), about halfway to their warm figures, and the gap to Spark's cold q18 goes from 2.3x to
1.6x with no kernel change. That cache was trained on q18 itself; a cache trained on q67, q22 and q4 and measured on q18, which it never saw, still gives
12.0/12.3 s and 828/816 s against 15.1 s and 1120 s in the same window (-27% executor time), the
class linking being shared by every query and the query-specific profiles adding the rest. The
cache is therefore worth baking into the image at build time from a synthetic training workload
over every operator; it is the cheap half of the cold-start cost, the branchless kernels the other.

**The AOT cache, productized: trained on the cluster, not at build (#416).** The build-time cache was
built, accepted by the executors, and did nothing at 1 TB (q18 16.0/15.9 s against 15.3/15.2 without
it): a local training run in the Dockerfile links the driver's and local-mode paths, while an
executor's cold minutes go to the S3A/Parquet client, the Flight transport over the network and the
executor backend, none of which a local run can reach. The cache now comes from the cluster.
`benchmarks/k8s/aot/train-cluster.sh` runs the vector-shuffle configuration over a training set
(q18, q67, q22, q4 by default) with the executors in `-XX:AOTMode=record`, each writing its
configuration to a per-node directory under the image tag; a DaemonSet then assembles the cache on
every node holding a recording (`assemble.sh`, the executor's exact option set from `aot-env.sh`,
which the executor's JVM checks against its own) and uploads it under the image tag; `render-run.sh`
fetches that object in an init container at every executor start and points the JVM at it, and an
image without a trained cache simply runs without one. Seven minutes of cluster time per image; the
cache is 188 MB with 19,277 AOT-linked classes. Measured cold at 1000 partitions, alternating legs
in one window: q18 **11.2 / 10.9 s** with the fetched cache against 15.9 / 15.1 without (-29% / -27%),
q67 unchanged (112.2 / 107.2 against 109.6 / 109.7; not warm-up bound). The build-time run stays in
the Dockerfile as the smoke test of the runtime -- the build fails when no class links -- and its
cache is discarded. One lesson that cost a run: the kubelet creates the hostPath directory root-owned
and the executor runs as the image's user; a JVM in record mode that cannot open its configuration
file dies at start, and every executor did until a root init container opened the directory first.

**The AOT cache, measured on the heavy queries and switched off for the runs (#248, #416).** The
q18 gain above is a cold-start gain, and the heavy queries pay for it. Once every full run had the
trained cache and every diagnostic window ran without it, the two disagreed on the same image and the
same plans by 10--30 % on q9, q28, q23a, q24a and q67, always in the cache's disfavour. Measured
directly, alone on the cluster, back to back at 300 partitions (v23, main at #462): cache on / off --
q9 **122.4 / 110.7 s** (+11 %), q28 **152.4 / 133.5** (+14 %), q23a **149.4 / 121.7** (+23 %), q14a
**146.2 / 92.6** (+58 %), q4 **85.4 / 78.2** (+9 %); 655.8 against 536.7 s over the five, **+22 %
with the cache**. The cache's profiles come from a four-query training run and drive the JIT's early
decisions for the operators' hot loops; on a query that runs for minutes those decisions are worse
than the ones the JIT makes on its own from the query's own profile, and the start-up seconds saved
are a rounding error against it. The decision rule, fixed before the leg ran: the cache stays on for
the benchmark runs only if the five queries' total with it is within 5 % of without; it was not, so
every benchmark run from here (and every number in the tables that follow) is measured **without the
AOT cache**; the pipeline stays in the tree for what it is good at -- a short-query, cold-start
deployment -- and `AOT_CACHE=0` is the run scripts' default. The `ours` full runs earlier in this
section that carried the cache (v18, v19, v23 at 300 partitions) read 3--10 % over what the same image
reads without it on the heavy half of the suite, and are not to be compared with cache-off numbers
query by query.

**The per-block cost of the shuffle at 1000 partitions, taken apart (#416, items 1-6).** With a
thousand reduce partitions a map task's output to one reducer is a few hundred rows, so everything
that is paid once per block -- a dictionary per string column per block, a 64 KB chunk buffer, a
record batch's header, a kernel's set-up on the reduce side -- is paid a thousand times per map. The
series: one dictionary per string column per map file with a 12-byte unit header per IPC message
(#439; the map's dictionary section is prepended to every partition fetch by the Flight producer), a
producer chunk buffer that starts at the chunk size (#441), a column frozen before its first flush
decoding its pending ids in place instead of carrying a dictionary it will not use (#442), the
reader's decode of a small encoded block through heap arrays instead of per-element `MemorySegment`
accesses (#443), and an encoded block of 128 rows or more handed to the operators as ids rather than
decoded into a coalesced plain batch (#444). The transport benchmark
(`FlightShuffleBenchmark`, two loopback Flight servers over real map files) is the ruler for each
step: the reduce task at 1000 partitions over mixed strings went 8.39 -> 2.00 ms across the series,
eight concurrent reducers 17.98 -> 3.59 ms. On the cluster the rollup stage of q67 writes 16% fewer
bytes and 25% less shuffle-write time (1968 -> 1445-1748 s at 300 partitions). The wall clock was a
longer road: the first image with the series (v13) was 12-32% *slower* on q67 at 1000 partitions
because #438's coalescing decoded every small encoded block into plain strings and the aggregate
then hashed UTF8 bytes where it had consumed ids -- a JFR of the final stage put a quarter of the
samples there -- and two attempts were needed to take it back (#443 removed the decode's segment
checks, #444 restored the id path for blocks above a floor). Where it ended, in one window each:
1000 partitions q67 v15 111.2 / 110.7 s against 106.1 s before the series (about 5% behind, and the
16% smaller shuffle makes AQE coalesce the final stage into 125 tasks instead of 143 -- a worse tail
on 104 cores, a consequence of the smaller shuffle rather than a cost in it); 300 partitions q67
95.8 s against 108.1 / 102.8 (ahead), q18 within the window's drift at both. Two findings for the
record: coalescing encoded blocks *as ids* is not available at 1000 partitions because consecutive
blocks come from different maps with different dictionaries, and the transport benchmark's short
strings did not show the decode cost that the 1 TB profile did, so the benchmark is the ruler for
the transport and the profile for the operators.

**The dictionary-id design, closed (#377).** After the three-step design (#436: string group keys by
id through the group table, the aggregate's output and the shuffle writer) the final aggregate at
1 TB read slower on ids at 1000 partitions, and #437 put the result modes on contiguous byte records;
then a 300-partition reading in another window pointed the other way, and #445 put every mode back
on ids. The decision was taken in one quiet window (the untouched rollup stage within 2-3% across
four legs), q67 at 300 partitions, records / ids / records / ids: the final stage on ids uses 6.5% and
11.5% more executor CPU than on records (3777 / 3932 s against 3546 / 3525) and *less* run time and
wall (4026 / 4183 against 4195 / 4232; 51.1 / 49.0 s against 53.6 / 50.3), because the record layout's
GC is twice the id layout's (32-37 s against 19-22) and GC pauses are run time that is not CPU. Ids
stay everywhere; the record layout is kept in the kernel behind a flag. The two earlier readings
that disagreed were both two-window comparisons; the rule that came out of it is the one this
document already uses for engines: a design decision at 1 TB needs both sides in one window with a
control stage that the change does not touch.

**q64 at 1 TB: a Comet native-execution issue, filed (#248).** The one result that differed from
Spark's in the full runs -- q64 returning 0 rows in the two configurations with Comet's scan, 12,185
in ours and Spark's -- was taken apart to the last discriminator: a new `comet-scan` configuration
(Comet's scan alone, Spark's operators and shuffle) returns 12,185, with the event log confirming it
is the same `CometNativeScan` (48 instances in the plan) that feeds the two failing configurations.
So the scan's values are right when read through Spark's `ColumnarToRow`, and the rows are lost only
when the batches are consumed by a native operator pipeline -- Comet's own or ours -- through the
Arrow C data export; the loss sits in the second `cs_ui` instance's join against the broadcast of
`store_sales(2000) ⋈ store_returns`, and SF10 never reproduces it. Reported upstream as
apache/datafusion-comet#6133 (intermittent -- see the four-configuration table below); q64 stays marked as a Comet issue in the tables and our own row is right.

**Comet's scan against ours, per query (v10, 300 partitions, one window).** Over 101 queries our
scan totals 2752 s and Comet's native scan over the same operators and shuffle 2730 s -- a wash that
is two large effects cancelling: 22 queries are more than 10% faster with Comet's scan (the wide
`store_sales` aggregates: q28 134 -> 96 s, q67 116 -> 93, q9 105 -> 86, q44 -24%, q59 -16%), 21 are
more than 10% slower (join-heavy plans over the smaller fact tables: q10 6.7 -> 16.2 s, q36 +70%,
q94 +48%, q90 +26%, q95 +23%). The best of both per query would be 2581 s, 6% under either. The wins
say Comet's reader decodes wide Parquet scans about 20% faster than ours; the losses say the boundary
gives it back -- Comet's vectors cross into our operators through the Arrow C export and lose our
reader's dictionary encoding of strings and our batch sizing, so joins and aggregates on string keys
run on plain strings, the input-side twin of the #416 finding above. The converter is the cheaper
lever; the reader the larger project.

**The four configurations on one day (v19 image, 300 partitions, 2026-09-23).** The v19 image is main
at #451: the 32 MB spilling sort (#448/#451), the merge join under `auto`, the cluster-trained AOT
cache; Spark's leg ran in the morning, the other three in the afternoon on the same cluster, one run
per query. `comet` is Comet 1.0.0 end to end (scan, native operators, native shuffle); `csvo` is
Comet's scan under our shuffle and operators; `ours` is our reader, shuffle and operators. Seconds.

| query | Spark | Comet | csvo | ours |
|---|---|---|---|---|
| q2 | 42.8 | 46.9 | 39.5 | 52.3 |
| q4 | 93.2 | 58.8 | 77.6 | 96.8 |
| q9 | 95.5 | 79.8 | 76.4 | 142.1 |
| q11 | 43.0 | 35.2 | 42.1 | 53.1 |
| q14a | 100.8 | 76.4 | 96.6 | 123.7 |
| q14b | 97.7 | 66.6 | 80.3 | 93.6 |
| q16 | 36.6 | 20.7 | 21.9 | 27.1 |
| q23a | 210.9 | 137.0 | 125.5 | 148.5 |
| q23b | 293.9 | 163.8 | 134.2 | 172.8 |
| q24a | 113.2 | 101.1 | 116.0 | 134.6 |
| q24b | 110.6 | 98.8 | 110.0 | 130.8 |
| q28 | 116.6 | 107.7 | 101.0 | 169.8 |
| q38 | 32.6 | 19.8 | 18.1 | 21.6 |
| q44 | 38.8 | 38.9 | 32.2 | 40.8 |
| q49 | 43.7 | 51.9 | 37.1 | 41.4 |
| q50 | 69.6 | 58.2 | 38.8 | 41.6 |
| q51 | 32.8 | 14.0 | 15.7 | 23.4 |
| q59 | 37.6 | 38.5 | 36.7 | 42.8 |
| q64 | 95.6 | 68.0 | 60.7 | 62.2 |
| q65 | 30.1 | 16.8 | 22.7 | 27.9 |
| q67 | 127.6 | 60.6 | 98.3 | 114.6 |
| q72 | 33.1 | 38.2 | 34.4 | 39.9 |
| q74 | 43.9 | 30.5 | 43.3 | 42.2 |
| q75 | 75.4 | 80.5 | 76.8 | 76.5 |
| q76 | 42.5 | 49.9 | 47.0 | 46.2 |
| q78 | 123.3 | 80.7 | 103.5 | 111.0 |
| q80 | 54.1 | 48.2 | 46.5 | 43.2 |
| q87 | 33.8 | 19.2 | 17.0 | 23.4 |
| q88 | 140.6 | 141.3 | 131.8 | 142.2 |
| q90 | 41.5 | 32.0 | 37.4 | 28.1 |
| q93 | 143.8 | 90.9 | 69.9 | 67.9 |
| q94 | 68.3 | 53.0 | 61.5 | 56.6 |
| q95 | 144.1 | 68.3 | 61.3 | 82.4 |
| q97 | 37.4 | 20.4 | 17.6 | 23.3 |
| 68 queries under 30 s | 563 | 442 | 452 | 515 |
| **all 102** | **3408** | **2555** | **2581** | **3059** |

`csvo` is faster than Spark on 85 of 102 queries (24% less total time) and
faster than Comet on 52 of 102, its total within 1% of Comet's: ahead on the join-heavy
shapes (q23a/q23b, q50, q64, q93, q95 -- our shuffle, hash joins and the spilling merge join under
Comet's scan), behind on the scan-and-aggregate ones where Comet's native aggregate follows its own
scan with no boundary (q4, q67, q74, q78). `ours` is faster than Spark on 64 of 102 but its
leg ran degraded -- right after a node-group stall, with the join window pulling from S3 on the other
node group -- and shows it on scan-bound queries the image did not touch (q9 142 s where the same
image reads 76 under Comet's scan and its own v18 leg read 106; q28 170 versus 137); its figure is the
reader gap plus that noise and is re-measured on a quiet cluster before it is read as one. Comet's own
leg is 4% faster today than its run of the day before on the same image (2555 versus 2660 s) -- the
run-to-run band on this cluster, and the reason every comparison here is drawn within one day.

Two rows are read with care. **q5** fails under `csvo` on this image
(`SubqueryAdaptiveBroadcastExec does not support the execute() code path`: the dynamic-partition-pruning
subquery on `ws_sold_date_sk` evaluated before adaptive execution rewrote it; it passes under `ours`
and passed under `csvo` on v10) and is missing from the table. **q65**'s checksum differs in every
configuration, Spark against Comet included: its `ORDER BY s_store_name, i_item_desc LIMIT 100` has
ties, so the hundred rows follow the physical order. Every other query's checksum agrees across the
four. And the q64 zero-row result reported above is intermittent: the two configurations that lost
the rows in the morning returned the correct 12,185 in the afternoon on the same image and data --
q64 too reads `store_sales` through a dynamic-partition-pruning filter, so one timing-dependent
evaluation of the Comet scan's pruning subquery would account for both it and q5.

**The four configurations in one window, cache off, AQE bounded to 208 (v23 image, 300 partitions,
2026-09-23/24).** The run the campaign was for: the v23 image is main at #462 (the threshold planner,
both grace-join memory fixes, the 1 GiB sort budget), the AOT cache is off for every leg after the
measurement above, and the run configuration adds `spark.sql.adaptive.coalescePartitions.minPartitionNum=208`
to the 300 shuffle partitions and the 128 MB advisory size (deprecated in Spark 3.2+; later TPC-DS runs
leave it unset, #253). Spark, `ours`, `csvo` and `comet` ran back to
back on the same nine nodes with nothing else on the cluster (22:28 to 01:40), one run per query, 103
queries each. `csvo` pins Comet's scan to `spark.comet.scan.impl=native_datafusion`. Seconds.

| query | Spark | Comet | csvo | ours |
|---|---|---|---|---|
| q1 | 13.1 | **12.3** | 15.3 | 13.2 |
| q2 | 51.4 | 42.8 | 50.6 | **42.2** |
| q3 | 3.9 | **3.8** | 4.5 | 4.1 |
| q4 | 91.2 | **60.1** | 97.6 | 90.1 |
| q5 | 46.0 | 31.2 | 27.6 | **24.4** |
| q6 | 9.8 | 5.0 | 4.6 | **2.8** |
| q7 | 6.2 | **6.2** | 7.4 | 6.9 |
| q8 | 7.6 | **3.7** | 4.9 | 4.6 |
| q9 | 92.8 | **78.1** | 93.3 | 90.3 |
| q10 | 7.5 | **6.7** | 6.8 | 7.0 |
| q11 | 44.1 | **35.6** | 46.3 | 47.6 |
| q12 | 2.3 | **1.8** | 2.6 | 2.9 |
| q13 | 7.9 | **7.7** | 8.6 | 8.0 |
| q14a | 102.5 | **76.1** | 96.2 | 88.6 |
| q14b | 94.6 | **69.7** | 92.3 | 82.8 |
| q15 | 8.8 | **4.4** | 4.8 | 5.6 |
| q16 | 30.8 | **20.9** | 23.3 | 20.9 |
| q17 | 12.1 | 9.3 | 8.8 | **7.2** |
| q18 | 8.7 | **6.7** | 11.7 | 8.1 |
| q19 | 5.3 | **2.8** | 3.4 | 3.1 |
| q20 | 2.2 | **1.6** | 2.6 | 2.3 |
| q21 | 1.8 | **1.2** | 1.7 | 1.6 |
| q22 | 8.5 | **3.7** | 6.4 | 7.6 |
| q23a | 210.6 | 131.7 | 131.3 | **117.6** |
| q23b | 291.1 | 153.4 | 137.0 | **124.8** |
| q24a | 110.0 | **101.1** | 120.8 | 110.9 |
| q24b | 102.7 | **97.7** | 111.0 | 108.2 |
| q25 | 9.4 | **6.5** | 11.4 | 10.9 |
| q26 | 3.8 | **3.0** | 3.5 | 3.2 |
| q27 | 6.2 | 6.6 | 6.8 | **6.2** |
| q28 | 114.0 | **102.5** | 110.1 | 113.7 |
| q29 | 25.1 | 14.8 | 11.4 | **9.8** |
| q30 | 18.7 | **12.7** | 16.1 | 14.4 |
| q31 | 14.0 | 13.9 | 14.9 | **12.5** |
| q32 | 1.5 | 1.7 | 1.4 | **0.9** |
| q33 | 4.4 | **2.4** | 4.4 | 3.8 |
| q34 | 5.8 | **4.0** | 5.2 | 4.5 |
| q35 | 18.5 | 13.9 | 12.8 | **11.4** |
| q36 | **5.6** | 6.5 | 8.0 | 7.8 |
| q37 | 7.4 | **6.4** | 7.9 | 6.9 |
| q38 | 33.5 | 22.3 | 22.3 | **21.2** |
| q39a | 5.8 | **3.7** | 5.3 | 5.1 |
| q39b | 6.0 | **2.9** | 4.9 | 4.4 |
| q40 | 8.3 | 13.9 | 12.1 | **6.2** |
| q41 | 0.8 | **0.5** | 0.6 | 0.6 |
| q42 | 1.6 | 1.4 | 1.4 | **1.2** |
| q43 | 5.6 | **4.6** | 5.2 | 5.5 |
| q44 | 36.8 | **34.0** | 35.8 | 35.1 |
| q45 | 7.8 | 5.1 | 5.8 | **4.0** |
| q46 | 8.9 | 8.1 | 7.2 | **7.1** |
| q47 | 12.9 | **10.3** | 13.6 | 14.3 |
| q48 | 8.8 | 6.3 | **6.0** | 7.0 |
| q49 | 44.4 | 47.4 | 49.8 | **38.9** |
| q50 | 68.2 | 54.4 | 38.4 | **37.8** |
| q51 | 24.5 | **13.6** | 15.1 | 14.6 |
| q52 | 1.5 | 1.1 | 1.2 | **1.1** |
| q53 | 4.8 | 4.7 | 4.4 | **4.3** |
| q54 | 7.2 | 6.3 | 4.1 | **4.1** |
| q55 | 1.7 | 1.7 | 1.8 | **1.4** |
| q56 | 3.8 | **1.8** | 3.9 | 3.4 |
| q57 | 7.1 | **5.2** | 8.1 | 8.5 |
| q58 | 3.0 | 3.4 | 2.8 | **2.7** |
| q59 | 35.8 | 35.7 | 37.3 | **33.0** |
| q60 | 3.5 | **2.4** | 4.3 | 3.9 |
| q61 | 5.1 | **2.8** | 3.6 | 3.7 |
| q62 | 23.0 | 25.1 | 26.5 | **22.9** |
| q63 | 5.1 | **4.3** | 5.1 | 4.5 |
| q64 | 92.7 | 56.3 | 53.9 | **51.4** |
| q65 | 29.7 | **15.5** | 24.0 | 23.4 |
| q66 | 9.5 | **9.5** | 11.1 | 9.9 |
| q67 | 126.5 | **60.7** | 71.2 | 72.9 |
| q68 | 6.7 | **3.3** | 4.2 | 3.7 |
| q69 | 7.2 | 5.1 | 5.2 | **4.5** |
| q70 | 11.1 | 12.1 | 10.4 | **9.8** |
| q71 | 3.2 | 3.4 | **2.9** | 3.0 |
| q72 | **29.6** | 34.1 | 33.3 | 32.3 |
| q73 | 5.3 | 2.7 | **2.6** | 2.8 |
| q74 | 43.7 | **31.0** | 40.8 | 39.5 |
| q75 | 76.8 | 73.3 | 80.2 | **69.3** |
| q76 | 46.3 | 43.4 | 46.9 | **38.7** |
| q77 | 2.8 | 2.6 | 2.6 | **1.9** |
| q78 | 123.3 | **74.6** | 92.8 | 79.3 |
| q79 | 5.7 | **4.5** | 5.7 | 5.4 |
| q80 | 48.3 | 40.4 | 43.6 | **39.1** |
| q81 | 18.0 | **12.0** | 13.0 | 14.4 |
| q82 | 20.6 | **17.9** | 19.3 | 19.3 |
| q83 | 1.7 | **1.2** | 1.7 | 1.6 |
| q84 | 19.4 | **17.2** | 19.1 | 18.0 |
| q85 | 22.7 | **18.4** | 20.9 | 21.9 |
| q86 | 5.8 | **5.1** | 5.6 | 5.7 |
| q87 | 31.2 | **19.0** | 27.9 | 20.3 |
| q88 | **125.5** | 158.1 | 126.6 | 141.7 |
| q89 | 5.7 | 20.2 | **5.6** | 6.0 |
| q90 | 35.1 | **34.0** | 35.8 | 36.9 |
| q91 | 4.2 | **1.8** | 2.9 | 2.5 |
| q92 | 2.4 | **1.8** | 1.9 | 2.1 |
| q93 | 136.6 | 85.4 | 67.9 | **65.9** |
| q94 | 59.7 | 57.9 | 59.5 | **57.3** |
| q95 | 106.4 | **57.5** | 58.3 | 69.3 |
| q96 | 18.5 | 16.9 | **16.7** | 19.8 |
| q97 | 31.1 | **13.6** | 13.8 | 14.3 |
| q98 | 3.7 | **2.3** | 2.8 | 2.7 |
| q99 | 8.5 | **8.4** | 13.3 | 14.4 |
| **total** | **3309** | **2514** | **2706** | **2557** |

`ours` totals **2557 s** against Spark's 3309 (23% less), faster than Spark on 82 of 103
queries, faster than `csvo` on 77 and than Comet on 38; Comet totals 2514, `csvo` 2706. On the
heavy joins `ours` leads every engine -- q23a 118 against Comet's 132 and Spark's 211, q23b 125 against
153 and 291, q93 66, q64 51, q50 38 -- and trails Comet where the scan and the
aggregate dominate: q4 90 against 60, q67 73 against 61, q95 69 against 58, q14a/b by 12 s each. Against Spark
the one heavy loss is q88 (142 against 126; Comet 158), the #409 thread's query.

Read against the cache-on `ours` legs above (2856 on v18 in the morning, 3059 on v19 beside the
windows, 3097 on v23 alone), this leg is the same code path on the same data 10-17% faster, all of it
the AOT cache's cost on the heavy half of the suite. The 208 minimum on its own, from the paired test
and the plain-300 legs of the same engines: Spark 3408 to 3263 (-4%), Comet 2555 to 2483 (-3%), `csvo`
2581 to 2678 (+4%), `ours` q67 -12% and the rest inside the band -- it keeps the sort stages of the
window queries wide enough and costs the Comet-scan configurations a little on the small ones.

Correctness: every checksum equals Spark's except q65 (ties, in every engine) and q64 under `csvo`
(0 rows against 12,185; the stale pruning value of comet#6133). q5, which failed under `csvo` on every
earlier run, completes with the scan implementation pinned and matches Spark's 100 rows.

**The Spark reference at three memory splits, and the prefetching converter (same window
configuration, the legs after the table).** Every engine above has 50 GB per executor; they differ in
where the engine puts it. Spark ran at 20 g heap / 30 g overhead, `ours` at 30 / 20, `csvo` at 18 / 16
with 16 g off-heap, Comet at 20 / 6 with 24 g off-heap -- each engine's split follows where it
allocates, and a plain-Spark leg at 30 g of heap is the check that the reference column is not
handicapped by its own. Two more Spark legs, 40 / 10 and 30 / 20, and one `ours` leg with the
prefetching scan converter (#465, `spark.vector.scan.prefetch=2`), all on the v24/v25 images of the
same tree:

| set | Spark 20/30 | Spark 30/20 | Spark 40/10 | Comet | `csvo` | `ours` | `ours` prefetch 2 |
|---|---|---|---|---|---|---|---|
| all 103 | **3309** | -- | -- | 2514 | 2706 | **2557** | 2615 |
| the 99 every split completed | 2796 | 2863 | 2860 | 2155 | 2326 | 2202 | 2253 |

| query | Spark 20/30 | 30/20 | 40/10 | Comet | `csvo` | `ours` | `ours` prefetch 2 |
|---|---|---|---|---|---|---|---|
| q23b | 291.1 | evicted | evicted | 153.4 | 137.0 | **124.8** | 135.0 |
| q23a | 210.6 | 188.1 | 185.5 | 131.7 | 131.3 | **117.6** | 121.7 |
| q93 | 136.6 | 146.1 | 164.8 | 85.4 | 67.9 | **65.9** | 66.1 |
| q67 | 126.5 | 118.9 | 124.9 | **60.7** | 71.2 | 72.9 | 85.4 |
| q88 | 125.5 | 125.3 | **124.4** | 158.1 | 126.6 | 141.7 | 129.7 |
| q78 | 123.3 | 119.0 | 123.4 | **74.6** | 92.8 | 79.3 | 92.8 |
| q28 | 114.0 | 235.1 | 183.3 | **102.5** | 110.1 | 113.7 | 119.0 |
| q95 | 106.4 | 96.4 | 92.8 | **57.5** | 58.3 | 69.3 | 60.4 |
| q14a | 102.5 | 93.8 | 97.5 | **76.1** | 96.2 | 88.6 | 91.4 |
| q14b | 94.6 | 90.3 | 88.3 | **69.7** | 92.3 | 82.8 | 87.7 |
| q9 | 92.8 | 83.3 | 83.7 | **78.1** | 93.3 | 90.3 | 90.8 |
| q64 | 92.7 | 91.5 | 91.2 | 56.3 | 53.9 | **51.4** | 52.3 |
| q4 | 91.2 | 90.9 | 88.6 | **60.1** | 97.6 | 90.1 | 90.4 |
| q50 | 68.2 | 72.7 | 80.2 | 54.4 | 38.4 | **37.8** | 37.8 |

The heap does move Spark: at 30 / 20 it is faster than at 20 / 30 on 62 of the 99 shared queries -- q9
92.8 to 83.3, q14a 102.5 to 93.8, q23a 210.6 to 188.1, q67 126.5 to 118.9, q95 106.4 to 96.4 -- and
without the two damaged queries below the 99 total is 2482 against 2545 (-2.5%). But both heavier-heap
legs lose q23b, q24a and q24b, and the loss is the node's disk, not memory: the executors are evicted
for ephemeral storage during q23b (`The node was low on resource: ephemeral-storage`, then
`DiskPressure`). Spark's q23b writes 83 GB of shuffle (`ours` 37) and spills 503 GB across nine nodes
with 20 GB volumes; a larger heap spills larger files, and the 20 / 30 split survives on the margin. A
repeat of the 30 / 20 leg with `spark.cleaner.periodicGC.interval=1min`, so the previous query's
shuffle files could not be the difference, evicted the same three -- the footprint is q23b's own. The
executors replaced after the eviction then cost q28 (235 and 183 against 114) and q93 (146 and 165
against 137) their cached inputs, which is why the 30 / 20 total is *higher* on paper. The reference
column stays the complete 20 / 30 run; against the best of the three splits per query (3165 s) `ours`
is 19% faster and ahead on 75 of 103, against the complete run 23% and 82. A 30 / 20 reference that
completes needs a larger node volume.

The prefetching converter changes 44 queries for the better and 59 for the worse, +2.3% in total:
gains q88 141.7 to 129.7, q95 69.3 to 60.4, q94 57.3 to 53.6; losses q78 79.3 to 92.8, q67 72.9 to
85.4, q23b 124.8 to 135.0. The node's own metrics say why. On q9 (six scans wrapped, 710 k batches
each) the task thread waited 28.1 min per scan for converted batches, the helper thread waited
27.7 min *on the Parquet reader*, and converting took 1.2 min: the conversion this operator overlaps is
4% of the read, and the queue hand-off plus one batch copy per 710 k batches is the tax the losses
show. The default stays `spark.vector.scan.prefetch=0`; the operator remains as an opt-in instrument
with its three wait metrics. The scan-side cost is the reader, which is also what `csvo`'s wins on
q88, q95 and q28 measure -- Comet's DataFusion reader, not the Arrow boundary.

**The read path itself: the S3A Analytics Accelerator stream, tuned against its defaults.** Every
Parquet leg reads through the accelerator: Hadoop 3.4.3's S3A defaults `fs.s3a.input.stream.type` to
`analytics` (the executor profiles show `AnalyticsStream` in the read path, the library at 1.3.1),
and the image pins the jar explicitly. One Spark leg on the same window configuration with the stream
tuned -- read-ahead 4 MB (default 64 KB), 16 MB blocks, ranges and parts (default 8 MB), whole-object
prefetch up to 16 MB (default 8), `prefetching.mode=ALL` (default `ROW_GROUP`), a 500-connection pool
(200), 256 threads -- was slower on 61 of 93 queries, 2534 s against 2440 (+3.8%, the median per-query
change the same), most on the scan-heavy ones: q9 96.1 against 92.8, q23a 219.9 against 210.6, q28
123.8 against 114.0, q67 133.1 against 126.5, q88 133.0 against 125.5. The files are 7-15 MB, so the
default already fetched most of them whole and the row-group prefetch already brought in exactly the
columns in flight; larger units and `ALL` fetch more bytes per file than the query uses, thirteen
tasks at a time. The library's defaults stay the configuration. With the converter result above, the
scan cost is the reader's per-file request latency, and neither overlapping the conversion nor
fetching bigger units moves it.


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

**Update (#264, multi-wildcard `LIKE`).** `o_comment NOT LIKE '%special%requests%'` (Q13) and
`s_comment LIKE '%Customer%Complaints%'` (Q16) were the last expression reasons on TPC-H: `Like` with
several wildcards is left alone by `LikeSimplification` and fell back as `unsupported expression Like`,
taking the projection above it. As a multi-token matcher both compile: at SF1 (decimals, `vector`,
one run) Q13 is at 8/13 accelerated operators with the global sort (by design) as its only remaining
reason, and Q16's list no longer mentions `Like` -- what is left there is the null-aware anti join
(#265), whose `child ... is not columnar` cascade still takes the joins, projection and aggregate above it.

**Update (#265, null-aware anti join).** With `ps_suppkey NOT IN (SELECT s_suppkey ...)` compiled --
Spark's single-key null-aware anti join over a broadcast, the two singleton relations (empty build:
keep all; a null build key: keep none) and, otherwise, the null-key streamed rows dropped -- Q16 goes
from 7 to 11 of 17 operators on our side (SF1 decimals, `vector`, one run), and like Q13 its only
remaining reason is the global sort. Neither query has an expression or join reason left.

## Iceberg merge-on-read: v2 positional and equality deletes, v3 deletion vectors (#260 harness, #261, #262)

The local harness (`gen-iceberg-mor.sh`, `run-tpch.sh --iceberg ... --variant ...`, `docs/iceberg.md`)
builds `lineitem` variants with the delete shapes a lakehouse table carries between compactions and
runs every configuration over them. This is the v2 study over Iceberg's JVM reader (`BatchScanExec`
under both engines): SF1 decimals, the 8-core x86 host of the decimal tables above, `local[8]`, 5
warm-up and 10 measured iterations per cell, `spark` against `vector`, a quiet machine (load 1.9). All
50 cells returned identical checksums, and the adapter's counters read exactly what the generator's
README says (5402462 live of 6001215 rows read at 10 %, 4201747 at 30 %). The Comet configurations
were not measured (no Comet jar on this host); v3 deletion vectors are #262.

Medians in milliseconds; in parentheses the ratio to the same engine on `plain` (what the deletes
cost that engine: below 1 is slower), then `vector`'s speedup over `spark` on the same variant.

| variant | live % | probe-sum spark | probe-sum vector | probe-group spark | probe-group vector | q6 spark | q6 vector | q1 spark | q1 vector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `plain` | 100 | 104 | 76 (1.38x) | 148 | 92 (1.61x) | 149 | 145 (1.03x) | 1650 | 379 (4.36x) |
| `pos_2` | 98 | 140 (0.74) | 128 (0.59, 1.10x) | 182 (0.81) | 153 (0.60, 1.19x) | 176 (0.85) | 215 (0.67, 0.82x) | 1653 (1.00) | 416 (0.91, 3.98x) |
| `pos_10` | 90 | 125 (0.83) | 108 (0.70, 1.16x) | 170 (0.87) | 142 (0.65, 1.20x) | 166 (0.89) | 204 (0.71, 0.82x) | 1516 (1.09) | 381 (0.99, 3.98x) |
| `pos_30` | 70 | 127 (0.82) | 116 (0.66, 1.10x) | 167 (0.89) | 134 (0.69, 1.24x) | 158 (0.94) | 194 (0.75, 0.81x) | 1263 (1.31) | 382 (0.99, 3.30x) |
| `pos_10_clustered` | 92 | 124 (0.84) | 124 (0.61, 1.00x) | 181 (0.82) | 140 (0.66, 1.29x) | 165 (0.90) | 215 (0.67, 0.77x) | 1523 (1.08) | 400 (0.95, 3.81x) |
| `pos_30_clustered` | 71 | 124 (0.84) | 116 (0.65, 1.07x) | 169 (0.88) | 138 (0.67, 1.23x) | 159 (0.93) | 184 (0.79, 0.87x) | 1224 (1.35) | 389 (0.97, 3.14x) |
| `pos_upd_1` | 88 | 147 (0.71) | 131 (0.58, 1.12x) | 188 (0.79) | 156 (0.59, 1.20x) | 178 (0.84) | 232 (0.62, 0.76x) | 1606 (1.03) | 450 (0.84, 3.57x) |
| `pos_upd_5` | 84 | 159 (0.65) | 148 (0.51, 1.08x) | 225 (0.66) | 160 (0.58, 1.40x) | 212 (0.70) | 233 (0.62, 0.91x) | 1807 (0.91) | 392 (0.97, 4.61x) |
| `eq_2` | 98 | 231 (0.45) | 201 (0.38, 1.15x) | 285 (0.52) | 234 (0.39, 1.22x) | 275 (0.54) | 311 (0.46, 0.88x) | 1786 (0.92) | 510 (0.74, 3.50x) |
| `eq_10` | 90 | 330 (0.32) | 372 (0.20, 0.89x) | 377 (0.39) | 355 (0.26, 1.06x) | 386 (0.38) | 451 (0.32, 0.86x) | 1872 (0.88) | 674 (0.56, 2.78x) |

`probe-count` is left out: on `plain` it is Iceberg's manifest lookup (`LocalTableScanExec`, 38 ms for
both engines) and on the deleted variants it costs both engines the same 95-105 ms (positional) or
190-330 ms (equality) -- the delete cost alone, with nothing to compute.

### What the numbers say

**The hypothesis is refuted, and the refutation is informative.** The margin of `vector` over `spark`
does not grow with the delete share; it shrinks. On the pure-merge probe (`probe-sum`) `vector` is
1.38x faster on `plain` and 1.00-1.16x on every positional variant, whatever the percentage and the
layout; on `probe-group` 1.61x becomes 1.19-1.40x; on Q1 4.36x becomes 3.1-4.0x. The reason is in the
`vs plain` columns: **the deletes cost a fixed price per batch that is flat in the delete share** --
`spark` pays 20-25 ms on `probe-sum` at 2 %, 10 % and 30 % alike, `vector` pays 30-50 ms -- and a
fixed price hurts the faster engine's ratio more. The reasoning of the hypothesis was about the
per-row indirection (`mapping[i]` per column access in Spark's codegen against our in-place kernels
over a bitmap); at these sizes that term is invisible against the per-batch work.

**Where the fixed price goes: Iceberg's own delete handling, in both engines.** JFR on `vector`
(`probe-sum`, `pos_30` scattered and clustered, 8 iterations) has Iceberg at the top of every
list: `ColumnarBatchUtil.buildRowIdMapping` (the `int[]` the reader builds per batch),
`Deletes.toPositionIndexes` and `JavaHashes.hashCode(CharSequence)` (the position index built per
task from the delete files, keyed by data-file path), the roaring-bitmap search under it. The two
costs the issue named on our side barely register: `ArrowLayout.selectionFromIndices` is one sample
in either profile and `validityFromNullBytes` none, so the mapping-to-bitmap conversion is not the
problem and neither is the validity copy. Our extra 10-25 ms over Spark's price is the kernels
walking every physical row of a forwarded selection to use 70-98 % of them (the `ColumnarBatchRow`
path Spark uses skips the deleted rows before codegen sees them), plus the aggregate's selection
handling.

**Scattered and clustered deletes cost the same.** `pos_10` against `pos_10_clustered` and `pos_30`
against `pos_30_clustered` are within noise on every query for both engines. Whole inactive 64-row
blocks -- the shape `EvalContext`'s active-block skipping exists for -- buy nothing here, because the
per-batch price is paid before any kernel runs and the kernels' per-row work over a 70 %-dense
selection is already small.

**Equality deletes are the expensive kind, for both engines and more so for ours.** `eq_10` costs
`spark` 3x on `probe-sum` (104 to 330 ms) and `vector` 5x (76 to 372 ms); `eq_10` is the one
variant where `vector` loses the probe (0.89x). Iceberg's JVM reader evaluates the equality predicate
per row on a `ColumnarBatchRow` to build the mapping (`ColumnarBatchUtil.buildRowIdMapping` with an
`EqualityDeleteFilter`), which is row-at-a-time work in front of a columnar reader and dominates
everything else. The candidate optimisation the issue describes -- take the position-filtered batch
and evaluate the equality-delete set as our own `IN` / anti-join over the batch -- is the only lever
that would move these numbers; recorded here, not implemented (it needs the reader to expose the
un-applied equality deletes, which the 1.11 API does not).

**The forwarding threshold is not the lever.** `sparkvector.selection.minFraction` decides whether a
filter or project forwards a selection or compacts (0.5: forward when at least half the rows
survive). On `pos_30` under `vector` (7 iterations) raising it to 0.8 changes nothing on the probes
(within 3 %); compacting always (1.01) makes `probe-sum` 130 to 112 ms, `probe-group` 160 to 143 ms
and Q6 232 to 195 ms, but Q1 404 to 468 ms -- the compaction of seven columns costs more than the
kernels save. A delete-derived selection is not different enough from a predicate's to earn its own
knob; the default stays.

**Q6 is slower under `vector` than under `spark` on every deleted variant** (0.76-0.91x) while it is
even on `plain` (1.03x). Q6 keeps 2 % of the rows: on `plain` the filter compacts once and the
aggregate sees a tiny batch; on a deleted variant the filter's predicate runs over a batch that
already carries a selection, and the compaction is of a selection-over-selection. Spark's codegen
fuses the delete mapping and the predicate into one row loop. This is the one shape where our
per-row work over the physical rows shows, and it is at most 40 ms per query at SF1.

**A transient, not a cliff.** In two of the sixteen `vector` JVMs of this and the harness's
validation run, Q1 took about 8 s on two or three consecutive iterations (once after five warm-up
runs on `pos_upd_5`: 8035, 7963, 679, then 380-460 ms) with the partial `VectorHashAggregateExec`
reporting 60 s of kernel time summed over tasks and the other operators unchanged. It does not
reproduce on demand: 24 iterations under JFR on the two variants it first appeared on ran at 450 ms
with the aggregate at 1.0 s. The signature -- growth across iterations inside one JVM, then recovery
-- is a JIT deoptimisation storm in the aggregate rather than an algorithmic cost of the delete
layout (the harness's first write-up read it as the latter; this run corrects it). It is worth
catching with `-XX:+PrintCompilation` on a run that shows it; it is listed in AGENTS.md section 7.

### v3: deletion vectors (#262)

The same protocol over the v3 tables -- the `pos_*` mutation scripts encoded as deletion vectors in
Puffin files, one roaring bitmap per data file -- in one session with its own `plain` run (5 warm-up,
10 measured iterations, quiet host). 40 cells, identical checksums between `spark` and `vector`, and
every `dv_*` checksum equals the `pos_*` twin of the same mutation: same rows, different encoding. On
Comet 1.0 every configuration reads v3 through Iceberg's JVM reader (`BatchScanExec`), so this is the
case where a JVM columnar merge is the only accelerated path a user has; the Comet configurations were
not run on this host.

| variant | live % | probe-count spark | probe-count vector | probe-sum spark | probe-sum vector | probe-group spark | probe-group vector | q6 spark | q6 vector | q1 spark | q1 vector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `plain` | 100 | 38 | 40 | 98 | 79 (1.23x) | 146 | 94 (1.55x) | 132 | 170 (0.78x) | 1656 | 380 (4.35x) |
| `dv_2` | 98 | 91 | 95 | 127 (0.77) | 112 (0.71, 1.14x) | 173 (0.85) | 132 (0.72, 1.31x) | 183 | 186 (0.99x) | 1595 | 406 (3.93x) |
| `dv_10` | 90 | 68 | 77 | 113 (0.87) | 101 (0.78, 1.12x) | 160 (0.91) | 121 (0.78, 1.31x) | 158 | 206 (0.76x) | 1517 | 419 (3.62x) |
| `dv_30` | 70 | 79 | 83 | 113 (0.86) | 96 (0.83, 1.18x) | 152 (0.96) | 120 (0.78, 1.26x) | 168 | 199 (0.85x) | 1223 | 391 (3.13x) |
| `dv_10_clustered` | 92 | 86 | 76 | 113 (0.87) | 102 (0.77, 1.10x) | 155 (0.94) | 127 (0.74, 1.22x) | 168 | 184 (0.91x) | 1535 | 424 (3.62x) |
| `dv_30_clustered` | 71 | 85 | 80 | 108 (0.90) | 95 (0.84, 1.14x) | 152 (0.96) | 118 (0.80, 1.28x) | 157 | 218 (0.72x) | 1225 | 411 (2.98x) |
| `dv_upd_1` | 88 | 114 | 110 | 137 (0.71) | 112 (0.71, 1.23x) | 176 (0.83) | 132 (0.72, 1.34x) | 174 | 194 (0.90x) | 1676 | 413 (4.06x) |
| `dv_upd_5` | 84 | 121 | 115 | 142 (0.69) | 126 (0.63, 1.13x) | 198 (0.74) | 145 (0.65, 1.37x) | 193 | 206 (0.94x) | 1808 | 439 (4.12x) |

**Deletion vectors are the cheaper encoding, for both engines.** Against the v2 cells of the same
mutation (previous table, same host and protocol): the pure-merge probe costs `spark` 113 ms on `dv_10`
against 125 on `pos_10` and `vector` 101 against 108; `probe-count`, which is nothing but the merge,
68-79 ms on `dv_*` against 99-101 on `pos_*` for `spark` and 77-83 against 93-96 for `vector`. The
delete price over `plain` is +15 ms for `spark` and +17-22 ms for `vector` on the probe, where the v2
position-delete files cost +21-23 and +32-40. What disappears is the per-task index construction:
in the v2 profile `Deletes.toPositionIndexes` and the path hashing under it were a third of the
Iceberg samples; on v3 they are gone and `RoaringPositionBitmap.contains` takes their place at a
fraction of the cost -- one blob per data file, deserialised once, against thousands of delete rows
to sort into an index. The update/merge shape (`dv_upd_*`: 18 data files, several snapshots, the
one v3 is meant for) pays the same as its v2 twin within noise, since its extra cost is the small
files, not the deletes.

**The ratio looks like v2's, as the hypothesis said; the margin does not grow, as v2 found.**
`vector` over `spark`: 1.10-1.23x on `probe-sum`, 1.22-1.37x on `probe-group`, 3.0-4.1x on Q1 --
flat in the delete share and a little under the `plain` ratios of the same session (1.23x, 1.55x,
4.35x), the fixed-price effect of the v2 write-up in a smaller dose. Q6 is again slower under `vector`
on every variant, `plain` included this session (0.72-0.99x): the selection-over-selection compaction.

**Where the DV cost goes, and the candidate optimisation.** JFR on `probe-sum` over `dv_30`, 10
iterations, both engines: `ColumnarBatchUtil.buildRowIdMapping` is the top method for both -- 19.0 %
of `vector`'s 621 samples and 15.0 % of `spark`'s 713 -- with `RoaringPositionBitmap.contains` and
`DeleteFilter.incrementDeleteCount` under it; the Puffin read itself does not register (it is once per
file). On our side `selectionFromIndices` is 1.45 % (9 samples) and `validityFromNullBytes` absent.
The mapping construction is therefore clearly visible, which is the issue's condition for the direct
bitmap path (build our selection from the `PositionDeleteIndex` and the batch's starting position,
skipping the `int[]`). But the `int[]` is built inside Iceberg's `ColumnarBatchReader` before the batch
reaches either engine: a direct path on our side removes our 1.5 %, not the 19 %, unless Iceberg's
reader can be told to hand out the index and the row offset instead of wrapping the vectors. That is
an Iceberg-side option (a reader flag, or the `_deleted` metadata-column path with the wrapping
skipped), worth proposing upstream with these numbers; nothing in our operators buys it. The
pipeline stays bitmap -> `int[]` -> bitmap with the middle step Iceberg's and the middle step the
expensive one.

**The Q1 transient, third and fourth sightings.** Both `dv_upd_*` variants showed it in this run, on
the first iterations after five warm-ups (`dv_upd_1`: 8954, 9323, 3262 then 430 ms; `dv_upd_5`: 8364,
8473, 8384, 1731 then 380-440 ms), and it again refused to appear under JFR (10 iterations at 441 ms)
or under `-Xlog:deoptimization=debug` (395 ms median, an ordinary 648-line log). Four of the five
sightings are on the update/merge shape -- 18 data files, two of them small ones from the UPDATE and
the MERGE -- which is the one pattern in it; the recovery inside the same JVM says the code recompiles
its way out. It is in AGENTS.md section 7 with the recipe to catch it: run the sweep command (not the
profiler) with `-XX:+PrintCompilation` and keep the output of a JVM that shows the 8 s iterations.

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

Refreshed after the 128-bit decimal lane (#257: the lane and its scan, #258: the kernels that compute on
it, #259: every operator carries it -- the aggregate's keys, inputs and wide results, the column movers,
the hash joins' keys and payloads, the window's frames and keys): **3425 of 4655 operators ours (73%)**,
65 queries at 75% or more (from the low forties), 37 between 50% and 75%, none between 25% and 50%; q9 stays the one
fully accelerated plan and q17 is empty at SF1 (0/1 under both configurations). All 103 checksums agree
with Spark. The queries the wide-decimal issues named: q1 28/36, q30 34/44, q81 33/43 (their `decimal(24,7)`
averages and `decimal(19,2)` totals were the inputs no lane could carry), q64 126/169, q51 23/35 (its
`decimal(27,2)` running totals through the window and the join on them), q66 38/48 (the union of channel
sums), q67 20/26 (the group limit over a wide key), and the windows q12 13/18, q20 13/18, q47 and q57
50/70, q53 and q63 18/24, q89 17/23, q98 12/20. What still names a decimal is computed, not carried, and
all of it is the *result* of a wide division: `round` over a `decimal(37,20)` quotient (q2, seven
columns), `CASE WHEN ... THEN a / b END` whose result is `decimal(37,20)` or `decimal(38,14)` (q4, q11,
q31, q74 -- the year-over-year ratios), and a scalar subquery whose value is `decimal(32,6)` to
`decimal(38,8)` (q14a/b, q23a/b, q24a/b). Those three are one follow-up to the #258 kernel set: the
conditional, the rounding and the literal over the wide lane. The other reasons are the ones this
section has carried since the start -- Spark's global `Sort` over its row shuffle (21 plans read a bare
`AQEShuffleRead`), the `TINYINT` grouping id of the `ROLLUP` plans (q36, q70, q86), sort-merge joins
without the opt-in flag -- and the cascades below them (a `Project` whose child is not columnar, 11).
The tally is not #259's alone: the run counts everything merged since the last refresh, and the
operators that count moved from 4736 to 4655 with the plans that changed in between, so the percentages
and the per-query readings are the comparable figures, not the raw counts.
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

## Hybrid planning study (#279)

Running Comet's native operator instead of ours where it is measurably ahead pays the crossing twice
-- our batch into Comet, Comet's batch back -- so the study starts with that price, measured on its
own (`CrossingBenchmark`, JMH, in `benchmarks`; the Comet 1.0 jar on the classpath; JDK 25, x86-64,
one thread). Into Comet is the real path (`CometBatchBridge.convert`: `ArrowCData` export through the
C Data interface, Comet's `ArrowImporter`, the imported vectors released); back is
`ColumnVectorAdapters.adapt` through the registered `CometVectorAdapter` over the Comet vectors a
conversion produced.

### The crossing cost

Microseconds per batch at 8 columns, and nanoseconds per row per column (2 warm-up, 3 measured
iterations of 1 s; the 4- and 16-column runs scale linearly with the column count and are omitted):

| lane | into Comet, 4096 rows | into Comet, 8192 rows | back, 4096 rows | back, 8192 rows |
|---|---:|---:|---:|---:|
| INT64 (no validity) | 23.3 µs (0.71 ns) | 21.1 µs (0.32 ns) | 0.96 µs (0.029 ns) | 1.55 µs (0.024 ns) |
| INT64, 10 % nulls | 22.8 µs (0.70 ns) | 24.3 µs (0.37 ns) | 1.13 µs (0.034 ns) | 1.93 µs (0.029 ns) |
| FLOAT64 | 23.7 µs (0.72 ns) | 21.7 µs (0.33 ns) | 0.96 µs (0.029 ns) | 1.65 µs (0.025 ns) |
| UTF8, 12-40 bytes | 25.7 µs (0.78 ns) | 23.9 µs (0.37 ns) | 1.10 µs (0.034 ns) | 1.94 µs (0.030 ns) |
| UTF8 dictionary, 1000 values | 796 µs (24.3 ns) | 1359 µs (20.7 ns) | 1.10 µs (0.034 ns) | 2.00 µs (0.030 ns) |
| decimal(12,2) on the INT64 lane | 97.7 µs (2.98 ns) | 160.5 µs (2.45 ns) | 14.1 µs (0.43 ns) | 32.8 µs (0.50 ns) |

**Reading.** For a fixed-width lane and for plain strings the export is a pointer hand-over: the cost
is per *column*, about 2.7-3 µs each (the C Data structs, the JNI import, the release), and does not
grow with the rows -- 4096 and 8192 rows cost the same 21-25 µs for eight columns, which is why the
per-row figure halves between them. The way back is zero-copy for those lanes: 0.1-0.25 µs per
column. Two lanes pay per row. A dictionary-encoded string column is decoded on the way into Comet
(its reader would decode it anyway): 20-25 ns per row per column, the price of a kernel pass, and the
one crossing that dominates a swap -- TPC-DS's dictionary-heavy dimension columns cross at 8192 rows
× 8 columns for 1.4 ms, more than most of our operators spend on such a batch. A decimal on the
INT64 lane is widened to Arrow Decimal128 going in (2.5-3 ns per row per column) and narrowed back
(0.43-0.5 ns), a copy each way.

The rule the study applies: a swap candidate must beat ours by more than *twice* the crossing of the
columns it touches -- for fixed-width lanes that is a few microseconds per batch and any real operator
gap clears it; for dictionary strings it is ~45 ns per row per column, which only a large kernel gap
clears; for INT64 decimals ~6 ns per row per column.

### The operator matrix: TPC-H at SF10

Four configurations, one JVM each, `local[8]`, 8 GB heap, 5 warm-up and 7 measured runs (the
protocol's SF10 counts), the doubles schema of `gen-tpch.sh`; Comet 1.0.0; x86-64 (an 8-vCPU EC2
host, not the development laptop, so the absolute numbers are not comparable with the earlier
tables). `vector` is the harness's default (`sortMergeJoin.mode=auto` since #287);
`comet-scan-vector-shuffle` is Comet's scan and shuffle with our operators in between; `comet` is all
Comet. All checksums equal across the four.

| | spark | vector | comet-scan-vector-shuffle | comet |
|---|---:|---:|---:|---:|
| total of the 22 medians | 80.6 s | 112.3 s | 42.9 s | 39.0 s |
| q21 | 16.8 s | 44.9 s | 6.5 s | 10.9 s |
| total without q21 | 63.8 s | 67.4 s | 36.4 s | 28.1 s |

Milliseconds per operator kind, summed over the plan's nodes and over tasks for the last measured
run of every query (ours: kernel time inside the operator; Comet: native `elapsed_compute`; neither
includes the wait on children; the clocks are not identical, see `docs/comet.md`):

| operator kind | vector | comet-scan-vector-shuffle | comet |
|---|---:|---:|---:|
| SortMergeJoin | 272 236 (2 queries) | 461 (1) | 215 903 (13) |
| HashAggregate | 48 124 (21) | 44 267 (21) | 12 340 (21) |
| Filter | 44 403 (21) | 20 445 (21) | 9 102 (21) |
| BroadcastHashJoin | 27 132 (15) | 14 857 (15) | 5 705 (15) |
| ShuffledHashJoin | 20 983 (14) | 32 314 (14) | -- |
| Sort | -- | 226 (12) | 39 077 (16) |
| Project | 7 504 (21) | 2 135 (21) | 328 (21) |
| TakeOrderedAndProject | 103 (5) | 109 (5) | 5 (5) |

**Reading, query by query where it matters.**

- **The merge join at scale is the headline, and it is ours.** q21 runs two of our merge joins
  (`auto` chose them: the upper join relies on the lower's ordering, so the chain takes the merge)
  over 2.0M- and 36.6M-row outputs: 271 s of task time, 44.9 s of wall clock against Spark's own
  sort-merge join at 16.8 s and Comet's at 10.9 s. The same query under Comet's shuffle plans three
  of our *hash* joins and runs in 6.5 s -- the fastest of the four. The per-run bookkeeping named in
  the #286 notes (a run object, a compare through the columns, a buffer append per run; lineitem's
  order keys make runs of about four rows) is the whole difference. Two conclusions for #287: at SF10
  `auto` *is* slower than `hash` on TPC-H, so its third condition for the default is not met; and
  the merge join must not be chosen for large inputs until it walks runs without a run object --
  `auto` should take the merge only where the hash rewrite cannot go *and* the inputs are small by
  statistics, otherwise leave the join to Spark. q2 is the same story in miniature (merge join
  1359 ms of task time; wall 1855 ms against Spark's 1218 ms).
- **Hash aggregate with many groups: confirmed.** q18 (1.5M groups over lineitem) 13.7 s of task
  time against Comet's 1.7 s, q17 (per-part averages) 23.4 s against 6.6 s, q13 (customer counts)
  2.2 s against 0.06 s -- 4-40x. With few groups the two are level: q1 (4 groups, 60M rows) 4.1 s
  against 3.1 s. The gap is the group-key table, not the reductions. The crossing to reach Comet's
  aggregate is a pointer hand-over for these fixed-width keys, so the candidate clears the rule by
  seconds.
- **Broadcast hash join: behind, half of it the scan copy.** q9 8.9 s against 1.9 s, q8 5.8 s against
  0.86 s, q2 2.1 s against 0.3 s. Under Comet's scan (the middle column) the same joins cost 3.6 s
  and 1.8 s: reading Spark's Parquet vectors into our buffers is a copy the first operator above the
  scan pays, and it lands on the filter or the join. The remaining 2x is the probe itself.
- **Filter: the copy again, then a real gap.** Over Spark's scan our filter's time is dominated by
  the input adaptation (q20 5.4 s of task time; over Comet's scan 0.66 s; Comet's own 0.22 s; q15 3.9
  s / -- / 0.12 s; q8 3.6 s / 0.09 s / 0.09 s). Where the input is already Arrow the filter kernel is
  1-3x behind Comet's (q6 0.93 s against 0.28 s; q13's `like` filters 3.8 s against 1.2 s; q19 4.5 s
  against 2.0 s), never ahead. The control operator does not come out level, so the study has to say
  it: on this x86 host our kernels lose to DataFusion's on the plain filter too, and the question the
  hybrid issues (#280, #281) have to answer is not only "which operator" but "why our per-row cost is
  higher on the simplest kernel" -- the JFR profiles below are the start.
- **Project: a materialising projection is slow.** q21 6.1 s of task time over 37.9M rows (160 ns per
  row) against Comet's 12 ms; q3 0.81 s against 26 ms. Comet's projection over a join output is
  near-free; ours copies the columns it keeps. A pass-through projection (`isPassThrough`) is not the
  issue; the gather after a join is.
- **Shuffled hash join: ours only.** Comet has no shuffled hash join in these plans (its plans keep
  the sort-merge join, 216 s of task time over 13 queries, plus 39 s of sorts); ours runs 14 of them
  in 21 s. q5 (3.6 s against Comet's merge join 12.1 s + sort 7.9 s), q7 (5.0 s against 9.4 + 5.0 s)
  and q9 (3.9 s against 7.3 + 4.3 s) are the queries where our operator is the better one *by
  construction* -- a hash join instead of a sort and a merge -- and the wall clock still favours
  Comet there (q5 7.0 s against 3.7 s) because of the filter, aggregate and broadcast-join gaps
  above. Keeping ours here is the right call; the rest of the chain is what loses the query.
- **String functions.** q13 (`like`), q16 (`not like`, `in`), q22 (`substr`): the filters carrying
  them are 1-3x behind Comet's (q13 3.8 s against 1.2 s over Comet's scan; q22 0.33 s against 0.09
  s); no candidate stands out beyond the general filter gap.
- **Comet's fallbacks.** One: q11's `EmptyRelation is not supported`. Ours are the sorts over row
  exchanges (`Sort: child AQEShuffleRead is not columnar`, twelve queries), which stay Spark's by
  design; Comet sorts natively (39 s of task time) and it is part of why its merge-join plans still
  win the wall clock.

### Decimal arithmetic: TPC-H at SF1, the decimal schema

`gen-tpch.sh --decimals` keeps DuckDB's `DECIMAL(15,2)` columns, so the price arithmetic runs on
decimals -- narrow ones through #26's INT64 path, the wide products (`l_extendedprice * (1 -
l_discount)` is decimal(33,4)) through #258's two-limb kernels. Three configurations, 3 warm-up and 5
measured runs; totals of the 22 medians: `spark` 12.8 s, `vector` 13.0 s, `comet` 6.9 s; all
checksums equal.

- **Narrow decimals: refuted as a swap candidate.** q1 (four groups, the sums and averages over
  6M rows of decimal(15,2)) runs 347 ms under ours against 355 ms under Comet, and our
  aggregate's task time is *lower* (585 ms against 1037 ms). Where the INT64 path applies there is
  nothing to gain from Comet's i128 arithmetic.
- **Wide decimals: confirmed.** q6 sums a decimal(33,4) product into one group: our aggregate spends
  283 ms of task time, Comet's 2 ms -- the two-limb scalar reduction against a native i128 add. The
  wall clock is 161 ms against 64 ms. The crossing for these columns is the decimal widening, 2.5-3 ns
  per row per column each way; against a 100x operator gap it does not matter.
- **The shapes we refuse** stay refused: q8 and q14 fall back on `sum(CASE WHEN ... decimal)` (the
  #258 follow-up named in the expressions notes); Comet runs them natively. That is not an operator
  comparison but a coverage gap, listed here so the matrix is not misread.
- The many-group aggregate gap of SF10 repeats at SF1 (q18 1099 ms against 203 ms of task time,
  q17 1261 against 421).

### Profiles: the two largest gaps under a recording

The JFR-first protocol (AGENTS.md section 4.7) for the two gaps with the most task time behind them,
each `profile-query.sh` at SF10 under `vector`, 2 warm-up and 3 measured runs.

- **q18, the many-group hash aggregate.** Inside our operator the hottest method is
  `GroupKeyTable.lookupOrInsert` (13.6 % of samples; `rehash`, `GroupedAccumulators.regroup`,
  `insert`, `equals` and the hash mix another 4 %) -- the probe of a 1.5M-entry key table, which is
  what the swap candidate would replace. But three quarters of the samples carry no plugin frame at
  all: `BufferedInputStream.read1` 10 %, `BufferedOutputStream.growIfNeeded` 9 %,
  `UnsafeRowWriter.zeroOutNullBytes` 5 %, `ShuffleExchangeExec.prepareShuffleDependency` 4 %,
  `RowToColumnConverter.append` 3 % -- Spark's *row* shuffle of the 1.5M-group partial aggregate:
  every group serialised to an `UnsafeRow`, written, read, and converted back into columns above
  the exchange. Comet's aggregate is faster, and Comet's shuffle carries Arrow batches; the wall clock
  (8.7 s against 3.4 s) is mostly the second. The finding belongs to the columnar shuffle (#288) as
  much as to hybrid planning: swapping the aggregate alone would leave the shuffle.
- **q20, the filter over Spark's scan.** `SparkColumnVectorBuffers.decodeDictionary` is 25 % of all
  samples (28 % of plugin self time), ahead of everything in the query: the strings the Parquet
  reader hands us dictionary-encoded are decoded into our buffers before the first kernel runs, the
  same copy "Q6 revisited" found at SF1. The filter kernels themselves (`CompactKernels`,
  `StringMatchKernels.match`, `CompareKernels`) are 7 % together. Under Comet's scan the same
  filter's task time falls from 5.4 s to 0.66 s. The remedy is not a swap: it is keeping the
  dictionary through the adapter (or the Arrow scan) -- the same lesson as #14, at scale.

### The operator matrix: TPC-DS at SF1

`spark`, `vector`, `comet`; 2 warm-up and 3 measured runs (the protocol's 10 and 10 were a
multi-hour session for three configurations on this host; the medians below are steady to a few
percent between runs, the outliers are named). Totals of the 103 medians: `spark` 47.7 s, `vector`
82.2 s, `comet` 34.4 s; all checksums equal. Comet is fully native on 95 of the 103 queries; ours is
fully accelerated on one (sorts over row exchanges and Spark's exchanges stay Spark's by design).

| operator kind | vector | comet |
|---|---:|---:|
| ShuffledHashJoin | 161 757 (13 queries; q72 alone most of it) | -- |
| BroadcastHashJoin | 85 839 (100) | 19 748 (100) |
| SortMergeJoin | 280 (5) | 40 029 (7) |
| HashAggregate | 18 123 (100) | 4 848 (99) |
| Filter | 13 114 (102) | 4 425 (103) |
| Sort | 1 008 (8) | 2 963 (29) |
| Project | 1 771 (102) | 1 555 (102) |
| Window | 647 (13) | -- (Comet has no window operator; `WindowGroupLimit is not supported` on 3) |
| TakeOrderedAndProject | 187 (60) | 2 (64) |
| Expand | 137 (10) | -- |

- **Broadcast hash join is the TPC-DS gap.** 85.8 s of task time against Comet's 19.7 s over the
  same hundred queries, 4x, and it is the operator on the critical path of nearly every query here
  (a fact table probing a chain of dimension broadcasts): q4 5.0 s against 1.9 s, q11 3.1 against
  1.5, q14a/b 1.3-1.5 against 0.3-0.4, q64 1.4 against 0.2, q58 0.72 against 0.12, q15 0.75 against
  0.18. At SF1 the dimension tables are small and the probe dominates: this is our probe, not a copy
  (the input copy is a smaller share here than at TPC-H SF10 because the streamed fact-table columns
  are mostly fixed-width). A candidate that clears the rule: the crossing for a probe's key and
  payload columns is a few microseconds per batch.
- **Shuffled hash join: q72.** 27.0 s of wall clock against Comet's 5.8 s (Spark's is the same order
  as ours); the per-query issue #168 has the shape. Our operator, not Comet's, so not a swap
  question -- a fix.
- **Hash aggregate 3.7x, filter 3x** -- the SF10 reading at small scale; the many-group cases (q4,
  q11, q74: 1.27 s / 0.84 s / 0.35 s against 0.40 / 0.25 / 0.09) carry it, q23a/b's 1.1-1.3 s
  against 0.3 s too.
- **Window: not a candidate -- Comet has none.** Comet 1.0 leaves every window to Spark (and refuses
  `WindowGroupLimit`); ours runs 13 windows in 0.65 s of task time and three group limits in 0.18 s.
  The window queries (q47, q51, q57, q67) lose their wall clock to the joins and aggregates around
  the window, not to the window (q51: window 481 ms of task time, Comet's plan runs it in Spark).
- **Project is level** (1.8 s against 1.6 s over 102 queries) -- the SF10 project gap was the
  materialising gather after a merge join, absent here; on q64 Comet's project is the slower one
  (936 ms against 4 ms).
- **String functions** (q8, q15, q19, q45, q75, q79, q85): the filters carrying `substr`, `like` and
  the `in` lists are 1.5-5x behind (q8 174 ms against 37, q19 88 against 48, q79 110 against 84);
  the joins around them are the larger share of the wall clock in every one of those queries.
- **Decimal-heavy queries** (q1, q14, q24, q30, q58, q64, q65, q81, q83): no decimal-specific gap
  stands out from the join and aggregate gaps above; q83 is level (180 ms against 197).
- **Comet's fallbacks**: `WindowGroupLimit` (3 queries), an aggregate whose child aggregate is not
  Comet's (2), `Spark's BigDecimal rounding` (1). Ours: sorts over row exchanges and the
  expression shapes named in `docs/expressions.md`.

### Candidates and controls: the verdict

Against the rule -- *a swap must beat ours by more than twice the crossing of the columns it
touches* -- and with the profiles read:

| candidate | verdict | evidence |
|---|---|---|
| Hash aggregate, many groups | **Comet wins by 4-40x on the operator**, but the wall clock is Spark's row shuffle of the many-group partial result (q18 profile: 75 % of samples). A swap of the aggregate alone leaves most of the time where it is; the columnar shuffle (#288) is the larger lever. | TPC-H SF10 q13/q17/q18; TPC-DS q4/q11/q74; the q18 profile |
| Hash aggregate, few groups | Level. Keep ours. | q1 SF10 4.1 s vs 3.1 s; decimal q1 ours ahead |
| Wide-decimal reduction (two-limb) | **Comet wins, ~100x on the operator.** | decimal q6 283 ms vs 2 ms |
| Narrow decimals (INT64 path) | Refuted. Keep ours. | decimal q1 347 vs 355 ms wall, ours lower task time |
| Sort + sort-merge join at scale | **Comet's (and Spark's) win**; our merge join must not be chosen on large inputs (#287 comment). Where the hash rewrite applies, ours is the better plan by construction. | SF10 q21 44.9 s vs 16.8 / 10.9 s; q5/q7/q9 hash vs Comet's sort+merge |
| Window | Not a candidate: Comet has no window operator. Keep ours. | TPC-DS: Comet 0 windows, `WindowGroupLimit` refused |
| String functions | No specific gap beyond the filter gap; the joins around them dominate. | TPC-DS q8/q19/q79; SF10 q13/q16/q22 |
| Broadcast hash join | **Comet ahead 4x on the probe** at TPC-DS SF1, 2x at TPC-H SF10 once the input copy is separated. The largest TPC-DS lever and a candidate that clears the rule. | TPC-DS 85.8 s vs 19.7 s; SF10 q8/q9 |
| Filter, project (the controls) | Project level (TPC-DS); filter 1-3x behind even over Arrow input, so the controls do **not** all come out level: on this x86 host our per-row cost on the simplest kernel is higher than DataFusion's. #280/#281 need that answered before any allowlist -- profile the filter kernel itself (AVX-512 vs the 256-bit shapes, #282-#284). | SF10 q6/q13/q19 over Comet's scan; TPC-DS Project 1.8 vs 1.6 s |
| The input copy over Spark's scan | Not a swap: half of filter and join time at SF10 is decoding Spark's dictionary vectors (q20 profile, 25 %). Keep the dictionary through the adapter, or scan through Arrow. | SF10 q20 5.4 s → 0.66 s under Comet's scan |

### Not measured here

TPC-DS at SF10: the dataset is not on this host (`gen-tpcds.sh 10`, about 12 GB, then three
configurations at the protocol's counts are a multi-hour session); the harness runs it unchanged and
the report prints the same matrix. The x86 host is an 8-vCPU EC2 instance; the kernel-lab issues
(#282-#284) are where the filter-kernel question above gets its AVX-512 answer.

## The allowlist decided (#281)

The candidates of `spark.vector.comet.preferComet` measured as the harness's `hybrid` configuration
(`comet-scan-vector-shuffle` plus the mixed pass and one candidate's Comet toggle), TPC-H SF10 on the
same host and protocol as the #279 matrices above. The decision table with the three rules per entry
is in `docs/comet.md`. Protocol note: the #279 baselines and the first joins run kept Spark's shuffle
files in the host's RAM-backed `/tmp`; the later runs keep them on disk and set Comet's off-heap pool
(the shipped `hybrid` configuration), so they are compared with the unswapped plan rerun under the
same conditions -- that alone moves q5 from 2.9 to 7.4 s. Every run had a 13 GB memory cap.

| candidate | queries | hybrid | unswapped, same conditions | comet | wins > 5% vs unswapped | losses > 5% vs unswapped |
|---|---:|---:|---:|---:|---|---|
| `hashJoin,broadcastHashJoin` (RAM temp, first run) | 22 | 43.2 s | 42.9 s | 39.0 s | q12 1.22x, q3 1.10x | none |
| `hashJoin,broadcastHashJoin` (disk, after the fixes) | 20 | 40.7 s | 40.9 s | -- | q12 1.30x, q3 1.12x, q19 1.07x | q7 0.88x; q21 exceeded the memory cap |
| `sort` | died on q5 | -- | -- | -- | -- | q3 0.66x; 12.7 GB resident, killed |
| `filter,project` | 22 | 44.2 s | 48.4 s | 39.0 s | q19 1.72x, q6 1.70x, q14 1.43x, q15 1.30x, q13 1.29x, q20 1.26x, q12 1.21x, q4 1.17x, q5 1.16x, q3 1.15x, q21 1.10x, q16 1.05x, q22 1.05x | q8 0.86x, q11 0.86x |
| `filter` | 20 | 37.6 s | 40.9 s | -- | q19 1.74x, q14 1.45x, q15 1.29x, q20 1.29x, q12 1.26x, q6 1.23x, q13 1.22x, q2 1.19x, q7 1.12x, q3 1.09x, q16 1.07x, q4 1.06x, q10 1.06x | none past noise; q21 exceeded the memory cap |
| `project:wideDecimal`, `filter:wideDecimal` | -- | -- | -- | -- | never fired: the decimal schema is narrow | -- |

Where Comet's join crossed it was 2-3.6x faster than ours on the same rows (q3 568 -> 157 ms on
1.46M rows, q12 742 -> 334, q19 43 -> 25); where Comet's filter replaced ours on the scan it won by
the #14 dictionary decode, 8% over the suite; a projection between two of our operators cost its two
crossings (q8, q11); a Comet sort above our chain pinned every exported batch until the kernel killed
the JVM; and two candidates pushed q21 past a memory cap the unswapped plan fits under. No entry met
the three rules as written; the default allowlist stays empty, and the two costs the study names --
our join probe and our dictionary decode -- are the work to do on our side.

The same runs on the SF1 decimal schema found the hybrid configuration returning wrong results on
q11, q15, q17 and q18 while both pure configurations agreed with Spark: the C Data export widened a
wide decimal's two limbs into two rows. Fixed in the same change; all 22 decimal checksums equal
Spark's afterwards, and wide decimals now also cross Comet's native shuffle.

## x86 kernel lab: the AVX-512 loop (#283)

The hand-tuning loop of #283 run on the one x86 pool available before the lab of #282 exists: an
Intel Xeon Platinum 8488C (Sapphire Rapids; 8 vCPUs; AVX-512 F/BW/DQ/VL/VBMI/VNNI/VPOPCNTDQ -- the
lab's `x86-spr` pool in all but name), Corretto 25.0.4.8.1 (25.0.4.1+8-LTS), base commit `e00cc27`.
The evidence rule is the project's: a JMH number before and after, at `vectorBits=512` and `256`, and
under `-XX:UseAVX=2` for the AVX2 row. Both paths of a decision run on the same build, one of them
forced through `-Dsparkvector.platform` (the probe of `kernels/Platform.java`, which reads HotSpot's
`UseAVX`/`UseSVE`/`MaxVectorSize` once and folds the answer into `static final` booleans the JIT
constant-folds). Not measured here, for #282/#284: Ice Lake, Genoa (its double-pumped 512 and
`vpcompress` latency), the Arm pools (the NEON path is untouched by construction), the `hsdis` dumps.

### Decision 1: mask construction

`AggBenchmark`, 8192 rows, one thread, `-wi 3 -i 5 -w 1 -r 1`, ops/us (higher is better); the
aggregate kernels' lane masks from a validity word, built by `VectorMask.fromLong` (one `kmov` into
a `k` register: `Platform.MASK_REGISTERS`) against the broadcast-AND-compare form chosen on NEON.

| kernel | nulls | 512 bits, `fromLong` | 512 bits, broadcast | 256 bits, `fromLong` | 256 bits, broadcast | `-XX:UseAVX=2`, 256 bits (broadcast) |
|---|---:|---:|---:|---:|---:|---:|
| sumDouble | 1% | 6156 ± 94 | 5281 ± 185 | 3895 ± 125 | 3470 ± 98 | 2909 ± 51 |
| sumDouble | 30% | 4795 ± 154 | 3850 ± 229 | 2516 ± 27 | 2632 ± 59 | 2020 ± 74 |
| sumLong | 1% | 6189 ± 61 | 5276 ± 225 | 3876 ± 87 | 3408 ± 87 | 2908 ± 44 |
| sumLong | 30% | 4814 ± 83 | 3919 ± 181 | 2497 ± 69 | 2639 ± 76 | 2005 ± 87 |
| minDouble | 1% | 2905 ± 167 | 2807 ± 275 | 1127 ± 57 | 1127 ± 44 | 1035 ± 13 |
| minDouble | 30% | 2834 ± 44 | 2472 ± 457 | 1044 ± 21 | 1073 ± 29 | 899 ± 24 |

Reading: at 512 bits the mask register wins 17-25% on the sums' null paths and 15% on the minimum
with dense nulls; at 256 bits on AVX-512VL it is a wash (+12-14% with sparse nulls, -4-5% with dense,
at the edge of the error bars); the AVX2 row runs the broadcast form by construction and is slower
than either 256-bit AVX-512 row for the platform's own sake (masked operations through `k` registers
even when the mask comes from a compare). Decision: `MASK_REGISTERS` on AVX-512 at every width; the
preferred width on this host is 512 anyway. NEON and AVX2 keep the broadcast form.

### Decision 2: compaction

The rule before the lab was `compress` only for the 16-lane species (512-bit int) and the 256-entry
shuffle table for every species of 8 lanes or fewer, so on AVX-512 the long and double compactions
at 512 bits and every compaction at 256 bits took the table. `CompactBenchmark` (rows per
microsecond, `-wi 2 -i 3 -w 1 -r 1`, 1M rows; both forms on one build through the
`sparkvector.platform` override, `avx512` = `compress`, `avx2` = the table; the last column is the JIT
held at `-XX:UseAVX=2`, where `compress` does not exist and the table is the only form):

| type, selectivity, nulls | 512 `compress` | 512 table | 256 `compress` | 256 table | AVX2 table |
|---|---|---|---|---|---|
| INT32 2% | 15116 | 15147 | 15156 | 15129 | 15091 |
| INT32 2%, nulls | 8982 | 8989 | 9036 | 9003 | 9038 |
| INT32 50% | 5501 | 5342 | 2909 | 2284 | 2465 |
| INT32 50%, nulls | 4652 | 4657 | 2603 | 2240 | 2289 |
| INT32 98% | 4945 | 5193 | 4474 | 3830 | 3822 |
| INT32 98%, nulls | 3903 | 4250 | 3761 | 3207 | 3431 |
| FLOAT64 2% | 7247 | 7326 | 6030 | 5864 | 5797 |
| FLOAT64 2%, nulls | 5664 | 5635 | 4798 | 4629 | 4619 |
| FLOAT64 50% | 2035 | 1594 | 1641 | 1385 | 1222 |
| FLOAT64 50%, nulls | 2446 | 1946 | 1574 | 1262 | 1124 |
| FLOAT64 98% | 3759 | 3262 | 2825 | 2568 | 2200 |
| FLOAT64 98%, nulls | 3190 | 2828 | 2442 | 2230 | 1905 |

Reading. The INT32 rows at 512 bits are 16 lanes, `compress` in both columns by construction, and
agree within the error bars: the measurement's own sanity check. Where the two forms differ,
`compress` wins on every dense selection: 8-lane doubles at 512 bits +28% at 50% (+26% with nulls)
and +15% at 98% (+13%); at 256 bits on AVX-512VL, 8-lane ints +27% at 50% and +17% at 98%, 4-lane
doubles +18% and +10%. At 2% the two forms tie because the sparse path (a scalar walk over the set
bits of the selection word) serves both. The full-word bulk copy stays: the 98% rows are within reach
of the 2% rows on ints only because most words are full and copied whole, and a masked store would
replace a `memcpy` with a per-word instruction. Decision: `compress` wherever the platform has it
(`Platform.NATIVE_COMPRESS`, AVX-512 and SVE), at every width and for every species; the table
stays for the 8-lane species on NEON and AVX2, the scalar walk for the 2-lane species there. A
second reading from the same table: compaction at 512 bits is 1.3-1.9x its 256-bit self on dense
selections (INT32 50% 5501 against 2909), an early data point for decision 5.

### Decision 3: the grouped-aggregation thresholds

`GroupedAggBenchmark` (4096-row double sum, rows per microsecond, `-wi 2 -i 3 -w 1 -r 1`): the masked
path (one masked SIMD reduction per group) against the scatter (a scalar `sum[g] += x` loop), for 1
to 32 uniformly random groups, at both widths. `interleave` is the scatter's accumulator copies;
`interleave=1` also switches every double sum to Spark's sequential order, which is what
`strictFloatingPoint` (the default) does at run time, so the `i1` columns are the production rounding
mode and the `i4` columns the fast one.

| groups | 512 i4 mask / scatter | 512 i2 | 512 i1 (Spark's order) | 256 i4 | 256 i1 |
|---:|---|---|---|---|---|
| 1 | 5829 / 753 | 7615 / 691 | 1681 / 425 | 3982 / 754 | 1622 / 427 |
| 2 | 2066 / 746 | 2249 / 747 | 1202 / 803 | 1337 / 728 | 1205 / 802 |
| 4 | 1163 / 740 | 1210 / 760 | 1108 / 1070 | 826 / 747 | 1129 / 1055 |
| 6 | 855 / 743 | 882 / 766 | 1087 / 1234 | 604 / 740 | 1095 / 1227 |
| 8 | 686 / 748 | 702 / 764 | 1051 / 1250 | 433 / 740 | 1049 / 1252 |
| 12 | 490 / 748 | 497 / 762 | 941 / 1234 | 369 / 742 | 952 / 1234 |
| 16 | 215 / 738 | 240 / 761 | 870 / 1243 | 124 / 745 | 875 / 1240 |
| 32 | 158 / 731 | 161 / 763 | 656 / 1278 | 111 / 742 | 668 / 1285 |

Reading, threshold. The masked path wins up to 4 groups at both widths in both rounding modes (512
bits, 4 groups: +57% fast, a tie in Spark's order; 256 bits: +11% and +7%) and loses from 6 in
Spark's order (-12%) and from 8 (512) or 6 (256) in the fast mode. The old default of 8 on 8-lane
species was a guess and is 8% slow at 8 groups; the old default of 1 on 4-lane species was NEON's
measurement (2 lanes, `docs/results.md` above) and left +84% at 2 groups and +11% at 4 on the 256-bit
AVX-512 species. Decision: `maskPathMaxGroups` defaults to 4 where the platform has mask registers and
the double species has at least 4 lanes, 1 elsewhere (NEON measured; AVX2 unmeasured and kept
conservative). TPC-H Q1's 4 groups stay on the masked path.

Reading, interleave. The scatter's copies were measured on NEON (+40% for 4 copies at 4 groups,
above). On this host the picture inverts: 4 copies win only at 1-2 groups (753 against 425 at one
group), which the masked path owns, and from 4 groups one copy is 45-70% faster (1070 against 740 at
4, 1250 against 748 at 8, 1278 against 731 at 32) -- the four-way loop's extra indexing costs more
than the store-to-load chains it breaks, which random group ids already break. Decision: one copy on
AVX-512 by default, four elsewhere; the explicit `sparkvector.agg.interleave=1` keeps its meaning
(Spark's order in every double sum, `GroupedAccumulators.SEQUENTIAL_SUMS`), the default of one copy
does not imply it. Also visible: in Spark's order the masked path walks only the group's rows while
the lane-parallel one re-reads the batch per group, so from 8 groups the sequential masked sum is the
faster of the two masked forms (1051 against 686) -- moot, since the scatter owns that range.

### Decision 5: the default width

TPC-H Q1 and Q6 at SF10, the `vector` configuration (`local[8]`, 8g, 5 warm-ups, 7 measured), the
only change between the two runs `-Dsparkvector.vectorBits`; same checksums.

| query | 512 bits | 256 bits | 512 / 256 | where |
|---|---:|---:|---:|---|
| Q1 | 983 ms (min 965) | 1272 ms (min 1167) | 0.77x | the grouped aggregate: 2913 against 4676 ms of task time; the filter 289 against 339 |
| Q6 | 553 ms (min 520) | 552 ms (min 544) | 1.00x | the filter is 1239 ms of task time at either width: scan and dictionary decode, not the lanes |

Reading. Q1 is the kernel query and 512 bits win it by 23%, almost all of it in the 4-group masked
aggregation (decisions 1 and 3 are both on that path: `fromLong` masks and the masked path up to 4
groups); Q6 is a scan-bound filter whose time does not move with the width at all. Decision: the
preferred width (`Species.SHAPE`, 512 on this host) stays the default; `-Dsparkvector.vectorBits=256`
stays the override for a host where the JIT's 512-bit code is slower than its 256-bit code (Ice Lake's
frequency licence, Genoa's double-pumped units -- the measurement for #282/#284 to make). The
decision-2 table agrees from the other side: compaction at 512 bits is 1.3-1.9x its 256-bit self.

### Decision 6: gather

Not taken. No profile in this loop showed a gather (the hash join probes and the dictionary decode
walk scalar indices by design, #14), so there is nothing to measure `VectorSpecies.fromArray` with an
index map against.

### The loop, closed

| decision | before | after | evidence |
|---|---|---|---|
| 1, mask construction | broadcast-AND-compare on every platform | `VectorMask.fromLong` where the platform has mask registers | +14-25% on the null paths at 512, a wash at 256 (#312) |
| 2, compaction | `compress` only on 16-lane species, the shuffle table below | `compress` at every width where the platform has it | +10-28% on dense selections, a tie at 2% (#313) |
| 3, grouped thresholds | masked path up to 8 groups on 8 lanes, 1 on 4; four scatter copies | masked path up to 4 groups where masks are registers and the species has 4+ lanes; one scatter copy on AVX-512 | the masked path never loses up to 4; one copy +45-70% from 4 groups (#314) |
| 4, the platform switch | none | `kernels/Platform.java`, `-Dsparkvector.platform` override | both paths of every decision measured on one build (#312) |
| 5, default width | preferred (512 here) | unchanged | Q1 0.77x at 512, Q6 flat |
| 6, gather | -- | not taken | no profile shows one |

What this lab did not measure is the other pools: Ice Lake and Genoa (#282, #284: the 512-bit
frequency and double-pumping questions, `vpcompress` latency, whether decision 5 flips) and Graviton
(#253: SVE, where `MASK_REGISTERS` and `NATIVE_COMPRESS` are true by construction and unmeasured).
Every switch above reads `Platform`, so those pools are a measurement away, not a code change.


## Graviton4 (#253): the kernels on SVE, and TPC-DS 1 TB against Spark

Nodes: `m8g.4xlarge` (AWS Graviton4, Neoverse V2, `sve sve2 svebitperm`). Corretto 25.0.4.1 runs there with `UseSVE = 2` and `MaxVectorSize = 16`, so the Vector API's species are 128 bits wide, as on NEON.

### The kernels

Each JMH benchmark ran three ways on one node:

- **`sve`:** SVE codegen, with the platform switch on `sve`.
- **`sve-neon`:** SVE codegen, with the switch forced to `neon`.
- **`neon`:** `UseSVE=0`.

The first two isolate our SVE paths on the same machine code. The findings:

- **Native `compress`:** `compress` (SVE `COMPACT`) beats the shuffle table: 1.37x / 1.36x at 50 % selectivity on INT32, without and with nulls, and neutral elsewhere.
- **`fromLong` masks:** the `fromLong` lane masks lose to broadcast-AND-compare on the same SVE code. Double-checked with 2 forks × 5 iterations:
  - `minDouble_simd`: 0.77x / 0.50x (1 % / 30 % nulls)
  - `sumLong_simd`: 0.78x / 0.62x
  - `sumDouble_simd`: 0.76x / 0.93x
  - `sumMaskPath` with 4 groups: 0.72x
- **Cause:** with `-XX:+PrintIntrinsics`, the SVE run inlines `jdk.incubator.vector.VectorMask::lambda$fromLong$0`, the Vector API's Java fallback. So `fromLong` is not intrinsified at 128-bit SVE on this JDK.
- **Everything else:** SVE codegen against NEON codegen is up to 3.1x on compaction with nulls. The sort and group-key table are neutral.

Decision (#484): `Platform.MASK_REGISTERS` is AVX-512 only, while `NATIVE_COMPRESS` stays on for AVX-512 and SVE. `-Dsparkvector.maskRegisters=true` re-measures SVE after a JDK update.

### TPC-DS 1 TB

The published x86 setup, moved to 9 × `m8g.4xlarge`:

- **Settings:** 8 executors × 13 cores × 50 GB, Spark 20/30 and ours 30/20; 300 shuffle partitions, advisory 128m, `minPartitionNum=208` (as the x86 run, for comparability; later runs leave it unset); event logs on, AOT cache off.
- **Data and runs:** the same S3 Parquet data, one measured iteration.
- **Image:** arm64, built from main + #481 + #484.
- **Engines:** OSS Spark against `vector-shuffle`. Comet was not run.

The page is [benchmarks/tpcds-1tb-graviton.html](benchmarks/tpcds-1tb-graviton.html).

| | Spark (s) | ours (s) | speedup | geomean |
|---|---|---|---|---|
| Graviton4 (m8g.4xlarge) | 2,695.5 | 2,158.5 | 1.25x | 1.22x |
| x86 (m5.4xlarge, AVX-512) | 3,308.7 | 2,556.5 | 1.29x | 1.24x |
| Graviton4 / x86 per query (geomean) | 0.81 | 0.82 | | |

- **Correctness:** all 103 queries ran under both engines, with equal row counts. Checksums are equal except q65, whose result has ties.
- **Speed:** we are faster than Spark on 86 of 103 queries. Executor time is 53.7 h against 67.3 h, and shuffle read 0.52 TB against 0.94 TB.
- **Where the lead shrinks:** mostly queries where Spark itself gains more on Graviton4 (q45, q6, q29, q77, q78, q93).
- **Where it grows:** queries at parity or behind on x86 (q7 0.90x → 1.71x, q25 0.86x → 1.39x, q11 0.93x → 1.22x, q9 1.03x → 1.45x).
