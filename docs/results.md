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
