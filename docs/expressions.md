# Spark expression support

One row per Spark expression: whether spark-vector compiles it, which lane types it accepts, and the
incompatibility that still makes it fall back where one exists. Modelled on Comet's
[Spark Expression Support](https://datafusion.apache.org/comet/user-guide/latest/expressions.html)
page; the companion for operators is [docs/operators.md](operators.md).

**How to read it.** `ExpressionCompiler.compile` (`spark/src/main/scala/io/sparkvector/spark/expr/`)
is one `match` over Catalyst expressions. Every case either produces a `VectorExpr` node with a kernel
behind it or returns a `Left(reason)`; the reason is what the operator records (`VectorFallback`), what
the UI tooltip and `spark.vector.explainFallback.enabled` show, and what the suites assert on with
`checkFallback(..., reasonContains)`. An expression that is not matched at all records
`unsupported expression <Class>: <sql>` -- so a row missing from the **Supported** table below is a
`not yet`, and the **Planned** table says which issue tracks it.

**Types** are the `VecType` lanes, since that is what decides support here:

| Lane | Spark types |
|---|---|
| BOOL | `boolean` |
| INT32 | `int`, `date` |
| INT64 | `bigint`, `timestamp`, `decimal(p <= 18)` (unscaled value; the scale stays in the Spark type) |
| FLOAT64 | `double` |
| UTF8 | `string` (carried through filters, projections and as a grouping/join key; compared in UTF8_BINARY order by `StringCompareKernels`) |

Anything else -- `decimal(p > 18)`, `float`, `short`, `byte`, `binary`, `array`, `map`, `struct`,
intervals -- has no lane: a column of such a type records `unsupported type <type> for <name>`, a
result of such a type `unsupported output type <type> for <name>`, whatever the expression. Wide
decimals are the usual reason in practice (see the TPC-H decimal measurement in
[docs/results.md](results.md) and #26 / #27 / #28).

**Literals** are compiled only as operands of a supported type: `int`, `bigint`, `double`, `date`,
`timestamp`, `decimal(p <= 18)`, `string`. A `boolean` literal records `unsupported literal type
boolean`; a `NULL` literal `null literal`. An expression made only of literals is
refused where it would be pointless as a kernel (`comparison of two literals`, `arithmetic on two
literals`, `cast of a literal`, `literal predicate`, ...) -- Spark's optimizer normally folds those away
before we see them; a bare literal projection (`SELECT 1 FROM t`) is supported and materialised as a
constant column.

Spark 4 defaults to **ANSI mode**. Where ANSI changes the semantics (integer overflow, division by
zero, decimal overflow) the row says whether the kernel raises the same error Spark does or falls back.
Errors are raised only for rows that are active -- survivors of earlier conjuncts / the selection --
matching Spark's short-circuit behaviour.

## Supported

| Expression | Types | Notes / fallback reasons |
|---|---|---|
| Column reference (`AttributeReference`, `BoundReference`) | all lanes | `unbound attribute <name>` if the attribute is not in the operator's input; `unsupported type <type> for <name>` otherwise |
| `Alias` | any | Transparent |
| Literal | INT32, INT64, FLOAT64, date, timestamp, decimal(<=18), string | Operand only, or a whole projected column (a string literal becomes a constant UTF8 column). `unsupported literal type <type>`, `null literal` |
| `=` `<` `<=` `>` `>=` and `!=` / `<>` (`Not(EqualTo)`) | INT32, INT64, FLOAT64 (incl. date, timestamp, decimal(<=18) as their lane), UTF8 | Operands must have the **same** Spark type -- Spark's coercion inserts casts, which then have to compile (see Cast): `comparison operands differ: <t1> vs <t2>`. Strings compare in Spark's default `UTF8_BINARY` order (unsigned byte-wise, a prefix first) against a literal or another string column; a dictionary-encoded column is compared once per dictionary entry. Booleans: `comparison not supported for boolean` (#32). Doubles compare with Spark's ordering (NaN equal to NaN and greatest, `-0.0 == 0.0`); `KnownFloatingPointNormalized` / `NormalizeNaNAndZero` wrappers are identities. |
| `IN (v1, ..., vN)` | any comparable lane incl. UTF8 | Every element must be a non-null literal of the value's type (`InExpr`: the value is evaluated once, one equality pass per literal, through the dictionary for dictionary-encoded strings). `NULL in IN list`, `IN list is not all literals`, `IN operands differ: ...`, `IN over a literal`, `empty IN list`; above `spark.sql.optimizer.inSetConversionThreshold` literals Spark rewrites to `InSet` -- see its row |
| `InSet` (an `IN` list above `spark.sql.optimizer.inSetConversionThreshold`, 10 by default) | INT32, INT64, FLOAT64 (incl. date, decimal(<=18) as their lane), UTF8 | `InSetExpr` over `PredicateKernels.inSet`: the set's values sorted once at compile time, one binary search per row; doubles are matched on their bit images, which is Spark's boxed-set semantics (NaN matches NaN, `-0.0` does not match `0.0`). Strings take the `IN` path above (one equality pass per element, through the dictionary). `NULL in IN set` (Spark's result is then null for non-members), `IN not supported for <type>` |
| `<=>` (`EqualNullSafe`) | every comparable lane incl. UTF8 and BOOL | `NullSafeEqExpr`: the ordinary equality, then `(eq AND valid) OR (both null)` on the validity bitmaps -- never null. `x <=> NULL` compiles to `x IS NULL`. A boolean literal side falls back (`<=> of a boolean literal`); same-type rule as `=` |
| `isnan` | FLOAT64 | `IsNaNExpr` over a Vector API `IS_NAN` test packed into the bitmap; false (not null) where the argument is null, as Spark's. `isnan over <type>` for a non-double |
| `=` `<>` `<` `<=` `>` `>=` on BOOL | BOOL column vs boolean literal or BOOL column | `BoolCompareExpr` / `BoolCompareScalarExpr` on the packed words (`false < true`); null where an operand is. `comparison against NULL` |
| `BETWEEN` | as its comparisons | Spark rewrites it to two comparisons before planning; pinned by a test |
| `startswith`, `endswith`, `contains`; `LIKE 'p%'`, `LIKE '%p'`, `LIKE '%p%'` (Spark's `LikeSimplification` rewrites these three shapes into the functions) | UTF8 vs a string literal | `StringMatchExpr` over `StringMatchKernels`: byte-level under the default `UTF8_BINARY` collation (a UTF-8 pattern can only match at character boundaries), the empty pattern matches everything, once per dictionary entry on dictionary-encoded columns; prefix and suffix are one `MemorySegment.mismatch` over a fixed range, contains scans for the first byte and confirms with `mismatch`. `null pattern`, `string pattern is not a literal`, `string pattern of type <t>`, `string match not supported for <t>`, `string match on a literal`. A `LIKE` the optimizer does not simplify -- inner wildcards (`'%a%b%'`), `_`, a bare `'%'`, `'a%b'` (needs `length`) -- stays `Like` and falls back with `unsupported expression Like` (#3 follow-up: run Spark's matcher per dictionary entry) |
| `AND`, `OR`, `NOT` | BOOL | Operands must be non-literal booleans: `boolean literal operand`, `expected boolean, got <type>` |
| `IS NULL`, `IS NOT NULL` | all lanes | `null test on literal` |
| `+` `-` `*` on integers | INT32, INT64 | Same-typed operands (`arithmetic operands differ`). Both modes: legacy wraps like Spark; in ANSI mode (Spark's default) the wrapped result is checked with an overflow lane mask (`OverflowKernels`: sign trick for `+ -`, exact product for `*`) and Spark's `ARITHMETIC_OVERFLOW` is raised -- `integer overflow` / `long overflow` with the `try_add` / `try_subtract` / `try_multiply` hint -- only if an **active** row overflowed, so rows a filter removed or an earlier conjunct decided never raise. `try_*`: `try_* arithmetic not supported` (#47) |
| `+` `-` `*` `/` on doubles | FLOAT64 | Bit-identical in ANSI and legacy mode; `/` by zero yields null (legacy) or raises `DIVIDE_BY_ZERO` (ANSI) with Spark's error context. Integer division falls back: `division not supported for <type>` |
| `+` `-` `*` `/` on decimals | INT64 (decimal(<=18) operands **and** result) | Both operands decimal (`mixed decimal and non-decimal arithmetic` otherwise); Spark's result type must fit 18 digits: `decimal result <type> exceeds 18 digits` -- `decimal(15,2) * decimal(15,2)` is `decimal(31,4)` and falls back (#26). Only `/` can overflow and it checks; ANSI raises `NUMERIC_VALUE_OUT_OF_RANGE`, legacy yields null. Division is Spark-exact (round half up at the result scale). `unexpected decimal result scale` guards against an operand shape the kernel does not expect |
| Unary minus | INT32, INT64, FLOAT64, decimal(<=18) | ANSI mode raises `ARITHMETIC_OVERFLOW` (`integer overflow` / `long overflow`, no hint) for an active `MIN_VALUE`; doubles and decimals never overflow here |
| `Cast` | INT32 -> INT64, INT32 -> FLOAT64, INT64 -> FLOAT64; int/bigint/double -> decimal(<=18); decimal -> decimal / double / bigint / int; timestamp -> date under a UTC or fixed-offset session zone (`TimestampToDateExpr`: `floorDiv(micros + offset, micros per day)`; Spark inserts this cast under `year(ts)` etc.); a cast to the operand's own type | Everything else: `unsupported cast <from> -> <to>` / `unsupported cast target <type>` (#43 -- narrowing, boolean, string <-> number, string <-> date/timestamp, date -> timestamp). timestamp -> date under a zone with rules: `cast timestamp -> date needs a fixed-offset session zone, not <zone>`. Decimal casts check the range; ANSI raises, legacy nulls. `try_cast not supported` (#47) |
| `year`, `month`, `dayofmonth` / `day`, `dayofyear`, `quarter`, `dayofweek`, `weekday`, `extract(<field> FROM date)` | INT32 days -> INT32 | `DateFieldExpr` over `DateKernels.field`: branch-free civil-from-days per lane (no `LocalDate`), valid for negative days and every leap rule; Spark's numbering (`dayofweek` 1 = Sunday, `weekday` 0 = Monday). Over a timestamp Spark first casts to date (see Cast). `date function over <type>`, `date function on a literal` |
| `trunc(date, unit)` | INT32 -> INT32 | Units `YEAR`/`YYYY`/`YY`, `QUARTER`, `MONTH`/`MON`/`MM`, `WEEK` (Monday), case-insensitive, as a string literal. `trunc unit '<u>' not supported`, `trunc unit is not a string literal` |
| `date_add`, `date_sub`, `datediff` | INT32 lanes | `ArithExpr` add/subtract on days (Spark does not overflow-check these): `date +/- int` and `date - date`; either side may be a literal. `date arithmetic over <type>`, `date arithmetic with <type> days`, `arithmetic on two literals` |
| `hour`, `minute`, `second` | INT64 micros -> INT32 | Under a UTC or fixed-offset session zone only (`TimeFieldExpr`: local micros = `micros + offset`); a zone with rules falls back: `time field needs a fixed-offset session zone, not <zone>`. `time field over <type>` |
| `monotonically_increasing_id()` | -> INT64 | `MonotonicIdExpr`: `partitionIndex << 33` (from the task context) plus a running row number carried across the partition's batches -- Spark's contract, and identical values for the same plan since the numbering follows the rows that reach the expression: with a forwarded selection only the selected rows are numbered (`SequenceKernels.iotaSelected`). Rows outside a CASE branch's active mask are still numbered (Spark would skip them); the id is a tag, and uniqueness and monotonicity within the partition hold either way |
| `abs`, `positive`, `negative` | INT32, INT64, FLOAT64 | `AbsExpr` over `MathKernels.abs`; ANSI mode raises `ARITHMETIC_OVERFLOW` (`integer overflow` / `long overflow`, no hint -- `negateExact`'s message) for an active `MIN_VALUE`, legacy wraps. `positive` is the identity, `negative` is `UnaryMinus`. `abs of a literal`, `abs over <type> not supported` (decimals: #49) |
| `sign` / `signum` | FLOAT64 -> FLOAT64 | Spark casts the argument to double first (see Cast); NaN stays NaN, `-0.0` stays `-0.0` |
| `%` / `mod`, `pmod` | INT32, INT64, FLOAT64 | `DivideLikeExpr` over `MathKernels.remainder`: Java's truncated remainder for `%`, Spark's `r = a % n; r < 0 ? (r + n) % n : r` for `pmod` (int arithmetic wraps like Spark's). A zero divisor never reaches the arithmetic: on an active row ANSI raises Spark's `REMAINDER_BY_ZERO`, legacy nulls the lane; rows a filter removed or an earlier conjunct decided never raise. Same Spark type on both sides; `try_*` falls back. `% over <type> not supported` (decimals: #49), `% operands differ: ...` |
| `div` (`IntegralDivide`) | INT32 / INT64 operands -> INT64 | Truncation toward zero, long result as Spark's; zero divisor as above but with `DIVIDE_BY_ZERO`; ANSI raises `ARITHMETIC_OVERFLOW` (`Overflow in integral divide`, hint `try_divide`) for an active `Long.MIN_VALUE div -1`. `div over double not supported`, decimals #49 |
| `greatest`, `least` | INT32, INT64, FLOAT64 | `PickExpr` over `MathKernels.pick`: variadic, every child the same Spark type after Spark's casts, null operands ignored, all-null is null, doubles in Spark's ordering (NaN greatest, `-0.0 == 0.0`); literal children become constant columns. `greatest operands differ: ...`, `greatest over <type> not supported`, `greatest of literals only` |
| `nanvl` | FLOAT64 | `NanvlExpr` over `MathKernels.nanvl` with Spark's eval semantics: null where the first argument is null, the first argument where it is not NaN (whatever the second is), otherwise the second (null if it is null). `nanvl over <type>`, `nanvl of two literals` |
| `ceil`, `floor` (one argument) | INT64 (identity), FLOAT64 -> INT64, decimal(p,s) -> decimal(p-s+1, 0) | `CeilFloorExpr` over `RoundKernels`: Spark casts an int argument to long first; a double becomes a long through Java's `(long) Math.ceil(d)` (NaN is 0, the infinities saturate); a decimal is an unscaled divide by `10^s` in CEILING / FLOOR mode to Spark's `bounded(p - s + 1, 0)` type. `ceil over decimal(18,4) -> decimal(19,0) not supported` when the widened result passes 18 digits |
| `rint` | FLOAT64 | `RintExpr`, `Math.rint` |
| `round(x[, k])`, `bround(x[, k])` | INT32, INT64, FLOAT64, decimal | `RoundExpr` -- HALF_UP for `round`, HALF_EVEN for `bround`, each as Spark's `RoundBase`: a double is `BigDecimal(Double.toString(d)).setScale(k, mode).doubleValue()` (the shortest decimal representation is rounded, so `round(2.675, 2)` is 2.68; NaN and the infinities pass through; scalar path); an integer with `k >= 0` is itself, with `k < 0` rounded to a power of ten -- a result outside the type raises `ARITHMETIC_OVERFLOW` (`Overflow`, no hint) in ANSI mode for active rows and wraps like `BigDecimal.intValue()` otherwise; a decimal is an unscaled-value operation to the type Spark computed (`(p - s + 1 + min(s, k), min(s, k))`, or `(max(p - s + 1, -k + 1), 0)` for a negative `k`). `k` must be an int literal (Spark requires it foldable). `round over decimal(18,4) -> decimal(19,4) not supported` when Spark's widened result passes 18 digits |
| `ceil(x, k)`, `floor(x, k)` | decimal (int arguments arrive cast to decimal(10,0)) | `RoundExpr` in CEILING / FLOOR mode (`RoundCeil` / `RoundFloor`), same result-type rule as `round`. Spark casts a double argument to decimal(30,15) first, which stays a fallback (`cast ... not supported`, #49) |
| `&`, `\|`, `^`, `~` | INT32, INT64 | `BitBinaryExpr` / `BitNotExpr` over `BitKernels`, Vector API lanewise ops; both operands the same integral type after Spark's coercion; a literal on either side. Bytes, shorts and booleans have no lane here (`& over tinyint, tinyint not supported`); boolean `and`/`or` are the logical path above |
| `shiftleft`, `shiftright`, `shiftrightunsigned` (`<<`, `>>`, `>>>`) | INT32, INT64 value, INT32 amount | `BitBinaryExpr` with Java's (= Spark's) semantics: the amount is masked by the lane width (`x << 33` on an int is `x << 1`, negative amounts wrap the same way), `>>>` on INT32 is computed in 32 bits, the amount may be a literal or a column, and a literal value shifted by a column (`1 << i`) is handled too |
| `bit_count` | INT32, INT64 -> INT32 | `BitCountExpr`: `java.lang.Long.bitCount` of the value **widened to a long**, exactly as Spark computes it -- `bit_count(CAST(-1 AS INT))` is 64, not 32. `bit_count over boolean not supported` (booleans have no INT lane to widen) |
| `bit_get` / `getbit` | fallback | Returns TINYINT, which has no lane in this project (no INT8): `bit_get returns tinyint, which has no lane` |
| `UnscaledValue`, `MakeDecimal` | INT64 (decimal(<=18)) | The optimizer's `DecimalAggregates` rewrite of `sum(decimal(p <= 8))` and `avg(decimal(p <= 11))`; `MakeDecimal` into more than 18 digits falls back (`make_decimal into <type> exceeds 18 digits`), overflow nulls or raises per `nullOnOverflow` (#49 covers `CheckOverflow` and the rest of that family) |
| `KnownFloatingPointNormalized`, `NormalizeNaNAndZero` | FLOAT64 | Identities: the compare kernels already use the normalised ordering and double grouping keys are refused |
| `CASE WHEN ... THEN ... [ELSE ...] END` | result of any lane; conditions BOOL | `CaseWhenExpr` over `SelectKernels`: each condition is evaluated only on the rows no earlier branch took, its winning rows are `condition is true` (a null condition counts as false), each branch value only on its winning rows; the result is null where the winner is null or no branch matched and there is no `ELSE`. Branches must share the result's Spark type (`branch type <t> differs from <t>`); `NULL` and literal branches -- string and boolean literals included -- are materialised as constant columns. `unsupported result type <type> for <sql>` for a wide decimal or nested result |
| `IF(c, a, b)` | as `CASE WHEN` | The one-branch case with an `ELSE` |
| `COALESCE(a, b, ...)`, `NVL`, `NVL2`, `NULLIF`, `IFNULL` | as `CASE WHEN` | `COALESCE` is `IS NOT NULL` conditions over the operands with the last as `ELSE` (an operand is evaluated once for its test and once for its value); `NVL`/`NVL2`/`NULLIF`/`IFNULL` arrive as `COALESCE`/`IF` through Spark's own rewrites |

### Aggregate functions

Compiled by `VectorAggregates` for `HashAggregateExec` in every mode (see
[docs/operators.md](operators.md) for the operator's own conditions).

| Function | Types | Notes / fallback reasons |
|---|---|---|
| `count(*)`, `count(x)` | any lane | `count with several arguments not supported` |
| `sum` | INT32, INT64, FLOAT64, decimal with a buffer of <= 18 digits | ANSI `sum(bigint)` is overflow-checked (a sign-trick overflow lane, `Math.addExact` on the grouped path). `sum(decimal(p <= 8))` arrives as `MakeDecimal(sum(UnscaledValue))` and is supported; wider: `sum buffer <type> exceeds 18 digits` (#27), `sum over <type> producing <type> not supported`. Double sums add in lane-parallel, interleaved order (`sparkvector.agg.interleave`, default 4): results can differ from Spark's in the last bits; `interleave=1` reproduces Spark's rounding |
| `min`, `max` | INT32, INT64, FLOAT64 (incl. date, timestamp, decimal(<=18)) | `min/max over <type> not supported` (strings, booleans) |
| `avg` | INT32, INT64, FLOAT64 producing `double`; decimal only through the optimizer's rewrite for p <= 11 | `avg producing <type> not supported`, `avg buffer <type> exceeds 18 digits` (#27) |
| `first(x, ignoreNulls = true)` / `first_value(x, true)` | INT32, INT64, FLOAT64, date, boolean | `FirstAgg` / `FirstMergeAgg` with Spark's `(first, valueSet)` buffer: the first non-null value in row order per group (row order within a task; across the shuffle Spark's own result is order-dependent too). Spark's distinct rewrite wraps every plain aggregate in this form. `first over <type> without ignoreNulls not supported` (a null first value is a real value there), `first over <type> not supported` (strings, decimals) |
| any of the above with `FILTER (WHERE ...)` | the predicate must compile as a boolean | `FilteredAgg`: applied in the update modes only (`Partial`, `Complete`) as a narrowed selection for that function -- Spark drops the clause in the merge modes. `FILTER <sql>: <reason>` when the predicate does not compile |
| any of the above with `DISTINCT` | as the function | The `isDistinct` flag is only a marker in a physical plan: Spark's `planAggregateWithOneDistinct` (one distinct group) groups by the distinct column in the two inner stages and the flagged function runs over deduplicated input as a plain one; several distinct groups go through Expand with a keys-only first aggregate, `FILTER (WHERE gid = k)` on every function and `first(..., true)` around the plain ones -- all of which compile. A distinct **with** `FILTER` folds its condition with `max` over a boolean in the first aggregate, which falls back (`min/max over boolean not supported`, #45) |
| any other function | -- | `unsupported aggregate function <Class>: <sql>` (#45, #46) |

`aggregate over a literal`, `aggregate over <type> not supported` (non-numeric input) and
`literal grouping key` / `grouping key type <type> not supported` (double keys, unsupported types) are
the remaining reasons on the aggregate's inputs.

## Planned

| Family | Issue |
|---|---|
| General `LIKE` (inner wildcards, `_`, `'a%b'`), `rlike` | #3 follow-up (the `LikeSimplification` shapes are Supported above) |
| Decimal results wider than 18 digits on narrow operands (`decimal(15,2) * decimal(15,2)`) | #26 |
| 128-bit `sum` / `avg` buffers for decimals beyond 8 / 11 digits | #27 |
| Genuinely wide declared decimals (`p > 18`) | #28 (design note first) |
| Transcendental and trigonometric functions | #35 |
| `bit_get` / `getbit` (needs an INT8 lane) | #36 |
| String length and character functions | #37 |
| String case and trim | #38 |
| `substring`, padding, repetition | #39 |
| String search and replace (`instr`, `locate`, `replace`, `translate`, `split_part`) | #40 |
| `concat`, `concat_ws`, `elt` | #41 |
| Hash functions (`hash`, `xxhash64`, `md5`, `sha1`, `sha2`, `crc32`) | #42 |
| The rest of the cast matrix (narrowing, boolean, string <-> number, string <-> date/timestamp, date -> timestamp, timestamp -> date under zone rules) | #43 |
| The rest of the datetime family (`date_trunc` on timestamps, ISO weeks, `add_months`, `last_day`, `unix_timestamp`, timestamp functions under zones with rules) | #44 |
| Aggregate functions beyond `count`/`sum`/`min`/`max`/`avg`/`first(ignoreNulls)`; `min`/`max` over booleans and strings | #45 |
| `stddev`, `variance`, `covar`, `corr` | #46 |
| `try_add`, `try_divide`, `try_cast`, `try_sum`, `try_avg` | #47 |
| Optimizer-injected: `InSet`, `ScalarSubquery`, bloom-filter probes, normalisation | #48 |
| `CheckOverflow` and the internal decimal family | #49 |
| Nested-type accessors (struct field, array element, map value) | #50 |
| A row-based escape hatch for expressions with no kernel, including Scala UDFs | #51 |
| Structural audit: literals, aliases, sort orders, `CASE WHEN` shapes | #52 |

## Not planned

Recorded here so that an absence is a decision rather than an omission (#66). Open to revisiting on
demand, not permanent exclusions. Comet's list is the reference for what a columnar engine reasonably
leaves out.

| Family | Reason |
|---|---|
| Probabilistic sketches (`approx_count_distinct` / HyperLogLog++, `approx_percentile`, `count_min_sketch`, `hll_*` functions) | Sketch state is a Spark-defined binary buffer with its own merge semantics; the buffers are the compatibility surface, not the arithmetic, and Spark's implementation is already tight |
| Geospatial functions | Outside the project's data model; no lane type and no plan to add one |
| Avro / Protobuf codecs (`from_avro`, `to_avro`, `from_protobuf`, `to_protobuf`) | Schema-driven record (de)serialisation, row by row by construction |
| JVM reflection (`java_method`, `reflect`) | Calls arbitrary JVM code per row; nothing to vectorise |
| Niche string validators and encoders (`is_valid_utf8`, `make_valid_utf8`, `validate_utf8`, `try_validate_utf8`, `luhn_check`, `soundex`, `levenshtein`) | Byte-level per-value algorithms with little SIMD upside; Spark's implementations are adequate. The common string functions are planned (#37-#41) |
| Pickled (non-Arrow) Python UDFs, Scala UDFs *as kernels* | The data has to become objects row by row. The row-based escape hatch (#51) is the path for these, not a kernel |

## Keeping this page honest

Every expression issue names this file as part of its definition of done: the row lands in the same
commit as the kernel, the scalar reference and the Spark comparison test. The reason strings above are
quoted from `ExpressionCompiler`, `VectorAggregates` and `VectorHashAggregateExec`; a string that
changes without its row changing fails a `checkFallback` assertion in the suites before it reaches a
user. The intended end state is to generate the Supported table from the compiler itself -- walking
Spark's `FunctionRegistry` against a dummy schema and recording each `Left(reason)` -- so that a
regression shows up as a diff of this file rather than as a stale row.
