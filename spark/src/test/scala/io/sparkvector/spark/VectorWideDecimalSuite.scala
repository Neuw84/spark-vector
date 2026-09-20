package io.sparkvector.spark

import io.sparkvector.spark.adapter.SparkColumnVectorBuffers
import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.vector.{VectorFilterExec, VectorProjectExec}

/**
 * Wide decimals (p > 18) as a DECIMAL128 lane (#257, the storage half of #28): a column read from
 * Parquet -- plain or dictionary encoded, Spark's big-endian byte strings -- becomes two little-endian
 * limbs, is compacted by our filter and forwarded by our projection as an Arrow Decimal128 vector,
 * and reads back through Spark's `getDecimal` exactly. No kernel computes on the lane yet (#258), so
 * an expression over a wide column still falls back with its type reason.
 */
class VectorWideDecimalSuite extends VectorQuerySuite {
  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]

  private val wideExprs = Seq(
    "cast(id as int) as i",
    // decimal(38,10): the extremes of the type, values straddling the 64-bit limb boundary, nulls.
    "case when id % 101 = 1 then cast('9999999999999999999999999999.9999999999' as decimal(38,10)) " +
      "     when id % 101 = 2 then cast('-9999999999999999999999999999.9999999999' as decimal(38,10)) " +
      "     when id % 101 = 3 then cast('922337203.6854775807' as decimal(38,10)) " +
      "     when id % 101 = 4 then cast('-922337203.6854775808' as decimal(38,10)) " +
      "     when id % 101 = 5 then cast('922337203.6854775808' as decimal(38,10)) " +
      "     when id % 101 = 6 then cast(0 as decimal(38,10)) " +
      "     when id % 13 = 0 then null " +
      "     else cast(cast(id as decimal(38,10)) * cast('1234567890123.0000000001' as decimal(38,10)) as decimal(38,10)) end as w38",
    // decimal(27,2): TPC-DS's running-total shape, negative and small values, few distinct values (dictionary).
    "case when id % 17 = 0 then null else cast(cast((id % 40) - 20 as decimal(27,2)) * cast('1000000000000000.25' as decimal(27,2)) as decimal(27,2)) end as w27",
    // decimal(20,0): fits 9 bytes, the shortest wide form Parquet emits.
    "cast(cast(id as decimal(20,0)) * 10000000000 as decimal(20,0)) as w20")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    val df = spark.range(0, 20000).selectExpr(wideExprs: _*)
    df.repartition(3).write.mode("overwrite").option("parquet.enable.dictionary", "true").parquet(newTempPath("wide/dict"))
    df.repartition(3).write.mode("overwrite").option("parquet.enable.dictionary", "false").parquet(newTempPath("wide/plain"))
    spark.read.parquet(newTempPath("wide/dict")).createOrReplaceTempView("tw_dict")
    spark.read.parquet(newTempPath("wide/plain")).createOrReplaceTempView("tw_plain")
  }

  private def lanesAdaptedBy[T](f: => T): Long = {
    val before = SparkColumnVectorBuffers.wideDecimalColumns()
    f
    SparkColumnVectorBuffers.wideDecimalColumns() - before
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: a filter compacts wide decimal columns as DECIMAL128 lanes and the rows read back exactly") {
      val adapted = lanesAdaptedBy {
        checkVectorized(s"SELECT i, w38, w27, w20 FROM $t WHERE i % 3 = 0 AND i > 10", Seq(Filter))
      }
      // Three wide columns per batch, several batches: the lane path ran (a RemappedColumnVector never adapts).
      assert(adapted >= 3, s"expected the wide columns to be adapted into lanes, saw $adapted")
      // Every extreme survives the round trip.
      checkVectorized(s"SELECT w38 FROM $t WHERE i % 101 IN (1, 2, 3, 4, 5, 6) AND i < 2000", Seq(Filter))
      // A sparse selection (one survivor per word) and a dense one exercise both compaction paths.
      checkVectorized(s"SELECT w38, w27 FROM $t WHERE i % 64 = 5", Seq(Filter))
      checkVectorized(s"SELECT w38, w20 FROM $t WHERE i % 64 <> 5", Seq(Filter))
    }

    test(s"$t: a projection forwards wide decimal columns beside computed ones, compacting them under a selection") {
      val adapted = lanesAdaptedBy {
        checkVectorized(s"SELECT w38, i + 1 AS j, w27, w20 FROM $t WHERE i % 5 = 0", Seq(Filter, Project))
      }
      assert(adapted >= 3, s"expected the wide columns to be adapted into lanes, saw $adapted")
      // Without a selection the column is borrowed as is (no adaptation, no copy).
      checkVectorized(s"SELECT w38, i * 2 AS k FROM $t", Seq(Project))
    }
  }

  test("expressions over a wide decimal still fall back with the type reason until #258") {
    // String parsing into the lane and half-even rounding are the rest of #258; arithmetic, comparisons, casts, abs, negation and round compile.
    checkFallback("SELECT bround(w27, 1) AS r FROM tw_plain", Seq(Project), "decimal")
    checkFallback("SELECT cast(cast(w20 AS string) AS decimal(20,0)) AS s FROM tw_plain", Seq(Project), "unsupported cast string -> decimal(20,0)")
    checkFallback("SELECT w27 % cast(7 AS decimal(27,2)) AS m FROM tw_plain", Seq(Project), "% over decimal(27,2) not supported")
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: casts to and from the wide lane, negation and abs, with Spark's overflow semantics (#258)") {
      for (ansi <- Seq("false", "true")) {
        withConf("spark.sql.ansi.enabled" -> ansi) {
          // Wide to wide: widening, a scale change up and down (half up), the shortest wide form.
          checkVectorized(s"SELECT i, cast(w27 AS decimal(38,10)) AS c FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, cast(w38 AS decimal(38,2)) AS c FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, cast(w20 AS decimal(25,5)) AS c FROM $t", Seq(Project))
          // Wide to narrow (the INT64 lane) where every row fits, and narrow to wide.
          checkVectorized(s"SELECT i, cast(w20 AS decimal(18,0)) AS c FROM $t WHERE w20 < 100000000000000000", Seq(Filter, Project))
          checkVectorized(s"SELECT i, cast(cast(i AS decimal(10,2)) AS decimal(30,4)) AS c FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, cast(i AS decimal(25,3)) AS c, cast(cast(i AS bigint) * 1000000000 AS decimal(38,0)) AS d FROM $t", Seq(Project))
          // Out of the lane: double, long, int, string.
          checkVectorized(s"SELECT i, cast(w20 AS double) AS d, cast(w38 AS double) AS e FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, cast(w27 AS bigint) AS l FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, cast(w38 AS bigint) AS l FROM $t WHERE w38 > -1000000000 AND w38 < 1000000000", Seq(Filter, Project))
          checkVectorized(s"SELECT i, cast(w38 AS int) AS n FROM $t WHERE w38 > -1000000 AND w38 < 1000000", Seq(Filter, Project))
          checkVectorized(s"SELECT i, cast(w38 AS string) AS s, cast(w27 AS string) AS s2, cast(w20 AS string) AS s3 FROM $t", Seq(Project))
          // Negation and abs, also under a comparison and inside arithmetic. The extreme decimal(38,10) rows are excluded from
          // the w38 statements: Spark itself raises NUMERIC_VALUE_OUT_OF_RANGE negating them under ANSI (its Decimal for the
          // literal-derived value holds a rounded form), where the limbs negate exactly.
          checkVectorized(s"SELECT i, abs(w27) AS a FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, -w27 AS n FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, -w38 AS n FROM $t WHERE i % 101 > 6", Seq(Filter, Project))
          checkVectorized(s"SELECT i FROM $t WHERE abs(w27) > 1000000000000000 AND -w20 < 0", Seq(Filter))
          checkVectorized(s"SELECT i, abs(w38 - w27) AS d FROM $t WHERE i % 101 > 6", Seq(Filter, Project))
        }
      }
      // A narrowing cast that overflows: null rows in legacy mode; in ANSI mode Spark's CAST_OVERFLOW,
      // but only when an overflowing row is active.
      withConf("spark.sql.ansi.enabled" -> "false") {
        checkVectorized(s"SELECT i, cast(w38 AS decimal(20,10)) AS c FROM $t", Seq(Project))
        checkVectorized(s"SELECT i, cast(w27 AS decimal(10,2)) AS c FROM $t", Seq(Project))
      }
      withConf("spark.sql.ansi.enabled" -> "true") {
        checkVectorized(s"SELECT i, cast(w38 AS decimal(20,10)) AS c FROM $t WHERE w38 > -1000000000 AND w38 < 1000000000", Seq(Filter, Project))
        val e = intercept[Exception](withPlugin(enabled = true)(spark.sql(s"SELECT i, cast(w38 AS decimal(20,10)) AS c FROM $t").collect()))
        assert(e.getMessage.contains("CAST_OVERFLOW") || e.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE"), e.getMessage)
      }
    }
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: wide decimal + - * / compile on the limbs with Spark's rounding, overflow and division semantics (#258)") {
      for (ansi <- Seq("false", "true")) {
        withConf("spark.sql.ansi.enabled" -> ansi) {
          // Sums and differences: wide with wide at different scales, wide with a literal, wide with a narrow column.
          checkVectorized(s"SELECT i, w38 + w27 AS s FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, w27 - w20 AS d FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, w38 - 1234567890123.0000000001 AS d FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, 100 + w20 AS s FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, w27 + cast(i AS decimal(10,2)) AS s FROM $t", Seq(Project))
          // A sum whose exact scale Spark caps: decimal(38,10) + decimal(38,37) rounds half up to the result scale.
          // A product whose exact scale Spark caps: decimal(38,10) * decimal(27,2) is decimal(38,6), rounded half up.
          checkVectorized(s"SELECT i, w38 * w27 AS p FROM $t WHERE i % 101 > 6 AND w27 > -100 AND w27 < 100", Seq(Filter, Project))
          // Products: the exact 128-bit product, and products Spark rounds; a wide result over a wide column and a literal.
          checkVectorized(s"SELECT i, w27 * w20 AS p FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, w20 * 3 AS p FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, w38 * cast(i % 7 AS decimal(3,1)) AS p FROM $t", Seq(Project))
          // Division: Spark's divide(38, HALF_UP) then toPrecision; w27 has zero rows (null divisor result in legacy mode).
          checkVectorized(s"SELECT i, w38 / 3 AS q FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, w20 / w38 AS q FROM $t WHERE i % 101 <> 6", Seq(Filter, Project))
          checkVectorized(s"SELECT i, w38 / w27 AS q FROM $t WHERE w27 <> 0", Seq(Filter, Project))
          if (ansi == "false") checkVectorized(s"SELECT i, w38 / w27 AS q FROM $t", Seq(Project))
          // Arithmetic under a comparison, and nested arithmetic.
          checkVectorized(s"SELECT i FROM $t WHERE w27 + 1 > 0", Seq(Filter))
          checkVectorized(s"SELECT i, (w38 + w27) * 2 - w20 AS x FROM $t", Seq(Project))
        }
      }
      // Overflow: w27 * w27 is decimal(38,4) and the extreme rows overflow it -- null in legacy mode, an
      // error in ANSI mode only when an overflowing row is active.
      withConf("spark.sql.ansi.enabled" -> "false") {
        checkVectorized(s"SELECT i, w27 * w27 AS p FROM $t", Seq(Project))
        checkVectorized(s"SELECT i, w38 * w38 AS p FROM $t", Seq(Project))
      }
      withConf("spark.sql.ansi.enabled" -> "true") {
        checkVectorized(s"SELECT i, w27 * w27 AS p FROM $t WHERE w27 > -1000000 AND w27 < 1000000", Seq(Filter, Project))
        val e = intercept[Exception](withPlugin(enabled = true)(spark.sql(s"SELECT i, w38 * w38 AS p FROM $t").collect()))
        assert(e.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE") || e.getMessage.contains("cannot be represented"), e.getMessage)
        val z = intercept[Exception](withPlugin(enabled = true)(spark.sql(s"SELECT i, w38 / w27 AS q FROM $t").collect()))
        assert(z.getMessage.contains("DIVIDE_BY_ZERO"), z.getMessage)
      }
    }
  }

  test("narrow operands with a wide declared result keep the speculative INT64 path under sum (#26 precedence)") {
    import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Multiply, Subtract, Literal}
    import org.apache.spark.sql.types.{Decimal, DecimalType}
    val price = AttributeReference("l_extendedprice", DecimalType(15, 2))()
    val discount = AttributeReference("l_discount", DecimalType(15, 2))()
    val one = Literal(Decimal(1), DecimalType(1, 0))
    val product = Multiply(price, Subtract(one, discount))
    val speculative = io.sparkvector.spark.expr.ExpressionCompiler.speculativeDecimalArithmetic(product, Seq(price, discount))
    assert(speculative.exists(_.exists(_.isInstanceOf[io.sparkvector.spark.expr.SpeculativeDecimalMulExpr])), speculative.toString)
    // The same product as a projected value compiles onto the wide lane.
    val projected = io.sparkvector.spark.expr.ExpressionCompiler.compile(product, Seq(price, discount))
    assert(projected.exists(_.isInstanceOf[io.sparkvector.spark.expr.WideDecimalArithExpr]), projected.toString)
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: comparisons and IN over wide decimal columns and literals compile on the limbs (#258)") {
      for (ansi <- Seq("true", "false")) {
        withConf("spark.sql.ansi.enabled" -> ansi) {
          // Column against a wide literal, every operator, literals at the extremes and straddling the limb boundary.
          checkVectorized(s"SELECT i, w38 FROM $t WHERE w38 > 922337203.6854775807", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w38 >= cast('922337203.6854775808' as decimal(38,10))", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w38 < cast('-922337203.6854775808' as decimal(38,10))", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w38 <= 0", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w38 = cast('9999999999999999999999999999.9999999999' as decimal(38,10))", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w38 <> cast('-9999999999999999999999999999.9999999999' as decimal(38,10))", Seq(Filter))
          // Literal on the left, and the shortest wide form.
          checkVectorized(s"SELECT i FROM $t WHERE 0 > w27", Seq(Filter))
          checkVectorized(s"SELECT i, w20 FROM $t WHERE w20 >= 100000000000000", Seq(Filter))
          // IN over wide literals, with the nulls of w27 staying null.
          checkVectorized(s"SELECT i, w27 FROM $t WHERE w27 IN (0, -20000000000000005.00, 19000000000000004.75)", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w27 NOT IN (0, 1000000000000000.25)", Seq(Filter))
          // Under AND/OR with narrow predicates, and as a projected boolean.
          checkVectorized(s"SELECT i FROM $t WHERE (w38 > 0 AND w27 < 0) OR i % 101 = 3", Seq(Filter))
          checkVectorized(s"SELECT i, w38 > 0 AS pos, w27 = 0 AS zero FROM $t", Seq(Project))
        }
      }
    }
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: wide decimals as values -- CASE WHEN results, scalar-subquery filters, round (#326)") {
      for (ansi <- Seq("true", "false")) {
        withConf("spark.sql.ansi.enabled" -> ansi) {
          // TPC-DS q4/q11/q74's shape: a CASE whose branch is a wide division, no ELSE (null), then compared.
          checkVectorized(s"SELECT i, CASE WHEN w27 > 0 THEN w38 / w27 END AS ratio FROM $t", Seq(Project))
          checkVectorized(s"SELECT i FROM $t WHERE CASE WHEN w27 > 0 THEN w38 / w27 END > CASE WHEN w20 > 0 THEN w38 / w20 END", Seq(Filter))
          // Wide branches: columns, a wide literal, an ELSE; a null branch.
          checkVectorized(s"SELECT i, CASE WHEN i % 3 = 0 THEN w38 WHEN i % 3 = 1 THEN cast('1.5' as decimal(38,10)) ELSE null END AS c FROM $t", Seq(Project))
          checkVectorized(s"SELECT i, IF(w27 < 0, w27, cast(0 as decimal(27,2))) AS nonpos FROM $t", Seq(Project))
          // TPC-DS q14/q23/q24's shape: a scalar subquery of a wide decimal in a filter (avg over w27: the sum of w38 overflows Spark's own decimal(38) accumulator).
          checkVectorized(s"SELECT i, w27 FROM $t WHERE w27 > (SELECT avg(w27) * 1.1 FROM $t)", Seq(Filter))
          checkVectorized(s"SELECT i FROM $t WHERE w27 <= (SELECT max(w27) FROM $t WHERE i % 2 = 0)", Seq(Filter))
          // A subquery over no rows: a null literal, so nothing passes.
          checkVectorized(s"SELECT i FROM $t WHERE w27 > (SELECT max(w27) FROM $t WHERE i < 0)", Seq(Filter))
          // TPC-DS q2/q59's shape: round of a wide quotient to a narrow result.
          checkVectorized(s"SELECT i, round(w38 / w27, 2) AS r2, round(w38 / 7, 0) AS r0 FROM $t WHERE w27 <> 0", Seq(Project))
          checkVectorized(s"SELECT i, round(w38, 3) AS r3 FROM $t WHERE w38 IS NOT NULL", Seq(Project))
        }
      }
    }
  }

  private val Agg = classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]
  private val Window = classOf[org.apache.spark.sql.vector.VectorWindowExec]

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: window aggregates over wide decimals -- whole partition and running -- and wide window keys (#259)") {
      for (ansi <- Seq("false", "true")) {
        withConf("spark.sql.ansi.enabled" -> ansi) {
          // Whole-partition frames: the aggregate machinery over the lane, one value per partition.
          checkVectorized(s"SELECT i, sum(w27) OVER (PARTITION BY i % 7) AS s, avg(w27) OVER (PARTITION BY i % 7) AS a FROM $t WHERE i < 4000", Seq(Window))
          checkVectorized(s"SELECT i, min(w38) OVER (PARTITION BY i % 5) AS lo, max(w38) OVER (PARTITION BY i % 5) AS hi, count(w38) OVER (PARTITION BY i % 5) AS c FROM $t WHERE i < 4000", Seq(Window))
          // Running frames (q51's shape): the partial buffers combined per row, finalised as the merge would.
          checkVectorized(s"SELECT i, sum(w27) OVER (PARTITION BY i % 7 ORDER BY i) AS running FROM $t WHERE i < 4000", Seq(Window))
          checkVectorized(s"SELECT i, avg(w27) OVER (PARTITION BY i % 7 ORDER BY i) AS running_avg, count(w27) OVER (PARTITION BY i % 7 ORDER BY i) AS c FROM $t WHERE i < 4000", Seq(Window))
          checkVectorized(s"SELECT i, sum(w38) OVER (PARTITION BY i % 7 ORDER BY i) AS running, min(w38) OVER (PARTITION BY i % 7 ORDER BY i) AS lo FROM $t WHERE i < 4000 AND i % 101 > 6", Seq(Window))
          // Wide partition and order keys: the rank family and offsets over the lane, every null ordering.
          for (order <- Seq("ASC NULLS FIRST", "ASC NULLS LAST", "DESC NULLS FIRST", "DESC NULLS LAST")) {
            checkVectorized(s"SELECT i, row_number() OVER (ORDER BY w38 $order, i) AS rn, rank() OVER (ORDER BY w27 $order, i) AS r FROM $t WHERE i < 3000", Seq(Window))
          }
          checkVectorized(s"SELECT i, lag(w38, 1) OVER (ORDER BY i) AS prev, lead(w27, 2) OVER (ORDER BY i) AS nxt FROM $t WHERE i < 3000", Seq(Window))
          checkVectorized(s"SELECT i, count(*) OVER (PARTITION BY w27) AS c, sum(i) OVER (PARTITION BY w27 ORDER BY i) AS s FROM $t WHERE i < 3000", Seq(Window))
        }
      }
      // A running sum past the buffer precision: null from that row on in legacy mode, ANSI raises.
      withConf("spark.sql.ansi.enabled" -> "false") {
        checkVectorized(s"SELECT i, sum(cast(cast(w38 AS decimal(38,0)) * 1000000000 AS decimal(38,0))) OVER (ORDER BY i) AS s FROM $t WHERE i % 101 = 1 AND i < 3000", Seq(Window))
      }
      withConf("spark.sql.ansi.enabled" -> "true") {
        val e = intercept[Exception](withPlugin(enabled = true)(spark.sql(s"SELECT i, sum(cast(cast(w38 AS decimal(38,0)) * 1000000000 AS decimal(38,0))) OVER (ORDER BY i) AS s FROM $t WHERE i % 101 = 1 AND i < 3000").collect()))
        assert(e.getMessage.contains("ARITHMETIC_OVERFLOW") || e.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE"), e.getMessage)
      }
      // Sliding ROWS frames over decimals stay refused, with a reason naming the frame.
      checkFallback(s"SELECT i, sum(w27) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s FROM $t WHERE i < 2000", Seq(Window), "sliding frame")
    }
  }
  private val BHJ = classOf[org.apache.spark.sql.vector.VectorBroadcastHashJoinExec]
  private val SHJ = classOf[org.apache.spark.sql.vector.VectorShuffledHashJoinExec]

  test("wide decimal join keys and payloads in the broadcast and shuffled hash joins (#259)") {
    // A small dimension keyed on a wide decimal (the 40 distinct w27 values), with wide payloads on both sides.
    spark.sql("SELECT w27 AS dk, min(w38) AS dw38, max(w20) AS dw20, count(*) AS dc FROM tw_plain WHERE w27 IS NOT NULL GROUP BY w27")
      .write.mode("overwrite").parquet(newTempPath("wide/dim"))
    spark.read.parquet(newTempPath("wide/dim")).createOrReplaceTempView("tw_dim")
    for (hint <- Seq("BROADCAST(d)", "SHUFFLE_HASH(d)")) {
      val join = if (hint.startsWith("BROADCAST")) BHJ else SHJ
      // The build side's wide key and payloads come as rows (broadcast) or as batches (shuffled); the
      // streamed side's wide key is a lane; null keys never match.
      checkVectorized(s"SELECT /*+ $hint */ t.i, t.w27, d.dw38, d.dc FROM tw_dict t JOIN tw_dim d ON t.w27 = d.dk WHERE t.i < 6000", Seq(join, Filter))
      checkVectorized(s"SELECT /*+ $hint */ t.i, t.w38, d.dw20 FROM tw_dict t LEFT JOIN tw_dim d ON t.w27 = d.dk WHERE t.i < 6000", Seq(join, Filter))
      // A right outer join builds the left (non-preserved) side, so the hint names it: the wide build key comes from batches or rows.
      checkVectorized(s"SELECT /*+ ${hint.replace("(d)", "(t)")} */ t.i, d.dk FROM tw_dict t RIGHT JOIN tw_dim d ON t.w27 = d.dk AND t.i < 500", Seq(join))
      checkVectorized(s"SELECT /*+ $hint */ t.i FROM tw_dict t LEFT SEMI JOIN tw_dim d ON t.w27 = d.dk AND d.dc > 400", Seq(join))
      checkVectorized(s"SELECT /*+ $hint */ t.i, t.w27 FROM tw_dict t LEFT ANTI JOIN tw_dim d ON t.w27 = d.dk", Seq(join))
      // A wide key beside a narrow one, and a wide non-equi condition over the lane (#258's compare).
      checkVectorized(s"SELECT /*+ $hint */ t.i, d.dc FROM tw_dict t JOIN tw_dim d ON t.w27 = d.dk AND t.i % 40 = d.dc % 40 WHERE t.i < 3000", Seq(join, Filter))
      checkVectorized(s"SELECT /*+ $hint */ t.i, d.dw38 FROM tw_dict t JOIN tw_dim d ON t.w27 = d.dk AND t.w38 > d.dw38 WHERE t.i < 6000", Seq(join, Filter))
    }
    // A shuffled full outer join on the wide key, wide payloads on both sides.
    checkVectorized("SELECT /*+ SHUFFLE_HASH(d) */ t.i, t.w38, d.dk, d.dw20 FROM tw_dict t FULL OUTER JOIN tw_dim d ON t.w27 = d.dk WHERE t.i IS NULL OR t.i < 2000", Seq(SHJ))
    // Two wide keys (w27, w20) joining the two fixture tables to themselves.
    checkVectorized("SELECT /*+ SHUFFLE_HASH(b) */ a.i, b.i AS j FROM tw_dict a JOIN tw_plain b ON a.w27 = b.w27 AND a.w20 = b.w20 WHERE a.i < 2000", Seq(SHJ, Filter))
  }
  private val TopN = classOf[org.apache.spark.sql.vector.VectorTakeOrderedAndProjectExec]
  private val Collect = classOf[org.apache.spark.sql.vector.VectorCollectLimitExec]
  private val LocalLimit = classOf[org.apache.spark.sql.vector.VectorLocalLimitExec]
  private val Sample = classOf[org.apache.spark.sql.vector.VectorSampleExec]
  private val Expand = classOf[org.apache.spark.sql.vector.VectorExpandExec]
  private val Rollup = classOf[org.apache.spark.sql.vector.VectorRollupExec]
  private val Union = classOf[org.apache.spark.sql.vector.VectorUnionExec]
  private val Coalesce = classOf[org.apache.spark.sql.vector.VectorCoalesceExec]

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: the column movers carry wide decimals -- take-ordered, limits, sample, expand, union, coalesce (#259)") {
      // ORDER BY a wide key with LIMIT: Spark's top-k over our batches, the projection applied by Spark's
      // row projection, the result a batch with a DECIMAL128 lane. All four null orderings.
      for (order <- Seq("ASC NULLS FIRST", "ASC NULLS LAST", "DESC NULLS FIRST", "DESC NULLS LAST")) {
        checkVectorized(s"SELECT i, w38, w27 FROM $t ORDER BY w38 $order, i LIMIT 25", Seq(TopN))
      }
      checkVectorized(s"SELECT w27, w20 * 2 AS d FROM $t WHERE i > 100 ORDER BY w27 DESC, i LIMIT 40", Seq(TopN, Filter))
      // Limits: a collect limit and a local limit over wide columns, across a batch boundary.
      checkVectorized(s"SELECT i, w38, w20 FROM $t LIMIT 5000", Seq(Collect))
      assert(nodesOf[org.apache.spark.sql.vector.VectorLocalLimitExec](
        checkVectorized(s"SELECT count(*) AS c, sum(w20) AS s FROM (SELECT w20 FROM $t LIMIT 7000)", Seq(Agg))).nonEmpty)
      // A sample keeps the wide columns with their rows.
      checkVectorized(s"SELECT i, w38, w27 FROM $t TABLESAMPLE (30 PERCENT) REPEATABLE (7)", Seq(Sample))
      // Expand: a wide grouping key nulled per grouping set (a wide null constant column), a wide sum through it.
      checkVectorized(s"SELECT w27, i % 3 AS k, count(*) AS c, sum(w20) AS s, grouping_id() AS gid FROM $t WHERE i < 4000 GROUP BY ROLLUP(w27, i % 3)", Seq(Rollup, Agg)) // a rollup is the chain since #383
      checkVectorized(s"SELECT w27, w20, count(*) AS c FROM $t WHERE i < 3000 GROUP BY GROUPING SETS ((w27), (w20), ())", Seq(Expand, Agg))
      // Union of two wide-sum aggregates (q66's shape: two channels summed then unioned), and a wide key coalesced.
      checkVectorized(
        s"SELECT k, s FROM (SELECT i % 5 AS k, sum(w27) AS s FROM $t WHERE i % 2 = 0 GROUP BY i % 5 UNION ALL SELECT i % 5 AS k, sum(w20) AS s FROM $t WHERE i % 2 = 1 GROUP BY i % 5) u",
        Seq(Union, Agg))
      checkVectorized(s"SELECT /*+ COALESCE(1) */ i, w38 FROM $t WHERE i % 7 = 0", Seq(Coalesce, Filter))
    }
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: wide decimal grouping keys, inputs and result arithmetic in the hash aggregate (#259)") {
      for (ansi <- Seq("false", "true")) {
        withConf("spark.sql.ansi.enabled" -> ansi) {
          // Wide grouping keys (two limbs hashed and compared), alone and beside a narrow key; nulls group together.
          checkVectorized(s"SELECT w27, count(*) AS c, sum(i) AS s FROM $t GROUP BY w27", Seq(Agg))
          checkVectorized(s"SELECT w20, i % 3 AS k, count(*) AS c FROM $t WHERE i < 3000 GROUP BY w20, i % 3", Seq(Agg))
          checkVectorized(s"SELECT w38, w27, count(*) AS c FROM $t WHERE i % 101 IN (1, 2, 3, 4, 5, 6) GROUP BY w38, w27", Seq(Agg))
          // Wide inputs: sum and average into the 128-bit accumulator, min / max on the limbs, count, first / last.
          // The decimal(38,10) extremes are excluded from the sums: an intermediate total past the
          // precision poisons Spark's buffer in row order, which no engine reproduces for another.
          checkVectorized(s"SELECT i % 7 AS k, sum(w38) AS s, sum(w27) AS s2, sum(w20) AS s3 FROM $t WHERE i % 101 > 6 GROUP BY i % 7", Seq(Agg))
          checkVectorized(s"SELECT i % 7 AS k, avg(w27) AS a, avg(w20) AS a2 FROM $t GROUP BY i % 7", Seq(Agg))
          checkVectorized(s"SELECT i % 7 AS k, min(w38) AS lo, max(w38) AS hi, min(w27) AS lo2, max(w20) AS hi2 FROM $t GROUP BY i % 7", Seq(Agg))
          checkVectorized(s"SELECT i % 7 AS k, count(w38) AS c, count(w27) AS c2 FROM $t GROUP BY i % 7", Seq(Agg))
          checkVectorized(s"SELECT i % 5 AS k, first(w38) AS f, last(w27) AS l FROM $t WHERE i < 400 AND i % 13 <> 0 AND i % 17 <> 0 GROUP BY i % 5", Seq(Agg))
          // Ungrouped.
          checkVectorized(s"SELECT sum(w27) AS s, avg(w27) AS a, min(w38) AS lo, max(w38) AS hi, count(w20) AS c FROM $t", Seq(Agg))
          checkVectorized(s"SELECT sum(w38) AS s FROM $t WHERE i % 101 > 6", Seq(Agg))
          // Arithmetic over wide sums in the result projection (#245's shapes), and a wide sum under a narrow one.
          checkVectorized(s"SELECT i % 7 AS k, sum(w27) / 7.0 AS q, 0.5 * sum(w20) AS h, sum(w27) - sum(w20) AS d FROM $t GROUP BY i % 7", Seq(Agg))
          checkVectorized(s"SELECT i % 7 AS k, sum(w27) + sum(i) AS m, avg(w27) * 2 AS a2 FROM $t GROUP BY i % 7", Seq(Agg))
          // The wide sum's result compared and cast in the same projection.
          checkVectorized(s"SELECT i % 7 AS k, cast(sum(w27) AS double) AS d, sum(w20) > 0 AS pos FROM $t GROUP BY i % 7", Seq(Agg))
        }
      }
      // A total past the sum's precision: 18 rows of 10^37 leave 128 bits as well as the 38 digits.
      // Null in legacy mode; Spark's numeric range error in ANSI mode.
      withConf("spark.sql.ansi.enabled" -> "false") {
        checkVectorized(s"SELECT sum(cast(cast(w38 AS decimal(38,0)) * 1000000000 AS decimal(38,0))) AS s FROM $t WHERE i % 101 = 1 AND i < 1900", Seq(Agg))
        checkVectorized(s"SELECT sum(w38) AS s FROM $t WHERE i % 101 = 1", Seq(Agg))
      }
      withConf("spark.sql.ansi.enabled" -> "true") {
        val e = intercept[Exception](withPlugin(enabled = true)(spark.sql(s"SELECT sum(cast(cast(w38 AS decimal(38,0)) * 1000000000 AS decimal(38,0))) AS s FROM $t WHERE i % 101 = 1 AND i < 1900").collect()))
        assert(e.getMessage.contains("NUMERIC_VALUE_OUT_OF_RANGE") || e.getMessage.contains("ARITHMETIC_OVERFLOW"), e.getMessage)
      }
    }
  }
}

/** Sorting over the lane: the four-pass two-limb key order equals Spark's in every partition. */
class VectorWideDecimalSortSuite extends VectorQuerySuite {
  private val Sort = classOf[org.apache.spark.sql.vector.VectorSortExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark.range(0, 20000).selectExpr(
      "cast(id as int) as i",
      "case when id % 101 = 1 then cast('9999999999999999999999999999.9999999999' as decimal(38,10)) " +
        "     when id % 101 = 2 then cast('-9999999999999999999999999999.9999999999' as decimal(38,10)) " +
        "     when id % 101 = 3 then cast('922337203.6854775807' as decimal(38,10)) " +
        "     when id % 101 = 4 then cast('-922337203.6854775808' as decimal(38,10)) " +
        "     when id % 101 = 5 then cast('922337203.6854775808' as decimal(38,10)) " +
        "     when id % 13 = 0 then null " +
        "     else cast(cast((id % 4000) - 2000 as decimal(38,10)) * cast('1234567890123.0000000001' as decimal(38,10)) as decimal(38,10)) end as w38",
      "case when id % 17 = 0 then null else cast(cast((id % 40) - 20 as decimal(27,2)) * cast('1000000000000000.25' as decimal(27,2)) as decimal(27,2)) end as w27")
      .repartition(3).write.mode("overwrite").parquet(newTempPath("wide/sort"))
    spark.read.parquet(newTempPath("wide/sort")).createOrReplaceTempView("tws")
  }

  private def keySequences(sql: String, numKeys: Int, enabled: Boolean): Seq[Seq[Seq[Any]]] =
    withPlugin(enabled) {
      spark.sql(sql).rdd.glom().collect().toSeq.map(_.toSeq.map(r => (0 until numKeys).map(c => r.get(c))))
    }

  private def checkSorted(sql: String, numKeys: Int): Unit = {
    checkVectorized(sql, Seq(Sort))
    val expected = keySequences(sql, numKeys, enabled = false)
    val actual = keySequences(sql, numKeys, enabled = true)
    assert(actual.length === expected.length, s"partition count for: $sql")
    expected.zip(actual).zipWithIndex.foreach { case ((e, a), p) => assert(a === e, s"key order differs in partition $p for: $sql") }
  }

  test("a wide decimal sort key in every direction and null order, with wide payload columns") {
    checkSorted("SELECT w38, i, w27 FROM tws SORT BY w38", 1)
    checkSorted("SELECT w38, i FROM tws SORT BY w38 DESC", 1)
    checkSorted("SELECT w38, i FROM tws SORT BY w38 ASC NULLS FIRST", 1)
    checkSorted("SELECT w38, i FROM tws SORT BY w38 DESC NULLS LAST", 1)
    checkSorted("SELECT w27, w38, i FROM tws SORT BY w27 DESC NULLS FIRST", 1)
    // Wide payloads gathered under an int key; a wide second key deciding ties of a coarse first key.
    checkSorted("SELECT i, w38, w27 FROM tws SORT BY i DESC", 1)
    checkSorted("SELECT i % 3, w38, w27 FROM tws SORT BY i % 3, w38 DESC NULLS LAST", 2)
  }

  test("a computed expression over a wide decimal is a sort key since #258") {
    checkVectorized("SELECT w38, i FROM tws SORT BY w38 * 2", Seq(Sort))
  }
}
