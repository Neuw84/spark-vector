/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.vector.AggSpillPolicy
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec}

class VectorAggregateSuite extends VectorQuerySuite {

  private val Agg = classOf[VectorHashAggregateExec]
  private val Filter = classOf[VectorFilterExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("agg/t"))
    TestTables.createLineitem(spark, newTempPath("agg/lineitem"))
  }

  test("count, sum, min, max, avg over doubles, ints and dates") {
    checkVectorized("SELECT count(*), count(d), sum(d2), min(d2), max(d2), avg(d2) FROM t", Seq(Agg))
    checkVectorized("SELECT count(i), sum(i), min(i), max(i), avg(i) FROM t", Seq(Agg))
    checkVectorized("SELECT min(dt), max(dt), count(dt) FROM t", Seq(Agg))
    checkVectorized("SELECT count(l), min(l), max(l), avg(l) FROM t", Seq(Agg))
  }

  test("NaN and infinities follow Spark's ordering in min and max") {
    // d contains NaN, +Inf and -Inf: max is NaN, min is -Inf.
    val df = checkVectorized("SELECT max(d), min(d) FROM t", Seq(Agg))
    val row = df.collect().head
    assert(row.getDouble(0).isNaN)
    assert(row.getDouble(1) === Double.NegativeInfinity)
  }

  test("aggregates over expressions and filtered input") {
    checkVectorized("SELECT sum(d2 * 2.0), avg(d2 - 1.0), count(d2 * d2) FROM t WHERE i > 1000", Seq(Filter, Agg))
    checkVectorized("SELECT sum(d2 * (1.0 - d2) * (1.0 + d2)) FROM t WHERE d2 > 0.25", Seq(Filter, Agg))
  }

  test("all-null and empty inputs produce Spark's results") {
    checkVectorized("SELECT sum(d), count(d), min(d), max(d), avg(d) FROM t WHERE d IS NULL", Seq(Filter, Agg))
    checkVectorized("SELECT sum(d2), count(*), count(d2), min(i), avg(d2) FROM t WHERE i < 0", Seq(Filter, Agg))
    checkVectorized("SELECT count(*) FROM t WHERE i < 0", Seq(Filter, Agg))
  }

  test("grouping keys emitted under an alias (#328): keys-only and with an aggregate") {
    // TPC-DS q97's shape: a distinct whose result renames its own keys.
    checkVectorized("SELECT s AS key_s, i AS key_i FROM t GROUP BY s, i", Seq(Agg))
    checkVectorized("SELECT s AS key_s, count(*) AS n, sum(d) AS total FROM t GROUP BY s", Seq(Agg))
  }

  test("#377: string keys are emitted dictionary-encoded, ids over the group table's dictionary") {
    import org.apache.spark.sql.vectorized.ColumnarBatch
    import io.sparkvector.spark.arrow.{VectorArrowColumnVector, VectorDictionaryColumnVector}
    def keyColumns(sql: String): Seq[Class[_]] = withPlugin(enabled = true) {
      val plan = finalPlan { val d = spark.sql(sql); d.collect(); d }
      val agg = org.apache.spark.sql.vector.PlanUtils.allNodes(plan).collect { case a: VectorHashAggregateExec =>
        a
      }.last // the partial, over the scan
      // Classified inside the task: a ColumnarBatch does not travel to the driver.
      org.apache.spark.sql.execution.SQLExecution.withSQLConfPropagated(
        spark.asInstanceOf[org.apache.spark.sql.classic.SparkSession]
      ) {
        agg.executeColumnar().mapPartitions(
          _.filter(_.numRows() > 0).map(_.column(0).getClass.getName)
        ).collect().toSeq.distinct
      }.map(Class.forName)
    }
    // 45 distinct strings (plus the null) over 20000 rows in the partial, 45 groups in the final.
    checkVectorized("SELECT s, count(*) AS n, sum(d) AS total FROM t GROUP BY s", Seq(Agg))
    assert(
      keyColumns("SELECT s, count(*) AS n FROM t GROUP BY s") == Seq(classOf[VectorDictionaryColumnVector]),
      "a low-cardinality key goes out as ids over the table's dictionary"
    )
    // Every row its own key: still ids plus a dictionary of the same bytes; the shuffle writer decides per block.
    checkVectorized("SELECT concat(s, '-', i) AS k, count(*) AS n FROM t GROUP BY concat(s, '-', i)", Seq(Agg))
    assert(keyColumns("SELECT concat(s, '-', i) AS k, count(*) AS n FROM t GROUP BY concat(s, '-', i)") == Seq(
      classOf[VectorDictionaryColumnVector]
    ))
    // Switched off, the low-cardinality key is plain too, and the results are the same.
    withConf(VectorConf.AggDictionaryKeys -> "false") {
      checkVectorized("SELECT s, count(*) AS n, sum(d) AS total FROM t GROUP BY s", Seq(Agg))
      assert(keyColumns("SELECT s, count(*) AS n FROM t GROUP BY s") == Seq(classOf[VectorArrowColumnVector]))
    }
  }

  test("count(*) over a scan with no projected columns") {
    checkVectorized("SELECT count(*) FROM t", Seq(Agg))
  }

  test(
    "#363: past its budget the aggregate spills -- the partial emits and restarts, the final merges bucket by bucket"
  ) {
    // 20000 distinct keys of every lane kind, an 8 KB budget: the partial emits many times, the final spills into buckets.
    withConf(AggSpillPolicy.ThresholdKey -> "8k", AggSpillPolicy.BucketsKey -> "4") {
      def spilled(df: org.apache.spark.sql.DataFrame): Unit = {
        val aggs = nodesOf[VectorHashAggregateExec](df)
        val fin = aggs.filter(_.isFinal)
        assert(fin.nonEmpty, df.queryExecution.executedPlan.treeString)
        assert(
          fin.exists(_.metrics("spills").value > 0L),
          s"the final should have spilled: ${fin.map(_.metrics("spills").value)}"
        )
        assert(
          aggs.filterNot(_.isFinal).exists(_.metrics("spills").value > 0L),
          "the partial should have emitted early"
        )
      }
      spilled(checkVectorized(
        "SELECT i, count(*) AS n, sum(l) AS sl, min(d) AS mn, max(d2) AS mx, avg(d2) AS av FROM t GROUP BY i",
        Seq(Agg)
      ))
      spilled(checkVectorized("SELECT l, dt, count(*) AS n, sum(d2) AS s FROM t GROUP BY l, dt", Seq(Agg)))
      spilled(checkVectorized(
        "SELECT s, i % 400 AS k, count(*) AS n, sum(i) AS si FROM t GROUP BY s, i % 400",
        Seq(Agg)
      ))
      spilled(checkVectorized("SELECT b, d, count(*) AS n, min(l) AS ml FROM t GROUP BY b, d", Seq(Agg)))
      // Keys only (a distinct): no Final mode to look for -- both of its aggregates must have spilled.
      val distinct = checkVectorized("SELECT DISTINCT i, s FROM t", Seq(Agg))
      val distinctAggs = nodesOf[VectorHashAggregateExec](distinct)
      assert(
        distinctAggs.size === 2 && distinctAggs.forall(_.metrics("spills").value > 0L),
        distinctAggs.map(_.metrics("spills").value).toString
      )
      spilled(checkVectorized("SELECT i, sum(l) AS sl FROM t WHERE i % 3 <> 1 GROUP BY i", Seq(Filter, Agg)))
    }
    // The budget off: nothing spills.
    withConf(AggSpillPolicy.ThresholdKey -> "0") {
      val df = checkVectorized("SELECT i, count(*) AS n FROM t GROUP BY i", Seq(Agg))
      assert(nodesOf[VectorHashAggregateExec](df).forall(_.metrics("spills").value === 0L))
    }
  }

  test("#376: a partial aggregate that does not reduce passes the rest of its input through after one emit") {
    // i is unique: a full table reduces nothing. With the ratio on, the partial emits its first table (one
    // spill counted) and then hands every batch on; with it off, it keeps filling and emitting. Same answer.
    def partialSpills(df: org.apache.spark.sql.DataFrame): Seq[Long] =
      nodesOf[VectorHashAggregateExec](df).filterNot(_.isFinal).map(_.metrics("spills").value)
    withConf(AggSpillPolicy.ThresholdKey -> "8k", AggSpillPolicy.BucketsKey -> "4") {
      val on = partialSpills(checkVectorized("SELECT i, sum(l) AS sl, count(*) AS n FROM t GROUP BY i", Seq(Agg)))
      assert(on.nonEmpty && on.sum >= 1L, s"the first table is emitted and counted: $on")
      withConf(AggSpillPolicy.PassThroughKey -> "0") {
        val off = partialSpills(checkVectorized("SELECT i, sum(l) AS sl, count(*) AS n FROM t GROUP BY i", Seq(Agg)))
        assert(off.sum > on.sum, s"without pass-through the partial emits many more tables: $off vs $on")
      }
    }
  }

  test("final aggregation merges the partial buffers over the shuffle") {
    // Ungrouped: one partial row per partition, one final row; avg is sum / count of the merged buffers.
    val df = checkVectorized(
      "SELECT sum(d2), count(*), count(l), min(d), max(i), avg(d2), avg(i) FROM t WHERE i >= 0",
      Seq(Agg)
    )
    assert(nodesOf[HashAggregateExec](df).isEmpty, df.queryExecution.executedPlan.treeString)
    val aggs = nodesOf[VectorHashAggregateExec](df)
    assert(aggs.size === 2 && aggs.count(_.isFinal) === 1, df.queryExecution.executedPlan.treeString)
    // Grouped, with result expressions that are not plain attributes.
    val grouped = checkVectorized(
      "SELECT s, count(*) AS n, sum(d) + 1.0 AS s1, avg(d2) * 2.0 AS a2, max(dt) FROM t GROUP BY s",
      Seq(Agg)
    )
    assert(nodesOf[HashAggregateExec](grouped).isEmpty)
    assert(nodesOf[VectorHashAggregateExec](grouped).count(_.isFinal) === 1)
    // Empty input: Spark's initial buffers (count 0, sum null, avg null).
    checkVectorized("SELECT count(*), sum(d), avg(d), min(i) FROM t WHERE i < 0", Seq(Agg))
    // (With AQE on, Spark replaces an empty grouped aggregate by EmptyRelation before we run.)
    withConf("spark.sql.adaptive.enabled" -> "false") {
      checkVectorized("SELECT s, count(*) FROM t WHERE i < 0 GROUP BY s", Seq(Agg))
    }
  }

  test("final aggregation stays in Spark when disabled by configuration") {
    withConf(VectorConf.FinalAggregateEnabled -> "false") {
      val df = checkVectorized("SELECT sum(d2), count(*) FROM t WHERE i >= 0", Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).nonEmpty, "expected Spark's Final HashAggregateExec")
      assert(nodesOf[VectorHashAggregateExec](df).size === 1)
    }
  }

  test("sum of bigint is overflow-checked in ANSI mode and wraps in legacy mode") {
    checkVectorized("SELECT sum(l), sum(i) FROM t", Seq(Agg))
    checkVectorized("SELECT s, sum(l) FROM t GROUP BY s", Seq(Agg))
    // Values whose sum leaves the long range, read from Parquet so the aggregate's input is columnar.
    spark.range(0, 3000).selectExpr("6000000000000000000L + id AS v", "cast(id % 2 as boolean) AS g")
      .write.mode("overwrite").parquet(newTempPath("agg/big"))
    spark.read.parquet(newTempPath("agg/big")).createOrReplaceTempView("big")
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT sum(v) FROM big", Seq(Agg)) // wraps like Spark
      checkVectorized("SELECT g, sum(v) FROM big GROUP BY g", Seq(Agg))
    }
    withPlugin(enabled = true) {
      Seq("SELECT sum(v) FROM big", "SELECT g, sum(v) FROM big GROUP BY g").foreach { sql =>
        val df = spark.sql(sql)
        val e = intercept[Exception](df.collect())
        assert(
          Iterator.iterate(e: Throwable)(
            _.getCause
          ).takeWhile(_ != null).take(10).exists(_.getMessage.contains("ARITHMETIC_OVERFLOW")),
          s"expected ARITHMETIC_OVERFLOW for $sql, got $e"
        )
        assert(
          nodesOf[VectorHashAggregateExec](df).nonEmpty,
          s"the failing aggregate should be ours: ${finalPlan(df).treeString}"
        )
      }
    }
  }

  test("unsupported aggregates fall back") {
    checkFallback("SELECT approx_count_distinct(i) FROM t", Seq(Agg), "unsupported aggregate function")
    checkVectorized(
      "SELECT first(d2, true), first(l, true), first(b, true), first(dt, true) FROM t WHERE i > 3",
      Seq(Agg)
    )
  }

  test("PartialMerge: the second stage of a count(distinct) plan merges over the exchange") {
    // Spark's rewrite: Partial(sum) over (s, i) -> exchange -> PartialMerge(sum) over (s, i) ->
    // PartialMerge(sum) + Partial(count, distinct) over s -> exchange -> Final + Final(distinct).
    // All four stages are ours: the distinct function runs over already-deduplicated input.
    val df = checkVectorized("SELECT s, count(DISTINCT i) AS cd, sum(l) AS sl FROM t GROUP BY s", Seq(Agg))
    val ours = nodesOf[VectorHashAggregateExec](df)
    assert(
      ours.size === 4,
      org.apache.spark.sql.vector.VectorFallback.reasons(finalPlan(df)).map(_._2).mkString("; ") + "\n" + finalPlan(
        df
      ).treeString
    )
    assert(nodesOf[HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
    val merge = ours.find(_.modes == Seq(org.apache.spark.sql.catalyst.expressions.aggregate.PartialMerge))
    assert(merge.isDefined, finalPlan(df).treeString)
    assert(!merge.get.emitsResults && !merge.get.isFinal)
    assert(
      ours.exists(_.modes.toSet == Set(
        org.apache.spark.sql.catalyst.expressions.aggregate.PartialMerge,
        org.apache.spark.sql.catalyst.expressions.aggregate.Partial
      )),
      finalPlan(df).treeString
    )
    // Ungrouped: the same shape with an empty key set, plus a second non-distinct function.
    val df2 = checkVectorized("SELECT count(DISTINCT s), sum(d2), max(i) FROM t WHERE i > 100", Seq(Agg))
    assert(nodesOf[HashAggregateExec](df2).isEmpty, finalPlan(df2).treeString)
  }

  test("distinct aggregation: one distinct group, TPC-H Q16 shape, nullable columns, sum(distinct)") {
    // planAggregateWithOneDistinct (no Expand): every stage is ours. Nulls are never counted.
    checkVectorized("SELECT count(DISTINCT i) AS c FROM t", Seq(Agg))
    checkVectorized("SELECT count(DISTINCT l) AS c, count(l) AS all FROM t", Seq(Agg))
    checkVectorized(
      "SELECT s, count(DISTINCT l) AS cl, sum(DISTINCT i % 5) AS sd, max(d2) AS m FROM t GROUP BY s",
      Seq(Agg)
    )
    checkVectorized("SELECT b, count(DISTINCT s) AS cs, avg(d2) AS a FROM t WHERE i > 10 GROUP BY b", Seq(Agg))
    val q16 = checkVectorized(
      "SELECT l_returnflag, count(DISTINCT l_suppkey) AS supplier_cnt FROM lineitem WHERE l_quantity > 40 GROUP BY l_returnflag",
      Seq(Agg)
    )
    assert(nodesOf[HashAggregateExec](q16).isEmpty, finalPlan(q16).treeString)
  }

  test("distinct aggregation: several distinct groups and a plain aggregate go through Expand") {
    // RewriteDistinctAggregates: a keys-only first aggregate over (s, i, l, gid), then a second whose
    // functions all carry FILTER (WHERE gid = k) and whose plain aggregate is first(..., ignoreNulls).
    val df = checkVectorized(
      "SELECT s, count(DISTINCT i) AS di, count(DISTINCT l) AS dl, sum(i) AS si, max(d2) AS m FROM t GROUP BY s",
      Seq(Agg)
    )
    assert(nodesOf[HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
    // With plain aggregates present the first aggregate carries them; without, it is keys-only (below).
    val ko = checkVectorized("SELECT count(DISTINCT i) AS di, count(DISTINCT s) AS ds FROM t WHERE i > 1000", Seq(Agg))
    assert(
      nodesOf[VectorHashAggregateExec](ko).exists(_.aggregateExpressions.isEmpty),
      "the keys-only stage is ours\n" + finalPlan(ko).treeString
    )
    checkVectorized(
      "SELECT count(DISTINCT i) AS di, count(DISTINCT s) AS ds, count(*) AS c FROM t WHERE i > 1000",
      Seq(Agg)
    )
    // A distinct with FILTER is also rewritten through Expand; its first aggregate folds the filter
    // condition with max over a boolean, which is an accumulator since #45: every stage is ours.
    val f = checkVectorized(
      "SELECT s, count(DISTINCT i) FILTER (WHERE l IS NOT NULL) AS d, sum(l) AS sl FROM t GROUP BY s",
      Seq(Agg)
    )
    assert(
      nodesOf[org.apache.spark.sql.execution.aggregate.HashAggregateExec](f).isEmpty,
      org.apache.spark.sql.vector.VectorFallback.reasons(finalPlan(f)).map(_._2).mkString("; ") + "\n" + finalPlan(
        f
      ).treeString
    )
  }

  test("aggregate functions beyond count/sum/min/max/avg: booleans, strings, bits, first/last, min_by/max_by") {
    import org.apache.spark.sql.execution.aggregate.HashAggregateExec
    def allOurs(sql: String): Unit = {
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).isEmpty, "every stage should be ours\n" + finalPlan(df).treeString)
    }
    // min/max over booleans and strings (Spark's binary order; s has nulls), bool_and/bool_or and their aliases.
    allOurs(
      "SELECT i % 5 AS g, min(b), max(b), min(s), max(s), bool_and(b), bool_or(b), every(b), any(b), some(b) FROM t GROUP BY i % 5"
    )
    allOurs("SELECT min(b), max(b), min(s), max(s), bool_and(b), bool_or(b) FROM t WHERE i > 17")
    // count_if is a count over a rewritten boolean.
    allOurs("SELECT i % 3 AS g, count_if(b), count_if(l > 500), count_if(s IS NULL) FROM t GROUP BY i % 3")
    allOurs("SELECT count_if(b), count_if(l IS NULL) FROM t")
    // Bit aggregates over int and bigint (l has nulls), grouped and not.
    allOurs(
      "SELECT i % 4 AS g, bit_and(i), bit_or(i), bit_xor(i), bit_and(l), bit_or(l), bit_xor(l) FROM t GROUP BY i % 4"
    )
    allOurs("SELECT bit_and(i), bit_or(l), bit_xor(i) FROM t WHERE i BETWEEN 100 AND 1000")
    // first/last with and without ignoreNulls: deterministic shapes -- a value constant within its group,
    // and a single ordered partition for the null-sensitive forms.
    allOurs("SELECT i % 5 AS g, first(i % 5), last(i % 5), first(s, true), last(s, true) FROM t GROUP BY i % 5")
    allOurs(
      "SELECT first(l), last(l), first(l, true), last(l, true), first(s), last(s, true) FROM (SELECT * FROM t WHERE i < 40 ORDER BY i)"
    )
    // min_by / max_by with unique orderings (ties are order-dependent in Spark too), int/double/string values and keys.
    allOurs(
      "SELECT i % 7 AS g, max_by(s, i), min_by(s, i), max_by(i, d2), min_by(l, i), max_by(d, i), min_by(b, d2) FROM t WHERE i < 5000 GROUP BY i % 7"
    )
    allOurs("SELECT max_by(s, i), min_by(i, i), max_by(l, d2) FROM t WHERE i > 100")
    // Empty groups and no rows: nulls (zero for count_if), like Spark.
    allOurs("SELECT min(s), max(b), bit_and(i), first(l), last(s), max_by(s, i), count_if(b) FROM t WHERE i < 0")
    allOurs(
      "SELECT i % 3 AS g, min(s), bool_or(b), bit_xor(l), last(s, true), min_by(s, i) FROM t WHERE i < 0 GROUP BY i % 3"
    )
    // Mixed with the existing functions and a FILTER clause in one operator.
    allOurs(
      "SELECT i % 6 AS g, count(*), sum(l), min(s) FILTER (WHERE b), max_by(s, l), bit_or(i) FILTER (WHERE i % 2 = 0), avg(i) FROM t GROUP BY i % 6"
    )
  }

  test("FILTER clauses apply in the update stage, grouped and ungrouped") {
    checkVectorized(
      "SELECT sum(d2) FILTER (WHERE i > 5) AS s1, count(*) FILTER (WHERE b) AS c1, count(*) AS c, min(i) FILTER (WHERE l IS NULL) AS m FROM t",
      Seq(Agg)
    )
    val grouped = checkVectorized(
      "SELECT s, sum(i) FILTER (WHERE i % 2 = 0) AS even, avg(d2) FILTER (WHERE d2 > 1.0) AS a, count(l) FILTER (WHERE i > 19990) AS few FROM t GROUP BY s",
      Seq(Agg)
    )
    assert(nodesOf[HashAggregateExec](grouped).isEmpty, finalPlan(grouped).treeString)
    // A FILTER whose predicate is never true leaves Spark's initial buffers: null sums, zero counts.
    checkVectorized(
      "SELECT s, sum(i) FILTER (WHERE i < 0) AS none, count(*) FILTER (WHERE i < 0) AS zero FROM t GROUP BY s",
      Seq(Agg)
    )
  }

  test("keys-only aggregates: SELECT DISTINCT and UNION are ours in both stages") {
    val df = checkVectorized("SELECT DISTINCT s, b FROM t WHERE i > 100", Seq(Agg))
    assert(nodesOf[HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
    assert(nodesOf[VectorHashAggregateExec](df).size === 2, finalPlan(df).treeString)
    checkVectorized("SELECT DISTINCT i % 7 AS m FROM t", Seq(Agg))
    checkVectorized("SELECT count(*) FROM (SELECT DISTINCT s, l FROM t)", Seq(Agg))
  }

  test("Complete: update functions with the Final result expressions in one operator") {
    import org.apache.spark.sql.catalyst.expressions.aggregate.{Complete, Final, Partial}
    import org.apache.spark.sql.execution.ColumnarToRowExec
    import org.apache.spark.sql.vector.VectorAggregatePlanner
    // Batch planning in Spark 4.1 never emits Complete (only the streaming planners do), so build the
    // operator from a real plan's two stages: the Partial stage's columnar child and functions, the
    // Final stage's result expressions.
    val sql = "SELECT s, sum(d2) AS sd, count(*) AS c, min(i) AS mi, avg(l) AS al FROM t WHERE i > 50 GROUP BY s"
    val expected = withPlugin(enabled = false)(spark.sql(sql).collect())
    val vectorPlan = withPlugin(enabled = true) { val d = spark.sql(sql); d.collect(); finalPlan(d) }
    val aggs = org.apache.spark.sql.vector.PlanUtils.allNodes(vectorPlan).collect { case h: VectorHashAggregateExec =>
      h
    }
    val partial = aggs.find(_.modes == Seq(Partial)).get
    val fin = aggs.find(_.modes == Seq(Final)).get
    val complete = HashAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = partial.groupingExpressions,
      aggregateExpressions = fin.aggregateExpressions.map(_.copy(mode = Complete)),
      aggregateAttributes = fin.aggregateAttributes,
      initialInputBufferOffset = 0,
      resultExpressions = fin.resultExpressions,
      child = partial.child
    )
    val planned = VectorAggregatePlanner.plan(complete, finalEnabled = true)
    assert(planned.isRight, planned.left.toOption.getOrElse(""))
    val ours = planned.toOption.get
    assert(ours.modes === Seq(Complete) && ours.emitsResults && !ours.isFinal)
    // A Complete aggregate emits one group set per partition, so run it over a single partition.
    val single = ours.copy(child = org.apache.spark.sql.vector.VectorCoalesceExec(1, partial.child))
    val rows = ColumnarToRowExec(single).executeCollect().map(_.copy())
    val actual = rows.map(r =>
      org.apache.spark.sql.Row.fromSeq(single.output.indices.map(i =>
        r.get(i, single.output(i).dataType) match {
          case u: org.apache.spark.unsafe.types.UTF8String => u.toString
          case v => v
        }
      ))
    )
    assertRowsEqual(expected, actual, 1e-9, "Complete " + sql)
  }

  test("an operator mixing buffer and result modes is refused with a reason") {
    import org.apache.spark.sql.catalyst.expressions.aggregate.{Final, Partial}
    import org.apache.spark.sql.vector.VectorAggregatePlanner
    val sparkPlan = withPlugin(enabled =
      false
    )(spark.sql("SELECT s, sum(d2), count(*) FROM t GROUP BY s").queryExecution.executedPlan)
    val fin = org.apache.spark.sql.vector.PlanUtils.allNodes(
      sparkPlan
    ).collect { case h: HashAggregateExec => h }.find(_.aggregateExpressions.forall(_.mode == Final)).get
    val mixed = fin.copy(aggregateExpressions =
      fin.aggregateExpressions.head.copy(mode = Partial) +: fin.aggregateExpressions.tail
    )
    val planned = VectorAggregatePlanner.plan(mixed, finalEnabled = true)
    assert(planned.isLeft && planned.left.toOption.get.contains("mix buffer and result output"), planned.toString)
  }

  test("aggregate conversion can be disabled by configuration") {
    withConf(VectorConf.AggregateEnabled -> "false") {
      val df = withPlugin(enabled = true)(spark.sql("SELECT sum(d2) FROM t"))
      df.collect()
      assert(nodesOf[VectorHashAggregateExec](df).isEmpty)
    }
  }

  test("grouped aggregation with few groups uses the mask path") {
    // s has 49 distinct values plus null: at most 50 groups, below the low-cardinality threshold.
    checkVectorized(
      "SELECT s, count(*), count(d), sum(d2), avg(d2), min(i), max(l), min(dt) FROM t GROUP BY s",
      Seq(Agg)
    )
    checkVectorized("SELECT b, sum(d2), count(*) FROM t GROUP BY b", Seq(Agg))
    checkVectorized("SELECT s, b, sum(d2 * 2.0), avg(d2) FROM t WHERE i > 100 GROUP BY s, b", Seq(Filter, Agg))
  }

  test("grouped aggregation with many groups uses the scatter path") {
    checkVectorized("SELECT i, sum(d2), count(*), max(d2) FROM t GROUP BY i", Seq(Agg))
    checkVectorized("SELECT l, count(*), sum(d2), min(i) FROM t GROUP BY l", Seq(Agg)) // nullable bigint key
    checkVectorized("SELECT dt, count(*), avg(d2) FROM t GROUP BY dt", Seq(Agg)) // 730 date keys
  }

  test("grouping keys with nulls and NaN values in aggregates") {
    checkVectorized("SELECT s, max(d), min(d), sum(d) FROM t GROUP BY s", Seq(Agg))
    checkVectorized("SELECT l, max(d) FROM t WHERE d IS NOT NULL GROUP BY l", Seq(Filter, Agg))
  }

  test("grouped aggregation over empty input emits no rows") {
    // With AQE the empty partial output makes Spark replace the stage by EmptyRelation, so only
    // check the plan shape with AQE off.
    checkVectorized("SELECT s, count(*) FROM t WHERE i < 0 GROUP BY s", Seq.empty)
    withConf("spark.sql.adaptive.enabled" -> "false") {
      val df = checkVectorized("SELECT s, count(*) FROM t WHERE i < 0 GROUP BY s", Seq(Filter, Agg))
      assert(df.count() === 0)
    }
  }

  test("strict floating point reproduces Spark's double sums bit for bit; fast mode may differ") {
    // Sums and averages, ungrouped and grouped, Partial and Final: tolerance 0 means bit equality.
    val sql = "SELECT l_returnflag, sum(l_extendedprice * (1 - l_discount)), avg(l_extendedprice), sum(l_quantity) " +
      "FROM lineitem GROUP BY l_returnflag"
    val strict = checkVectorized(sql, Seq(Agg), tolerance = 0.0)
    assert(
      nodesOf[VectorHashAggregateExec](strict).forall(_.strictFloatingPoint),
      strict.queryExecution.executedPlan.treeString
    )
    checkVectorized(
      "SELECT sum(l_extendedprice * (1 - l_discount)), avg(l_discount) FROM lineitem WHERE l_quantity < 30",
      Seq(Agg),
      tolerance = 0.0
    )
    withConf(VectorConf.StrictFloatingPoint -> "false") {
      val fast = checkVectorized(sql, Seq(Agg), tolerance = 1e-9)
      assert(
        nodesOf[VectorHashAggregateExec](fast).forall(!_.strictFloatingPoint),
        fast.queryExecution.executedPlan.treeString
      )
    }
    // TPC-H Q15's shape: a double sum compared for equality with the maximum of the same sums.
    // With interleaved accumulators this returned no rows at SF1; strict rounding keeps the row.
    val q15 =
      """WITH revenue AS (
        |  SELECT l_suppkey AS supplier_no, sum(l_extendedprice * (1 - l_discount)) AS total_revenue
        |  FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        |  GROUP BY l_suppkey)
        |SELECT supplier_no, total_revenue FROM revenue
        |WHERE total_revenue = (SELECT max(total_revenue) FROM revenue)""".stripMargin
    val rows = checkVectorized(q15, Seq(Agg), tolerance = 0.0).collect()
    assert(rows.nonEmpty, "the maximum revenue row must match itself")
  }

  test("double grouping keys group by bits after Spark's normalisation") {
    checkVectorized("SELECT d2, count(*) FROM t GROUP BY d2", Seq(Agg))
    // d has NaNs, infinities and nulls.
    checkVectorized("SELECT d, count(*), sum(i) FROM t GROUP BY d", Seq(Agg))
    checkVectorized("SELECT s, d2, min(i), max(i) FROM t GROUP BY s, d2", Seq(Agg))
  }

  test("TPC-H Q1 end to end") {
    val df = checkVectorized(TestTables.TpchQ1, Seq(Filter, Agg), tolerance = 1e-9)
    assert(nodesOf[HashAggregateExec](df).isEmpty, "both aggregation stages should be vectorized")
    val rows = df.collect()
    assert(rows.length === 3, rows.mkString("\n")) // synthetic data yields (A,F), (N,O), (R,F)
    info(s"Q1 rows:\n${rows.mkString("\n")}\n${df.queryExecution.executedPlan.treeString}")
  }

  test("TPC-H Q6 end to end") {
    val df = checkVectorized(TestTables.TpchQ6, Seq(Filter, Agg), tolerance = 1e-9)
    val revenue = df.collect().head.getDouble(0)
    assert(revenue > 0.0)
    info(s"Q6 revenue = $revenue\n${df.queryExecution.executedPlan.treeString}")
  }

  test("string predicates over Parquet dictionary pages, the TPC-H Q10/Q12 shapes") {
    // l_returnflag / l_linestatus / l_comment are low-cardinality strings the Parquet writer
    // dictionary-encodes, so the literal is compared once per dictionary entry.
    val q10 = checkVectorized(
      "SELECT l_linestatus, count(*) AS c, sum(l_extendedprice * (1 - l_discount)) AS rev FROM lineitem WHERE l_returnflag = 'R' GROUP BY l_linestatus",
      Seq(Filter, Agg),
      tolerance = 1e-6
    )
    assert(nodesOf[HashAggregateExec](q10).isEmpty)
    val q12 = checkVectorized(
      "SELECT l_returnflag, count(*) AS c FROM lineitem WHERE l_comment IN ('cmt1', 'cmt50', 'cmt96') AND l_linestatus <> 'O' GROUP BY l_returnflag",
      Seq(Filter, Agg)
    )
    assert(q12.collect().map(_.getLong(1)).sum > 0)
    checkVectorized("SELECT count(*) FROM lineitem WHERE l_returnflag < l_linestatus", Seq(Filter, Agg))
  }

  test("TPC-H Q7/Q8/Q9 shape: extract(year) as a group key above our Final aggregate") {
    val df = checkVectorized(
      "SELECT extract(year FROM l_shipdate) AS l_year, l_returnflag, sum(l_extendedprice * (1 - l_discount)) AS volume, count(*) AS c FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate <= DATE '1996-12-31' GROUP BY extract(year FROM l_shipdate), l_returnflag",
      Seq(Filter, Agg),
      tolerance = 1e-6
    )
    assert(nodesOf[HashAggregateExec](df).isEmpty)
    assert(df.collect().map(_.getInt(0)).toSet === Set(1995, 1996))
    checkVectorized(
      "SELECT year(l_shipdate) AS y, month(l_shipdate) AS m, count(*) FROM lineitem GROUP BY year(l_shipdate), month(l_shipdate)",
      Seq(Agg)
    )
    checkVectorized(
      "SELECT trunc(l_shipdate, 'quarter') AS q, sum(l_quantity) FROM lineitem WHERE datediff(l_receiptdate, l_shipdate) > 10 GROUP BY trunc(l_shipdate, 'quarter')",
      Seq(Filter, Agg),
      tolerance = 1e-6
    )
    // Q9 / Q14 / Q16 shapes: LIKE with a leading or trailing wildcard over dictionary pages.
    checkVectorized(
      "SELECT l_returnflag, count(*) FROM lineitem WHERE l_comment LIKE 'cmt1%' GROUP BY l_returnflag",
      Seq(Filter, Agg)
    )
    checkVectorized(
      "SELECT count(*) FROM lineitem WHERE l_comment NOT LIKE '%9%' AND l_comment LIKE '%5'",
      Seq(Filter, Agg)
    )
  }

  test("statistical aggregates: stddev, variance, skewness, kurtosis, covar, corr, regr_*") {
    // Compared with Spark at a relative tolerance (never bit equality: the merge order across the
    // shuffle is not ours to fix). Every stage ours, Partial and Final.
    def allOurs(sql: String): Unit = {
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).isEmpty, "every stage should be ours\n" + finalPlan(df).treeString)
    }
    // The family over doubles, integers (cast by the analyzer) and a nullable bigint, grouped and not.
    allOurs(
      "SELECT i % 7 AS g, stddev(d2), stddev_samp(i), stddev_pop(l), std(d2), variance(d2), var_samp(l), var_pop(i) FROM t GROUP BY i % 7"
    )
    allOurs(
      "SELECT stddev(d2), stddev_pop(i), variance(l), var_pop(d2), skewness(d2), kurtosis(i), skewness(l), kurtosis(l) FROM t"
    )
    allOurs("SELECT i % 11 AS g, skewness(d2), kurtosis(d2), skewness(l), kurtosis(i) FROM t GROUP BY i % 11")
    // A large mean with a small spread: the streaming update keeps the digits a sum of squares would lose.
    allOurs("SELECT i % 5 AS g, stddev_samp(d2 + 1e9), var_pop(d2 + 1e9), stddev_pop(l + 1e12) FROM t GROUP BY i % 5")
    allOurs("SELECT var_samp(d2 + 1e9), stddev(i + 1e9), skewness(d2 + 1e6), kurtosis(d2 + 1e6) FROM t")
    // NaN and infinities in d propagate as NaN, as in Spark.
    allOurs("SELECT stddev(d), variance(d), skewness(d), kurtosis(d) FROM t")
    // Groups of one row (stddev_samp / var_samp / covar_samp / corr null, the population forms 0), constant columns
    // (variance 0, skewness / kurtosis null on m2 = 0, corr NaN), no rows (null), and a filtered ungrouped shape.
    allOurs(
      "SELECT i AS g, stddev_samp(d2), stddev_pop(d2), var_samp(d2), var_pop(d2), covar_samp(d2, i), covar_pop(d2, i), corr(d2, i) FROM t WHERE i < 50 GROUP BY i"
    )
    allOurs(
      "SELECT i % 4 AS g, stddev_samp(i div 20000), var_pop(i div 20000), skewness(i div 20000), kurtosis(i div 20000), covar_pop(i div 20000, d2) FROM t GROUP BY i % 4"
    )
    // corr over a constant column divides by sqrt(0): Spark's own evaluation raises DIVIDE_BY_ZERO under ANSI, and so does ours.
    Seq(false, true).foreach { on =>
      val e = intercept[Exception](
        withPlugin(enabled = on)(spark.sql("SELECT corr(i div 20000, d2) FROM t")).collect()
      ).asInstanceOf[_root_.org.apache.spark.SparkThrowable]
      assert(e.getCondition == "DIVIDE_BY_ZERO", s"plugin=$on: $e")
    }
    allOurs("SELECT stddev(d2), var_samp(l), covar_samp(d2, l), corr(d2, l), skewness(d2) FROM t WHERE i < 0")
    allOurs("SELECT stddev_samp(d2), var_pop(l), corr(i, l) FROM t WHERE i > 19990")
    // Covariance and correlation: nulls in either argument skip the pair; correlated, anti-correlated and independent shapes.
    allOurs(
      "SELECT i % 7 AS g, covar_samp(d2, i), covar_pop(l, d2), corr(i, d2), corr(l, i), corr(d2, -d2), covar_samp(i, 3.0 * i) FROM t GROUP BY i % 7"
    )
    allOurs(
      "SELECT covar_samp(d2, l), covar_pop(i, l), corr(d2, l), corr(i, i % 13), covar_pop(d2 + 1e9, l + 1e9) FROM t"
    )
    // The regr_* family rewrites onto counts, averages and these moments.
    allOurs(
      "SELECT i % 7 AS g, regr_count(d2, l), regr_avgx(d2, l), regr_avgy(d2, l), regr_sxx(d2, l), regr_syy(d2, l), regr_sxy(d2, l), regr_slope(d2, l), regr_intercept(d2, l), regr_r2(d2, l) FROM t GROUP BY i % 7"
    )
    allOurs(
      "SELECT regr_count(i, d2), regr_slope(i, d2), regr_intercept(i, d2), regr_r2(i, d2), regr_sxy(i, d2) FROM t WHERE i > 100"
    )
    // With FILTER, mixed with the plain aggregates, and the lineitem shape.
    allOurs(
      "SELECT i % 3 AS g, stddev(d2) FILTER (WHERE b), corr(d2, i) FILTER (WHERE l IS NOT NULL), sum(l), avg(d2), count(*) FROM t GROUP BY i % 3"
    )
    allOurs(
      "SELECT l_returnflag, stddev_samp(l_quantity), var_pop(l_extendedprice), corr(l_quantity, l_extendedprice), covar_samp(l_discount, l_tax) FROM lineitem GROUP BY l_returnflag"
    )
  }

  test("try_sum nulls the whole group on overflow; try_avg is avg over doubles") {
    def allOurs(sql: String): Unit = {
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).isEmpty, "every stage should be ours\n" + finalPlan(df).treeString)
    }
    // l is 0..~60000 (null where id % 7 = 3): scaled by 5e13 every value fits a long but any group's
    // sum overflows; unscaled sums never do. Group 0 is poisoned, groups 1 and 2 are exact.
    val mixed = "CASE WHEN i % 3 = 0 THEN l * 50000000000000 ELSE l END"
    for (ansi <- Seq("true", "false")) {
      withConf("spark.sql.ansi.enabled" -> ansi) {
        allOurs(
          s"SELECT i % 3 AS g, try_sum($mixed) AS a, try_sum(l) AS b, try_sum(i) AS c, sum(l) AS d1, count(*) AS n FROM t GROUP BY i % 3"
        )
        allOurs(s"SELECT try_sum($mixed) AS a, try_sum(l) AS b, try_sum(i) AS c FROM t")
        // A group with nulls only is null (isEmpty), a group with one value is that value, a filtered subset agrees.
        allOurs(
          "SELECT i % 7 AS g, try_sum(l) AS a, try_sum(CASE WHEN i % 7 = 3 THEN l END) AS b, try_sum(CASE WHEN i = 5 THEN l * 50000000000000 END) AS c FROM t GROUP BY i % 7"
        )
        allOurs(s"SELECT i % 3 AS g, try_sum($mixed) AS a FROM t WHERE i % 2 = 0 GROUP BY i % 3")
        // Doubles: Spark's Sum in TRY mode has no isEmpty buffer and never overflows; try_avg sums in doubles.
        allOurs(
          "SELECT i % 5 AS g, try_sum(d2) AS a, try_avg(l) AS b, try_avg(i) AS c, try_avg(d2) AS d1, try_sum(d) AS e0 FROM t GROUP BY i % 5"
        )
        allOurs(s"SELECT try_sum(d2) AS a, try_avg($mixed) AS b, try_avg(d) AS c FROM t")
      }
    }
    // Decimals keep their own path.
    checkFallback("SELECT try_sum(CAST(l AS DECIMAL(12, 2))) AS a FROM t", Seq(Agg), "try_sum over a decimal")
  }

  test("literal and typed-null result columns beside the aggregates") {
    def allOurs(sql: String): Unit = {
      val df = checkVectorized(sql, Seq(Agg))
      assert(nodesOf[HashAggregateExec](df).isEmpty, "every stage should be ours\n" + finalPlan(df).treeString)
    }
    // The TPC-DS channel shape: a string literal tagging each grouped aggregate, then a union of them.
    allOurs(
      "SELECT 'store' AS channel, i % 3 AS g, sum(l) AS s, count(*) AS n, CAST(NULL AS BIGINT) AS pad, 1 AS one FROM t GROUP BY i % 3"
    )
    allOurs("SELECT 'all' AS tag, count(*) AS n, sum(d2) AS s, 2.5D AS w, CAST(NULL AS STRING) AS pad FROM t")
    allOurs(
      "SELECT channel, g, sum(s) AS total FROM (SELECT 'store' AS channel, i % 3 AS g, sum(l) AS s FROM t GROUP BY i % 3 UNION ALL SELECT 'web' AS channel, i % 2 AS g, sum(i) AS s FROM t GROUP BY i % 2) u GROUP BY channel, g"
    )
    allOurs(
      "SELECT i % 5 AS g, max(l) AS m, DATE '2020-01-01' AS d0, 'x' AS x FROM t WHERE i > 100 GROUP BY i % 5 HAVING max(l) > 0"
    )
  }

  test("NaN and negative zero grouping keys are normalised as Spark's") {
    // Spark wraps a double key in KnownFloatingPointNormalized(NormalizeNaNAndZero(...)); the compiler
    // runs the normalisation as a real pass, so the group table's bit comparison puts every NaN in one
    // group and both zeros in another, exactly as Spark does.
    checkVectorized(
      "SELECT k, count(*) AS n, sum(i) AS s FROM (SELECT CASE WHEN i % 4 = 0 THEN -0.0 WHEN i % 4 = 1 THEN 0.0 WHEN i % 4 = 2 THEN CAST('NaN' AS DOUBLE) ELSE d2 END AS k, i FROM t) GROUP BY k",
      Seq(Agg)
    )
    // In a comparison the normalisation wrapper does not appear; the compare kernels already treat NaN = NaN and -0.0 = 0.0.
    checkVectorized("SELECT count(*) FROM t WHERE (CASE WHEN i % 2 = 0 THEN -0.0 ELSE d2 END) = 0.0", Seq(Agg))
  }

  test("count over literal arguments after the distinct rewrite; a collated grouping key falls back") {
    // count(DISTINCT 3, 2) becomes count(3, 2) over the Expand: literal arguments never decide anything.
    checkVectorized(
      "SELECT count(DISTINCT 2) AS a, count(DISTINCT 3, 2) AS b, count(DISTINCT i, 5) AS c FROM t",
      Seq(Agg)
    )
    checkVectorized(
      "SELECT s, count(DISTINCT 2) AS a, count(DISTINCT 3, 2) FILTER (WHERE i > 100) AS b FROM t GROUP BY s",
      Seq(Agg)
    )
    checkVectorized("SELECT count(i, 7, l) AS a, count(1, 2) AS b FROM t", Seq(Agg))
    // Kernels compare bytes: a collated string has no lane, and Spark's RowToColumnarExec cannot convert one either.
    checkFallback(
      "SELECT count(*) AS n FROM t GROUP BY s COLLATE UTF8_LCASE",
      Seq(Agg),
      "unsupported column type string collate"
    )
    checkFallback(
      "SELECT s COLLATE UTF8_LCASE AS c, i FROM t WHERE i < 10",
      Seq(classOf[org.apache.spark.sql.vector.VectorProjectExec]),
      "collate"
    )
  }
}
