package io.sparkvector.spark

import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.joins.SortMergeJoinExec
import org.apache.spark.sql.vector.{VectorShuffledHashJoinExec, VectorSortMergeJoinExec}

/**
 * The merge join (#286) against Spark's sort-merge join, row order included: both are
 * order-preserving, so the comparison is positional -- unlike the hash join suites, which compare
 * row sets.
 */
class VectorSortMergeJoinSuite extends VectorQuerySuite {

  private val SMJ = classOf[VectorSortMergeJoinExec]

  /** Spark plans a sort-merge join, and we take it as our merge join. */
  private val merge = Seq(
    "spark.sql.autoBroadcastJoinThreshold" -> "-1",
    "spark.sql.join.preferSortMergeJoin" -> "true",
    VectorConf.SortMergeJoinMode -> "merge")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    val a = newTempPath("smj/a")
    spark
      .range(0, 3000)
      .selectExpr(
        "if(id % 37 = 0, null, cast(id % 300 as int)) as ki", // ~10 rows per key, nulls
        "if(id % 41 = 0, null, cast(id % 300 as bigint) * 3000000000) as kl",
        "case when id % 53 = 0 then cast('NaN' as double) when id % 300 = 7 then -0.0 when id % 59 = 0 then null else cast(id % 300 as double) / 4 end as kd",
        "if(id % 43 = 0, null, concat('k', id % 300)) as ks",
        "if(id % 47 = 0, null, cast(id % 300 as decimal(12,2)) / 8) as kdec",
        "date_add(date '2021-01-01', cast(id % 300 as int)) as kdt",
        "cast(id % 300 as decimal(27,2)) * 100000000000 as kw",
        "cast(id as int) as v",
        "concat('left', id) as name")
      .write
      .mode("overwrite")
      .parquet(a)
    spark.read.parquet(a).createOrReplaceTempView("a")
    val b = newTempPath("smj/b")
    spark
      .range(0, 1200)
      .selectExpr(
        "if(id % 23 = 0, null, cast(id % 400 as int)) as ki", // keys 300..399 only here; ~3 rows per key
        "if(id % 29 = 0, null, cast(id % 400 as bigint) * 3000000000) as kl",
        "case when id % 31 = 0 then cast('NaN' as double) when id % 400 = 7 then 0.0 when id % 61 = 0 then null else cast(id % 400 as double) / 4 end as kd",
        "if(id % 19 = 0, null, concat('k', id % 400)) as ks",
        "if(id % 17 = 0, null, cast(id % 400 as decimal(12,2)) / 8) as kdec",
        "date_add(date '2021-01-01', cast(id % 400 as int)) as kdt",
        "cast(id % 400 as decimal(27,2)) * 100000000000 as kw",
        "cast(id as int) as w",
        "concat('right', id) as tag")
      .write
      .mode("overwrite")
      .parquet(b)
    spark.read.parquet(b).createOrReplaceTempView("b")
  }

  /** Rows equal to Spark's in order, and the plan carries our merge join (and nothing of the hash rewrite). */
  private def checkOrdered(sql: String): Unit = withConf(merge: _*) {
    val expected = withPlugin(false) { spark.sql(sql).collect() }
    val df = checkVectorized(sql, Seq(SMJ))
    assert(nodesOf[VectorShuffledHashJoinExec](df).isEmpty, s"hash rewrite took the join for: $sql")
    assert(nodesOf[SortMergeJoinExec](df).isEmpty, s"Spark's merge join remains for: $sql")
    val actual = withPlugin(true) { spark.sql(sql).collect() }
    assert(actual.length === expected.length, s"row count for: $sql")
    var i = 0
    while (i < expected.length) {
      assert(same(expected(i), actual(i)), s"row $i differs for: $sql\n  spark: ${expected(i)}\n  ours:  ${actual(i)}")
      i += 1
    }
  }

  private def same(e: Row, a: Row): Boolean =
    e.length == a.length && (0 until e.length).forall { c =>
      (e.get(c), a.get(c)) match {
        case (x: Double, y: Double) => java.lang.Double.compare(x, y) == 0 || (x.isNaN && y.isNaN)
        case (x, y) => x == y
      }
    }

  test("inner join on every key lane, Spark's order") {
    checkOrdered("SELECT a.ki, a.v, b.w FROM a JOIN b ON a.ki = b.ki")
    checkOrdered("SELECT a.kl, a.v, b.w FROM a JOIN b ON a.kl = b.kl")
    checkOrdered("SELECT a.kd, a.v, b.w FROM a JOIN b ON a.kd = b.kd") // NaN = NaN, -0.0 = 0.0
    checkOrdered("SELECT a.ks, a.v, b.w FROM a JOIN b ON a.ks = b.ks") // dictionary strings
    checkOrdered("SELECT a.kdec, a.v, b.w FROM a JOIN b ON a.kdec = b.kdec")
    checkOrdered("SELECT a.kdt, a.v, b.w FROM a JOIN b ON a.kdt = b.kdt")
    checkOrdered("SELECT a.kw, a.v, b.w FROM a JOIN b ON a.kw = b.kw") // decimal(27,2): the 128-bit lane
    checkOrdered("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki AND a.ks = b.ks") // two keys
  }

  test("every join type, with and without a condition") {
    for (jt <- Seq("INNER", "LEFT OUTER", "RIGHT OUTER", "FULL OUTER", "LEFT SEMI", "LEFT ANTI")) {
      checkOrdered(s"SELECT * FROM a $jt JOIN b ON a.ki = b.ki")
      checkOrdered(s"SELECT * FROM a $jt JOIN b ON a.ki = b.ki AND a.v % 3 < b.w % 5")
    }
    // Existence: EXISTS used as a value.
    checkOrdered("SELECT a.v, EXISTS (SELECT 1 FROM b WHERE b.ki = a.ki) AS e FROM a")
    checkOrdered("SELECT a.v, EXISTS (SELECT 1 FROM b WHERE b.ki = a.ki AND b.w > a.v) AS e FROM a")
  }

  test("runs spanning batch boundaries: 64-row batches") {
    withConf("spark.sql.inMemoryColumnarStorage.batchSize" -> "64") {
      checkOrdered("SELECT a.ki, a.v, b.w FROM a JOIN b ON a.ki = b.ki")
      checkOrdered("SELECT * FROM a FULL OUTER JOIN b ON a.ki = b.ki")
      checkOrdered("SELECT * FROM a LEFT ANTI JOIN b ON a.kl = b.kl")
    }
  }

  test("null keys never match, an empty side, a skewed key") {
    checkOrdered("SELECT a.ki, b.ki FROM a FULL OUTER JOIN b ON a.ki = b.ki WHERE a.ki IS NULL OR b.ki IS NULL")
    // An empty side: with adaptive execution Spark optimises the join away at runtime, so plan it statically.
    withConf("spark.sql.adaptive.enabled" -> "false") {
      checkOrdered("SELECT * FROM a LEFT OUTER JOIN (SELECT * FROM b WHERE w < -1) b2 ON a.ki = b2.ki")
      checkOrdered("SELECT * FROM (SELECT * FROM a WHERE v < -1) a2 RIGHT OUTER JOIN b ON a2.ki = b.ki")
    }
    // One key on both sides: the cross product 2000 x 800 in chunks, Spark's order.
    checkOrdered("SELECT a.v, b.w FROM (SELECT pmod(v, 1) AS k, v FROM a WHERE v < 2000) a JOIN (SELECT pmod(w, 1) AS k, w FROM b WHERE w < 800) b ON a.k = b.k")
  }

  test("a parent relying on the join's ordering converts: a window on the key, a same-key chain") {
    withConf(merge: _*) {
      // The window's partition is the join key: it needs the join's ordering and gets it, no sort between.
      val w = checkVectorized(
        "SELECT a.ki, a.v, b.w, row_number() OVER (PARTITION BY a.ki ORDER BY a.ki) AS rn FROM a JOIN b ON a.ki = b.ki",
        Seq(SMJ, classOf[org.apache.spark.sql.vector.VectorWindowExec]))
      assert(nodesOf[org.apache.spark.sql.execution.SortExec](w).forall(s => !s.child.isInstanceOf[VectorSortMergeJoinExec]), "a sort was placed over the merge join")
      assert(nodesOf[org.apache.spark.sql.vector.VectorSortExec](w).forall(s => !s.child.isInstanceOf[VectorSortMergeJoinExec]), "our sort was placed over the merge join")
      // A chain on the same key: the second join reads the first's ordering.
      val c = checkVectorized(
        "SELECT a.v, b.w, c.w AS w2 FROM a JOIN b ON a.ki = b.ki JOIN (SELECT ki, w FROM b WHERE w % 2 = 0) c ON a.ki = c.ki",
        Seq(SMJ))
      assert(nodesOf[VectorSortMergeJoinExec](c).length === 2, "both joins of the chain")
      assert(nodesOf[SortMergeJoinExec](c).isEmpty)
    }
  }

  test("auto: the hash rewrite where a side's statistics fit, the merge join where the order can show or statistics are missing") {
    val auto = Seq("spark.sql.autoBroadcastJoinThreshold" -> "-1", "spark.sql.join.preferSortMergeJoin" -> "true", VectorConf.SortMergeJoinMode -> "auto")
    def why(df: org.apache.spark.sql.DataFrame): String =
      (nodesOf[VectorShuffledHashJoinExec](df) ++ nodesOf[VectorSortMergeJoinExec](df)).flatMap(_.getTagValue(org.apache.spark.sql.vector.VectorExecRule.SortMergeWhy)).mkString("; ")
    withConf(auto: _*) {
      // A small side with AQE statistics: the hash rewrite, the reason names the side and the budget.
      val h = checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki", Seq(classOf[VectorShuffledHashJoinExec]))
      assert(why(h).contains("as hash join"), why(h))
      // A LIMIT directly above: the rows it picks would differ under the hash rewrite, so the merge join.
      val l = checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki LIMIT 20", Seq(SMJ))
      assert(why(l).contains("reaches a limit or a sort"), why(l))
      // A global ORDER BY above (a range-partitioned exchange): ties would show, so the merge join.
      val o = checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki ORDER BY a.ki", Seq(SMJ))
      assert(why(o).contains("reaches a limit or a sort"), why(o))
      // A parent relying on the ordering: the merge join.
      val w = checkVectorized("SELECT a.ki, a.v, row_number() OVER (PARTITION BY a.ki ORDER BY a.ki) AS rn FROM a JOIN b ON a.ki = b.ki", Seq(SMJ))
      assert(why(w).contains("as merge join"), why(w)) // relied on by the parent, or through the sort Spark placed for the window
      // An aggregate above ends the visibility: the join's order cannot show through a GROUP BY.
      checkVectorized("SELECT a.ki, count(*) AS n FROM a JOIN b ON a.ki = b.ki GROUP BY a.ki ORDER BY a.ki", Seq(classOf[VectorShuffledHashJoinExec]))
    }
    // No statistics (adaptive execution off): the hash rewrite needs them, our merge join does not
    // (#416: the sort below it spills, so no input size is too large).
    withConf((auto :+ ("spark.sql.adaptive.enabled" -> "false")): _*) {
      checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki", Seq(SMJ))
    }
    // A tiny sort budget: the merge join's inputs spill and the join still answers.
    withConf((auto :+ (VectorConf.SortSpillBytes -> "1") :+ (VectorConf.SortRunRows -> "64")): _*) {
      checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki LIMIT 20", Seq(SMJ))
      checkVectorized("SELECT a.ki, a.v, b.w FROM a FULL OUTER JOIN b ON a.ki = b.ki ORDER BY a.ki, a.v, b.w", Seq(SMJ))
    }
    // The boolean flag reads as auto.
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1", VectorConf.SortMergeJoinEnabled -> "true") {
      checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki LIMIT 20", Seq(SMJ))
    }
  }

  test("the join's input estimate reads the stage that has run, not a sort's product estimate (#329)") {
    import org.apache.spark.sql.execution.adaptive.ShuffleQueryStageExec
    import org.apache.spark.sql.vector.VectorJoinPlanner
    // The q1/q30/q81 shape: an aggregate self-joined against its own average, a sort above the join.
    // The logical estimate of the join's inputs is derived from a join; the stages below have run.
    val ctr =
      """WITH t AS (SELECT ki, sum(v) AS total FROM a GROUP BY ki)
        |SELECT t1.ki, t1.total FROM t t1, (SELECT ki, avg(total) * 1.2 AS lim FROM t GROUP BY ki) t2
        |WHERE t1.ki = t2.ki AND t1.total > t2.lim ORDER BY t1.ki LIMIT 50""".stripMargin
    val auto = Seq("spark.sql.autoBroadcastJoinThreshold" -> "-1", "spark.sql.join.preferSortMergeJoin" -> "true", VectorConf.SortMergeJoinMode -> "auto")
    withConf(auto: _*) {
      val df = spark.sql(ctr)
      df.collect()
      val plan = finalPlan(df)
      val joins = nodesOf[VectorSortMergeJoinExec](df).flatMap(j => Seq(j.left, j.right)) ++
        nodesOf[VectorShuffledHashJoinExec](df).flatMap(j => Seq(j.left, j.right)) ++
        nodesOf[SortMergeJoinExec](df).flatMap(j => Seq(j.left, j.right))
      assume(joins.nonEmpty, plan.treeString)
      // Each input of the join estimates as no more than the runtime bytes of the stage below it.
      joins.foreach { child =>
        child.collectFirst { case q: ShuffleQueryStageExec => q }.foreach { q =>
          val runtime = q.computeStats().map(_.sizeInBytes.toLong)
          val est = VectorJoinPlanner.estimatedBuildSize(child)
          assert(est.isDefined && runtime.exists(_ >= est.get), s"estimate $est, stage $runtime\n${plan.treeString}")
        }
      }
      // And the join is ours.
      assert(nodesOf[VectorSortMergeJoinExec](df).nonEmpty || nodesOf[VectorShuffledHashJoinExec](df).nonEmpty, plan.treeString)
    }
  }

  test("the mode switch: off leaves Spark's join, hash takes the rewrite, a struct column falls back") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1", VectorConf.SortMergeJoinMode -> "off") {
      val df = checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki", Seq())
      assert(nodesOf[SortMergeJoinExec](df).nonEmpty)
    }
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1", VectorConf.SortMergeJoinMode -> "hash") {
      checkVectorized("SELECT a.v, b.w FROM a JOIN b ON a.ki = b.ki", Seq(classOf[VectorShuffledHashJoinExec]))
    }
    withConf(merge: _*) {
      checkFallback("SELECT a.v, s.st FROM a JOIN (SELECT ki, struct(w, tag) AS st FROM b) s ON a.ki = s.ki", Seq(SMJ), "unsupported column type struct")
    }
  }
}
