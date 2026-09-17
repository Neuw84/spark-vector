package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec}
import org.apache.spark.sql.vector.{VectorBroadcastHashJoinExec, VectorBroadcastNestedLoopJoinExec, VectorFilterExec, VectorHashAggregateExec, VectorShuffledHashJoinExec}

/**
 * Hash joins against Spark's. `t` (20k rows) is the streamed side, `dim` a small dimension table
 * joined on ints, longs (with nulls), strings and decimals; some dimension keys have no match and
 * some have several rows so inner, outer, semi and anti joins all exercise their edge.
 */
class VectorJoinSuite extends VectorQuerySuite {

  private val BHJ = classOf[VectorBroadcastHashJoinExec]
  private val SHJ = classOf[VectorShuffledHashJoinExec]
  private val BNLJ = classOf[VectorBroadcastNestedLoopJoinExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("join/t"))
    val dim = newTempPath("join/dim")
    spark
      .range(0, 60)
      .selectExpr(
        "cast(id % 50 as int) as di", // 50 distinct keys, ten of them twice; t.i % 50 style matches
        "if(id % 9 = 0, null, id * 3) as dl", // matches t.l for small ids, nulls never match
        "concat('s', id % 50) as ds", // matches t.s
        "cast(id as decimal(5,1)) as dd",
        "concat('name', id) as name",
        "cast(id as double) * 1.5 as weight")
      .write
      .mode("overwrite")
      .parquet(dim)
    spark.read.parquet(dim).createOrReplaceTempView("dim")
    // t.i is 0..19999; i50 = i % 50 (materialised: `%` is not a compiled expression) so every
    // streamed row has a dimension row.
    val tk = newTempPath("join/tk")
    spark.sql("SELECT *, cast(i % 50 as int) AS i50 FROM t").write.mode("overwrite").parquet(tk)
    spark.read.parquet(tk).createOrReplaceTempView("tk")
  }

  test("broadcast inner join on int, string and decimal keys, with multiple matches") {
    checkVectorized("SELECT tk.i, tk.s, dim.name FROM tk JOIN dim ON tk.i50 = dim.di", Seq(BHJ))
    checkVectorized("SELECT tk.i, dim.name, dim.weight FROM tk JOIN dim ON tk.s = dim.ds", Seq(BHJ))
    checkVectorized("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.l = dim.dl WHERE tk.i < 200", Seq(BHJ, classOf[VectorFilterExec]))
    checkVectorized("SELECT count(*), sum(dim.weight) FROM tk JOIN dim ON tk.i50 = dim.di AND tk.s = dim.ds", Seq(BHJ, classOf[VectorHashAggregateExec]))
    checkVectorized("SELECT tk.i, dim.name FROM tk JOIN dim ON cast(tk.i50 as decimal(5,1)) = dim.dd", Seq(BHJ))
  }

  test("broadcast outer, semi and anti joins keep or drop unmatched streamed rows") {
    checkVectorized("SELECT tk.i, tk.l, dim.name FROM tk LEFT JOIN dim ON tk.l = dim.dl", Seq(BHJ))
    checkVectorized("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.l = dim.dl", Seq(BHJ))
    checkVectorized("SELECT tk.i, tk.s FROM tk LEFT ANTI JOIN dim ON tk.s = dim.ds", Seq(BHJ))
    checkVectorized("SELECT tk.i FROM tk LEFT ANTI JOIN dim ON tk.l = dim.dl", Seq(BHJ))
    // Right outer with the broadcast side on the left.
    checkVectorized("SELECT dim.name, tk.i FROM dim RIGHT JOIN tk ON tk.l = dim.dl WHERE tk.i < 500", Seq(BHJ))
  }

  test("inner join with a non-equi condition filters the joined rows") {
    checkVectorized("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(BHJ))
    checkVectorized("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di AND dim.weight * 2.0 > 20.0", Seq(BHJ))
  }

  // `tk.i50 = dim.di` gives every streamed row one or two candidates (ten dimension keys appear
  // twice); `tk.d > dim.weight` passes for some candidates and fails for others, `dim.dl` is null
  // for every ninth dimension row so a condition on it is null for those candidates, and
  // `tk.l = dim.dl` leaves most streamed rows with no candidate at all.
  test("semi and anti joins with a non-equi condition") {
    checkVectorized("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(BHJ))
    checkVectorized("SELECT tk.i, tk.s FROM tk LEFT ANTI JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(BHJ))
    checkVectorized("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.i50 = dim.di AND tk.l > dim.dl", Seq(BHJ))
    checkVectorized("SELECT tk.i FROM tk LEFT ANTI JOIN dim ON tk.i50 = dim.di AND tk.l > dim.dl", Seq(BHJ))
    checkVectorized("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.l = dim.dl AND dim.weight > 10.0", Seq(BHJ))
    checkVectorized("SELECT tk.i FROM tk LEFT ANTI JOIN dim ON tk.l = dim.dl AND dim.weight > 10.0", Seq(BHJ))
    // The shape EXISTS / NOT EXISTS take after Spark's rewrite.
    checkVectorized("SELECT tk.i FROM tk WHERE EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight < tk.d)", Seq(BHJ))
    checkVectorized("SELECT count(*) FROM tk WHERE NOT EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight < tk.d)", Seq(BHJ, classOf[VectorHashAggregateExec]))
  }

  test("outer joins with a non-equi condition pad the rows that fail it") {
    checkVectorized("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(BHJ))
    checkVectorized("SELECT tk.i, dim.name, dim.weight FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.l > dim.dl", Seq(BHJ))
    checkVectorized("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.l = dim.dl AND dim.weight > 10.0", Seq(BHJ))
    checkVectorized("SELECT dim.name, tk.i FROM dim RIGHT JOIN tk ON tk.i50 = dim.di AND tk.d > dim.weight WHERE tk.i < 500", Seq(BHJ))
    checkVectorized("SELECT count(*), count(dim.name) FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(BHJ, classOf[VectorHashAggregateExec]))
  }

  test("full outer join emits the unmatched rows of both sides") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk FULL OUTER JOIN dim ON tk.i50 = dim.di", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, tk.l, dim.name, dim.dl FROM tk FULL OUTER JOIN dim ON tk.l = dim.dl", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk FULL OUTER JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(SHJ))
      // Build side on the left.
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ dim.name, tk.i FROM dim FULL OUTER JOIN tk ON tk.l = dim.dl AND dim.weight > 10.0", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ count(*), count(tk.i), count(dim.name) FROM tk FULL OUTER JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(SHJ, classOf[VectorHashAggregateExec]))
      // An empty side: AQE eliminates a join whose side is empty outright (it never reaches any
      // operator), so the shape that does reach us is a side reduced to one key -- every shuffle
      // partition but one then has an empty build or probe side and must still emit the other side.
      val emptyBuild = checkVectorized("SELECT /*+ SHUFFLE_HASH(d) */ tk.i, d.name FROM tk FULL OUTER JOIN (SELECT * FROM dim WHERE di = 7) d ON tk.i50 = d.di", Seq(SHJ))
      assert(emptyBuild.filter("name IS NULL").count() === 20000 - 400, "every probe row outside key 7 is unmatched")
      val emptyProbe = checkVectorized("SELECT /*+ SHUFFLE_HASH(d) */ tk.i, d.name FROM (SELECT * FROM tk WHERE i50 = 7 AND i < 1000) tk FULL OUTER JOIN dim d ON tk.i50 = d.di", Seq(SHJ))
      assert(emptyProbe.filter("i IS NULL").count() === 58, "58 dimension rows have no probe row")
      // Shuffle partitions that receive build rows but no probe rows at all still emit their build rows.
      val sparse = checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name, dim.di FROM (SELECT * FROM tk WHERE i50 = 3) tk FULL OUTER JOIN dim ON tk.i50 = dim.di", Seq(SHJ))
      assert(sparse.filter("i IS NULL").count() === 58, "58 of the 60 dimension rows have no probe row")
      // Null keys never match: the null-keyed rows of both sides appear once each, unpaired.
      val nulls = checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.l, dim.dl FROM tk FULL OUTER JOIN dim ON tk.l = dim.dl", Seq(SHJ))
      val nullKeyed = spark.table("tk").filter("l IS NULL").count() + spark.table("dim").filter("dl IS NULL").count()
      assert(nulls.filter("l IS NULL AND dl IS NULL").count() === nullKeyed, "each null-keyed row of either side comes out once, unpaired, with the other side null")
    }
  }

  test("a full outer join over a broadcast is refused, as Spark never plans one") {
    import org.apache.spark.sql.catalyst.plans.FullOuter
    import org.apache.spark.sql.execution.joins.BroadcastHashJoinExec
    import org.apache.spark.sql.vector.VectorJoinPlanner
    // Spark's JoinSelection excludes FullOuter from broadcasting, so build the operator by hand from a
    // real broadcast join and change only the join type.
    val plan = withPlugin(enabled = false)(spark.sql("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di").queryExecution.executedPlan)
    val bhj = org.apache.spark.sql.vector.PlanUtils.allNodes(plan).collect { case b: BroadcastHashJoinExec => b }.head
    val planned = VectorJoinPlanner.plan(bhj.copy(joinType = FullOuter))
    assert(planned.isLeft && planned.left.toOption.get.contains("full outer join over a broadcast"), planned.toString)
  }

  test("existence join: EXISTS used as a value emits every row plus a boolean") {
    import org.apache.spark.sql.catalyst.plans.ExistenceJoin
    def isExistence(df: org.apache.spark.sql.DataFrame): Boolean =
      nodesOf[org.apache.spark.sql.vector.VectorBroadcastHashJoinExec](df).exists(_.joinType.isInstanceOf[ExistenceJoin]) ||
        nodesOf[org.apache.spark.sql.vector.VectorShuffledHashJoinExec](df).exists(_.joinType.isInstanceOf[ExistenceJoin])
    // OR of two EXISTS: Spark plans two ExistenceJoins feeding one filter. Broadcast at this size.
    val or = checkVectorized("SELECT tk.i FROM tk WHERE tk.i < 500 AND (EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight > 60) OR EXISTS (SELECT 1 FROM dim WHERE dim.dl = tk.l))", Seq(BHJ))
    assert(isExistence(or), finalPlan(or).treeString)
    // CASE WHEN EXISTS as a projected value: the boolean itself is visible, false (not null) where nothing matched,
    // one boolean per streamed row even though di has duplicates on the build side.
    val cw = checkVectorized("SELECT tk.i, CASE WHEN EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight > 60) THEN 'hit' ELSE 'miss' END AS tag FROM tk", Seq(BHJ))
    assert(isExistence(cw), finalPlan(cw).treeString)
    assert(cw.count() === 20000 && cw.filter("tag = 'hit'").count() > 0 && cw.filter("tag = 'miss'").count() > 0)
    // Null keys never match: rows with a null l report false, and appear once.
    val nk = checkVectorized("SELECT tk.i, tk.l, EXISTS (SELECT 1 FROM dim WHERE dim.dl = tk.l) AS e FROM tk WHERE tk.i < 2000", Seq(BHJ))
    assert(isExistence(nk), finalPlan(nk).treeString)
    assert(nk.filter("l IS NULL AND e").count() === 0 && nk.filter("l IS NULL").count() > 0)
    // A non-equi condition alongside the key: evaluated per candidate and ANDed into the match.
    checkVectorized("SELECT count(*), count_if(e) FROM (SELECT EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight > tk.d) AS e FROM tk)", Seq(BHJ, classOf[VectorHashAggregateExec]))
    // The shuffled hash join plans it too.
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1", "spark.sql.join.preferSortMergeJoin" -> "false") {
      val shj = checkVectorized("SELECT tk.i FROM tk WHERE tk.i < 500 AND (EXISTS (SELECT /*+ SHUFFLE_HASH(dim) */ 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight > 60) OR tk.i < 10)", Seq(SHJ))
      assert(isExistence(shj), finalPlan(shj).treeString)
    }
  }

  test("unsupported joins fall back with a reason") {
    checkFallback("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.d = dim.weight", Seq(BHJ), "join key type double")
    checkFallback("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.s LIKE 'x%y%z'", Seq(BHJ), "unsupported expression")
    // A simplified LIKE (StartsWith) in the condition is compiled.
    checkVectorized("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.s LIKE 'x%'", Seq(BHJ))
  }

  test("shuffled hash join over Spark's row shuffle on both sides") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      val df = checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di", Seq(SHJ))
      assert(nodesOf[ShuffledHashJoinExec](df).isEmpty, finalPlan(df).treeString)
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk LEFT JOIN dim ON tk.l = dim.dl", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i FROM tk LEFT SEMI JOIN dim ON tk.s = dim.ds", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ count(*) FROM tk JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(SHJ))
    }
  }

  test("join conversion can be disabled by configuration") {
    withConf(VectorConf.BroadcastHashJoinEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di"); d.collect(); d }
      assert(nodesOf[VectorBroadcastHashJoinExec](df).isEmpty)
      assert(nodesOf[BroadcastHashJoinExec](df).nonEmpty)
    }
  }

  test("NaN and negative zero join keys are normalised as Spark's") {
    // Both sides of a double-keyed join carry -0.0, 0.0 and NaN: Spark wraps the keys in
    // KnownFloatingPointNormalized(NormalizeNaNAndZero(...)), which the compiler unwraps -- but a double
    // join key is refused by type, so the plan falls back with that reason and the rows still agree.
    // Pinned so that a future double-key path inherits the test.
    val sql = "SELECT count(*), sum(a.i), sum(b.i) FROM " +
      "(SELECT i, CASE WHEN i % 3 = 0 THEN -0.0 WHEN i % 3 = 1 THEN CAST('NaN' AS DOUBLE) ELSE d2 END AS k FROM t WHERE i < 300) a JOIN " +
      "(SELECT i, CASE WHEN i % 3 = 0 THEN 0.0 WHEN i % 3 = 1 THEN CAST('NaN' AS DOUBLE) ELSE d2 END AS k FROM t WHERE i < 300) b ON a.k = b.k"
    checkFallback(sql, Seq(BHJ), "join key type double not supported")
    checkFallback(sql.replace("SELECT count(*)", "SELECT /*+ SHUFFLE_HASH(b) */ count(*)"), Seq(SHJ), "join key type double not supported")
  }

  /** A nested loop join over a bounded streamed slice; asserts Spark's operator is gone. */
  private def checkNested(sql: String, extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil): Unit = {
    val df = checkVectorized(sql, BNLJ +: extra)
    assert(nodesOf[org.apache.spark.sql.execution.joins.BroadcastNestedLoopJoinExec](df).isEmpty, finalPlan(df).treeString)
  }

  test("nested loop inner and cross joins: conditions matching none, some and all rows, with a literal") {
    checkNested("SELECT tk.i, dim.di, dim.name FROM tk JOIN dim ON tk.i50 < dim.di WHERE tk.i < 2000")
    checkNested("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 + dim.di > 60 AND dim.weight < 40.0 WHERE tk.i < 2000")
    checkNested("SELECT tk.i, dim.di FROM tk JOIN dim ON tk.i50 > 100 WHERE tk.i < 2000") // none
    checkNested("SELECT tk.i, dim.di FROM tk JOIN dim ON tk.i50 >= 0 AND dim.weight >= 0 WHERE tk.i < 300") // all
    checkNested("SELECT tk.i, tk.s, dim.di, dim.name FROM tk CROSS JOIN dim WHERE tk.i < 500")
    checkNested("SELECT dim.name, tk.i FROM dim JOIN tk ON dim.di * 100 < tk.i WHERE tk.i < 3000") // build side left
    checkNested("SELECT count(*) AS n, sum(dim.weight) AS w FROM tk JOIN dim ON tk.d > dim.weight AND tk.l IS NOT NULL WHERE tk.i < 5000", Seq(classOf[VectorHashAggregateExec]))
  }

  test("nested loop semi, anti and existence joins: non-equi EXISTS") {
    checkNested("SELECT tk.i FROM tk WHERE tk.i < 3000 AND EXISTS (SELECT 1 FROM dim WHERE dim.di > tk.i50 + 40)")
    checkNested("SELECT tk.i, tk.s FROM tk WHERE tk.i < 3000 AND NOT EXISTS (SELECT 1 FROM dim WHERE dim.di > tk.i50 + 40)")
    checkNested("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.i50 < dim.di - 45 WHERE tk.i < 3000")
    checkNested("SELECT tk.i FROM tk LEFT ANTI JOIN dim ON tk.d < dim.weight WHERE tk.i < 3000")
    checkNested("SELECT tk.i, EXISTS (SELECT 1 FROM dim WHERE dim.di > tk.i50 + 40) AS e FROM tk WHERE tk.i < 1000")
  }

  test("nested loop outer joins with the streamed side preserved; the broadcast-preserved forms fall back") {
    checkNested("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 + 30 < dim.di WHERE tk.i < 2000")
    checkNested("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 > 100 WHERE tk.i < 500") // nothing matches: null padded
    checkNested("SELECT dim.name, tk.i FROM dim RIGHT JOIN tk ON tk.i50 + 30 < dim.di WHERE tk.i < 2000")
    checkFallback("SELECT /*+ BROADCAST(tk) */ tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 < dim.di WHERE tk.i < 100", Seq(BNLJ), "preserved side broadcast")
    // A WHERE on one side would turn the full join into a left join; filter beneath it instead.
    checkFallback("SELECT t2.i, dim.name FROM (SELECT * FROM tk WHERE i < 100) t2 FULL JOIN dim ON t2.i50 < dim.di", Seq(BNLJ), "full outer nested loop join")
  }
}
