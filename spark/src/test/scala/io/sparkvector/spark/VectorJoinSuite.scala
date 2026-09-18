package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec}
import org.apache.spark.sql.vector.{VectorBroadcastHashJoinExec, VectorBroadcastNestedLoopJoinExec, VectorFilterExec, VectorHashAggregateExec, VectorJoinPlanner, VectorShuffledHashJoinExec}

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
    checkFallback("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND soundex(tk.s) = 'X000'", Seq(BHJ), "unsupported expression")
    // A simplified LIKE (StartsWith) in the condition is compiled.
    checkVectorized("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.s LIKE 'x%'", Seq(BHJ))
  }

  test("null-aware anti join (NOT IN over nullable columns): the four regimes (#265)") {
    def nullAware(df: DataFrame): Seq[VectorBroadcastHashJoinExec] =
      nodesOf[VectorBroadcastHashJoinExec](df).filter(_.isNullAwareAntiJoin)
    // Plain regime: a null-free, non-empty build side. tk.l is null at id % 7 = 3 and those rows
    // must go (NULL NOT IN (...) is unknown); the others drop on a match as in a plain anti join.
    // dim.dl has nulls, so the subquery filters them and Spark keeps the null-aware shape only for
    // the streamed side's nullability.
    val plain = checkVectorized("SELECT i, l FROM tk WHERE l NOT IN (SELECT dl FROM dim WHERE dl IS NOT NULL)", Seq(BHJ))
    assert(nullAware(plain).nonEmpty, "expected the null-aware anti join to be ours\n" + finalPlan(plain).treeString)
    assert(plain.collect().forall(!_.isNullAt(1)), "a streamed row with a null key must be dropped")
    // Long and string keys, a null-free build that is nullable only by schema, the q16 shape (a
    // multi-wildcard LIKE inside the subquery) and an aggregate over the result.
    checkVectorized("SELECT i, s FROM tk WHERE s NOT IN (SELECT ds FROM dim WHERE name LIKE 'name%1%')", Seq(BHJ))
    checkVectorized("SELECT i FROM tk WHERE i50 NOT IN (SELECT di FROM dim WHERE weight > 40)", Seq(BHJ))
    checkVectorized("SELECT count(*), count(l) FROM tk WHERE l NOT IN (SELECT dl FROM dim WHERE dl IS NOT NULL)", Seq(BHJ, classOf[VectorHashAggregateExec]))
    // The two singleton regimes. Adaptive execution short-circuits them before any join runs (an
    // empty build side becomes the streamed side, a build side with a null key an empty relation),
    // so the operator itself is exercised with it off; with it on only the results are compared.
    Seq(true, false).foreach { aqe =>
      withConf("spark.sql.adaptive.enabled" -> aqe.toString) {
        val ops = if (aqe) Seq() else Seq(BHJ)
        // Build side holding a null key: NOT IN is never true, no row survives -- null keys or not.
        val allNull = checkVectorized("SELECT i, l FROM tk WHERE l NOT IN (SELECT dl FROM dim)", ops)
        assert(allNull.collect().isEmpty)
        if (!aqe) assert(nullAware(allNull).nonEmpty, finalPlan(allNull).treeString)
        // Empty build side: every streamed row is kept, the null keys too.
        val empty = checkVectorized("SELECT i, l FROM tk WHERE l NOT IN (SELECT dl FROM dim WHERE dl > 1000000)", ops)
        assert(empty.collect().length === spark.table("tk").count())
        if (!aqe) assert(nullAware(empty).nonEmpty, finalPlan(empty).treeString)
        // And the plain regime with adaptive execution off, for completeness.
        checkVectorized("SELECT i, s FROM tk WHERE s NOT IN (SELECT ds FROM dim WHERE di < 10)", Seq(BHJ))
      }
    }
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
    // KnownFloatingPointNormalized(NormalizeNaNAndZero(...)) and the compiler runs the normalisation
    // as a real pass, so the tables' bit comparison sees one NaN and one zero, exactly as Spark does.
    val sql = "SELECT count(*), sum(a.i), sum(b.i) FROM " +
      "(SELECT i, CASE WHEN i % 3 = 0 THEN -0.0 WHEN i % 3 = 1 THEN CAST('NaN' AS DOUBLE) ELSE d2 END AS k FROM t WHERE i < 300) a JOIN " +
      "(SELECT i, CASE WHEN i % 3 = 0 THEN 0.0 WHEN i % 3 = 1 THEN CAST('NaN' AS DOUBLE) ELSE d2 END AS k FROM t WHERE i < 300) b ON a.k = b.k"
    checkVectorized(sql, Seq(BHJ))
    checkVectorized(sql.replace("SELECT count(*)", "SELECT /*+ SHUFFLE_HASH(b) */ count(*)"), Seq(SHJ))
    // Plain double keys straight off the scan, with a matching filter above.
    checkVectorized("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.d = dim.weight", Seq(BHJ))
    checkVectorized("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.d2 = dim.weight WHERE tk.i < 300", Seq(BHJ))
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

  test("a build side estimated above spark.vector.join.maxBuildSize stays with Spark; unknown sizes convert") {
    withConf(VectorConf.JoinMaxBuildSize -> "1") {
      checkFallback("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di", Seq(BHJ), "exceeds spark.vector.join.maxBuildSize=1")
      checkFallback("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 < dim.di WHERE tk.i < 500", Seq(BNLJ), "exceeds spark.vector.join.maxBuildSize=1")
      checkFallback("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di", Seq(SHJ), "exceeds spark.vector.join.maxBuildSize=1")
    }
    // Size strings are accepted, and a generous threshold converts as before.
    withConf(VectorConf.JoinMaxBuildSize -> "512m") {
      checkVectorized("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di WHERE tk.i < 300", Seq(BHJ))
    }
    // The default is a per-core share of the off-heap budget, or 1 GiB without one.
    assert(VectorConf.joinMaxBuildSize(spark.sessionState.conf, new org.apache.spark.SparkConf(false)) === (1L << 30))
    val offHeap = new org.apache.spark.SparkConf(false).set("spark.memory.offHeap.enabled", "true").set("spark.memory.offHeap.size", "4g").set("spark.executor.cores", "4")
    assert(VectorConf.joinMaxBuildSize(spark.sessionState.conf, offHeap) === (1L << 30))
    // No logical link, no statistics: unknown, which converts rather than refuses.
    val orphan = org.apache.spark.sql.execution.LocalTableScanExec(Nil, Nil, None)
    assert(VectorJoinPlanner.estimatedBuildSize(orphan).isEmpty)
    assert(VectorJoinPlanner.buildSizeReason(orphan, 1L).isEmpty)
  }

  test("a cross join whose build side is pruned to no columns still pairs every row") {
    checkNested("SELECT tk.i, tk.s FROM tk CROSS JOIN dim WHERE tk.i < 100")
    checkNested("SELECT count(*) AS n FROM tk CROSS JOIN dim WHERE tk.i < 500", Seq(classOf[VectorHashAggregateExec]))
    checkNested("SELECT dim.name FROM tk JOIN dim ON tk.i < 5")
  }

  // Spark's default for large equi joins: no broadcast, sort-merge preferred, and our rewrite opted in.
  private val SortMerge = Seq(
    "spark.sql.autoBroadcastJoinThreshold" -> "-1",
    "spark.sql.join.preferSortMergeJoin" -> "true",
    "spark.vector.exec.sortMergeJoin.enabled" -> "true")

  private def checkSortMerge(sql: String, extra: Seq[Class[_ <: org.apache.spark.sql.execution.SparkPlan]] = Nil): org.apache.spark.sql.DataFrame = {
    // Spark plans the sort-merge join for this query, and re-expressed as our hash join neither the join nor
    // the two sorts placed for the merge survive.
    val planned = withPlugin(enabled = false) { val d = spark.sql(sql); d.collect(); d }
    assert(nodesOf[org.apache.spark.sql.execution.joins.SortMergeJoinExec](planned).nonEmpty, finalPlan(planned).treeString)
    val df = checkVectorized(sql, SHJ +: extra)
    assert(nodesOf[org.apache.spark.sql.execution.joins.SortMergeJoinExec](df).isEmpty, finalPlan(df).treeString)
    assert(nodesOf[org.apache.spark.sql.execution.SortExec](df).isEmpty && nodesOf[org.apache.spark.sql.vector.VectorSortExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  test("sort-merge joins re-expressed as the shuffled hash join: every join type, duplicate and null keys, conditions") {
    withConf(SortMerge: _*) {
      checkSortMerge("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di")
      checkSortMerge("SELECT tk.i, tk.l, dim.name FROM tk LEFT JOIN dim ON tk.l = dim.dl")
      checkSortMerge("SELECT tk.i, dim.name FROM tk RIGHT JOIN dim ON tk.i50 = dim.di")
      checkSortMerge("SELECT tk.i, dim.name, dim.di FROM tk FULL OUTER JOIN dim ON tk.i50 = dim.di")
      checkSortMerge("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.s = dim.ds")
      checkSortMerge("SELECT tk.i FROM tk LEFT ANTI JOIN dim ON tk.i50 = dim.di AND dim.weight > tk.d")
      checkSortMerge("SELECT count(*), count(dim.name) FROM tk JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(classOf[VectorHashAggregateExec]))
      val existence = checkSortMerge("SELECT tk.i, EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.weight > 60) AS e FROM tk WHERE tk.i < 3000")
      def isExistence(df: org.apache.spark.sql.DataFrame): Boolean =
      nodesOf[org.apache.spark.sql.vector.VectorBroadcastHashJoinExec](df).exists(_.joinType.isInstanceOf[org.apache.spark.sql.catalyst.plans.ExistenceJoin]) ||
        nodesOf[org.apache.spark.sql.vector.VectorShuffledHashJoinExec](df).exists(_.joinType.isInstanceOf[org.apache.spark.sql.catalyst.plans.ExistenceJoin])
      assert(isExistence(existence), finalPlan(existence).treeString)
      // Null keys pair with nothing, on both sides of an outer join (rows compared with Spark's above).
      val nulls = checkSortMerge("SELECT tk.l, dim.dl FROM tk FULL OUTER JOIN dim ON tk.l = dim.dl")
      assert(nulls.filter("l IS NULL").count() > 0 && nulls.filter("dl IS NULL").count() > 0 && nulls.filter("l = dl").count() > 0)
      // Two joins on different keys: a shuffle sits between them, so each converts on its own.
      checkSortMerge("SELECT tk.i, a.name, b.name FROM tk JOIN dim a ON tk.i50 = a.di JOIN dim b ON tk.l = b.dl WHERE tk.i < 4000")
      // A chain on the same key with no shuffle between the joins: the upper join reads the lower one's
      // output directly, columnar once the lower converts, so the whole chain converts (TPC-DS q10, q35,
      // q69, q95 chain semi and existence joins on the customer key).
      checkSortMerge("SELECT tk.i, a.name, b.name FROM tk JOIN dim a ON tk.i50 = a.di JOIN dim b ON tk.i50 = b.di WHERE tk.i < 2000")
      checkSortMerge("SELECT tk.i FROM tk WHERE tk.i50 IN (SELECT di FROM dim WHERE weight > 10) AND EXISTS (SELECT 1 FROM dim WHERE dim.di = tk.i50 AND dim.name > 'n') AND tk.i < 3000")
      val three = checkSortMerge("SELECT tk.i, a.name, b.name, c.name FROM tk JOIN dim a ON tk.i50 = a.di JOIN dim b ON tk.i50 = b.di JOIN dim c ON tk.i50 = c.di WHERE tk.i < 1000")
      assert(nodesOf[org.apache.spark.sql.vector.VectorShuffledHashJoinExec](three).length === 3, finalPlan(three).treeString)
    }
  }

  test("a sort-merge join stays Spark's when its ordering is relied on, its statistics are missing or too large, or by default") {
    // Off by default: the rewrite changes the memory profile of Spark's default join, so the maintainer opts in.
    withConf(SortMerge.take(2): _*) {
      val df = withPlugin(enabled = true) { val d = spark.sql("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di"); d.collect(); d }
      assert(nodesOf[org.apache.spark.sql.execution.joins.SortMergeJoinExec](df).nonEmpty, finalPlan(df).treeString)
    }
    withConf(SortMerge: _*) {
      // A window partitioned by the join key is planned directly over the merge join, on its ordering.
      checkFallback(
        "SELECT tk.i, count(*) OVER (PARTITION BY tk.i50) AS c FROM tk JOIN dim ON tk.i50 = dim.di WHERE tk.i < 2000",
        Seq(SHJ), "output ordering required by the parent operator")
      // A chain of merge joins on the same key with no shuffle between them, under a window on that key:
      // the window relies on the top join's ordering, so it stays, and the join below it -- whose
      // ordering the top one reads -- stays with it.
      val chained = checkFallback(
        "SELECT tk.i, a.name, b.name, count(*) OVER (PARTITION BY tk.i50) AS c FROM tk JOIN dim a ON tk.i50 = a.di JOIN dim b ON tk.i50 = b.di WHERE tk.i < 2000",
        Seq(SHJ), "output ordering required by the parent operator")
      assert(nodesOf[org.apache.spark.sql.execution.joins.SortMergeJoinExec](chained).length === 2, finalPlan(chained).treeString)
    }
    withConf(SortMerge: _*) {
      // The pre-pass judges inputs optimistically over Spark's plan; here the upper join's right input is an
      // aggregate that will not convert (approx_count_distinct), so the upper join stays after all while the
      // lower one, freed by the pre-pass, converted: the ordering the lower one no longer offers is restored
      // by a sort over our hash join, and Spark's merge join reads sorted input.
      val healed = checkVectorized(
        "SELECT tk.i, a.name, x.c FROM tk JOIN dim a ON tk.i50 = a.di JOIN (SELECT di, approx_count_distinct(name) AS c FROM dim GROUP BY di) x ON tk.i50 = x.di WHERE tk.i < 2000",
        Seq(SHJ))
      val smj = nodesOf[org.apache.spark.sql.execution.joins.SortMergeJoinExec](healed)
      assert(smj.length === 1, finalPlan(healed).treeString)
      val resort = nodesOf[org.apache.spark.sql.vector.VectorSortExec](healed).filter(sort => !sort.global && sort.child.collectFirst { case j: org.apache.spark.sql.vector.VectorShuffledHashJoinExec => j }.isDefined)
      assert(resort.nonEmpty, finalPlan(healed).treeString)
    }
    withConf((SortMerge :+ ("spark.vector.join.maxBuildSize" -> "1")): _*) {
      checkFallback("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di", Seq(SHJ), "exceeds spark.vector.join.maxBuildSize=1")
    }
    withConf((SortMerge :+ ("spark.sql.adaptive.enabled" -> "false")): _*) {
      // Without adaptive execution the shuffles Spark inserts carry no statistics: no assumption is made.
      checkFallback("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.i50 = dim.di", Seq(SHJ), "no size statistics")
    }
  }

  test("a broadcast join whose streamed side is a bare shuffle read is ours (AQE's runtime broadcast conversion)") {
    import org.apache.spark.sql.execution.RowToColumnarExec
    import org.apache.spark.sql.execution.adaptive.AQEShuffleReadExec
    // No broadcast at planning time (a shuffled join is planned), but adaptive execution sees the dimension's
    // real size and re-plans the join as a broadcast join at runtime: the streamed side is then the bare
    // shuffle read of the fact table, which Spark converts below us with RowToColumnarExec as it does for
    // the shuffled hash join. Eleven TPC-DS queries at SF1 have this shape.
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1", "spark.sql.adaptive.autoBroadcastJoinThreshold" -> "10MB") {
      val df = checkVectorized("SELECT tk.i, tk.l, dim.name FROM tk JOIN dim ON tk.i50 = dim.di WHERE tk.i < 4000", Seq(BHJ))
      val bhj = nodesOf[org.apache.spark.sql.vector.VectorBroadcastHashJoinExec](df)
      assert(bhj.nonEmpty && nodesOf[org.apache.spark.sql.execution.joins.BroadcastHashJoinExec](df).isEmpty, finalPlan(df).treeString)
      val streamed = if (bhj.head.buildSide == org.apache.spark.sql.catalyst.optimizer.BuildRight) bhj.head.left else bhj.head.right
      assert(streamed.isInstanceOf[RowToColumnarExec] && streamed.children.head.isInstanceOf[AQEShuffleReadExec], finalPlan(df).treeString)
      // Everything above the join stays columnar and ours: the aggregate no longer falls back on "child ... is not columnar".
      checkVectorized("SELECT dim.name, count(*) AS n, sum(tk.l) AS s FROM tk JOIN dim ON tk.i50 = dim.di GROUP BY dim.name", Seq(BHJ, classOf[VectorHashAggregateExec]))
      checkVectorized("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.l = dim.dl AND dim.weight > tk.d WHERE tk.i < 3000", Seq(BHJ))
      checkVectorized("SELECT tk.i FROM tk LEFT SEMI JOIN dim ON tk.i50 = dim.di", Seq(BHJ))
    }
  }

  test("payload columns without a lane (struct, array, map) pass through the joins on the streamed side (#273)") {
    // A copy of the streamed table with nested payloads beside plain and dictionary strings; the struct
    // (Iceberg's `_partition` shape) has nulls at two levels.
    val nk = newTempPath("join/nk")
    spark.sql(
      "SELECT i, l, s, i50, if(i % 11 = 4, null, named_struct('a', i, 'b', s, 'c', if(i % 5 = 0, null, named_struct('d', l, 'e', d)))) AS st, " +
        "if(i % 13 = 6, null, array(i, i + 1, l)) AS arr, if(i % 17 = 2, null, map(coalesce(s, 'none'), l)) AS mp FROM tk WHERE i < 6000")
      .repartition(2).write.mode("overwrite").parquet(nk)
    spark.read.parquet(nk).createOrReplaceTempView("nk")
    for (hint <- Seq("BROADCAST(dim)", "SHUFFLE_HASH(dim)")) {
      val join = if (hint.startsWith("BROADCAST")) BHJ else SHJ
      // Inner and left outer: the struct rides with its row; the outer's unmatched rows keep their payloads.
      checkVectorized(s"SELECT /*+ $hint */ nk.i, nk.st, nk.arr, dim.name FROM nk JOIN dim ON nk.i50 = dim.di", Seq(join))
      checkVectorized(s"SELECT /*+ $hint */ nk.i, nk.st, nk.mp, dim.name FROM nk LEFT JOIN dim ON nk.i50 = dim.di AND dim.weight > 30.0", Seq(join))
      // With a condition (evaluated on the joined batch, which carries the pass-through column too) and a struct field read above.
      checkVectorized(s"SELECT /*+ $hint */ nk.i, nk.st.a AS a, nk.st.c.d AS d, dim.name FROM nk JOIN dim ON nk.i50 = dim.di AND nk.l > dim.dl", Seq(join))
      // Semi and anti never output the payload but the streamed batch still carries it.
      checkVectorized(s"SELECT /*+ $hint */ nk.i, nk.st FROM nk LEFT SEMI JOIN dim ON nk.i50 = dim.di AND dim.di < 30", Seq(join))
      checkVectorized(s"SELECT /*+ $hint */ nk.i, nk.arr FROM nk LEFT ANTI JOIN dim ON nk.i50 = dim.di AND dim.di < 30", Seq(join))
    }
    // A right outer join with the build on the left: the streamed (preserved) side is the right table here,
    // so its payload passes through and the unmatched build rows pad the streamed struct with nulls.
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ dim.name, nk.i, nk.st FROM dim RIGHT JOIN nk ON dim.di = nk.i50 AND dim.di < 40", Seq(SHJ))
    // A full outer join: pass-through rows and null-padded rows in the same output.
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ dim.name, nk.i, nk.st FROM dim FULL OUTER JOIN nk ON dim.di = nk.i50 AND nk.i < 3000", Seq(SHJ))
    // The build side is laid out in lanes: a struct there is still refused, with its reason.
    checkFallback("SELECT /*+ BROADCAST(nk) */ nk.i, nk.st, dim.name FROM dim JOIN nk ON dim.di = nk.i50 WHERE dim.di < 10", Seq(BHJ), "unsupported column type struct")
    checkFallback("SELECT /*+ SHUFFLE_HASH(nk) */ nk.i, nk.st, dim.name FROM dim JOIN nk ON dim.di = nk.i50 WHERE dim.di < 10", Seq(SHJ), "unsupported column type struct")
  }

  test("outer joins that preserve the build side in the shuffled hash join (#273)") {
    // RIGHT JOIN built from the right (preserved) side and LEFT JOIN built from the left: the matched
    // bitmap of the full outer join emits the unmatched build rows once, with null streamed columns.
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk RIGHT JOIN dim ON tk.i50 = dim.di AND tk.i < 500", Seq(SHJ))
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ dim.name, tk.i FROM dim LEFT JOIN tk ON dim.di = tk.i50 AND tk.i < 500", Seq(SHJ))
    // With a condition on the pairs (duplicate build keys: ten dimension keys appear twice), and with none.
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name, dim.weight FROM tk RIGHT JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight WHERE tk.i IS NULL OR tk.i < 3000", Seq(SHJ))
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ dim.name, tk.i FROM dim LEFT JOIN tk ON dim.dl = tk.l", Seq(SHJ))
    // Null build keys never match and come out null-padded; a build side almost nobody matches comes out (nearly) whole.
    // (An empty streamed side would let AQE rewrite the join to a projection of nulls, so one streamed row stays.)
    checkVectorized("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.dl FROM tk RIGHT JOIN dim ON tk.l = dim.dl AND tk.i < 40", Seq(SHJ))
  }
}
