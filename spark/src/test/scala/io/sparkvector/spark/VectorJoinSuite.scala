package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec}
import org.apache.spark.sql.vector.{VectorBroadcastHashJoinExec, VectorFilterExec, VectorHashAggregateExec, VectorShuffledHashJoinExec}

/**
 * Hash joins against Spark's. `t` (20k rows) is the streamed side, `dim` a small dimension table
 * joined on ints, longs (with nulls), strings and decimals; some dimension keys have no match and
 * some have several rows so inner, outer, semi and anti joins all exercise their edge.
 */
class VectorJoinSuite extends VectorQuerySuite {

  private val BHJ = classOf[VectorBroadcastHashJoinExec]
  private val SHJ = classOf[VectorShuffledHashJoinExec]

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
}
