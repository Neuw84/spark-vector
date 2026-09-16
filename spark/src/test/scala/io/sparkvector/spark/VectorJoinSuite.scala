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

  test("unsupported joins fall back with a reason") {
    checkFallback("SELECT tk.i, dim.name FROM tk LEFT JOIN dim ON tk.i50 = dim.di AND tk.d > dim.weight", Seq(BHJ), "non-equi condition")
    checkFallback("SELECT tk.i, dim.name FROM tk JOIN dim ON tk.d = dim.weight", Seq(BHJ), "join key type double")
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      checkFallback("SELECT /*+ SHUFFLE_HASH(dim) */ tk.i, dim.name FROM tk FULL OUTER JOIN dim ON tk.i50 = dim.di", Seq(BHJ, SHJ), "join type")
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
}
