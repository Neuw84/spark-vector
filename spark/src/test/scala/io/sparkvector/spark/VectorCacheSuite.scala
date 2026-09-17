package io.sparkvector.spark

import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.RowToColumnarExec
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorProjectExec}

/**
 * Cached tables (#55, level 1: consume the cache). `InMemoryTableScanExec` emits columnar batches
 * only when Spark's `DefaultCachedBatchSerializer` agrees to, and it agrees only for a cached
 * relation whose *whole* schema is boolean/byte/short/int/long/float/double -- a string, date,
 * timestamp or decimal column anywhere in the cached relation makes the scan a row scan, whatever
 * the query projects. When it is columnar, the batches (Spark's on-heap vectors) feed our operators
 * through the adapter seam's copy like any other Spark columnar scan.
 */
class VectorCacheSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("cache/t"))
    // Primitive-only: the one shape Spark's default cache serializer emits columnar batches for.
    spark.sql("CACHE TABLE t_num AS SELECT i, l, d, d2, b, i % 7 = 0 AS b2 FROM t")
    spark.sql("SELECT count(*) FROM t_num").collect()
    // The full mixed table (string, date): cached as rows as far as the scan is concerned.
    spark.sql("CACHE TABLE t_cached AS SELECT * FROM t")
    spark.sql("SELECT count(*) FROM t_cached").collect()
  }

  override protected def afterAll(): Unit = {
    try spark.catalog.clearCache() finally super.afterAll()
  }

  private def scanIsInput(df: DataFrame): Unit = {
    val scans = nodesOf[InMemoryTableScanExec](df)
    assert(scans.nonEmpty && scans.forall(_.supportsColumnar), finalPlan(df).treeString)
    assert(nodesOf[RowToColumnarExec](df).forall(!_.child.isInstanceOf[InMemoryTableScanExec]),
      "the cache's batches should be consumed directly\n" + finalPlan(df).treeString)
  }

  test("a primitive-only cached table is a columnar input: our operators sit on the scan, copied once per batch") {
    val copiedBefore = ColumnVectorAdapters.copiedColumns()
    scanIsInput(checkVectorized("SELECT i, l, d FROM t_num WHERE i % 3 = 0 AND l IS NOT NULL", Seq(Filter)))
    scanIsInput(checkVectorized("SELECT i + 1 AS j, d * 2 AS e, d2, b, NOT b2 AS nb2 FROM t_num WHERE i < 5000", Seq(Filter, Project)))
    scanIsInput(checkVectorized("SELECT b, count(*) AS n, sum(l) AS sl, avg(d) AS ad, sum(d2) AS sd, max(i) AS mi FROM t_num GROUP BY b", Seq(Agg)))
    scanIsInput(checkVectorized("SELECT count(*) AS n, count(l) AS nl, count(d) AS nd FROM t_num", Seq(Agg)))
    scanIsInput(checkVectorized("SELECT b2, count(*) AS n FROM t_num WHERE d IS NULL OR l IS NULL GROUP BY b2", Seq(Filter, Agg)))
    assert(ColumnVectorAdapters.copiedColumns() > copiedBefore, "cache vectors are Spark's on-heap vectors, copied once per batch")
  }

  test("df.cache() over primitives, and the cache's own filter pushdown beneath our filter") {
    val df = TestTables.mixedDataFrame(spark, 5000).selectExpr("i", "l", "d", "d2", "b").cache()
    df.count()
    df.createOrReplaceTempView("t_df_cached")
    scanIsInput(checkVectorized("SELECT i, l, d, d2, b FROM t_df_cached WHERE i % 2 = 0", Seq(Filter)))
    scanIsInput(checkVectorized("SELECT d2 + 1 AS d1, NOT b AS nb, l * 2 AS l2 FROM t_df_cached WHERE i < 100 AND d > 1", Seq(Filter, Project)))
    df.unpersist()
  }

  test("a cached relation with a string or date column is a row scan (Spark's serializer), as is the reader switched off") {
    val df = checkFallback("SELECT i, l FROM t_cached WHERE i % 5 = 0", Seq(Filter), "is not columnar")
    assert(nodesOf[InMemoryTableScanExec](df).forall(!_.supportsColumnar), finalPlan(df).treeString)
    // Projecting only primitives does not help: the serializer decides on the cached relation's schema.
    checkFallback("SELECT i + 1 AS j FROM t_cached", Seq(Project), "is not columnar")
    // A merging aggregate reads the shuffle, so the Final stage is still ours above a row cache scan.
    checkVectorized("SELECT s, count(*) AS n, sum(l) AS sl FROM t_cached GROUP BY s", Seq(Agg))
    withConf("spark.sql.inMemoryColumnarStorage.enableVectorizedReader" -> "false") {
      val off = checkFallback("SELECT i, l FROM t_num WHERE i % 5 = 0", Seq(Filter), "is not columnar")
      assert(nodesOf[InMemoryTableScanExec](off).forall(!_.supportsColumnar))
    }
  }
}
