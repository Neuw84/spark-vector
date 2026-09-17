package io.sparkvector.spark

import io.sparkvector.spark.adapter.{ColumnVectorAdapters, SparkColumnVectorBuffers}
import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorProjectExec}

/**
 * What counts as a columnar input (#61): Spark's vectorized Parquet reader, on-heap (copied once per
 * batch) or off-heap (fixed-width columns handed over as views), the ORC vectorized reader through
 * the generic copy, and the two ways a scan stops being one -- the reader switched off, or a column
 * type with no lane.
 */
class VectorScanSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("scan/t"))
    TestTables.mixedDataFrame(spark).repartition(3).write.mode("overwrite").orc(newTempPath("scan/t_orc"))
    spark.read.orc(newTempPath("scan/t_orc")).createOrReplaceTempView("t_orc")
    TestTables.mixedDataFrame(spark).selectExpr("i", "l", "s", "named_struct('a', i, 'b', s) AS st")
      .repartition(2).write.mode("overwrite").parquet(newTempPath("scan/t_struct"))
    spark.read.parquet(newTempPath("scan/t_struct")).createOrReplaceTempView("t_struct")
  }

  test("the vectorized Parquet reader switched off leaves a row scan: our operators fall back with the reason") {
    withConf("spark.sql.parquet.enableVectorizedReader" -> "false") {
      val df = checkFallback("SELECT i, l FROM t WHERE i % 5 = 0", Seq(Filter), "is not columnar")
      assert(nodesOf[FileSourceScanExec](df).forall(!_.supportsColumnar))
      checkFallback("SELECT i + 1 AS j, upper(s) AS us FROM t", Seq(Project), "is not columnar")
      // A merging aggregate reads the shuffle, not the scan, so the Final stage stays ours over RowToColumnar.
      checkVectorized("SELECT s, count(*) AS n FROM t GROUP BY s", Seq(Agg))
    }
  }

  test("off-heap Parquet batches: fixed-width columns are views, not copies, and the rows match") {
    val wrappedBefore = SparkColumnVectorBuffers.wrappedOffHeapColumns()
    withConf("spark.sql.columnVector.offheap.enabled" -> "true") {
      checkVectorized("SELECT i, l, d, dt FROM t WHERE d > 100 AND l IS NOT NULL", Seq(Filter))
      checkVectorized("SELECT i + 1 AS j, l * 2 AS m, d / 2 AS h, year(dt) AS y, s, b FROM t WHERE i < 5000", Seq(Filter, Project))
      checkVectorized("SELECT s, count(*) AS n, sum(l) AS sl, avg(d) AS ad, min(dt) AS mdt, max(i) AS mi FROM t GROUP BY s", Seq(Agg))
      checkVectorized("SELECT count(*) AS n, count(l) AS nl, count(d) AS nd FROM t", Seq(Agg))
    }
    val wrapped = SparkColumnVectorBuffers.wrappedOffHeapColumns() - wrappedBefore
    // t is 20000 rows in 3 partitions of 4096-row batches: at least 15 batches, each with fixed-width columns.
    assert(wrapped >= 15, s"expected off-heap fixed-width columns to be wrapped, saw $wrapped")
    // On-heap (the default) never wraps.
    val onHeapBefore = SparkColumnVectorBuffers.wrappedOffHeapColumns()
    checkVectorized("SELECT i, l FROM t WHERE i % 7 = 0", Seq(Filter))
    assert(SparkColumnVectorBuffers.wrappedOffHeapColumns() === onHeapBefore)
  }

  test("an ORC table read by Spark's ORC vectorized reader is a columnar input through the generic copy") {
    val copiedBefore = ColumnVectorAdapters.copiedColumns()
    val df = checkVectorized("SELECT i, l, d, s FROM t_orc WHERE i % 3 = 0 AND l IS NOT NULL", Seq(Filter))
    assert(nodesOf[FileSourceScanExec](df).exists(_.supportsColumnar), finalPlan(df).treeString)
    checkVectorized("SELECT i + 1 AS j, upper(s) AS us, dt, d2, b FROM t_orc WHERE i < 3000", Seq(Filter, Project))
    checkVectorized("SELECT s, count(*) AS n, sum(l) AS sl, max(dt) AS mdt, sum(d2) AS sd FROM t_orc GROUP BY s", Seq(Agg))
    assert(ColumnVectorAdapters.copiedColumns() > copiedBefore, "ORC vectors are foreign and copied")
  }

  test("a nested column keeps Spark's scan columnar but our operators fall back on the type") {
    val df = checkFallback("SELECT i, st.a AS a FROM t_struct WHERE i > 10", Seq(Filter), "unsupported column type struct")
    assert(nodesOf[FileSourceScanExec](df).exists(_.supportsColumnar), "Spark 4's nested vectorized reader keeps the scan columnar\n" + finalPlan(df).treeString)
    // Pruned away, the struct column no longer stands in the way.
    checkVectorized("SELECT i, l FROM t_struct WHERE i > 10 AND l IS NOT NULL", Seq(Filter))
  }
}
