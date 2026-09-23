package io.sparkvector.spark

import io.sparkvector.spark.adapter.ColumnVectorAdapters
import io.sparkvector.spark.test.{UnknownColumnarSource, VectorQuerySuite}
import org.apache.spark.sql.execution.RowToColumnarExec
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec
import org.apache.spark.sql.vector.{VectorFilterExec, VectorHashAggregateExec, VectorProjectExec}

/**
 * An unknown columnar DSv2 source -- `ColumnVector`s no adapter recognises -- is accepted as input
 * and copied once per batch through the adapter seam's fallback (#88, split out of #62). The rows
 * must match Spark's, our operators must sit directly on the scan, and the counters must show every
 * column went through the copy, not a zero-copy adapter.
 */
class VectorUnknownSourceSuite extends VectorQuerySuite {

  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]
  private val source = classOf[UnknownColumnarSource].getName

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark.read.format(source).load().createOrReplaceTempView("u")
    spark.read.format(source).option("struct", "true").load().createOrReplaceTempView("u_struct")
  }

  test("an unknown columnar source feeds our filter, projection and aggregate, copied once per batch") {
    val adaptedBefore = ColumnVectorAdapters.adaptedColumns()
    val copiedBefore = ColumnVectorAdapters.copiedColumns()
    val batches = UnknownColumnarSource.Partitions * UnknownColumnarSource.BatchesPerPartition

    val f = checkVectorized("SELECT i, l, d, s FROM u WHERE i % 3 = 0 AND l IS NOT NULL", Seq(Filter))
    val p = checkVectorized("SELECT i + 1 AS j, d * 2 AS e, upper(s) AS us, dt, ts, dec, b FROM u", Seq(Project))
    val a = checkVectorized(
      "SELECT s, count(*) AS n, sum(l) AS sl, max(dec) AS md, min(dt) AS mdt, count(ts) AS nts, sum(d) AS sd FROM u GROUP BY s",
      Seq(Agg)
    )
    for (df <- Seq(f, p, a)) {
      assert(nodesOf[BatchScanExec](df).nonEmpty, finalPlan(df).treeString)
      assert(
        nodesOf[RowToColumnarExec](df).forall(!_.child.isInstanceOf[BatchScanExec]),
        "the source's batches should be consumed directly\n" + finalPlan(df).treeString
      )
    }
    // Every column of every batch read with the plugin on went through the copy fallback; the
    // filter reads 8 columns (the scan is not pruned), the projection 8, the aggregate 8.
    val copied = ColumnVectorAdapters.copiedColumns() - copiedBefore
    assert(copied >= 3L * batches * 8, s"expected at least ${3 * batches * 8} copied columns, saw $copied")
    assert(
      ColumnVectorAdapters.adaptedColumns() === adaptedBefore,
      "no zero-copy adapter should claim an unknown vector"
    )
  }

  test("nulls, dictionaries of one and every supported type survive the copy") {
    checkVectorized("SELECT count(*) AS n, count(l) AS nl, count(d) AS nd, count(ts) AS nts FROM u", Seq(Agg))
    checkVectorized("SELECT i, l, d, b, dt, ts, dec, s FROM u WHERE l IS NULL OR d IS NULL OR ts IS NULL", Seq(Filter))
    checkVectorized("SELECT b, count(*) AS n, sum(i) AS si FROM u GROUP BY b", Seq(Agg))
    checkVectorized(
      "SELECT year(dt) AS y, hour(ts) AS h, dec + 1 AS d1, length(s) AS ls FROM u WHERE i < 100",
      Seq(Filter, Project)
    )
  }

  test(
    "a struct column from a foreign source passes through the filter as the source's vector; its field is read from the source's child vector"
  ) {
    checkVectorized(
      "SELECT i, st.a AS a FROM u_struct WHERE i > 10",
      Seq(Filter, classOf[org.apache.spark.sql.vector.VectorProjectExec])
    )
    checkVectorized("SELECT i, st FROM u_struct WHERE i > 10", Seq(Filter))
  }
}
