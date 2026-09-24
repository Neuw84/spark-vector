package io.sparkvector.spark

import java.util.concurrent.atomic.AtomicInteger

import io.sparkvector.spark.arrow.VectorArrowColumnVector
import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.{FileSourceScanExec, SparkPlan}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType, StructType}
import org.apache.spark.sql.vector.{
  PlanUtils,
  PrefetchingBatchConverter,
  PrefetchMetrics,
  VectorFilterExec,
  VectorHashAggregateExec,
  VectorPlan,
  VectorPrefetchScanExec,
  VectorRowStages
}
import org.apache.spark.sql.vector.ui.{Engine, PlanAcceleration}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.types.UTF8String

/**
 * The prefetching scan converter (#403, lever 2): the plan rule's insertion under the first operator
 * of ours above a Spark file scan, the rows it hands over, and the converter itself over a slow and
 * a failing child (overlap, error propagation, early close without leaks).
 */
class VectorPrefetchScanSuite extends VectorQuerySuite {

  private val Prefetch = classOf[VectorPrefetchScanExec]
  private val Filter = classOf[VectorFilterExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("prefetch/t"))
    TestTables
      .mixedDataFrame(spark, 5000)
      .repartition(2)
      .write
      .mode("overwrite")
      .parquet(newTempPath("prefetch/t2"))
    spark.read.parquet(newTempPath("prefetch/t2")).createOrReplaceTempView("t2")
    // Decimals of every width beside the mixed types: narrow (int and long lanes) and a wide one.
    spark
      .range(0, 8000)
      .selectExpr(
        "cast(id as int) as i",
        "cast(id % 4 as int) as k",
        "if(id % 7 = 3, null, cast(cast(id % 10007 as double) / 4 - 900 as decimal(7,2))) as dec7",
        "cast(cast(id * 37 % 1000003 as double) / 100 - 3000 as decimal(12,2)) as dec12",
        "if(id % 5 = 0, null, cast(cast(id as double) / 7 as decimal(18,4))) as dec18",
        "if(id % 11 = 0, null, cast(id as decimal(25,3)) * cast('1234567.891' as decimal(25,3))) as wide"
      )
      .repartition(2)
      .write
      .mode("overwrite")
      .parquet(newTempPath("prefetch/dec"))
    spark.read.parquet(newTempPath("prefetch/dec")).createOrReplaceTempView("dec")
  }

  private def prefetchNodes(df: DataFrame): Seq[VectorPrefetchScanExec] = nodesOf[VectorPrefetchScanExec](df)

  /** The parents of every prefetch node in the final plan. */
  private def parentsOfPrefetch(df: DataFrame): Seq[SparkPlan] =
    PlanUtils.allNodes(finalPlan(df)).filter(_.children.exists(_.isInstanceOf[VectorPrefetchScanExec]))

  /** Rows of `sql` with prefetch `depth`, sorted for comparison. */
  private def rowsWith(sql: String, depth: Int): Seq[Row] =
    withConf(VectorConf.ScanPrefetch -> depth.toString)(spark.sql(sql).collect().toSeq)

  private def checkPrefetched(sql: String, operators: Seq[Class[_ <: SparkPlan]], scans: Int = 1): DataFrame =
    withConf(VectorConf.ScanPrefetch -> "2") {
      val df = checkVectorized(sql, operators :+ Prefetch)
      val nodes = prefetchNodes(df)
      assert(nodes.size === scans, s"expected $scans prefetch node(s):\n${finalPlan(df).treeString}")
      nodes.foreach { n =>
        assert(n.depth === 2)
        assert(n.child.isInstanceOf[FileSourceScanExec], s"prefetch over ${n.child.nodeName}")
      }
      val parents = parentsOfPrefetch(df)
      assert(parents.size === scans && parents.forall(_.isInstanceOf[VectorPlan]), finalPlan(df).treeString)
      // The rows are Spark's (checkVectorized) and the prefetch-off plugin's.
      val off = rowsWith(sql, 0)
      assertRowsEqual(off.toArray, df.collect(), 1e-9, s"prefetch on vs off: $sql")
      df
    }

  test("scan, filter and aggregate over every type read through one prefetch node") {
    val sql = "SELECT s, count(*), sum(l), max(d), min(dt), sum(if(b, 1, 0)), min(i) FROM t WHERE i > 100 GROUP BY s"
    val df = checkPrefetched(sql, Seq(Filter, Agg))
    // The node under the first operator of ours -- the filter -- and nowhere else.
    assert(parentsOfPrefetch(df).forall(_.isInstanceOf[VectorFilterExec]), finalPlan(df).treeString)
    // Plumbing in the acceleration view: the operator count is that of the prefetch-off plan.
    val on = PlanAcceleration.fromPlan(finalPlan(df))
    val offPlan = withConf(VectorConf.ScanPrefetch -> "0") {
      val d = spark.sql(sql)
      d.collect()
      finalPlan(d)
    }
    val off = PlanAcceleration.fromPlan(offPlan)
    assert(on.operatorCount === off.operatorCount)
    assert(on.countBy(Engine.Transition) === off.countBy(Engine.Transition) + 1)
    // Dictionary-encoded strings and the projection's pass-through; every batch of the table.
    checkPrefetched("SELECT i, s, d2 FROM t WHERE s IS NOT NULL", Seq(Filter))
    checkPrefetched("SELECT count(*), sum(d2), count(s) FROM t", Seq(Agg))
    // Decimals of every width.
    checkPrefetched("SELECT k, sum(dec7), sum(dec12), max(dec18), min(dec7), count(wide) FROM dec GROUP BY k", Seq(Agg))
    checkPrefetched("SELECT i, dec12, wide FROM dec WHERE dec18 IS NOT NULL AND i % 3 = 0", Seq(Filter))
  }

  test("a join with two scan sides gets a prefetch node on each") {
    checkPrefetched(
      "SELECT a.i, b.l, a.s FROM t a JOIN t2 b ON a.i = b.i WHERE a.d > 1 AND b.d2 > 0.5",
      Seq(Filter),
      scans = 2
    )
  }

  test("prefetch off (the default) leaves the plan unchanged") {
    val sql = "SELECT s, count(*) FROM t WHERE i > 100 GROUP BY s"
    val df = checkVectorized(sql, Seq(Filter, Agg))
    assert(prefetchNodes(df).isEmpty)
    val filters = nodesOf[VectorFilterExec](df)
    assert(filters.nonEmpty && filters.forall(_.child.isInstanceOf[FileSourceScanExec]), finalPlan(df).treeString)
    assert(VectorConf.scanPrefetchDepth(spark.sessionState.conf) === 0)
  }

  // ------------------------------------------------------------------ the converter on its own

  private val schema = new StructType()
    .add("i", IntegerType)
    .add("l", LongType)
    .add("d", DoubleType)
    .add("s", StringType)

  private def batch(b: Int, rows: Int = 1024): ColumnarBatch =
    VectorRowStages.toBatch(
      schema,
      Array.tabulate(rows) { r =>
        val id = b * rows + r
        InternalRow(id, if (id % 7 == 3) null else id.toLong * 3, id / 7.0, UTF8String.fromString(s"s${id % 50}"))
      }
    )

  private def metrics: PrefetchMetrics =
    PrefetchMetrics(new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("sum"))

  /** A child that sleeps `sleepMs` before each batch and throws `failure` instead of batch `failAt`. */
  private final class SlowChild(numBatches: Int, sleepMs: Long, failAt: Int = -1, failure: Throwable = null)
      extends Iterator[ColumnarBatch] {
    val produced = new AtomicInteger()
    override def hasNext: Boolean = produced.get() < numBatches
    override def next(): ColumnarBatch = {
      Thread.sleep(sleepMs)
      val b = produced.getAndIncrement()
      if (b == failAt) throw failure
      batch(b)
    }
  }

  private def sumOf(b: ColumnarBatch): Long = {
    var s = 0L
    var i = 0
    while (i < b.numRows()) { s += b.column(0).getInt(i); i += 1 }
    s
  }

  test("the converter overlaps the read with the consumer and hands out our own vectors") {
    val n = 10
    val sleep = 40L
    def run(depth: Int): (Long, Long, PrefetchMetrics) = {
      val m = metrics
      val child = new SlowChild(n, sleep)
      val it = new PrefetchingBatchConverter(child, schema.fields.map(f => (f.name, f.dataType)), depth, m)
      val start = System.nanoTime()
      var total = 0L
      try {
        while (it.hasNext) {
          val b = it.next()
          assert(b.numRows() === 1024)
          (0 until b.numCols()).foreach(c => assert(b.column(c).isInstanceOf[VectorArrowColumnVector], s"column $c"))
          total += sumOf(b)
          Thread.sleep(sleep) // the consumer's own work per batch
        }
      } finally it.close()
      assert(child.produced.get() === n)
      assert(it.allocator.getAllocatedMemory === 0L)
      ((System.nanoTime() - start) / 1000000L, total, m)
    }
    val (elapsed, total, m) = run(depth = 2)
    val expected = (0 until n * 1024).map(_.toLong).sum
    assert(total === expected)
    assert(m.batches.value === n)
    // Sequential would be at least 2 x n x sleep = 800 ms; the overlap keeps it well under that.
    assert(elapsed < 2 * n * sleep - 3 * sleep, s"no overlap: $elapsed ms")
    assert(m.readWaitMs.value >= (n - 1) * sleep, s"read wait ${m.readWaitMs.value} ms")
    assert(m.prefetchWaitMs.value < 2 * n * sleep, s"prefetch wait ${m.prefetchWaitMs.value} ms")
  }

  test("an exception thrown by the child surfaces on the consumer as the same exception") {
    val boom = new IllegalStateException("boom on batch 3")
    val m = metrics
    val child = new SlowChild(10, 1, failAt = 3, failure = boom)
    val it = new PrefetchingBatchConverter(child, schema.fields.map(f => (f.name, f.dataType)), 2, m)
    var seen = 0
    val thrown = intercept[IllegalStateException] {
      while (it.hasNext) {
        it.next()
        seen += 1
      }
    }
    assert(thrown eq boom)
    assert(seen === 3)
    it.close()
    assert(it.allocator.getAllocatedMemory === 0L)
  }

  test("closing early stops the helper within a second and leaks no Arrow memory") {
    val m = metrics
    val child = new SlowChild(10, 20)
    val it = new PrefetchingBatchConverter(child, schema.fields.map(f => (f.name, f.dataType)), 2, m)
    assert(it.hasNext)
    val first = it.next()
    val firstSum = sumOf(first) // read before the next take closes it (the columnar contract)
    assert(it.hasNext)
    val second = it.next()
    assert(sumOf(second) > firstSum)
    val start = System.nanoTime()
    it.close()
    val closeMs = (System.nanoTime() - start) / 1000000L
    assert(!it.isHelperAlive, "helper still running after close")
    assert(closeMs < 1000L, s"close took $closeMs ms")
    assert(it.allocator.getAllocatedMemory === 0L)
    // The helper stopped between batches: at most the two handed out plus the queue's depth and one in flight.
    val produced = child.produced.get()
    assert(produced <= 2 + 2 + 1 && produced >= 2, s"child produced $produced batches")
    Thread.sleep(100)
    assert(child.produced.get() === produced, "the child kept being read after close")
    assert(!it.hasNext)
  }
}
