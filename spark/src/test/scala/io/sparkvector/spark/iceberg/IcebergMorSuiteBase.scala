package io.sparkvector.spark.iceberg

import java.nio.file.{Files, Path}

import io.sparkvector.spark.VectorConf
import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec, VectorHashAggregateExec, VectorProjectExec}
import org.scalatest.Tag

/**
 * Queries over Iceberg merge-on-read tables, shared by the suite that reads them through Comet's
 * native Iceberg scan and the one that reads them through Iceberg's own JVM vectorized reader.
 * Both must produce Spark's answer with spark-vector operators directly above the scan.
 */
abstract class IcebergMorSuiteBase extends VectorQuerySuite {

  protected val Filter = classOf[VectorFilterExec]
  protected val Project = classOf[VectorProjectExec]
  protected val Agg = classOf[VectorHashAggregateExec]

  private lazy val warehouse: Path = Files.createTempDirectory("spark-vector-iceberg")

  /** Catalog and extension configuration; subclasses add their reader configuration on top. */
  protected def icebergConf: Map[String, String] = IcebergTables.catalogConf(warehouse.toString)

  /** Tags every test in the suite carries (which Maven profiles it needs). */
  protected def suiteTags: Seq[Tag]

  /** Simple class name of the columnar scan expected under our operators. */
  protected def expectedScanClass: String

  /** Per table, for readers that only handle some of them; defaults to [[expectedScanClass]]. */
  protected def expectedScanClassFor(table: String): String = expectedScanClass

  /** The table the `t` view currently points at (set by [[useTable]]). */
  private var currentTable: String = _

  protected def useTable(name: String): Unit = {
    currentTable = name
    IcebergTables.useAsT(spark, s"${IcebergTables.Db}.$name")
  }

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    IcebergTables.createAll(spark)
  }

  override protected def afterAll(): Unit = {
    try super.afterAll()
    finally {
      if (Files.exists(warehouse)) {
        Files.walk(warehouse).sorted(java.util.Comparator.reverseOrder[Path]()).forEach(f => Files.deleteIfExists(f))
      }
    }
  }

  protected def icebergTest(name: String)(body: => Unit): Unit = test(name, suiteTags: _*)(body)

  /** The leaf feeding the first spark-vector operator of `df` must be the expected scan. */
  protected def assertScanUnder(df: DataFrame, op: Class[_ <: SparkPlan]): Unit = {
    val ops = PlanUtils.allNodes(finalPlan(df)).filter(op.isInstance)
    assert(ops.nonEmpty, s"expected ${op.getSimpleName}:\n${finalPlan(df).treeString}")
    val leaves = ops.collect { case u: UnaryExecNode => u.child }.filter(_.children.isEmpty)
    assert(leaves.nonEmpty, s"${op.getSimpleName} is not directly above a scan:\n${finalPlan(df).treeString}")
    val expected = expectedScanClassFor(currentTable)
    leaves.foreach { leaf =>
      assert(
        leaf.getClass.getSimpleName == expected,
        s"expected $expected under ${op.getSimpleName}, got ${leaf.getClass.getName}:\n${finalPlan(df).treeString}")
    }
  }

  protected def count(sql: String): Long = spark.sql(sql).collect()(0).getLong(0)

  /** The standard battery over whatever table the `t` view points at. */
  protected def checkMixedQueries(): Unit = {
    val f = checkVectorized("SELECT i, l, d, dt, b, s FROM t WHERE i > 100 AND d IS NOT NULL", Seq(Filter))
    assertScanUnder(f, Filter)
    checkVectorized("SELECT i, d * 2.0 AS x, s FROM t WHERE d2 > 0.5 AND l IS NOT NULL", Seq(Filter, Project))
    val a = checkVectorized("SELECT s, count(*), sum(d2), avg(d2), min(i), max(l) FROM t GROUP BY s", Seq(Agg))
    assertScanUnder(a, Agg)
    checkVectorized("SELECT count(*), sum(d2), max(d), min(dt) FROM t WHERE i > 10", Seq(Filter, Agg))
    checkVectorized("SELECT i, count(*), sum(d2) FROM t GROUP BY i", Seq(Agg))
    checkVectorized("SELECT s, b, count(*), sum(d2) FROM t WHERE i > 5 GROUP BY s, b", Seq(Filter, Agg))
    // Selective and non-selective predicates exercise both the compacting and the forwarding path.
    checkVectorized("SELECT i, s FROM t WHERE i > 19000", Seq(Filter))
    checkVectorized("SELECT i, d2 FROM t WHERE b OR i < 15000", Seq(Filter))
    // Rows re-inserted by UPDATE live in newer data files with no deletes attached; d2 + 1.0 lifts
    // them above the original range [0, 3], so this predicate selects exactly those rows.
    checkVectorized("SELECT count(*), sum(d2), min(i), max(i) FROM t WHERE d2 > 3.0", Seq(Filter, Agg))
  }

  icebergTest("tables carry delete files") {
    Seq("t_pos", "t_dv", "t_eq", "lineitem").foreach { n =>
      assert(IcebergTables.deleteFileCount(spark, s"${IcebergTables.Db}.$n") > 0, s"$n has no delete files")
    }
  }

  icebergTest("positional deletes (v2) are merged before our operators see the batch") {
    useTable("t_pos")
    assert(count("SELECT count(*) FROM t") === IcebergTables.MixedLiveRows)
    assert(count("SELECT count(*) FROM t WHERE i % 5 = 0") === 0)
    checkMixedQueries()
  }

  icebergTest("deletion vectors (v3) are merged before our operators see the batch") {
    useTable("t_dv")
    assert(count("SELECT count(*) FROM t") === IcebergTables.MixedLiveRows)
    checkMixedQueries()
  }

  icebergTest("equality deletes written with the Java API apply together with positional ones") {
    useTable("t_eq")
    assert(count("SELECT count(*) FROM t") === IcebergTables.EqLiveRows)
    assert(count("SELECT count(*) FROM t WHERE i IN (1, 2, 3, 19999)") === 0)
    checkMixedQueries()
  }

  icebergTest("TPC-H Q1 and Q6 over a merge-on-read lineitem") {
    currentTable = "lineitem"
    IcebergTables.useAsLineitem(spark, s"${IcebergTables.Db}.lineitem")
    assert(count("SELECT count(*) FROM lineitem WHERE pmod(l_orderkey, 50) = 7") === 0)
    val q6 = checkVectorized(TestTables.TpchQ6, Seq(Filter, Agg))
    assertScanUnder(q6, Filter)
    val q1 = checkVectorized(TestTables.TpchQ1, Seq(Filter, Agg))
    assertScanUnder(q1, Filter)
    info(finalPlan(q1).treeString)
  }

  /**
   * The end-to-end merge scenario: a table that already carries many rounds of merge-on-read
   * deletes and re-inserts receives an incoming batch through `MERGE INTO` (updates, deletes and
   * inserts). Two identical copies are merged, one with the plugin on and one with it off, and
   * must end up identical; the merged table is then queried with our operators, which now have to
   * see through the merge's own position deletes as well as the earlier ones.
   */
  icebergTest("MERGE INTO over a heavily mutated merge-on-read table, then vectorized reads") {
    val on = s"${IcebergTables.Db}.t_merge_on"
    val off = s"${IcebergTables.Db}.t_merge_off"
    IcebergTables.createHeavilyMutated(spark, on)
    IcebergTables.createHeavilyMutated(spark, off)
    IcebergTables.createMergeSource(spark)
    val deletesBefore = IcebergTables.deleteFileCount(spark, on)
    assert(deletesBefore >= 7, s"expected many delete files before the merge, got $deletesBefore")

    // Flagged keys still present in the target are deleted by the merge; flagged keys already
    // removed by the earlier rounds do not match and are inserted (with their negative d2).
    val flagged = IcebergTables.MergeDeletedKeys.toSeq.sorted.mkString(", ")
    val flaggedPresent = count(s"SELECT count(*) FROM $on WHERE i IN ($flagged)")
    val flaggedAbsent = IcebergTables.MergeDeletedKeys.size - flaggedPresent

    withPlugin(enabled = false)(spark.sql(IcebergTables.mergeSql(off)).collect())
    val merge = withPlugin(enabled = true) {
      val d = spark.sql(IcebergTables.mergeSql(on))
      d.collect()
      d
    }
    val vectorNodes = PlanUtils.allNodes(finalPlan(merge)).filter(_.isInstanceOf[org.apache.spark.sql.vector.VectorExec])
    info(s"spark-vector operators in the MERGE plan: ${vectorNodes.map(_.nodeName).mkString(", ")}")
    info(finalPlan(merge).treeString)
    assert(IcebergTables.deleteFileCount(spark, on) > deletesBefore, "MERGE should have added delete files, not rewritten data files")

    // The write must not depend on which engine evaluated the plan: compare both tables row by row.
    val readAll = "SELECT i, l, d, d2, dt, b, s FROM %s"
    val expected = withPlugin(enabled = false)(spark.sql(readAll.format(off)).collect())
    val actual = withPlugin(enabled = false)(spark.sql(readAll.format(on)).collect())
    assertRowsEqual(expected, actual, 1e-9, "table contents after MERGE INTO")
    assert(expected.length === count(s"SELECT count(*) FROM $off"))
    assert(count(s"SELECT count(*) FROM $on WHERE i >= ${IcebergTables.MixedRows}") === IcebergTables.MergeInsertedKeys.size)
    assert(count(s"SELECT count(*) FROM $on WHERE d2 < 0") === flaggedAbsent, "only unmatched flagged rows keep a negative d2")
    assert(count(s"SELECT count(*) FROM $on WHERE d2 < 0 AND i IN ($flagged)") === flaggedAbsent)

    // Reads over the merged table: the merge's position deletes stack on the earlier rounds.
    useTable("t_merge_on")
    checkMixedQueries()
    val m = checkVectorized("SELECT s, count(*), sum(l), max(d2) FROM t WHERE d2 >= 10.0 GROUP BY s", Seq(Filter, Agg))
    assertScanUnder(m, Filter)
  }

  /**
   * The columnar MergeRows (#21) is planned over its join: it converts when MergeRowsExec's child is
   * columnar. Today the merge's join is never ours -- the target side carries Iceberg's struct
   * `_partition` metadata column, which the hash join does not pass through (the sort-merge route
   * refuses it with `unsupported column type struct<> for _partition`) -- so the operator records
   * why and Spark's runs. `VectorMergeRowsSuite` exercises the operator itself over a columnar child.
   */
  icebergTest("MergeRows records why the merge's join is not ours, and the merge is unchanged (#21)") {
    val Db = IcebergTables.Db
    IcebergTables.createMergeSource(spark)
    val on = s"$Db.m_row_on"
    val off = s"$Db.m_row_off"
    IcebergTables.createMixedMor(spark, on, formatVersion = 2)
    IcebergTables.createMixedMor(spark, off, formatVersion = 2)
    withPlugin(enabled = false)(spark.sql(IcebergTables.mergeSql(off)).collect())
    withConf(VectorConf.SortMergeJoinEnabled -> "true", "spark.sql.autoBroadcastJoinThreshold" -> "-1", VectorConf.ExplainFallbackEnabled -> "true") {
      withPlugin(enabled = true)(spark.sql(IcebergTables.mergeSql(on)).collect())
    }
    val readAll = "SELECT i, l, d, d2, dt, b, s FROM %s ORDER BY i"
    val expected = withPlugin(enabled = false)(spark.sql(readAll.format(off)).collect())
    val actual = withPlugin(enabled = false)(spark.sql(readAll.format(on)).collect())
    assertRowsEqual(expected, actual, 1e-9, "table contents after MERGE INTO")
  }

  icebergTest("plugin disabled leaves the same scan feeding Spark operators") {
    useTable("t_pos")
    withConf(VectorConf.Enabled -> "false") {
      val df = spark.sql("SELECT count(*) FROM t WHERE i > 10")
      df.collect()
      val leaves = PlanUtils.allNodes(finalPlan(df)).filter(_.children.isEmpty)
      assert(leaves.map(_.getClass.getSimpleName).contains(expectedScanClassFor("t_pos")), finalPlan(df).treeString)
      assert(nodesOf[VectorFilterExec](df).isEmpty)
    }
  }
}
