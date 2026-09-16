package io.sparkvector.spark

import io.sparkvector.spark.test.{TestTables, VectorQuerySuite}

import org.apache.spark.sql.vector.{VectorFilterExec, VectorProjectExec}

class VectorProjectSuite extends VectorQuerySuite {

  private val Project = classOf[VectorProjectExec]
  private val Filter = classOf[VectorFilterExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("project/t"))
  }

  test("double arithmetic with columns and literals (ANSI mode, Spark 4 default)") {
    assert(spark.conf.get("spark.sql.ansi.enabled") === "true")
    checkVectorized("SELECT i, d * 2.0 AS x, d + d2 AS y, d - d2 AS z, 1.0 - d2 AS w, -d AS neg FROM t WHERE i > 100", Seq(Filter, Project))
    checkVectorized("SELECT d * (1.0 - d2) * (1.0 + d2) AS q1_style FROM t WHERE d IS NOT NULL", Seq(Filter, Project))
    checkVectorized("SELECT d2 / 4.0 AS a, 8.0 / (d2 + 1.0) AS b FROM t WHERE i < 5000", Seq(Filter, Project))
  }

  test("division by zero raises in ANSI mode, yields null in legacy mode") {
    // d2 is zero where id % 13 = 0.
    val sql = "SELECT i, d / d2 AS r FROM t WHERE i < 100"
    withPlugin(enabled = true) {
      val e = intercept[Exception](spark.sql(sql).collect())
      assert(causes(e).exists(c => c.isInstanceOf[ArithmeticException] && c.getMessage.contains("DIVIDE_BY_ZERO")), s"expected DIVIDE_BY_ZERO, got $e")
    }
    withConf("spark.sql.ansi.enabled" -> "false") {
      val df = checkVectorized(sql, Seq(Project))
      assert(df.filter("r IS NULL").count() === 17) // d null where i % 11 = 0 (10 rows) plus d2 = 0 where i % 13 = 0 (8 rows), overlapping at 0
      checkVectorized("SELECT i, 1.0 / d2 AS r, d / 0.0 AS all_null FROM t WHERE i < 100", Seq(Project))
    }
  }

  test("integer arithmetic is vectorized in legacy mode and falls back in ANSI mode") {
    withConf("spark.sql.ansi.enabled" -> "false") {
      checkVectorized("SELECT i + 1 AS a, i * 3 AS b, l - 7 AS c, 100 - i AS d, -i AS e, l * l AS f FROM t WHERE i > 10", Seq(Project))
      checkVectorized("SELECT i + i AS a, l + l AS b, CAST(i AS BIGINT) + l AS c FROM t WHERE d IS NULL", Seq(Project))
    }
    checkFallback("SELECT i + 1 AS a FROM t WHERE i > 10", Seq(Project), "ANSI integer arithmetic")
  }

  test("widening casts and implicit casts inserted by the analyzer") {
    checkVectorized("SELECT CAST(i AS BIGINT) AS a, CAST(i AS DOUBLE) AS b, CAST(l AS DOUBLE) AS c FROM t WHERE i > 5", Seq(Project))
    checkVectorized("SELECT i + 1.5D AS a, l * 0.5D AS b, i / 2 AS c FROM t WHERE i > 5", Seq(Project))
    // A plain decimal literal makes Spark cast the int to a decimal; both run on long lanes.
    checkVectorized("SELECT i + 1.5 AS a FROM t WHERE i > 5", Seq(Project))
    checkFallback("SELECT CAST(d2 AS INT) AS a FROM t WHERE i > 5", Seq(Project), "unsupported cast")
  }

  test("forwarded and reordered columns, including strings and nulls") {
    checkVectorized("SELECT s, i, l, b, dt FROM t WHERE d > 1", Seq(Filter, Project))
    checkVectorized("SELECT dt AS when, s AS name, d2 * 2.0 AS twice FROM t WHERE l IS NOT NULL", Seq(Project))
  }

  test("chained filter and project both stay columnar") {
    val df = checkVectorized("SELECT i, d * d2 AS p FROM t WHERE d > 5 AND d2 > 0.5", Seq(Filter, Project))
    val project = nodesOf[VectorProjectExec](df).head
    assert(project.child.isInstanceOf[VectorFilterExec], project.treeString)
  }

  test("a filter below a projection forwards a selection bitmap instead of compacting") {
    val df = checkVectorized("SELECT s, i, d * d2 AS p, l FROM t WHERE d > 5 AND d2 > 0.5 AND s IS NOT NULL", Seq(Filter, Project))
    val filter = nodesOf[VectorFilterExec](df).head
    val project = nodesOf[VectorProjectExec](df).head
    assert(filter.emitSelection, "filter feeding our projection should emit a selection")
    assert(!project.emitSelection, "projection feeding Spark must compact")
    // Same query with selections disabled: identical rows, dense batches everywhere.
    withConf(VectorConf.SelectionEnabled -> "false") {
      val dense = checkVectorized("SELECT s, i, d * d2 AS p, l FROM t WHERE d > 5 AND d2 > 0.5 AND s IS NOT NULL", Seq(Filter, Project))
      assert(!nodesOf[VectorFilterExec](dense).head.emitSelection)
    }
  }

  test("ANSI errors are not raised for rows the filter removed") {
    // d2 is zero where id % 13 = 0; Spark never evaluates the projection for those rows.
    checkVectorized("SELECT i, 10.0 / d2 AS r FROM t WHERE d2 > 0.0", Seq(Filter, Project))
    // Same inside a conjunction: the right operand only matters where the left one holds.
    checkVectorized("SELECT i FROM t WHERE d2 > 0.0 AND 10.0 / d2 > 3.0", Seq(Filter))
  }

  test("selection survives an identity projection and reaches the aggregate") {
    val df = checkVectorized("SELECT count(*), sum(d), min(i), max(l), avg(d2) FROM t WHERE d > 5 AND d2 > 0.5", Seq(Filter, classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]))
    assert(nodesOf[VectorFilterExec](df).head.emitSelection)
    val grouped = checkVectorized("SELECT s, count(*), sum(d), count(l) FROM t WHERE d > 5 GROUP BY s", Seq(Filter, classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]))
    assert(nodesOf[VectorFilterExec](grouped).head.emitSelection)
    // Groups that only occur in filtered-out rows must not appear.
    checkVectorized("SELECT s, count(*) FROM t WHERE i < 30 GROUP BY s", Seq(Filter, classOf[org.apache.spark.sql.vector.VectorHashAggregateExec]))
  }

  test("unsupported expressions in a projection fall back") {
    checkFallback("SELECT concat(s, 'x') AS c FROM t WHERE i > 5", Seq(Project), "unsupported expression")
    checkFallback("SELECT i % 3 AS m FROM t WHERE i > 5", Seq(Project), "unsupported expression")
  }

  test("project conversion can be disabled by configuration") {
    withConf(VectorConf.ProjectEnabled -> "false") {
      val df = withPlugin(enabled = true)(spark.sql("SELECT d * 2.0 AS x FROM t WHERE i > 5"))
      df.collect()
      assert(nodesOf[VectorProjectExec](df).isEmpty)
      assert(nodesOf[VectorFilterExec](df).nonEmpty)
    }
  }

  private def causes(t: Throwable): Seq[Throwable] =
    Iterator.iterate(t)(_.getCause).takeWhile(_ != null).take(10).toSeq
}
