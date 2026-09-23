package io.sparkvector.spark

import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.vector.{
  VectorFilterExec,
  VectorHashAggregateExec,
  VectorProjectExec,
  VectorSortExec,
  VectorWindowExec
}

/**
 * TINYINT and SMALLINT (#327): columns ride INT32 lanes and Spark reads the declared type back
 * through `VectorNarrowIntColumnVector`. Casts in (with Spark's legacy / ANSI / try semantics), casts
 * out, arithmetic narrowed into the type, and the operators the ROLLUP queries of TPC-DS (q27, q36,
 * q70, q86) run them through: grouping keys, window partitions, sort keys, joins.
 */
class VectorNarrowIntSuite extends VectorQuerySuite {
  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]
  private val Agg = classOf[VectorHashAggregateExec]
  private val Sort = classOf[VectorSortExec]
  private val Window = classOf[VectorWindowExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    val df = spark.range(0, 20000).selectExpr(
      "cast(id as int) as i",
      "cast(id as bigint) as l",
      "cast((id % 1000) / 7.0 as double) as d",
      // The extremes and nulls: tinyint over the whole range, smallint likewise.
      "case when id % 97 = 1 then cast(127 as tinyint) when id % 97 = 2 then cast(-128 as tinyint) when id % 13 = 0 then null else cast(id % 200 - 100 as tinyint) end as t",
      "case when id % 89 = 1 then cast(32767 as smallint) when id % 89 = 2 then cast(-32768 as smallint) when id % 11 = 0 then null else cast(id % 50000 - 25000 as smallint) end as s",
      "cast(id % 5 as tinyint) as g",
      "concat('k', id % 40) as k"
    )
    df.repartition(3).write.mode("overwrite").option(
      "parquet.enable.dictionary",
      "true"
    ).parquet(newTempPath("narrow/dict"))
    df.repartition(3).write.mode("overwrite").option(
      "parquet.enable.dictionary",
      "false"
    ).parquet(newTempPath("narrow/plain"))
    spark.read.parquet(newTempPath("narrow/dict")).createOrReplaceTempView("tn_dict")
    spark.read.parquet(newTempPath("narrow/plain")).createOrReplaceTempView("tn_plain")
  }

  Seq("tn_dict", "tn_plain").foreach { t =>
    test(s"$t: tinyint and smallint columns filter, project and read back as the declared type") {
      checkVectorized(s"SELECT i, t, s FROM $t WHERE t > 50 OR s < -20000", Seq(Filter))
      checkVectorized(s"SELECT t, s, g FROM $t WHERE t IS NOT NULL AND g = cast(3 as tinyint)", Seq(Filter))
      checkVectorized(s"SELECT i, t AS tt, s AS ss, t = s AS eq FROM $t", Seq(Project))
      checkVectorized(
        s"SELECT i FROM $t WHERE t IN (cast(1 as tinyint), cast(-100 as tinyint), cast(127 as tinyint))",
        Seq(Filter)
      )
    }

    test(s"$t: casts into tinyint / smallint wrap in legacy mode, raise under ANSI, null under try_cast") {
      withConf("spark.sql.ansi.enabled" -> "false") {
        checkVectorized(
          s"SELECT i, cast(i AS tinyint) AS ti, cast(l AS tinyint) AS tl, cast(d AS tinyint) AS td, cast(s AS tinyint) AS ts FROM $t",
          Seq(Project)
        )
        checkVectorized(
          s"SELECT i, cast(i AS smallint) AS si, cast(l * 3 AS smallint) AS sl, cast(d * 100 AS smallint) AS sd, cast(t AS smallint) AS st FROM $t",
          Seq(Project)
        )
        checkVectorized(
          s"SELECT i, cast(i % 3 AS tinyint) + cast(g AS tinyint) AS a, t - g AS b, t * cast(2 as tinyint) AS c FROM $t",
          Seq(Project)
        )
      }
      withConf("spark.sql.ansi.enabled" -> "true") {
        // In range: the same values under ANSI.
        checkVectorized(
          s"SELECT i, cast(i % 100 AS tinyint) AS ti, cast(l % 30000 AS smallint) AS sl, cast(d / 2 AS tinyint) AS td FROM $t",
          Seq(Project)
        )
        checkVectorized(s"SELECT i, cast(i % 3 AS tinyint) + g AS a, g - cast(1 as tinyint) AS b FROM $t", Seq(Project))
        checkVectorized(
          s"SELECT i, try_cast(i AS tinyint) AS ti, try_cast(l * 5 AS smallint) AS sl, try_cast(d * 10 AS tinyint) AS td FROM $t",
          Seq(Project)
        )
        // Out of range raises the cast overflow, as Spark.
        val e = intercept[Exception] { spark.sql(s"SELECT cast(i AS tinyint) FROM $t").collect() }
        assert(e.getMessage.contains("CAST_OVERFLOW") || e.getMessage.contains("overflow"), e.getMessage)
        val e2 = intercept[Exception] { spark.sql(s"SELECT t + cast(100 as tinyint) FROM $t WHERE t > 100").collect() }
        assert(e2.getMessage.contains("ARITHMETIC_OVERFLOW") || e2.getMessage.contains("overflow"), e2.getMessage)
      }
    }

    test(s"$t: casts out of tinyint / smallint, and their aggregates") {
      checkVectorized(
        s"SELECT i, cast(t AS int) AS ti, cast(t AS bigint) AS tl, cast(s AS double) AS sd, cast(t AS string) AS tstr, cast(s AS int) + 1 AS si FROM $t",
        Seq(Project)
      )
      checkVectorized(
        s"SELECT g, count(*) AS n, sum(t) AS st, min(t) AS mn, max(s) AS mx, avg(s) AS av FROM $t GROUP BY g",
        Seq(Agg)
      )
      checkVectorized(s"SELECT t, count(*) AS n FROM $t GROUP BY t", Seq(Agg))
      checkVectorized(
        s"SELECT k, min(g) AS mg, max(t) AS mt, sum(cast(s AS bigint)) AS ss FROM $t GROUP BY k",
        Seq(Agg)
      )
    }

    test(s"$t: the ROLLUP shape -- grouping-id casts, a window partitioned by tinyint, sorted by it") {
      // TPC-DS q36/q70/q86: lochierarchy is a sum of two tinyint casts of grouping-id bits; the window
      // ranks within it; the order by is on it.
      checkVectorized(
        s"""SELECT sum(l) AS total, g, k,
           |  grouping(g) + grouping(k) AS lochierarchy,
           |  rank() OVER (PARTITION BY grouping(g) + grouping(k), CASE WHEN grouping(k) = 0 THEN g END ORDER BY sum(l) DESC) AS rank_within_parent
           |FROM $t GROUP BY ROLLUP(g, k) ORDER BY lochierarchy DESC, CASE WHEN lochierarchy = 0 THEN g END, rank_within_parent LIMIT 100""".stripMargin,
        Seq(Agg, Window)
      )
      checkVectorized(
        s"SELECT t, s, i FROM $t ORDER BY t NULLS FIRST, s DESC, i LIMIT 500",
        Seq(classOf[org.apache.spark.sql.vector.VectorTakeOrderedAndProjectExec])
      )
      checkVectorized(
        s"SELECT g, rank() OVER (PARTITION BY g ORDER BY t, i) AS r, lag(t, 1) OVER (PARTITION BY g ORDER BY i) AS prev FROM $t WHERE i < 3000",
        Seq(Window)
      )
    }
  }

  test("tinyint join keys through our exchange and joins") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      checkVectorized(
        "SELECT a.i, a.g, b.n FROM tn_dict a JOIN (SELECT g, count(*) AS n FROM tn_plain GROUP BY g) b ON a.g = b.g WHERE a.i < 2000",
        Seq(Agg)
      )
    }
    checkVectorized(
      "SELECT a.i, a.t, b.k FROM tn_dict a JOIN (SELECT DISTINCT t, k FROM tn_plain WHERE t > 100) b ON a.t = b.t",
      Seq(Agg)
    )
  }
}
