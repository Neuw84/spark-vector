/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.vecruntime.spark

import io.vecruntime.spark.test.{TestTables, VectorQuerySuite}
import org.apache.spark.sql.execution.{CoalesceExec, RowToColumnarExec, UnionExec}
import org.apache.spark.sql.vector.{
  VectorCoalesceExec,
  VectorFallback,
  VectorFilterExec,
  VectorHashAggregateExec,
  VectorUnionExec
}

/** The structural operators: union and coalesce keep a columnar chain whole without computing anything. */
class VectorUnionSuite extends VectorQuerySuite {

  private val Union = classOf[VectorUnionExec]
  private val Coalesce = classOf[VectorCoalesceExec]
  private val Filter = classOf[VectorFilterExec]
  private val Agg = classOf[VectorHashAggregateExec]

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("union/t"))
    TestTables.createLineitem(spark, newTempPath("union/lineitem"))
  }

  test("UNION ALL of two scans feeding a filter and an aggregate") {
    val df = checkVectorized(
      "SELECT s, count(*) AS c, sum(l) AS sl FROM (SELECT i, l, s FROM t UNION ALL SELECT i, l, s FROM t WHERE i < 5000) WHERE i >= 100 GROUP BY s",
      Seq(Union, Filter, Agg)
    )
    assert(nodesOf[UnionExec](df).isEmpty, "Spark's UnionExec should be gone")
    checkVectorized(
      "SELECT count(*), sum(i), min(d), max(dt) FROM (SELECT * FROM t UNION ALL SELECT * FROM t)",
      Seq(Union, Agg)
    )
    checkVectorized(
      "SELECT i, s FROM (SELECT i, s FROM t WHERE s = 's1' UNION ALL SELECT i, s FROM t WHERE s = 's2') WHERE i > 10",
      Seq(Union, Filter)
    )
    // Three children, and a union under a local sort.
    checkVectorized(
      "SELECT count(*) FROM (SELECT i FROM t UNION ALL SELECT i FROM t UNION ALL SELECT i FROM t WHERE i < 7)",
      Seq(Union, Agg)
    )
    checkVectorized(
      "SELECT count(*) FROM (SELECT l_orderkey FROM lineitem WHERE l_quantity > 40 UNION ALL SELECT l_orderkey FROM lineitem WHERE l_discount = 0.0)",
      Seq(Union, Filter, Agg)
    )
  }

  test("union with mismatched nullability and coerced types") {
    // l is nullable, cast(i) is not: the output is nullable and nulls survive the union.
    val df = checkVectorized(
      "SELECT v, count(*) AS c FROM (SELECT l AS v FROM t UNION ALL SELECT CAST(i AS BIGINT) AS v FROM t) GROUP BY v IS NULL, v",
      Seq(Union, Agg)
    )
    assert(nodesOf[VectorUnionExec](df).head.output.head.nullable)
    checkVectorized(
      "SELECT count(*) FROM (SELECT l AS v FROM t UNION ALL SELECT CAST(i AS BIGINT) FROM t) WHERE v IS NULL",
      Seq(Union, Filter, Agg)
    )
    // int UNION ALL bigint: Spark widens the int side with a cast in a project.
    checkVectorized(
      "SELECT sum(v) FROM (SELECT i AS v FROM t UNION ALL SELECT l AS v FROM t WHERE l IS NOT NULL)",
      Seq(Union, Agg)
    )
  }

  test("union with a row child: the VALUES side goes through RowToColumnar, the chain stays columnar") {
    val df = checkVectorized(
      "SELECT count(*), sum(v) FROM (SELECT i AS v FROM t UNION ALL SELECT * FROM VALUES (1), (2), (3) AS x(v))",
      Seq(Union, Agg)
    )
    assert(nodesOf[RowToColumnarExec](df).nonEmpty, finalPlan(df).treeString)
    checkVectorized(
      "SELECT v, count(*) FROM (SELECT s AS v FROM t UNION ALL SELECT * FROM VALUES ('x'), ('s1'), (NULL) AS x(v)) GROUP BY v",
      Seq(Union, Agg)
    )
    // No columnar child at all: nothing to keep, Spark's union stays.
    val rows = withPlugin(enabled = true) {
      val d = spark.sql(
        "SELECT sum(v) FROM (SELECT * FROM VALUES (1), (2) AS a(v) UNION ALL SELECT * FROM VALUES (3) AS b(v))"
      ); d.collect(); d
    }
    assert(nodesOf[VectorUnionExec](rows).isEmpty)
    val reasons = VectorFallback.reasons(finalPlan(rows)).map(_._2)
    assert(reasons.exists(_.contains("no columnar child")), reasons.mkString("; "))
  }

  test("union of equally partitioned children keeps Spark's partitioning contract (TPC-DS q33 shape)") {
    import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioningLike, UnknownPartitioning}
    import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike
    // Every child is a Final aggregate hash-partitioned by s, so Spark's union reports that
    // partitioning and EnsureRequirements plans no shuffle above it: the outer aggregate runs
    // directly over the union, which must keep each key in one partition. With a plain
    // concatenation every key came out once per child (#128: q33, q56, q60).
    val channel = (k: Int) => s"SELECT s, sum(l) AS total FROM t WHERE i % 3 = $k GROUP BY s"
    val sql =
      s"SELECT s, sum(total) AS total, count(*) AS parts FROM (${channel(0)} UNION ALL ${channel(1)} UNION ALL ${channel(2)}) u GROUP BY s"
    val expected = withPlugin(enabled = false)(spark.sql(sql).collect())
    val df = withPlugin(enabled = true) { val d = spark.sql(sql); d.collect(); d }
    assertRowsEqual(expected, df.collect(), 1e-9, sql)
    val union = nodesOf[VectorUnionExec](df).head
    assert(union.outputPartitioning.isInstanceOf[HashPartitioningLike], union.outputPartitioning.toString)
    // The three inner shuffles are the only exchanges: none was planned above the union.
    assert(nodesOf[ShuffleExchangeLike](df).size === 3, finalPlan(df).treeString)
    assert(df.collect().forall(_.getLong(2) <= 3) && df.collect().map(_.getString(0)).distinct.length === df.count())
    // Children partitioned on different keys: Spark's union reports nothing, and a shuffle is planned.
    val mixed =
      "SELECT k, sum(total) FROM (SELECT s AS k, sum(l) AS total FROM t GROUP BY s UNION ALL SELECT CAST(i % 7 AS STRING) AS k, sum(l) FROM t GROUP BY i % 7) u GROUP BY k"
    val m = checkVectorized(mixed, Seq(Union, Agg))
    assert(nodesOf[VectorUnionExec](m).head.outputPartitioning.isInstanceOf[UnknownPartitioning])
    assert(nodesOf[ShuffleExchangeLike](m).size === 3, finalPlan(m).treeString)
  }

  test("UNION (distinct) is a keys-only aggregate over our union") {
    // The aggregate has no functions: a keys-only aggregate, ours in both stages, over our union.
    val df = checkVectorized("SELECT s FROM (SELECT s FROM t UNION SELECT s FROM t WHERE i < 100)", Seq(Union))
    assert(nodesOf[UnionExec](df).isEmpty)
    assert(nodesOf[org.apache.spark.sql.execution.aggregate.HashAggregateExec](df).isEmpty, finalPlan(df).treeString)
  }

  test("coalesce below an aggregate and at the top of a query") {
    // The DataFrame API is the way to put a coalesce under an aggregate; compare with the plugin off.
    def query() = spark.table("t").filter("i > 100").coalesce(1).groupBy("s").agg(
      org.apache.spark.sql.functions.count("*").as("c"),
      org.apache.spark.sql.functions.sum("l").as("sl")
    )
    val expected = withPlugin(enabled = false)(query().collect())
    val df = withPlugin(enabled = true) { val d = query(); d.collect(); d }
    assert(df.collect().map(_.toString).sorted.toSeq === expected.map(_.toString).sorted.toSeq)
    assert(nodesOf[VectorCoalesceExec](df).nonEmpty, finalPlan(df).treeString)
    assert(nodesOf[CoalesceExec](df).isEmpty)
    assert(nodesOf[VectorHashAggregateExec](df).nonEmpty)
    assert(df.rdd.getNumPartitions === 1)
    // At the top: every row survives a coalesce into two partitions.
    val top = spark.table("t").filter("i < 1000").coalesce(2)
    val topExpected = withPlugin(enabled = false)(top.collect())
    val topDf = withPlugin(enabled = true) { val d = spark.table("t").filter("i < 1000").coalesce(2); d.collect(); d }
    assert(topDf.collect().map(_.getInt(0)).sorted.toSeq === topExpected.map(_.getInt(0)).sorted.toSeq)
    assert(nodesOf[VectorCoalesceExec](topDf).nonEmpty)
    assert(topDf.rdd.getNumPartitions === 2)
    // A coalesce below a union below an aggregate.
    val mixed = spark.table("t").coalesce(1).union(spark.table("t").filter("i < 10")).groupBy("s").count()
    val mixedExpected = withPlugin(enabled = false)(mixed.collect())
    val mixedDf = withPlugin(enabled = true) {
      val d = spark.table("t").coalesce(1).union(spark.table("t").filter("i < 10")).groupBy("s").count(); d.collect(); d
    }
    assert(mixedDf.collect().map(_.toString).sorted.toSeq === mixedExpected.map(_.toString).sorted.toSeq)
    assert(
      nodesOf[VectorCoalesceExec](mixedDf).nonEmpty && nodesOf[VectorUnionExec](mixedDf).nonEmpty,
      finalPlan(mixedDf).treeString
    )
  }

  test("union and coalesce can be disabled") {
    withConf(VectorConf.UnionEnabled -> "false") {
      val df = withPlugin(enabled = true) {
        val d = spark.sql("SELECT count(*) FROM (SELECT i FROM t UNION ALL SELECT i FROM t)"); d.collect(); d
      }
      assert(nodesOf[VectorUnionExec](df).isEmpty)
      assert(nodesOf[UnionExec](df).nonEmpty)
    }
    withConf(VectorConf.CoalesceEnabled -> "false") {
      val df = withPlugin(enabled = true) { val d = spark.table("t").coalesce(1); d.collect(); d }
      assert(nodesOf[VectorCoalesceExec](df).isEmpty)
      assert(nodesOf[CoalesceExec](df).nonEmpty)
    }
  }
}
