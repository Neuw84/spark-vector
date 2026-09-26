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

import io.vecruntime.spark.test.VectorQuerySuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.joins.{BroadcastNestedLoopJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vector.{
  VectorBroadcastHashJoinExec,
  VectorBroadcastNestedLoopJoinExec,
  VectorHashAggregateExec,
  VectorShuffledHashJoinExec,
  VectorSortMergeJoinExec
}

/**
 * SQL correctness coverage ported from DataFusion Comet's `CometJoinSuite`, adapted to
 * spark-vector's plugin-on/plugin-off comparison model: every join query is run twice on one session
 * with `spark.vector.enabled` toggled, the rows compared, and the right vector join operator asserted
 * in the accelerated plan (a join we don't accelerate asserts fallback with its reason).
 *
 * This is the join slice of the SQL-coverage survey. It complements the hand-written
 * `VectorJoinSuite` and `VectorSortMergeJoinSuite` (which already pin the fused non-equi residual,
 * the null-aware anti join's four regimes, the grace-hash spill, the payload pass-through and the
 * sort-merge rewrite in depth) by adding Comet's data-driven join matrix in the exact shapes its
 * suite checks -- every join type over the two-column `(k, v)` tables Comet builds with
 * `withParquetTable`, build-left and build-right, with and without a join filter, the broadcast
 * nested loop join's inequality / cross / semi / anti / outer forms and the two combinations it
 * refuses, `NOT IN` null-aware anti joins over every null regime, and the sort-merge join over
 * timestamp, composite and nullable keys.
 *
 * The tables mirror Comet's fixtures rather than `TestTables.createMixed`: `a(k, v)` = `(i, i % 5)`
 * and `b(k, v)` = `(i % 10, i + 2)` so keys repeat on both sides (Comet's `tbl_a` / `tbl_b`), plus
 * small null-bearing tables for the anti-join and nested-loop regimes. Only join types and key types
 * `docs/operators.md` pins as supported are asserted vectorized; the combinations it names refused
 * (a full outer nested loop join, an outer nested loop join whose preserved side is broadcast) assert
 * fallback with the documented reason. Where a shape uncovered an engine bug the case is marked
 * `ignore` with a one-line reason and a minimal repro in the PR, per the task's rule not to fix
 * engine code in a coverage change.
 */
class VectorPortedCometJoinSuite extends VectorQuerySuite {

  private val BHJ = classOf[VectorBroadcastHashJoinExec]
  private val SHJ = classOf[VectorShuffledHashJoinExec]
  private val BNLJ = classOf[VectorBroadcastNestedLoopJoinExec]
  private val MergeJoin = classOf[VectorSortMergeJoinExec]

  // Comet's tbl_a / tbl_b: two-column tables whose keys repeat on both sides so every join type
  // exercises multiple-match, no-match and (for b) the many-rows-per-key edge. Written as Parquet
  // and re-read so the scan is columnar.
  override protected def beforeAll(): Unit = {
    super.beforeAll()
    writeInts("a", (0 until 1000).map(i => (i, i % 5)))
    writeInts("b", (0 until 1000).map(i => (i % 10, i + 2)))
    // A larger streamed table with nulls for the broadcast nested loop join inequality forms.
    writeNullable("nl", (0 until 100).map(i => (Some(i), i % 5)) ++ Seq((None, 7), (Some(50), -1)))
    writeNullable("nr", (0 until 10).map(i => (Some(i), i + 5)) ++ Seq[(Option[Int], Int)]((None, 1)))
  }

  private val ktv = StructType(
    Seq(StructField("k", IntegerType, nullable = false), StructField("v", IntegerType, nullable = false))
  )
  private val ktvNull = StructType(
    Seq(StructField("k", IntegerType, nullable = true), StructField("v", IntegerType, nullable = false))
  )

  /** A two-int-column view written as Parquet (columnar scan). */
  private def writeInts(view: String, rows: Seq[(Int, Int)]): Unit = {
    val data = rows.map { case (k, v) => Row(k, v) }.toList
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data, 3), ktv)
    val path = newTempPath(s"pjoin/$view")
    df.write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView(view)
  }

  /** A two-column view whose key is a nullable Integer. */
  private def writeNullable(view: String, rows: Seq[(Option[Int], Int)]): Unit = {
    val data = rows.map { case (k, v) => Row(k.map(Int.box).orNull, v) }.toList
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data, 2), ktvNull)
    val path = newTempPath(s"pjoin/$view")
    df.write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView(view)
  }

  /** A single nullable-int-column view (the all-null side of Comet's full outer join case). */
  private def writeInts1(view: String, col: String, rows: Seq[Option[Int]]): Unit = {
    val schema = StructType(Seq(StructField(col, IntegerType, nullable = true)))
    val data = rows.map(o => Row(o.map(Int.box).orNull)).toList
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data, 2), schema)
    val path = newTempPath(s"pjoin/$view")
    df.write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView(view)
  }

  /** A (name/id, timestamp) view; the timestamp is parsed from an ISO string, nullable where None. */
  private def writeTs(view: String, idCol: String, idType: DataType, rows: Seq[(Any, Option[String])]): Unit = {
    val schema = StructType(
      Seq(StructField(idCol, idType, nullable = false), StructField("ts", StringType, nullable = true))
    )
    val data = rows.map { case (id, ts) => Row(id, ts.orNull) }.toList
    val df = spark
      .createDataFrame(spark.sparkContext.parallelize(data, 2), schema)
      .selectExpr(idCol, "cast(ts as timestamp) as time")
    val path = newTempPath(s"pjoin/$view")
    df.write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView(view)
  }

  // Comet forces the shuffled hash join with these; the sort-merge rewrite mode selects our operator.
  private val ShuffleHash = Seq(
    "spark.sql.autoBroadcastJoinThreshold" -> "-1",
    "spark.sql.join.preferSortMergeJoin" -> "false"
  )
  private val Broadcast = Seq(
    "spark.sql.join.preferSortMergeJoin" -> "false",
    "spark.sql.adaptive.autoBroadcastJoinThreshold" -> "10485760",
    "spark.sql.autoBroadcastJoinThreshold" -> "10485760"
  )

  // ---------------------------------------------------------------------------
  // Broadcast hash join, build left and build right, with and without a filter
  // (Comet's "Broadcast HashJoin without/with join filter"). The build side is
  // hinted; inner and right outer are the shapes Spark broadcasts.
  // ---------------------------------------------------------------------------

  test("broadcast hash join without a join filter, inner and right outer, either build side") {
    withConf(Broadcast: _*) {
      checkVectorized("SELECT /*+ BROADCAST(b) */ a.k, a.v, b.v FROM a JOIN b ON a.v = b.k", Seq(BHJ))
      checkVectorized("SELECT /*+ BROADCAST(b) */ a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k", Seq(BHJ))
      // Build the left (the streamed side is then the right table).
      checkVectorized("SELECT /*+ BROADCAST(a) */ a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k", Seq(BHJ))
    }
  }

  test("broadcast hash join with a non-equi join filter, inner and right outer") {
    withConf(Broadcast: _*) {
      checkVectorized(
        "SELECT /*+ BROADCAST(b) */ a.k, a.v, b.v FROM a JOIN b ON a.v = b.k AND a.k > b.v",
        Seq(BHJ)
      )
      checkVectorized(
        "SELECT /*+ BROADCAST(b) */ a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k AND a.k > b.v",
        Seq(BHJ)
      )
    }
  }

  // ---------------------------------------------------------------------------
  // Shuffled hash join, both build sides, every join type Spark plans for it
  // (Comet's "HashJoin without/with join filter"). preferSortMergeJoin=false is
  // what makes Spark pick the hash join at this size.
  // ---------------------------------------------------------------------------

  test("shuffled hash join without a filter: inner, right, full, semi, anti; both build sides") {
    withConf(ShuffleHash: _*) {
      // Build right (the dimension `b`).
      checkVectorized("SELECT /*+ SHUFFLE_HASH(b) */ a.k, b.v FROM a JOIN b ON a.v = b.k", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(b) */ a.k, b.v FROM a LEFT JOIN b ON a.v = b.k", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(b) */ a.k FROM a LEFT SEMI JOIN b ON a.v = b.k", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(b) */ a.k FROM a LEFT ANTI JOIN b ON a.v = b.k", Seq(SHJ))
      // Build left (the streamed side is then the right table), so right and full outer.
      checkVectorized("SELECT /*+ SHUFFLE_HASH(a) */ a.k, b.v FROM a JOIN b ON a.v = b.k", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(a) */ a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(a) */ a.k, b.v FROM a FULL JOIN b ON a.v = b.k", Seq(SHJ))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(b) */ a.k, b.v FROM a FULL JOIN b ON a.v = b.k", Seq(SHJ))
    }
  }

  test("shuffled hash join with a non-equi filter: inner, right, full outer, semi, anti") {
    withConf(ShuffleHash: _*) {
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(a) */ a.k, b.v FROM a JOIN b ON a.v = b.k AND a.k > b.v",
        Seq(SHJ)
      )
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(a) */ a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k AND a.k > b.v",
        Seq(SHJ)
      )
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(a) */ a.k, b.v FROM a FULL JOIN b ON a.v = b.k AND a.k > b.v",
        Seq(SHJ)
      )
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(b) */ a.k FROM a LEFT SEMI JOIN b ON a.v = b.k AND a.v >= b.k",
        Seq(SHJ)
      )
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(b) */ a.k FROM a LEFT ANTI JOIN b ON a.v = b.k AND a.v >= b.k",
        Seq(SHJ)
      )
    }
  }

  // ---------------------------------------------------------------------------
  // Sort-merge join re-expressed as our shuffled hash join (docs: SortMergeJoin
  // -> VectorShuffledHashJoinExec under sortMergeJoin.mode=hash). Comet's
  // "SortMergeJoin without/with join filter" over every join type.
  // ---------------------------------------------------------------------------

  private val SortMerge = Seq(
    "spark.sql.autoBroadcastJoinThreshold" -> "-1",
    "spark.sql.join.preferSortMergeJoin" -> "true",
    "spark.vector.exec.sortMergeJoin.mode" -> "hash"
  )

  private def checkMerge(sql: String): org.apache.spark.sql.DataFrame = {
    // Spark plans a sort-merge join for the query; re-expressed as our hash join the merge join and
    // its two sorts are gone.
    val planned = withPlugin(enabled = false) { val d = spark.sql(sql); d.collect(); d }
    assert(nodesOf[SortMergeJoinExec](planned).nonEmpty, finalPlan(planned).treeString)
    val df = checkVectorized(sql, Seq(SHJ))
    assert(nodesOf[SortMergeJoinExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  test("sort-merge join without a filter: inner, left, right, full, semi, anti (both directions)") {
    withConf(SortMerge: _*) {
      checkMerge("SELECT a.k, b.v FROM a JOIN b ON a.v = b.k")
      checkMerge("SELECT a.k, b.v FROM a LEFT JOIN b ON a.v = b.k")
      checkMerge("SELECT b.k, a.v FROM b LEFT JOIN a ON a.v = b.k")
      checkMerge("SELECT a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k")
      checkMerge("SELECT b.k, a.v FROM b RIGHT JOIN a ON a.v = b.k")
      checkMerge("SELECT a.k, b.v FROM a FULL JOIN b ON a.v = b.k")
      checkMerge("SELECT a.k FROM a LEFT SEMI JOIN b ON a.v = b.k")
      checkMerge("SELECT a.k FROM a LEFT ANTI JOIN b ON a.v = b.k")
    }
  }

  test("sort-merge join with a non-equi filter: inner, outer, semi, anti") {
    withConf(SortMerge: _*) {
      checkMerge("SELECT a.k, b.v FROM a JOIN b ON a.v = b.k AND a.k > b.v")
      checkMerge("SELECT a.k, b.v FROM a LEFT JOIN b ON a.v = b.k AND a.k > b.v")
      checkMerge("SELECT a.k, b.v FROM a RIGHT JOIN b ON a.v = b.k AND a.k > b.v")
      checkMerge("SELECT a.k, b.v FROM a FULL JOIN b ON a.v = b.k AND a.k > b.v")
      checkMerge("SELECT a.k FROM a LEFT SEMI JOIN b ON a.v = b.k AND a.v >= b.k")
      checkMerge("SELECT a.k FROM a LEFT ANTI JOIN b ON a.v = b.k AND a.v >= b.k")
    }
  }

  // ---------------------------------------------------------------------------
  // A real merge join, ours, over a timestamp / composite / nullable-timestamp
  // key (Comet's "SortMergeJoin with TimestampType key ..."). mode=merge keeps
  // it a merge join (VectorSortMergeJoinExec) instead of the hash rewrite.
  // ---------------------------------------------------------------------------

  private val MergeMode = Seq(
    "spark.sql.autoBroadcastJoinThreshold" -> "-1",
    "spark.sql.join.preferSortMergeJoin" -> "true",
    "spark.vector.exec.sortMergeJoin.mode" -> "merge"
  )

  private def checkTrueMerge(sql: String): org.apache.spark.sql.DataFrame = {
    val df = checkVectorized(sql, Seq(MergeJoin))
    assert(nodesOf[SortMergeJoinExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  test("a real merge join over timestamp keys: inner, the three outer joins, composite and nullable") {
    withConf(MergeMode: _*) {
      // Two timestamp-keyed tables; some instants shared, one side nullable, a composite (string,ts) key.
      writeTs("ts1", "name", StringType, Seq(("a", Some("2019-01-01 11:11:11")), ("b", Some("2020-05-05 05:05:05"))))
      writeTs("ts2", "name", StringType, Seq(("a", Some("2019-01-01 11:11:11")), ("c", Some("2021-07-07 07:07:07"))))
      checkTrueMerge("SELECT ts1.name, ts2.name FROM ts1 JOIN ts2 ON ts1.time = ts2.time")
      for (jt <- Seq("LEFT OUTER", "RIGHT OUTER", "FULL OUTER"))
        checkTrueMerge(s"SELECT ts1.name, ts2.name FROM ts1 $jt JOIN ts2 ON ts1.time = ts2.time")
      // Composite (string, timestamp) key.
      checkTrueMerge("SELECT ts1.name FROM ts1 JOIN ts2 ON ts1.name = ts2.name AND ts1.time = ts2.time")
      // Nullable timestamp key: NULL never matches (inner) and surfaces unmatched (full outer).
      writeTs(
        "tn1",
        "id",
        IntegerType,
        Seq((1, Some("2019-01-01 11:11:11")), (2, None), (3, Some("2020-05-05 05:05:05")))
      )
      writeTs(
        "tn2",
        "id",
        IntegerType,
        Seq((10, Some("2019-01-01 11:11:11")), (20, None), (30, Some("2022-02-02 02:02:02")))
      )
      checkTrueMerge("SELECT tn1.id, tn2.id FROM tn1 JOIN tn2 ON tn1.time = tn2.time")
      checkTrueMerge("SELECT tn1.id, tn2.id FROM tn1 FULL OUTER JOIN tn2 ON tn1.time = tn2.time")
    }
  }

  // ---------------------------------------------------------------------------
  // NOT IN over nullable columns: the null-aware anti join across Comet's null
  // regimes ("BroadcastHashJoin with LeftAnti and NOT IN subquery"). Rows are
  // compared against Spark; the operator is asserted where AQE does not fold it.
  // ---------------------------------------------------------------------------

  test("NOT IN null-aware anti join across the null regimes") {
    withConf(Broadcast: _*) {
      // No null on the right: plain anti-semantics; the streamed rows with a matching key drop.
      writeNullable("naR0", Seq((Some(0), 100), (Some(1), 101), (Some(2), 102), (Some(3), 103), (Some(4), 104)))
      writeInts("naL0", (0 until 10).map(i => (i, i % 5)))
      checkVectorized("SELECT k FROM naL0 WHERE v NOT IN (SELECT k FROM naR0)", Seq(BHJ))
      // Right side contains NULL: null-aware suppresses every left row. The build side is Spark's
      // HashedRelationWithAllNullKeys singleton, and adaptive execution folds the anti join over it
      // to an EmptyRelation before any operator runs (as VectorJoinSuite's null-aware case notes), so
      // the operator is asserted with AQE off; with it on only the (empty) result is compared.
      writeNullable("naR1", Seq((Some(1), 100), (None, 200)))
      writeNullable("naL1", Seq((Some(1), 1), (Some(2), 2), (Some(3), 3)))
      Seq(true, false).foreach { aqe =>
        withConf("spark.sql.adaptive.enabled" -> aqe.toString) {
          val ops = if (aqe) Seq() else Seq(BHJ)
          val allNull = checkVectorized("SELECT k FROM naL1 WHERE v NOT IN (SELECT k FROM naR1)", ops)
          assert(allNull.collect().isEmpty, finalPlan(allNull).treeString)
        }
      }
      // Left side has NULL values: NOT IN filters them (NULL vs anything is unknown).
      writeNullable("naL2", Seq((Some(1), 1), (None, 2), (Some(3), 3)))
      writeNullable("naR2", Seq((Some(2), 100), (Some(4), 200)))
      checkVectorized("SELECT v FROM naL2 WHERE k NOT IN (SELECT k FROM naR2)", Seq(BHJ))
      // Empty subquery: NOT IN against an empty set keeps every left row, the NULL probe included.
      // Like the all-null regime, adaptive execution folds the anti join over an empty build side to a
      // bare scan, so the operator is asserted with AQE off and only the result compared with it on.
      Seq(true, false).foreach { aqe =>
        withConf("spark.sql.adaptive.enabled" -> aqe.toString) {
          val ops = if (aqe) Seq() else Seq(BHJ)
          checkVectorized("SELECT v FROM naL2 WHERE k NOT IN (SELECT k FROM naR2 WHERE k > 1000000)", ops)
        }
      }
    }
  }

  test("LEFT ANTI (non-null-aware) with an explicit build side and nulls on both sides") {
    withConf(Broadcast: _*) {
      writeInts("aaL", (0 until 10).map(i => (i, i % 5)))
      writeNullable("aaR", Seq((Some(0), 100), (Some(1), 101), (Some(2), 102), (Some(3), 103), (Some(4), 104)))
      checkVectorized(
        "SELECT /*+ BROADCAST(aaR) */ aaL.k FROM aaL LEFT ANTI JOIN aaR ON aaL.v = aaR.k",
        Seq(BHJ)
      )
      // NULL keys on both sides: non-null-aware semantics -- a NULL key matches nothing.
      writeNullable("aaL2", Seq((Some(1), 1), (None, 2), (Some(3), 3)))
      writeNullable("aaR2", Seq((Some(1), 100), (None, 200)))
      checkVectorized(
        "SELECT /*+ BROADCAST(aaR2) */ aaL2.v FROM aaL2 LEFT ANTI JOIN aaR2 ON aaL2.k = aaR2.k",
        Seq(BHJ)
      )
    }
  }

  // ---------------------------------------------------------------------------
  // Broadcast nested loop join (Comet's BroadcastNestedLoopJoin ...): inequality,
  // cross, semi, anti, left/right outer with the supported build side; and the
  // two combinations docs/operators.md names refused.
  // ---------------------------------------------------------------------------

  private def checkNested(sql: String): org.apache.spark.sql.DataFrame = {
    val df = checkVectorized(sql, Seq(BNLJ))
    assert(nodesOf[BroadcastNestedLoopJoinExec](df).isEmpty, finalPlan(df).treeString)
    df
  }

  test("broadcast nested loop join: inequality, cross count, left outer/semi/anti, right outer") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      // Inequality inner: a NULL key operand makes `k > k` unknown, so those rows do not contribute.
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k, nr.k FROM nl JOIN nr ON nl.k > nr.k")
      // Cross join with count-only output.
      checkNested("SELECT /*+ BROADCAST(nr) */ count(*) AS n FROM nl, nr")
      // Left outer preserves the NULL-keyed left rows.
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k, nr.k FROM nl LEFT JOIN nr ON nl.k > nr.k")
      // Left semi and left anti (NULL keys never match the predicate).
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k FROM nl LEFT SEMI JOIN nr ON nl.k > nr.k")
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k FROM nl LEFT ANTI JOIN nr ON nl.k > nr.k")
      // Right outer with the broadcast (build) side on the left: the swap path.
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k, nr.k FROM nr RIGHT JOIN nl ON nr.k < nl.k")
      // Cross join with no condition (materialised pairs).
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k, nr.v FROM nl, nr")
    }
  }

  // A boolean-literal join condition now compiles (ConstBoolExpr): `ON true` is every pair, `ON
  // false` matches nothing (each preserved-side row null-padded once for an outer join). Comet's
  // "BroadcastNestedLoopJoin LEFT OUTER without condition" -- the shape that used to fall back with
  // `unsupported literal type boolean` -- is the LEFT OUTER `ON true` case here.
  //
  // `ON true` always keeps a nested loop join in the plan; `ON false` is checked for row-equality
  // only, because Spark's optimizer may prune a `false` join to an empty relation with no join
  // operator (in which case there is nothing for us to accelerate, and nothing that fell back).
  test("broadcast nested loop join: literal true condition (inner, left outer, right outer, semi)") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      // LEFT OUTER, build-right: the case Comet names, previously ignored.
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k, nr.v FROM nl LEFT JOIN nr ON true")
      // INNER `ON true` is the cross product.
      checkNested("SELECT /*+ BROADCAST(nr) */ nl.k, nr.v FROM nl JOIN nr ON true")
      // RIGHT OUTER with the broadcast (build) side on the left: the swap path.
      checkNested("SELECT /*+ BROADCAST(nl) */ nl.k, nr.v FROM nl RIGHT JOIN nr ON true")
      // Left semi `ON true` reduces (in Spark) to an existence check that keeps every left row when
      // nr is non-empty; Spark prunes the join to a bare scan, so no operator is asserted -- only
      // that the vector path agrees on the rows.
      checkVectorized("SELECT /*+ BROADCAST(nr) */ nl.k FROM nl LEFT SEMI JOIN nr ON true", Seq.empty)
    }
  }

  test("broadcast nested loop join: literal false condition produces Spark-equal rows") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      // No operator asserted: Spark may fold a `false` join to EmptyRelation. What matters is that
      // when the join does run (ConstBoolExpr(false) as the residual) the rows match Spark exactly:
      // an inner/semi join is empty, a left outer null-pads every left row, an anti keeps them all.
      checkVectorized("SELECT /*+ BROADCAST(nr) */ nl.k, nr.v FROM nl LEFT JOIN nr ON false", Seq.empty)
      checkVectorized("SELECT /*+ BROADCAST(nr) */ nl.k, nr.v FROM nl JOIN nr ON false", Seq.empty)
      checkVectorized("SELECT /*+ BROADCAST(nl) */ nl.k, nr.v FROM nl RIGHT JOIN nr ON false", Seq.empty)
      checkVectorized("SELECT /*+ BROADCAST(nr) */ nl.k FROM nl LEFT ANTI JOIN nr ON false", Seq.empty)
    }
  }

  // A bare boolean literal is also a valid filter predicate and a projected value now. Spark prunes
  // a `WHERE true`, so no Filter node is asserted; what is checked is that the plugin path agrees
  // with Spark on the rows (and, for the projection, on the constant column).
  test("boolean literal as a filter predicate and a projected column") {
    checkVectorized("SELECT k, v FROM nl WHERE true", Seq.empty)
    checkVectorized("SELECT k, true AS flag, false AS off FROM nl", Seq.empty)
  }

  test("broadcast nested loop joins that fall back: full outer, and a preserved broadcast side") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      // A full outer nested loop join needs a matched bitmap over the shared broadcast side.
      checkFallback(
        "SELECT /*+ BROADCAST(nr) */ nl.k, nr.k FROM nl FULL OUTER JOIN nr ON nl.k > nr.k",
        Seq(BNLJ),
        "full outer nested loop join"
      )
      // A left outer join whose preserved (left) side is the broadcast one.
      checkFallback(
        "SELECT /*+ BROADCAST(nl) */ nl.k, nr.k FROM nl LEFT OUTER JOIN nr ON nl.k > nr.k",
        Seq(BNLJ),
        "preserved side broadcast"
      )
    }
  }

  // ---------------------------------------------------------------------------
  // The full outer join Comet's `full outer join` case checks: an all-null side,
  // outer with a filter on either side, and count grouped by a nullable key.
  // ---------------------------------------------------------------------------

  test("full outer join with an all-null side and a filter on either side") {
    withConf(ShuffleHash: _*) {
      writeInts1("uL", "n", Seq(Some(1), Some(2), Some(3), Some(4)))
      writeInts1("uR", "n", Seq(Some(3), Some(4), Some(5), Some(6)))
      writeInts1("allNulls", "x", Seq(None, None, None))
      checkVectorized("SELECT /*+ SHUFFLE_HASH(uR) */ uL.n, uR.n FROM uL FULL JOIN uR ON uL.n = uR.n", Seq(SHJ))
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(uR) */ uL.n, uR.n FROM uL FULL JOIN uR ON uL.n = uR.n AND uL.n <> 3",
        Seq(SHJ)
      )
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(uR) */ uL.n, uR.n FROM uL FULL JOIN uR ON uL.n = uR.n AND uR.n <> 3",
        Seq(SHJ)
      )
      // count grouped by a nullable key of the full outer join (the all-null side never matches).
      checkVectorized(
        "SELECT /*+ SHUFFLE_HASH(uR) */ l.x, count(*) AS n FROM allNulls l FULL OUTER JOIN uR r ON l.x = r.n GROUP BY l.x",
        Seq(SHJ, classOf[VectorHashAggregateExec])
      )
    }
  }
}
