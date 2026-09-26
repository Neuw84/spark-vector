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
import org.apache.spark.sql.vecruntime.VectorFallback

/**
 * #402: the merge join's size gate on q47's shape -- a windowed CTE self-joined on rn-1 / rn+1 under
 * ORDER BY ... LIMIT with broadcasts off. The second self-join's input is the first join, which has no
 * stage of its own; measured by the logical product it read as 11.7 GB for a few thousand rows and
 * the join was left to Spark, taking the Project and the top-N above it along.
 */
class VectorMergeJoinGateSuite extends VectorQuerySuite {
  override def beforeAll(): Unit = {
    super.beforeAll()
    TestTables.createMixed(spark, newTempPath("diag402/t"))
    val tk = newTempPath("diag402/tk")
    spark.sql(
      "SELECT *, cast(i % 50 as int) AS i50, cast(i % 7 as int) AS m FROM t"
    ).write.mode("overwrite").parquet(tk)
    spark.read.parquet(tk).createOrReplaceTempView("tk")
  }

  test("both self-joins of a windowed row set under a top-N are our merge joins (#402)") {
    withConf("spark.sql.autoBroadcastJoinThreshold" -> "-1", "spark.sql.join.preferSortMergeJoin" -> "true") {
      val sql =
        """WITH v1 AS (
          |  SELECT i50, m, sum(d) AS sum_d,
          |         avg(sum(d)) OVER (PARTITION BY i50) AS avg_d,
          |         rank() OVER (PARTITION BY i50 ORDER BY m) AS rn
          |  FROM tk GROUP BY i50, m),
          |v2 AS (
          |  SELECT v1.i50, v1.m, v1.avg_d, v1.sum_d, v1_lag.sum_d AS psum, v1_lead.sum_d AS nsum
          |  FROM v1, v1 v1_lag, v1 v1_lead
          |  WHERE v1.i50 = v1_lag.i50 AND v1.i50 = v1_lead.i50 AND v1.rn = v1_lag.rn + 1 AND v1.rn = v1_lead.rn - 1)
          |SELECT * FROM v2 WHERE avg_d > 0 ORDER BY sum_d - avg_d, i50, m LIMIT 100""".stripMargin
      val df = checkVectorized(
        sql,
        Seq(
          classOf[org.apache.spark.sql.vecruntime.VectorSortMergeJoinExec],
          classOf[org.apache.spark.sql.vecruntime.VectorTakeOrderedAndProjectExec]
        )
      )
      val reasons = VectorFallback.reasons(finalPlan(df)).map(_._2)
      assert(nodesOf[org.apache.spark.sql.execution.joins.SortMergeJoinExec](df).isEmpty, finalPlan(df).treeString)
      assert(
        nodesOf[org.apache.spark.sql.vecruntime.VectorSortMergeJoinExec](df).length === 2,
        finalPlan(df).treeString
      )
    }
  }
}
