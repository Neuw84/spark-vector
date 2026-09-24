/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.spark.ui

import org.apache.spark.sql.catalyst.expressions.{Ascending, AttributeReference, SortOrder}
import org.apache.spark.sql.comet.{FakeCometColumnarExchangeExec, FakeCometScanExec, FakeCometShuffleExchangeExec}
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlanInfo}
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.vector.VectorSortExec
import org.apache.spark.sql.vector.ui.{Engine, PlanAcceleration}
import org.scalatest.funsuite.AnyFunSuite

/**
 * Classification rules that do not need a SparkSession: Comet operators (matched by package, so
 * they can be exercised without Comet installed) and the name-based path used for running queries.
 */
class PlanAccelerationSuite extends AnyFunSuite {

  private val attrs = Seq(AttributeReference("a", LongType)())

  test("operators in Comet's package are coloured as Comet") {
    val plan = PlanAcceleration.fromPlan(FakeCometScanExec(attrs))
    assert(plan.nodes.map(_.engine) === Seq(Engine.Comet))
    // A Comet scan is a real acceleration, not plumbing, so a plan made of one qualifies.
    assert(plan.fullyAccelerated)
  }

  test("a Comet exchange over a Comet scan is fully accelerated") {
    val plan = PlanAcceleration.fromPlan(
      FakeCometShuffleExchangeExec(FakeCometScanExec(attrs))
    )
    assert(plan.countBy(Engine.Comet) === 2)
    assert(plan.countBy(Engine.Spark) === 0)
    assert(plan.fullyAccelerated)
  }

  test("a row transition is plumbing and does not make a plan accelerated on its own") {
    val plan = PlanAcceleration.fromPlan(ColumnarToRowExec(FakeCometScanExec(attrs)))
    assert(plan.countBy(Engine.Transition) === 1)
    assert(plan.countBy(Engine.Comet) === 1)
    assert(plan.operatorCount === 1, "the transition must not count as an operator")
    assert(plan.fullyAccelerated)
  }

  test("the name-based path recognises our operators, Comet's and the bridge") {
    def info(name: String, children: SparkPlanInfo*): SparkPlanInfo =
      new SparkPlanInfo(name, name, children.toSeq, Map.empty, Nil)

    val plan = PlanAcceleration.fromInfo(
      info(
        "CometShuffleExchange",
        info("VectorToComet", info("VectorHashAggregate", info("VectorFilter", info("Scan parquet"))))
      )
    )

    assert(plan.nodes.map(n => (n.name, n.engine)) === Seq(
      ("CometShuffleExchange", Engine.Comet),
      ("VectorToComet", Engine.Bridge),
      ("VectorHashAggregate", Engine.Vector),
      ("VectorFilter", Engine.Vector),
      ("Scan parquet", Engine.ColumnarSource)
    ))
    assert(plan.fullyAccelerated)
  }

  test("wrappers are unwrapped so the graph shows real operators") {
    def info(name: String, children: SparkPlanInfo*): SparkPlanInfo =
      new SparkPlanInfo(name, name, children.toSeq, Map.empty, Nil)

    val plan = PlanAcceleration.fromInfo(
      info(
        "AdaptiveSparkPlan",
        info("WholeStageCodegen (1)", info("Filter", info("InputAdapter", info("Scan parquet"))))
      )
    )

    assert(plan.nodes.map(_.name) === Seq("Filter", "Scan parquet"))
    assert(!plan.fullyAccelerated, "a Spark filter is a missed operator")
  }

  test("edges point from child to parent") {
    val plan = PlanAcceleration.fromPlan(
      FakeCometShuffleExchangeExec(FakeCometScanExec(attrs))
    )
    // Root is id 0, its child id 1, and the edge runs child -> parent.
    assert(plan.edges === Seq(org.apache.spark.sql.vector.ui.PlanEdge(1, 0)))
    assert(plan.toDotFile.contains("1->0"))
  }

  test("AQE's shuffle read and reuse markers are plumbing, not transitions") {
    def info(name: String, children: SparkPlanInfo*): SparkPlanInfo =
      new SparkPlanInfo(name, name, children.toSeq, Map.empty, Nil)
    val plan = PlanAcceleration.fromInfo(
      info("ColumnarToRow", info("VectorSort", info("AQEShuffleRead", info("CometExchange", info("ReusedExchange")))))
    )
    assert(plan.nodes.map(n => (n.name, n.engine)) === Seq(
      ("ColumnarToRow", Engine.Transition),
      ("VectorSort", Engine.Vector),
      ("AQEShuffleRead", Engine.ShuffleRead),
      ("CometExchange", Engine.Comet),
      ("ReusedExchange", Engine.ShuffleRead)
    ))
    // The reader does not count against the plan, and only the ColumnarToRow is a transition.
    assert(plan.countBy(Engine.Transition) === 1)
    assert(plan.operatorCount === 2, "VectorSort and CometExchange")
    assert(plan.fullyAccelerated)
  }

  test("a Comet JVM shuffle over one of our operators is flagged as a row round-trip") {
    val ours = VectorSortExec(Seq(SortOrder(attrs.head, Ascending)), global = false, FakeCometScanExec(attrs))
    val flagged = PlanAcceleration.fromPlan(FakeCometColumnarExchangeExec(ours))
    val exchange = flagged.nodes.head
    assert(exchange.engine === Engine.Comet, "it is still Comet's operator")
    assert(exchange.detail.startsWith(PlanAcceleration.RowRoundTripNote))
    // Over a Comet child there is nothing to flag.
    val plain = PlanAcceleration.fromPlan(FakeCometColumnarExchangeExec(FakeCometScanExec(attrs)))
    assert(!plain.nodes.head.detail.contains("rows"))
  }
}
