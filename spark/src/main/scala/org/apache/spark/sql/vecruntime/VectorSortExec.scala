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
package org.apache.spark.sql.vecruntime

import io.vecruntime.spark.VectorConf
import io.vecruntime.spark.adapter.TypeMapping
import io.vecruntime.spark.expr.{ColumnRef, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, NullsFirst, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{
  Distribution,
  OrderedDistribution,
  Partitioning,
  UnspecifiedDistribution
}
import org.apache.spark.sql.execution.{SortExec, SparkPlan}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Columnar replacement for SortExec.
 *
 * A blocking operator: every batch of the partition is appended to one [[ColumnBuilder]] per
 * output attribute and per computed key (Spark's columnar contract lets the producer reuse a batch
 * as soon as the next one is requested). Every `spark.vector.sort.runRows` rows the builders are
 * sealed into a run -- its columns and the permutation [[SortKernels.sortIndices]] computed over
 * them -- so the sort's scratch is bounded by the run, not the partition. One run emits its rows
 * through the permutation; several are k-way merged by [[RunMerge]] in the same total order, ties by
 * run then position (stable), and gathered run by run. Output batches are 4096 rows of Arrow
 * vectors; a dictionary-encoded string column is decoded once when its run is sealed (#285). Sort
 * keys that are plain column references reuse the output column rather than being copied twice.
 *
 * Same distribution contract as SortExec: a global sort requires the range partitioning the
 * exchange below already provides, a local sort accepts anything. The operator is only planned
 * over a columnar child (Comet's columnar shuffle, or a spark-vector operator for a local sort);
 * over Spark's row shuffle the rule leaves SortExec in place, since converting rows to columns just
 * to sort them buys nothing.
 *
 * Memory is bounded by `spark.vector.sort.spillBytes` (#416): runs past the budget are written to
 * local disk in sorted order and merged from there, so a partition of any size sorts in bounded
 * memory -- which is what lets the merge join above it take large inputs.
 */
case class VectorSortExec(sortOrder: Seq[SortOrder], global: Boolean, child: SparkPlan) extends VectorExec {

  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = sortOrder
  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def requiredChildDistribution: Seq[Distribution] =
    if (global) OrderedDistribution(sortOrder) :: Nil else UnspecifiedDistribution :: Nil

  @transient private lazy val compiledKeys: Array[VectorExpr] = sortOrder.map { o =>
    VectorSortPlanner.compileKey(o, child.output) match {
      case Right(k) => k
      case Left(reason) => throw new IllegalStateException(s"cannot vectorize sort key ${o.sql}: $reason")
    }
  }.toArray

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val keys = compiledKeys
    val ascending = sortOrder.map(_.direction == Ascending).toArray
    val nullsFirst = sortOrder.map(_.nullOrdering == NullsFirst).toArray
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    val runRows = VectorConf.sortRunRows(conf)
    val spillBytes = VectorConf.sortSpillBytes(conf, sparkContext.getConf)
    child.executeColumnar().mapPartitionsInternal { iter =>
      new VectorSortIterator(
        iter,
        keys,
        ascending,
        nullsFirst,
        outputAttrs,
        m,
        runRows = runRows,
        spillBytes = spillBytes
      )
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)

  override def verboseStringWithOperatorId(): String = {
    s"""$formattedNodeName
       |Sort: ${sortOrder.map(_.sql).mkString(", ")}
       |Global: $global
       |Output: ${output.map(_.name).mkString(", ")}
       |""".stripMargin
  }
}

/** Planning-time checks shared by the rule and the operator. */
object VectorSortPlanner {

  /**
   * Any compiled non-literal expression of a supported type can be a sort key, and so can a bare
   * wide decimal column (a DECIMAL128 lane the sort kernel orders limb by limb, #257).
   */
  def compileKey(order: SortOrder, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compileLaneColumn(order.child, input).flatMap {
      case _: LiteralExpr => Left("literal sort key")
      case k if !TypeMapping.hasLane(order.child.dataType) =>
        Left(s"sort key type ${order.child.dataType.simpleString} not supported")
      case k => Right(k)
    }

  /** Attempts to convert a Spark SortExec; Left explains the fallback. */
  def plan(s: SortExec): Either[String, VectorSortExec] = {
    val failures = s.sortOrder.flatMap(o => compileKey(o, s.child.output).left.toOption.map(r => s"${o.sql}: $r"))
    if (s.sortOrder.isEmpty) Left("sort without keys")
    else if (failures.nonEmpty) Left(failures.mkString("; "))
    else Right(VectorSortExec(s.sortOrder, s.global, s.child))
  }
}
