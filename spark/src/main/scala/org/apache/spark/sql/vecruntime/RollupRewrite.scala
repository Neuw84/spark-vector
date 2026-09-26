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
package org.apache.spark.sql.vector

import org.apache.spark.sql.catalyst.expressions.{
  Attribute,
  AttributeReference,
  AttributeSet,
  Expression,
  ExprId,
  Literal
}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial, PartialMerge}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.aggregate.HashAggregateExec

/**
 * `ROLLUP` without hashing every input row once per grouping set (#383).
 *
 * Spark plans `GROUP BY ... WITH ROLLUP` as an `Expand` -- one copy of each input row per grouping
 * set, the rolled-up keys nulled, a grouping id -- under an ordinary partial aggregate, so the
 * partial hashes N times the input on the full key tuple (q67 at 1 TB: 540 M rows x 9 sets = 4.86 B
 * rows into the aggregate, all of the query's cost). A rollup's sets are nested, and every
 * aggregate we run is decomposable in buffer form, so the same answer comes from a chain: the input
 * aggregated once on the finest set, and each coarser set merged from the groups of the set before
 * it -- N + G1 + G2 + ... rows hashed instead of N x sets. [[VectorRollupExec]] runs the chain in
 * one pass and emits every level under the original partial's output attributes, so the exchange
 * and the final aggregate above are untouched.
 *
 * Applies when the partial aggregate's child is our Expand, the key slots are column references or
 * null literals, the aggregate-input slots are the same column in every projection (Spark's
 * distinct rewrite nulls inputs per projection: not applicable), and the sets are nested (a CUBE or
 * general GROUPING SETS keeps Spark's plan). Every function must be in Partial mode.
 */
object RollupRewrite {

  /** The rewritten operator, or the original when the shape does not fit or a level does not plan. */
  def apply(agg: VectorHashAggregateExec, strictFloatingPoint: Boolean): SparkPlan = agg.child match {
    case expand: VectorExpandExec if applicable(agg, expand) => rewrite(agg, expand, strictFloatingPoint).getOrElse(agg)
    case _ => agg
  }

  private def applicable(agg: VectorHashAggregateExec, expand: VectorExpandExec): Boolean =
    agg.aggregateExpressions.nonEmpty &&
      agg.aggregateExpressions.forall(_.mode == Partial) &&
      expand.projections.size > 1 &&
      agg.groupingExpressions.forall(_.isInstanceOf[Attribute])

  private def rewrite(agg: VectorHashAggregateExec, expand: VectorExpandExec, strict: Boolean): Option[SparkPlan] = {
    val output = expand.output
    val slotOf: Map[ExprId, Int] = output.map(_.exprId).zipWithIndex.toMap
    val groupingAttrs = agg.groupingExpressions.map(_.asInstanceOf[Attribute])
    val groupingSlots: Seq[Int] = groupingAttrs.map(a => slotOf.getOrElse(a.exprId, return None))
    val keySlots = groupingSlots.toSet

    // Every slot an aggregate expression reads must be the same child column in every projection.
    val inputAttrs = AttributeSet(agg.aggregateExpressions.flatMap(_.references))
    val passThrough: Map[Int, Attribute] = inputAttrs.toSeq.map { attr =>
      val slot = slotOf.getOrElse(attr.exprId, return None)
      val exprs = expand.projections.map(_(slot))
      exprs.head match {
        case a: Attribute if exprs.forall { case b: Attribute => b.exprId == a.exprId; case _ => false } => slot -> a
        case _ => return None
      }
    }.toMap
    if (keySlots.intersect(passThrough.keySet).nonEmpty) return None

    // Key slots: the same child column wherever they are not a literal (a null, or the grouping id).
    val childAttrs = AttributeSet(expand.child.output)
    keySlots.foreach { slot =>
      val refs = expand.projections.map(_(slot)).collect { case a: Attribute => a }
      val ok =
        expand.projections.forall(p => p(slot) match { case _: Attribute | _: Literal => true; case _ => false }) &&
          refs.map(_.exprId).distinct.size <= 1 && refs.forall(childAttrs.contains)
      if (!ok) return None
    }
    // The grouping set of each projection, as child columns; nested means each is a subset of the one before.
    val setOf: Seq[Set[ExprId]] = expand.projections.map(p =>
      groupingSlots.flatMap(slot => p(slot) match { case a: Attribute => Some(a.exprId); case _ => None }).toSet
    )
    val order: Seq[Int] = setOf.indices.sortBy(j => -setOf(j).size)
    val nested = order.sliding(2).forall { case Seq(a, b) => setOf(b).subsetOf(setOf(a)); case _ => true }
    if (!nested || setOf(order.head).isEmpty) return None
    val keyByExprId: Map[ExprId, Attribute] = expand.child.output.map(a => a.exprId -> a).toMap
    def keysOf(j: Int): Seq[Attribute] = expand.child.output.filter(a => setOf(j).contains(a.exprId))

    // Level 0: the partial aggregate on the finest set over the Expand's child, its functions reading
    // the child columns directly (the Expand passed them through under their own attributes).
    val inputRewrite: Map[ExprId, Attribute] = passThrough.map { case (slot, childAttr) =>
      output(slot).exprId -> childAttr
    }
    if (!inputRewrite.values.forall(childAttrs.contains)) return None
    val partialAggs = agg.aggregateExpressions.map { ae =>
      ae.transform { case a: AttributeReference => inputRewrite.getOrElse(a.exprId, a) }.asInstanceOf[
        AggregateExpression
      ]
    }
    // The levels' functions may be new objects (their inputs were rewritten), with buffer attributes of
    // their own; the projections map those onto the ORIGINAL buffer attributes, which the final
    // aggregate above reads by identity.
    val levelBuffers: Seq[Attribute] = partialAggs.flatMap(_.aggregateFunction.inputAggBufferAttributes)
    val buffers: Seq[Attribute] = agg.aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes)
    if (levelBuffers.length != buffers.length || levelBuffers.map(_.dataType) != buffers.map(_.dataType)) return None
    val mergeAggs = partialAggs.map(ae => ae.copy(mode = PartialMerge, filter = None))

    val levels = Seq.newBuilder[VectorHashAggregateExec]
    var below: SparkPlan = expand.child
    order.zipWithIndex.foreach { case (j, k) =>
      val keys = keysOf(j)
      val spark = HashAggregateExec(
        requiredChildDistributionExpressions = None,
        isStreaming = false,
        numShufflePartitions = None,
        groupingExpressions = keys,
        aggregateExpressions = if (k == 0) partialAggs else mergeAggs,
        aggregateAttributes = agg.aggregateAttributes,
        initialInputBufferOffset = if (k == 0) 0 else keys.length,
        resultExpressions = keys ++ levelBuffers,
        child = below
      )
      val planned = VectorAggregatePlanner.plan(spark, finalEnabled = true, strict).toOption.getOrElse(return None)
      levels += planned
      below = planned
    }
    // Each level's projection: the original Expand's key slots (over that level's keys) and the buffers.
    val projections: Seq[Seq[Expression]] = order.map { j =>
      val p = expand.projections(j)
      groupingSlots.map(slot =>
        p(slot) match { case a: Attribute => keyByExprId(a.exprId); case e => e }
      ) ++ levelBuffers
    }
    val out = groupingAttrs ++ buffers
    if (out.map(_.exprId) != agg.output.map(_.exprId)) return None
    Some(VectorRollupExec(levels.result(), projections, out, expand.child))
  }
}
