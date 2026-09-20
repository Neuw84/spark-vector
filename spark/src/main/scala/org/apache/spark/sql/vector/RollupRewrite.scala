package org.apache.spark.sql.vector

import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference, AttributeSet, Expression, Literal, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial, PartialMerge}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.aggregate.HashAggregateExec

/**
 * `ROLLUP`, `CUBE` and `GROUPING SETS` without hashing every input row once per grouping set (#383).
 *
 * Spark plans them as an `Expand` -- one copy of each input row per grouping set, the rolled-up keys
 * nulled, a grouping id -- under an ordinary partial aggregate, so the partial hashes N times the
 * input on the full key tuple (q67 at 1 TB: 540 M rows x 9 sets = 4.86 B rows into the aggregate, all
 * of the query's cost). Every aggregate we run is decomposable in buffer form, so the same answer
 * comes from aggregating the input ONCE on the finest grouping -- the union of every set's keys --
 * expanding those partial rows per grouping set, and merging the copies:
 *
 * {{{
 *   Partial(keys, aggs)                 Partial-merge(keys, aggs)          <- the same output as before
 *     Expand(sets)              =>        Expand(sets over partial rows)
 *       child                               Partial(finest keys, aggs)
 *                                             child
 * }}}
 *
 * Rows hashed: N + (finest groups x sets) instead of N x sets. The rewrite applies when the partial
 * aggregate's child is our Expand whose key slots are column references or null literals and whose
 * aggregate-input slots are the same column in every projection (Spark's distinct rewrite nulls
 * inputs per projection: not applicable, it keeps its plan); every function must be in Partial mode.
 * The exchange and the final aggregate above are untouched: the output attributes are the same.
 */
object RollupRewrite {

  /** The rewritten operator, or the original when the shape does not fit or a stage does not plan. */
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
    val slotOf: Map[org.apache.spark.sql.catalyst.expressions.ExprId, Int] = output.map(_.exprId).zipWithIndex.toMap
    val groupingAttrs = agg.groupingExpressions.map(_.asInstanceOf[Attribute])
    val groupingSlots = groupingAttrs.map(a => slotOf.get(a.exprId))
    if (groupingSlots.exists(_.isEmpty)) return None
    val keySlots = groupingSlots.flatten.toSet

    // Every slot an aggregate expression reads must be the same child column in every projection.
    val inputAttrs = AttributeSet(agg.aggregateExpressions.flatMap(_.references))
    val inputSlots = inputAttrs.toSeq.map(a => slotOf.get(a.exprId))
    if (inputSlots.exists(_.isEmpty)) return None
    val passThrough: Map[Int, Attribute] = inputSlots.flatten.map { slot =>
      val exprs = expand.projections.map(_(slot))
      exprs.head match {
        case a: Attribute if exprs.forall { case b: Attribute => b.exprId == a.exprId; case _ => false } => slot -> a
        case _ => return None
      }
    }.toMap
    if (keySlots.intersect(passThrough.keySet).nonEmpty) return None

    // Key slots: a column reference (the same child column wherever it is not nulled) or a literal.
    val keySource: Map[Int, Option[Attribute]] = keySlots.toSeq.map { slot =>
      val refs = expand.projections.map(_(slot)).collect { case a: Attribute => a }
      val ok = expand.projections.forall(p => p(slot) match { case _: Attribute | _: Literal => true; case _ => false }) &&
        refs.map(_.exprId).distinct.size <= 1
      if (!ok) return None
      slot -> refs.headOption
    }.toMap
    val finestKeys: Seq[Attribute] = keySource.values.flatten.toSeq.distinct.sortBy(a => expand.child.output.indexWhere(_.exprId == a.exprId))
    if (finestKeys.isEmpty) return None

    // Stage 1: the partial aggregate on the finest grouping, over the Expand's child. Its functions
    // read the child columns directly (the Expand passed them through under their own attributes).
    val childAttrs = AttributeSet(expand.child.output)
    val inputRewrite: Map[org.apache.spark.sql.catalyst.expressions.ExprId, Attribute] =
      passThrough.map { case (slot, childAttr) => output(slot).exprId -> childAttr }
    if (!inputRewrite.values.forall(childAttrs.contains) || !finestKeys.forall(childAttrs.contains)) return None
    val stage1Aggs = agg.aggregateExpressions.map { ae =>
      ae.transform { case a: AttributeReference => inputRewrite.getOrElse(a.exprId, a) }.asInstanceOf[AggregateExpression]
    }
    // Stage 1's functions may be new objects (their inputs were rewritten), with buffer attributes of
    // their own; the Expand maps those onto the ORIGINAL buffer attributes, which the merge below and
    // the final aggregate above read by identity.
    val stage1Buffers: Seq[Attribute] = stage1Aggs.flatMap(_.aggregateFunction.inputAggBufferAttributes)
    val buffers: Seq[Attribute] = agg.aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes)
    if (stage1Buffers.length != buffers.length || stage1Buffers.map(_.dataType) != buffers.map(_.dataType)) return None
    val stage1 = HashAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = finestKeys,
      aggregateExpressions = stage1Aggs,
      aggregateAttributes = agg.aggregateAttributes,
      initialInputBufferOffset = 0,
      resultExpressions = finestKeys ++ stage1Buffers,
      child = expand.child)
    val planned1 = VectorAggregatePlanner.plan(stage1, finalEnabled = true, strict).toOption.getOrElse(return None)

    // Stage 2: the Expand over the partial rows -- the key slots as they were (over stage 1's key
    // attributes, which are the child's), the aggregate inputs replaced by the buffers.
    val keyOutput: Seq[Attribute] = groupingAttrs
    val projections2: Seq[Seq[Expression]] = expand.projections.map { p =>
      groupingSlots.flatten.map(slot => p(slot)) ++ stage1Buffers
    }
    val expand2 = VectorExpandExec(projections2, keyOutput ++ buffers, planned1)

    // Stage 3: the merge of the copies, emitting exactly what the original partial emitted.
    val stage3Aggs = agg.aggregateExpressions.map(ae => ae.copy(mode = PartialMerge, filter = None))
    val stage3 = HashAggregateExec(
      requiredChildDistributionExpressions = agg.requiredChildDistributionExpressions,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = agg.groupingExpressions,
      aggregateExpressions = stage3Aggs,
      aggregateAttributes = agg.aggregateAttributes,
      initialInputBufferOffset = agg.groupingExpressions.length,
      resultExpressions = agg.resultExpressions,
      child = expand2)
    VectorAggregatePlanner.plan(stage3, finalEnabled = true, strict).toOption.filter(_.output.map(_.exprId) == agg.output.map(_.exprId))
  }
}
