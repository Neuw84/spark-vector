package org.apache.spark.sql.vector

import io.sparkvector.spark.VectorConf
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.expr.ExpressionCompiler
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import io.sparkvector.spark.comet.CometBatchBridge
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, RangePartitioning}
import org.apache.spark.sql.execution.{CoalesceExec, CollectLimitExec, ColumnarRule, ExpandExec, FilterExec, GlobalLimitExec, LocalLimitExec, LocalTableScanExec, ProjectExec, SampleExec, SortExec, SparkPlan, TakeOrderedAndProjectExec, UnionExec}
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.adaptive.{AQEShuffleReadExec, QueryStageExec}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, ShuffledHashJoinExec}
import org.apache.spark.sql.catalyst.expressions.aggregate.Final // still used below
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.internal.SQLConf

/** Tags and helpers for explaining why an operator was left to Spark. */
object VectorFallback {
  val Tag: TreeNodeTag[String] = TreeNodeTag[String]("io.sparkvector.fallback")

  def reason(plan: SparkPlan): Option[String] = plan.getTagValue(Tag)

  /** All fallback reasons in a plan tree, including inside adaptive query stages. */
  def reasons(plan: SparkPlan): Seq[(SparkPlan, String)] =
    PlanUtils.allNodes(plan).flatMap(p => reason(p).map(r => (p, r)))
}

/** Registered through SparkSessionExtensions.injectColumnar. */
case class VectorColumnarRule(session: SparkSession) extends ColumnarRule {
  override def preColumnarTransitions: Rule[SparkPlan] = VectorExecRule(session)
}

/**
 * Bottom-up replacement of Spark operators by spark-vector ones. An operator is converted only if
 * its child already produces columnar batches of supported types (a vectorized Parquet scan, a
 * Comet scan, or another spark-vector operator) and every expression compiles; otherwise the
 * reason is attached as a tag and the Spark operator stays.
 */
case class VectorExecRule(session: SparkSession) extends Rule[SparkPlan] with Logging {

  override def apply(plan: SparkPlan): SparkPlan = {
    val conf = session.sessionState.conf
    if (!VectorConf.isEnabled(conf)) {
      plan
    } else {
      val converted = plan.transformUp {
        case f @ FilterExec(condition, child) if VectorConf.filterEnabled(conf) =>
          columnarInputReason(child) match {
            case Some(reason) => fallback(f, reason)
            case None =>
              ExpressionCompiler.compilePredicate(condition, child.output) match {
                case Right(_) => VectorFilterExec(condition, child)
                case Left(reason) => fallback(f, reason)
              }
          }

        case p @ ProjectExec(projectList, child) if VectorConf.projectEnabled(conf) =>
          columnarInputReason(child) match {
            case Some(reason) => fallback(p, reason)
            case None =>
              val failures = projectList.flatMap { e =>
                val compiled = ExpressionCompiler.compile(e, child.output)
                val typeCheck =
                  if (TypeMapping.isSupported(e.dataType)) Right(())
                  else Left(s"unsupported output type ${e.dataType.simpleString} for ${e.name}")
                compiled.flatMap(_ => typeCheck).left.toOption.map(r => s"${e.sql}: $r")
              }
              if (failures.isEmpty) VectorProjectExec(projectList, child)
              else fallback(p, failures.mkString("; "))
          }

        case e: ExpandExec if VectorConf.expandEnabled(conf) =>
          columnarInputReason(e.child) match {
            case Some(reason) => fallback(e, reason)
            case None => VectorExpandPlanner.plan(e).fold(reason => fallback(e, reason), v => v)
          }

        case u: UnionExec if VectorConf.unionEnabled(conf) =>
          // Columnar whatever the children are, as long as one of them is: Spark's transitions
          // convert the row children through RowToColumnarExec below us.
          VectorStructuralPlanner.planUnion(u).fold(reason => fallback(u, reason), v => v)

        case c: CoalesceExec if VectorConf.coalesceEnabled(conf) =>
          columnarInputReason(c.child) match {
            case Some(reason) => fallback(c, reason)
            case None => VectorStructuralPlanner.planCoalesce(c).fold(reason => fallback(c, reason), v => v)
          }

        case s: SampleExec if VectorConf.sampleEnabled(conf) =>
          columnarInputReason(s.child) match {
            case Some(reason) => fallback(s, reason)
            case None => VectorSamplePlanner.plan(s).fold(reason => fallback(s, reason), v => v)
          }

        case l: LocalTableScanExec if VectorConf.localTableScanEnabled(conf) =>
          VectorSamplePlanner.planLocalTableScan(l).fold(reason => fallback(l, reason), v => v)

        case l: LocalLimitExec if VectorConf.limitEnabled(conf) =>
          columnarInputReason(l.child) match {
            case Some(reason) => fallback(l, reason)
            case None => VectorLimitPlanner.planLocal(l).fold(reason => fallback(l, reason), v => v)
          }

        case g: GlobalLimitExec if VectorConf.limitEnabled(conf) =>
          // Only over a columnar child: above Spark's row shuffle the limit stays Spark's.
          columnarInputReason(g.child) match {
            case Some(reason) => fallback(g, reason)
            case None => VectorLimitPlanner.planGlobal(g).fold(reason => fallback(g, reason), v => v)
          }

        case c: CollectLimitExec if VectorConf.limitEnabled(conf) =>
          // Per-partition cut stays columnar; the final take goes through Spark's single-partition
          // shuffle like the top-N operator.
          columnarInputReason(c.child) match {
            case Some(reason) => fallback(c, reason)
            case None => VectorLimitPlanner.planCollect(c).fold(reason => fallback(c, reason), v => v)
          }

        case t: TakeOrderedAndProjectExec if VectorConf.takeOrderedEnabled(conf) =>
          // ORDER BY ... LIMIT over a columnar child: the per-partition top-N is ours, the final
          // merge of at most limit rows per partition goes through Spark's single-partition shuffle.
          columnarInputReason(t.child) match {
            case Some(reason) => fallback(t, reason)
            case None =>
              VectorTakeOrderedPlanner.plan(t) match {
                case Right(v) => v
                case Left(reason) => fallback(t, reason)
              }
          }

        case s: SortExec if VectorConf.sortEnabled(conf) =>
          // Only over a columnar child: a sort above Spark's row shuffle would need a
          // RowToColumnarExec first and gain nothing over Spark's own sort.
          columnarInputReason(s.child) match {
            case Some(reason) => fallback(s, reason)
            case None =>
              VectorSortPlanner.plan(s) match {
                case Right(v) => v
                case Left(reason) => fallback(s, reason)
              }
          }

        case j: BroadcastHashJoinExec if VectorConf.broadcastHashJoinEnabled(conf) =>
          // The build side is Spark's broadcast relation whatever it is; the streamed side must be
          // columnar.
          val (buildPlan, streamedPlan) = j.buildSide match {
            case org.apache.spark.sql.catalyst.optimizer.BuildLeft => (j.left, j.right)
            case org.apache.spark.sql.catalyst.optimizer.BuildRight => (j.right, j.left)
          }
          columnarInputReason(streamedPlan).orElse(typeReason(buildPlan)) match {
            case Some(reason) => fallback(j, reason)
            case None =>
              VectorJoinPlanner.plan(j) match {
                case Right(v) => v
                case Left(reason) => fallback(j, reason)
              }
          }

        case j: BroadcastNestedLoopJoinExec if VectorConf.broadcastNestedLoopJoinEnabled(conf) =>
          val (buildPlan, streamedPlan) = j.buildSide match {
            case org.apache.spark.sql.catalyst.optimizer.BuildLeft => (j.left, j.right)
            case org.apache.spark.sql.catalyst.optimizer.BuildRight => (j.right, j.left)
          }
          columnarInputReason(streamedPlan).orElse(typeReason(buildPlan)) match {
            case Some(reason) => fallback(j, reason)
            case None =>
              VectorJoinPlanner.plan(j) match {
                case Right(v) => v
                case Left(reason) => fallback(j, reason)
              }
          }

        case j: ShuffledHashJoinExec if VectorConf.shuffledHashJoinEnabled(conf) =>
          // Both inputs are exchanges: Spark's row shuffle is converted below us by
          // RowToColumnarExec, Comet's columnar one is read directly.
          exchangeInputReason(j.left).orElse(exchangeInputReason(j.right)) match {
            case Some(reason) => fallback(j, reason)
            case None =>
              VectorJoinPlanner.plan(j) match {
                case Right(v) => v
                case Left(reason) => fallback(j, reason)
              }
          }

        case a: HashAggregateExec if VectorConf.aggregateEnabled(conf) => planAggregate(a, a, conf)

        case s: SortAggregateExec if VectorConf.aggregateEnabled(conf) =>
          // Spark plans a SortAggregate when an aggregation buffer holds a string (min/max/first/last
          // over strings): not mutable in an UnsafeRow, so no hash aggregate for Spark. Our group table
          // has no such limit, so the same hash operator serves, built from the identical fields. Two
          // contracts to keep: the sort Spark placed below is not needed by a hash aggregate and is
          // dropped when it is exactly the required one, and a result-emitting stage keeps Spark's
          // output ordering (the keys ascending) through our sort, since parents were planned on it.
          val child = s.child match {
            case VectorSortExec(order, false, c) if sameOrder(order, s.requiredChildOrdering.head) => c
            case org.apache.spark.sql.execution.SortExec(order, false, c, _) if sameOrder(order, s.requiredChildOrdering.head) => c
            case c => c
          }
          planAggregate(s, s, conf) match {
            case v: VectorHashAggregateExec =>
              val unsorted = if (child eq s.child) v else v.copy(child = child)
              if (v.emitsResults && s.outputOrdering.nonEmpty) VectorSortExec(s.outputOrdering, global = false, unsorted) else unsorted
            case other => other
          }
      }
      val withSelections = if (VectorConf.selectionEnabled(conf)) markSelectionProducers(converted) else converted
      val withShuffles =
        if (VectorConf.cometShuffleEnabled(conf) && CometShuffle.isEnabled(conf, session.sparkContext.getConf.get("spark.shuffle.manager", "sort")))
          useCometShuffle(withSelections, conf)
        else withSelections
      if (VectorConf.explainFallback(conf)) {
        VectorFallback.reasons(withShuffles).foreach { case (node, reason) =>
          logInfo(s"spark-vector fallback for ${node.nodeName}: $reason")
        }
      }
      withShuffles
    }
  }

  /**
   * An exchange fed by one of our operators becomes Comet's native shuffle over a
   * [[VectorToCometExec]], whether Spark still owns it or Comet already turned it into its
   * row-based columnar shuffle (which would have converted our batches to rows and back). Comet's
   * own planner declines because it does not recognise our operators; at run time its native writer
   * only needs `CometVector` batches, which the bridge provides. Range partitioning is left alone:
   * it samples the child, which a native shuffle over a non-native child does twice.
   */
  private def useCometShuffle(plan: SparkPlan, conf: SQLConf): SparkPlan = plan.transformUp {
    case s: ShuffleExchangeExec if s.child.isInstanceOf[VectorPlan] && bridgeable(s.child, s.outputPartitioning, conf) =>
      CometShuffle.native(s, VectorToCometExec(s.child))
    case c if CometShuffle.isCometExchange(c) && !CometShuffle.isNative(c) &&
        c.children.head.isInstanceOf[VectorPlan] && bridgeable(c.children.head, c.outputPartitioning, conf) =>
      CometShuffle.toNative(c, VectorToCometExec(c.children.head))
  }

  /**
   * Range partitioning is bridged too when Comet's native range partitioning is on (its default):
   * Comet samples the child for the bounds through Spark's `RangePartitioner`, exactly as Spark's
   * own exchange does, so the child runs twice in both cases. `spark.vector.comet.shuffle.range.enabled`
   * turns just this part off.
   */
  private def bridgeable(child: SparkPlan, partitioning: Partitioning, conf: SQLConf): Boolean = {
    val partitioningOk = partitioning match {
      case r: RangePartitioning =>
        VectorConf.cometRangeShuffleEnabled(conf) && CometShuffle.rangePartitioningEnabled(conf) &&
          r.ordering.forall(o => CometBatchBridge.isSupported(o.dataType))
      case _ => true
    }
    partitioningOk && child.output.forall(a => CometBatchBridge.isSupported(a.dataType))
  }

  /**
   * A filter or projection whose parent is another spark-vector operator forwards its child's
   * columns with a selection bitmap instead of compacting them; the consumer folds the bitmap into
   * its own evaluation. Anything else (Spark operators, exchanges) needs dense batches.
   */
  private def markSelectionProducers(plan: SparkPlan): SparkPlan = plan.transformDown {
    case parent: VectorPlan if !parent.isInstanceOf[VectorPassThrough] =>
      parent.withNewChildren(parent.children.map {
        case f: VectorFilterExec if !f.emitSelection => f.copy(emitSelection = true)
        case p: VectorProjectExec if !p.emitSelection => p.copy(emitSelection = true)
        case s: VectorSampleExec if !s.emitSelection => s.copy(emitSelection = true)
        case other => other
      })
  }

  private def fallback(plan: SparkPlan, reason: String): SparkPlan = {
    plan.setTagValue(VectorFallback.Tag, reason)
    plan
  }

  /** None if `plan` is an acceptable columnar input, else the reason it is not. */
  private def columnarInputReason(plan: SparkPlan): Option[String] = {
    if (!plan.supportsColumnar) Some(s"child ${plan.nodeName} is not columnar") else typeReason(plan)
  }

  /**
   * A join input that is an exchange (or its adaptive stage) is accepted on types alone, like the
   * Final aggregate's input: Spark inserts RowToColumnarExec below us when the shuffle is row based.
   */
  private def exchangeInputReason(plan: SparkPlan): Option[String] = plan match {
    case _: ShuffleExchangeLike | _: QueryStageExec | _: AQEShuffleReadExec => typeReason(plan)
    case other => columnarInputReason(other)
  }

  /** Converts an aggregate (`a`, possibly a SortAggregate re-expressed as a hash one); `original` takes the fallback. */
  private def planAggregate(a: org.apache.spark.sql.execution.aggregate.BaseAggregateExec, original: SparkPlan, conf: SQLConf): SparkPlan = {
    // A merging aggregate (Final, PartialMerge) reads an exchange; Spark inserts RowToColumnarExec below us when the
    // shuffle is row based (Comet's shuffle is columnar already), so only the types matter.
    val isFinal = VectorAggregatePlanner.readsExchange(a)
    // The wide decimal sum buffer (Decimal(p > 18)) is the one wide column a merging aggregate reads.
    val inputReason = if (isFinal) typeReason(a.child, VectorAggregatePlanner.wideSumBuffers(a)) else columnarInputReason(a.child)
    inputReason match {
      case Some(reason) => fallback(original, reason)
      case None =>
        VectorAggregatePlanner.plan(a, VectorConf.finalAggregateEnabled(conf)) match {
          case Right(v) => v
          case Left(reason) => fallback(original, reason)
        }
    }
  }

  private def sameOrder(a: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder], b: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]): Boolean =
    a.length == b.length && a.zip(b).forall { case (x, y) => x.child.semanticEquals(y.child) && x.direction == y.direction && x.nullOrdering == y.nullOrdering }

  private def typeReason(plan: SparkPlan, allowed: Set[org.apache.spark.sql.catalyst.expressions.ExprId] = Set.empty): Option[String] =
    plan.output.find(a => !TypeMapping.isSupported(a.dataType) && !allowed.contains(a.exprId)).map { a =>
      s"unsupported column type ${a.dataType.simpleString} for ${a.name}"
    }
}

object PlanUtils {
  import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
  import org.apache.spark.sql.execution.exchange.ReusedExchangeExec

  /** Every node in the final physical plan, descending into adaptive stages and reused exchanges. */
  def allNodes(plan: SparkPlan): Seq[SparkPlan] = {
    val inner = plan match {
      case a: AdaptiveSparkPlanExec => Seq(a.executedPlan)
      case q: QueryStageExec => Seq(q.plan)
      case r: ReusedExchangeExec => Seq(r.child)
      case _ => Nil
    }
    plan +: (plan.children ++ inner).flatMap(allNodes)
  }
}
