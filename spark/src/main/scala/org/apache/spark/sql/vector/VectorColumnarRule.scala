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
import org.apache.spark.sql.execution.{CoalesceExec, CollectLimitExec, ColumnarRule, ExpandExec, FilterExec, GenerateExec, GlobalLimitExec, LocalLimitExec, LocalTableScanExec, ProjectExec, SampleExec, SortExec, SparkPlan, TakeOrderedAndProjectExec, UnionExec}
import org.apache.spark.sql.execution.datasources.v2.MergeRowsExec
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.adaptive.{AQEShuffleReadExec, QueryStageExec}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.execution.window.{WindowExec, WindowGroupLimitExec}
import org.apache.spark.sql.catalyst.expressions.aggregate.Final // still used below
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.internal.SQLConf

object VectorExecRule {
  /**
   * Set on every SortMergeJoinExec by the pre-pass: the build side of the shuffled hash join it becomes,
   * or the reason it stays Spark's. The transform builds from this decision and never re-derives it.
   */
  val SortMergeDecision: TreeNodeTag[Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide]] =
    TreeNodeTag("io.sparkvector.sortMergeJoin.decision")
}

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
    lazy val maxBuildSize = VectorConf.joinMaxBuildSize(conf, session.sparkContext.getConf)
    if (!VectorConf.isEnabled(conf)) {
      plan
    } else {
      if (VectorConf.sortMergeJoinEnabled(conf)) {
        markSortMergeJoins(plan, orderingNeeded = false, maxBuildSize, new java.util.IdentityHashMap[SparkPlan, Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide]])
      }
      val converted = plan.transformUp {
        case f @ FilterExec(condition, child) if VectorConf.filterEnabled(conf) =>
          forwardingInputReason(child).orElse(filterReason(f)) match {
            case Some(reason) => fallback(f, reason)
            case None => VectorFilterExec(condition, child)
          }

        case p @ ProjectExec(projectList, child) if VectorConf.projectEnabled(conf) =>
          forwardingInputReason(child).orElse(projectReason(p)) match {
            case Some(reason) => fallback(p, reason)
            case None => VectorProjectExec(projectList, child)
          }

        // The row-level operator of a MERGE INTO (#21): its child is the merge's join, so it converts
        // when that join is ours (the shuffled hash join, or a sort-merge join re-expressed as one, #10).
        case m: MergeRowsExec if VectorConf.mergeRowsEnabled(conf) =>
          columnarInputReason(m.child).orElse(VectorMergeRowsPlanner.reason(m)) match {
            case Some(reason) => fallback(m, reason)
            case None =>
              VectorMergeRowsExec(m.isSourceRowPresent, m.isTargetRowPresent, m.matchedInstructions, m.notMatchedInstructions,
                m.notMatchedBySourceInstructions, m.checkCardinality, m.output, m.child)
          }

        case e: ExpandExec if VectorConf.expandEnabled(conf) =>
          laneInputReason(e.child) match {
            case Some(reason) => fallback(e, reason)
            case None => VectorExpandPlanner.plan(e).fold(reason => fallback(e, reason), v => v)
          }

        case u: UnionExec if VectorConf.unionEnabled(conf) =>
          // Columnar whatever the children are, as long as one of them is: Spark's transitions
          // convert the row children through RowToColumnarExec below us.
          VectorStructuralPlanner.planUnion(u).fold(reason => fallback(u, reason), v => v)

        case c: CoalesceExec if VectorConf.coalesceEnabled(conf) =>
          laneInputReason(c.child) match {
            case Some(reason) => fallback(c, reason)
            case None => VectorStructuralPlanner.planCoalesce(c).fold(reason => fallback(c, reason), v => v)
          }

        case g: GenerateExec if VectorConf.generateEnabled(conf) =>
          // The array column has no lane and is read as Spark's vector, so only the columnar contract is required.
          forwardingInputReason(g.child) match {
            case Some(reason) => fallback(g, reason)
            case None => VectorGeneratePlanner.plan(g).fold(reason => fallback(g, reason), v => v)
          }

        case s: SampleExec if VectorConf.sampleEnabled(conf) =>
          laneInputReason(s.child) match {
            case Some(reason) => fallback(s, reason)
            case None => VectorSamplePlanner.plan(s).fold(reason => fallback(s, reason), v => v)
          }

        case l: LocalTableScanExec if VectorConf.localTableScanEnabled(conf) =>
          VectorSamplePlanner.planLocalTableScan(l).fold(reason => fallback(l, reason), v => v)

        case l: LocalLimitExec if VectorConf.limitEnabled(conf) =>
          laneInputReason(l.child) match {
            case Some(reason) => fallback(l, reason)
            case None => VectorLimitPlanner.planLocal(l).fold(reason => fallback(l, reason), v => v)
          }

        case g: GlobalLimitExec if VectorConf.limitEnabled(conf) =>
          // Only over a columnar child: above Spark's row shuffle the limit stays Spark's.
          laneInputReason(g.child) match {
            case Some(reason) => fallback(g, reason)
            case None => VectorLimitPlanner.planGlobal(g).fold(reason => fallback(g, reason), v => v)
          }

        case c: CollectLimitExec if VectorConf.limitEnabled(conf) =>
          // Per-partition cut stays columnar; the final take goes through Spark's single-partition
          // shuffle like the top-N operator.
          laneInputReason(c.child) match {
            case Some(reason) => fallback(c, reason)
            case None => VectorLimitPlanner.planCollect(c).fold(reason => fallback(c, reason), v => v)
          }

        case t: TakeOrderedAndProjectExec if VectorConf.takeOrderedEnabled(conf) =>
          // ORDER BY ... LIMIT over a columnar child: the per-partition top-N is ours, the final
          // merge of at most limit rows per partition goes through Spark's single-partition shuffle.
          laneInputReason(t.child) match {
            case Some(reason) => fallback(t, reason)
            case None =>
              VectorTakeOrderedPlanner.plan(t) match {
                case Right(v) => v
                case Left(reason) => fallback(t, reason)
              }
          }

        case w: WindowExec if VectorConf.windowEnabled(conf) =>
          // Over any child on types alone: Spark plans Window above Sort above an exchange, and without a
          // columnar shuffle that sort is Spark's, so RowToColumnarExec is inserted below us -- the window
          // itself computes, and from here up the chain is columnar again.
          typeReason(w.child) match {
            case Some(reason) => fallback(w, reason)
            case None => VectorWindowPlanner.plan(w).fold(reason => fallback(w, reason), v => v)
          }

        case g: WindowGroupLimitExec if VectorConf.windowEnabled(conf) =>
          // Spark's per-partition top-k under a ranking window: Partial sits over whatever produced the rows
          // (often ours), Final over Spark's sort -- either way the child is accepted on types alone.
          typeReason(g.child) match {
            case Some(reason) => fallback(g, reason)
            case None => VectorWindowGroupLimitPlanner.plan(g).fold(reason => fallback(g, reason), v => v)
          }

        case s: SortExec if VectorConf.sortEnabled(conf) =>
          // Only over a columnar child: a sort above Spark's row shuffle would need a
          // RowToColumnarExec first and gain nothing over Spark's own sort. The sort only moves
          // its columns (append, order, gather), so a DECIMAL128 lane is as good as any (#257).
          laneInputReason(s.child) match {
            case Some(reason) => fallback(s, reason)
            case None =>
              VectorSortPlanner.plan(s) match {
                case Right(v) => v
                case Left(reason) => fallback(s, reason)
              }
          }

        case j: BroadcastHashJoinExec if VectorConf.broadcastHashJoinEnabled(conf) =>
          // The build side is Spark's broadcast relation whatever it is; the streamed side must be
          // columnar -- or an exchange (its AQE stage, a shuffle read), which Spark converts below us
          // with RowToColumnarExec exactly as for the shuffled hash join. That shape is what adaptive
          // execution leaves when it re-plans a shuffled join as a broadcast join at runtime: the
          // streamed side is then the bare shuffle read, and refusing it left the whole chain above
          // (projects, aggregates) to Spark in eleven TPC-DS queries.
          val (buildPlan, streamedPlan) = j.buildSide match {
            case org.apache.spark.sql.catalyst.optimizer.BuildLeft => (j.left, j.right)
            case org.apache.spark.sql.catalyst.optimizer.BuildRight => (j.right, j.left)
          }
          exchangeInputReason(streamedPlan).orElse(typeReason(buildPlan)).orElse(VectorJoinPlanner.buildSizeReason(buildPlan, maxBuildSize)) match {
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
          exchangeInputReason(streamedPlan).orElse(typeReason(buildPlan)).orElse(VectorJoinPlanner.buildSizeReason(buildPlan, maxBuildSize)) match {
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
          val shjBuild = j.buildSide match {
            case org.apache.spark.sql.catalyst.optimizer.BuildLeft => j.left
            case org.apache.spark.sql.catalyst.optimizer.BuildRight => j.right
          }
          exchangeInputReason(j.left).orElse(exchangeInputReason(j.right)).orElse(VectorJoinPlanner.buildSizeReason(shjBuild, maxBuildSize)) match {
            case Some(reason) => fallback(j, reason)
            case None =>
              VectorJoinPlanner.plan(j) match {
                case Right(v) => v
                case Left(reason) => fallback(j, reason)
              }
          }

        case j: SortMergeJoinExec if VectorConf.sortMergeJoinEnabled(conf) =>
          // Re-expressed as our shuffled hash join (#10): same distribution, same rows, no need for
          // the sorts Spark placed below. Two contracts decide it. The parent may have been planned
          // on the merge join's output ordering (a Window over the same keys, a merge join above on
          // the same keys with no shuffle in between): such a join stays Spark's, decided by the
          // pre-pass over the whole plan (markSortMergeJoins), since a hash join has no ordering to
          // offer. And the build side must fit the budget by statistics, not by assumption.
          // The decision is the pre-pass's, taken over the original plan: by now the children are the
          // transformed ones (a merge join below that converted is our hash join, columnar; a required
          // sort is stripped), and re-deriving anything here -- a size estimate, say, which our operators
          // do not carry -- could contradict the pass and leave a Spark merge join reading unsorted input.
          j.getTagValue(VectorExecRule.SortMergeDecision) match {
            case Some(Right(buildSide)) =>
              val (left, right) = sortMergeInputs(j)
              exchangeInputReason(left).orElse(exchangeInputReason(right)) match {
                case None => VectorShuffledHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, buildSide, j.condition, left, right)
                case Some(reason) => fallback(resorted(j), reason)
              }
            case Some(Left(reason)) => fallback(resorted(j), reason)
            case None => fallback(resorted(j), "sort-merge join not examined by the pre-pass")
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
   * The input of an operator that forwards its child's columns without reading them all (filter,
   * project): a column of a type the kernels have no lane for is passed through as Spark's own vector
   * (see [[VectorProjectExec]]), so only the columnar contract is required here; an expression that
   * reads such a column is refused by the compiler with the type named.
   */
  private def forwardingInputReason(plan: SparkPlan): Option[String] =
    if (!plan.supportsColumnar) Some(s"child ${plan.nodeName} is not columnar") else None

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
    // Any lane will do as input (a wide decimal key, input or buffer is a DECIMAL128 lane, #259): the
    // planner below refuses a column a function or a key cannot read, with the function's own reason.
    val inputReason = if (isFinal) laneTypeReason(a.child) else laneInputReason(a.child)
    inputReason match {
      case Some(reason) => fallback(original, reason)
      case None =>
        VectorAggregatePlanner.plan(a, VectorConf.finalAggregateEnabled(conf), VectorConf.strictFloatingPoint(conf)) match {
          case Right(v) => v
          case Left(reason) => fallback(original, reason)
        }
    }
  }

  private def sameOrder(a: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder], b: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]): Boolean =
    a.length == b.length && a.zip(b).forall { case (x, y) => x.child.semanticEquals(y.child) && x.direction == y.direction && x.nullOrdering == y.nullOrdering }

  /** The merge join's inputs without the sorts Spark placed for the merge (exactly the required ones, local). */
  private def sortMergeInputs(j: SortMergeJoinExec): (SparkPlan, SparkPlan) = {
    def strip(child: SparkPlan, required: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]): SparkPlan = child match {
      case VectorSortExec(order, false, c) if sameOrder(order, required) => c
      case SortExec(order, false, c, _) if sameOrder(order, required) => c
      case c => c
    }
    (strip(j.left, j.requiredChildOrdering.head), strip(j.right, j.requiredChildOrdering(1)))
  }

  /** Why a filter's predicate would not compile over its child's output, input aside. */
  private def filterReason(f: FilterExec): Option[String] =
    ExpressionCompiler.compilePredicate(f.condition, f.child.output).left.toOption

  /** Why a projection would not compile over its child's output (an expression, an output type), input aside. */
  private def projectReason(p: ProjectExec): Option[String] = {
    val failures = p.projectList.filterNot(VectorProjectExec.isPassThrough).flatMap { e =>
      val compiled = ExpressionCompiler.compile(e, p.child.output)
      // A decimal output wider than 18 digits is a DECIMAL128 lane when its expression compiled (#258).
      val typeCheck =
        if (TypeMapping.isSupported(e.dataType) || (e.dataType.isInstanceOf[org.apache.spark.sql.types.DecimalType] && TypeMapping.hasLane(e.dataType))) Right(())
        else Left(s"unsupported output type ${e.dataType.simpleString} for ${e.name}")
      compiled.flatMap(_ => typeCheck).left.toOption.map(r => s"${e.sql}: $r")
    }
    if (failures.isEmpty) None else Some(failures.mkString("; "))
  }

  /**
   * A merge join that stays Spark's must read sorted inputs: a merge join below it that converted (the
   * pre-pass expected this one to convert too) no longer offers its ordering, so the sort comes back over
   * that columnar child. A child that still satisfies the requirement -- Spark's own, or a sort left in
   * place -- is untouched.
   */
  private def resorted(j: SortMergeJoinExec): SortMergeJoinExec = {
    def fix(child: SparkPlan, required: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]): SparkPlan =
      if (org.apache.spark.sql.catalyst.expressions.SortOrder.orderingSatisfies(child.outputOrdering, required) || !child.supportsColumnar) child
      else VectorSortExec(required, global = false, child)
    val left = fix(j.left, j.requiredChildOrdering.head)
    val right = fix(j.right, j.requiredChildOrdering(1))
    if ((left eq j.left) && (right eq j.right)) j else j.copy(left = left, right = right)
  }

  private type SortMergeMemo = java.util.IdentityHashMap[SparkPlan, Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide]]

  /**
   * Whether a sort-merge join could become our hash join, ordering aside: its inputs (sorts stripped)
   * are exchanges, or merge joins that could themselves convert -- a chain of merge joins on the same
   * key has no shuffle between its links, and the upper link reads the lower one's output, which is
   * columnar once the lower converts (TPC-DS q10, q35, q69, q95 chain semi and existence joins on the
   * customer key). Bottom-up, memoised over the original nodes; the ordering question is the pre-pass's.
   */
  private def sortMergeEligibility(j: SortMergeJoinExec, maxBuildSize: Long, memo: SortMergeMemo): Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide] = {
    val known = memo.get(j)
    if (known != null) known
    else {
      val conf = session.sessionState.conf
      val (left, right) = sortMergeInputs(j)
      // The pass runs over Spark's plan, before anything is converted, so an input is judged by what it
      // will be: an exchange is read through RowToColumnarExec; a merge join is columnar if it converts
      // (recursively -- Spark prunes and filters between the links of a chain, so projects and filters are
      // seen through with the rule's own checks); any other operator the rule replaces is assumed to
      // convert when its output types are lanes. That assumption is optimistic on purpose: the transform
      // verifies the real children and, when a join has to stay after all, restores the ordering a
      // converted child below it no longer offers (`resorted`), so a wrong guess costs a sort, never a row.
      def inputReason(p: SparkPlan): Option[String] = p match {
        case inner: SortMergeJoinExec =>
          sortMergeEligibility(inner, maxBuildSize, memo).left.toOption.map(r => s"child SortMergeJoin stays with Spark: $r").orElse(typeReason(inner))
        case proj: ProjectExec if VectorConf.projectEnabled(conf) => projectReason(proj).orElse(inputReason(proj.child))
        case filt: FilterExec if VectorConf.filterEnabled(conf) => filterReason(filt).orElse(inputReason(filt.child))
        case _: ShuffleExchangeLike | _: QueryStageExec | _: AQEShuffleReadExec => typeReason(p)
        case _: UnionExec | _: ExpandExec | _: CoalesceExec | _: SampleExec | _: SortExec | _: LocalLimitExec | _: GlobalLimitExec |
             _: BroadcastHashJoinExec | _: BroadcastNestedLoopJoinExec | _: ShuffledHashJoinExec | _: HashAggregateExec | _: SortAggregateExec =>
          typeReason(p)
        case other => columnarInputReason(other)
      }
      val decision = inputReason(left).orElse(inputReason(right)) match {
        case Some(reason) => Left(reason)
        case None => VectorJoinPlanner.sortMergeBuildSide(j.leftKeys, j.rightKeys, j.joinType, j.condition, j.isSkewJoin, left, right, maxBuildSize)
      }
      memo.put(j, decision)
      decision
    }
  }

  /**
   * Top-down pre-pass deciding every sort-merge join: it stays Spark's when some ancestor relies on its
   * output ordering -- a parent that requires an ordering of it (Spark placed no sort in between only
   * because the join's ordering satisfied it), or an ordered ancestor whose own ordering is relied on
   * and derives from it -- or when it is not eligible; otherwise it converts with the build side the
   * eligibility chose. A join that converts requires nothing of its children, so a merge join below it
   * on the same key is free to convert too; one that stays requires its children sorted, and a merge
   * join among them then stays as well. The decisions ride along the copies transformUp makes.
   */
  private def markSortMergeJoins(plan: SparkPlan, orderingNeeded: Boolean, maxBuildSize: Long, memo: SortMergeMemo): Unit = {
    val converts = plan match {
      case j: SortMergeJoinExec =>
        val decision = if (orderingNeeded) Left("output ordering required by the parent operator") else sortMergeEligibility(j, maxBuildSize, memo)
        j.setTagValue(VectorExecRule.SortMergeDecision, decision)
        decision.isRight
      case _ => false
    }
    plan.children.zipWithIndex.foreach { case (child, i) =>
      val required = !converts && plan.requiredChildOrdering(i).nonEmpty
      val passes = orderingNeeded && plan.outputOrdering.nonEmpty &&
        org.apache.spark.sql.catalyst.expressions.SortOrder.orderingSatisfies(child.outputOrdering, plan.outputOrdering)
      markSortMergeJoins(child, required || passes, maxBuildSize, memo)
    }
  }

  /** Like [[columnarInputReason]] for an operator that only moves columns: any lane type will do. */
  private def laneInputReason(plan: SparkPlan): Option[String] =
    if (!plan.supportsColumnar) Some(s"child ${plan.nodeName} is not columnar")
    else plan.output.find(a => !TypeMapping.hasLane(a.dataType)).map(a => s"unsupported column type ${a.dataType.simpleString} for ${a.name}")

  /** Like [[typeReason]] for an operator that reads any lane type (the aggregate over the DECIMAL128 lane, #259). */
  private def laneTypeReason(plan: SparkPlan): Option[String] =
    plan.output.find(a => !TypeMapping.hasLane(a.dataType)).map(a => s"unsupported column type ${a.dataType.simpleString} for ${a.name}")

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
