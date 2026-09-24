package org.apache.spark.sql.vector

import io.sparkvector.spark.VectorConf
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute}
import io.sparkvector.spark.expr.ExpressionCompiler
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import io.sparkvector.spark.comet.{CometBatchBridge, CometMixedBridge}
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, RangePartitioning}
import org.apache.spark.sql.execution.{
  CoalesceExec,
  CollectLimitExec,
  ColumnarRule,
  ExpandExec,
  FilterExec,
  GenerateExec,
  GlobalLimitExec,
  LocalLimitExec,
  LocalTableScanExec,
  ProjectExec,
  SampleExec,
  SortExec,
  SparkPlan,
  TakeOrderedAndProjectExec,
  UnionExec
}
import org.apache.spark.sql.execution.datasources.v2.MergeRowsExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.adaptive.{AQEShuffleReadExec, QueryStageExec}
import org.apache.spark.sql.execution.joins.{
  BroadcastHashJoinExec,
  BroadcastNestedLoopJoinExec,
  ShuffledHashJoinExec,
  SortMergeJoinExec
}
import org.apache.spark.sql.execution.window.{WindowExec, WindowGroupLimitExec}
import org.apache.spark.sql.catalyst.expressions.aggregate.Final // still used below
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, ObjectHashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.internal.SQLConf

object VectorExecRule {

  /**
   * Set on every SortMergeJoinExec by the pre-pass: the build side of the shuffled hash join it becomes,
   * or the reason it stays Spark's. The transform builds from this decision and never re-derives it.
   */
  val SortMergeDecision: TreeNodeTag[Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide]] =
    TreeNodeTag("io.sparkvector.sortMergeJoin.decision")

  /**
   * Under `auto`, what a sort-merge join becomes and why (#287): `Left(why)` the merge join, `Right((side, why))`
   * the hash rewrite building `side`. The reason is printed on the operator so the plan shows the decision.
   */
  val SortMergeChoice: TreeNodeTag[Either[String, (org.apache.spark.sql.catalyst.optimizer.BuildSide, String)]] =
    TreeNodeTag("io.sparkvector.sortMergeJoin.choice")

  /** The decision's reason, carried onto the operator the join became. */
  val SortMergeWhy: TreeNodeTag[String] = TreeNodeTag("io.sparkvector.sortMergeJoin.why")
}

/** Tags and helpers for explaining why an operator was left to Spark. */
object VectorFallback {
  val Tag: TreeNodeTag[String] = TreeNodeTag[String]("io.sparkvector.fallback")

  /** Set on an operator our rule left to Comet on the allowlist's request (#281), so the mixed pass knows to run ours if Comet declines. */
  val Delegated: TreeNodeTag[Boolean] = TreeNodeTag[Boolean]("io.sparkvector.delegated")

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
      val sortMergeMode = VectorConf.sortMergeJoinMode(conf)
      if (sortMergeMode == "hash" || sortMergeMode == "auto") {
        markSortMergeJoins(
          plan,
          orderingNeeded = false,
          orderVisible = false,
          sortMergeMode == "auto",
          maxBuildSize,
          HashBudget(
            VectorConf.joinSpillBytes(conf, session.sparkContext.getConf),
            VectorConf.joinSpillBuckets(conf),
            VectorConf.joinHashMaxBuildBytes(conf, session.sparkContext.getConf)
          ),
          new java.util.IdentityHashMap[SparkPlan, Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide]]
        )
      }
      val bridge = if (VectorConf.cometMixedEnabled(conf)) CometMixedBridge.tryCreate() else null
      val prefer = if (bridge != null) PreferComet.parse(VectorConf.cometPreferComet(conf)) else PreferComet.Empty
      // The allowlist (#281): a listed operator the mixed pass will be able to offer (its inputs are ours, bottom-up)
      // is left to that pass; if Comet declines it there, ours converts it after all, so a requested swap never
      // ends on Spark's operator.
      val delegation: PartialFunction[SparkPlan, SparkPlan] = {
        case p if !prefer.isEmpty && !p.isInstanceOf[VectorPlan] && prefer.wants(p) && offerable(p, bridge) =>
          p.setTagValue(VectorFallback.Delegated, true)
          fallback(p, PreferComet.Reason)
      }
      val conversions: PartialFunction[SparkPlan, SparkPlan] = {
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
          // A lane-less child column (Iceberg's struct `_partition`) is forwarded by the operator (#273).
          forwardingInputReason(m.child).orElse(VectorMergeRowsPlanner.reason(m)) match {
            case Some(reason) => fallback(m, reason)
            case None =>
              VectorMergeRowsExec(
                m.isSourceRowPresent,
                m.isTargetRowPresent,
                m.matchedInstructions,
                m.notMatchedInstructions,
                m.notMatchedBySourceInstructions,
                m.checkCardinality,
                m.output,
                m.child
              )
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
          laneTypeReason(w.child) match {
            case Some(reason) => fallback(w, reason)
            case None => VectorWindowPlanner.plan(w).fold(reason => fallback(w, reason), v => v)
          }

        case g: WindowGroupLimitExec if VectorConf.windowEnabled(conf) =>
          // Spark's per-partition top-k under a ranking window: Partial sits over whatever produced the rows
          // (often ours), Final over Spark's sort -- either way the child is accepted on types alone.
          laneTypeReason(g.child) match {
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
          streamedInputReason(streamedPlan).orElse(laneTypeReason(buildPlan)).orElse(VectorJoinPlanner.buildSizeReason(
            buildPlan,
            maxBuildSize
          )) match {
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
          streamedInputReason(streamedPlan).orElse(laneTypeReason(buildPlan)).orElse(VectorJoinPlanner.buildSizeReason(
            buildPlan,
            maxBuildSize
          )) match {
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
          val shjStreamed = if (shjBuild eq j.left) j.right else j.left
          // No size gate on the build side (#416): past the budget the join splits both sides into buckets on disk.
          streamedInputReason(shjStreamed).orElse(laneExchangeInputReason(shjBuild)) match {
            case Some(reason) => fallback(j, reason)
            case None =>
              VectorJoinPlanner.plan(j) match {
                case Right(v) => v
                case Left(reason) => fallback(j, reason)
              }
          }

        case j: SortMergeJoinExec if VectorConf.sortMergeJoinMode(conf) == "merge" =>
          // Our own merge join (#286): Spark's contract kept -- the sorts below stay (ours over a columnar
          // child, Spark's with a RowToColumnarExec above), the output ordering is the join's own -- so no
          // pre-pass and no build side: every partition streams both sides.
          VectorJoinPlanner.planMergeJoin(j) match {
            case Right(v) => v
            case Left(reason) => fallback(j, reason)
          }

        case j: SortMergeJoinExec if VectorConf.sortMergeJoinMode(conf) == "auto" =>
          // The pre-pass chose (#287): the merge join where a parent relies on the join's ordering, where
          // the order can reach a LIMIT or a sort without an exchange in between (the hash rewrite's tie
          // order would show), or where the hash rewrite is not allowed (no statistics, both sides large,
          // a skew join); the hash rewrite where a side's statistics fit the budget. Either operator
          // carries the reason.
          j.getTagValue(VectorExecRule.SortMergeChoice) match {
            case Some(Right((buildSide, why))) =>
              val (left, right) = sortMergeInputs(j)
              val (buildPlan, streamedPlan) =
                if (buildSide == org.apache.spark.sql.catalyst.optimizer.BuildLeft) (left, right) else (right, left)
              streamedInputReason(streamedPlan).orElse(laneExchangeInputReason(buildPlan)) match {
                case None =>
                  val v =
                    VectorShuffledHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, buildSide, j.condition, left, right)
                  v.setTagValue(VectorExecRule.SortMergeWhy, why)
                  v
                case Some(reason) =>
                  // The rewrite's inputs must be exchanges or ours. The merge join could take the join
                  // instead (it reads Spark's sorted inputs as they are), but measured on TPC-DS q97 --
                  // a full outer join of two 300k-row unique-key sides over Spark's aggregates -- it ran
                  // 2960 ms against 385 ms for Spark's own merge join (the per-run bookkeeping, #286), so
                  // the join stays Spark's until the merge join walks runs without a run object.
                  fallback(resorted(j), reason)
              }
            case Some(Left(why)) if why.startsWith("left to Spark") =>
              // The size gate (#311): the merge join would be ours but the inputs are large or unsized.
              fallback(resorted(j), why)
            case Some(Left(why)) =>
              VectorJoinPlanner.planMergeJoin(j) match {
                case Right(v) =>
                  // The mixed case: a hash join below (its sorts stripped) no longer offers the ordering this
                  // merge join requires, so a sort comes back over that child.
                  val fixed = resortedMerge(v)
                  fixed.setTagValue(VectorExecRule.SortMergeWhy, why)
                  fixed
                case Left(reason) => fallback(resorted(j), reason)
              }
            case None => fallback(resorted(j), "sort-merge join not examined by the pre-pass")
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
              val (buildPlan, streamedPlan) =
                if (buildSide == org.apache.spark.sql.catalyst.optimizer.BuildLeft) (left, right) else (right, left)
              streamedInputReason(streamedPlan).orElse(laneExchangeInputReason(buildPlan)) match {
                case None =>
                  VectorShuffledHashJoinExec(j.leftKeys, j.rightKeys, j.joinType, buildSide, j.condition, left, right)
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
            case org.apache.spark.sql.execution.SortExec(order, false, c, _)
                if sameOrder(order, s.requiredChildOrdering.head) => c
            case c => c
          }
          planAggregate(s, s, conf) match {
            case v: VectorHashAggregateExec =>
              val unsorted = if (child eq s.child) v else v.copy(child = child)
              if (v.emitsResults && s.outputOrdering.nonEmpty)
                VectorSortExec(s.outputOrdering, global = false, unsorted)
              else unsorted
            case other => other
          }
      }
      val converted = plan.transformUp(delegation.orElse(conversions))
      val prefetchDepth = VectorConf.scanPrefetchDepth(conf)
      val withPrefetch = if (prefetchDepth > 0) prefetchScans(converted, prefetchDepth) else converted
      val withSelections = if (VectorConf.selectionEnabled(conf)) markSelectionProducers(withPrefetch) else withPrefetch
      val withMixed = if (bridge != null) mixedChains(withSelections, bridge, prefer, conversions) else withSelections
      val withShuffles =
        if (
          VectorConf.cometShuffleEnabled(conf) && CometShuffle.isEnabled(
            conf,
            session.sparkContext.getConf.get("spark.shuffle.manager", "sort")
          )
        )
          useCometShuffle(withMixed, conf)
        else withMixed
      // Our own columnar exchange (#288) where Comet's did not take the shuffle: the module is
      // optional on the classpath and needs its shuffle manager, so the hook is reflective.
      val withOurShuffles =
        if (VectorConf.shuffleEnabled(conf) && VectorShuffle.isAvailable(session.sparkContext.getConf))
          withShuffles.transformUp {
            // Hash keys that are expressions rather than columns (q47/q57's self-join on `rn + 1`):
            // the keys are materialised by a projection under the exchange, which declares the
            // original partitioning, and dropped by a projection above it. Otherwise the shuffle
            // stays Spark's row exchange and drags a row Sort and a RowToColumnar with it.
            case s: ShuffleExchangeExec
                if s.child.supportsColumnar && VectorConf.projectEnabled(conf) && computedHashKeys(s, conf).isDefined =>
              computedHashKeys(s, conf).get
            case s: ShuffleExchangeExec
                if s.child.supportsColumnar && VectorShuffle.supports(s.outputPartitioning, s.child.output) =>
              VectorShuffle.exchange(s)
          }
        else withShuffles
      if (VectorConf.explainFallback(conf)) {
        VectorFallback.reasons(withOurShuffles).foreach { case (node, reason) =>
          logInfo(s"spark-vector fallback for ${node.nodeName}: $reason")
        }
      }
      withOurShuffles
    }
  }

  /**
   * Mixed chains (#280): a Spark operator left to Spark whose children are ours goes to Comet's native
   * operator when Comet can plan it -- our chain ends in the leaf Comet's native block reads from
   * ([[CometMixedBridge.leaf]]: Comet's sink placeholder over a [[VectorToCometExec]]), and Comet's own
   * rule builds the operator above it. Comet's rule ran before ours and never saw our operators, so
   * this is the only way a Comet operator ends up above ours; the split is the two engines' per-operator
   * toggles (an operator ours refused, or `spark.vector.exec.<op>.enabled=false`, with
   * `spark.comet.exec.<op>.enabled=true`). Two boundaries the planner keeps: an aggregate pair stays on
   * one engine (Comet's final needs Comet's partial buffers, ours needs ours), and a selection is
   * compacted by the export itself. Comet's fallback reasons, when it declines, are its own explain's.
   */
  private def ours(p: SparkPlan): Boolean = p.isInstanceOf[VectorPlan] || p.isInstanceOf[VectorToCometExec]

  /**
   * Whether the mixed pass can offer this operator to Comet: not ours, not Comet's, not an exchange, and every
   * input ours or a block Comet already owns -- a broadcast exchange over such an input is looked through, since
   * Comet converts one only together with the join above it.
   */
  private def offerable(p: SparkPlan, bridge: CometMixedBridge): Boolean =
    !ours(p) && !bridge.isComet(p) && !p.isInstanceOf[org.apache.spark.sql.execution.exchange.Exchange] &&
      p.children.nonEmpty && p.children.forall(c =>
        ours(c) || bridge.isNative(c) || (c match {
          case b: org.apache.spark.sql.execution.exchange.BroadcastExchangeExec =>
            ours(b.child) || bridge.isNative(b.child)
          case _ => false
        })
      )

  private def mixedChains(
      plan: SparkPlan,
      bridge: CometMixedBridge,
      prefer: PreferComet,
      conversions: PartialFunction[SparkPlan, SparkPlan]
  ): SparkPlan = {
    {
      // Comet declined an operator the allowlist asked for: ours takes it after all (#281).
      def declined(p: SparkPlan, reason: String): SparkPlan =
        if (p.getTagValue(VectorFallback.Delegated).isDefined) {
          p.unsetTagValue(VectorFallback.Delegated)
          p.unsetTagValue(VectorFallback.Tag)
          logInfo(s"spark-vector: ${p.nodeName} was requested for Comet but $reason; ours runs")
          conversions.applyOrElse(p, (q: SparkPlan) => fallback(q, s"$reason; ours could not take it either"))
        } else fallback(p, reason)
      plan.transformUp {
        // A shuffle over ours is Comet's native shuffle already (useCometShuffle); a broadcast exchange over
        // ours is offered like an operator, since Comet's broadcast join needs Comet's broadcast below it.
        // Bottom-up, so a parent above an operator this pass just gave to Comet is offered too (up to and
        // including a broadcast exchange): Comet's rule ran before ours and never saw a native child there.
        // A Comet-native child needs no leaf; a parent Comet already declined is declined again, cheaply.
        // Comet converts a broadcast exchange only together with the join above it, so the exchange is looked
        // through here and the join is what gets offered.
        case p if prefer.wants(p) && offerable(p, bridge) =>
          aggregatePairReason(p, bridge) match {
            case Some(reason) => declined(p, reason)
            case None =>
              def leafOf(c: SparkPlan): java.util.Optional[SparkPlan] = c match {
                case b: org.apache.spark.sql.execution.exchange.BroadcastExchangeExec =>
                  val under = leafOf(b.child)
                  if (under.isEmpty) under else java.util.Optional.of(b.withNewChildren(Seq(under.get)))
                case c if ours(c) => bridge.leaf(c, VectorToCometExec(c))
                case c => java.util.Optional.of(c)
              }
              val leaves = p.children.map(leafOf)
              if (leaves.exists(_.isEmpty)) declined(p, "mixed: Comet's sink refuses a column type of the input")
              else {
                val converted = bridge.convertAbove(session, p.withNewChildren(leaves.map(_.get)))
                // Native, or one of Comet's JVM sinks (a union, a limit): Comet's own block pass has already
                // unwrapped the placeholders a JVM sink does not need.
                if (bridge.isComet(converted)) {
                  // The swap the allowlist asked for, shown where the operator now runs (#281).
                  if (p.getTagValue(VectorFallback.Delegated).isDefined)
                    converted.setTagValue(VectorFallback.Tag, PreferComet.Reason)
                  converted
                } else {
                  val reasons = bridge.declineReasons(converted)
                  declined(
                    p,
                    if (reasons.isEmpty) "mixed: Comet declined the operator"
                    else s"mixed: Comet declined -- ${reasons.mkString("; ")}"
                  )
                }
              }
          }
      }
    }
  }

  /**
   * An aggregate half may change engine only when its functions' intermediate buffers are laid out the
   * same way by Spark and by Comet (Comet's own predicate, `allAggsSupportMixedExecution`: sum, min, max,
   * the bit aggregates and a non-decimal avg are; count is not, nor a decimal sum or avg; a Comet final
   * over a foreign partial of an incompatible function is refused by Comet's rule as well). Otherwise the
   * pair stays where the bottom-up transform left it, with the reason.
   */
  private def aggregatePairReason(p: SparkPlan, bridge: CometMixedBridge): Option[String] = p match {
    case a: HashAggregateExec if !bridge.aggregatesMix(a.aggregateExpressions) =>
      Some("mixed: aggregate halves cannot be split across engines (intermediate buffer formats differ)")
    case _ => None
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
    case s: ShuffleExchangeExec
        if s.child.isInstanceOf[VectorPlan] && bridgeable(s.child, s.outputPartitioning, conf) =>
      CometShuffle.native(s, VectorToCometExec(s.child))
    case c
        if CometShuffle.isCometExchange(c) && !CometShuffle.isNative(c) &&
          c.children.head.isInstanceOf[VectorPlan] && bridgeable(c.children.head, c.outputPartitioning, conf) =>
      CometShuffle.toNative(c, VectorToCometExec(c.children.head))
    // Comet's JVM shuffle over a block the mixed pass gave to Comet after Comet's own rule had run: its
    // native shuffle reads a native child directly (#280).
    case c
        if CometShuffle.isCometExchange(c) && !CometShuffle.isNative(c) && VectorConf.cometMixedEnabled(conf) &&
          c.children.head.getClass.getName.startsWith(
            "org.apache.spark.sql.comet."
          ) && c.children.head.supportsColumnar =>
      CometShuffle.toNative(c, c.children.head)
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

  /**
   * The prefetching scan converter (#403, lever 2): under `spark.vector.scan.prefetch > 0`, every
   * Spark vectorized file scan that feeds one of our operators directly is wrapped in a
   * [[VectorPrefetchScanExec]], so the first operator of ours above a scan -- a filter, a projection,
   * an aggregate, a join side, a sort, an expand -- reads batches of our own vectors converted on a
   * helper thread. Applied once the operators are built, which is the same as wrapping the child
   * before each conversion (the conversions only judge the scan's columnar contract and types, which
   * the wrapper reports unchanged) without touching every case. A scan whose parent stayed Spark's is
   * not wrapped: nothing of ours would read the converted batches. Idempotent under adaptive
   * re-planning: a wrapper is never a scan, so a stage plan that already carries one is left alone.
   */
  private def prefetchScans(plan: SparkPlan, depth: Int): SparkPlan = plan.transformUp {
    case parent: VectorPlan if parent.children.exists(prefetchableScan) =>
      parent.withNewChildren(parent.children.map(c => if (prefetchableScan(c)) VectorPrefetchScanExec(c, depth) else c))
  }

  /**
   * A Spark file scan the converter can take: Spark's own vectorized Parquet / ORC scan
   * (`FileSourceScanExec`) or a DSv2 batch scan (`BatchScanExec`: Iceberg's JVM vectorized reader,
   * whose `scan` is an `org.apache.iceberg.spark.source.*` class; any other columnar DSv2 source is
   * converted through the copying adapter just as it is today), columnar, every column with a lane --
   * a column without one (struct, array, map) is passed through by a filter or projection as Spark's
   * vector, which the reader recycles, so such a scan keeps the lazy per-column adaptation. Comet's
   * scans (`CometScanExec`, `CometBatchScanExec`, the native Iceberg scan) are their own classes in
   * Comet's packages and are read zero-copy already; our operators and a local table scan are not
   * scans.
   */
  private def prefetchableScan(plan: SparkPlan): Boolean = {
    def lanes: Boolean = plan.supportsColumnar && plan.output.forall(a => TypeMapping.hasLane(a.dataType))
    def cometClass: Boolean = {
      val name = plan.getClass.getName
      name.startsWith("org.apache.spark.sql.comet.") || name.startsWith("org.apache.comet.")
    }
    plan match {
      case _: org.apache.spark.sql.execution.FileSourceScanExec => lanes && !cometClass
      case _: org.apache.spark.sql.execution.datasources.v2.BatchScanExec => lanes && !cometClass
      case _ => false
    }
  }

  /**
   * A child the allowlist left to Comet (#281) is still Spark's operator when its parent is planned, but
   * it will run on Comet or -- if Comet declines -- on ours, columnar either way; refusing the parent over
   * it would break the chain above the swap (TPC-H q5: a Spark broadcast join over rows above the swap).
   */
  private def columnarChild(plan: SparkPlan): Boolean =
    plan.supportsColumnar || plan.getTagValue(VectorFallback.Delegated).isDefined

  /** None if `plan` is an acceptable columnar input, else the reason it is not. */
  private def columnarInputReason(plan: SparkPlan): Option[String] = {
    if (!columnarChild(plan)) Some(s"child ${plan.nodeName} is not columnar") else typeReason(plan)
  }

  /**
   * The input of an operator that forwards its child's columns without reading them all (filter,
   * project): a column of a type the kernels have no lane for is passed through as Spark's own vector
   * (see [[VectorProjectExec]]), so only the columnar contract is required here; an expression that
   * reads such a column is refused by the compiler with the type named.
   */
  /**
   * The streamed side of a hash join: columnar or an exchange, any column type -- a payload without a
   * lane is passed through as a remapped view of the streamed batch (#273); the keys are checked by
   * the planner. The build side keeps [[laneTypeReason]]: its rows are laid out in lanes.
   */
  private def streamedInputReason(plan: SparkPlan): Option[String] = plan match {
    case _: ShuffleExchangeLike | _: QueryStageExec | _: AQEShuffleReadExec => None
    case other => forwardingInputReason(other)
  }

  private def forwardingInputReason(plan: SparkPlan): Option[String] =
    if (!columnarChild(plan)) Some(s"child ${plan.nodeName} is not columnar") else None

  /**
   * A join input that is an exchange (or its adaptive stage) is accepted on types alone, like the
   * Final aggregate's input: Spark inserts RowToColumnarExec below us when the shuffle is row based.
   */
  /** Like [[exchangeInputReason]] for the joins, whose keys and payloads may be any lane type (a wide decimal included, #259). */
  private def laneExchangeInputReason(plan: SparkPlan): Option[String] = plan match {
    case _: ShuffleExchangeLike | _: QueryStageExec | _: AQEShuffleReadExec => laneTypeReason(plan)
    case other => laneInputReason(other)
  }

  private def exchangeInputReason(plan: SparkPlan): Option[String] = plan match {
    case _: ShuffleExchangeLike | _: QueryStageExec | _: AQEShuffleReadExec => typeReason(plan)
    case other => columnarInputReason(other)
  }

  /** Converts an aggregate (`a`, possibly a SortAggregate re-expressed as a hash one); `original` takes the fallback. */
  private def planAggregate(
      a: org.apache.spark.sql.execution.aggregate.BaseAggregateExec,
      original: SparkPlan,
      conf: SQLConf
  ): SparkPlan = {
    // A merging aggregate (Final, PartialMerge) reads an exchange; Spark inserts RowToColumnarExec below us when the
    // shuffle is row based (Comet's shuffle is columnar already), so only the types matter.
    val isFinal = VectorAggregatePlanner.readsExchange(a)
    // Any lane will do as input (a wide decimal key, input or buffer is a DECIMAL128 lane, #259): the
    // planner below refuses a column a function or a key cannot read, with the function's own reason.
    val inputReason = if (isFinal) laneTypeReason(a.child) else laneInputReason(a.child)
    inputReason match {
      case Some(reason) => fallback(original, reason)
      case None =>
        VectorAggregatePlanner.plan(
          a,
          VectorConf.finalAggregateEnabled(conf),
          VectorConf.strictFloatingPoint(conf)
        ) match {
          // A partial aggregate over our Expand (ROLLUP, CUBE, GROUPING SETS): aggregate the finest grouping
          // once and roll the partials up, instead of hashing every input row once per grouping set (#383).
          case Right(v) if VectorConf.rollupRewriteEnabled(conf) =>
            RollupRewrite(v, VectorConf.strictFloatingPoint(conf))
          case Right(v) => v
          case Left(reason) => fallback(original, reason)
        }
    }
  }

  private def sameOrder(
      a: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder],
      b: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]
  ): Boolean =
    a.length == b.length && a.zip(b).forall { case (x, y) =>
      x.child.semanticEquals(y.child) && x.direction == y.direction && x.nullOrdering == y.nullOrdering
    }

  /** The merge join's inputs without the sorts Spark placed for the merge (exactly the required ones, local). */
  private def sortMergeInputs(j: SortMergeJoinExec): (SparkPlan, SparkPlan) = {
    def strip(child: SparkPlan, required: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]): SparkPlan =
      child match {
        case VectorSortExec(order, false, c) if sameOrder(order, required) => c
        case SortExec(order, false, c, _) if sameOrder(order, required) => c
        case c => c
      }
    (strip(j.left, j.requiredChildOrdering.head), strip(j.right, j.requiredChildOrdering(1)))
  }

  /** Why a filter's predicate would not compile over its child's output, input aside. */
  private def filterReason(f: FilterExec): Option[String] =
    ExpressionCompiler.compilePredicate(f.condition, f.child.output).left.toOption

  /**
   * Our exchange for a hash shuffle whose keys include computed expressions, when every such key
   * compiles to a lane: `VectorShuffleExchange(original partitioning, materializedKeys = n) <-
   * VectorProject(child.output ++ keys) <- child`. The exchange finds a computed key's column in the
   * projection under it (`keysAsColumns`), partitions on it, and neither writes nor outputs it, so
   * its output is the original and AQE's stage root stays an exchange. It hashes the same values
   * Spark's expression would have, so it pairs with the other side of a join whichever exchange that
   * side runs on. None when the keys are plain columns (the plain case applies) or one does not compile.
   */
  private def computedHashKeys(s: ShuffleExchangeExec, conf: SQLConf): Option[SparkPlan] = s.outputPartitioning match {
    case h: org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
        if h.expressions.exists(!_.isInstanceOf[Attribute]) =>
      val input = s.child.output
      val compiles = h.expressions.forall {
        case _: Attribute => true
        case e => TypeMapping.isSupported(e.dataType) && ExpressionCompiler.compile(e, input).isRight
      }
      if (!compiles) None
      else {
        val keys = h.expressions.zipWithIndex.collect {
          case (e, i) if !e.isInstanceOf[Attribute] => Alias(e, s"_shuffle_key_$i")()
        }
        val projected = VectorProjectExec(input ++ keys, s.child)
        val asColumns = h.copy(expressions = h.expressions.map {
          case a: Attribute => a
          case e => keys.find(_.child.semanticEquals(e)).get.toAttribute
        })
        if (!VectorShuffle.supports(asColumns, projected.output)) None
        else Some(VectorShuffle.exchange(s.copy(child = projected), keys.size))
      }
    case _ => None
  }

  /** Why a projection would not compile over its child's output (an expression, an output type), input aside. */
  private def projectReason(p: ProjectExec): Option[String] = {
    val failures = p.projectList.filterNot(e =>
      VectorProjectExec.isPassThrough(e) || VectorProjectExec.constantSlot(e).isDefined
    ).flatMap { e =>
      val compiled = ExpressionCompiler.compile(e, p.child.output)
      // A decimal output wider than 18 digits is a DECIMAL128 lane when its expression compiled (#258).
      val typeCheck =
        if (
          TypeMapping.isSupported(
            e.dataType
          ) || (e.dataType.isInstanceOf[org.apache.spark.sql.types.DecimalType] && TypeMapping.hasLane(e.dataType))
        ) Right(())
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
      if (
        org.apache.spark.sql.catalyst.expressions.SortOrder.orderingSatisfies(
          child.outputOrdering,
          required
        ) || !child.supportsColumnar
      ) child
      else VectorSortExec(required, global = false, child)
    val left = fix(j.left, j.requiredChildOrdering.head)
    val right = fix(j.right, j.requiredChildOrdering(1))
    if ((left eq j.left) && (right eq j.right)) j else j.copy(left = left, right = right)
  }

  private type SortMergeMemo =
    java.util.IdentityHashMap[SparkPlan, Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide]]

  /** The hash join's per-task budgets `mode=auto` judges a build side by: in memory, one bucketing pass, cap. */
  private case class HashBudget(spillBytes: Long, buckets: Int, hashMaxBuildBytes: Long)

  /**
   * Whether a sort-merge join could become our hash join, ordering aside: its inputs (sorts stripped)
   * are exchanges, or merge joins that could themselves convert -- a chain of merge joins on the same
   * key has no shuffle between its links, and the upper link reads the lower one's output, which is
   * columnar once the lower converts (TPC-DS q10, q35, q69, q95 chain semi and existence joins on the
   * customer key). Bottom-up, memoised over the original nodes; the ordering question is the pre-pass's.
   */
  private def sortMergeEligibility(
      j: SortMergeJoinExec,
      maxBuildSize: Long,
      memo: SortMergeMemo
  ): Either[String, org.apache.spark.sql.catalyst.optimizer.BuildSide] = {
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
      // The streamed side is judged as the hash join reads it: any column type, a payload without a lane
      // passed through (#273); the build side needs lanes. So the build side is chosen first.
      def inputReason(p: SparkPlan, streamed: Boolean): Option[String] = {
        def types(plan: SparkPlan): Option[String] = if (streamed) None else laneTypeReason(plan)
        p match {
          case inner: SortMergeJoinExec =>
            sortMergeEligibility(inner, maxBuildSize, memo).left.toOption.map(r =>
              s"child SortMergeJoin stays with Spark: $r"
            ).orElse(types(inner))
          case proj: ProjectExec if VectorConf.projectEnabled(conf) =>
            projectReason(proj).orElse(inputReason(proj.child, streamed))
          case filt: FilterExec if VectorConf.filterEnabled(conf) =>
            filterReason(filt).orElse(inputReason(filt.child, streamed))
          case _: ShuffleExchangeLike | _: QueryStageExec | _: AQEShuffleReadExec => types(p)
          case _: UnionExec | _: ExpandExec | _: CoalesceExec | _: SampleExec | _: SortExec | _: LocalLimitExec | _: GlobalLimitExec |
              _: BroadcastHashJoinExec | _: BroadcastNestedLoopJoinExec | _: ShuffledHashJoinExec | _: HashAggregateExec | _: SortAggregateExec =>
            types(p)
          case other => if (streamed) forwardingInputReason(other) else laneInputReason(other)
        }
      }
      val decision = VectorJoinPlanner.sortMergeBuildSide(
        j.leftKeys,
        j.rightKeys,
        j.joinType,
        j.condition,
        j.isSkewJoin,
        left,
        right,
        maxBuildSize
      ).flatMap { buildSide =>
        val (buildPlan, streamedPlan) =
          if (buildSide == org.apache.spark.sql.catalyst.optimizer.BuildLeft) (left, right) else (right, left)
        inputReason(streamedPlan, streamed = true).orElse(inputReason(buildPlan, streamed = false)) match {
          case Some(reason) => Left(reason)
          case None => Right(buildSide)
        }
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
  private def markSortMergeJoins(
      plan: SparkPlan,
      orderingNeeded: Boolean,
      orderVisible: Boolean,
      auto: Boolean,
      maxBuildSize: Long,
      budget: HashBudget,
      memo: SortMergeMemo
  ): Unit = {
    // Order visibility (#287): below a limit, a take-ordered, a sort, or a range-partitioned exchange (a
    // global sort's), a join's row order can show -- ties under ORDER BY, the rows a LIMIT picks -- and
    // the hash rewrite's order differs from Spark's. An aggregate or any other exchange ends it.
    val visibleHere = orderVisible || (plan match {
      case _: LocalLimitExec | _: GlobalLimitExec | _: CollectLimitExec | _: TakeOrderedAndProjectExec | _: SortExec =>
        true
      case e: ShuffleExchangeExec =>
        e.outputPartitioning.isInstanceOf[org.apache.spark.sql.catalyst.plans.physical.RangePartitioning]
      case _ => false
    })
    val converts = plan match {
      case j: SortMergeJoinExec =>
        val eligibility = sortMergeEligibility(j, maxBuildSize, memo)
        val decision = if (orderingNeeded) Left("output ordering required by the parent operator") else eligibility
        j.setTagValue(VectorExecRule.SortMergeDecision, decision)
        if (auto) {
          val choice: Either[String, (org.apache.spark.sql.catalyst.optimizer.BuildSide, String)] =
            if (orderingNeeded) Left("as merge join: ordering relied on by the parent")
            else if (visibleHere) Left("as merge join: the row order reaches a limit or a sort")
            else eligibility match {
              case Right(side) => hashOrMerge(j, side, budget)
              case Left(reason) => Left(s"as merge join: $reason")
            }
          j.setTagValue(VectorExecRule.SortMergeChoice, choice)
          choice.isRight
        } else decision.isRight
      case _ => false
    }
    val childVisible = visibleHere && (plan match {
      case _: HashAggregateExec | _: SortAggregateExec | _: ObjectHashAggregateExec => false
      case e: ShuffleExchangeExec =>
        e.outputPartitioning.isInstanceOf[org.apache.spark.sql.catalyst.plans.physical.RangePartitioning]
      case _: BroadcastExchangeExec => false
      case _ => true
    })
    plan.children.zipWithIndex.foreach { case (child, i) =>
      val required = !converts && plan.requiredChildOrdering(i).nonEmpty
      val passes = orderingNeeded && plan.outputOrdering.nonEmpty &&
        org.apache.spark.sql.catalyst.expressions.SortOrder.orderingSatisfies(child.outputOrdering, plan.outputOrdering)
      markSortMergeJoins(child, required || passes, childVisible, auto, maxBuildSize, budget, memo)
    }
  }

  /**
   * The size rule of `mode=auto` (#416), once the order is known not to show and a build side is chosen:
   * what the build side weighs per task, by statistics, against the hash join's two budgets. Within
   * `spark.vector.join.spillBytes` it builds in memory; within `spark.vector.join.hashMaxBuildSize`
   * (one bucketing pass: `spillBuckets` buckets of at most `spillBytes`) it splits into buckets on
   * disk; past that, or with no estimate to judge by, the merge join over the spilling sort takes it --
   * its memory is bounded by the sort's budget whatever the inputs weigh, where a hash join's grows
   * with the build side.
   */
  private def hashOrMerge(
      j: SortMergeJoinExec,
      side: org.apache.spark.sql.catalyst.optimizer.BuildSide,
      budget: HashBudget
  ): Either[String, (org.apache.spark.sql.catalyst.optimizer.BuildSide, String)] = {
    val name = if (side == org.apache.spark.sql.catalyst.optimizer.BuildLeft) "left" else "right"
    val buildPlan = if (side == org.apache.spark.sql.catalyst.optimizer.BuildLeft) j.left else j.right
    val tasks = math.max(1, buildPlan.outputPartitioning.numPartitions)
    VectorJoinPlanner.estimatedBuildSize(buildPlan).map(total => (total, total / tasks)) match {
      case Some((total, perTask)) if perTask <= budget.spillBytes =>
        Right((side, s"as hash join: built from the $name side in memory ($total bytes over $tasks tasks)"))
      case Some((total, perTask)) if perTask <= budget.hashMaxBuildBytes =>
        Right((
          side,
          s"as hash join: built from the $name side, split into ${budget.buckets} buckets on disk " +
            s"($total bytes over $tasks tasks, past ${VectorConf.JoinSpillBytes})"
        ))
      case Some((total, _)) =>
        Left(
          s"as merge join: the $name side weighs $total bytes over $tasks tasks, past the hash join's " +
            s"${VectorConf.JoinHashMaxBuildSize} (${budget.hashMaxBuildBytes} per task)"
        )
      case None if budget.hashMaxBuildBytes == Long.MaxValue =>
        Right((side, s"as hash join: built from the $name side (no size cap)"))
      case None => Left(s"as merge join: no size estimate for the $name side to judge the hash join's budget by")
    }
  }

  /** A merge join of ours whose child lost its ordering (a hash join below, sorts stripped) gets the sort back. */
  private def resortedMerge(v: VectorSortMergeJoinExec): VectorSortMergeJoinExec = {
    def fix(child: SparkPlan, required: Seq[org.apache.spark.sql.catalyst.expressions.SortOrder]): SparkPlan =
      if (
        org.apache.spark.sql.catalyst.expressions.SortOrder.orderingSatisfies(
          child.outputOrdering,
          required
        ) || !child.supportsColumnar
      ) child
      else VectorSortExec(required, global = false, child)
    val left = fix(v.left, v.requiredChildOrdering.head)
    val right = fix(v.right, v.requiredChildOrdering(1))
    if ((left eq v.left) && (right eq v.right)) v else v.copy(left = left, right = right)
  }

  /** Like [[columnarInputReason]] for an operator that only moves columns: any lane type will do. */
  private def laneInputReason(plan: SparkPlan): Option[String] =
    if (!columnarChild(plan)) Some(s"child ${plan.nodeName} is not columnar")
    else plan.output.find(a => !TypeMapping.hasLane(a.dataType)).map(a =>
      s"unsupported column type ${a.dataType.simpleString} for ${a.name}"
    )

  /** Like [[typeReason]] for an operator that reads any lane type (the aggregate over the DECIMAL128 lane, #259). */
  private def laneTypeReason(plan: SparkPlan): Option[String] =
    plan.output.find(a => !TypeMapping.hasLane(a.dataType)).map(a =>
      s"unsupported column type ${a.dataType.simpleString} for ${a.name}"
    )

  private def typeReason(
      plan: SparkPlan,
      allowed: Set[org.apache.spark.sql.catalyst.expressions.ExprId] = Set.empty
  ): Option[String] =
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
      // A command (MERGE INTO, a write) holds its physical plan beside its children.
      case c: org.apache.spark.sql.execution.CommandResultExec => Seq(c.commandPhysicalPlan)
      case _ => Nil
    }
    plan +: (plan.children ++ inner).flatMap(allNodes)
  }
}
