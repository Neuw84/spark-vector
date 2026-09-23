package org.apache.spark.sql.vector.ui

import scala.collection.mutable

import org.apache.commons.text.StringEscapeUtils
import org.apache.spark.sql.execution.{SparkPlan, SparkPlanInfo}
import org.apache.spark.sql.vector.{VectorFallback, VectorPlan}

/**
 * Which engine executes an operator. This is what the UI colours nodes by.
 *
 * The distinction is drawn from the operator's identity, not from a tag we set: a spark-vector
 * operator *is* a [[VectorPlan]], a Comet operator *is* a class in Comet's packages, and an
 * operator we declined to convert is left as the original Spark class (carrying a
 * [[VectorFallback]] tag with the reason). See `VectorExecRule`.
 */
sealed abstract class Engine(val id: String, val label: String, val shortLabel: String) {

  /** CSS class applied to the node in the DAG and to its legend swatch. */
  def cssClass: String = s"sv-engine-$id"

  /** Counted against "fully accelerated": only plain Spark operators are. */
  def isAccelerated: Boolean = true
}

object Engine {

  /** Executed by our SIMD kernels on Arrow-layout batches. */
  case object Vector extends Engine("vector", "Java Vector API", "Vector")

  /** Executed by Comet's native (DataFusion) operators. */
  case object Comet extends Engine("comet", "Comet native", "Comet")

  /**
   * A Spark scan that already produces columnar batches, i.e. the vectorized Parquet reader. Not
   * our engine, but the designed input for it rather than an operator we failed to convert, so it
   * does not disqualify a plan from being fully accelerated.
   */
  case object ColumnarSource extends Engine("source", "Spark columnar scan", "Scan")

  /** The zero-copy hand-off from our batches to Comet's, and Arrow/Comet input adapters. */
  case object Bridge extends Engine("bridge", "Vector to Comet bridge", "Bridge")

  /** Left to Spark: row-based codegen or a columnar operator we do not implement. */
  case object Spark extends Engine("spark", "Spark (not accelerated)", "Spark") {
    override def isAccelerated: Boolean = false
  }

  /**
   * Row/columnar transitions. They cost real time but are not an operator we either did or did
   * not accelerate, so they do not by themselves disqualify a plan.
   */
  case object Transition extends Engine("transition", "Row/columnar transition", "Transition")

  /**
   * Adaptive execution's shuffle reader (coalescing, skew splitting) and exchange/subquery reuse
   * markers. They move nothing between row and columnar form: the reader hands over whatever the
   * exchange wrote, columnar when the exchange is. Plumbing, shown in a neutral colour.
   */
  case object ShuffleRead extends Engine("shuffleread", "Shuffle read / reuse (AQE)", "Read")

  val all: Seq[Engine] = Seq(Vector, Comet, Bridge, ColumnarSource, Spark, Transition, ShuffleRead)

  /** Engines that are shown but are neither an acceleration nor a missed one. */
  val plumbing: Set[Engine] = Set(ColumnarSource, Transition, ShuffleRead)
}

/** One operator in the visualised plan. */
final case class PlanNode(
    id: Long,
    name: String,
    detail: String,
    engine: Engine,
    fallbackReason: Option[String]
) {

  /** `label` is what shows in the box, `tooltip` what shows on hover. */
  private def dotAttrs: String = {
    val tooltip = {
      val reason = fallbackReason.map(r => s"\nnot accelerated: $r").getOrElse("")
      s"${engine.label}\n$detail$reason"
    }
    Seq(
      s"""id="node$id"""",
      s"""label="${StringEscapeUtils.escapeJava(name)}"""",
      s"""tooltip="${StringEscapeUtils.escapeJava(tooltip)}"""",
      s"""class="${engine.cssClass}""""
    ).mkString(" ")
  }

  def toDot: String = s"  $id [$dotAttrs];"
}

/** Child-to-parent edge, matching the direction Spark's own plan graph uses. */
final case class PlanEdge(fromId: Long, toId: Long) {
  def toDot: String = s"  $fromId->$toId;"
}

/**
 * A classified physical plan, ready to render.
 *
 * @param nodes operators, root first
 * @param edges child -> parent
 */
final case class AcceleratedPlan(nodes: Seq[PlanNode], edges: Seq[PlanEdge]) {

  def countBy(engine: Engine): Int = nodes.count(_.engine == engine)

  /**
   * Operators that acceleration applies to: everything but plumbing (columnar sources and
   * row/columnar transitions), which is neither accelerated nor a missed opportunity.
   */
  def operatorCount: Int = nodes.count(n => !Engine.plumbing.contains(n.engine))

  /**
   * True when every operator runs on our kernels or on Comet. Plumbing does not count against it,
   * so a plan reading a vectorized Parquet scan and filtering it with our kernels qualifies; one
   * with a Spark exchange or sort does not, because those are operators we do not implement.
   *
   * Requires at least one accelerated operator, so a plan made only of a scan does not qualify.
   */
  def fullyAccelerated: Boolean =
    !nodes.exists(_.engine == Engine.Spark) && operatorCount > 0

  /** Share of the operators above that we or Comet execute, for the summary column. */
  def acceleratedFraction: Double = {
    val total = operatorCount
    if (total == 0) 0.0
    else nodes.count(n => !Engine.plumbing.contains(n.engine) && n.engine.isAccelerated).toDouble / total
  }

  /**
   * Operators the planner rule tried to convert and could not, with the reason. A Spark-coloured
   * operator without a reason (a scan, an exchange, a sort) was never a candidate.
   */
  def fallbacks: Seq[(String, String)] =
    nodes.flatMap(n => n.fallbackReason.map(r => (n.name, r)))

  def toDotFile: String =
    (Seq("digraph G {") ++ nodes.map(_.toDot) ++ edges.map(_.toDot) ++ Seq("}")).mkString("\n")
}

/**
 * Classifies a physical plan into [[AcceleratedPlan]].
 *
 * Two entry points, because the two places we can observe a query give different fidelity:
 * [[fromPlan]] sees the real operator objects (exact types, fallback tags) and is used once a
 * query finishes or while it is still live; [[fromInfo]] only has the serialised
 * [[SparkPlanInfo]] tree from the listener event and has to match on node names, which is enough
 * to render a running query before its final plan exists.
 */
object PlanAcceleration {

  /** Comet's operators all live under these packages; we never depend on Comet at compile time. */
  private val CometPackages = Seq("org.apache.spark.sql.comet.", "org.apache.comet.")

  /** Wrappers that carry no execution of their own; unwrapped so the DAG shows real operators. */
  private val UnwrappedNames = Set(
    "AdaptiveSparkPlan",
    "WholeStageCodegen",
    "InputAdapter",
    "ResultQueryStage",
    "ShuffleQueryStage",
    "BroadcastQueryStage",
    "TableCacheQueryStage"
  )

  /**
   * Row/columnar transitions: the only nodes that convert between rows and batches. Neither is an
   * operator we either did or did not convert, so they are shown but do not count as a fallback.
   */
  private val TransitionNames = Set("ColumnarToRow", "RowToColumnar")

  /** AQE's shuffle reader and reuse markers: plumbing that changes nothing about the format. */
  private val ShuffleReadNames = Set("AQEShuffleRead", "ReusedExchange", "ReusedSubquery")

  /**
   * Comet's JVM ("columnar") shuffle reads its child through `execute()`, i.e. as rows, and its
   * writer re-encodes them as Arrow. Over one of our operators that is a hidden row round-trip the
   * node's own colour would not show, so the tooltip says so. Its native shuffle takes batches.
   */
  private val CometJvmShuffleNames = Set("CometColumnarExchange")

  /** Prepended to the tooltip of a Comet JVM shuffle over one of our operators. */
  val RowRoundTripNote: String =
    "reads its child as rows (Comet's JVM shuffle calls execute()): our batches are converted to " +
      "rows and re-encoded as Arrow. Comet's native shuffle would take the batches directly."

  private def isUnwrapped(nodeName: String): Boolean =
    UnwrappedNames.exists(n => nodeName == n || nodeName.startsWith(s"$n ")) ||
      nodeName.startsWith("WholeStageCodegen")

  private def isTransition(nodeName: String): Boolean = TransitionNames.contains(nodeName)

  private def isShuffleRead(nodeName: String): Boolean = ShuffleReadNames.contains(nodeName)

  private def isCometJvmShuffle(nodeName: String): Boolean = CometJvmShuffleNames.contains(nodeName)

  private def isCometClass(clazz: Class[_]): Boolean = {
    val name = clazz.getName
    CometPackages.exists(name.startsWith)
  }

  /** Comet's own node names all start with `Comet`, which is what the info-based path matches. */
  private def isCometName(nodeName: String): Boolean = nodeName.startsWith("Comet")

  private def isVectorName(nodeName: String): Boolean =
    nodeName.startsWith("Vector") && nodeName != "VectorToComet"

  // ---------------------------------------------------------------- from real operators

  /**
   * Classify from the live plan. Exact, because it can test the operator's type and read the
   * fallback tag the planner rule left behind.
   */
  def fromPlan(plan: SparkPlan): AcceleratedPlan = {
    val nodes = mutable.ArrayBuffer.empty[PlanNode]
    val edges = mutable.ArrayBuffer.empty[PlanEdge]
    var nextId = 0L

    def engineOf(p: SparkPlan): Engine = p match {
      case _: VectorPlan => Engine.Vector
      case _ if p.nodeName == "VectorToComet" => Engine.Bridge
      // The mixed-chain leaf (#280): Comet's one-child union over our export node is the hand-off, not an operator.
      case _ if isCometClass(p.getClass) && p.children.size == 1 && p.children.head.nodeName == "VectorToComet" =>
        Engine.Bridge
      case _ if isCometClass(p.getClass) => Engine.Comet
      case _ if isTransition(p.nodeName) => Engine.Transition
      case _ if isShuffleRead(p.nodeName) => Engine.ShuffleRead
      // A leaf that already emits batches is the vectorized reader our operators are built on; a
      // non-vectorized scan has supportsColumnar false and stays an unaccelerated Spark operator.
      case _ if p.children.isEmpty && p.supportsColumnar => Engine.ColumnarSource
      case _ => Engine.Spark
    }

    def detailOf(p: SparkPlan): String = {
      val base = truncate(p.simpleString(maxFields = 20))
      if (isCometJvmShuffle(p.nodeName) && p.children.exists(_.isInstanceOf[VectorPlan])) {
        RowRoundTripNote + "\n" + base
      } else base
    }

    def visit(p: SparkPlan, parent: Option[Long]): Unit = {
      if (isUnwrapped(p.nodeName)) {
        childrenOf(p).foreach(visit(_, parent))
      } else {
        val id = nextId
        nextId += 1
        nodes += PlanNode(
          id,
          p.nodeName,
          detailOf(p),
          engineOf(p),
          VectorFallback.reason(p)
        )
        parent.foreach(pid => edges += PlanEdge(id, pid))
        childrenOf(p).foreach(visit(_, Some(id)))
      }
    }

    visit(plan, None)
    AcceleratedPlan(nodes.toSeq, edges.toSeq)
  }

  /** Children plus the inner plan of adaptive stages and reused exchanges. */
  private def childrenOf(p: SparkPlan): Seq[SparkPlan] = {
    import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
    import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
    p match {
      case a: AdaptiveSparkPlanExec => Seq(a.executedPlan)
      case q: QueryStageExec => Seq(q.plan)
      case r: ReusedExchangeExec => Seq(r.child)
      case _ => p.children
    }
  }

  // ---------------------------------------------------------------- from listener event info

  /**
   * Classify from the serialised plan on a listener event. Matches on node names, which is why
   * [[fromPlan]] is preferred whenever the real plan is reachable.
   */
  def fromInfo(info: SparkPlanInfo): AcceleratedPlan = {
    val nodes = mutable.ArrayBuffer.empty[PlanNode]
    val edges = mutable.ArrayBuffer.empty[PlanEdge]
    var nextId = 0L

    // Without the operator objects there is no `supportsColumnar` to consult, so a leaf scan is
    // assumed to be the vectorized reader. Pages built this way are labelled approximate.
    def engineOf(i: SparkPlanInfo): Engine = {
      val name = i.nodeName
      if (name == "VectorToComet") Engine.Bridge
      else if (isVectorName(name)) Engine.Vector
      else if (isCometName(name)) Engine.Comet
      else if (isTransition(name)) Engine.Transition
      else if (isShuffleRead(name)) Engine.ShuffleRead
      // "Scan parquet", "Scan In-memory table <name>", or an unnamed cache's "InMemoryTableScan".
      else if (i.children.isEmpty && (name.startsWith("Scan") || name.startsWith("InMemoryTableScan")))
        Engine.ColumnarSource
      else Engine.Spark
    }

    def visit(i: SparkPlanInfo, parent: Option[Long]): Unit = {
      if (isUnwrapped(i.nodeName)) {
        i.children.foreach(visit(_, parent))
      } else {
        val id = nextId
        nextId += 1
        nodes += PlanNode(id, i.nodeName, truncate(i.simpleString), engineOf(i), None)
        parent.foreach(pid => edges += PlanEdge(id, pid))
        i.children.foreach(visit(_, Some(id)))
      }
    }

    visit(info, None)
    AcceleratedPlan(nodes.toSeq, edges.toSeq)
  }

  private def truncate(s: String): String = if (s.length <= 400) s else s.take(400) + "..."
}
