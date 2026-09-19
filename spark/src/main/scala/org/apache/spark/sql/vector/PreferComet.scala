package org.apache.spark.sql.vector

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, ObjectHashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.execution.window.{WindowExec, WindowGroupLimitExec}
import org.apache.spark.sql.types.{DecimalType, StringType}

/**
 * The operator allowlist of `spark.vector.comet.preferComet` (#281): which Spark operators above one of
 * our chains are offered to Comet by the mixed-chain pass (#280), and which of the operators our rule
 * would take itself are left to Comet instead. Comma-separated entries, each an operator kind with an
 * optional predicate the planner evaluates from the plan alone:
 *
 *   - kinds: `filter`, `project`, `sort`, `sortMergeJoin`, `hashJoin`, `broadcastHashJoin`, `window`,
 *     `expand`, `union`, `limit`; `all` stands for every kind with no predicate.
 *   - predicates: `wideDecimal` (an input or output column of `decimal(p > 18)`), `strings` (a string
 *     input), `estimatedRows>N` (the operator's logical estimate, rows or bytes over the row width).
 *
 * `aggregate` is refused with a warning: an aggregate pair cannot be split across the engines; under
 * `all` an aggregate half is offered and Comet's own buffer rule decides whether it may cross (#280). An empty list under
 * `spark.vector.comet.mixed.enabled=true` allows mixed plans but requests none.
 */
final case class PreferComet(entries: Seq[PreferComet.Entry]) {

  /** Whether the allowlist asks for this operator to run on Comet. */
  def wants(plan: SparkPlan): Boolean = {
    val kind = PreferComet.kindOf(plan)
    kind.nonEmpty && entries.exists(e => (e.kind == "all" || e.kind == kind) && e.predicate.forall(_.holds(plan)))
  }

  def isEmpty: Boolean = entries.isEmpty
}

object PreferComet extends Logging {

  val Reason = "delegated to Comet (spark.vector.comet.preferComet)"

  val Kinds: Set[String] = Set("filter", "project", "sort", "sortMergeJoin", "hashJoin", "broadcastHashJoin",
    "window", "expand", "union", "limit")

  sealed trait Predicate { def holds(plan: SparkPlan): Boolean }

  /** An input or output column of a wide decimal. */
  case object WideDecimal extends Predicate {
    override def holds(plan: SparkPlan): Boolean =
      (plan.output ++ plan.children.flatMap(_.output)).exists(isWide)
    private def isWide(a: Attribute): Boolean = a.dataType match {
      case d: DecimalType => d.precision > 18
      case _ => false
    }
  }

  /** A string input column. */
  case object Strings extends Predicate {
    override def holds(plan: SparkPlan): Boolean =
      plan.children.flatMap(_.output).exists(_.dataType.isInstanceOf[StringType])
  }

  /** The logical estimate of the operator's input rows exceeds `n` (rows, or bytes over the row width). */
  final case class EstimatedRows(n: Long) extends Predicate {
    override def holds(plan: SparkPlan): Boolean = estimatedRows(plan).exists(_ > n)
  }

  def estimatedRows(plan: SparkPlan): Option[Long] = plan.logicalLink.flatMap { l =>
    val stats = l.stats
    stats.rowCount.map(_.toLong).orElse {
      val width = math.max(1L, l.output.map(_.dataType.defaultSize.toLong).sum)
      if (stats.sizeInBytes == Long.MaxValue) None else Some((stats.sizeInBytes / width).toLong)
    }
  }

  final case class Entry(kind: String, predicate: Option[Predicate])

  val Empty: PreferComet = PreferComet(Nil)

  private val EstimatedRowsPattern = """estimatedRows\s*>\s*(\d+)""".r

  /** Parses the conf value; unknown kinds and predicates are dropped with a warning, never fatal. */
  def parse(value: String): PreferComet = {
    if (value == null || value.trim.isEmpty) Empty
    else PreferComet(value.split(",").map(_.trim).filter(_.nonEmpty).toSeq.flatMap { raw =>
      val (kind, pred) = raw.indexOf(':') match {
        case -1 => (raw, None)
        case i => (raw.substring(0, i).trim, Some(raw.substring(i + 1).trim))
      }
      if (kind == "aggregate") {
        logWarning(s"spark.vector.comet.preferComet: '$raw' ignored -- an aggregate pair cannot be split across the engines (#280)")
        None
      } else if (kind != "all" && !Kinds.contains(kind)) {
        logWarning(s"spark.vector.comet.preferComet: unknown operator kind '$kind' in '$raw' ignored; known: ${Kinds.toSeq.sorted.mkString(", ")}, all")
        None
      } else pred match {
        case None => Some(Entry(kind, None))
        case Some("wideDecimal") => Some(Entry(kind, Some(WideDecimal)))
        case Some("strings") => Some(Entry(kind, Some(Strings)))
        case Some(EstimatedRowsPattern(n)) => Some(Entry(kind, Some(EstimatedRows(n.toLong))))
        case Some(other) =>
          logWarning(s"spark.vector.comet.preferComet: unknown predicate '$other' in '$raw' ignored; known: wideDecimal, strings, estimatedRows>N")
          None
      }
    })
  }

  /** The allowlist kind of a Spark operator, empty for operators no entry can name. */
  def kindOf(plan: SparkPlan): String = plan match {
    case _: FilterExec => "filter"
    case _: ProjectExec => "project"
    case _: SortExec => "sort"
    case _: SortMergeJoinExec => "sortMergeJoin"
    case _: ShuffledHashJoinExec => "hashJoin"
    case _: BroadcastHashJoinExec | _: BroadcastNestedLoopJoinExec => "broadcastHashJoin"
    case _: WindowExec | _: WindowGroupLimitExec => "window"
    case _: ExpandExec => "expand"
    case _: UnionExec => "union"
    case _: LocalLimitExec | _: GlobalLimitExec | _: CollectLimitExec | _: TakeOrderedAndProjectExec => "limit"
    // Only `all` reaches an aggregate: the half that may cross is decided by Comet's buffer rule (#280).
    case _: HashAggregateExec | _: ObjectHashAggregateExec | _: SortAggregateExec => "aggregate"
    case _ => ""
  }
}
