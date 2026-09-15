package org.apache.spark.sql.vector

import io.sparkvector.spark.VectorConf
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.expr.ExpressionCompiler
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.{ColumnarRule, FilterExec, ProjectExec, SparkPlan}
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
      }
      if (VectorConf.explainFallback(conf)) {
        VectorFallback.reasons(converted).foreach { case (node, reason) =>
          logInfo(s"spark-vector fallback for ${node.nodeName}: $reason")
        }
      }
      converted
    }
  }

  private def fallback(plan: SparkPlan, reason: String): SparkPlan = {
    plan.setTagValue(VectorFallback.Tag, reason)
    plan
  }

  /** None if `plan` is an acceptable columnar input, else the reason it is not. */
  private def columnarInputReason(plan: SparkPlan): Option[String] = {
    if (!plan.supportsColumnar) {
      Some(s"child ${plan.nodeName} is not columnar")
    } else {
      plan.output.find(a => !TypeMapping.isSupported(a.dataType)).map { a =>
        s"unsupported column type ${a.dataType.simpleString} for ${a.name}"
      }
    }
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
