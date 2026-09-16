package org.apache.spark.sql.vector.ui

import scala.util.control.NonFatal

import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerEvent}
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.sql.execution.ui.{
  SparkListenerSQLAdaptiveExecutionUpdate,
  SparkListenerSQLExecutionEnd,
  SparkListenerSQLExecutionStart
}

/**
 * Records the acceleration of every SQL execution for [[VectorAccelerationTab]].
 *
 * Spark's own SQL listener events carry a [[org.apache.spark.sql.execution.SparkPlanInfo]], a
 * serialised tree that keeps node names but loses the operator objects, so it cannot be asked
 * whether a node is a `VectorExec` or why one was left to Spark. The listener therefore upgrades
 * its classification as better information arrives:
 *
 *  - on start, from the event's plan info, so a running query already appears in the tab;
 *  - on each adaptive update, from the live `QueryExecution` if it is still registered;
 *  - on end, from `SparkListenerSQLExecutionEnd.qe.executedPlan`, the final plan with real types
 *    and fallback tags. This is the classification the tab ends up showing.
 *
 * Failures are swallowed: a UI listener must never break a query.
 */
class VectorAccelerationListener(store: VectorAccelerationStore) extends SparkListener with Logging {

  override def onOtherEvent(event: SparkListenerEvent): Unit = event match {
    case e: SparkListenerSQLExecutionStart => onStart(e)
    case e: SparkListenerSQLAdaptiveExecutionUpdate => onAdaptiveUpdate(e)
    case e: SparkListenerSQLExecutionEnd => onEnd(e)
    case _ =>
  }

  private def onStart(e: SparkListenerSQLExecutionStart): Unit = guard(e.executionId) {
    store.update(ExecutionSummary(
      executionId = e.executionId,
      description = e.description,
      submissionTime = e.time,
      completionTime = None,
      failed = false,
      plan = livePlan(e.executionId).getOrElse(PlanAcceleration.fromInfo(e.sparkPlanInfo)),
      exact = livePlan(e.executionId).isDefined,
      physicalPlanDescription = e.physicalPlanDescription))
  }

  /** AQE replans mid-flight; each update supersedes what we recorded before. */
  private def onAdaptiveUpdate(e: SparkListenerSQLAdaptiveExecutionUpdate): Unit =
    guard(e.executionId) {
      val exact = livePlan(e.executionId)
      store.updateWith(e.executionId) { existing =>
        existing.copy(
          plan = exact.getOrElse(PlanAcceleration.fromInfo(e.sparkPlanInfo)),
          exact = exact.isDefined,
          physicalPlanDescription = e.physicalPlanDescription)
      }
    }

  private def onEnd(e: SparkListenerSQLExecutionEnd): Unit = guard(e.executionId) {
    // `qe` is set by SQLExecution before the event is posted and gives us the final plan objects.
    val finalPlan = Option(e.qe).flatMap(qe => classify(qe.executedPlan))
    store.updateWith(e.executionId) { existing =>
      existing.copy(
        completionTime = Some(e.time),
        failed = e.errorMessage.exists(_.nonEmpty),
        plan = finalPlan.getOrElse(existing.plan),
        exact = finalPlan.isDefined || existing.exact)
    }
  }

  /** The plan of a still-registered execution, when the driver can still reach it. */
  private def livePlan(executionId: Long): Option[AcceleratedPlan] =
    Option(SQLExecution.getQueryExecution(executionId)).flatMap(qe => classify(qe.executedPlan))

  /** `executedPlan` throws if the query failed to plan, so classification is best effort. */
  private def classify(plan: => org.apache.spark.sql.execution.SparkPlan): Option[AcceleratedPlan] =
    try Some(PlanAcceleration.fromPlan(plan))
    catch { case NonFatal(_) => None }

  private def guard(executionId: Long)(body: => Unit): Unit =
    try body
    catch {
      case NonFatal(e) =>
        logWarning(s"spark-vector UI could not record execution $executionId", e)
    }
}
