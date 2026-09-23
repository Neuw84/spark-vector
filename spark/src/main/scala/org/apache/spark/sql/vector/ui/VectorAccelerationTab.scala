package org.apache.spark.sql.vector.ui

import scala.xml.Node

import jakarta.servlet.http.HttpServletRequest
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.ui.{SparkUI, SparkUITab}

/**
 * The "Vector Acceleration" tab: a list of SQL executions and, per execution, the physical plan
 * as a DAG with each operator coloured by the engine that runs it.
 *
 * Lives under `org.apache.spark` because `SparkUITab`, `WebUIPage` and `UIUtils` are all
 * `private[spark]`; Spark exposes no public API for adding a tab.
 */
class VectorAccelerationTab(val store: VectorAccelerationStore, sparkUI: SparkUI)
    extends SparkUITab(sparkUI, "vector") with Logging {

  override val name: String = "Vector Acceleration"

  def conf: SparkConf = sparkUI.conf
  val parent: SparkUI = sparkUI

  attachPage(new AllExecutionsPage(this))
  attachPage(new ExecutionPlanPage(this))
  parent.attachTab(this)
  parent.addStaticHandler(VectorAccelerationTab.StaticResourceDir, "/static/vector")

  /** After the SQL tab (0), before the rest. */
  override def displayOrder: Int = 1
}

object VectorAccelerationTab {
  val StaticResourceDir = "org/apache/spark/sql/vector/ui/static"

  /** Shared legend, so the list and the plan page agree on what each colour means. */
  def legend: Seq[Node] =
    <div class="sv-legend">
      {
      Engine.all.map { e =>
        <span class="sv-legend-item">
          <span class={s"sv-legend-swatch ${e.cssClass}"}></span>
          <span class="sv-legend-label">{e.label}</span>
        </span>
      }
    }
    </div>

  /**
   * The badge shown when nothing in the plan fell back to Spark. Otherwise the accelerated share,
   * with a tooltip separating operators we tried and failed to convert (a recorded reason) from
   * operators that were never candidates, such as a scan, an exchange or a sort.
   */
  def fullyAcceleratedBadge(plan: AcceleratedPlan): Seq[Node] =
    if (plan.fullyAccelerated) {
      <span class="sv-badge sv-badge-full" title="Every operator in this plan runs on the Vector API or on Comet">
        Fully Accelerated
      </span>
    } else {
      val onSpark = plan.countBy(Engine.Spark)
      val missed = plan.fallbacks.size
      val tip =
        s"$onSpark operator(s) run on Spark; $missed of them were candidates we could not convert " +
          s"(see the reasons on the plan page)"
      <span class="sv-badge sv-badge-partial" title={tip}>
        {f"${plan.acceleratedFraction * 100}%.0f%% accelerated"}
      </span>
    }

  def resourceUri(request: HttpServletRequest, file: String): String =
    org.apache.spark.ui.UIUtils.prependBaseUri(request, s"/static/vector/$file")
}
