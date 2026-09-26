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
package org.apache.spark.sql.vecruntime.ui

import scala.xml.Node

import jakarta.servlet.http.HttpServletRequest
import org.apache.spark.ui.{UIUtils, WebUIPage}

/** Landing page of the tab: every recorded SQL execution and how much of it was accelerated. */
class AllExecutionsPage(parent: VectorAccelerationTab) extends WebUIPage("") {

  override def render(request: HttpServletRequest): Seq[Node] = {
    val executions = parent.store.list()

    val content =
      <div>
        <link rel="stylesheet" type="text/css"
              href={VectorAccelerationTab.resourceUri(request, "vector-acceleration.css")}/>
        {summary(executions)}
        {VectorAccelerationTab.legend}
        {if (executions.isEmpty) emptyState else table(request, executions)}
      </div>

    UIUtils.headerSparkPage(request, "Vector Acceleration", content, parent)
  }

  private def emptyState: Seq[Node] =
    <p class="sv-empty">
      No SQL execution recorded yet. Run a query and reload this page.
    </p>

  private def summary(executions: Seq[ExecutionSummary]): Seq[Node] = {
    val full = executions.count(_.plan.fullyAccelerated)
    <ul class="sv-summary list-unstyled">
      <li><strong>Queries recorded:</strong> {executions.size} (last {parent.store.retained} kept)</li>
      <li><strong>Fully accelerated:</strong> {full}</li>
      <li>
        <strong>Vector API operators:</strong>{" "}
        {executions.map(_.plan.countBy(Engine.Vector)).sum}
        <span class="sv-muted">
          , Comet {executions.map(_.plan.countBy(Engine.Comet)).sum}
          , on Spark {executions.map(_.plan.countBy(Engine.Spark)).sum}
          {" "}(of which {executions.map(_.plan.fallbacks.size).sum} were candidates we could not convert)
        </span>
      </li>
    </ul>
  }

  private def table(request: HttpServletRequest, executions: Seq[ExecutionSummary]): Seq[Node] = {
    val base = UIUtils.prependBaseUri(request, parent.basePath)
    <table class="table table-bordered table-sm table-striped sortable sv-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Description</th>
          <th>Submitted</th>
          <th>Duration</th>
          <th>Status</th>
          <th>Acceleration</th>
          <th>Operators</th>
        </tr>
      </thead>
      <tbody>
        {executions.map(row(base, _))}
      </tbody>
    </table>
  }

  private def row(base: String, e: ExecutionSummary): Seq[Node] = {
    val plan = e.plan
    <tr>
      <td><a href={s"$base/vector/execution/?id=${e.executionId}"}>{e.executionId}</a></td>
      <td>
        <a href={s"$base/vector/execution/?id=${e.executionId}"}>{e.description}</a>
        {
      if (!e.exact) <span class="sv-muted" title={
        "Classified from the plan on the listener event; the final plan was not reachable"
      }> (approximate)</span>
      else scala.xml.NodeSeq.Empty
    }
      </td>
      <td sorttable_customkey={e.submissionTime.toString}>{UIUtils.formatDate(e.submissionTime)}</td>
      <td sorttable_customkey={e.duration.getOrElse(-1L).toString}>
        {e.duration.map(UIUtils.formatDuration).getOrElse("")}
      </td>
      <td>{e.status}</td>
      <td sorttable_customkey={f"${plan.acceleratedFraction}%.4f"}>
        {VectorAccelerationTab.fullyAcceleratedBadge(plan)}
      </td>
      <td>{operatorCounts(plan)}</td>
    </tr>
  }

  /** A compact per-engine breakdown, so the list is readable without opening each plan. */
  private def operatorCounts(plan: AcceleratedPlan): Seq[Node] =
    <span class="sv-counts">
      {
      Engine.all.filter(e => plan.countBy(e) > 0).map { e =>
        <span class={s"sv-count ${e.cssClass}"} title={e.label}>
          {e.shortLabel} {plan.countBy(e)}
        </span>
      }
    }
    </span>
}
