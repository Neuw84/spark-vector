package org.apache.spark.sql.vector.ui

import scala.xml.Node

import jakarta.servlet.http.HttpServletRequest
import org.apache.spark.ui.{UIUtils, WebUIPage}

/**
 * One execution: the plan as a coloured DAG, the fallback reasons behind every uncoloured node,
 * and the plan text.
 *
 * The graph is emitted as a DOT file and rendered client side by `vector-acceleration.js` with the
 * d3 / dagre-d3 / graphlib-dot bundles Spark's UI already ships. Each node carries the engine's
 * CSS class, which is what produces the colours.
 */
class ExecutionPlanPage(parent: VectorAccelerationTab) extends WebUIPage("execution") {

  override def render(request: HttpServletRequest): Seq[Node] = {
    val idParam = Option(request.getParameter("id")).map(_.trim).getOrElse("")
    // A missing or malformed id is a bad link, not a server error: show the same page as for an
    // evicted execution rather than letting toLong throw a 500.
    val executionId = scala.util.Try(idParam.toLong).toOption

    val content = executionId.flatMap(parent.store.get) match {
      case None =>
        <div>
          <link rel="stylesheet" type="text/css"
                href={VectorAccelerationTab.resourceUri(request, "vector-acceleration.css")}/>
          <p class="sv-empty">
            No acceleration information for query {idParam}. It may have been evicted
            (only the last {parent.store.retained} executions are kept).
          </p>
        </div>
      case Some(e) => body(request, e)
    }

    val title = executionId.map(id => s"Vector Acceleration for Query $id").getOrElse("Vector Acceleration")
    UIUtils.headerSparkPage(request, title, content, parent)
  }

  private def body(request: HttpServletRequest, e: ExecutionSummary): Seq[Node] = {
    val plan = e.plan
    <div>
      <link rel="stylesheet" type="text/css"
            href={VectorAccelerationTab.resourceUri(request, "vector-acceleration.css")}/>
      {header(e)}
      {VectorAccelerationTab.legend}
      <div id="sv-plan-graph" class="sv-plan-graph"></div>
      <div id="sv-plan-dot" style="display:none">{plan.toDotFile}</div>
      {fallbackTable(plan)}
      {planText(e)}
      <script src={UIUtils.prependBaseUri(request, "/static/d3.min.js")}></script>
      <script src={UIUtils.prependBaseUri(request, "/static/dagre-d3.min.js")}></script>
      <script src={UIUtils.prependBaseUri(request, "/static/graphlib-dot.min.js")}></script>
      <script src={VectorAccelerationTab.resourceUri(request, "vector-acceleration.js")}></script>
    </div>
  }

  private def header(e: ExecutionSummary): Seq[Node] = {
    val plan = e.plan
    <div class="sv-header">
      <div class="sv-header-badge">{VectorAccelerationTab.fullyAcceleratedBadge(plan)}</div>
      <ul class="sv-summary list-unstyled">
        <li><strong>Description:</strong> {e.description}</li>
        <li><strong>Status:</strong> {e.status}
          {e.duration.map(d => <span> in {UIUtils.formatDuration(d)}</span>).getOrElse(scala.xml.NodeSeq.Empty)}
        </li>
        <li>
          <strong>Operators:</strong>{" "}
          {Engine.all.filter(en => plan.countBy(en) > 0).map { en =>
            <span class={s"sv-count ${en.cssClass}"}>{en.label} {plan.countBy(en)}</span>
          }}
        </li>
        {if (!e.exact) {
          <li class="sv-muted">
            Classified from the plan attached to the listener event: node names were matched rather
            than operator types, so fallback reasons are not available for this query.
          </li>
        } else scala.xml.NodeSeq.Empty}
      </ul>
    </div>
  }

  /** Why each Spark-coloured operator stayed with Spark; the reasons the planner rule recorded. */
  private def fallbackTable(plan: AcceleratedPlan): Seq[Node] = {
    val fallbacks = plan.fallbacks
    if (fallbacks.isEmpty) {
      scala.xml.NodeSeq.Empty
    } else {
      <div class="sv-section">
        <h4>Why operators were not accelerated ({fallbacks.size.toString})</h4>
        <table class="table table-bordered table-sm sv-table">
          <thead><tr><th>Operator</th><th>Reason</th></tr></thead>
          <tbody>
            {fallbacks.map { case (name, reason) =>
              <tr><td>{name}</td><td>{reason}</td></tr>
            }}
          </tbody>
        </table>
      </div>
    }
  }

  /**
   * The plan text, collapsed by default. Toggled with `collapseTable` from Spark's `webui.js`, the
   * same mechanism the Stage page uses for its metrics tables: Spark 4 bundles Bootstrap 4, whose
   * collapse plugin does not react to the Bootstrap 5 `data-bs-toggle` attributes.
   */
  private def planText(e: ExecutionSummary): Seq[Node] =
    <div class="sv-section">
      <span class="collapse-sv-plan-details collapse-table"
            onClick="collapseTable('collapse-sv-plan-details','sv-plan-details')">
        <h4><span class="collapse-table-arrow arrow-closed"></span><a>Plan Details</a></h4>
      </span>
      <div class="sv-plan-details collapsible-table collapsed" id="sv-plan-details">
        <pre>{e.physicalPlanDescription}</pre>
      </div>
    </div>
}
