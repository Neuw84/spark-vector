/*
 * Renders the coloured plan DAG on the spark-vector acceleration tab.
 *
 * The Scala page writes a DOT file into #sv-plan-dot, where every node carries the CSS class of the
 * engine that executes it. dagre-d3 copies that class onto the node's SVG group, so all the
 * colouring lives in vector-acceleration.css; this file only lays the graph out and sizes the SVG.
 *
 * Uses the d3, dagre-d3 and graphlib-dot bundles that Spark's UI already serves from /static.
 */

/* global d3, dagreD3, graphlibDot, collapseTablePageLoad */

(function () {
  "use strict";

  var MARGIN = 16;

  function container() {
    return d3.select("#sv-plan-graph");
  }

  function render() {
    var graphContainer = container();
    if (graphContainer.empty() || !graphContainer.selectAll("svg").empty()) {
      return;
    }

    var dotNode = document.getElementById("sv-plan-dot");
    if (!dotNode) {
      return;
    }
    var dot = dotNode.textContent.trim();
    if (!dot) {
      return;
    }

    var g;
    try {
      g = graphlibDot.read(dot);
    } catch (e) {
      console.error("spark-vector: could not parse the plan DOT file", e);
      graphContainer.append("p").attr("class", "sv-empty")
        .text("Could not render the plan graph.");
      return;
    }

    // Children point at their parent in the DOT file, the same direction Spark's SQL plan graph
    // uses, so the root ends up at the bottom. Flip it so the scan is at the bottom and the final
    // operator on top, matching how `explain` prints a plan.
    g.graph().rankdir = "BT";
    g.graph().ranksep = 30;
    g.graph().nodesep = 20;

    g.nodes().forEach(function (v) {
      var node = g.node(v);
      node.padding = 6;
      node.rx = 4;
      node.ry = 4;
    });

    g.edges().forEach(function (e) {
      g.setEdge(e.v, e.w, { curve: d3.curveBasis });
    });

    var svg = graphContainer.append("svg");
    var inner = svg.append("g");
    new dagreD3.render()(inner, g);

    addTooltips(g);
    resize(svg);
  }

  /*
   * DOT `tooltip` attributes become node.tooltip. Render them as a native <title>, which needs no
   * bootstrap and survives the SVG being scrolled.
   */
  function addTooltips(g) {
    g.nodes().forEach(function (v) {
      var node = g.node(v);
      if (!node.tooltip) {
        return;
      }
      var group = d3.select("#" + node.id);
      if (!group.empty()) {
        group.append("title").text(node.tooltip);
      }
    });
  }

  /* Size the SVG to its contents so the surrounding div can scroll rather than clip. */
  function resize(svg) {
    var group = svg.select("g").node();
    if (!group) {
      return;
    }
    var box = group.getBBox();
    var width = box.width + 2 * MARGIN;
    var height = box.height + 2 * MARGIN;
    svg
      .attr("viewBox", (box.x - MARGIN) + " " + (box.y - MARGIN) + " " + width + " " + height)
      .attr("width", width)
      .attr("height", height);
  }

  /* The "Plan Details" section remembers whether it was expanded, like Spark's own collapsible tables. */
  function restorePlanDetails() {
    if (typeof collapseTablePageLoad === "function") {
      collapseTablePageLoad("collapse-sv-plan-details", "sv-plan-details");
    }
  }

  function init() {
    render();
    restorePlanDetails();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
