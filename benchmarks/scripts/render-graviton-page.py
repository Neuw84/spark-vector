#!/usr/bin/env python3
# Copyright 2025-2026 Angel Conde and the spark-vector contributors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the
# License.
"""Render the Graviton TPC-DS page (docs/benchmarks/tpcds-1tb-graviton.html, #253): Apache Spark against
spark-vector on AWS Graviton4, from the cluster runner's two result files, with the published x86 run
(docs/benchmarks/tpcds-1tb.html, whose per-query data is inlined in the page) as the reference.

    render-graviton-page.py --spark spark.jsonl --vector vector-shuffle.jsonl \\
        --x86-page docs/benchmarks/tpcds-1tb.html --out docs/benchmarks/tpcds-1tb-graviton.html [--meta meta.json]

`meta.json` overrides the environment / configuration text below (see DEFAULT_META). The helpers and the
look are render-benchmark-page.py's; the page is self-contained (data inlined, Chart.js from a CDN).
"""
import argparse, html, importlib.util, json, math, re, sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the sibling script must not leave a __pycache__ in the repo
_spec = importlib.util.spec_from_file_location("render_benchmark_page", Path(__file__).with_name("render-benchmark-page.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
load, qkey, fmt_s, distribution, table = _base.load, _base.qkey, _base.fmt_s, _base.distribution, _base.table

DEFAULT_META = {
    "title": "Apache Spark vs spark-vector on TPC-DS 1 TB, AWS Graviton4",
    "dataset": "TPC-DS scale factor 1000 (1 TB), Parquet on Amazon S3 (103 query variants, one measured iteration each, no warm-up)",
    "cluster": "Amazon EKS 1.36; 9 x m8g.4xlarge (16 vCPU AWS Graviton4 / Neoverse V2 with SVE2, 64 GB, arm64), 300 GB root volume; one node group, the driver on the ninth node",
    "executors": "8 executors x 13 cores x 50 GB each (Spark: 20 GB heap / 30 GB overhead; spark-vector: 30 GB heap / 20 GB overhead); driver 2 cores x 4 GB",
    "storage": "Amazon S3 through Hadoop 3.4.3 S3A with the Analytics Accelerator input stream (the default in 3.4.3)",
    "versions": {
        "Spark": "4.1.3",
        "Scala": "2.13",
        "JDK": "Amazon Corretto 25.0.4.1 (aarch64)",
        "spark-vector": "main at #480 + #481 (arm64 image) + #484 (SVE mask construction)",
        "Hadoop": "3.4.3",
    },
    "common_conf": [
        "spark.sql.shuffle.partitions=300",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes=128m",
        "spark.sql.adaptive.coalescePartitions.minPartitionNum=208   # as the x86 run, for comparability",
        "spark.eventLog.enabled=true",
    ],
    "vector_conf": [
        "spark.plugins=io.sparkvector.spark.VectorPlugin",
        "spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager",
        "spark.vector.exec.strictFloatingPoint=false   # Comet's default too",
        "AOT class-data cache off",
        "--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED (driver and executors)",
    ],
    "platform": [
        "HotSpot on Graviton4: UseSVE=2, MaxVectorSize=16 -- the Vector API's species are 128 bits wide, as on NEON.",
        "spark-vector's platform probe picks its SVE paths: native compress (SVE COMPACT) for selection, and the broadcast-AND-compare lane masks, because VectorMask.fromLong is not intrinsified at 128-bit SVE on JDK 25 (#253, #484).",
        "The native codecs (snappy, zstd, lz4) and Netty's epoll transport load their aarch64 libraries; neither architecture has libhadoop, as on x86.",
    ],
    "notes": [
        "Both engines returned the same row counts on every query and the same checksums on every query except q65, whose result has ties that each engine orders differently (as on x86).",
        "The two legs ran one after another on the same nodes, each alone on the cluster, with the settings of the published x86 run -- only the instance type and the image's architecture differ; the x86 numbers are that run's (Comet is left out here).",
        "coalescePartitions.minPartitionNum=208 is kept so the two architectures compare; it is deprecated in Spark 3.2+, and runs after this one leave it unset.",
    ],
    "results_doc": "https://github.com/spark-vector/spark-vector/blob/main/docs/results.md",
    "repo": "https://github.com/spark-vector/spark-vector",
    "run_doc": "https://github.com/spark-vector/spark-vector/blob/main/benchmarks/k8s/README.md",
    "x86_page": "tpcds-1tb.html",
    "issue": "https://github.com/spark-vector/spark-vector/issues/253",
    # How the x86 reference run relates to this one: the lede says "the published x86 run <x86_relation>",
    # the architecture section "..., <x86_settings>." Override both when the two runs' settings differ.
    "x86_relation": "with the same settings",
    "x86_settings": "the same data, executors and Spark settings",
}

COLORS = {"spark": "#6b7280", "vector": "#2563eb"}
LABELS = {"spark": "Apache Spark 4.1.3", "vector": "spark-vector (plugin + Flight shuffle)"}


def x86_from_page(path):
    """Seconds per query of the published x86 run, per engine, from the page's inlined chart data."""
    m = re.search(r"const D = (\{.*?\});\n", Path(path).read_text(), re.S)
    if not m:
        sys.exit(f"no chart data in {path}")
    d = json.loads(m.group(1))
    return {e: dict(zip(d["queries"], d["series"][e])) for e in ("spark", "vector")}


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True); ap.add_argument("--vector", required=True)
    ap.add_argument("--x86-page", required=True); ap.add_argument("--out", required=True); ap.add_argument("--meta")
    ap.add_argument("--web-out", help="also write a standalone web/ page (no Jekyll) to this path")
    ap.add_argument("--web-base", default="/spark-vector", help="base URL prefix for the web/ page (default /spark-vector)")
    a = ap.parse_args()
    meta = dict(DEFAULT_META)
    if a.meta:
        meta.update(json.loads(Path(a.meta).read_text()))
    S, V = load(a.spark), load(a.vector)
    X = x86_from_page(a.x86_page)
    queries = sorted(set(S) & set(V), key=qkey)
    if not queries:
        sys.exit("no common queries in the two result files")
    missing = [q for q in queries if q not in X["spark"] or q not in X["vector"]]
    if missing:
        sys.exit(f"queries missing from the x86 page: {missing}")
    n = len(queries)
    data = {"spark": S, "vector": V}
    tot = {e: sum(data[e][q]["medianMs"] for q in queries) / 1000 for e in data}
    x86tot = {e: sum(X[e][q] for q in queries) for e in X}
    gc = {e: sum(data[e][q].get("gcTimeMs", 0) for q in queries) / 3.6e6 for e in data}
    shuf = {e: sum(data[e][q].get("shuffleReadBytes", 0) for q in queries) / 1e12 for e in data}
    exe = {e: sum(data[e][q].get("executorRunTimeMs", 0) for q in queries) / 3.6e6 for e in data}
    faster = sum(1 for q in queries if V[q]["medianMs"] < S[q]["medianMs"])
    rows_equal = sum(1 for q in queries if S[q].get("rows") == V[q].get("rows"))
    checksum_diff = [q for q in queries if S[q].get("checksum") != V[q].get("checksum")]

    def speed(q):
        return S[q]["medianMs"] / V[q]["medianMs"]

    def x86_speed(q):
        return X["spark"][q] / X["vector"][q]

    geo = geomean([speed(q) for q in queries]); x86_geo = geomean([x86_speed(q) for q in queries])
    arch = {e: geomean([data[e][q]["medianMs"] / 1000 / X[e][q] for q in queries]) for e in data}
    best, worst = max(queries, key=speed), min(queries, key=speed)

    def top(k=10):
        qs = sorted(queries, key=speed, reverse=True)[:k]
        return [(q, fmt_s(S[q]["medianMs"]), fmt_s(V[q]["medianMs"]), f"<b>{speed(q):.2f}x</b> ({100 - 100 / speed(q):+.0f}%)", f"{x86_speed(q):.2f}x") for q in qs]

    def regressions(k=10):
        qs = [q for q in sorted(queries, key=speed) if speed(q) < 1.0][:k]
        return [(q, fmt_s(S[q]["medianMs"]), fmt_s(V[q]["medianMs"]), f"<b>{100 / speed(q) - 100:.0f}%</b> slower ({speed(q):.2f}x)", f"{x86_speed(q):.2f}x") for q in qs]

    shift = sorted(queries, key=lambda q: speed(q) / x86_speed(q))

    def shift_rows(qs):
        return [(q, f"{speed(q):.2f}x", f"{x86_speed(q):.2f}x", f"{S[q]['medianMs'] / 1000 / X['spark'][q]:.2f}", f"{V[q]['medianMs'] / 1000 / X['vector'][q]:.2f}") for q in qs]

    dist = distribution(S, V, queries)
    chart_data = {
        "queries": queries,
        "series": {e: [round(data[e][q]["medianMs"] / 1000, 2) for q in queries] for e in data},
        "x86": {e: [round(X[e][q], 2) for q in queries] for e in X},
        "labels": LABELS, "colors": COLORS,
        "totals": {e: round(tot[e], 1) for e in data}, "x86totals": {e: round(x86tot[e], 1) for e in X},
    }
    per_query_rows = []
    for q in queries:
        sv, vv = S[q]["medianMs"], V[q]["medianMs"]
        cell = lambda x: f"<b>{fmt_s(x)}</b>" if x == min(sv, vv) else fmt_s(x)
        per_query_rows.append((q, cell(sv), cell(vv), f"{sv / vv:.2f}x", f"{X['spark'][q]:.1f}", f"{X['vector'][q]:.1f}", f"{x86_speed(q):.2f}x",
                               "ties" if q in checksum_diff else ""))

    versions = table(["Component", "Version"], [(k, v) for k, v in meta["versions"].items()], "kv")
    env = table(["Component", "Configuration"], [
        ("Dataset", meta["dataset"]), ("Cluster", meta["cluster"]), ("Executors", meta["executors"]), ("Storage", meta["storage"])], "kv")
    conf_common = "\n".join(meta["common_conf"]); conf_v = "\n".join(meta["vector_conf"])
    platform = "".join(f"<li>{html.escape(x)}</li>" for x in meta["platform"])
    notes = "".join(f"<li>{html.escape(x)}</li>" for x in meta["notes"])
    csum = ("every query" if not checksum_diff else
            f"every query except {', '.join(checksum_diff)} (ties in the result, ordered differently by each engine)")

    page = f"""---
layout: default
title: {json.dumps(meta["title"])}
---
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
/* Benchmark-page components. Scoped under .bench so they never restyle the site
   shell; colours come from the site theme variables (light/dark toggle works). */
.doc:has(.bench) {{ max-width: none; }}
.bench .lede {{ color: var(--muted); }}
.bench .tldr {{ background: var(--card); border-left: 4px solid var(--accent); padding: 14px 18px; border-radius: 6px; margin: 1.2em 0; }}
.bench .cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 1em 0; }}
.bench .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
.bench .card .n {{ font-size: 1.7rem; font-weight: 700; }} .bench .card .l {{ color: var(--muted); font-size: .9rem; }}
.bench table td:nth-child(n+2):not(:last-child) {{ font-variant-numeric: tabular-nums; }}
.bench table.kv td:first-child {{ font-weight: 600; white-space: nowrap; }}
.bench .chart {{ position: relative; height: 340px; margin: 1em 0 2em; }} .bench .chart.tall {{ height: 420px; }}
.bench .two {{ display:grid; grid-template-columns: 1fr 1fr; gap: 24px; }} @media (max-width: 860px) {{ .bench .two {{ grid-template-columns: 1fr; }} }}
.bench details summary {{ cursor: pointer; font-weight: 600; margin: 1em 0; }}
.bench .foot {{ color: var(--muted); font-size: .85rem; margin-top: 3em; }}
</style>
<div class="bench">
<h1>{html.escape(meta["title"])}</h1>
<p class="lede"><a href="{meta["repo"]}">spark-vector</a> runs Spark SQL's filters, projections, aggregates, sorts and joins on Arrow-layout
batches with the Java Vector API -- on the JVM, no native code -- and moves batches between executors over its own Arrow Flight shuffle.
This page compares it against plain Apache Spark on the TPC-DS 1 TB workload on Amazon EKS with AWS Graviton4 (arm64) nodes, and sets both against
the <a href="{meta["x86_page"]}">published x86 run</a> {html.escape(meta["x86_relation"])} (<a href="{meta["issue"]}">#253</a>).</p>

<div class="tldr"><b>TL;DR.</b> On Graviton4, over the {n} TPC-DS queries at 1 TB, <b>spark-vector</b> finished in <b>{tot["vector"]:,.0f} s</b> against Spark's
{tot["spark"]:,.0f} s -- <b>{tot["spark"] / tot["vector"]:.2f}x</b>, {100 - 100 * tot["vector"] / tot["spark"]:.0f}% less runtime (geometric mean {geo:.2f}x),
faster than Spark on {faster} of {n} queries (best {best}: {speed(best):.2f}x; largest regression {worst}: {100 / speed(worst) - 100:.0f}%).
On the x86 nodes the same comparison is {x86tot["spark"] / x86tot["vector"]:.2f}x (geometric mean {x86_geo:.2f}x): the lead carries over to arm64.
Both engines run faster on Graviton4 than on the m5.4xlarge nodes -- Spark's time is {arch["spark"]:.2f} of its x86 time and spark-vector's
{arch["vector"]:.2f}, per query (geometric mean). Row counts equal Spark's on {rows_equal} of {n} queries; checksums on {csum}.</div>

<div class="cards">
<div class="card"><div class="l">Apache Spark 4.1.3 on Graviton4</div><div class="n">{tot["spark"]:,.0f} s</div><div class="l">baseline, {n} queries (x86: {x86tot["spark"]:,.0f} s)</div></div>
<div class="card"><div class="l">spark-vector on Graviton4</div><div class="n">{tot["vector"]:,.0f} s</div><div class="l">{tot["spark"] / tot["vector"]:.2f}x · {100 - 100 * tot["vector"] / tot["spark"]:.0f}% less runtime (x86: {x86tot["vector"]:,.0f} s)</div></div>
<div class="card"><div class="l">Graviton4 against x86</div><div class="n">{arch["spark"]:.2f} / {arch["vector"]:.2f}</div><div class="l">time per query, Spark / spark-vector (geometric mean)</div></div>
</div>

<h2 id="summary">Summary</h2>
{table(["Engine", "Completion time (s)", "Speedup", "Faster than Spark on", "Executor time (h)", "GC (h)", "Shuffle read (TB)"], [
    ("Apache Spark 4.1.3", f"{tot['spark']:,.1f}", "baseline", "--", f"{exe['spark']:.1f}", f"{gc['spark']:.2f}", f"{shuf['spark']:.2f}"),
    ("spark-vector", f"{tot['vector']:,.1f}", f"<b>{tot['spark'] / tot['vector']:.2f}x</b> ({100 - 100 * tot['vector'] / tot['spark']:.0f}% less)", f"{faster} / {n}", f"{exe['vector']:.1f}", f"{gc['vector']:.2f}", f"{shuf['vector']:.2f}"),
])}
<div class="chart"><canvas id="totals"></canvas></div>

<h2 id="arch">Graviton4 against x86</h2>
<p>The x86 numbers are the <a href="{meta["x86_page"]}">published run</a>: 9 x m5.4xlarge (16 vCPU, 64 GB, AVX-512), {html.escape(meta["x86_settings"])}.</p>
{table(["", "Spark (s)", "spark-vector (s)", "spark-vector speedup", "geometric mean"], [
    ("Graviton4 (m8g.4xlarge)", f"{tot['spark']:,.1f}", f"{tot['vector']:,.1f}", f"<b>{tot['spark'] / tot['vector']:.2f}x</b>", f"{geo:.2f}x"),
    ("x86 (m5.4xlarge)", f"{x86tot['spark']:,.1f}", f"{x86tot['vector']:,.1f}", f"{x86tot['spark'] / x86tot['vector']:.2f}x", f"{x86_geo:.2f}x"),
    ("Graviton4 / x86, per query (geomean)", f"{arch['spark']:.2f}", f"{arch['vector']:.2f}", "", ""),
])}
<div class="two">
<div><p>Where spark-vector's lead over Spark <b>shrinks</b> most on Graviton4 -- mostly queries where Spark itself gains more from Graviton4:</p>
{table(["Query", "Graviton4", "x86", "Spark G/x", "spark-vector G/x"], shift_rows(shift[:8]))}</div>
<div><p>Where it <b>grows</b> most -- mostly queries at parity or behind on x86:</p>
{table(["Query", "Graviton4", "x86", "Spark G/x", "spark-vector G/x"], shift_rows(list(reversed(shift[-8:]))))}</div>
</div>
<p>"G/x" is the query's time on Graviton4 over its time on x86 (below 1 = faster on Graviton4).</p>
<div class="chart tall"><canvas id="archspeed"></canvas></div>

<h2 id="infrastructure">Benchmark infrastructure</h2>
<p><b>Methodology.</b> The two engines ran one after another on the same nodes, each alone on the cluster, over the same S3 data with the same
Spark settings; only the execution engine and its own memory split differ (every engine has 50 GB per executor). Each query ran once after the plan
was compiled; the time is the wall-clock of the query's execution as the runner measures it.</p>
<h3>Test environment</h3>
{env}
<h3>Versions</h3>
{versions}
<h3>Configuration</h3>
<p>Common to both engines:</p><pre>{html.escape(conf_common)}</pre>
<p>spark-vector:</p><pre>{html.escape(conf_v)}</pre>
<h3>The arm64 platform</h3>
<ul>{platform}</ul>

<h2 id="results">Performance results</h2>
<h3>Per query</h3>
<p>Seconds per query on Graviton4, both engines side by side (hover for values; click a legend entry to hide an engine).</p>
<div class="chart tall"><canvas id="perquery1"></canvas></div>
<div class="chart tall"><canvas id="perquery2"></canvas></div>
<div class="chart tall"><canvas id="perquery3"></canvas></div>
<h3>Performance distribution</h3>
{table(["spark-vector vs Spark", "Queries", "Share"], [(k, v, f"{100 * v / n:.0f}%") for k, v in dist.items()])}
<h3>Top 10 improvements</h3>
{table(["Query", "Spark (s)", "spark-vector (s)", "Speedup", "x86 speedup"], top())}
<h3>Regressions</h3>
{table(["Query", "Spark (s)", "spark-vector (s)", "Degradation", "x86 speedup"], regressions()) if regressions() else "<p>None.</p>"}
<h3>Notes</h3>
<ul>{notes}</ul>

<h2 id="table">All queries</h2>
<details open><summary>Seconds per query on Graviton4, the faster engine in bold, with the x86 run alongside</summary>
{table(["Query", "Spark", "spark-vector", "Speedup", "x86 Spark", "x86 spark-vector", "x86 speedup", "Note"], per_query_rows, "all")}
</details>

<h2 id="running">Running the benchmark</h2>
<p>The cluster runner, the Spark-on-Kubernetes manifests, the image and the data generation are in
<a href="{meta["run_doc"]}">the benchmark runner's README</a>; on Graviton the image builds for arm64 on an arm64 node and the runs select the arm64 node group.
<code>run-matrix.sh</code> writes one JSON-lines result file per run; this page is rendered from two of them, with the x86 page as the reference,
by <code>benchmarks/scripts/render-graviton-page.py</code>.</p>

<p class="foot">TPC-DS is a benchmark of the Transaction Processing Performance Council; these results are not audited TPC results and are not comparable to
published TPC-DS results. Times are one measured iteration per query on the cluster described above; run-to-run variation on the heavy queries is a few percent.</p>
</div>
<script>
const D = {json.dumps(chart_data)};
const opts = (title, yTitle, extra={{}}) => Object.assign({{
  responsive: true, maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ title: {{ display: !!title, text: title }}, legend: {{ position: 'top' }} }},
  scales: {{ y: {{ title: {{ display: true, text: yTitle }} }} }}
}}, extra);
const engines = ['spark', 'vector'];
new Chart(document.getElementById('totals'), {{ type: 'bar', data: {{ labels: ['Graviton4 (m8g.4xlarge)', 'x86 (m5.4xlarge)'],
  datasets: engines.map(e => ({{ label: D.labels[e], data: [D.totals[e], D.x86totals[e]], backgroundColor: D.colors[e] }})) }},
  options: opts('Completion time, {n} queries', 'seconds') }});
const chunks = [D.queries.slice(0, 35), D.queries.slice(35, 70), D.queries.slice(70)];
chunks.forEach((keys, i) => new Chart(document.getElementById('perquery' + (i + 1)), {{ type: 'bar',
  data: {{ labels: keys, datasets: engines.map(e => ({{ label: D.labels[e], backgroundColor: D.colors[e], borderColor: D.colors[e],
    data: keys.map(q => D.series[e][D.queries.indexOf(q)]) }})) }},
  options: opts(i === 0 ? 'Seconds per query, Graviton4' : '', 'seconds') }}));
new Chart(document.getElementById('archspeed'), {{ type: 'line',
  data: {{ labels: D.queries, datasets: [
    {{ label: 'Graviton4', borderColor: D.colors.vector, backgroundColor: D.colors.vector, pointRadius: 3, showLine: false,
      data: D.queries.map((q, i) => +(D.series.spark[i] / D.series.vector[i]).toFixed(3)) }},
    {{ label: 'x86', borderColor: D.colors.spark, backgroundColor: D.colors.spark, pointRadius: 3, showLine: false,
      data: D.queries.map((q, i) => +(D.x86.spark[i] / D.x86.vector[i]).toFixed(3)) }}] }},
  options: opts('spark-vector speedup over Spark per query, Graviton4 and x86 (log scale; 1 = Spark)', 'x faster than Spark',
    {{ scales: {{ y: {{ type: 'logarithmic', min: 0.4, max: 4, title: {{ display: true, text: 'x faster than Spark' }} }} }} }}) }});
</script>
"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page)
    print(f"wrote {a.out}: {n} queries; Graviton4 spark={tot['spark']:.1f}s vector={tot['vector']:.1f}s ({tot['spark'] / tot['vector']:.2f}x); "
          f"x86 {x86tot['spark'] / x86tot['vector']:.2f}x; geomean {geo:.3f} vs {x86_geo:.3f}; checksum differences {checksum_diff}")
    if a.web_out:
        from webwrap import to_web_page
        Path(a.web_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.web_out).write_text(to_web_page(page, a.web_base, description=meta.get("dataset", "")))
        print(f"wrote {a.web_out}: web/ page under {a.web_base}")


if __name__ == "__main__":
    main()
