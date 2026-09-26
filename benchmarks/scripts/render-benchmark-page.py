#!/usr/bin/env python3
# Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
"""Render the TPC-DS benchmark page (docs/benchmarks/tpcds-1tb.html) from the cluster runner's result
files -- one JSON-lines file per engine, one row per query with `query` and `medianMs` (plus the
metrics the runner records: executorRunTimeMs, gcTimeMs, shuffleReadBytes, checksum, rows).

    render-benchmark-page.py --spark spark.jsonl --vector vector.jsonl --comet comet.jsonl \
        --out docs/benchmarks/tpcds-1tb.html [--meta meta.json]

`meta.json` overrides the environment / configuration text embedded below (see DEFAULT_META). The page
is self-contained: the data is inlined, Chart.js comes from a CDN, no build step.
"""
import argparse, html, json, re, sys
from pathlib import Path

DEFAULT_META = {
    "title": "Apache Spark vs vecruntime vs DataFusion Comet on TPC-DS 1 TB",
    "dataset": "TPC-DS scale factor 1000 (1 TB), Parquet on Amazon S3 (103 query variants, one measured iteration each, no warm-up)",
    "cluster": "Amazon EKS 1.36; 9 x m5.4xlarge (16 vCPU, 64 GB, x86-64 with AVX-512), 20 GB root volume; one node group, the driver on the ninth node",
    "executors": "8 executors x 13 cores x 50 GB each (Spark: 20 GB heap / 30 GB overhead; vecruntime: 30 GB heap / 20 GB overhead; Comet: 20 GB heap / 6 GB overhead / 24 GB off-heap); driver 2 cores x 6 GB",
    "storage": "Amazon S3 through Hadoop 3.4.3 S3A with the Analytics Accelerator input stream (the default in 3.4.3)",
    "versions": {"Spark": "4.1.3", "Scala": "2.13", "JDK": "Amazon Corretto 25", "vecruntime": "main at #465 (0.0.1)", "Comet": "1.0.0", "Hadoop": "3.4.3"},
    "common_conf": [
        "spark.sql.shuffle.partitions=300",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes=128m",
        "spark.sql.adaptive.coalescePartitions.minPartitionNum=208",
        "spark.eventLog.enabled=true",
    ],
    "vector_conf": [
        "spark.plugins=io.vecruntime.spark.VectorPlugin",
        "spark.shuffle.manager=org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager",
        "spark.vector.exec.strictFloatingPoint=false   # Comet's default too",
        "spark.vector.scan.prefetch=0                 # measured: off",
        "AOT class-data cache off                     # measured: costs the heavy queries 20%",
        "--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED (driver and executors)",
    ],
    "comet_conf": [
        "spark.plugins=org.apache.spark.CometPlugin",
        "spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager",
        "spark.memory.offHeap.enabled=true",
        "spark.memory.offHeap.size=24g",
        "spark.comet.exec.enabled=true",
        "spark.comet.exec.shuffle.enabled=true",
        "spark.comet.exec.shuffle.mode=auto",
    ],
    "notes": [
        "Every engine returned Spark's row counts and checksums on every query except q65 (its result has ties, ordered differently by every engine) and, for Comet, q64 (0 rows against 12,185 -- a stale dynamic-pruning value, apache/datafusion-comet#6133).",
        "The four legs ran in one window on the same nodes, one after another (Spark, vecruntime, Comet), each alone on the cluster; the AQE minimum of 208 partitions keeps the sort stages of the window queries wide (measured: Spark -4%, Comet -3%, vecruntime q67 -12%, the rest inside the run-to-run band).",
        "Spark's memory split is the one that completes: at 30 GB heap / 20 GB overhead Spark loses q23b, q24a and q24b to node-disk evictions during q23b's 503 GB spill; per query the heavier heap is 2-3% faster where it completes. Every split is measured in docs/results.md.",
        "coalescePartitions.minPartitionNum=208 is set for this run (and the Graviton4 run that compares with it); it is deprecated in Spark 3.2+, and later TPC-DS runs leave it unset (#253).",
    ],
    "results_doc": "https://github.com/vecruntime/vecruntime/blob/main/docs/results.md",
    "repo": "https://github.com/vecruntime/vecruntime",
    "run_doc": "https://github.com/vecruntime/vecruntime/blob/main/benchmarks/k8s/README.md",
}

ENGINES = [("spark", "Apache Spark 4.1.3", "#6b7280"), ("vector", "vecruntime (plugin + Flight shuffle)", "#2563eb"), ("comet", "DataFusion Comet 1.0.0", "#f59e0b")]


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        if "query" in j and "medianMs" in j:
            rows[j["query"]] = j
    return rows


def qkey(q):
    m = re.match(r"q(\d+)([a-z]?)", q)
    return (int(m.group(1)), m.group(2))


def fmt_s(ms):
    return f"{ms / 1000:.1f}"


def distribution(base, other, queries):
    b = {"20%+ improvement": 0, "10-20% improvement": 0, "within ±10%": 0, "10-20% degradation": 0, "20%+ degradation": 0}
    for q in queries:
        r = other[q]["medianMs"] / base[q]["medianMs"]
        if r <= 0.8: b["20%+ improvement"] += 1
        elif r <= 0.9: b["10-20% improvement"] += 1
        elif r < 1.1: b["within ±10%"] += 1
        elif r < 1.2: b["10-20% degradation"] += 1
        else: b["20%+ degradation"] += 1
    return b


def table(headers, rows, cls=""):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True); ap.add_argument("--vector", required=True); ap.add_argument("--comet", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--meta")
    ap.add_argument("--web-out", help="also write a standalone web/ page (no Jekyll) to this path")
    ap.add_argument("--web-base", default="/vecruntime", help="base URL prefix for the web/ page (default /vecruntime)")
    a = ap.parse_args()
    meta = dict(DEFAULT_META)
    if a.meta:
        meta.update(json.loads(Path(a.meta).read_text()))
    data = {"spark": load(a.spark), "vector": load(a.vector), "comet": load(a.comet)}
    queries = sorted(set(data["spark"]) & set(data["vector"]) & set(data["comet"]), key=qkey)
    if not queries:
        sys.exit("no common queries in the three result files")
    tot = {e: sum(data[e][q]["medianMs"] for q in queries) / 1000 for e, _, _ in ENGINES}
    gc = {e: sum(data[e][q].get("gcTimeMs", 0) for q in queries) / 3.6e6 for e, _, _ in ENGINES}
    shuf = {e: sum(data[e][q].get("shuffleReadBytes", 0) for q in queries) / 1e12 for e, _, _ in ENGINES}
    exe = {e: sum(data[e][q].get("executorRunTimeMs", 0) for q in queries) / 3.6e6 for e, _, _ in ENGINES}
    faster = {e: sum(1 for q in queries if data[e][q]["medianMs"] < data["spark"][q]["medianMs"]) for e in ("vector", "comet")}
    v_lt_c = sum(1 for q in queries if data["vector"][q]["medianMs"] < data["comet"][q]["medianMs"])
    S, V, C = data["spark"], data["vector"], data["comet"]

    def speed(e, q):
        return S[q]["medianMs"] / data[e][q]["medianMs"]

    def top(e, n=10):
        qs = sorted(queries, key=lambda q: speed(e, q), reverse=True)[:n]
        return [(q, fmt_s(S[q]["medianMs"]), fmt_s(data[e][q]["medianMs"]), f"<b>{speed(e, q):.2f}x</b> ({100 - 100 / speed(e, q):+.0f}%)") for q in qs]

    def regressions(e):
        qs = [q for q in sorted(queries, key=lambda q: speed(e, q)) if speed(e, q) < 1.0]
        return [(q, fmt_s(S[q]["medianMs"]), fmt_s(data[e][q]["medianMs"]), f"<b>{100 / speed(e, q) - 100:.0f}%</b> slower ({speed(e, q):.2f}x)") for q in qs[:10]]

    dist_v, dist_c = distribution(S, V, queries), distribution(S, C, queries)
    n = len(queries)
    best_v = max(queries, key=lambda q: speed("vector", q)); worst_v = min(queries, key=lambda q: speed("vector", q))
    best_c = max(queries, key=lambda q: speed("comet", q)); worst_c = min(queries, key=lambda q: speed("comet", q))

    chart_data = {
        "queries": queries,
        "series": {e: [round(data[e][q]["medianMs"] / 1000, 2) for q in queries] for e, _, _ in ENGINES},
        "labels": {e: label for e, label, _ in ENGINES},
        "colors": {e: color for e, _, color in ENGINES},
        "totals": {e: round(tot[e], 1) for e, _, _ in ENGINES},
    }

    per_query_rows = []
    for q in queries:
        sv, vv, cv = S[q]["medianMs"], V[q]["medianMs"], C[q]["medianMs"]
        best = min(sv, vv, cv)
        cell = lambda x: f"<b>{fmt_s(x)}</b>" if x == best else fmt_s(x)
        note = ""
        if q == "q65": note = "ties"
        if q == "q64": note = "Comet: 0 rows (comet#6133)"
        per_query_rows.append((q, cell(sv), cell(vv), cell(cv), f"{sv / vv:.2f}x", f"{sv / cv:.2f}x", note))

    versions = table(["Component", "Version"], [(k, v) for k, v in meta["versions"].items()], "kv")
    env = table(["Component", "Configuration"], [
        ("Dataset", meta["dataset"]), ("Cluster", meta["cluster"]), ("Executors", meta["executors"]), ("Storage", meta["storage"])], "kv")
    conf_common = "\n".join(meta["common_conf"]); conf_v = "\n".join(meta["vector_conf"]); conf_c = "\n".join(meta["comet_conf"])
    notes = "".join(f"<li>{html.escape(x)}</li>" for x in meta["notes"])

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
<p class="lede"><a href="{meta["repo"]}">vecruntime</a> runs Spark SQL's filters, projections, aggregates, sorts and joins on Arrow-layout
batches with the Java Vector API -- on the JVM, no native code -- and moves batches between executors over its own Arrow Flight shuffle.
<a href="https://github.com/apache/datafusion-comet">Apache DataFusion Comet</a> offloads the same operators to a native Rust engine. This page
compares both against plain Apache Spark on the TPC-DS 1 TB workload on Amazon EKS, all three on identical hardware, data and Spark settings.</p>

<div class="tldr"><b>TL;DR.</b> Over the 103 TPC-DS queries at 1 TB, <b>vecruntime</b> finished in <b>{tot["vector"]:,.0f} s</b> against Spark's
{tot["spark"]:,.0f} s -- <b>{tot["spark"] / tot["vector"]:.2f}x</b>, {100 - 100 * tot["vector"] / tot["spark"]:.0f}% less runtime, faster than Spark on {faster["vector"]} of {n} queries
(best {best_v}: {speed("vector", best_v):.2f}x; largest regression {worst_v}: {100 / speed("vector", worst_v) - 100:.0f}%).
<b>Comet</b> finished in {tot["comet"]:,.0f} s -- {tot["spark"] / tot["comet"]:.2f}x, {100 - 100 * tot["comet"] / tot["spark"]:.0f}% less, faster on {faster["comet"]} of {n}
(best {best_c}: {speed("comet", best_c):.2f}x; largest regression {worst_c}: {100 / speed("comet", worst_c) - 100:.0f}%).
Between the two, vecruntime is faster on {v_lt_c} of {n} queries and leads on the heavy joins; Comet leads on the scan- and aggregate-bound ones.
Every engine returned Spark's results (two footnoted exceptions).</div>

<div class="cards">
<div class="card"><div class="l">Apache Spark 4.1.3</div><div class="n">{tot["spark"]:,.0f} s</div><div class="l">baseline, 103 queries</div></div>
<div class="card"><div class="l">vecruntime</div><div class="n">{tot["vector"]:,.0f} s</div><div class="l">{tot["spark"] / tot["vector"]:.2f}x · {100 - 100 * tot["vector"] / tot["spark"]:.0f}% less runtime</div></div>
<div class="card"><div class="l">DataFusion Comet 1.0.0</div><div class="n">{tot["comet"]:,.0f} s</div><div class="l">{tot["spark"] / tot["comet"]:.2f}x · {100 - 100 * tot["comet"] / tot["spark"]:.0f}% less runtime</div></div>
</div>

<h2 id="summary">Summary</h2>
{table(["Engine", "Completion time (s)", "Speedup", "Faster than Spark on", "Executor time (h)", "GC (h)", "Shuffle read (TB)"], [
    ("Apache Spark 4.1.3", f"{tot['spark']:,.1f}", "baseline", "--", f"{exe['spark']:.1f}", f"{gc['spark']:.2f}", f"{shuf['spark']:.2f}"),
    ("vecruntime", f"{tot['vector']:,.1f}", f"<b>{tot['spark'] / tot['vector']:.2f}x</b> ({100 - 100 * tot['vector'] / tot['spark']:.0f}% less)", f"{faster['vector']} / {n}", f"{exe['vector']:.1f}", f"{gc['vector']:.2f}", f"{shuf['vector']:.2f}"),
    ("DataFusion Comet 1.0.0", f"{tot['comet']:,.1f}", f"<b>{tot['spark'] / tot['comet']:.2f}x</b> ({100 - 100 * tot['comet'] / tot['spark']:.0f}% less)", f"{faster['comet']} / {n}", f"{exe['comet']:.1f}", f"{gc['comet']:.2f}", f"{shuf['comet']:.2f}"),
])}
<div class="chart"><canvas id="totals"></canvas></div>

<h2 id="infrastructure">Benchmark infrastructure</h2>
<p><b>Methodology.</b> The three engines ran one after another in the same window on the same nodes, each alone on the cluster, over the same
S3 data with the same Spark settings; only the execution engine and its own memory split differ (every engine has 50 GB per executor; where it
puts them follows where it allocates). Each query ran once after the plan was compiled; the time is the wall-clock of the query's execution as the
runner measures it. The result files, event logs and every study behind a number are in
<a href="{{{{ '/results.html' | relative_url }}}}">Benchmark results</a>.</p>
<h3>Test environment</h3>
{env}
<h3>Versions</h3>
{versions}
<h3>Configuration</h3>
<p>Common to all three engines:</p><pre>{html.escape(conf_common)}</pre>
<div class="two"><div><p>vecruntime:</p><pre>{html.escape(conf_v)}</pre></div><div><p>Comet:</p><pre>{html.escape(conf_c)}</pre></div></div>

<h2 id="results">Performance results</h2>
<h3>Per query</h3>
<p>Seconds per query, all 103, the three engines side by side (hover for values; click a legend entry to hide an engine).</p>
<div class="chart tall"><canvas id="perquery1"></canvas></div>
<div class="chart tall"><canvas id="perquery2"></canvas></div>
<div class="chart tall"><canvas id="perquery3"></canvas></div>
<h3>Speedup over Spark, per query</h3>
<p>Spark's time divided by the engine's; above 1 is faster than Spark. Log scale.</p>
<div class="chart tall"><canvas id="speedup"></canvas></div>

<h3>Performance distribution</h3>
<div class="two">
<div><p><b>vecruntime</b> vs Spark</p>{table(["Range", "Queries", "Share"], [(k, v, f"{100 * v / n:.0f}%") for k, v in dist_v.items()])}</div>
<div><p><b>Comet</b> vs Spark</p>{table(["Range", "Queries", "Share"], [(k, v, f"{100 * v / n:.0f}%") for k, v in dist_c.items()])}</div>
</div>

<h3>Top 10 improvements -- vecruntime</h3>
{table(["Query", "Spark (s)", "vecruntime (s)", "Speedup"], top("vector"))}
<h3>Regressions -- vecruntime</h3>
{table(["Query", "Spark (s)", "vecruntime (s)", "Degradation"], regressions("vector"))}
<h3>Top 10 improvements -- Comet</h3>
{table(["Query", "Spark (s)", "Comet (s)", "Speedup"], top("comet"))}
<h3>Regressions -- Comet</h3>
{table(["Query", "Spark (s)", "Comet (s)", "Degradation"], regressions("comet"))}

<h3>Where each engine wins</h3>
<p>The heavy joins are vecruntime's: q23a {fmt_s(V["q23a"]["medianMs"])} s against Spark's {fmt_s(S["q23a"]["medianMs"])} and Comet's {fmt_s(C["q23a"]["medianMs"])};
q23b {fmt_s(V["q23b"]["medianMs"])} against {fmt_s(S["q23b"]["medianMs"])} and {fmt_s(C["q23b"]["medianMs"])}; q93 {fmt_s(V["q93"]["medianMs"])} against {fmt_s(S["q93"]["medianMs"])} and {fmt_s(C["q93"]["medianMs"])};
q64 {fmt_s(V["q64"]["medianMs"])} against {fmt_s(S["q64"]["medianMs"])} and {fmt_s(C["q64"]["medianMs"])}; q50 {fmt_s(V["q50"]["medianMs"])} against {fmt_s(S["q50"]["medianMs"])} and {fmt_s(C["q50"]["medianMs"])}.
Its shuffled hash join is a grace hash join over dictionary-encoded Arrow batches, and its shuffle moves {shuf["vector"]:.2f} TB where Spark's moves {shuf["spark"]:.2f}
(q23a: 37 GB written against Spark's 83). Comet leads where the scan and the aggregate dominate: q67 {fmt_s(C["q67"]["medianMs"])} against vecruntime's {fmt_s(V["q67"]["medianMs"])},
q4 {fmt_s(C["q4"]["medianMs"])} against {fmt_s(V["q4"]["medianMs"])}, q95 {fmt_s(C["q95"]["medianMs"])} against {fmt_s(V["q95"]["medianMs"])}, q14a/b by about 12 s each -- its native Parquet reader
and off-heap execution (GC {gc["comet"]:.2f} h against Spark's {gc["spark"]:.2f} and vecruntime's {gc["vector"]:.2f}) pay on the queries that are read-bound.
vecruntime's one heavy loss to Spark is q88 ({fmt_s(V["q88"]["medianMs"])} against {fmt_s(S["q88"]["medianMs"])}; Comet {fmt_s(C["q88"]["medianMs"])}): eight scans of <code>store_sales</code>
through Spark's own Parquet reader, where the reader's per-file request latency is the cost and the operators have little to add; its other regressions are short queries (2-9 s)
where the columnar boundary and shuffle set-up outweigh the operator gains. Each is analysed in <a href="{{{{ '/results.html' | relative_url }}}}">Benchmark results</a>.</p>

<h3>Notes</h3>
<ul>{notes}</ul>

<h2 id="table">All queries</h2>
<details open><summary>Seconds per query, the fastest engine in bold</summary>
{table(["Query", "Spark", "vecruntime", "Comet", "vecruntime speedup", "Comet speedup", "Note"], per_query_rows, "all")}
</details>

<h2 id="running">Running the benchmark</h2>
<p>The cluster runner, the Spark-on-Kubernetes manifests, the image and the data generation are in
<a href="{meta["run_doc"]}">the benchmark runner's README</a>: <code>run-matrix.sh</code> renders a <code>SparkApplication</code> per engine configuration
(<code>spark</code>, <code>vector-shuffle</code>, <code>comet</code>, ...) and writes one JSON-lines result file per run; this page is rendered from three of them by
<code>benchmarks/scripts/render-benchmark-page.py</code>.</p>

<p class="foot">TPC-DS is a benchmark of the Transaction Processing Performance Council; these results are not audited TPC results and are not comparable to
published TPC-DS results. Times are the median of one measured iteration per query on the cluster described above; run-to-run variation on the heavy queries is a few percent.</p>
</div>
<script>
const D = {json.dumps(chart_data)};
const opts = (title, yTitle, extra={{}}) => Object.assign({{
  responsive: true, maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ title: {{ display: !!title, text: title }}, legend: {{ position: 'top' }} }},
  scales: {{ y: {{ title: {{ display: true, text: yTitle }} }} }}
}}, extra);
const ds = (e, keys) => ({{ label: D.labels[e], data: keys.map(q => D.series[e][D.queries.indexOf(q)]), backgroundColor: D.colors[e], borderColor: D.colors[e] }});
new Chart(document.getElementById('totals'), {{ type: 'bar', data: {{ labels: ['total, 103 queries (s)'],
  datasets: ['spark','vector','comet'].map(e => ({{ label: D.labels[e], data: [D.totals[e]], backgroundColor: D.colors[e] }})) }},
  options: opts('Completion time, all 103 queries', 'seconds') }});
const chunks = [D.queries.slice(0, 35), D.queries.slice(35, 70), D.queries.slice(70)];
chunks.forEach((keys, i) => new Chart(document.getElementById('perquery' + (i + 1)), {{ type: 'bar',
  data: {{ labels: keys, datasets: ['spark','vector','comet'].map(e => ds(e, keys)) }},
  options: opts(i === 0 ? 'Seconds per query' : '', 'seconds') }}));
new Chart(document.getElementById('speedup'), {{ type: 'line',
  data: {{ labels: D.queries, datasets: ['vector','comet'].map(e => ({{ label: D.labels[e] + ' / Spark', borderColor: D.colors[e], backgroundColor: D.colors[e], pointRadius: 3, showLine: false,
    data: D.queries.map((q, i) => +(D.series.spark[i] / D.series[e][i]).toFixed(3)) }})) }},
  options: opts('Speedup over Spark (log scale; 1 = Spark)', 'x faster than Spark', {{ scales: {{ y: {{ type: 'logarithmic', min: 0.4, max: 4, title: {{ display: true, text: 'x faster than Spark' }} }} }} }}) }});
</script>
"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page)
    print(f"wrote {a.out}: {n} queries; totals " + ", ".join(f"{e}={tot[e]:.0f}s" for e, _, _ in ENGINES))
    if a.web_out:
        from webwrap import to_web_page
        Path(a.web_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.web_out).write_text(to_web_page(page, a.web_base, description=meta.get("dataset", "")))
        print(f"wrote {a.web_out}: web/ page under {a.web_base}")


if __name__ == "__main__":
    main()
