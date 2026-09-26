// Copyright 2025-2026 Angel Conde and the vecruntime contributors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

// Build the spark-vector static site into web/site/ as plain files -- no Jekyll.
//
//   * Markdown docs (../docs/*.md) are converted with markdown-it and wrapped in the shared shell.
//   * index.md becomes the home page: its body drops below an animated-terminal hero.
//   * The three benchmark pages (../docs/benchmarks/*.html) are GENERATED artifacts. This build does
//     NOT recompute them; it strips their Jekyll front matter and rewrites the one Liquid link they
//     carry, then wraps them in the same shell. Every number is copied through untouched, so the
//     figures are byte-identical to the docs/ pages (verified in build output).
//   * Links ending in .html that point at a ported doc are rewritten to BASE-prefixed URLs so the
//     site works under /vecruntime/.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import MarkdownIt from "markdown-it";
import anchor from "markdown-it-anchor";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(__dirname, "..");
const REPO = path.resolve(WEB, "..");
const DOCS = path.join(REPO, "docs");
const OUT = path.join(WEB, "site");

const nav = JSON.parse(fs.readFileSync(path.join(WEB, "nav.json"), "utf8"));
const BASE = nav.site.baseurl.replace(/\/$/, ""); // "/vecruntime"
const REPO_URL = nav.site.repo_url;
const SITE_TITLE = nav.site.title;

const pageTpl = fs.readFileSync(path.join(WEB, "templates", "page.html"), "utf8");
const homeTpl = fs.readFileSync(path.join(WEB, "templates", "home.html"), "utf8");

// Markdown pages to port: filename -> nav url (must match nav.json).
const MD_PAGES = [
  ["index.md", "/index.html"],
  ["operators.md", "/operators.html"],
  ["expressions.md", "/expressions.html"],
  ["configuration.md", "/configuration.html"],
  ["comet.md", "/comet.html"],
  ["iceberg.md", "/iceberg.html"],
  ["flight-shuffle.md", "/flight-shuffle.html"],
  ["results.md", "/results.html"],
];
const BENCH_PAGES = [
  "tpcds-1tb.html",
  "tpcds-1tb-graviton.html",
  "iceberg-mor.html",
];

// The set of internal doc targets, for link rewriting.
const INTERNAL = new Set([
  ...MD_PAGES.map(([, u]) => u.replace(/^\//, "")),
  ...BENCH_PAGES.map((f) => "benchmarks/" + f),
]);

const md = new MarkdownIt({ html: true, linkify: false, typographer: false });
md.use(anchor, {
  level: [2, 3],
  slugify: (s) => slugify(s),
  permalink: anchor.permalink.linkInsideHeader({ symbol: "#", placement: "before", ariaHidden: true, class: "head-anchor" }),
});

// Match the client-side slugify in site.js so anchors and TOC agree.
const _used = {};
function slugify(text) {
  let s = String(text).toLowerCase().trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
  if (!s) s = "section";
  if (_used[s] != null) { _used[s]++; s = s + "-" + _used[s]; } else { _used[s] = 0; }
  return s;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function stripFrontMatter(src) {
  const m = /^---\n([\s\S]*?)\n---\n/.exec(src);
  let title = "", description = "";
  if (m) {
    const t = /^title:\s*(.+)$/m.exec(m[1]);
    if (t) title = t[1].trim().replace(/^['"]|['"]$/g, "");
    const d = /^description:\s*(.+)$/m.exec(m[1]);
    if (d) description = d[1].trim().replace(/^['"]|['"]$/g, "");
    return { title, description, body: src.slice(m.index + m[0].length) };
  }
  return { title, description, body: src };
}

// Rewrite href="foo.html" / "benchmarks/foo.html" (and .html#frag) to BASE-prefixed absolute URLs.
function rewriteLinks(html) {
  return html.replace(/href="([^":#?]+\.html)(#[^"]*)?"/g, (full, file, frag) => {
    const clean = file.replace(/^\.\//, "");
    if (INTERNAL.has(clean)) return `href="${BASE}/${clean}${frag || ""}"`;
    return full;
  });
}

function renderSidebar(activeUrl) {
  const out = [];
  for (const group of nav.nav) {
    out.push('<div class="nav-group">');
    out.push(`<p class="nav-section">${esc(group.section)}</p>`);
    out.push("<ul>");
    for (const item of group.items) {
      const active = item.url === activeUrl ? ' class="active" aria-current="page"' : "";
      out.push(`<li><a href="${BASE}${item.url}"${active}>${esc(item.title)}</a></li>`);
    }
    out.push("</ul></div>");
  }
  return out.join("\n        ");
}

function shell({ title, description, content, activeUrl, headExtra = "", bodyClass = "" }) {
  const fullTitle = title && title !== SITE_TITLE ? `${title} \u00b7 ${SITE_TITLE}` : SITE_TITLE;
  let html = pageTpl
    .replace("__TITLE__", esc(fullTitle))
    .replace("__DESCRIPTION__", esc(description || nav.site.description))
    .replaceAll("__BASE__", BASE)
    .replaceAll("__REPO_URL__", REPO_URL)
    .replace("__HEAD_EXTRA__", headExtra)
    .replace("__SIDEBAR__", renderSidebar(activeUrl))
    .replace("__CONTENT__", content);
  if (bodyClass) html = html.replace('<div class="layout">', `<div class="layout ${bodyClass}">`);
  return html;
}

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function copyDir(src, dst) {
  ensureDir(dst);
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    if (e.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
  }
}

// -------------------------------------------------------------------------- build
fs.rmSync(OUT, { recursive: true, force: true });
ensureDir(OUT);

let count = 0;

// Markdown pages.
for (const [file, url] of MD_PAGES) {
  const src = fs.readFileSync(path.join(DOCS, file), "utf8");
  const { title, description, body } = stripFrontMatter(src);
  Object.keys(_used).forEach((k) => delete _used[k]); // reset per-page anchor dedup
  let contentHtml = rewriteLinks(md.render(body));

  let content, bodyClass = "";
  if (file === "index.md") {
    // Home: hero + terminal, then the overview body (drop its leading duplicate <h1>).
    contentHtml = contentHtml.replace(/^\s*<h[12][^>]*>[\s\S]*?<\/h[12]>/, "");
    content = homeTpl.replaceAll("__BASE__", BASE).replaceAll("__REPO_URL__", REPO_URL)
      .replace("__OVERVIEW_BODY__", contentHtml);
    bodyClass = "home";
  } else {
    content = contentHtml;
  }

  const outPath = path.join(OUT, url.replace(/^\//, ""));
  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, shell({ title, description, content, activeUrl: url, bodyClass }));
  count++;
}

// Benchmark pages: transform generated HTML, numbers untouched.
for (const file of BENCH_PAGES) {
  const src = fs.readFileSync(path.join(DOCS, "benchmarks", file), "utf8");
  const { title, description } = stripFrontMatter(src);
  let body = src.replace(/^---\n[\s\S]*?\n---\n/, "");
  // Rewrite the Liquid links Jekyll would resolve.
  body = body.replace(/\{\{\s*'(\/[^']*)'\s*\|\s*relative_url\s*\}\}/g, (m, p) => BASE + p);
  // Lift the leading Chart.js <script src> into <head>.
  const heads = [];
  body = body.replace(/^\s*(<script\s+src="[^"]+"><\/script>)\s*/g, (m, tag) => { heads.push(tag); return ""; });
  const url = "/benchmarks/" + file;
  const outPath = path.join(OUT, "benchmarks", file);
  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, shell({
    title, description, content: body, activeUrl: url, headExtra: heads.join("\n  "),
  }));
  count++;
}

// Assets + images.
copyDir(path.join(WEB, "assets"), path.join(OUT, "assets"));
if (fs.existsSync(path.join(REPO, "images"))) copyDir(path.join(REPO, "images"), path.join(OUT, "images"));

console.log(`built ${count} pages into ${path.relative(REPO, OUT)}/ (base ${BASE})`);
