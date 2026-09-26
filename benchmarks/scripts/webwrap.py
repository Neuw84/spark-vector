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
"""Optional `web/` output mode for the benchmark-page generators.

The generators build one string, `page`, that begins with Jekyll front matter
(``---\\nlayout: default\\ntitle: ...\\n---``) and, in the body, a ``<script>``
tag pulling Chart.js plus a ``<style>`` block and the ``<div class="bench">``
content. That string is what the `docs/` (Jekyll) build consumes, unchanged.

`to_web_page()` takes the SAME `page` string and re-wraps it as a self-contained
static HTML file for the plain-file site under `web/`:

  * the Jekyll front matter is stripped,
  * the ``{{ '/PATH' | relative_url }}`` Liquid links Jekyll would resolve are
    rewritten to ``<base>/PATH`` (e.g. ``/spark-vector/results.html``),
  * the Chart.js ``<script>`` that lived at the top of the body is lifted into
    ``<head>`` (kept, not dropped), and
  * everything else -- every number, table, chart series and note -- is copied
    verbatim, so the rendered figures are byte-identical to the `docs/` page.

Nothing here recomputes a result: the numbers come from `page` as the generator
already produced them.
"""
import re
from pathlib import Path

_TEMPLATE = None


def _template():
    global _TEMPLATE
    if _TEMPLATE is None:
        # web/templates/page.html, relative to benchmarks/scripts/ -> ../../web/...
        p = Path(__file__).resolve().parent.parent.parent / "web" / "templates" / "page.html"
        _TEMPLATE = p.read_text()
    return _TEMPLATE


def _strip_front_matter(page: str) -> tuple[str, str]:
    """Return (title, body) splitting the leading Jekyll front matter off `page`."""
    m = re.match(r"^---\n(.*?)\n---\n", page, re.DOTALL)
    title = ""
    if m:
        block = m.group(1)
        tm = re.search(r'^title:\s*(.+)$', block, re.MULTILINE)
        if tm:
            title = tm.group(1).strip().strip("'\"")
        body = page[m.end():]
    else:
        body = page
    return title, body


def _lift_head_scripts(body: str) -> tuple[str, str]:
    """Pull leading external <script src=...> tags (Chart.js) out of the body into head."""
    head = []
    while True:
        m = re.match(r'\s*(<script\s+src="[^"]+"></script>)\s*', body)
        if not m:
            break
        head.append(m.group(1))
        body = body[m.end():]
    return "\n  ".join(head), body


def to_web_page(page: str, base: str, description: str = "") -> str:
    """Wrap a generator `page` string as a standalone site page for `web/` under `base`."""
    base = base.rstrip("/")
    title, body = _strip_front_matter(page)
    # Rewrite the Liquid links Jekyll would have resolved.
    body = re.sub(r"\{\{\s*'(/[^']*)'\s*\|\s*relative_url\s*\}\}", lambda m: base + m.group(1), body)
    head_extra, body = _lift_head_scripts(body)

    site_title = "spark-vector"
    full_title = f"{title} \u00b7 {site_title}" if title and title != site_title else site_title
    desc = description or title or site_title

    # Sidebar is rendered by the site build; the generator standalone page uses a
    # minimal sidebar so the page is viewable on its own, but the site build
    # overwrites __SIDEBAR__ via its own renderer when it consumes this. For the
    # generator-direct path we inject the shared nav here.
    sidebar = _render_sidebar(base, active="/" + Path(_guess_url(title)).name if title else "")

    html = _template()
    html = html.replace("__TITLE__", _esc(full_title))
    html = html.replace("__DESCRIPTION__", _esc(desc))
    html = html.replace("__BASE__", base)
    html = html.replace("__REPO_URL__", "https://github.com/spark-vector/spark-vector")
    html = html.replace("__HEAD_EXTRA__", head_extra)
    html = html.replace("__SIDEBAR__", sidebar)
    html = html.replace("__CONTENT__", body)
    return html


def _guess_url(title: str) -> str:
    return "results.html"


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _render_sidebar(base: str, active: str = "") -> str:
    import json
    nav_path = Path(__file__).resolve().parent.parent.parent / "web" / "nav.json"
    nav = json.loads(nav_path.read_text())["nav"]
    out = []
    for group in nav:
        out.append('<div class="nav-group">')
        out.append(f'<p class="nav-section">{_esc(group["section"])}</p>')
        out.append("<ul>")
        for item in group["items"]:
            href = base + item["url"]
            cls = ' class="active" aria-current="page"' if item["url"] == active else ""
            out.append(f'<li><a href="{href}"{cls}>{_esc(item["title"])}</a></li>')
        out.append("</ul></div>")
    return "\n        ".join(out)
