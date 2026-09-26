# spark-vector website (`web/`)

A static documentation site for spark-vector, modelled on the
[Apache DataFusion Comet](https://datafusion.apache.org/comet/) site: a hero with an animated
terminal, a top nav with a GitHub link, a docs sidebar, an on-page table of contents, and light/dark
themes. **No Jekyll.** The build output is plain HTML/CSS/JS files.

This is a *second* site living alongside the existing Jekyll site in `../docs/`. `../docs/` is
untouched and still works; nothing here changes it.

## Tooling

A tiny, pinned Node generator — not a framework:

| Tool | Version | Why |
|---|---|---|
| [Node.js](https://nodejs.org/) | >= 20 (built on v22) | runs the build/serve scripts |
| [`markdown-it`](https://github.com/markdown-it/markdown-it) | `15.0.2` (exact) | Markdown → HTML for the doc pages |
| [`markdown-it-anchor`](https://github.com/valeriangalliat/markdown-it-anchor) | `10.0.0` (exact) | heading ids/anchors matching the client-side TOC slugs |

Versions are pinned exactly in `package.json` and locked in `package-lock.json`. The site shell
(header, sidebar, TOC, footer, themes, hero, terminal) is our own HTML/CSS/JS in `templates/` and
`assets/` — the same Comet-style shell the `../docs/` Jekyll site already used, ported to plain-file
tokens instead of Liquid.

### Why not Comet's own stack?

Comet's site is [Sphinx](https://www.sphinx-doc.org/) (`apache/datafusion-comet/docs`: `build.sh`,
`requirements.txt`, `source/conf.py`), and it renders Mermaid diagrams by driving **headless Chrome**
through `mermaid-cli`/puppeteer. That toolchain is heavier than this site needs and its Chrome step is
fragile under the AppArmor user-namespace restriction on recent Linux (the very issue Comet's own docs
call out). Our content is the existing Markdown docs plus three already-generated benchmark HTML pages,
so a small Markdown-to-HTML generator into a shared shell is a better fit and keeps the output as plain
files.

## Build

```bash
cd web
npm ci            # or: npm install  (restores the pinned deps)
npm run build     # writes the site into web/site/
```

## Preview

```bash
cd web
npm run serve             # serves web/site/ at http://127.0.0.1:8080/spark-vector/
npm run serve -- 8099     # choose a port
```

The preview server binds to `127.0.0.1` only and serves under `/spark-vector/`, mirroring the path a
GitHub Pages **project** site uses, so links and assets resolve exactly as they will in production.

## Content

- **Overview / home** (`../docs/index.md`) — rendered with the animated-terminal hero.
- **User guide** — `operators.md`, `expressions.md`, `configuration.md`, `comet.md`, `iceberg.md`,
  `flight-shuffle.md`.
- **Benchmarks** — `results.md` (the lab notebook) plus the three generated benchmark pages.

Doc content is read straight from `../docs/*.md`. To change wording, edit those Markdown files (they
serve both this site and the Jekyll site) and rebuild.

### Benchmark pages stay generated

`../docs/benchmarks/{tpcds-1tb,tpcds-1tb-graviton,iceberg-mor}.html` are generated artifacts. This
build does **not** recompute any number: it strips the Jekyll front matter, rewrites the one Liquid
`relative_url` link they carry, lifts their Chart.js `<script>` into `<head>`, and wraps the body in
this site's shell. Every figure is copied through byte-for-byte identical to the `../docs/` page.

The two Python generators grew an **optional** `web/` output mode that does the same wrapping at
generation time, without changing their `docs/` output:

```bash
# docs/ output is exactly as before; --web-out ADDITIONALLY writes a standalone web/ page:
benchmarks/scripts/render-benchmark-page.py \
  --spark spark.jsonl --vector vector.jsonl --comet comet.jsonl \
  --out   docs/benchmarks/tpcds-1tb.html \
  --web-out web/site/benchmarks/tpcds-1tb.html --web-base /spark-vector

benchmarks/scripts/render-graviton-page.py \
  --spark spark.jsonl --vector vector.jsonl \
  --x86-page docs/benchmarks/tpcds-1tb.html \
  --out   docs/benchmarks/tpcds-1tb-graviton.html \
  --web-out web/site/benchmarks/tpcds-1tb-graviton.html --web-base /spark-vector
```

For a normal `npm run build` you do **not** need to run the generators: it transforms the committed
`docs/benchmarks/*.html` directly. Re-run the generators only when you have fresh result files, then
rebuild.

## Build output is not committed

`web/site/` (the built HTML) and `web/node_modules/` are git-ignored (`web/.gitignore`). The output is
fully reproducible from the sources with `npm ci && npm run build`, so committing it would only add
churn and merge conflicts. Publish by building in CI (below).

## Publishing

The site is built for a GitHub Pages **project** site under `/spark-vector/` (the same base URL the
current Jekyll site uses). Two ways to publish it — the owner chooses; this PR touches **no**
`.github/` files:

1. **GitHub Actions (recommended).** Add a Pages workflow that runs `cd web && npm ci && npm run build`
   and uploads `web/site/` with `actions/upload-pages-artifact` + `actions/deploy-pages`, then set
   **Settings → Pages → Source = GitHub Actions**. This replaces the Jekyll-from-`/docs` publish.

2. **Publish the built files from a branch.** Run `npm run build`, copy `web/site/` to the branch/dir
   GitHub Pages serves (e.g. a `gh-pages` branch, or move it under `docs/` if you retire the Jekyll
   site), and point **Settings → Pages → Source** at it.

Either way the base URL stays `/spark-vector/`. If you ever serve at the domain root instead, change
`baseurl` in `web/nav.json` and rebuild.
