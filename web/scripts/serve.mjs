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

// Serve web/site/ under the baseurl (default /vecruntime) on 127.0.0.1, mirroring GitHub Pages'
// project-site path. Usage: node scripts/serve.mjs [port]   (default 8080). Preview only.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(__dirname, "..");
const OUT = path.join(WEB, "site");
const nav = JSON.parse(fs.readFileSync(path.join(WEB, "nav.json"), "utf8"));
const BASE = nav.site.baseurl.replace(/\/$/, "");
const PORT = parseInt(process.argv[2] || "8080", 10);
const HOST = "127.0.0.1";

const TYPES = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
  ".jpg": "image/jpeg", ".json": "application/json", ".ico": "image/x-icon",
};

const server = http.createServer((req, res) => {
  let url = decodeURIComponent(req.url.split("?")[0]);
  if (BASE && url === BASE) { res.writeHead(302, { Location: BASE + "/" }); return res.end(); }
  if (BASE && url.startsWith(BASE + "/")) url = url.slice(BASE.length);
  else if (BASE && url !== "/") { res.writeHead(404); return res.end("Not found (serve under " + BASE + "/)"); }
  if (url.endsWith("/")) url += "index.html";
  const fp = path.join(OUT, path.normalize(url).replace(/^(\.\.[/\\])+/, ""));
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404, { "Content-Type": "text/plain" }); return res.end("404: " + url); }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(fp)] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`serving ${path.relative(WEB, OUT)}/ at http://${HOST}:${PORT}${BASE}/`);
});
