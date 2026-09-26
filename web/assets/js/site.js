(function () {
  "use strict";

  var root = document.documentElement;

  /* ---- Theme toggle (initial theme is set inline in <head>) ---- */
  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("sv-theme", next); } catch (e) {}
    });
  }

  /* ---- Mobile sidebar ---- */
  var menuBtn = document.getElementById("menu-toggle");
  var sidebar = document.getElementById("sidebar");
  var backdrop = document.getElementById("sidebar-backdrop");
  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove("open");
    if (backdrop) backdrop.hidden = true;
    if (menuBtn) menuBtn.setAttribute("aria-expanded", "false");
  }
  if (menuBtn && sidebar) {
    menuBtn.addEventListener("click", function () {
      var open = sidebar.classList.toggle("open");
      if (backdrop) backdrop.hidden = !open;
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }
  if (backdrop) backdrop.addEventListener("click", closeSidebar);

  /* ---- Terminal animation (home page) ---- */
  var term = document.getElementById("terminal-body");
  if (term) {
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    // The lines to type. Each: {html: markup, pause: ms after}. Instant lines have no typing.
    var lines;
    try { lines = JSON.parse(term.getAttribute("data-lines") || "[]"); } catch (e) { lines = []; }

    function finalFrame() {
      var html = lines.map(function (l) { return l.html; }).join("\n");
      term.innerHTML = html;
      term.setAttribute("aria-busy", "false");
    }

    if (reduce || !lines.length) {
      finalFrame();
    } else {
      term.setAttribute("aria-busy", "true");
      var cursor = document.createElement("span");
      cursor.className = "terminal-cursor";
      var idx = 0;
      var buf = "";
      function typeLine() {
        if (idx >= lines.length) {
          term.innerHTML = buf;
          term.appendChild(cursor);
          term.setAttribute("aria-busy", "false");
          return;
        }
        var line = lines[idx];
        // Type a "typed" line character-by-character on its plain text, then
        // swap in the highlighted markup; "output" lines appear at once.
        if (line.typed) {
          var plain = line.text || "";
          var i = 0;
          (function tick() {
            term.innerHTML = buf + escapeHtml(plain.slice(0, i));
            term.appendChild(cursor);
            term.scrollTop = term.scrollHeight;
            if (i < plain.length) { i++; setTimeout(tick, 14); }
            else { buf += line.html + "\n"; idx++; setTimeout(typeLine, line.pause || 260); }
          })();
        } else {
          buf += line.html + "\n";
          term.innerHTML = buf;
          term.appendChild(cursor);
          term.scrollTop = term.scrollHeight;
          idx++;
          setTimeout(typeLine, line.pause || 140);
        }
      }
      function escapeHtml(s) {
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      }
      // Start when the terminal scrolls into view (or immediately if already visible).
      if ("IntersectionObserver" in window) {
        var io = new IntersectionObserver(function (entries) {
          if (entries[0].isIntersecting) { io.disconnect(); typeLine(); }
        }, { threshold: 0.2 });
        io.observe(term);
      } else {
        typeLine();
      }
    }
  }

  /* ---- Slugify + heading anchors + TOC (doc pages) ---- */
  var article = document.querySelector(".doc");
  var tocNav = document.getElementById("toc-nav");
  if (!article) return;

  var used = {};
  function slugify(text) {
    var s = text.toLowerCase().trim()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");
    if (!s) s = "section";
    if (used[s] != null) { used[s]++; s = s + "-" + used[s]; } else { used[s] = 0; }
    return s;
  }

  var heads = article.querySelectorAll("h2, h3");
  var items = [];
  heads.forEach(function (h) {
    if (!h.id) h.id = slugify(h.textContent);
    var a = document.createElement("a");
    a.className = "head-anchor";
    a.href = "#" + h.id;
    a.setAttribute("aria-hidden", "true");
    a.textContent = "#";
    h.insertBefore(a, h.firstChild);
    items.push(h);
  });

  if (tocNav && items.length) {
    items.forEach(function (h) {
      var link = document.createElement("a");
      link.href = "#" + h.id;
      link.textContent = (h.querySelector(".head-anchor") ? h.textContent.replace(/^#/, "") : h.textContent).trim();
      if (h.tagName === "H3") link.className = "toc-h3";
      link.addEventListener("click", closeSidebar);
      tocNav.appendChild(link);
    });

    var links = Array.prototype.slice.call(tocNav.querySelectorAll("a"));
    var byId = {};
    links.forEach(function (l) { byId[l.getAttribute("href").slice(1)] = l; });

    var spy = function () {
      var top = window.scrollY + parseInt(getComputedStyle(root).getPropertyValue("--header-h")) + 24;
      var current = items[0];
      for (var i = 0; i < items.length; i++) {
        if (items[i].offsetTop <= top) current = items[i]; else break;
      }
      links.forEach(function (l) { l.classList.remove("active"); });
      if (current && byId[current.id]) byId[current.id].classList.add("active");
    };
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (!ticking) { window.requestAnimationFrame(function () { spy(); ticking = false; }); ticking = true; }
    }, { passive: true });
    spy();
  }
})();
