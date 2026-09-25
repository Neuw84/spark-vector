(function () {
  "use strict";

  /* ---- Theme (light default, dark optional, persisted) ---- */
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("sv-theme");
    if (saved === "dark" || saved === "light") root.setAttribute("data-theme", saved);
    else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches)
      root.setAttribute("data-theme", "dark");
  } catch (e) {}

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

  /* ---- Slugify + heading anchors + TOC ---- */
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
    // clickable anchor marker
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

    /* ---- Scroll-spy ---- */
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
