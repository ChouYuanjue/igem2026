(() => {
  const data = window.LINEAGE_DATA;
  if (!data || !data.atlas || !data.atlas.root) {
    throw new Error("Atlas Engineering data is not loaded");
  }

  const atlas = data.atlas;
  const root = atlas.root;
  const storyRoot = document.getElementById("storyRoot");
  const architectureRoot = document.getElementById("architectureRoot");
  const sceneNav = document.getElementById("sceneNav");
  const searchInput = document.getElementById("storySearch");
  const searchResults = document.getElementById("searchResults");
  const dialog = document.getElementById("detailDialog");
  const detailBody = document.getElementById("detailBody");
  const loopIndex = new Map();
  const loopParent = new Map();
  const loopSystems = new Map();
  const recordIndex = new Map((data.nodes || []).map((row) => [row.id, row]));
  let focusId = root.id;
  let selectedLeaf = null;
  let resizeTimer = null;

  const esc = (value) =>
    String(value == null ? "" : value).replace(/[&<>'"]/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    })[char]);

  function walk(loop, parent = null, inheritedSystems = []) {
    const systems = loop.systems && loop.systems.length ? loop.systems : inheritedSystems;
    loopIndex.set(loop.id, loop);
    loopParent.set(loop.id, parent);
    loopSystems.set(loop.id, systems);
    (loop.children || []).forEach((child) => walk(child, loop.id, systems));
  }
  walk(root);

  const overviewLayout = new Map();
  let overviewMaxDepth = 1;

  function hierarchyDepth(loop, depth = 0) {
    overviewMaxDepth = Math.max(overviewMaxDepth, depth);
    (loop.children || []).forEach((child) => hierarchyDepth(child, depth + 1));
  }

  function buildOverviewLayout() {
    hierarchyDepth(root);
    const width = 320;
    const topMargin = 16;
    const sideMargin = 12;
    const usableWidth = width - sideMargin * 2;
    const usableHeight = 188;
    const top = root.children || [];
    overviewLayout.set(root.id, { x: width / 2, y: topMargin, depth: 0 });

    function place(loop, x0, x1, depth) {
      const x = (x0 + x1) / 2;
      const y = topMargin + (usableHeight * depth) / Math.max(1, overviewMaxDepth);
      overviewLayout.set(loop.id, { x, y, depth });
      const children = loop.children || [];
      if (!children.length) return;
      const span = (x1 - x0) / children.length;
      children.forEach((child, index) => {
        place(child, x0 + span * index, x0 + span * (index + 1), depth + 1);
      });
    }

    if (top.length) {
      const sector = usableWidth / top.length;
      top.forEach((child, index) => {
        place(
          child,
          sideMargin + sector * index,
          sideMargin + sector * (index + 1),
          1
        );
      });
    }
  }
  buildOverviewLayout();

  function phase(loop, key) {
    const value = (loop.phases || {})[key] || {};
    return {
      label: value.label || key,
      text: value.text || "",
    };
  }

  function systems(loop) {
    return loopSystems.get(loop.id) || [];
  }

  function systemChips(loop) {
    return systems(loop).map((id) => {
      const meta = atlas.systems[id] || { label: id };
      return `<span class="system-chip ${esc(id)}">${esc(meta.label)}</span>`;
    }).join("");
  }

  function ancestry(loopId) {
    const out = [];
    let cursor = loopId;
    while (cursor) {
      out.unshift(cursor);
      cursor = loopParent.get(cursor);
    }
    return out;
  }

  function overviewSystemClass(loop) {
    const ids = systems(loop);
    if (ids.length !== 1) return "mixed";
    return ids[0] || "mixed";
  }

  function overviewBranchMarkup(loop, activePath) {
    const parentPoint = overviewLayout.get(loop.id);
    if (!parentPoint) return "";
    return (loop.children || []).map((child) => {
      const childPoint = overviewLayout.get(child.id);
      if (!childPoint) return "";
      const middleY = (parentPoint.y + childPoint.y) / 2;
      const active = activePath.has(loop.id) && activePath.has(child.id);
      const cls = overviewSystemClass(child);
      return `<path class="tree-branch ${esc(cls)} ${active ? "active" : ""}" d="M ${parentPoint.x} ${parentPoint.y} C ${parentPoint.x} ${middleY}, ${childPoint.x} ${middleY}, ${childPoint.x} ${childPoint.y}"></path>${overviewBranchMarkup(child, activePath)}`;
    }).join("");
  }

  function topProgramId(loopId) {
    let cursor = loopId;
    let parent = loopParent.get(cursor);
    while (parent && parent !== root.id) {
      cursor = parent;
      parent = loopParent.get(cursor);
    }
    return cursor;
  }

  function overviewCouplingMarkup() {
    return atlas.handoffs
      .filter((edge) => topProgramId(edge.from.loop) !== topProgramId(edge.to.loop))
      .map((edge) => {
        const from = overviewLayout.get(edge.from.loop);
        const to = overviewLayout.get(edge.to.loop);
        if (!from || !to) return "";
        const midY = Math.min(from.y, to.y) - Math.max(7, Math.abs(from.x - to.x) * 0.05);
        return `<path class="tree-coupling" d="M ${from.x} ${from.y} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y}"></path>`;
      })
      .join("");
  }

  function overviewNodeMarkup(activeId, activePath, neighborhood) {
    return [...loopIndex.values()].map((loop) => {
      const point = overviewLayout.get(loop.id);
      if (!point) return "";
      const isRoot = loop.id === root.id;
      const isCurrent = loop.id === activeId;
      const isAncestor = activePath.has(loop.id) && !isCurrent;
      const isNearby = neighborhood.has(loop.id) && !isCurrent;
      const className = [
        "tree-node",
        overviewSystemClass(loop),
        isRoot ? "root" : "",
        isCurrent ? "current" : "",
        isAncestor ? "ancestor" : "",
        isNearby ? "nearby" : "",
      ].filter(Boolean).join(" ");
      const radius = isCurrent ? 4.8 : isNearby ? 3.5 : isAncestor ? 3.1 : isRoot ? 3.8 : 2.15;
      return `<circle class="${esc(className)}" cx="${point.x}" cy="${point.y}" r="${radius}"></circle>`;
    }).join("");
  }

  function overviewRegionMarkup(focus) {
    if (!focus || focus.id === root.id) return "";
    const ids = [focus.id, ...(focus.children || []).map((child) => child.id)];
    const points = ids.map((id) => overviewLayout.get(id)).filter(Boolean);
    if (!points.length) return "";
    const minX = Math.max(3, Math.min(...points.map((p) => p.x)) - 9);
    const maxX = Math.min(317, Math.max(...points.map((p) => p.x)) + 9);
    const minY = Math.max(3, Math.min(...points.map((p) => p.y)) - 9);
    const maxY = Math.min(215, Math.max(...points.map((p) => p.y)) + 9);
    return `<rect class="tree-focus-region" x="${minX}" y="${minY}" width="${Math.max(18, maxX - minX)}" height="${Math.max(18, maxY - minY)}" rx="9"></rect>`;
  }

  function overviewMarkup(focus) {
    const activeId = selectedLeaf || focus.id;
    const activePath = new Set(ancestry(activeId));
    const neighborhood = new Set([
      focus.id,
      ...(focus.children || []).map((child) => child.id),
    ]);
    const pathLabels = ancestry(activeId)
      .slice(1)
      .map((id) => loopIndex.get(id))
      .filter(Boolean)
      .map((loop) => loop.title);
    const locator = pathLabels.length ? pathLabels[pathLabels.length - 1] : "Atlas Engineering";
    return `<aside class="atlas-overview-panel" aria-label="Atlas Engineering overview">
      <header class="overview-head">
        <div><small>Whole Atlas</small><strong>Engineering tree</strong></div>
        <span>${loopIndex.size} loops</span>
      </header>
      <div class="overview-tree">
        <svg class="overview-tree-svg" viewBox="0 0 320 220" role="img" aria-label="Whole Atlas Engineering tree with the current focus highlighted">
          <g class="tree-regions">${overviewRegionMarkup(focus)}</g>
          <g class="tree-branches">${overviewBranchMarkup(root, activePath)}</g>
          <g class="tree-couplings">${overviewCouplingMarkup()}</g>
          <g class="tree-nodes">${overviewNodeMarkup(activeId, activePath, neighborhood)}</g>
        </svg>
      </div>
      <div class="overview-legend" aria-hidden="true">
        <span class="edge"><i></i>EDGE</span>
        <span class="bridge"><i></i>BRIDGE</span>
        <span class="compass"><i></i>COMPASS</span>
        <span class="mixed"><i></i>Shared</span>
      </div>
      <div class="overview-locator">
        <small>You are here</small>
        <strong>${esc(locator)}</strong>
        <span>${activeId === root.id ? "Whole system" : `${ancestry(activeId).length - 1} levels from Atlas`}</span>
      </div>
      ${focus.id !== root.id ? '<button type="button" class="overview-home" data-overview-home>See whole tree</button>' : ""}
    </aside>`;
  }

  function focusBreadcrumb() {
    return `<nav class="focus-breadcrumb" aria-label="Engineering focus path">${ancestry(focusId).map((id, index, rows) => {
      const loop = loopIndex.get(id);
      return `${index ? '<span>›</span>' : ''}<button type="button" data-focus="${esc(id)}" ${id === focusId ? 'aria-current="page"' : ''}>${esc(id === root.id ? "Atlas" : loop.title)}</button>`;
    }).join("")}</nav>`;
  }

  function parentSummary(loop) {
    if (loop.id === root.id) return "";
    return `<section class="focus-summary">
      <div class="focus-summary-head">
        <div>
          <small>${esc(loop.eyebrow || "DBTL loop")}</small>
          <h2>${esc(loop.title)}</h2>
        </div>
        <div class="focus-systems">${systemChips(loop)}</div>
      </div>
      <div class="phase-strip">
        ${["design", "build", "test", "learn"].map((key) => {
          const item = phase(loop, key);
          return `<article class="phase-strip-item ${key}"><b>${key[0].toUpperCase()}</b><div><small>${esc(item.label)}</small><span>${esc(item.text)}</span></div></article>`;
        }).join("")}
      </div>
      <div class="focus-outcome"><small>Outcome</small><strong>${esc(loop.outcome || "")}</strong></div>
    </section>`;
  }

  function loopCard(loop) {
    const childCount = (loop.children || []).length;
    const outgoing = atlas.handoffs.filter((edge) =>
      edge.from.loop === loop.id &&
      edge.from.phase === "learn" &&
      loopParent.get(edge.to.loop) === focusId
    );
    const pos = loop.position || {};
    const posStyle = focusId === root.id && pos.column
      ? `style="--map-row:${Number(pos.row || 1)};--map-column:${Number(pos.column)};--map-span:${Number(pos.span || 1)}"`
      : "";
    return `<article class="loop-cell" data-loop-cell="${esc(loop.id)}" ${posStyle}>
      <button class="loop-node" type="button" data-loop="${esc(loop.id)}" aria-label="Open ${esc(loop.title)}">
        <span class="phase-port design" data-port="design" aria-hidden="true">D</span>
        <span class="phase-port build" data-port="build" aria-hidden="true">B</span>
        <span class="phase-port test" data-port="test" aria-hidden="true">T</span>
        <span class="phase-port learn" data-port="learn" aria-hidden="true">L</span>
        <span class="loop-node-inner">
          <span class="loop-node-systems">${systemChips(loop)}</span>
          <small>${esc(loop.eyebrow || "DBTL loop")}</small>
          <strong>${esc(loop.title)}</strong>
          <em>${childCount ? `${childCount} subloops · focus` : "open details"}</em>
        </span>
      </button>
      <p class="loop-cell-outcome">${esc(loop.outcome || "")}</p>
      ${outgoing.length ? `<div class="mobile-handoffs">${outgoing.map((edge) => {
        const target = loopIndex.get(edge.to.loop);
        return `<span><b>L → D</b> ${esc(edge.label)} <i>→ ${esc(target ? target.title : edge.to.loop)}</i></span>`;
      }).join("")}</div>` : ""}
    </article>`;
  }

  function visibleHandoffs(children) {
    const ids = new Set(children.map((loop) => loop.id));
    return atlas.handoffs.filter((edge) => ids.has(edge.from.loop) && ids.has(edge.to.loop));
  }

  function mapMarkup(loop) {
    const children = loop.children || [];
    const rootClass = loop.id === root.id ? "overview-grid" : "focus-grid";
    return `<section class="causal-map" data-focus-map="${esc(loop.id)}">
      <div class="map-guide">
        <span>Fixed causal view</span>
        <strong>${children.length} loops</strong>
        <small>Arrows connect the phase that learned something to the phase whose design changed.</small>
      </div>
      <div class="map-stage">
        <svg class="handoff-svg" aria-hidden="true"><defs><marker id="arrowHead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z"></path></marker></defs><g></g></svg>
        <div class="loop-grid ${rootClass}">${children.map(loopCard).join("")}</div>
      </div>
      ${!children.length ? '<p class="empty-layer">This loop has no lower-level loops. Its DBTL record is shown above.</p>' : ""}
    </section>`;
  }

  function detailMarkup(loop) {
    if (!loop) return "";
    return `<section class="inline-detail" id="loopDetail">
      <header>
        <div><small>Loop detail</small><h3>${esc(loop.title)}</h3></div>
        <button type="button" data-close-detail aria-label="Close loop details">×</button>
      </header>
      <div class="detail-phases">
        ${["design", "build", "test", "learn"].map((key) => {
          const item = phase(loop, key);
          return `<article class="${key}"><b>${key[0].toUpperCase()}</b><div><small>${esc(item.label)}</small><p>${esc(item.text)}</p></div></article>`;
        }).join("")}
      </div>
      <div class="detail-outcome"><small>Outcome</small><strong>${esc(loop.outcome || "")}</strong></div>
      ${(loop.evidence || []).length ? `<details class="detail-evidence"><summary>Evidence anchors <span>${loop.evidence.length}</span></summary><ul>${loop.evidence.map((item) => `<li>${esc(item)}</li>`).join("")}</ul></details>` : ""}
    </section>`;
  }

  function render() {
    const focus = loopIndex.get(focusId) || root;
    storyRoot.innerHTML = `<div class="engineering-layout">
      ${overviewMarkup(focus)}
      <div class="focus-shell">
        <div class="focus-toolbar">
          <div>
            ${focusBreadcrumb()}
            <p>${focus.id === root.id
              ? "The tree at left is the whole system; this area shows only one local layer at a time."
              : "The whole-tree map keeps your position visible while this area shows only the current local branch."}</p>
          </div>
          ${focus.id !== root.id ? '<button type="button" class="focus-up" data-focus-up>← Parent</button>' : ""}
        </div>
        ${parentSummary(focus)}
        ${mapMarkup(focus)}
        ${selectedLeaf ? detailMarkup(loopIndex.get(selectedLeaf)) : ""}
      </div>
    </div>`;
    bindMap();
    requestAnimationFrame(drawHandoffs);
  }

  function focusLoop(loopId) {
    const loop = loopIndex.get(loopId);
    if (!loop) return;
    if ((loop.children || []).length) {
      focusId = loop.id;
      selectedLeaf = null;
      render();
      storyRoot.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    selectedLeaf = loop.id;
    render();
    const detail = document.getElementById("loopDetail");
    if (detail) detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function bindMap() {
    storyRoot.querySelectorAll("[data-loop]").forEach((button) => {
      button.addEventListener("click", () => focusLoop(button.dataset.loop));
    });
    storyRoot.querySelectorAll("[data-focus]").forEach((button) => {
      button.addEventListener("click", () => {
        focusId = button.dataset.focus;
        selectedLeaf = null;
        render();
      });
    });
    const up = storyRoot.querySelector("[data-focus-up]");
    if (up) up.addEventListener("click", () => {
      const parent = loopParent.get(focusId);
      if (parent) {
        focusId = parent;
        selectedLeaf = null;
        render();
      }
    });
    const overviewHome = storyRoot.querySelector("[data-overview-home]");
    if (overviewHome) overviewHome.addEventListener("click", () => {
      focusId = root.id;
      selectedLeaf = null;
      render();
    });
    const close = storyRoot.querySelector("[data-close-detail]");
    if (close) close.addEventListener("click", () => {
      selectedLeaf = null;
      render();
    });
  }

  function portPoint(loopId, phaseName, stageRect) {
    const cell = [...storyRoot.querySelectorAll("[data-loop-cell]")]
      .find((item) => item.dataset.loopCell === loopId);
    if (!cell) return null;
    const port = cell.querySelector(`[data-port="${phaseName}"]`);
    if (!port) return null;
    const rect = port.getBoundingClientRect();
    return {
      x: rect.left + rect.width / 2 - stageRect.left,
      y: rect.top + rect.height / 2 - stageRect.top,
    };
  }

  function drawHandoffs() {
    const map = storyRoot.querySelector("[data-focus-map]");
    if (!map) return;
    const stage = map.querySelector(".map-stage");
    const svg = map.querySelector(".handoff-svg");
    const group = svg && svg.querySelector("g");
    if (!stage || !svg || !group) return;
    const children = (loopIndex.get(focusId) || root).children || [];
    const edges = visibleHandoffs(children);
    const rect = stage.getBoundingClientRect();
    svg.setAttribute("viewBox", `0 0 ${Math.max(1, rect.width)} ${Math.max(1, rect.height)}`);
    group.innerHTML = "";
    if (window.matchMedia("(max-width: 760px)").matches) return;

    edges.forEach((edge, index) => {
      const from = portPoint(edge.from.loop, edge.from.phase || "learn", rect);
      const to = portPoint(edge.to.loop, edge.to.phase || "design", rect);
      if (!from || !to) return;
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const horizontal = Math.abs(dx) >= Math.abs(dy);
      const bend = Math.max(42, Math.min(120, Math.hypot(dx, dy) * 0.32));
      const c1 = horizontal
        ? { x: from.x + Math.sign(dx || 1) * bend, y: from.y }
        : { x: from.x, y: from.y + Math.sign(dy || 1) * bend };
      const c2 = horizontal
        ? { x: to.x - Math.sign(dx || 1) * bend, y: to.y }
        : { x: to.x, y: to.y - Math.sign(dy || 1) * bend };
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${from.x} ${from.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${to.x} ${to.y}`);
      path.setAttribute("class", "handoff-path");
      path.setAttribute("marker-end", "url(#arrowHead)");
      group.appendChild(path);

      if (edge.label && edges.length <= 8) {
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String((from.x + to.x) / 2));
        text.setAttribute("y", String((from.y + to.y) / 2 - 5 - (index % 2) * 5));
        text.setAttribute("class", "handoff-label");
        text.textContent = edge.label;
        group.appendChild(text);
      }
    });
  }

  function architectureMarkup() {
    return `<div class="architecture-wrap">
      <header><small>Current Atlas system</small><h2>Known graph → candidate frontier → scientific action</h2><p>Each layer keeps its own authority, while DBTL handoffs carry learned constraints across the system.</p></header>
      <div class="atlas-architecture">
        <article class="edge"><small>EDGE</small><strong>Known graph</strong><span>Canonical entities, relations, sources and evidence state</span></article>
        <i>→</i>
        <article class="bridge"><small>BRIDGE</small><strong>Candidate frontier</strong><span>Full-universe Broad order with query-gated local authority</span></article>
        <i>→</i>
        <article class="compass"><small>COMPASS</small><strong>Scientific action</strong><span>Verified research state, orchestration and next-step design</span></article>
      </div>
      <div class="architecture-return"><span>verified evidence · changing graph · new research state</span><b>↩</b><span>changes the next Design</span></div>
    </div>`;
  }

  function buildNav() {
    const targets = [
      ["atlas-root", "Overview"],
      ["edge-program", "EDGE"],
      ["bridge-program", "BRIDGE"],
      ["compass-program", "COMPASS"],
      ["knowledge-boundary", "Evaluation"],
    ];
    sceneNav.innerHTML = targets.map(([id, label]) =>
      `<button type="button" data-nav-focus="${id}">${label}</button>`
    ).join("");
    sceneNav.querySelectorAll("[data-nav-focus]").forEach((button) => {
      button.addEventListener("click", () => {
        focusId = button.dataset.navFocus;
        selectedLeaf = null;
        render();
        storyRoot.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  }

  function openRecord(id) {
    const row = recordIndex.get(id);
    if (!row) return;
    detailBody.innerHTML = `<span class="dialog-kicker">${esc((data.families || {})[row.family] || row.family)} · ${esc(row.status)}</span>
      <h3>${esc(row.label)}</h3>
      <section><small>Why</small><p>${esc(row.why)}</p></section>
      <section><small>Result</small><p>${esc(row.result)}</p></section>
      <section><small>What survived</small><p>${esc(row.legacy)}</p></section>`;
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  }

  function updateSearch() {
    const query = searchInput.value.trim().toLowerCase();
    if (!query) {
      searchResults.hidden = true;
      searchResults.innerHTML = "";
      return;
    }
    const loopMatches = [...loopIndex.values()]
      .filter((loop) => `${loop.title} ${loop.outcome || ""} ${Object.values(loop.phases || {}).map((x) => x.text || "").join(" ")}`.toLowerCase().includes(query))
      .slice(0, 8)
      .map((loop) => ({ type: "loop", id: loop.id, title: loop.title, note: loop.outcome || "DBTL loop" }));
    const recordMatches = (data.nodes || [])
      .filter((row) => `${row.label} ${row.why} ${row.result} ${row.legacy}`.toLowerCase().includes(query))
      .slice(0, Math.max(0, 10 - loopMatches.length))
      .map((row) => ({ type: "record", id: row.id, title: row.label, note: row.result || row.why }));
    const matches = [...loopMatches, ...recordMatches];
    searchResults.innerHTML = matches.map((item) =>
      `<button type="button" data-search-type="${item.type}" data-search-id="${esc(item.id)}"><strong>${esc(item.title)}</strong><small>${esc(item.note)}</small></button>`
    ).join("");
    searchResults.hidden = !matches.length;
    searchResults.querySelectorAll("[data-search-id]").forEach((button) => {
      button.addEventListener("click", () => {
        searchInput.value = "";
        searchResults.hidden = true;
        if (button.dataset.searchType === "loop") {
          const loop = loopIndex.get(button.dataset.searchId);
          if ((loop.children || []).length) {
            focusId = loop.id;
            selectedLeaf = null;
          } else {
            focusId = loopParent.get(loop.id) || root.id;
            selectedLeaf = loop.id;
          }
          render();
          storyRoot.scrollIntoView({ behavior: "smooth", block: "start" });
        } else {
          openRecord(button.dataset.searchId);
        }
      });
    });
  }

  buildNav();
  render();
  architectureRoot.innerHTML = architectureMarkup();

  dialog.querySelector(".dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  searchInput.addEventListener("input", updateSearch);
  searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") searchResults.hidden = true;
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-box")) searchResults.hidden = true;
  });
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawHandoffs, 80);
  });
})();
