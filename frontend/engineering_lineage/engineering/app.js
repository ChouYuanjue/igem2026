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
  const topProgram = new Map();
  const storylineMeta = new Map((atlas.storylines || []).map((row) => [row.id, row]));

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

  function walk(loop, parent = null, inheritedSystems = [], trunk = null) {
    const systems = loop.systems && loop.systems.length ? loop.systems : inheritedSystems;
    const nextTrunk = parent === root.id ? loop.id : (trunk || loop.id);
    loopIndex.set(loop.id, loop);
    loopParent.set(loop.id, parent);
    loopSystems.set(loop.id, systems);
    topProgram.set(loop.id, nextTrunk);
    (loop.children || []).forEach((child) => walk(child, loop.id, systems, nextTrunk));
  }
  loopIndex.set(root.id, root);
  loopParent.set(root.id, null);
  loopSystems.set(root.id, root.systems || []);
  topProgram.set(root.id, root.id);
  (root.children || []).forEach((child) => walk(child, root.id, child.systems || [], child.id));

  function systems(loop) {
    return loopSystems.get(loop.id) || [];
  }

  function phase(loop, key) {
    const value = (loop.phases || {})[key] || {};
    return { label: value.label || key, text: value.text || "" };
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

  function systemChips(loop) {
    return systems(loop).map((id) => {
      const meta = atlas.systems[id] || { label: id };
      return `<span class="system-chip ${esc(id)}">${esc(meta.label)}</span>`;
    }).join("");
  }

  function isJunction(loop) {
    return systems(loop).length > 1;
  }

  function causalNext(loopId) {
    return (atlas.handoffs || []).filter((edge) =>
      edge.from.loop === loopId &&
      edge.from.phase === "learn" &&
      edge.to.phase === "design" &&
      loopIndex.has(edge.to.loop)
    );
  }

  function navigateToLoop(loopId, options = {}) {
    const loop = loopIndex.get(loopId);
    if (!loop) return;
    const hasChildren = (loop.children || []).length > 0;
    if (hasChildren || options.forceFocus) {
      focusId = loop.id;
      selectedLeaf = null;
    } else {
      focusId = loopParent.get(loop.id) || root.id;
      selectedLeaf = loop.id;
    }
    render();
    if (options.scroll !== false) {
      storyRoot.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function nextActionsMarkup(loop, compact = false) {
    const edges = causalNext(loop.id);
    if (!edges.length) return "";
    return `<div class="${compact ? "focus-nexts" : "loop-nexts"}" aria-label="Next engineering loops">
      ${edges.map((edge) => {
        const target = loopIndex.get(edge.to.loop);
        if (!target) return "";
        return `<button type="button" class="causal-next" data-next-loop="${esc(target.id)}" title="${esc(edge.label || "Continue from Learn to the next Design")}">
          <span>Next</span>
          <strong>${esc(target.title)}</strong>
          <i aria-hidden="true">→</i>
        </button>`;
      }).join("")}
    </div>`;
  }

  function focusBreadcrumb() {
    return `<nav class="focus-breadcrumb" aria-label="Engineering focus path">${ancestry(focusId).map((id, index) => {
      const loop = loopIndex.get(id);
      const title = id === root.id ? "Atlas" : loop.title;
      return `${index ? '<span>›</span>' : ''}<button type="button" data-focus="${esc(id)}" ${id === focusId ? 'aria-current="page"' : ''}>${esc(title)}</button>`;
    }).join("")}</nav>`;
  }

  function storylineForLoop(loopId) {
    const trunkId = topProgram.get(loopId);
    if (trunkId === "edge-program") return "edge";
    if (trunkId === "bridge-program") return "bridge";
    if (trunkId === "compass-program") return "compass";
    return "mixed";
  }

  function layoutStorylines() {
    const width = 390;
    const xStart = 72;
    const xEnd = 370;
    const lanes = {
      edge: { y: 54, direction: -1, root: "edge-program" },
      bridge: { y: 112, direction: 1, root: "bridge-program" },
      compass: { y: 170, direction: 1, root: "compass-program" },
    };
    const positions = new Map();
    const hierarchyEdges = [];
    const depthHints = [];

    // One visual head for the three persistent storylines. Atlas Engineering is
    // already the semantic root; expose it in the overview instead of letting
    // EDGE / BRIDGE / COMPASS appear to start independently.
    positions.set(root.id, { x: 20, y: lanes.bridge.y, depth: -1, lane: "mixed", main: true });

    Object.entries(lanes).forEach(([laneId, lane]) => {
      const trunk = loopIndex.get(lane.root);
      if (!trunk) return;
      positions.set(trunk.id, { x: xStart - 16, y: lane.y, depth: 0, lane: laneId, main: true });
      hierarchyEdges.push([root.id, trunk.id]);
      const mains = trunk.children || [];
      const step = mains.length > 1 ? (xEnd - xStart) / (mains.length - 1) : 0;

      mains.forEach((main, index) => {
        const x = xStart + step * index;
        positions.set(main.id, { x, y: lane.y, depth: 1, lane: laneId, main: true });
        hierarchyEdges.push([trunk.id, main.id]);

        const left = index === 0 ? xStart - 12 : x - step / 2 + 3;
        const right = index === mains.length - 1 ? xEnd + 12 : x + step / 2 - 3;
        const baseDirection = laneId === "bridge" ? (index % 2 === 0 ? -1 : 1) : lane.direction;

        const children = main.children || [];
        if (children.length) {
          const span = (right - left) / children.length;
          children.forEach((child, childIndex) => {
            const cx = left + span * (childIndex + 0.5);
            const cy = lane.y + baseDirection * 19;
            positions.set(child.id, { x: cx, y: cy, depth: 2, lane: laneId, main: false });
            hierarchyEdges.push([main.id, child.id]);
            const deeperCount = countDescendants(child);
            if (deeperCount) {
              depthHints.push({
                x: cx,
                y: cy,
                lane: laneId,
                direction: baseDirection,
                count: deeperCount,
              });
            }
          });
        }
      });
    });

    return { width, height: 224, lanes, positions, hierarchyEdges, depthHints };
  }

  function countDescendants(loop) {
    return (loop.children || []).reduce(
      (total, child) => total + 1 + countDescendants(child),
      0
    );
  }

  const storylineLayout = layoutStorylines();

  function currentActiveId() {
    return selectedLeaf || focusId;
  }

  function activePathSet() {
    return new Set(ancestry(currentActiveId()));
  }

  function overviewActiveId() {
    const path = ancestry(currentActiveId()).reverse();
    return path.find((id) => storylineLayout.positions.has(id)) || root.id;
  }

  function overviewDepthHintsMarkup() {
    return storylineLayout.depthHints.map((hint) => {
      const length = Math.min(11, 4 + hint.count * 1.15);
      const endY = hint.y + hint.direction * length;
      const dotCount = Math.min(3, hint.count);
      const dots = Array.from({ length: dotCount }, (_, index) => {
        const offset = (index - (dotCount - 1) / 2) * 2.8;
        return `<circle cx="${hint.x + offset}" cy="${endY}" r="1.05"></circle>`;
      }).join("");
      return `<g class="story-depth-hint ${hint.lane}"><line x1="${hint.x}" y1="${hint.y}" x2="${hint.x}" y2="${endY}"></line>${dots}</g>`;
    }).join("");
  }

  function overviewHierarchyMarkup(activePath) {
    return storylineLayout.hierarchyEdges.map(([fromId, toId]) => {
      const from = storylineLayout.positions.get(fromId);
      const to = storylineLayout.positions.get(toId);
      if (!from || !to) return "";
      const active = activePath.has(fromId) && activePath.has(toId);
      const lane = to.lane;
      const midX = (from.x + to.x) / 2;
      const d = `M ${from.x} ${from.y} C ${midX} ${from.y}, ${midX} ${to.y}, ${to.x} ${to.y}`;
      return `<path class="story-branch-hit" d="${d}" data-overview-loop="${esc(toId)}" role="button" tabindex="0" aria-label="Open ${esc((loopIndex.get(toId) || {}).title || toId)}"></path><path class="story-branch ${lane} ${active ? "active" : ""}" d="${d}" pointer-events="none"></path>`;
    }).join("");
  }

  function overviewTrunksMarkup() {
    return (atlas.storylines || []).map((line) => {
      const lane = storylineLayout.lanes[line.id];
      if (!lane) return "";
      const trunkPoint = storylineLayout.positions.get(lane.root);
      const startX = trunkPoint ? trunkPoint.x : 56;
      return `<g class="story-trunk ${esc(line.id)}">
        <line class="story-trunk-hit" x1="${startX}" y1="${lane.y}" x2="376" y2="${lane.y}" data-overview-loop="${esc(lane.root)}" role="button" tabindex="0" aria-label="Open ${esc(line.label)}"></line>
        <line class="story-trunk-visible" x1="${startX}" y1="${lane.y}" x2="376" y2="${lane.y}"></line>
      </g>`;
    }).join("");
  }

  function overviewCrossMarkup(activePath) {
    return (atlas.handoffs || [])
      .filter((edge) => topProgram.get(edge.from.loop) !== topProgram.get(edge.to.loop))
      .map((edge) => {
        const from = storylineLayout.positions.get(edge.from.loop);
        const to = storylineLayout.positions.get(edge.to.loop);
        if (!from || !to) return "";
        const active = activePath.has(edge.from.loop) || activePath.has(edge.to.loop);
        const dx = Math.abs(to.x - from.x);
        const bend = Math.max(12, Math.min(34, dx * 0.22));
        const direction = to.x >= from.x ? 1 : -1;
        const c1x = from.x + direction * bend;
        const c2x = to.x - direction * bend;
        const d = `M ${from.x} ${from.y} C ${c1x} ${from.y}, ${c2x} ${to.y}, ${to.x} ${to.y}`;
        return `<path class="story-cross-hit" d="${d}" data-overview-loop="${esc(edge.to.loop)}" role="button" tabindex="0" aria-label="Follow to ${esc((loopIndex.get(edge.to.loop) || {}).title || edge.to.loop)}"></path><path class="story-cross ${active ? "active" : ""}" d="${d}" pointer-events="none"></path>`;
      }).join("");
  }

  function overviewNodesMarkup(activePath) {
    const activeId = overviewActiveId();
    return [...storylineLayout.positions.entries()].map(([id, point]) => {
      const loop = loopIndex.get(id);
      if (!loop) return "";
      const current = id === activeId;
      const ancestor = activePath.has(id) && !current;
      const junction = isJunction(loop);
      const cls = ["story-node", point.lane, point.main ? "main" : "sub", junction ? "junction" : "", current ? "current" : "", ancestor ? "ancestor" : ""].filter(Boolean).join(" ");
      const isAtlasHead = id === root.id;
      const r = current ? 5 : isAtlasHead ? 5 : point.main ? (point.lane === "bridge" ? 4.4 : 3.8) : 2.25;
      const hitR = isAtlasHead ? 10 : point.main ? 8.5 : 6.5;
      return `<g class="story-node-group">
        <circle class="story-node-hit" cx="${point.x}" cy="${point.y}" r="${hitR}" data-overview-loop="${esc(id)}" role="button" tabindex="0" aria-label="Open ${esc(loop.title)}"></circle>
        <circle class="${esc(cls)}" cx="${point.x}" cy="${point.y}" r="${r}" pointer-events="none"></circle>
      </g>`;
    }).join("");
  }

  function overviewMarkup() {
    const activePath = activePathSet();
    const current = loopIndex.get(currentActiveId()) || root;
    const currentLine = storylineForLoop(current.id);
    const depth = Math.max(0, ancestry(current.id).length - 2);
    return `<aside class="atlas-overview-panel" aria-label="Atlas Engineering overview">
      <header class="overview-head">
        <div><small>Whole system</small><strong>Three evolving storylines</strong></div>
        <span>DBTL overview</span>
      </header>
      <div class="storyline-map">
        <svg viewBox="0 0 ${storylineLayout.width} ${storylineLayout.height}" role="img" aria-label="EDGE, BRIDGE and COMPASS evolving in parallel with cross-system junctions">
          <g class="story-trunks">${overviewTrunksMarkup()}</g>
          <g class="story-branches">${overviewHierarchyMarkup(activePath)}</g>
          <g class="story-depth-hints">${overviewDepthHintsMarkup()}</g>
          <g class="story-crossings">${overviewCrossMarkup(activePath)}</g>
          <g class="story-nodes">${overviewNodesMarkup(activePath)}</g>
        </svg>
      </div>
      <div class="overview-legend">
        <span class="edge"><i></i>EDGE</span>
        <span class="bridge"><i></i>BRIDGE</span>
        <span class="compass"><i></i>COMPASS</span>
        <span class="junction"><i></i>junction</span>
      </div>
      <div class="overview-locator">
        <small>You are here</small>
        <strong>${esc(current.id === root.id ? "Atlas Engineering" : current.title)}</strong>
        <span>${current.id === root.id ? "whole system" : `${currentLine.toUpperCase()} · depth ${depth}`}</span>
      </div>
      ${focusId !== root.id ? '<button type="button" class="overview-home" data-overview-home>Back to the three trunks</button>' : ""}
    </aside>`;
  }

  function parentSummary(loop) {
    if (loop.id === root.id) return "";
    return `<section class="focus-summary ${isJunction(loop) ? "junction" : ""}">
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
      ${nextActionsMarkup(loop, true)}
    </section>`;
  }

  function visibleHandoffs(children) {
    const ids = new Set(children.map((loop) => loop.id));
    return (atlas.handoffs || []).filter((edge) => ids.has(edge.from.loop) && ids.has(edge.to.loop));
  }

  function loopCard(loop) {
    const childCount = (loop.children || []).length;
    const rootTrunk = loopParent.get(loop.id) === root.id;
    const cls = [
      "loop-cell",
      rootTrunk ? "root-trunk" : "",
      rootTrunk && systems(loop)[0] === "bridge" ? "bridge-trunk" : "",
      isJunction(loop) ? "junction-loop" : "",
    ].filter(Boolean).join(" ");
    return `<article class="${cls}" data-loop-cell="${esc(loop.id)}">
      <button class="loop-node" type="button" data-loop="${esc(loop.id)}" aria-label="Open ${esc(loop.title)}">
        <span class="phase-port design" data-port="design" aria-hidden="true">D</span>
        <span class="phase-port build" data-port="build" aria-hidden="true">B</span>
        <span class="phase-port test" data-port="test" aria-hidden="true">T</span>
        <span class="phase-port learn" data-port="learn" aria-hidden="true">L</span>
        <span class="loop-node-inner">
          <span class="loop-node-systems">${systemChips(loop)}</span>
          <small>${esc(loop.eyebrow || "DBTL loop")}</small>
          <strong>${esc(loop.title)}</strong>
          <em>${childCount ? `${childCount} subloops ↓` : "details"}</em>
        </span>
      </button>
      <p class="loop-cell-outcome">${esc(loop.outcome || "")}</p>
      ${nextActionsMarkup(loop)}
    </article>`;
  }

  function mapMarkup(loop) {
    const children = loop.children || [];
    const rootClass = loop.id === root.id ? "three-trunk-grid" : "focus-grid";
    const guide = loop.id === root.id
      ? '<span>Three engineering functions</span><strong>EDGE · BRIDGE · COMPASS</strong><small>Click a storyline, branch or station to jump there. Circle / ↓ opens sub-loops; Next / → follows a Learn → Design handoff.</small>'
      : `<span>Local branch</span><strong>${children.length} loops</strong><small>Circle / ↓ explores sub-loops. Next / → continues to the loop whose Design was triggered by this loop’s Learn.</small>`;
    return `<section class="causal-map" data-focus-map="${esc(loop.id)}">
      <div class="map-guide">${guide}</div>
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
      <header><div><small>Loop detail</small><h3>${esc(loop.title)}</h3></div><button type="button" data-close-detail aria-label="Close loop details">×</button></header>
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
      ${overviewMarkup()}
      <div class="focus-shell">
        <div class="focus-toolbar">
          <div>
            ${focusBreadcrumb()}
            <p>${focus.id === root.id
              ? "EDGE, BRIDGE and COMPASS each track a different engineering function. Open one storyline to inspect only that local branch."
              : "The overview keeps all three storylines visible while this panel expands only the current branch."}</p>
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
    navigateToLoop(loopId, { scroll: true });
    if (!(loop.children || []).length) {
      const detail = document.getElementById("loopDetail");
      if (detail) detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
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
    storyRoot.querySelectorAll("[data-next-loop]").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        navigateToLoop(button.dataset.nextLoop, { scroll: true });
      });
    });
    storyRoot.querySelectorAll("[data-overview-loop]").forEach((target) => {
      const activate = () => navigateToLoop(target.dataset.overviewLoop, { scroll: true });
      target.addEventListener("click", activate);
      target.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
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
    const home = storyRoot.querySelector("[data-overview-home]");
    if (home) home.addEventListener("click", () => {
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
      const bend = Math.max(38, Math.min(112, Math.hypot(dx, dy) * 0.30));
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
      <header><small>Atlas system</small><h2>Three engineering functions, one coupled research workflow</h2><p>Atlas EDGE organizes the known graph and evidence; Atlas BRIDGE ranks enzyme–reaction candidates in open space; Atlas COMPASS turns research intent into reusable scientific state. Crossings show where learning in one function changes the next design in another.</p></header>
      <div class="atlas-architecture three-lines">
        <article class="edge"><small>Atlas EDGE</small><strong>Known graph & evidence</strong><span>Identity, relations, provenance, evidence and graph growth</span></article>
        <article class="bridge"><small>Atlas BRIDGE</small><strong>Enzyme–reaction inference</strong><span>Full-universe Broad order with query-gated local authority</span></article>
        <article class="compass"><small>Atlas COMPASS</small><strong>Research orchestration</strong><span>Intent, verified workspace state, observations and iterative research</span></article>
      </div>
    </div>`;
  }

  function buildNav() {
    const targets = [
      ["atlas-root", "Overview"],
      ["edge-program", "EDGE"],
      ["bridge-program", "BRIDGE"],
      ["compass-program", "COMPASS"],
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
