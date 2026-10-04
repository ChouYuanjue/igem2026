(() => {
  const data = window.LINEAGE_DATA;
  if (!data) throw new Error('LINEAGE_DATA is not loaded');

  const nodes = data.nodes;
  const crossLinks = data.crossLinks;
  const families = data.families;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const actualChildren = new Map(nodes.map(node => [node.id, []]));
  nodes.forEach(node => {
    if (node.parent) actualChildren.get(node.parent).push(node.id);
  });

  const statusColors = {
    root: '#171914', keep: '#315f49', turn: '#9a6e2e', local: '#49697e',
    reject: '#a14d3e', historical: '#715878', considered: '#85867f'
  };

  const treeReader = document.getElementById('treeReader');
  const detailPanel = document.getElementById('detailPanel');
  const searchInput = document.getElementById('treeSearch');
  const overviewButton = document.getElementById('overviewMode');
  const expandAllButton = document.getElementById('expandAll');
  const collapseAllButton = document.getElementById('collapseAll');
  const backboneRail = document.getElementById('backboneRail');

  document.getElementById('nodeCount').textContent = `${nodes.length} nodes`;
  document.getElementById('graftCount').textContent = `${crossLinks.length} cross-links`;
  document.querySelector('.detail-index').textContent = String(nodes.length);
  expandAllButton.textContent = `Show all ${nodes.length}`;

  const descendantCount = new Map();
  function countDescendants(id) {
    if (descendantCount.has(id)) return descendantCount.get(id);
    const total = (actualChildren.get(id) || []).reduce((sum, child) => sum + 1 + countDescendants(child), 0);
    descendantCount.set(id, total);
    return total;
  }
  nodes.forEach(node => countDescendants(node.id));

  const overviewSet = new Set(nodes.filter(node => node.main || node.kind === 'program' || node.kind === 'milestone').map(node => node.id));
  [...overviewSet].forEach(id => {
    let parent = byId.get(id).parent;
    while (parent) {
      overviewSet.add(parent);
      parent = byId.get(parent).parent;
    }
  });

  const overviewChildren = new Map(nodes.map(node => [node.id, []]));
  const overviewParent = new Map();
  overviewSet.forEach(id => {
    let parent = byId.get(id).parent;
    while (parent && !overviewSet.has(parent)) parent = byId.get(parent).parent;
    overviewParent.set(id, parent || null);
    if (parent) overviewChildren.get(parent).push(id);
  });

  const backbone = nodes.filter(node => node.main);
  const expanded = new Set();
  let showAll = false;
  let selectedId = null;
  let searchTerm = '';
  let searchMatches = new Set();

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  }

  function incidentLinks(id) {
    return crossLinks.filter(link => link.source === id || link.target === id);
  }

  function ancestorIds(id) {
    const ids = [];
    let current = byId.get(id).parent;
    while (current) {
      ids.push(current);
      current = byId.get(current).parent;
    }
    return ids;
  }

  function subtreeIds(id) {
    const out = [];
    const stack = [id];
    while (stack.length) {
      const current = stack.pop();
      out.push(current);
      (actualChildren.get(current) || []).forEach(child => stack.push(child));
    }
    return out;
  }

  function displayChildren(id) {
    if (showAll || expanded.has(id)) {
      return (actualChildren.get(id) || []).map(child => ({id: child, compressed: false}));
    }
    return (overviewChildren.get(id) || []).map(child => ({
      id: child,
      compressed: byId.get(child).parent !== id
    }));
  }

  function searchVisible() {
    if (!searchTerm) return null;
    const visible = new Set();
    searchMatches = new Set();
    nodes.forEach(node => {
      const haystack = `${node.label} ${node.why} ${node.result} ${node.legacy} ${families[node.family] || ''}`.toLowerCase();
      if (haystack.includes(searchTerm)) {
        searchMatches.add(node.id);
        visible.add(node.id);
        ancestorIds(node.id).forEach(id => visible.add(id));
      }
    });
    return visible;
  }

  function rowHtml(node, level, compressed) {
    const childCount = descendantCount.get(node.id) || 0;
    const links = incidentLinks(node.id);
    const hasChildren = (actualChildren.get(node.id) || []).length > 0;
    const expandedHere = showAll || expanded.has(node.id);
    const status = node.status.toUpperCase();
    const isMatch = searchMatches.has(node.id);
    return `
      <div class="tree-row ${node.kind || 'experiment'} ${node.status} ${node.main ? 'main' : ''} ${selectedId === node.id ? 'selected' : ''} ${isMatch ? 'match' : ''} ${level === 0 ? 'root' : ''} ${compressed ? 'compressed-link' : ''}"
           id="row-${escapeHtml(node.id)}" data-id="${escapeHtml(node.id)}" role="treeitem" aria-level="${level + 1}" style="--level:${level}">
        <button class="expand-btn ${hasChildren ? '' : 'empty'}" type="button" data-expand="${escapeHtml(node.id)}" aria-label="${expandedHere ? 'Collapse' : 'Reveal'} ${escapeHtml(node.label)}">${expandedHere ? '−' : '+'}</button>
        <div class="node-title-wrap" data-select="${escapeHtml(node.id)}">
          <span class="node-kicker"><i class="status-dot" style="background:${statusColors[node.status] || '#777'}"></i>${escapeHtml(status)} · ${escapeHtml(families[node.family] || node.family)}</span>
          <strong class="node-title">${escapeHtml(node.label)}</strong>
        </div>
        <div class="node-summary" data-select="${escapeHtml(node.id)}" title="${escapeHtml(node.result)}">${escapeHtml(node.result)}</div>
        <div class="row-meta">
          ${childCount ? `<span class="count-badge">${childCount} below</span>` : ''}
          ${links.length ? `<span class="graft-badge">↗ ${links.length}</span>` : ''}
        </div>
      </div>`;
  }

  function renderSearchTree(visible) {
    if (!searchMatches.size) {
      treeReader.innerHTML = '<div class="tree-empty">No matching method, experiment or result.</div>';
      return;
    }
    const visibleChildren = new Map(nodes.map(node => [node.id, []]));
    visible.forEach(id => {
      const node = byId.get(id);
      if (node.parent && visible.has(node.parent)) visibleChildren.get(node.parent).push(id);
    });
    function walk(id, level) {
      let html = rowHtml(byId.get(id), level, false);
      visibleChildren.get(id).forEach(child => { html += walk(child, level + 1); });
      return html;
    }
    treeReader.innerHTML = walk(data.meta.root, 0);
  }

  function renderNormalTree() {
    function walk(id, level, compressed) {
      let html = rowHtml(byId.get(id), level, compressed);
      displayChildren(id).forEach(child => { html += walk(child.id, level + 1, child.compressed); });
      return html;
    }
    treeReader.innerHTML = walk(data.meta.root, 0, false);
  }

  function bindTreeEvents() {
    treeReader.querySelectorAll('[data-select]').forEach(element => {
      element.addEventListener('click', () => selectNode(element.dataset.select, false));
    });
    treeReader.querySelectorAll('[data-expand]').forEach(button => {
      button.addEventListener('click', event => {
        event.stopPropagation();
        toggleExpand(button.dataset.expand);
      });
    });
  }

  function renderTree() {
    searchTerm = searchInput.value.trim().toLowerCase();
    const visible = searchVisible();
    if (visible) renderSearchTree(visible); else renderNormalTree();
    bindTreeEvents();
    overviewButton.classList.toggle('active', !showAll && !searchTerm);
    expandAllButton.classList.toggle('active', showAll && !searchTerm);
    highlightBackbone();
  }

  function toggleExpand(id) {
    if (showAll) {
      showAll = false;
      expanded.clear();
      ancestorIds(id).forEach(parent => expanded.add(parent));
    }
    if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
    renderTree();
    selectNode(id, true, false);
  }

  function revealPath(id) {
    showAll = false;
    searchInput.value = '';
    searchTerm = '';
    ancestorIds(id).forEach(parent => expanded.add(parent));
    renderTree();
  }

  function scrollToRow(id) {
    const row = document.getElementById(`row-${id}`);
    if (row) row.scrollIntoView({behavior:'smooth', block:'center'});
  }

  function relationButton(id, label) {
    return `<button type="button" data-jump="${escapeHtml(id)}">${escapeHtml(label || byId.get(id).label)}</button>`;
  }

  function selectNode(id, scroll = true, rerender = true) {
    selectedId = id;
    const node = byId.get(id);
    const children = actualChildren.get(id) || [];
    const linksOut = crossLinks.filter(link => link.source === id);
    const linksIn = crossLinks.filter(link => link.target === id);
    detailPanel.innerHTML = `
      <span class="section-index">${escapeHtml(families[node.family] || node.family)}</span>
      <h3>${escapeHtml(node.label)}</h3>
      <div class="detail-meta">
        <span class="detail-chip">${escapeHtml(node.status)}</span>
        <span class="detail-chip">${escapeHtml(node.kind)}</span>
        ${node.main ? '<span class="detail-chip">backbone</span>' : ''}
        ${descendantCount.get(id) ? `<span class="detail-chip">${descendantCount.get(id)} descendants</span>` : ''}
      </div>
      <div class="detail-block"><b>Why we tried it</b><p>${escapeHtml(node.why)}</p></div>
      <div class="detail-block"><b>What happened</b><p>${escapeHtml(node.result)}</p></div>
      <div class="detail-block"><b>What survived</b><p>${escapeHtml(node.legacy)}</p></div>
      ${node.parent ? `<div class="detail-block"><b>Primary parent</b><div class="relations">${relationButton(node.parent)}</div></div>` : ''}
      ${children.length ? `<div class="detail-block"><b>Direct children</b><div class="relations">${children.map(child => relationButton(child)).join('')}</div></div>` : ''}
      ${(linksOut.length || linksIn.length) ? `<div class="detail-block"><b>Cross-branch inheritance</b><div class="relations">${linksOut.map(link => relationButton(link.target, `→ ${byId.get(link.target).label}: ${link.label}`)).join('')}${linksIn.map(link => relationButton(link.source, `← ${byId.get(link.source).label}: ${link.label}`)).join('')}</div></div>` : ''}
      ${children.length ? `<button class="detail-action" type="button" data-reveal="${escapeHtml(id)}">${expanded.has(id) ? 'Hide next experimental layer' : 'Reveal next experimental layer'}</button>` : ''}
      ${children.length && descendantCount.get(id) > children.length ? `<button class="detail-action secondary" type="button" data-branch="${escapeHtml(id)}">Open entire branch (${descendantCount.get(id)} nodes)</button>` : ''}
    `;
    detailPanel.querySelectorAll('[data-jump]').forEach(button => button.addEventListener('click', () => jumpToNode(button.dataset.jump)));
    const reveal = detailPanel.querySelector('[data-reveal]');
    if (reveal) reveal.addEventListener('click', () => toggleExpand(reveal.dataset.reveal));
    const branch = detailPanel.querySelector('[data-branch]');
    if (branch) branch.addEventListener('click', () => openWholeBranch(branch.dataset.branch));
    if (rerender) renderTree();
    if (scroll) requestAnimationFrame(() => scrollToRow(id));
  }

  function openWholeBranch(id) {
    showAll = false;
    searchInput.value = '';
    ancestorIds(id).forEach(parent => expanded.add(parent));
    subtreeIds(id).forEach(nodeId => {
      if ((actualChildren.get(nodeId) || []).length) expanded.add(nodeId);
    });
    renderTree();
    selectNode(id, true, false);
  }

  function jumpToNode(id) {
    if (overviewSet.has(id)) {
      showAll = false;
      expanded.clear();
      searchInput.value = '';
      searchTerm = '';
      renderTree();
    } else {
      revealPath(id);
    }
    selectNode(id, true, false);
  }

  function highlightBackbone() {
    backboneRail.querySelectorAll('.backbone-step').forEach(button => {
      button.classList.toggle('active', button.dataset.id === selectedId);
    });
  }

  function renderBackbone() {
    backboneRail.innerHTML = backbone.map((node, index) => `
      <button type="button" class="backbone-step" data-id="${escapeHtml(node.id)}">
        <small>${String(index + 1).padStart(2, '0')} · ${escapeHtml(node.status)}</small>
        <strong>${escapeHtml(node.label)}</strong>
      </button>`).join('');
    backboneRail.querySelectorAll('.backbone-step').forEach(button => {
      button.addEventListener('click', () => jumpToNode(button.dataset.id));
    });
  }

  searchInput.addEventListener('input', () => {
    showAll = false;
    renderTree();
  });

  overviewButton.addEventListener('click', () => {
    showAll = false;
    expanded.clear();
    searchInput.value = '';
    renderTree();
    treeReader.scrollTo({top:0, behavior:'smooth'});
  });

  expandAllButton.addEventListener('click', () => {
    showAll = true;
    expanded.clear();
    searchInput.value = '';
    renderTree();
    treeReader.scrollTo({top:0, behavior:'smooth'});
  });

  collapseAllButton.addEventListener('click', () => {
    showAll = false;
    expanded.clear();
    searchInput.value = '';
    renderTree();
    if (selectedId && overviewSet.has(selectedId)) scrollToRow(selectedId);
  });

  renderBackbone();
  renderTree();
})();
