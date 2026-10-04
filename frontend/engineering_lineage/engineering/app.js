(() => {
  const data = window.LINEAGE_DATA;
  if (!data) throw new Error('LINEAGE_DATA is not loaded');

  const nodes = data.nodes;
  const crossLinks = data.crossLinks;
  const families = data.families;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map(nodes.map(node => [node.id, []]));
  nodes.forEach(node => { if (node.parent) children.get(node.parent).push(node.id); });

  const familyColors = {
    root:'#252720', candidate:'#3f6f60', representation:'#506d83', training:'#637a8a', mechanism:'#a06c35',
    structure:'#745f7e', uncertainty:'#5b7480', graph:'#747b55', generalization:'#8c665e', routing:'#806d46',
    expert:'#4f6e80', evaluation:'#777970', fusion:'#825f4f', context:'#6f6a8c', fibre:'#765779', final:'#315f49', milestone:'#252720'
  };
  const statusColors = {root:'#252720', keep:'#315f49', turn:'#a1712c', local:'#4e6b7c', reject:'#a14d3e', historical:'#73587b', considered:'#85867f'};

  const stage = document.getElementById('graphStage');
  const svg = document.getElementById('graphEdges');
  const haloLayer = document.getElementById('haloLayer');
  const edgeLayer = document.getElementById('edgeLayer');
  const inheritLayer = document.getElementById('inheritLayer');
  const nodeLayer = document.getElementById('graphNodes');
  const detailPanel = document.getElementById('detailPanel');
  const contextStrip = document.getElementById('contextStrip');
  const viewTitle = document.getElementById('viewTitle');
  const viewSubtitle = document.getElementById('viewSubtitle');
  const searchInput = document.getElementById('graphSearch');
  const searchResults = document.getElementById('searchResults');
  const backButton = document.getElementById('backButton');
  const overviewButton = document.getElementById('overviewButton');
  const graftButton = document.getElementById('graftButton');

  document.getElementById('nodeCount').textContent = `${nodes.length} nodes`;
  document.getElementById('graftCount').textContent = `${crossLinks.length} inherited links`;
  document.querySelector('.detail-index').textContent = String(nodes.length);

  const descendantCount = new Map();
  function countDescendants(id) {
    if (descendantCount.has(id)) return descendantCount.get(id);
    const total = (children.get(id) || []).reduce((sum, child) => sum + 1 + countDescendants(child), 0);
    descendantCount.set(id, total);
    return total;
  }
  nodes.forEach(node => countDescendants(node.id));

  const overviewSet = new Set(nodes.filter(node => node.main || node.kind === 'program' || node.kind === 'milestone').map(node => node.id));
  [...overviewSet].forEach(id => {
    let parent = byId.get(id).parent;
    while (parent) { overviewSet.add(parent); parent = byId.get(parent).parent; }
  });
  const mainBackbone = nodes.filter(node => node.main);
  const mainSet = new Set(mainBackbone.map(node => node.id));
  const mainIndex = new Map(mainBackbone.map((node, index) => [node.id, index]));

  const overviewLabels = {
    enzymecage:'EnzymeCAGE', closed_pool:'Closed candidate pool', open_problem:'Open-world retrieval',
    representation_program:'Broad representation', dual_tower:'Dual tower', broad:'Broad Retrieval',
    fusion_program:'Expert fusion', r2e_lambdarank:'R2E LambdaRank', bime:'BiME-Rank',
    return_broad:'Return to Broad', dynamic_v4:'Dynamic router', dynamic_v6:'Permission levels',
    query_applicability:'Query applicability', integrated_specialists:'Gated specialists', bridge:'BRIDGE'
  };

  const constellationPlacement = {
    candidate_program:{side:'left',dy:-88}, tps_mech_program:{side:'right',dy:-20},
    evidence_program:{side:'left',dy:72}, graph_program:{side:'right',dy:132},
    generalization_program:{side:'left',dy:-84}, expert_program:{side:'right',dy:-18},
    stress_program:{side:'left',dy:82}, portfolio:{side:'right',dy:86}, fibre:{side:'right',dy:86}
  };

  let mode = 'overview';
  let focusId = null;
  let selectedId = null;
  let hoverId = null;
  let graftsOn = true;
  let history = [];
  let currentVisible = new Set();
  let currentPositions = new Map();
  let currentEdges = [];
  let resizeTimer = null;

  function esc(value) {
    return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  }
  function svgEl(name, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    return el;
  }
  function ancestors(id) {
    const out = [];
    let current = id;
    while (current) { out.push(current); current = byId.get(current).parent; }
    return out.reverse();
  }
  function nearestMainAncestor(id) {
    let current = byId.get(id).parent;
    while (current) {
      if (mainSet.has(current)) return current;
      current = byId.get(current).parent;
    }
    return mainBackbone[0].id;
  }
  function incidentLinks(id) {
    return crossLinks.filter(link => link.source === id || link.target === id);
  }
  function familyColor(id) {
    const node = byId.get(id);
    return familyColors[node.family] || statusColors[node.status] || '#777';
  }
  function graphWidth() {
    return Math.max(320, stage.getBoundingClientRect().width);
  }

  function overviewLayout(w) {
    const mobile = w < 720;
    const visible = new Set(overviewSet);
    const positions = new Map();
    const centerX = w / 2;
    const top = mobile ? 120 : 130;
    const gap = mobile ? 148 : 124;
    const height = top * 2 + gap * (mainBackbone.length - 1) + 120;

    mainBackbone.forEach((node, index) => {
      const drift = mobile ? 0 : Math.sin(index * .9) * 5;
      positions.set(node.id, {x:centerX + drift, y:top + index * gap});
    });

    [...visible].filter(id => !mainSet.has(id)).forEach(id => {
      const anchor = nearestMainAncestor(id);
      const a = positions.get(anchor);
      const spec = constellationPlacement[id] || {side:id.length % 2 ? 'left' : 'right', dy:0};
      const branchDistance = mobile ? Math.min(132, w * .335) : Math.min(320, Math.max(220, w * .275));
      const x = centerX + (spec.side === 'left' ? -branchDistance : branchDistance);
      positions.set(id, {x, y:a.y + spec.dy});
    });
    return {visible, positions, height, mobile};
  }

  function focusLayout(focus, w) {
    const mobile = w < 720;
    const visible = new Set([focus]);
    const positions = new Map();
    const parent = byId.get(focus).parent;
    if (parent) visible.add(parent);
    const kids = children.get(focus) || [];
    kids.forEach(id => visible.add(id));

    const centerX = w / 2;
    const parentY = 96;
    const focusY = mobile ? 235 : 240;
    const startY = mobile ? 430 : 455;
    const rowGap = mobile ? 118 : 126;
    const rows = Math.max(1, Math.ceil(kids.length / 2));
    const height = startY + rows * rowGap + 110;
    const branchDistance = mobile ? Math.min(118, w * .31) : Math.min(310, Math.max(225, w * .26));

    if (parent) positions.set(parent, {x:centerX, y:parentY});
    positions.set(focus, {x:centerX, y:focusY});
    kids.forEach((id, index) => {
      const row = Math.floor(index / 2);
      const side = index % 2 === 0 ? -1 : 1;
      const singleLast = kids.length % 2 === 1 && index === kids.length - 1;
      positions.set(id, {
        x: singleLast ? centerX : centerX + side * branchDistance,
        y: startY + row * rowGap
      });
    });
    return {visible, positions, height, mobile};
  }

  function verticalSpinePath(points) {
    if (!points.length) return '';
    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i += 1) {
      const a = points[i - 1], b = points[i];
      const my = (a.y + b.y) / 2;
      d += ` C ${a.x} ${my}, ${b.x} ${my}, ${b.x} ${b.y}`;
    }
    return d;
  }
  function verticalCurve(a, b) {
    const my = (a.y + b.y) / 2;
    return `M ${a.x} ${a.y} C ${a.x} ${my}, ${b.x} ${my}, ${b.x} ${b.y}`;
  }
  function branchCurve(a, b) {
    const dx = b.x - a.x;
    const controlY = a.y + (b.y - a.y) * .45;
    return `M ${a.x} ${a.y} C ${a.x + dx * .18} ${a.y}, ${b.x - dx * .22} ${controlY}, ${b.x} ${b.y}`;
  }

  function drawOverviewHalos(positions) {
    [...overviewSet].filter(id => !mainSet.has(id)).forEach(id => {
      const p = positions.get(id); if (!p) return;
      const count = descendantCount.get(id) || 0;
      const rx = Math.min(95, 50 + Math.sqrt(count) * 5.5);
      haloLayer.appendChild(svgEl('ellipse', {cx:p.x,cy:p.y,rx,ry:34,fill:familyColor(id),class:'cluster-halo'}));
    });
  }

  function drawEdges(visible, positions) {
    currentEdges = [];
    if (mode === 'overview') {
      const spinePoints = mainBackbone.map(node => positions.get(node.id)).filter(Boolean);
      edgeLayer.appendChild(svgEl('path', {d:verticalSpinePath(spinePoints),class:'spine-edge graph-edge','data-role':'spine'}));
      [...visible].filter(id => !mainSet.has(id)).forEach(id => {
        const anchor = nearestMainAncestor(id);
        const path = svgEl('path', {d:branchCurve(positions.get(anchor),positions.get(id)),class:'branch-edge graph-edge',stroke:familyColor(id),'data-source':anchor,'data-target':id});
        edgeLayer.appendChild(path);
        currentEdges.push({source:anchor,target:id,el:path});
      });
      for (let i=1;i<mainBackbone.length;i+=1) currentEdges.push({source:mainBackbone[i-1].id,target:mainBackbone[i].id,el:null});
    } else {
      const focus = focusId;
      const parent = byId.get(focus).parent;
      if (parent && positions.has(parent)) {
        const path = svgEl('path',{d:verticalCurve(positions.get(parent),positions.get(focus)),class:'parent-edge graph-edge','data-source':parent,'data-target':focus});
        edgeLayer.appendChild(path); currentEdges.push({source:parent,target:focus,el:path});
      }
      (children.get(focus)||[]).forEach(id => {
        const path = svgEl('path',{d:branchCurve(positions.get(focus),positions.get(id)),class:'focus-edge graph-edge',stroke:familyColor(id),'data-source':focus,'data-target':id});
        edgeLayer.appendChild(path); currentEdges.push({source:focus,target:id,el:path});
      });
    }
  }

  function drawInheritance(visible, positions) {
    if (!graftsOn) return;
    crossLinks.forEach(link => {
      if (!visible.has(link.source) || !visible.has(link.target)) return;
      const path = svgEl('path',{d:branchCurve(positions.get(link.source),positions.get(link.target)),class:'inherit-edge graph-edge','data-source':link.source,'data-target':link.target});
      inheritLayer.appendChild(path); currentEdges.push({source:link.source,target:link.target,el:path,inherit:true});
    });
  }

  function visualType(node, isCluster) {
    if (mode === 'overview' && mainSet.has(node.id)) return node.kind === 'milestone' ? 'milestone' : 'waypoint';
    if (mode === 'focus' && node.id === focusId) return 'focus';
    if (isCluster) return 'cluster';
    if (mode === 'focus' && node.id === byId.get(focusId).parent) return 'parent';
    return 'experiment';
  }

  function nodeMarkup(node, type) {
    const color = statusColors[node.status] || familyColor(node.id);
    const hidden = descendantCount.get(node.id) || 0;
    const label = mode === 'overview' && overviewLabels[node.id] ? overviewLabels[node.id] : node.label;
    if (type === 'milestone' || type === 'focus') {
      const number = type === 'milestone' ? String((mainIndex.get(node.id) || 0) + 1).padStart(2,'0') : '•';
      return `<span class="node-symbol" style="--node-color:${color}"><b>${number}</b></span><span class="node-copy"><span class="node-kicker">${type === 'focus' ? 'focused branch' : 'design milestone'}</span><strong class="node-label">${esc(label)}</strong>${hidden?`<span class="node-count">${hidden} descendants</span>`:''}</span>`;
    }
    if (type === 'waypoint') {
      return `<span class="waypoint-dot" style="--node-color:${color}"></span><span class="waypoint-label">${esc(label)}</span>`;
    }
    if (type === 'cluster') {
      return `<span class="cluster-card" style="--node-color:${color}"><span class="cluster-rule"></span><span class="node-kicker">${esc(families[node.family]||node.family)}</span><strong class="node-label">${esc(node.label)}</strong><span class="node-count">${hidden} nodes inside · open ↗</span></span>`;
    }
    const childHint = hidden ? `<span class="experiment-more">${hidden} ↘</span>` : '';
    return `<span class="experiment-card" style="--node-color:${color}"><span class="experiment-dot"></span><span class="experiment-copy"><span class="node-kicker">${esc(node.status)}</span><strong class="node-label">${esc(node.label)}</strong></span>${childHint}</span>`;
  }

  function renderNodes(visible, positions) {
    [...visible].forEach(id => {
      const node = byId.get(id), p = positions.get(id); if (!p) return;
      const isCluster = mode === 'overview' && !mainSet.has(id);
      const type = visualType(node, isCluster);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `graph-node type-${type} status-${node.status}${node.id==='bridge'?' bridge':''}${node.id==='fibre'?' fibre':''}${selectedId===id?' selected':''}`;
      button.dataset.id = id;
      button.style.left = `${p.x}px`;
      button.style.top = `${p.y}px`;
      button.innerHTML = nodeMarkup(node, type);
      button.title = node.label;
      button.addEventListener('mouseenter',()=>{hoverId=id;applyHighlight();});
      button.addEventListener('mouseleave',()=>{hoverId=null;applyHighlight();});
      button.addEventListener('click',()=>handleNodeClick(id));
      nodeLayer.appendChild(button);
    });
  }

  function neighborhood(id) {
    const set = new Set([id]);
    const node = byId.get(id);
    if (node.parent && currentVisible.has(node.parent)) set.add(node.parent);
    (children.get(id)||[]).forEach(child=>{if(currentVisible.has(child))set.add(child);});
    currentEdges.forEach(edge=>{if(edge.source===id)set.add(edge.target);if(edge.target===id)set.add(edge.source);});
    return set;
  }
  function applyHighlight() {
    const target = hoverId || selectedId;
    const related = target ? neighborhood(target) : null;
    nodeLayer.querySelectorAll('.graph-node').forEach(el=>{
      el.classList.toggle('dim',Boolean(related)&&!related.has(el.dataset.id));
      el.classList.toggle('selected',el.dataset.id===selectedId);
    });
    [...edgeLayer.querySelectorAll('.graph-edge'),...inheritLayer.querySelectorAll('.graph-edge')].forEach(el=>{
      if (!related) { el.classList.remove('dim','related'); return; }
      const s=el.dataset.source,t=el.dataset.target;
      const active=!s||!t||(related.has(s)&&related.has(t));
      el.classList.toggle('dim',!active);
      el.classList.toggle('related',active&&el.classList.contains('inherit-edge'));
    });
  }

  function contextForFocus() {
    if (mode === 'overview') { contextStrip.innerHTML='<span class="context-crumb current">EnzymeCAGE ↓ BRIDGE · vertical overview</span>'; return; }
    const path = ancestors(focusId);
    const compact = path.length > 5 ? [path[0],...path.slice(-4)] : path;
    contextStrip.innerHTML = compact.map((id,index)=>`<span class="context-crumb ${index===compact.length-1?'current':''}">${esc(byId.get(id).label)}</span>`).join('');
  }
  function updateHeading() {
    if (mode === 'overview') {
      viewTitle.textContent='The engineering tree grows downward';
      viewSubtitle.textContent='Read from EnzymeCAGE at the top to BRIDGE at the bottom. Parallel research programs grow from the spine on either side.';
      backButton.disabled=true; overviewButton.classList.add('active');
    } else {
      const node=byId.get(focusId);
      viewTitle.textContent=node.label;
      viewSubtitle.textContent=`${(children.get(focusId)||[]).length} direct branches · ${descendantCount.get(focusId)||0} total descendants. Follow the vertical branch downward.`;
      backButton.disabled=false; overviewButton.classList.remove('active');
    }
    graftButton.classList.toggle('active',graftsOn);
    graftButton.textContent=graftsOn?'Inheritance on':'Inheritance off';
  }

  function render() {
    const w = graphWidth();
    const layout = mode==='overview' ? overviewLayout(w) : focusLayout(focusId,w);
    stage.style.height = `${layout.height}px`;
    stage.classList.toggle('mobile-graph', layout.mobile);
    svg.setAttribute('viewBox',`0 0 ${w} ${layout.height}`);
    haloLayer.innerHTML=''; edgeLayer.innerHTML=''; inheritLayer.innerHTML=''; nodeLayer.innerHTML='';
    currentVisible=layout.visible; currentPositions=layout.positions;
    if (mode==='overview') drawOverviewHalos(layout.positions);
    drawEdges(layout.visible,layout.positions); drawInheritance(layout.visible,layout.positions); renderNodes(layout.visible,layout.positions);
    contextForFocus(); updateHeading(); applyHighlight();
  }

  function showDetail(id) {
    selectedId=id;
    detailPanel.classList.add('has-selection');
    const node=byId.get(id), kids=children.get(id)||[];
    const linksOut=crossLinks.filter(link=>link.source===id), linksIn=crossLinks.filter(link=>link.target===id);
    const relationButton=(target,label)=>`<button type="button" data-jump="${esc(target)}">${esc(label||byId.get(target).label)}</button>`;
    detailPanel.innerHTML=`<button type="button" class="detail-close" aria-label="Close details">×</button><span class="section-index">${esc(families[node.family]||node.family)}</span><h3>${esc(node.label)}</h3>
      <div class="detail-meta"><span class="detail-chip">${esc(node.status)}</span><span class="detail-chip">${esc(node.kind)}</span>${node.main?'<span class="detail-chip">design spine</span>':''}${descendantCount.get(id)?`<span class="detail-chip">${descendantCount.get(id)} descendants</span>`:''}</div>
      <div class="detail-block"><b>Why we tried it</b><p>${esc(node.why)}</p></div><div class="detail-block"><b>What happened</b><p>${esc(node.result)}</p></div><div class="detail-block"><b>What survived</b><p>${esc(node.legacy)}</p></div>
      ${node.parent?`<div class="detail-block"><b>Primary parent</b><div class="relations">${relationButton(node.parent)}</div></div>`:''}
      ${kids.length?`<div class="detail-block"><b>Direct children</b><div class="relations">${kids.slice(0,10).map(child=>relationButton(child)).join('')}${kids.length>10?`<span class="detail-chip">+${kids.length-10} more</span>`:''}</div></div>`:''}
      ${(linksOut.length||linksIn.length)?`<div class="detail-block"><b>Cross-branch inheritance</b><div class="relations">${linksOut.map(link=>relationButton(link.target,`→ ${byId.get(link.target).label}`)).join('')}${linksIn.map(link=>relationButton(link.source,`← ${byId.get(link.source).label}`)).join('')}</div></div>`:''}
      ${kids.length?`<button class="detail-action" type="button" data-open="${esc(id)}">Open this branch</button>`:''}`;
    const close=detailPanel.querySelector('.detail-close'); if(close) close.addEventListener('click',()=>detailPanel.classList.remove('has-selection'));
    detailPanel.querySelectorAll('[data-jump]').forEach(btn=>btn.addEventListener('click',()=>locateNode(btn.dataset.jump)));
    const open=detailPanel.querySelector('[data-open]'); if(open)open.addEventListener('click',()=>enterFocus(open.dataset.open));
    applyHighlight();
  }
  function handleNodeClick(id) {
    const hasChildren=(children.get(id)||[]).length>0;
    if (mode==='overview'&&!mainSet.has(id)&&hasChildren) { enterFocus(id); return; }
    if (mode==='focus'&&id!==focusId&&hasChildren&&id!==byId.get(focusId).parent) { enterFocus(id); return; }
    showDetail(id);
  }
  function enterFocus(id,push=true) {
    if (!(children.get(id)||[]).length) { showDetail(id); return; }
    if (push) history.push(mode==='overview'?null:focusId);
    mode='focus'; focusId=id; selectedId=id; hoverId=null; render(); showDetail(id);
    document.getElementById('atlas').scrollIntoView({behavior:'smooth',block:'start'});
  }
  function goOverview() {
    mode='overview'; focusId=null; selectedId=null; hoverId=null; history=[]; detailPanel.classList.remove('has-selection'); render();
    detailPanel.innerHTML=`<div class="detail-empty"><span class="detail-index">${nodes.length}</span><h3>One vertical lineage.</h3><p>The spine reads from EnzymeCAGE at the top to BRIDGE at the bottom. Parallel programs branch left and right; exact experiments appear when you open a branch.</p><div class="detail-rule"></div><p class="small">Select a node for motivation, result and inheritance.</p></div>`;
  }
  function goBack() {
    if(mode==='overview')return;
    const previous=history.pop();
    if(previous===null||previous===undefined){goOverview();return;}
    mode='focus';focusId=previous;selectedId=previous;render();showDetail(previous);
  }
  function locateNode(id) {
    const node=byId.get(id);
    if (overviewSet.has(id)) { goOverview(); selectedId=id; render(); showDetail(id); return; }
    if (node.parent) { mode='focus'; focusId=node.parent; selectedId=id; history=[]; render(); showDetail(id); }
    else { goOverview(); showDetail(id); }
  }
  function updateSearch() {
    const query=searchInput.value.trim().toLowerCase();
    if(!query){searchResults.hidden=true;return;}
    const matches=nodes.filter(node=>`${node.label} ${node.why} ${node.result} ${node.legacy} ${families[node.family]||''}`.toLowerCase().includes(query)).slice(0,12);
    searchResults.innerHTML=matches.map(node=>`<button type="button" data-result="${esc(node.id)}"><strong>${esc(node.label)}</strong><small>${esc(families[node.family]||node.family)} · ${esc(node.status)} · ${descendantCount.get(node.id)||0} descendants</small></button>`).join('');
    searchResults.hidden=!matches.length;
    searchResults.querySelectorAll('[data-result]').forEach(btn=>btn.addEventListener('click',()=>{searchResults.hidden=true;searchInput.value='';locateNode(btn.dataset.result);}));
  }

  backButton.addEventListener('click',goBack);
  overviewButton.addEventListener('click',goOverview);
  graftButton.addEventListener('click',()=>{graftsOn=!graftsOn;render();});
  searchInput.addEventListener('input',updateSearch);
  searchInput.addEventListener('keydown',event=>{if(event.key==='Enter'){const first=searchResults.querySelector('[data-result]');if(first){event.preventDefault();first.click();}}if(event.key==='Escape')searchResults.hidden=true;});
  document.addEventListener('click',event=>{if(!event.target.closest('.search-block'))searchResults.hidden=true;});

  const observer=new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,80);});
  observer.observe(stage);
  render();
})();
