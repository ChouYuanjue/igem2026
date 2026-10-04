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
    root:'#24251f',candidate:'#3f6f60',representation:'#506d83',training:'#637a8a',mechanism:'#a06c35',
    structure:'#745f7e',uncertainty:'#5b7480',graph:'#747b55',generalization:'#8c665e',routing:'#806d46',
    expert:'#4f6e80',evaluation:'#777970',fusion:'#825f4f',context:'#6f6a8c',fibre:'#765779',final:'#315f49',milestone:'#24251f'
  };
  const statusColors = {root:'#171914',keep:'#315f49',turn:'#9a6e2e',local:'#49697e',reject:'#a14d3e',historical:'#715878',considered:'#85867f'};

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
    while (current) {
      out.push(current);
      current = byId.get(current).parent;
    }
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

  function nearestVisibleParent(id, visible) {
    let current = byId.get(id).parent;
    while (current && !visible.has(current)) current = byId.get(current).parent;
    return current || null;
  }

  function incidentLinks(id) {
    return crossLinks.filter(link => link.source === id || link.target === id);
  }

  function graphSize() {
    const rect = stage.getBoundingClientRect();
    return {w: Math.max(720, rect.width), h: Math.max(620, rect.height)};
  }

  function overviewLayout(w, h) {
    const visible = new Set(overviewSet);
    const positions = new Map();
    const left = 72, right = 80;
    const yMid = h * 0.53;
    const usable = Math.max(500, w - left - right);

    mainBackbone.forEach((node, index) => {
      const t = mainBackbone.length === 1 ? 0 : index / (mainBackbone.length - 1);
      const x = left + t * usable;
      const wave = Math.sin(index * 0.86) * 20;
      const stagger = index % 2 === 0 ? -38 : 38;
      positions.set(node.id, {x, y: yMid + wave + stagger});
    });

    const satelliteLanes = {
      candidate_program: {dx:-58, lane:'topA'},
      tps_mech_program: {dx:82, lane:'topB'},
      evidence_program: {dx:-58, lane:'bottomA'},
      graph_program: {dx:92, lane:'bottomB'},
      generalization_program: {dx:-42, lane:'topA'},
      expert_program: {dx:115, lane:'topB'},
      stress_program: {dx:-18, lane:'bottomA'},
      portfolio: {dx:82, lane:'bottomB'},
      fibre: {dx:100, lane:'bottomA'}
    };
    const laneY = {topA:112, topB:218, bottomB:h-218, bottomA:h-112};
    [...visible].filter(id => !mainSet.has(id)).forEach(id => {
      const anchor = nearestMainAncestor(id);
      const a = positions.get(anchor) || {x:w/2,y:yMid};
      const spec = satelliteLanes[id] || {dx:0,lane:id.length % 2 ? 'topB' : 'bottomB'};
      positions.set(id, {
        x: Math.max(102, Math.min(w - 118, a.x + spec.dx)),
        y: laneY[spec.lane]
      });
    });

    return {visible, positions};
  }

  function focusLayout(focus, w, h) {
    const visible = new Set([focus]);
    const positions = new Map();
    const focusNode = byId.get(focus);
    const parent = focusNode.parent;
    if (parent) visible.add(parent);
    const kids = children.get(focus) || [];
    kids.forEach(id => visible.add(id));

    const fx = Math.max(300, Math.min(w * 0.35, 410));
    const fy = h * 0.53;
    positions.set(focus, {x:fx, y:fy});
    if (parent) positions.set(parent, {x:105, y:fy});

    const childCount = kids.length;
    const bowPoint = (index, total, baseX, bulge, top, bottom) => {
      const t = total <= 1 ? 0.5 : index / (total - 1);
      const yNorm = -1 + 2 * t;
      return {
        x: Math.min(w - 108, fx + baseX + bulge * (1 - yNorm * yNorm)),
        y: top + t * (bottom - top)
      };
    };
    if (childCount <= 8) {
      const baseX = Math.min(350, Math.max(245, w - fx - 175));
      kids.forEach((id, index) => positions.set(id, bowPoint(index, childCount, baseX, 62, 92, h - 86)));
    } else {
      const firstCount = Math.ceil(childCount / 2);
      const inner = kids.slice(0, firstCount);
      const outer = kids.slice(firstCount);
      const innerBase = Math.min(245, Math.max(205, (w - fx) * 0.29));
      const availableOuter = Math.max(innerBase + 150, w - fx - 125);
      const outerBase = Math.min(500, availableOuter);
      inner.forEach((id, index) => positions.set(id, bowPoint(index, inner.length, innerBase, 58, 92, h - 86)));
      outer.forEach((id, index) => positions.set(id, bowPoint(index, outer.length, outerBase, 34, 120, h - 114)));
    }
    return {visible, positions};
  }

  function smoothPath(points) {
    if (!points.length) return '';
    if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i += 1) {
      const a = points[i - 1], b = points[i];
      const mx = (a.x + b.x) / 2;
      d += ` C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`;
    }
    return d;
  }

  function curvePath(a, b) {
    const dx = Math.max(42, Math.abs(b.x - a.x) * 0.46);
    const dir = b.x >= a.x ? 1 : -1;
    return `M ${a.x} ${a.y} C ${a.x + dx * dir} ${a.y}, ${b.x - dx * dir} ${b.y}, ${b.x} ${b.y}`;
  }

  function familyColor(id) {
    const node = byId.get(id);
    return familyColors[node.family] || statusColors[node.status] || '#777';
  }

  function drawOverviewHalos(positions) {
    [...overviewSet].filter(id => !mainSet.has(id)).forEach(id => {
      const p = positions.get(id); if (!p) return;
      const count = descendantCount.get(id) || 0;
      const radius = Math.min(80, 42 + Math.sqrt(count) * 6);
      haloLayer.appendChild(svgEl('circle', {cx:p.x,cy:p.y,r:radius,fill:familyColor(id),class:'cluster-halo'}));
      haloLayer.appendChild(svgEl('circle', {cx:p.x,cy:p.y,r:radius * .82,stroke:familyColor(id),class:'cluster-ring'}));
    });
  }

  function drawEdges(visible, positions) {
    currentEdges = [];
    if (mode === 'overview') {
      const spinePoints = mainBackbone.map(node => positions.get(node.id)).filter(Boolean);
      edgeLayer.appendChild(svgEl('path', {d:smoothPath(spinePoints),class:'spine-edge graph-edge','data-role':'spine'}));
      [...visible].filter(id => !mainSet.has(id)).forEach(id => {
        const parent = nearestVisibleParent(id, visible);
        if (!parent) return;
        const path = svgEl('path', {d:curvePath(positions.get(parent),positions.get(id)),class:'branch-edge graph-edge',stroke:familyColor(id),'data-source':parent,'data-target':id});
        edgeLayer.appendChild(path); currentEdges.push({source:parent,target:id,el:path});
      });
      for (let i=1;i<mainBackbone.length;i+=1) currentEdges.push({source:mainBackbone[i-1].id,target:mainBackbone[i].id,el:null});
    } else {
      const focus = focusId;
      const focusPos = positions.get(focus);
      const parent = byId.get(focus).parent;
      if (parent && positions.has(parent)) {
        const path=svgEl('path',{d:curvePath(positions.get(parent),focusPos),class:'parent-edge graph-edge','data-source':parent,'data-target':focus});
        edgeLayer.appendChild(path);currentEdges.push({source:parent,target:focus,el:path});
      }
      (children.get(focus)||[]).forEach(id=>{
        const path=svgEl('path',{d:curvePath(focusPos,positions.get(id)),class:'focus-edge graph-edge',stroke:familyColor(id),'data-source':focus,'data-target':id});
        edgeLayer.appendChild(path);currentEdges.push({source:focus,target:id,el:path});
      });
      const r=Math.min(190,85+Math.sqrt(descendantCount.get(focus)||0)*11);
      haloLayer.appendChild(svgEl('circle',{cx:focusPos.x,cy:focusPos.y,r,fill:familyColor(focus),class:'cluster-halo'}));
      haloLayer.appendChild(svgEl('circle',{cx:focusPos.x,cy:focusPos.y,r:r*.83,stroke:familyColor(focus),class:'cluster-ring'}));
    }
  }

  function drawInheritance(visible, positions) {
    if (!graftsOn) return;
    crossLinks.forEach(link => {
      if (!visible.has(link.source) || !visible.has(link.target)) return;
      const path=svgEl('path',{d:curvePath(positions.get(link.source),positions.get(link.target)),class:'inherit-edge graph-edge','data-source':link.source,'data-target':link.target});
      inheritLayer.appendChild(path);currentEdges.push({source:link.source,target:link.target,el:path,inherit:true});
    });
  }

  const overviewLabels = {
    enzymecage:'EnzymeCAGE', closed_pool:'Closed pool', open_problem:'Open-world retrieval',
    representation_program:'Broad representation', dual_tower:'Dual tower', broad:'Broad Retrieval',
    fusion_program:'Expert fusion', r2e_lambdarank:'R2E LambdaRank', bime:'BiME-Rank',
    return_broad:'Return to Broad', dynamic_v4:'Dynamic router', dynamic_v6:'Permission levels',
    query_applicability:'Query applicability', integrated_specialists:'Gated specialists', bridge:'BRIDGE'
  };

  function nodeClass(node, isCluster) {
    const parts=['graph-node',node.status,node.kind||'experiment'];
    if(isCluster)parts.push('cluster');
    if(mode==='overview'&&mainSet.has(node.id))parts.push('spine-node');
    if(mode==='overview'&&mainSet.has(node.id)&&node.kind!=='milestone')parts.push('waypoint');
    if(node.id==='bridge')parts.push('bridge');
    if(node.id==='fibre'||node.family==='fibre')parts.push('fibre');
    if(selectedId===node.id)parts.push('selected');
    return parts.join(' ');
  }

  function renderNodes(visible, positions) {
    [...visible].forEach(id => {
      const node=byId.get(id);const p=positions.get(id);if(!p)return;
      const cluster = (mode==='overview' && !mainSet.has(id)) || (mode==='focus' && id===focusId && (children.get(id)||[]).length>0);
      const hidden = descendantCount.get(id)||0;
      const div=document.createElement('button');
      div.type='button';div.className=nodeClass(node,cluster);div.dataset.id=id;
      div.style.left=`${p.x}px`;div.style.top=`${p.y}px`;div.style.setProperty('--node-color',statusColors[node.status]||familyColor(id));
      const kicker = node.main ? 'design spine' : (families[node.family]||node.family);
      const displayLabel = mode==='overview' && overviewLabels[node.id] ? overviewLabels[node.id] : node.label;
      const showCount = hidden && !(mode==='overview' && mainSet.has(id));
      div.innerHTML=`<span class="graph-node-card"><span class="node-kicker">${esc(kicker)}</span><strong class="node-label">${esc(displayLabel)}</strong>${showCount?`<span class="node-count">${hidden} descendant${hidden===1?'':'s'}</span>`:''}</span>`;
      div.title=node.label;
      div.addEventListener('mouseenter',()=>{hoverId=id;applyHighlight();});
      div.addEventListener('mouseleave',()=>{hoverId=null;applyHighlight();});
      div.addEventListener('click',()=>handleNodeClick(id));
      nodeLayer.appendChild(div);
    });
  }

  function neighborhood(id) {
    const set=new Set([id]);
    const node=byId.get(id);
    if(node.parent&&currentVisible.has(node.parent))set.add(node.parent);
    (children.get(id)||[]).forEach(child=>{if(currentVisible.has(child))set.add(child);});
    currentEdges.forEach(edge=>{if(edge.source===id)set.add(edge.target);if(edge.target===id)set.add(edge.source);});
    return set;
  }

  function applyHighlight() {
    const target=hoverId||selectedId;
    const related=target?neighborhood(target):null;
    nodeLayer.querySelectorAll('.graph-node').forEach(el=>{
      el.classList.toggle('dim',Boolean(related)&&!related.has(el.dataset.id));
      el.classList.toggle('selected',el.dataset.id===selectedId);
    });
    [...edgeLayer.querySelectorAll('.graph-edge'),...inheritLayer.querySelectorAll('.graph-edge')].forEach(el=>{
      if(!related){el.classList.remove('dim');el.classList.remove('related');return;}
      const s=el.dataset.source,t=el.dataset.target;
      const active=!s||!t||(related.has(s)&&related.has(t));
      el.classList.toggle('dim',!active);
      el.classList.toggle('related',active&&el.classList.contains('inherit-edge'));
    });
  }

  function contextForFocus() {
    if(mode==='overview'){contextStrip.innerHTML='<span class="context-crumb current">EnzymeCAGE → BRIDGE · overview</span>';return;}
    const path=ancestors(focusId);
    const compact=path.length>6?[path[0],...path.slice(-5)]:path;
    contextStrip.innerHTML=compact.map((id,index)=>`<span class="context-crumb ${index===compact.length-1?'current':''}">${esc(byId.get(id).label)}</span>`).join('');
  }

  function updateHeading() {
    if(mode==='overview'){
      viewTitle.textContent='The whole research landscape';
      viewSubtitle.textContent='Follow the dark spine for the main story. Open any constellation to enter its local experiment graph.';
      backButton.disabled=true;overviewButton.classList.add('active');
    } else {
      const node=byId.get(focusId);
      viewTitle.textContent=node.label;
      viewSubtitle.textContent=`${(children.get(focusId)||[]).length} direct branches · ${descendantCount.get(focusId)||0} total descendants. Select a child to inspect it; open it to go one level deeper.`;
      backButton.disabled=false;overviewButton.classList.remove('active');
    }
    graftButton.classList.toggle('active',graftsOn);
    graftButton.textContent=graftsOn?'Inheritance on':'Inheritance off';
  }

  function render() {
    const {w,h}=graphSize();
    stage.classList.toggle('compact-graph', mode==='overview' && w < 1000);
    stage.classList.toggle('compact-focus', mode==='focus' && w < 1000);
    svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
    haloLayer.innerHTML='';edgeLayer.innerHTML='';inheritLayer.innerHTML='';nodeLayer.innerHTML='';
    const layout=mode==='overview'?overviewLayout(w,h):focusLayout(focusId,w,h);
    currentVisible=layout.visible;currentPositions=layout.positions;
    if(mode==='overview')drawOverviewHalos(layout.positions);
    drawEdges(layout.visible,layout.positions);drawInheritance(layout.visible,layout.positions);renderNodes(layout.visible,layout.positions);
    contextForFocus();updateHeading();applyHighlight();
  }

  function showDetail(id) {
    selectedId=id;
    const node=byId.get(id);const kids=children.get(id)||[];const linksOut=crossLinks.filter(link=>link.source===id);const linksIn=crossLinks.filter(link=>link.target===id);
    const relationButton=(target,label)=>`<button type="button" data-jump="${esc(target)}">${esc(label||byId.get(target).label)}</button>`;
    detailPanel.innerHTML=`
      <span class="section-index">${esc(families[node.family]||node.family)}</span>
      <h3>${esc(node.label)}</h3>
      <div class="detail-meta"><span class="detail-chip">${esc(node.status)}</span><span class="detail-chip">${esc(node.kind)}</span>${node.main?'<span class="detail-chip">design spine</span>':''}${descendantCount.get(id)?`<span class="detail-chip">${descendantCount.get(id)} descendants</span>`:''}</div>
      <div class="detail-block"><b>Why we tried it</b><p>${esc(node.why)}</p></div>
      <div class="detail-block"><b>What happened</b><p>${esc(node.result)}</p></div>
      <div class="detail-block"><b>What survived</b><p>${esc(node.legacy)}</p></div>
      ${node.parent?`<div class="detail-block"><b>Primary parent</b><div class="relations">${relationButton(node.parent)}</div></div>`:''}
      ${kids.length?`<div class="detail-block"><b>Direct children</b><div class="relations">${kids.slice(0,10).map(child=>relationButton(child)).join('')}${kids.length>10?`<span class="detail-chip">+${kids.length-10} more</span>`:''}</div></div>`:''}
      ${(linksOut.length||linksIn.length)?`<div class="detail-block"><b>Cross-branch inheritance</b><div class="relations">${linksOut.map(link=>relationButton(link.target,`→ ${byId.get(link.target).label}`)).join('')}${linksIn.map(link=>relationButton(link.source,`← ${byId.get(link.source).label}`)).join('')}</div></div>`:''}
      ${kids.length?`<button class="detail-action" type="button" data-open="${esc(id)}">Open this constellation</button>`:''}
      ${mode==='focus'&&node.parent?`<button class="detail-action secondary" type="button" data-parent="${esc(node.parent)}">Move to parent constellation</button>`:''}`;
    detailPanel.querySelectorAll('[data-jump]').forEach(btn=>btn.addEventListener('click',()=>locateNode(btn.dataset.jump)));
    const open=detailPanel.querySelector('[data-open]');if(open)open.addEventListener('click',()=>enterFocus(open.dataset.open));
    const parent=detailPanel.querySelector('[data-parent]');if(parent)parent.addEventListener('click',()=>enterFocus(parent.dataset.parent));
    applyHighlight();
  }

  function handleNodeClick(id) {
    const node=byId.get(id);const hasChildren=(children.get(id)||[]).length>0;
    if(mode==='overview'&&!mainSet.has(id)&&hasChildren){showDetail(id);enterFocus(id);return;}
    if(mode==='focus'&&id!==focusId&&id!==node.parent&&hasChildren){showDetail(id);enterFocus(id);return;}
    showDetail(id);
  }

  function enterFocus(id, push=true) {
    if(!(children.get(id)||[]).length){showDetail(id);return;}
    if(push)history.push(mode==='overview'?null:focusId);
    mode='focus';focusId=id;selectedId=id;hoverId=null;render();showDetail(id);
  }

  function goOverview() {
    mode='overview';focusId=null;selectedId=null;hoverId=null;history=[];render();
    detailPanel.innerHTML=`<div class="detail-empty"><span class="detail-index">${nodes.length}</span><h3>One lineage, several scales.</h3><p>The overview aggregates ${nodes.length} experiments into readable research constellations. Every experiment remains one click away inside its branch.</p><div class="detail-rule"></div><p class="small">Select a node for motivation, result, surviving idea, descendants and cross-branch inheritance.</p></div>`;
  }

  function goBack() {
    if(mode==='overview')return;
    const previous=history.pop();
    if(previous===null||previous===undefined){goOverview();return;}
    mode='focus';focusId=previous;selectedId=previous;render();showDetail(previous);
  }

  function locateNode(id) {
    const node=byId.get(id);
    if(overviewSet.has(id)){goOverview();selectedId=id;render();showDetail(id);return;}
    const parent=node.parent;
    if(parent){mode='focus';focusId=parent;selectedId=id;history=[];render();showDetail(id);}else{goOverview();showDetail(id);}
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
