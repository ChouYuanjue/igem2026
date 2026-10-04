(() => {
  const data = window.LINEAGE_DATA;
  if (!data) throw new Error('LINEAGE_DATA is not loaded');

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const nodes = data.nodes;
  const crossLinks = data.crossLinks;
  const families = data.families;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map(nodes.map(node => [node.id, []]));
  nodes.forEach(node => { if (node.parent) children.get(node.parent).push(node.id); });

  const stage = document.getElementById('graphStage');
  const svg = document.getElementById('graphSvg');
  const scene = document.getElementById('scene');
  const primaryEdges = document.getElementById('primaryEdges');
  const crossEdges = document.getElementById('crossEdges');
  const transitionLabels = document.getElementById('transitionLabels');
  const nodeLayer = document.getElementById('nodeLayer');
  const searchInput = document.getElementById('graphSearch');
  const searchResults = document.getElementById('searchResults');
  const detailPanel = document.getElementById('detailPanel');
  const detailClose = document.getElementById('detailClose');

  const familyColors = {
    root:'#252720', candidate:'#3f6f60', representation:'#506d83', training:'#637a8a', mechanism:'#a06c35',
    structure:'#745f7e', uncertainty:'#5b7480', graph:'#747b55', generalization:'#8c665e', routing:'#806d46',
    expert:'#4f6e80', evaluation:'#777970', fusion:'#825f4f', context:'#6f6a8c', fibre:'#73587b', final:'#315f49', milestone:'#252720'
  };
  const statusColors = {root:'#252720',keep:'#315f49',turn:'#a1712c',local:'#4f6d7f',reject:'#a14d3e',historical:'#73587b',considered:'#85867f'};
  const NODE_WIDTH = {milestone:190,program:178,experiment:164,considered:164};
  const NODE_HEIGHT = {milestone:76,program:66,experiment:56,considered:56};
  const H_GAP = 34;
  const DEPTH_GAP = 185;
  const PAD_X = 150;
  const PAD_Y = 105;

  const transitionNotes = {
    pocket_audit:'check the structural input first',
    full_library_structure:'pair score ≠ library ranking',
    reaction_transfer:'reaction neighbors rescue rank',
    gate_coverage_ceiling:'43.98% coverage creates a hard recall ceiling',
    open_problem:'new entities must enter from molecular inputs',
    dual_tower:'factorize reaction demand and enzyme capability',
    broad:'ordered full-space retrieval becomes the safe base',
    generalization_program:'larger coverage exposes forgetting trade-offs',
    fusion_program:'conflicting evidence forces explicit expert routing',
    bime:'organize admitted experts around protected routes',
    fibre:'test whether one relational geometry can replace the stack',
    return_broad:'replacement fails → restore Broad as the base order',
    query_applicability:'availability is weaker than query-specific usefulness',
    bridge:'Broad base + gated, bounded specialists',
    user_semantic_routing:'real users change scientific scope conversationally',
    wetlab_program:'ranking becomes an experimental decision'
  };

  const positions = new Map();
  const subtreeWidths = new Map();
  const layoutMeta = new Map();
  const depthMap = new Map();
  let canvasWidth = 0;
  let canvasHeight = 0;
  let selectedId = null;
  let hoverId = null;
  let inheritanceOn = true;
  let initialized = false;
  let transform = {x:0,y:0,k:1};
  const activePointers = new Map();
  let panStart = null;
  let pinchStart = null;
  let gestureMoved = false;

  function esc(value) {
    return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  }
  function svgEl(name, attrs) {
    const el = document.createElementNS(SVG_NS, name);
    Object.entries(attrs || {}).forEach(([key,value]) => el.setAttribute(key, String(value)));
    return el;
  }
  function nodeWidth(id) {
    const node = byId.get(id);
    return NODE_WIDTH[node.kind] || NODE_WIDTH.experiment;
  }
  function nodeHeight(id) {
    const node = byId.get(id);
    return NODE_HEIGHT[node.kind] || NODE_HEIGHT.experiment;
  }
  function sideTotal(ids) {
    if (!ids.length) return 0;
    return ids.reduce((sum,id) => sum + subtreeWidth(id),0) + H_GAP * (ids.length - 1);
  }
  function splitSides(ids) {
    const order = new Map(ids.map((id,index) => [id,index]));
    const sorted = ids.slice().sort((a,b) => subtreeWidth(b) - subtreeWidth(a));
    const left = [], right = [];
    let lw = 0, rw = 0;
    sorted.forEach(id => {
      const add = subtreeWidth(id) + H_GAP;
      if (lw <= rw) { left.push(id); lw += add; }
      else { right.push(id); rw += add; }
    });
    left.sort((a,b) => order.get(a) - order.get(b));
    right.sort((a,b) => order.get(a) - order.get(b));
    return [left,right];
  }
  function subtreeWidth(id) {
    if (subtreeWidths.has(id)) return subtreeWidths.get(id);
    const kids = children.get(id) || [];
    const own = nodeWidth(id);
    if (!kids.length) {
      subtreeWidths.set(id,own);
      layoutMeta.set(id,{left:[],main:null,right:[]});
      return own;
    }
    const mainChild = kids.find(child => byId.get(child).main) || null;
    if (mainChild) {
      const sides = kids.filter(child => child !== mainChild);
      const [left,right] = splitSides(sides);
      const sideSpan = Math.max(sideTotal(left),sideTotal(right));
      const span = Math.max(own, subtreeWidth(mainChild) + (sideSpan ? 2 * (sideSpan + H_GAP) : 0));
      subtreeWidths.set(id,span);
      layoutMeta.set(id,{left,main:mainChild,right});
      return span;
    }
    const span = Math.max(own, sideTotal(kids));
    subtreeWidths.set(id,span);
    layoutMeta.set(id,{left:kids.slice(),main:null,right:[]});
    return span;
  }
  function placeGroup(ids,left,depth) {
    let cursor = left;
    ids.forEach(id => {
      const width = subtreeWidth(id);
      placeNode(id,cursor,depth);
      cursor += width + H_GAP;
    });
  }
  function placeNode(id,left,depth) {
    depthMap.set(id,depth);
    const span = subtreeWidth(id);
    const meta = layoutMeta.get(id);
    let x;
    if (meta.main) {
      const mainWidth = subtreeWidth(meta.main);
      const center = left + span / 2;
      placeNode(meta.main, center - mainWidth / 2, depth + 1);
      const leftTotal = sideTotal(meta.left);
      if (leftTotal) placeGroup(meta.left, center - mainWidth/2 - H_GAP - leftTotal, depth + 1);
      if (meta.right.length) placeGroup(meta.right, center + mainWidth/2 + H_GAP, depth + 1);
      x = center;
    } else if (meta.left.length) {
      const total = sideTotal(meta.left);
      let cursor = left + (span - total) / 2;
      const centers = [];
      meta.left.forEach(child => {
        const width = subtreeWidth(child);
        placeNode(child,cursor,depth+1);
        centers.push(positions.get(child).x);
        cursor += width + H_GAP;
      });
      x = (centers[0] + centers[centers.length-1]) / 2;
    } else {
      x = left + span / 2;
    }
    positions.set(id,{x:x,y:PAD_Y + depth * DEPTH_GAP});
  }
  function buildLayout() {
    subtreeWidths.clear();layoutMeta.clear();positions.clear();depthMap.clear();
    const root = data.meta.root;
    const rootSpan = subtreeWidth(root);
    placeNode(root,PAD_X,0);
    const maxDepth = Math.max(...depthMap.values());
    canvasWidth = rootSpan + PAD_X * 2;
    canvasHeight = PAD_Y * 2 + maxDepth * DEPTH_GAP;
  }

  function primaryPath(a,b) {
    const ah = nodeHeight(a), bh = nodeHeight(b);
    const p = positions.get(a), q = positions.get(b);
    const sy = p.y + ah/2, ty = q.y - bh/2;
    const my = (sy + ty) / 2;
    return `M ${p.x} ${sy} C ${p.x} ${my}, ${q.x} ${my}, ${q.x} ${ty}`;
  }
  function crossPath(a,b) {
    const p = positions.get(a), q = positions.get(b);
    const sy = p.y, ty = q.y;
    const bend = Math.max(80,Math.abs(q.x-p.x)*.22);
    const dir = q.x >= p.x ? 1 : -1;
    const midY = Math.min(sy,ty) - Math.min(120,Math.abs(ty-sy)*.18 + 35);
    return `M ${p.x} ${sy} C ${p.x + bend*dir} ${midY}, ${q.x - bend*dir} ${midY}, ${q.x} ${ty}`;
  }
  function drawEdges() {
    primaryEdges.innerHTML='';crossEdges.innerHTML='';
    nodes.forEach(node => {
      if (!node.parent) return;
      const parent = byId.get(node.parent);
      const mainEdge = Boolean(node.main && parent && parent.main);
      const cls = `primary-edge${mainEdge?' main':''}${node.status==='turn'?' turn':''}`;
      const path = svgEl('path',{d:primaryPath(node.parent,node.id),class:cls,'data-source':node.parent,'data-target':node.id});
      primaryEdges.appendChild(path);
    });
    crossLinks.forEach(link => {
      const path = svgEl('path',{d:crossPath(link.source,link.target),class:'cross-edge','data-source':link.source,'data-target':link.target});
      const title = svgEl('title',{});title.textContent=link.label;path.appendChild(title);
      crossEdges.appendChild(path);
    });
  }

  function wrapLabel(label,maxChars,maxLines) {
    const tokens = String(label).replace(/\s*\/\s*/g,' / ').split(/\s+/);
    const lines=[];let line='';
    tokens.forEach(token => {
      const next=line?`${line} ${token}`:token;
      if(next.length>maxChars&&line){lines.push(line);line=token;}else line=next;
    });
    if(line)lines.push(line);
    if(lines.length>maxLines){
      const kept=lines.slice(0,maxLines);
      kept[maxLines-1]=`${kept[maxLines-1].slice(0,Math.max(4,maxChars-1))}…`;
      return kept;
    }
    return lines;
  }
  function nodeClass(node) {
    const classes=['node',`node-${node.kind}`,`node-${node.status}`];
    if(node.id==='enzymecage')classes.push('node-root');
    if(node.id==='bridge')classes.push('node-bridge');
    return classes.join(' ');
  }
  function familyLabel(node) {
    if(node.id==='enzymecage')return 'ROOT';
    if(node.id==='bridge')return 'CURRENT METHOD';
    if(node.kind==='milestone')return 'MILESTONE';
    if(node.kind==='program')return (families[node.family]||node.family).toUpperCase();
    return '';
  }
  function drawNodes() {
    nodeLayer.innerHTML='';
    nodes.forEach(node => {
      const pos=positions.get(node.id);const w=nodeWidth(node.id),h=nodeHeight(node.id);
      const g=svgEl('g',{class:nodeClass(node),transform:`translate(${pos.x} ${pos.y})`,'data-id':node.id,tabindex:'0'});
      const hit=svgEl('rect',{x:-w/2-7,y:-h/2-7,width:w+14,height:h+14,rx:16,class:'node-hit'});g.appendChild(hit);
      const box=svgEl('rect',{x:-w/2,y:-h/2,width:w,height:h,rx:node.kind==='milestone'?16:node.kind==='program'?13:10,class:'node-box'});g.appendChild(box);
      if(node.id!=='enzymecage'&&node.id!=='bridge'){
        const bar=svgEl('rect',{x:-w/2,y:-h/2,width:4,height:h,rx:2,fill:statusColors[node.status]||familyColors[node.family]||'#777',class:'node-status'});g.appendChild(bar);
      }
      const kicker=familyLabel(node);
      if(kicker){
        const kt=svgEl('text',{x:0,y:-h/2+15,'text-anchor':'middle',class:'node-kicker'});kt.textContent=kicker;g.appendChild(kt);
      } else {
        const dot=svgEl('circle',{cx:-w/2+13,cy:-h/2+13,r:3.3,fill:statusColors[node.status]||'#777',class:'node-status'});g.appendChild(dot);
      }
      const maxChars=node.kind==='milestone'?23:node.kind==='program'?24:25;
      const maxLines=node.kind==='milestone'?3:2;
      const lines=wrapLabel(node.label,maxChars,maxLines);
      const lineGap=node.kind==='milestone'?14:12;
      const total=(lines.length-1)*lineGap;
      const base=(kicker?5:2)-total/2;
      lines.forEach((line,index)=>{
        const t=svgEl('text',{x:0,y:base+index*lineGap,'text-anchor':'middle',class:'node-label'});t.textContent=line;g.appendChild(t);
      });
      const title=svgEl('title',{});title.textContent=`${node.label} — ${node.status.toUpperCase()}`;g.appendChild(title);
      g.addEventListener('mouseenter',()=>{hoverId=node.id;applyHighlight();});
      g.addEventListener('mouseleave',()=>{hoverId=null;applyHighlight();});
      g.addEventListener('click',event=>{event.stopPropagation();if(gestureMoved){gestureMoved=false;return;}selectNode(node.id,false);});
      g.addEventListener('dblclick',event=>{event.stopPropagation();focusNode(node.id,1.08,.48);selectNode(node.id,false);});
      g.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectNode(node.id,true);}});
      nodeLayer.appendChild(g);
    });
  }
  function drawTransitionLabels() {
    transitionLabels.innerHTML='';
    Object.entries(transitionNotes).forEach(([id,text])=>{
      const node=byId.get(id);if(!node||!node.parent)return;
      const p=positions.get(node.parent),q=positions.get(id);
      let x=(p.x+q.x)/2, y=(p.y+q.y)/2;
      if(Math.abs(p.x-q.x)<30)x+=118;
      const width=Math.max(92,Math.min(240,text.length*5.1+18));
      const g=svgEl('g',{class:`transition-note${node.main?' logic-strong':''}`,transform:`translate(${x} ${y})`});
      const rect=svgEl('rect',{x:-width/2,y:-11,width,height:22,rx:8});g.appendChild(rect);
      const t=svgEl('text',{x:0,y:3,'text-anchor':'middle'});t.textContent=text;g.appendChild(t);
      transitionLabels.appendChild(g);
    });
  }

  function ancestorSet(id) {
    const set=new Set();let cur=id;
    while(cur){set.add(cur);cur=byId.get(cur).parent;}
    return set;
  }
  function relatedSet(id) {
    const set=ancestorSet(id);
    (children.get(id)||[]).forEach(x=>set.add(x));
    crossLinks.forEach(link=>{if(link.source===id)set.add(link.target);if(link.target===id)set.add(link.source);});
    return set;
  }
  function applyHighlight() {
    const target=hoverId||selectedId;
    const related=target?relatedSet(target):null;
    nodeLayer.querySelectorAll('.node').forEach(el=>{
      const id=el.dataset.id;
      el.classList.toggle('dim',Boolean(related)&&!related.has(id));
      el.classList.toggle('selected',id===selectedId);
      el.classList.toggle('related',Boolean(related)&&related.has(id)&&id!==selectedId);
    });
    primaryEdges.querySelectorAll('path').forEach(el=>{
      if(!related){el.classList.remove('edge-dim');return;}
      const s=el.dataset.source,t=el.dataset.target;
      el.classList.toggle('edge-dim',!(related.has(s)&&related.has(t)));
    });
    crossEdges.querySelectorAll('path').forEach(el=>{
      if(!inheritanceOn){el.style.display='none';return;}else el.style.display='';
      if(!related){el.classList.remove('edge-dim','related');return;}
      const s=el.dataset.source,t=el.dataset.target;
      const active=related.has(s)&&related.has(t);
      el.classList.toggle('edge-dim',!active);
      el.classList.toggle('related',active);
    });
  }

  function relationButton(id,label) {
    return `<button type="button" data-focus="${esc(id)}">${esc(label||byId.get(id).label)}</button>`;
  }
  function selectNode(id,focus) {
    selectedId=id;
    const node=byId.get(id);
    const kids=children.get(id)||[];
    const linksOut=crossLinks.filter(link=>link.source===id);
    const linksIn=crossLinks.filter(link=>link.target===id);
    detailPanel.classList.add('has-selection');
    detailPanel.innerHTML=`<button type="button" class="detail-close" aria-label="Close details">×</button>
      <span class="section-index">${esc(families[node.family]||node.family)}</span><h3>${esc(node.label)}</h3>
      <div class="detail-meta"><span class="detail-chip">${esc(node.status)}</span><span class="detail-chip">${esc(node.kind)}</span>${node.main?'<span class="detail-chip">surviving descent</span>':''}</div>
      <div class="detail-block"><b>Why this appeared</b><p>${esc(node.why)}</p></div>
      <div class="detail-block"><b>What happened</b><p>${esc(node.result)}</p></div>
      <div class="detail-block"><b>What changed next</b><p>${esc(node.legacy)}</p></div>
      ${node.parent?`<div class="detail-block"><b>Direct parent</b><div class="relations">${relationButton(node.parent)}</div></div>`:''}
      ${kids.length?`<div class="detail-block"><b>Direct descendants</b><div class="relations">${kids.map(child=>relationButton(child)).join('')}</div></div>`:''}
      ${(linksOut.length||linksIn.length)?`<div class="detail-block"><b>Inherited across branches</b><div class="relations">${linksOut.map(link=>relationButton(link.target,`→ ${byId.get(link.target).label}`)).join('')}${linksIn.map(link=>relationButton(link.source,`← ${byId.get(link.source).label}`)).join('')}</div></div>`:''}
      <button type="button" class="detail-action" data-center="${esc(id)}">Center this node</button>`;
    const close=detailPanel.querySelector('.detail-close');if(close)close.addEventListener('click',()=>{detailPanel.classList.remove('has-selection');selectedId=null;applyHighlight();});
    detailPanel.querySelectorAll('[data-focus]').forEach(btn=>btn.addEventListener('click',()=>{focusNode(btn.dataset.focus,1.05,.48);selectNode(btn.dataset.focus,false);}));
    const center=detailPanel.querySelector('[data-center]');if(center)center.addEventListener('click',()=>focusNode(center.dataset.center,1.08,.48));
    applyHighlight();
    if(focus)focusNode(id,1.08,.48);
  }

  function clampScale(k){return Math.max(.025,Math.min(2.6,k));}
  function viewportSize(){const r=stage.getBoundingClientRect();return {w:r.width,h:r.height,left:r.left,top:r.top};}
  function applyTransform(){
    scene.setAttribute('transform',`translate(${transform.x} ${transform.y}) scale(${transform.k})`);
    stage.classList.toggle('zoom-far',transform.k<.48);
    stage.classList.toggle('zoom-very-far',transform.k<.18);
  }
  function zoomAt(newK,sx,sy){
    const k=clampScale(newK);const wx=(sx-transform.x)/transform.k,wy=(sy-transform.y)/transform.k;
    transform.x=sx-wx*k;transform.y=sy-wy*k;transform.k=k;applyTransform();
  }
  function focusNode(id,k,targetY){
    const size=viewportSize();const p=positions.get(id);const scale=clampScale(k||1);
    transform.k=scale;transform.x=size.w/2-p.x*scale;transform.y=size.h*(targetY==null ? .5 : targetY)-p.y*scale;applyTransform();
  }
  function focusRoot(){focusNode(data.meta.root,stage.clientWidth<720 ? .9 : .92,.24);}
  function fitTree(){
    const size=viewportSize();const k=clampScale(Math.min((size.w-34)/canvasWidth,(size.h-34)/canvasHeight));
    transform.k=k;transform.x=(size.w-canvasWidth*k)/2;transform.y=(size.h-canvasHeight*k)/2;applyTransform();
  }
  function zoomStep(mult){const size=viewportSize();zoomAt(transform.k*mult,size.w/2,size.h/2);}

  function pointerCenter(points){
    const arr=[...points.values()];return {x:(arr[0].x+arr[1].x)/2,y:(arr[0].y+arr[1].y)/2};
  }
  function pointerDistance(points){
    const arr=[...points.values()];return Math.hypot(arr[1].x-arr[0].x,arr[1].y-arr[0].y);
  }
  stage.addEventListener('pointerdown',event=>{
    stage.setPointerCapture(event.pointerId);activePointers.set(event.pointerId,{x:event.clientX,y:event.clientY});gestureMoved=false;
    const rect=stage.getBoundingClientRect();
    if(activePointers.size===1){panStart={clientX:event.clientX,clientY:event.clientY,x:transform.x,y:transform.y};stage.classList.add('dragging');}
    else if(activePointers.size>=2){
      const center=pointerCenter(activePointers);const dist=pointerDistance(activePointers);const sx=center.x-rect.left,sy=center.y-rect.top;
      pinchStart={dist,k:transform.k,worldX:(sx-transform.x)/transform.k,worldY:(sy-transform.y)/transform.k};panStart=null;
    }
  });
  stage.addEventListener('pointermove',event=>{
    if(!activePointers.has(event.pointerId))return;
    activePointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    const rect=stage.getBoundingClientRect();
    if(activePointers.size>=2&&pinchStart){
      const center=pointerCenter(activePointers);const dist=pointerDistance(activePointers);const sx=center.x-rect.left,sy=center.y-rect.top;
      const k=clampScale(pinchStart.k*(dist/pinchStart.dist));transform.k=k;transform.x=sx-pinchStart.worldX*k;transform.y=sy-pinchStart.worldY*k;gestureMoved=true;applyTransform();
    } else if(activePointers.size===1&&panStart){
      const dx=event.clientX-panStart.clientX,dy=event.clientY-panStart.clientY;
      if(Math.hypot(dx,dy)>5)gestureMoved=true;transform.x=panStart.x+dx;transform.y=panStart.y+dy;applyTransform();
    }
  });
  function endPointer(event){
    activePointers.delete(event.pointerId);pinchStart=null;
    if(activePointers.size===1){const p=[...activePointers.values()][0];panStart={clientX:p.x,clientY:p.y,x:transform.x,y:transform.y};}
    else {panStart=null;stage.classList.remove('dragging');}
  }
  stage.addEventListener('pointerup',endPointer);stage.addEventListener('pointercancel',endPointer);
  stage.addEventListener('wheel',event=>{event.preventDefault();const rect=stage.getBoundingClientRect();const sx=event.clientX-rect.left,sy=event.clientY-rect.top;zoomAt(transform.k*Math.exp(-event.deltaY*.00125),sx,sy);},{passive:false});

  function updateSearch(){
    const q=searchInput.value.trim().toLowerCase();if(!q){searchResults.hidden=true;return;}
    const matches=nodes.filter(node=>`${node.label} ${node.why} ${node.result} ${node.legacy} ${families[node.family]||''}`.toLowerCase().includes(q)).slice(0,14);
    searchResults.innerHTML=matches.map(node=>`<button type="button" data-result="${esc(node.id)}"><strong>${esc(node.label)}</strong><small>${esc(families[node.family]||node.family)} · ${esc(node.status)}</small></button>`).join('');
    searchResults.hidden=!matches.length;
    searchResults.querySelectorAll('[data-result]').forEach(btn=>btn.addEventListener('click',()=>{searchResults.hidden=true;searchInput.value='';selectNode(btn.dataset.result,true);}));
  }
  searchInput.addEventListener('input',updateSearch);
  searchInput.addEventListener('keydown',event=>{if(event.key==='Enter'){const first=searchResults.querySelector('[data-result]');if(first){event.preventDefault();first.click();}}if(event.key==='Escape')searchResults.hidden=true;});
  document.addEventListener('click',event=>{if(!event.target.closest('.search-block'))searchResults.hidden=true;});
  document.getElementById('rootButton').addEventListener('click',focusRoot);
  document.getElementById('fitButton').addEventListener('click',fitTree);
  document.getElementById('zoomOut').addEventListener('click',()=>zoomStep(.78));
  document.getElementById('zoomIn').addEventListener('click',()=>zoomStep(1.28));
  document.getElementById('inheritButton').addEventListener('click',event=>{inheritanceOn=!inheritanceOn;event.currentTarget.classList.toggle('active',inheritanceOn);event.currentTarget.textContent=inheritanceOn?'Inheritance on':'Inheritance off';applyHighlight();});
  detailClose.addEventListener('click',()=>{detailPanel.classList.remove('has-selection');selectedId=null;applyHighlight();});

  function render(){
    buildLayout();drawEdges();drawTransitionLabels();drawNodes();
    const rect=stage.getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${rect.width} ${rect.height}`);applyHighlight();
    if(!initialized){initialized=true;requestAnimationFrame(focusRoot);}
  }
  const resizeObserver=new ResizeObserver(()=>{const r=stage.getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);if(!initialized)requestAnimationFrame(focusRoot);});
  resizeObserver.observe(stage);
  render();
})();
