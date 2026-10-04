(() => {
  const data = window.LINEAGE_DATA;
  if (!data) throw new Error('LINEAGE_DATA is not loaded');

  const nodes = data.nodes;
  const crossLinks = data.crossLinks;
  const families = data.families;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map(nodes.map(node => [node.id, []]));
  nodes.forEach(node => { if (node.parent) children.get(node.parent).push(node.id); });

  const stage = document.getElementById('graphStage');
  const canvas = document.getElementById('edgeCanvas');
  const ctx = canvas.getContext('2d');
  const world = document.getElementById('world');
  const nodeLayer = document.getElementById('nodeLayer');
  const logicLabels = document.getElementById('logicLabels');
  const searchInput = document.getElementById('graphSearch');
  const searchResults = document.getElementById('searchResults');
  const detailPanel = document.getElementById('detailPanel');

  const familyColors = {
    root:'#252720',candidate:'#3f6f60',representation:'#506d83',training:'#637a8a',mechanism:'#a06c35',
    structure:'#745f7e',uncertainty:'#5b7480',graph:'#747b55',generalization:'#8c665e',routing:'#806d46',
    expert:'#4f6e80',evaluation:'#777970',fusion:'#825f4f',context:'#6f6a8c',fibre:'#73587b',final:'#315f49',milestone:'#252720'
  };
  const statusColors = {root:'#252720',keep:'#315f49',turn:'#a1712c',local:'#4f6d7f',reject:'#a14d3e',historical:'#73587b',considered:'#85867f'};
  const NODE_WIDTH = {milestone:188,program:176,experiment:160};
  const NODE_HEIGHT = {milestone:82,program:72,experiment:62};
  const SIBLING_GAP = 20;
  const GROUP_GAP = 48;
  const DEPTH_GAP = 150;
  const PAD_X = 170;
  const PAD_Y = 110;

  const transitionNotes = {
    pocket_audit:'verify pocket evidence before changing the ranker',
    full_library_structure:'pairwise compatibility still fails at library ranking',
    reaction_transfer:'reaction neighborhoods recover useful candidates',
    gate_coverage_ceiling:'hard candidate gates create an unrecoverable recall ceiling',
    open_problem:'new entities must enter from molecular inputs',
    dual_tower:'learn continuous reaction–enzyme compatibility',
    broad:'full-space retrieval becomes the safe global order',
    generalization_program:'broader coverage exposes forgetting trade-offs',
    fusion_program:'conditional evidence forces explicit expert routing',
    bime:'admit experts only where they add clean evidence',
    fibre:'test whether one relational core can replace the stack',
    return_broad:'replacement fails; restore Broad as ranking authority',
    query_applicability:'expert usefulness depends on query and direction',
    bridge:'protected Broad order + gated bounded specialists',
    user_semantic_routing:'real users change scientific scope conversationally',
    wetlab_program:'retrieval becomes an experimental decision'
  };

  const depthMap = new Map();
  const levels = new Map();
  const positions = new Map();
  const xRaw = new Map();
  let worldWidth = 0;
  let worldHeight = 0;
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
  function nodeWidth(id) {
    return NODE_WIDTH[byId.get(id).kind] || NODE_WIDTH.experiment;
  }
  function nodeHeight(id) {
    return NODE_HEIGHT[byId.get(id).kind] || NODE_HEIGHT.experiment;
  }
  function depthOf(id) {
    if (depthMap.has(id)) return depthMap.get(id);
    const parent = byId.get(id).parent;
    const depth = parent ? depthOf(parent) + 1 : 0;
    depthMap.set(id, depth);
    return depth;
  }
  function cleanPublicText(text) {
    return String(text || '')
      .replace(/\bV\d+\b/g, '')
      .replace(/\s{2,}/g, ' ')
      .replace(/\s+([,.;:])/g, '$1')
      .trim();
  }
  function shortOutcome(node) {
    const raw = cleanPublicText(node.result);
    const generic = /^(Rejected|Historical|Implemented|Insufficient|No improvement|No gain)\.?$/i.test(raw);
    return generic ? cleanPublicText(node.why) : raw;
  }

  function buildLevels() {
    depthMap.clear();
    levels.clear();
    nodes.forEach(node => {
      const depth = depthOf(node.id);
      if (!levels.has(depth)) levels.set(depth, []);
      levels.get(depth).push(node.id);
    });
  }
  function localGroup(kids) {
    const mainChild = kids.find(id => byId.get(id).main) || null;
    const local = new Map();
    if (mainChild) {
      const others = kids.filter(id => id !== mainChild);
      const left = [], right = [];
      let leftWidth = 0, rightWidth = 0;
      others.slice().sort((a,b) => nodeWidth(b) - nodeWidth(a)).forEach(id => {
        const add = nodeWidth(id) + SIBLING_GAP;
        if (leftWidth <= rightWidth) { left.push(id); leftWidth += add; }
        else { right.push(id); rightWidth += add; }
      });
      const order = new Map(kids.map((id,index) => [id,index]));
      left.sort((a,b) => order.get(a) - order.get(b));
      right.sort((a,b) => order.get(a) - order.get(b));
      local.set(mainChild,0);
      let cursor = -nodeWidth(mainChild)/2 - SIBLING_GAP;
      left.slice().reverse().forEach(id => {
        local.set(id,cursor-nodeWidth(id)/2);
        cursor -= nodeWidth(id) + SIBLING_GAP;
      });
      cursor = nodeWidth(mainChild)/2 + SIBLING_GAP;
      right.forEach(id => {
        local.set(id,cursor+nodeWidth(id)/2);
        cursor += nodeWidth(id) + SIBLING_GAP;
      });
    } else {
      const total = kids.reduce((sum,id) => sum + nodeWidth(id),0) + SIBLING_GAP*Math.max(0,kids.length-1);
      let cursor = -total/2;
      kids.forEach(id => {
        local.set(id,cursor+nodeWidth(id)/2);
        cursor += nodeWidth(id) + SIBLING_GAP;
      });
    }
    let lo = Infinity, hi = -Infinity;
    kids.forEach(id => {
      const x = local.get(id);
      lo = Math.min(lo,x-nodeWidth(id)/2);
      hi = Math.max(hi,x+nodeWidth(id)/2);
    });
    return {local,lo,hi,mainChild};
  }
  function buildCompactLayout() {
    buildLevels();
    xRaw.clear();
    const root = data.meta.root;
    xRaw.set(root,0);
    const maxDepth = Math.max.apply(null,[...levels.keys()]);

    for (let depth=1; depth<=maxDepth; depth+=1) {
      const level = levels.get(depth) || [];
      const parentIds = [];
      level.forEach(id => {
        const parent = byId.get(id).parent;
        if (parentIds.indexOf(parent) === -1) parentIds.push(parent);
      });
      parentIds.sort((a,b) => xRaw.get(a) - xRaw.get(b));

      const groups = [];
      let anchor = null;
      parentIds.forEach(parent => {
        const kids = children.get(parent) || [];
        const shape = localGroup(kids);
        const group = {parent,kids,shape,ideal:xRaw.get(parent)};
        groups.push(group);
        if (byId.get(parent).main && shape.mainChild) anchor = group;
      });

      const origins = new Map();
      if (anchor) {
        origins.set(anchor.parent,anchor.ideal);
        let rightBoundary = anchor.ideal + anchor.shape.lo - GROUP_GAP;
        groups.filter(group => group !== anchor && group.ideal <= anchor.ideal)
          .sort((a,b) => b.ideal-a.ideal)
          .forEach(group => {
            const origin = Math.min(group.ideal,rightBoundary-group.shape.hi);
            origins.set(group.parent,origin);
            rightBoundary = origin + group.shape.lo - GROUP_GAP;
          });
        let leftBoundary = anchor.ideal + anchor.shape.hi + GROUP_GAP;
        groups.filter(group => group !== anchor && group.ideal > anchor.ideal)
          .sort((a,b) => a.ideal-b.ideal)
          .forEach(group => {
            const origin = Math.max(group.ideal,leftBoundary-group.shape.lo);
            origins.set(group.parent,origin);
            leftBoundary = origin + group.shape.hi + GROUP_GAP;
          });
      } else {
        let rightEdge = null;
        groups.forEach(group => {
          let origin = group.ideal;
          if (rightEdge !== null) origin = Math.max(origin,rightEdge+GROUP_GAP-group.shape.lo);
          origins.set(group.parent,origin);
          rightEdge = origin + group.shape.hi;
        });
      }

      groups.forEach(group => {
        const origin = origins.get(group.parent);
        group.kids.forEach(id => xRaw.set(id,origin+group.shape.local.get(id)));
      });
    }

    let minX = Infinity, maxX = -Infinity;
    nodes.forEach(node => {
      const x = xRaw.get(node.id);
      minX = Math.min(minX,x-nodeWidth(node.id)/2);
      maxX = Math.max(maxX,x+nodeWidth(node.id)/2);
    });
    const shiftX = PAD_X - minX;
    positions.clear();
    nodes.forEach(node => {
      const depth = depthOf(node.id);
      positions.set(node.id,{x:xRaw.get(node.id)+shiftX,y:PAD_Y+depth*DEPTH_GAP});
    });
    worldWidth = maxX-minX+PAD_X*2;
    worldHeight = PAD_Y*2+maxDepth*DEPTH_GAP;
    world.style.width = `${worldWidth}px`;
    world.style.height = `${worldHeight}px`;
    nodeLayer.style.width = `${worldWidth}px`;
    nodeLayer.style.height = `${worldHeight}px`;
    logicLabels.style.width = `${worldWidth}px`;
    logicLabels.style.height = `${worldHeight}px`;
  }

  function drawNodes() {
    nodeLayer.innerHTML='';
    nodes.forEach(node => {
      const p = positions.get(node.id);
      const button = document.createElement('button');
      button.type='button';
      button.className=`graph-node ${node.kind} ${node.status}${node.id==='enzymecage'?' root':''}${node.id==='bridge'?' bridge':''}`;
      button.dataset.id=node.id;
      button.style.left=`${p.x}px`;
      button.style.top=`${p.y}px`;
      button.style.width=`${nodeWidth(node.id)}px`;
      button.style.height=`${nodeHeight(node.id)}px`;
      button.style.setProperty('--accent',statusColors[node.status]||familyColors[node.family]||'#777');
      button.innerHTML=`<span class="node-title">${esc(node.label)}</span><span class="node-summary">${esc(shortOutcome(node))}</span>`;
      button.title=`${node.label}: ${cleanPublicText(node.result)}`;
      button.addEventListener('mouseenter',()=>{hoverId=node.id;applyHighlight();});
      button.addEventListener('mouseleave',()=>{hoverId=null;applyHighlight();});
      button.addEventListener('click',event=>{event.stopPropagation();if(gestureMoved){gestureMoved=false;return;}selectNode(node.id,false);});
      button.addEventListener('dblclick',event=>{event.stopPropagation();focusNode(node.id,1.05,.47);selectNode(node.id,false);});
      nodeLayer.appendChild(button);
    });
  }
  function drawLogicLabels() {
    logicLabels.innerHTML='';
    Object.keys(transitionNotes).forEach(id => {
      const node=byId.get(id);
      if(!node||!node.parent)return;
      const a=positions.get(node.parent),b=positions.get(id);
      let x=(a.x+b.x)/2,y=(a.y+b.y)/2;
      if(Math.abs(a.x-b.x)<70)x+=128;
      const div=document.createElement('div');
      div.className=`logic-note${node.main?' main':''}`;
      div.style.left=`${x}px`;div.style.top=`${y}px`;
      div.textContent=transitionNotes[id];
      logicLabels.appendChild(div);
    });
  }

  function activeRelatedSet() {
    const id=hoverId||selectedId;
    if(!id)return null;
    const set=new Set();
    let current=id;
    while(current){set.add(current);current=byId.get(current).parent;}
    (children.get(id)||[]).forEach(child=>set.add(child));
    crossLinks.forEach(link=>{if(link.source===id)set.add(link.target);if(link.target===id)set.add(link.source);});
    return set;
  }
  function resizeCanvas() {
    const rect=stage.getBoundingClientRect();
    const dpr=Math.min(window.devicePixelRatio||1,2.5);
    const width=Math.max(1,Math.round(rect.width*dpr));
    const height=Math.max(1,Math.round(rect.height*dpr));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    return {w:rect.width,h:rect.height,dpr};
  }
  function edgePath(aId,bId) {
    const a=positions.get(aId),b=positions.get(bId);
    const sy=a.y+nodeHeight(aId)/2,ty=b.y-nodeHeight(bId)/2;
    const my=(sy+ty)/2;
    return {sx:a.x,sy,c1x:a.x,c1y:my,c2x:b.x,c2y:my,tx:b.x,ty};
  }
  function drawBezier(path) {
    ctx.beginPath();ctx.moveTo(path.sx,path.sy);ctx.bezierCurveTo(path.c1x,path.c1y,path.c2x,path.c2y,path.tx,path.ty);ctx.stroke();
  }
  function drawCross(link) {
    const a=positions.get(link.source),b=positions.get(link.target);
    const dir=b.x>=a.x?1:-1;
    const bend=Math.max(70,Math.abs(b.x-a.x)*.18);
    const lift=Math.min(a.y,b.y)-Math.min(105,Math.abs(b.y-a.y)*.18+28);
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.bezierCurveTo(a.x+bend*dir,lift,b.x-bend*dir,lift,b.x,b.y);ctx.stroke();
  }
  function drawEdges() {
    const size=resizeCanvas();
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.setTransform(size.dpr*transform.k,0,0,size.dpr*transform.k,size.dpr*transform.x,size.dpr*transform.y);
    ctx.lineCap='round';ctx.lineJoin='round';
    const related=activeRelatedSet();

    nodes.forEach(node=>{
      if(!node.parent)return;
      const parent=byId.get(node.parent);
      const active=!related||(related.has(node.id)&&related.has(node.parent));
      const mainEdge=Boolean(node.main&&parent.main);
      ctx.globalAlpha=active?(mainEdge?.86:node.status==='turn'?.52:.34):.055;
      ctx.strokeStyle=mainEdge?'#20221d':node.status==='turn'?'#9a6e2e':'#777a70';
      ctx.lineWidth=(mainEdge?3.0:1.25)/transform.k;
      ctx.setLineDash([]);
      drawBezier(edgePath(node.parent,node.id));
    });

    if(inheritanceOn){
      crossLinks.forEach(link=>{
        const active=!related||(related.has(link.source)&&related.has(link.target));
        ctx.globalAlpha=active?(related?.78:.20):.035;
        ctx.strokeStyle='#9a763d';ctx.lineWidth=(active&&related?1.8:1.05)/transform.k;
        ctx.setLineDash([5/transform.k,5/transform.k]);drawCross(link);
      });
    }
    ctx.setLineDash([]);ctx.globalAlpha=1;
  }
  function applyHighlight() {
    const related=activeRelatedSet();
    nodeLayer.querySelectorAll('.graph-node').forEach(el=>{
      const id=el.dataset.id;
      el.classList.toggle('dim',Boolean(related)&&!related.has(id));
      el.classList.toggle('selected',id===selectedId);
      el.classList.toggle('related',Boolean(related)&&related.has(id)&&id!==selectedId);
    });
    drawEdges();
  }

  function relationButton(id,label){return `<button type="button" data-focus="${esc(id)}">${esc(label||byId.get(id).label)}</button>`;}
  function selectNode(id,focus){
    selectedId=id;
    const node=byId.get(id),kids=children.get(id)||[];
    const linksOut=crossLinks.filter(link=>link.source===id),linksIn=crossLinks.filter(link=>link.target===id);
    detailPanel.classList.add('has-selection');
    detailPanel.innerHTML=`<button type="button" class="detail-close" aria-label="Close details">×</button>
      <span class="section-index">${esc(families[node.family]||node.family)}</span><h3>${esc(node.label)}</h3>
      <div class="detail-meta"><span class="detail-chip">${esc(node.status)}</span><span class="detail-chip">${esc(node.kind)}</span>${node.main?'<span class="detail-chip">surviving route</span>':''}</div>
      <div class="detail-block"><b>Why this route existed</b><p>${esc(cleanPublicText(node.why))}</p></div>
      <div class="detail-block"><b>What the experiment showed</b><p>${esc(cleanPublicText(node.result))}</p></div>
      <div class="detail-block"><b>What survived</b><p>${esc(cleanPublicText(node.legacy))}</p></div>
      ${node.parent?`<div class="detail-block"><b>Direct parent</b><div class="relations">${relationButton(node.parent)}</div></div>`:''}
      ${kids.length?`<div class="detail-block"><b>Direct descendants</b><div class="relations">${kids.map(child=>relationButton(child)).join('')}</div></div>`:''}
      ${(linksOut.length||linksIn.length)?`<div class="detail-block"><b>Inherited across branches</b><div class="relations">${linksOut.map(link=>relationButton(link.target,`→ ${byId.get(link.target).label}`)).join('')}${linksIn.map(link=>relationButton(link.source,`← ${byId.get(link.source).label}`)).join('')}</div></div>`:''}
      <button type="button" class="detail-action" data-center="${esc(id)}">Center this node</button>`;
    const close=detailPanel.querySelector('.detail-close');
    if(close)close.addEventListener('click',()=>{detailPanel.classList.remove('has-selection');selectedId=null;applyHighlight();});
    detailPanel.querySelectorAll('[data-focus]').forEach(btn=>btn.addEventListener('click',()=>{focusNode(btn.dataset.focus,1.03,.46);selectNode(btn.dataset.focus,false);}));
    const center=detailPanel.querySelector('[data-center]');if(center)center.addEventListener('click',()=>focusNode(center.dataset.center,1.05,.46));
    applyHighlight();if(focus)focusNode(id,1.05,.46);
  }

  function clampScale(k){return Math.max(.03,Math.min(2.4,k));}
  function viewportSize(){const r=stage.getBoundingClientRect();return {w:r.width,h:r.height,left:r.left,top:r.top};}
  function applyTransform(){
    world.style.transform=`translate(${transform.x}px,${transform.y}px) scale(${transform.k})`;
    stage.classList.toggle('zoom-far',transform.k<.52);
    stage.classList.toggle('zoom-very-far',transform.k<.20);
    stage.classList.toggle('zoom-overview',transform.k<.11);
    drawEdges();
  }
  function zoomAt(newK,sx,sy){
    const k=clampScale(newK);const wx=(sx-transform.x)/transform.k,wy=(sy-transform.y)/transform.k;
    transform.x=sx-wx*k;transform.y=sy-wy*k;transform.k=k;applyTransform();
  }
  function focusNode(id,k,targetY){
    const size=viewportSize(),p=positions.get(id),scale=clampScale(k||1);
    transform.k=scale;transform.x=size.w/2-p.x*scale;transform.y=size.h*(targetY==null?.5:targetY)-p.y*scale;applyTransform();
  }
  function focusRoot(){focusNode(data.meta.root,stage.clientWidth<720?.76:.92,.22);}
  function fitTree(){
    const size=viewportSize();const k=clampScale(Math.min((size.w-28)/worldWidth,(size.h-28)/worldHeight));
    transform.k=k;transform.x=(size.w-worldWidth*k)/2;transform.y=(size.h-worldHeight*k)/2;applyTransform();
  }
  function zoomStep(mult){const size=viewportSize();zoomAt(transform.k*mult,size.w/2,size.h/2);}

  function pointerCenter(){const arr=[...activePointers.values()];return {x:(arr[0].x+arr[1].x)/2,y:(arr[0].y+arr[1].y)/2};}
  function pointerDistance(){const arr=[...activePointers.values()];return Math.hypot(arr[1].x-arr[0].x,arr[1].y-arr[0].y);}
  stage.addEventListener('pointerdown',event=>{
    stage.setPointerCapture(event.pointerId);activePointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    if(activePointers.size===1)gestureMoved=false;
    const rect=stage.getBoundingClientRect();
    if(activePointers.size===1){panStart={clientX:event.clientX,clientY:event.clientY,x:transform.x,y:transform.y};stage.classList.add('dragging');}
    else if(activePointers.size>=2){
      const center=pointerCenter(),dist=pointerDistance(),sx=center.x-rect.left,sy=center.y-rect.top;
      pinchStart={dist,k:transform.k,worldX:(sx-transform.x)/transform.k,worldY:(sy-transform.y)/transform.k};panStart=null;
    }
  });
  stage.addEventListener('pointermove',event=>{
    if(!activePointers.has(event.pointerId))return;
    activePointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    const rect=stage.getBoundingClientRect();
    if(activePointers.size>=2&&pinchStart){
      const center=pointerCenter(),dist=pointerDistance(),sx=center.x-rect.left,sy=center.y-rect.top;
      const k=clampScale(pinchStart.k*(dist/pinchStart.dist));transform.k=k;transform.x=sx-pinchStart.worldX*k;transform.y=sy-pinchStart.worldY*k;gestureMoved=true;applyTransform();
    }else if(activePointers.size===1&&panStart){
      const dx=event.clientX-panStart.clientX,dy=event.clientY-panStart.clientY;
      if(Math.hypot(dx,dy)>5)gestureMoved=true;transform.x=panStart.x+dx;transform.y=panStart.y+dy;applyTransform();
    }
  });
  function endPointer(event){
    activePointers.delete(event.pointerId);pinchStart=null;
    if(activePointers.size===1){const p=[...activePointers.values()][0];panStart={clientX:p.x,clientY:p.y,x:transform.x,y:transform.y};}
    else{panStart=null;stage.classList.remove('dragging');}
  }
  stage.addEventListener('pointerup',endPointer);stage.addEventListener('pointercancel',endPointer);
  stage.addEventListener('wheel',event=>{event.preventDefault();const rect=stage.getBoundingClientRect();zoomAt(transform.k*Math.exp(-event.deltaY*.0012),event.clientX-rect.left,event.clientY-rect.top);},{passive:false});

  function updateSearch(){
    const q=searchInput.value.trim().toLowerCase();if(!q){searchResults.hidden=true;return;}
    const matches=nodes.filter(node=>`${node.label} ${node.why} ${node.result} ${node.legacy} ${families[node.family]||''}`.toLowerCase().includes(q)).slice(0,14);
    searchResults.innerHTML=matches.map(node=>`<button type="button" data-result="${esc(node.id)}"><strong>${esc(node.label)}</strong><small>${esc(shortOutcome(node))}</small></button>`).join('');
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
  document.getElementById('inheritButton').addEventListener('click',event=>{inheritanceOn=!inheritanceOn;event.currentTarget.classList.toggle('active',inheritanceOn);applyHighlight();});

  function render(){
    buildCompactLayout();drawNodes();drawLogicLabels();applyHighlight();
    if(!initialized){initialized=true;requestAnimationFrame(focusRoot);}
  }
  const resizeObserver=new ResizeObserver(()=>{resizeCanvas();drawEdges();});
  resizeObserver.observe(stage);
  render();
})();
