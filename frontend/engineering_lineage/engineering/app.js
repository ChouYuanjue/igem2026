(() => {
  const data = window.LINEAGE_DATA;
  if (!data) throw new Error('LINEAGE_DATA is not loaded');

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const nodes = data.nodes;
  const crossLinks = data.crossLinks;
  const families = data.families;
  const byId = new Map(nodes.map(n => [n.id, n]));
  const children = new Map(nodes.map(n => [n.id, []]));
  nodes.forEach(n => { if (n.parent) children.get(n.parent).push(n.id); });

  const familyColors = {
    root:'#24251f',candidate:'#3f6f60',representation:'#506d83',training:'#637a8a',mechanism:'#a06c35',structure:'#745f7e',
    uncertainty:'#5b7480',graph:'#747b55',generalization:'#8c665e',routing:'#806d46',expert:'#4f6e80',evaluation:'#777970',
    fusion:'#825f4f',context:'#6f6a8c',fibre:'#765779',final:'#315f49',milestone:'#24251f'
  };
  const statusColors = {root:'#171914',keep:'#315f49',turn:'#9a6e2e',local:'#49697e',reject:'#a14d3e',historical:'#715878',considered:'#85867f'};

  document.getElementById('nodeCount').textContent = `${nodes.length} nodes`;
  document.getElementById('graftCount').textContent = `${crossLinks.length} grafts`;
  document.querySelector('.detail-index').textContent = String(nodes.length);

  const familySelect = document.getElementById('familyFilter');
  [...new Set(nodes.map(n => n.family))].filter(f => families[f]).sort((a,b)=>families[a].localeCompare(families[b])).forEach(f => {
    const opt = document.createElement('option'); opt.value=f; opt.textContent=families[f]; familySelect.appendChild(opt);
  });

  const depth = new Map();
  function getDepth(id){
    if (depth.has(id)) return depth.get(id);
    const n=byId.get(id); const d=n.parent ? getDepth(n.parent)+1 : 0; depth.set(id,d); return d;
  }
  nodes.forEach(n=>getDepth(n.id));

  const leafOrder=[];
  function collectLeaves(id){
    const kids=children.get(id)||[];
    if(!kids.length){leafOrder.push(id);return;}
    kids.forEach(collectLeaves);
  }
  collectLeaves(data.meta.root);

  const positions = new Map();
  const LEAF_GAP=58, X_GAP=265, LEFT=90, TOP=90;
  leafOrder.forEach((id,i)=>positions.set(id,{x:LEFT+getDepth(id)*X_GAP,y:TOP+i*LEAF_GAP}));
  function placeInternal(id){
    const kids=children.get(id)||[];
    if(!kids.length) return positions.get(id);
    kids.forEach(placeInternal);
    const ys=kids.map(k=>positions.get(k).y);
    const p={x:LEFT+getDepth(id)*X_GAP,y:(Math.min(...ys)+Math.max(...ys))/2};positions.set(id,p);return p;
  }
  placeInternal(data.meta.root);

  const maxDepth=Math.max(...nodes.map(n=>getDepth(n.id)));
  const canvas={width:LEFT*2+maxDepth*X_GAP+300,height:TOP*2+(leafOrder.length-1)*LEAF_GAP};
  const svg=document.getElementById('treeSvg');
  svg.setAttribute('viewBox',`0 0 ${canvas.width} ${canvas.height}`);
  const scene=document.getElementById('scene');
  const edgeLayer=document.getElementById('primaryEdges');
  const crossLayer=document.getElementById('crossEdges');
  const nodeLayer=document.getElementById('nodesLayer');
  const regionLayer=document.getElementById('regions');

  function descendants(id){
    const out=[]; const stack=[id];
    while(stack.length){const cur=stack.pop();out.push(cur);(children.get(cur)||[]).forEach(k=>stack.push(k));}
    return out;
  }
  const fibreIds=descendants('fibre');
  function drawFibreRegion(){
    const ps=fibreIds.map(id=>positions.get(id));
    const minX=Math.min(...ps.map(p=>p.x))-45,maxX=Math.max(...ps.map(p=>p.x))+235;
    const minY=Math.min(...ps.map(p=>p.y))-42,maxY=Math.max(...ps.map(p=>p.y))+42;
    const rect=document.createElementNS(SVG_NS,'rect'); rect.setAttribute('x',minX);rect.setAttribute('y',minY);rect.setAttribute('width',maxX-minX);rect.setAttribute('height',maxY-minY);rect.setAttribute('rx',24);rect.setAttribute('class','fibre-region');regionLayer.appendChild(rect);
    const label=document.createElementNS(SVG_NS,'text');label.setAttribute('x',minX+18);label.setAttribute('y',minY+25);label.setAttribute('class','fibre-region-label');label.textContent='FIBRE · UNIFIED-MODEL DETOUR';regionLayer.appendChild(label);
  }
  drawFibreRegion();

  function edgePath(a,b){const dx=Math.max(55,(b.x-a.x)*.46);return `M ${a.x} ${a.y} C ${a.x+dx} ${a.y}, ${b.x-dx} ${b.y}, ${b.x} ${b.y}`;}
  nodes.forEach(n=>{
    if(!n.parent)return;
    const p=document.createElementNS(SVG_NS,'path');
    p.setAttribute('d',edgePath(positions.get(n.parent),positions.get(n.id)));
    p.setAttribute('class',`primary-edge${n.main&&byId.get(n.parent).main?' main':''}`);
    p.setAttribute('stroke',familyColors[n.family]||'#777');
    p.dataset.child=n.id;p.dataset.parent=n.parent;edgeLayer.appendChild(p);
  });

  function graftPath(a,b){
    const bend=Math.max(90,Math.abs(b.x-a.x)*.34);
    if(b.x>=a.x)return `M ${a.x} ${a.y} C ${a.x+bend} ${a.y}, ${b.x-bend} ${b.y}, ${b.x} ${b.y}`;
    const lift=Math.min(a.y,b.y)-90;
    return `M ${a.x} ${a.y} C ${a.x+80} ${lift}, ${b.x-80} ${lift}, ${b.x} ${b.y}`;
  }
  crossLinks.forEach((l,i)=>{
    const p=document.createElementNS(SVG_NS,'path');p.setAttribute('d',graftPath(positions.get(l.source),positions.get(l.target)));p.setAttribute('class','cross-edge');p.dataset.source=l.source;p.dataset.target=l.target;p.dataset.index=i;crossLayer.appendChild(p);
  });

  function labelWidth(text,kind){return Math.min(kind==='milestone'?230:210,Math.max(95,text.length*6.6));}
  nodes.forEach(n=>{
    const pos=positions.get(n.id);const g=document.createElementNS(SVG_NS,'g');
    g.setAttribute('class',`node ${n.kind||'experiment'} ${n.status}${n.main?' main':''}`);g.setAttribute('transform',`translate(${pos.x} ${pos.y})`);g.dataset.id=n.id;
    const hit=document.createElementNS(SVG_NS,'circle');hit.setAttribute('r',16);hit.setAttribute('class','node-hit');g.appendChild(hit);
    const dot=document.createElementNS(SVG_NS,'circle');dot.setAttribute('r',n.kind==='milestone'?7:n.kind==='program'?5.5:4.5);dot.setAttribute('class','node-dot');dot.setAttribute('fill',statusColors[n.status]||familyColors[n.family]||'#555');g.appendChild(dot);
    const fo=document.createElementNS(SVG_NS,'foreignObject');fo.setAttribute('x',12);fo.setAttribute('y',-19);fo.setAttribute('width',labelWidth(n.label,n.kind));fo.setAttribute('height',42);fo.setAttribute('class','node-label');
    const div=document.createElement('div');div.textContent=n.label;fo.appendChild(div);g.appendChild(fo);
    const title=document.createElementNS(SVG_NS,'title');title.textContent=`${n.label} — ${n.status.toUpperCase()}`;g.appendChild(title);
    g.addEventListener('click',e=>{e.stopPropagation();selectNode(n.id,false);});
    g.addEventListener('dblclick',e=>{e.stopPropagation();selectNode(n.id,true);});
    nodeLayer.appendChild(g);
  });

  const viewport=document.getElementById('treeViewport');
  let transform={x:0,y:0,k:1};let drag=null;let graftsOn=true;let selected=null;let activeStatus='all';let activeFamily='all';let searchTerm='';
  function applyTransform(){scene.setAttribute('transform',`translate(${transform.x} ${transform.y}) scale(${transform.k})`);}
  function boundsFor(ids){
    const ps=ids.map(id=>positions.get(id));return {minX:Math.min(...ps.map(p=>p.x))-70,maxX:Math.max(...ps.map(p=>p.x))+250,minY:Math.min(...ps.map(p=>p.y))-70,maxY:Math.max(...ps.map(p=>p.y))+70};
  }
  function fitIds(ids,pad=44){
    const b=boundsFor(ids);const r=viewport.getBoundingClientRect();const k=Math.max(.055,Math.min(1.45,(r.width-pad*2)/(b.maxX-b.minX),(r.height-pad*2)/(b.maxY-b.minY)));
    transform.k=k;transform.x=(r.width-(b.minX+b.maxX)*k)/2;transform.y=(r.height-(b.minY+b.maxY)*k)/2;applyTransform();
  }
  function focusNode(id){const r=viewport.getBoundingClientRect(),p=positions.get(id);transform.k=Math.max(.82,Math.min(1.65,transform.k*1.4));transform.x=r.width/2-p.x*transform.k;transform.y=r.height/2-p.y*transform.k;applyTransform();}
  viewport.addEventListener('wheel',e=>{e.preventDefault();const r=viewport.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;const old=transform.k;const next=Math.max(.055,Math.min(2.4,old*Math.exp(-e.deltaY*.0012)));const wx=(mx-transform.x)/old,wy=(my-transform.y)/old;transform.k=next;transform.x=mx-wx*next;transform.y=my-wy*next;applyTransform();},{passive:false});
  viewport.addEventListener('pointerdown',e=>{if(e.target.closest && e.target.closest('.node'))return;drag={x:e.clientX,y:e.clientY,tx:transform.x,ty:transform.y};viewport.classList.add('dragging');viewport.setPointerCapture(e.pointerId);});
  viewport.addEventListener('pointermove',e=>{if(!drag)return;transform.x=drag.tx+e.clientX-drag.x;transform.y=drag.ty+e.clientY-drag.y;applyTransform();});
  viewport.addEventListener('pointerup',()=>{drag=null;viewport.classList.remove('dragging');});
  viewport.addEventListener('pointercancel',()=>{drag=null;viewport.classList.remove('dragging');});

  function relatedIds(id){
    const set=new Set([id]);const n=byId.get(id);if(n.parent)set.add(n.parent);(children.get(id)||[]).forEach(x=>set.add(x));
    crossLinks.forEach(l=>{if(l.source===id)set.add(l.target);if(l.target===id)set.add(l.source);});return set;
  }
  function refreshStyles(){
    const hits=[];
    document.querySelectorAll('.node').forEach(el=>{
      const n=byId.get(el.dataset.id);const text=`${n.label} ${n.why} ${n.result} ${n.legacy}`.toLowerCase();const searchHit=searchTerm&&text.includes(searchTerm);if(searchHit)hits.push(n.id);
      const statusOk=activeStatus==='all'||n.status===activeStatus||n.status==='root';const familyOk=activeFamily==='all'||n.family===activeFamily||n.status==='root';
      el.classList.toggle('dim',!(statusOk&&familyOk) || (searchTerm&&!searchHit));el.classList.toggle('search-hit',Boolean(searchHit));
      el.classList.toggle('selected',n.id===selected);el.classList.toggle('related',selected&&relatedIds(selected).has(n.id)&&n.id!==selected);
    });
    document.querySelectorAll('.cross-edge').forEach(el=>{const rel=selected&&(el.dataset.source===selected||el.dataset.target===selected);el.classList.toggle('related',Boolean(rel));el.style.display=graftsOn?'':'none';});
    return hits;
  }

  const detail=document.getElementById('detailPanel');
  function relationButton(id){return `<button type="button" data-focus="${id}">${byId.get(id).label}</button>`;}
  function selectNode(id,focus){selected=id;const n=byId.get(id);const kids=children.get(id)||[];const graftOut=crossLinks.filter(l=>l.source===id);const graftIn=crossLinks.filter(l=>l.target===id);
    detail.innerHTML=`<span class="section-index">${families[n.family]||n.family}</span><h3>${n.label}</h3><div class="detail-meta"><span class="detail-chip">${n.status}</span><span class="detail-chip">${n.kind}</span>${n.main?'<span class="detail-chip">backbone</span>':''}</div>
      <div class="detail-block"><b>Why we tried it</b><p>${n.why}</p></div><div class="detail-block"><b>What happened</b><p>${n.result}</p></div><div class="detail-block"><b>What survived</b><p>${n.legacy}</p></div>
      ${n.parent?`<div class="detail-block"><b>Primary parent</b><div class="relations">${relationButton(n.parent)}</div></div>`:''}
      ${kids.length?`<div class="detail-block"><b>Primary descendants</b><div class="relations">${kids.map(relationButton).join('')}</div></div>`:''}
      ${(graftOut.length||graftIn.length)?`<div class="detail-block"><b>Cross-branch inheritance</b><div class="relations">${graftOut.map(l=>relationButton(l.target)).join('')}${graftIn.map(l=>relationButton(l.source)).join('')}</div></div>`:''}`;
    detail.querySelectorAll('[data-focus]').forEach(btn=>btn.addEventListener('click',()=>selectNode(btn.dataset.focus,true)));
    refreshStyles();if(focus)focusNode(id);
  }

  const search=document.getElementById('treeSearch'),searchResults=document.getElementById('searchResults');
  function updateSearch(){searchTerm=search.value.trim().toLowerCase();const hits=refreshStyles();if(!searchTerm){searchResults.hidden=true;return;}const show=hits.slice(0,12);searchResults.innerHTML=show.map(id=>{const n=byId.get(id);return `<button type="button" data-id="${id}"><strong>${n.label}</strong><small>${families[n.family]||n.family} · ${n.status}</small></button>`;}).join('');searchResults.hidden=!show.length;searchResults.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{selectNode(b.dataset.id,true);searchResults.hidden=true;}));}
  search.addEventListener('input',updateSearch);search.addEventListener('keydown',e=>{if(e.key==='Enter'){const first=searchResults.querySelector('button');if(first){e.preventDefault();first.click();}}if(e.key==='Escape')searchResults.hidden=true;});
  document.addEventListener('click',e=>{if(!e.target.closest('.search-block'))searchResults.hidden=true;});

  document.getElementById('statusFilters').addEventListener('click',e=>{const b=e.target.closest('button[data-status]');if(!b)return;activeStatus=b.dataset.status;document.querySelectorAll('#statusFilters button').forEach(x=>x.classList.toggle('active',x===b));refreshStyles();});
  familySelect.addEventListener('change',()=>{activeFamily=familySelect.value;refreshStyles();});
  document.getElementById('fitTree').addEventListener('click',()=>fitIds(nodes.map(n=>n.id)));
  const backbone=nodes.filter(n=>n.main).map(n=>n.id);
  document.getElementById('fitBackbone').addEventListener('click',()=>fitIds(backbone,70));
  document.getElementById('fitFibre').addEventListener('click',()=>fitIds(fibreIds,55));
  document.getElementById('toggleGrafts').addEventListener('click',e=>{graftsOn=!graftsOn;e.currentTarget.classList.toggle('active',graftsOn);e.currentTarget.textContent=graftsOn?'Grafts on':'Grafts off';refreshStyles();});
  window.addEventListener('resize',()=>fitIds(nodes.map(n=>n.id)));

  refreshStyles();requestAnimationFrame(()=>fitIds(nodes.map(n=>n.id)));
})();
