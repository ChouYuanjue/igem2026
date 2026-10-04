(() => {
  const data = window.LINEAGE_DATA;
  if (!data || !data.stages) throw new Error('Engineering atlas data is not loaded');

  const byId = new Map(data.nodes.map(n => [n.id,n]));
  const tracks = new Map(data.tracks.map(t => [t.id,t]));
  const atlasRoot = document.getElementById('atlasRoot');
  const architectureRoot = document.getElementById('architectureRoot');
  const stageNav = document.getElementById('stageNav');
  const searchInput = document.getElementById('atlasSearch');
  const searchResults = document.getElementById('searchResults');
  const dialog = document.getElementById('recordDialog');
  const dialogBody = document.getElementById('recordDialogBody');
  const phaseOrder = ['design','build','test','learn'];
  const phaseLetter = {design:'D',build:'B',test:'T',learn:'L'};

  function esc(v){return String(v==null?'':v).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
  function summary(n){const r=String(n.result||'').trim();return /^(Rejected|Historical|Implemented|Insufficient|No improvement|No gain)\.?$/i.test(r)?n.why:r;}

  function phaseMarkup(cycle,key){
    const p=cycle.phases[key];
    return `<div class="phase phase-${key}"><b>${phaseLetter[key]}</b><span>${esc(p.short)}</span></div>`;
  }

  function cycleMarkup(cycle, extraClass=''){
    return `<article class="cycle ${esc(cycle.size)} ${extraClass}" data-cycle="${esc(cycle.id)}">
      <header><h3>${esc(cycle.title)}</h3></header>
      <div class="cycle-diagram">
        <div class="cycle-ring"></div>
        ${phaseOrder.map(k=>phaseMarkup(cycle,k)).join('')}
        <button class="cycle-center" type="button" data-cycle-detail="${esc(cycle.id)}"><span>Learn</span><strong>${esc(cycle.outcome)}</strong></button>
      </div>
      ${cycle.micro.length?`<div class="micro-row">${cycle.micro.map(m=>`<span><strong>${esc(m.label)}</strong><small>${esc(m.result)}</small></span>`).join('')}</div>`:''}
    </article>`;
  }

  function trackMarkup(id){
    const track=tracks.get(id); if(!track) return '';
    return `<button class="track-chip ${esc(id)}" type="button" data-track="${esc(id)}">${esc(track.label)}</button>`;
  }

  function renderSerial(stage){
    return `<div class="serial-flow"><div class="flow-node">${esc(stage.entry)}</div>${stage.cycles.map((c,i)=>`${i?'<div class="flow-arrow">→</div>':''}${cycleMarkup(c)}`).join('')}<div class="flow-arrow">→</div><div class="flow-node exit">${esc(stage.exit)}</div></div>`;
  }

  function renderParallel(stage){
    return `<div class="network parallel-network">
      <div class="flow-node top">${esc(stage.entry)}</div>
      <div class="branch-stem"></div>
      <div class="branch-grid">${stage.cycles.map(c=>cycleMarkup(c)).join('')}</div>
      <div class="merge-stem"></div>
      <div class="flow-node exit">${esc(stage.exit)}</div>
    </div>`;
  }

  function renderNested(stage){
    const macro=stage.macro;
    return `<div class="nested-shell">
      <div class="macro-loop">
        <div class="macro-label"><strong>FIBRE macro loop</strong><span>${esc(stage.entry)} → ${esc(stage.exit)}</span></div>
        <div class="macro-phases"><span><b>D</b>${esc(macro.design)}</span><span><b>B</b>${esc(macro.build)}</span><span><b>T</b>${esc(macro.test)}</span><span><b>L</b>${esc(macro.learn)}</span></div>
        <div class="nested-grid">${stage.cycles.map(c=>cycleMarkup(c,'nested-cycle')).join('')}</div>
      </div>
      <div class="flow-node exit">${esc(stage.exit)}</div>
    </div>`;
  }

  function renderConverge(stage){
    return `<div class="network converge-network">
      <div class="flow-node top">${esc(stage.entry)}</div>
      <div class="branch-stem"></div>
      <div class="branch-grid">${stage.cycles.map(c=>cycleMarkup(c)).join('')}</div>
      <div class="merge-stem strong"></div>
      <div class="bridge-node">${esc(stage.exit)}</div>
    </div>`;
  }

  function stageBody(stage){
    if(stage.layout==='serial') return renderSerial(stage);
    if(stage.layout==='parallel') return renderParallel(stage);
    if(stage.layout==='nested') return renderNested(stage);
    return renderConverge(stage);
  }

  function stageMarkup(stage){
    return `<section class="macro-stage ${esc(stage.layout)} ${esc(stage.weight)}" id="stage-${esc(stage.id)}">
      <header class="stage-head"><span class="stage-index">${esc(stage.index)}</span><div><h2>${esc(stage.title)}</h2><p>${esc(stage.summary)}</p></div><button type="button" data-stage-records="${esc(stage.id)}">Full record</button></header>
      <div class="stage-visual">${stageBody(stage)}</div>
      ${stage.trackIds.length?`<div class="stage-tracks">${stage.trackIds.map(trackMarkup).join('')}</div>`:''}
    </section>`;
  }

  function architectureMarkup(){
    const a=data.architecture;
    return `<div class="architecture-wrap"><header><span>Current architecture</span><h2>BRIDGE</h2></header>
      <div class="arch-flow"><div class="arch-node"><small>base</small><strong>${esc(a.base)}</strong></div><i>→</i><div class="arch-node"><small>control</small><strong>${esc(a.control)}</strong></div><i>→</i><div class="arch-experts">${a.experts.map(x=>`<span>${esc(x)}</span>`).join('')}</div><i>→</i><div class="arch-node"><small>action</small><strong>${esc(a.correction)}</strong></div></div>
      <div class="formula">${esc(a.formula)}</div></div>`;
  }

  function openDialog(html){dialogBody.innerHTML=html;if(typeof dialog.showModal==='function')dialog.showModal();else dialog.setAttribute('open','');bindDialogButtons();}
  function closeDialog(){if(typeof dialog.close==='function'&&dialog.open)dialog.close();else dialog.removeAttribute('open');}

  function openRecord(id){
    const n=byId.get(id); if(!n) return;
    openDialog(`<span class="dialog-kicker">${esc(data.families[n.family]||n.family)} · ${esc(n.status)}</span><h3>${esc(n.label)}</h3><section><small>Why</small><p>${esc(n.why)}</p></section><section><small>Result</small><p>${esc(n.result)}</p></section><section><small>What survived</small><p>${esc(n.legacy)}</p></section>`);
  }

  function findCycle(cycleId){for(const s of data.stages){const c=s.cycles.find(x=>x.id===cycleId);if(c)return {stage:s,cycle:c};}return null;}
  function openCycle(cycleId){
    const found=findCycle(cycleId); if(!found) return; const c=found.cycle;
    openDialog(`<span class="dialog-kicker">DBTL loop</span><h3>${esc(c.title)}</h3><div class="dialog-cycle">${phaseOrder.map(k=>{const p=c.phases[k];return `<article><b>${phaseLetter[k]}</b><span><strong>${esc(p.label)}</strong><small>${esc(p.short)}</small></span><div>${p.keyIds.map(id=>{const n=byId.get(id);return n?`<button type="button" data-record="${esc(id)}">${esc(n.label)}</button>`:'';}).join('')}</div></article>`;}).join('')}</div><div class="dialog-outcome"><small>Learn</small><strong>${esc(c.outcome)}</strong></div>`);
  }

  function fullRecords(stageId){
    const s=data.stages.find(x=>x.id===stageId);if(!s)return;
    const groups=new Map();s.recordIds.forEach(id=>{const n=byId.get(id);if(!n)return;const g=data.families[n.family]||n.family;if(!groups.has(g))groups.set(g,[]);groups.get(g).push(n);});
    openDialog(`<span class="dialog-kicker">Stage ${esc(s.index)} · full record</span><h3>${esc(s.title)}</h3><div class="record-groups">${[...groups.entries()].map(([g,ns])=>`<section><h4>${esc(g)}</h4>${ns.map(n=>`<button type="button" data-record="${esc(n.id)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`).join('')}</section>`).join('')}</div>`);
  }

  function openTrack(id){const t=tracks.get(id);if(!t)return;openDialog(`<span class="dialog-kicker">Parallel project track</span><h3>${esc(t.label)}</h3><div class="track-list">${t.recordIds.map((rid,i)=>{const n=byId.get(rid);return n?`${i?'<i>↓</i>':''}<button type="button" data-record="${esc(rid)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`:'';}).join('')}</div>`);}

  function bindDialogButtons(){dialogBody.querySelectorAll('[data-record]').forEach(b=>b.addEventListener('click',()=>openRecord(b.dataset.record)));}

  function buildNav(){stageNav.innerHTML=data.stages.map(s=>`<button type="button" data-target="stage-${esc(s.id)}">${esc(s.index)}</button>`).join('')+'<button type="button" data-target="architecture">BRIDGE</button>';stageNav.querySelectorAll('[data-target]').forEach(b=>b.addEventListener('click',()=>{const t=document.getElementById(b.dataset.target);if(t)t.scrollIntoView({behavior:'smooth',block:'start'});}));}

  function bindMain(){
    document.querySelectorAll('[data-cycle-detail]').forEach(b=>b.addEventListener('click',()=>openCycle(b.dataset.cycleDetail)));
    document.querySelectorAll('[data-stage-records]').forEach(b=>b.addEventListener('click',()=>fullRecords(b.dataset.stageRecords)));
    document.querySelectorAll('[data-track]').forEach(b=>b.addEventListener('click',()=>openTrack(b.dataset.track)));
  }

  function locateRecord(id){for(const s of data.stages){if(s.recordIds.includes(id)){const el=document.getElementById(`stage-${s.id}`);if(el)el.scrollIntoView({behavior:'smooth',block:'center'});break;}}setTimeout(()=>openRecord(id),250);}
  function updateSearch(){const q=searchInput.value.trim().toLowerCase();if(!q){searchResults.hidden=true;searchResults.innerHTML='';return;}const matches=data.nodes.filter(n=>`${n.label} ${n.why} ${n.result} ${n.legacy}`.toLowerCase().includes(q)).slice(0,12);searchResults.innerHTML=matches.map(n=>`<button type="button" data-result="${esc(n.id)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`).join('');searchResults.hidden=!matches.length;searchResults.querySelectorAll('[data-result]').forEach(b=>b.addEventListener('click',()=>{searchInput.value='';searchResults.hidden=true;locateRecord(b.dataset.result);}));}

  function bindStageObserver(){
    const buttons=new Map([...stageNav.querySelectorAll('[data-target]')].map(b=>[b.dataset.target,b]));
    const sections=[...data.stages.map(s=>document.getElementById(`stage-${s.id}`)),document.getElementById('architecture')].filter(Boolean);
    const observer=new IntersectionObserver(entries=>{
      const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];
      if(!visible)return;
      buttons.forEach(b=>b.classList.remove('active'));
      const active=buttons.get(visible.target.id);if(active)active.classList.add('active');
    },{rootMargin:'-30% 0px -58% 0px',threshold:[0,.1,.25,.5]});
    sections.forEach(s=>observer.observe(s));
  }

  atlasRoot.innerHTML=data.stages.map(stageMarkup).join('');
  architectureRoot.innerHTML=architectureMarkup();
  buildNav();bindMain();bindStageObserver();
  dialog.querySelector('.dialog-close').addEventListener('click',closeDialog);dialog.addEventListener('click',e=>{if(e.target===dialog)closeDialog();});
  searchInput.addEventListener('input',updateSearch);searchInput.addEventListener('keydown',e=>{if(e.key==='Escape')searchResults.hidden=true;if(e.key==='Enter'){const first=searchResults.querySelector('[data-result]');if(first){e.preventDefault();first.click();}}});
  document.addEventListener('click',e=>{if(!e.target.closest('.search-box'))searchResults.hidden=true;});
})();
