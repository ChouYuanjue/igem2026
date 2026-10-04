(() => {
  const data = window.LINEAGE_DATA;
  if (!data || !data.scenes) throw new Error('Engineering scene data is not loaded');

  const byId = new Map(data.nodes.map(n => [n.id,n]));
  const tracks = new Map(data.tracks.map(t => [t.id,t]));
  const storyRoot = document.getElementById('storyRoot');
  const architectureRoot = document.getElementById('architectureRoot');
  const sceneNav = document.getElementById('sceneNav');
  const searchInput = document.getElementById('storySearch');
  const searchResults = document.getElementById('searchResults');
  const dialog = document.getElementById('detailDialog');
  const detailBody = document.getElementById('detailBody');
  const phaseOrder = ['design','build','test','learn'];
  const phaseLetter = {design:'D',build:'B',test:'T',learn:'L'};
  const recordToScene = new Map();
  data.scenes.forEach(scene => scene.recordIds.forEach(id => { if(!recordToScene.has(id)) recordToScene.set(id,scene.id); }));

  function esc(v){return String(v==null?'':v).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
  function summary(n){const r=String(n.result||'').trim();return /^(Rejected|Historical|Implemented|Insufficient|No improvement|No gain)\.?$/i.test(r)?n.why:r;}

  function stackMarkup(scene){
    const s=scene.stack;
    return `<div class="stack-card ${scene.detour?'with-side':''}">
      <div class="stack-label">system state</div>
      <div class="stack-layers">
        ${s.upper?`<div class="stack-layer upper"><small>upper</small><strong>${esc(s.upper.title)}</strong><span>${esc(s.upper.note)}</span></div>`:''}
        ${s.middle?`<div class="stack-arrow">↓</div><div class="stack-layer middle"><small>control</small><strong>${esc(s.middle.title)}</strong><span>${esc(s.middle.note)}</span></div>`:''}
        ${s.lower?`<div class="stack-arrow">↓</div><div class="stack-layer lower"><small>lower</small><strong>${esc(s.lower.title)}</strong><span>${esc(s.lower.note)}</span></div>`:''}
      </div>
      ${s.side?`<div class="stack-side"><small>side branch</small><strong>${esc(s.side.title)}</strong><span>${esc(s.side.note)}</span></div>`:''}
      <div class="role-strip"><span><b>CAGE</b>${esc(scene.cageRole)}</span>${scene.broadRole?`<span><b>Broad</b>${esc(scene.broadRole)}</span>`:''}</div>
    </div>`;
  }

  function radialLoop(loop){
    return `<article class="primary-loop">
      <header><span>Primary DBTL loop</span><h3>${esc(loop.title)}</h3></header>
      <div class="loop-layout">
        <div class="phase-callout phase-design"><b>D</b><div><small>${esc(loop.phases.design.label)}</small><span>${esc(loop.phases.design.text)}</span></div></div>
        <div class="phase-callout phase-learn"><b>L</b><div><small>${esc(loop.phases.learn.label)}</small><span>${esc(loop.phases.learn.text)}</span></div></div>
        <button class="radial-core" type="button" data-loop-detail="${esc(loop.id)}">
          <div class="radial-ring"><span>D</span><span>B</span><span>T</span><span>L</span></div>
          <div class="radial-center"><small>Learn</small><strong>${esc(loop.outcome)}</strong></div>
        </button>
        <div class="phase-callout phase-build"><b>B</b><div><small>${esc(loop.phases.build.label)}</small><span>${esc(loop.phases.build.text)}</span></div></div>
        <div class="phase-callout phase-test"><b>T</b><div><small>${esc(loop.phases.test.label)}</small><span>${esc(loop.phases.test.text)}</span></div></div>
      </div>
      <div class="why-next"><small>Why next</small><strong>${esc(loop.whyNext)}</strong></div>
    </article>`;
  }

  function miniLoop(loop){
    return `<button class="mini-loop" type="button" data-mini-loop="${esc(loop.id)}">
      <span class="mini-ring" aria-hidden="true"></span>
      <span><strong>${esc(loop.title)}</strong><small>${esc(loop.outcome)}</small></span>
      ${loop.children.length?`<em>${loop.children.length} local loops</em>`:''}
    </button>`;
  }

  function trackMarkup(id){const t=tracks.get(id);return t?`<button class="track-link ${esc(id)}" type="button" data-track="${esc(id)}">${esc(t.label)}</button>`:'';}

  function sceneMarkup(scene){
    return `<section class="story-scene ${scene.detour?'detour':''}" id="scene-${esc(scene.id)}" data-scene="${esc(scene.id)}">
      <header class="scene-head"><span>${esc(scene.number)}</span><div><small>${esc(scene.eyebrow)}</small><h2>${esc(scene.title)}</h2><p>${esc(scene.lead)}</p></div><button type="button" data-scene-records="${esc(scene.id)}">Full record</button></header>
      <div class="scene-grid">
        <aside>${stackMarkup(scene)}</aside>
        <div class="scene-main">
          ${radialLoop(scene.primary)}
          ${scene.secondary.length?`<div class="secondary-wrap"><div class="secondary-title">Secondary loops</div><div class="secondary-grid">${scene.secondary.map(miniLoop).join('')}</div></div>`:''}
          ${scene.evidence?`<div class="scene-evidence">${scene.evidence.map(e=>`<span><small>${esc(e.label)}</small><strong>${esc(e.value)}</strong></span>`).join('')}</div>`:''}
          ${scene.trackIds.length?`<div class="scene-tracks">${scene.trackIds.map(trackMarkup).join('')}</div>`:''}
        </div>
      </div>
    </section>`;
  }

  function architectureMarkup(){
    const a=data.architecture;
    return `<div class="architecture-wrap"><header><small>Current architecture</small><h2>BRIDGE</h2><p>The engineering story ends in a simple authority rule: Broad is always valid; optional evidence may modify it only when permitted.</p></header>
      <div class="arch-flow"><div class="arch-block"><small>base</small><strong>${esc(a.base)}</strong></div><i>→</i><div class="arch-block"><small>control</small><strong>${esc(a.control)}</strong></div><i>→</i><div class="expert-field">${a.experts.map(x=>`<span>${esc(x)}</span>`).join('')}</div><i>→</i><div class="arch-block"><small>action</small><strong>${esc(a.correction)}</strong></div></div>
      <div class="formula-card"><small>ranking rule</small><math class="bridge-math" display="block" aria-label="BRIDGE ranking formula"><mrow><msub><mi>S</mi><mi>BRIDGE</mi></msub><mo>(</mo><mi>q</mi><mo>,</mo><mi>e</mi><mo>)</mo><mo>=</mo><msub><mi>S</mi><mi>Broad</mi></msub><mo>(</mo><mi>q</mi><mo>,</mo><mi>e</mi><mo>)</mo><mo>+</mo><munder><mo>∑</mo><mi>k</mi></munder><msub><mi>g</mi><mi>k</mi></msub><mo>(</mo><mi>q</mi><mo>)</mo><msub><mi>Δ</mi><mi>k</mi></msub><mo>(</mo><mi>q</mi><mo>,</mo><mi>e</mi><mo>)</mo></mrow></math></div>
    </div>`;
  }

  function openDialog(html,wide=false){dialog.classList.toggle('wide',wide);detailBody.innerHTML=html;if(!dialog.open){if(typeof dialog.showModal==='function')dialog.showModal();else dialog.setAttribute('open','');}bindDialogButtons();}
  function closeDialog(){if(typeof dialog.close==='function'&&dialog.open)dialog.close();else dialog.removeAttribute('open');}

  function openRecord(id){const n=byId.get(id);if(!n)return;openDialog(`<span class="dialog-kicker">${esc(data.families[n.family]||n.family)} · ${esc(n.status)}</span><h3>${esc(n.label)}</h3><section><small>Why</small><p>${esc(n.why)}</p></section><section><small>Result</small><p>${esc(n.result)}</p></section><section><small>What survived</small><p>${esc(n.legacy)}</p></section>`);}

  function findLoop(id){for(const scene of data.scenes){if(scene.primary.id===id)return {scene,loop:scene.primary};const m=scene.secondary.find(x=>x.id===id);if(m)return {scene,loop:m,mini:true};}return null;}

  function openPrimaryLoop(id){const found=findLoop(id);if(!found||found.mini)return;const l=found.loop;openDialog(`<span class="dialog-kicker">DBTL loop</span><h3>${esc(l.title)}</h3><div class="dialog-phases">${phaseOrder.map(k=>{const p=l.phases[k];return `<article class="${k}"><b>${phaseLetter[k]}</b><div><strong>${esc(p.label)}</strong><p>${esc(p.text)}</p><div>${p.keyIds.map(rid=>{const n=byId.get(rid);return n?`<button type="button" data-record="${esc(rid)}">${esc(n.label)}</button>`:'';}).join('')}</div></div></article>`;}).join('')}</div><div class="dialog-outcome"><small>Learn</small><strong>${esc(l.outcome)}</strong><p>${esc(l.whyNext)}</p></div>`,true);}

  function openMiniLoop(id){const found=findLoop(id);if(!found||!found.mini)return;const l=found.loop;openDialog(`<span class="dialog-kicker">Secondary loop</span><h3>${esc(l.title)}</h3><p class="dialog-lede">${esc(l.outcome)}</p>${l.children.length?`<div class="child-loops">${l.children.map(c=>`<div><strong>${esc(c.label)}</strong><span>${esc(c.result)}</span></div>`).join('')}</div>`:''}<details class="supporting-records" open><summary>Supporting records <span>${l.recordIds.length}</span></summary><div>${l.recordIds.map(rid=>{const n=byId.get(rid);return n?`<button type="button" data-record="${esc(rid)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`:'';}).join('')}</div></details>`,true);}

  function openTrack(id){const t=tracks.get(id);if(!t)return;openDialog(`<span class="dialog-kicker">Parallel project track</span><h3>${esc(t.label)}</h3><div class="track-list">${t.recordIds.map((rid,i)=>{const n=byId.get(rid);return n?`${i?'<i>↓</i>':''}<button type="button" data-record="${esc(rid)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`:'';}).join('')}</div>`);}

  function sceneRecordHtml(scene){
    const keyIds=new Set();phaseOrder.forEach(k=>scene.primary.phases[k].keyIds.forEach(id=>keyIds.add(id)));scene.secondary.forEach(s=>s.recordIds.forEach(id=>keyIds.add(id)));
    const remaining=scene.recordIds.filter(id=>!keyIds.has(id));
    return `<span class="dialog-kicker">Scene ${esc(scene.number)} · full engineering record</span><h3>${esc(scene.title)}</h3><p class="dialog-lede">${esc(scene.lead)}</p>
      <section class="record-loop"><h4>${esc(scene.primary.title)}</h4><div class="record-phase-grid">${phaseOrder.map(k=>{const p=scene.primary.phases[k];return `<article class="${k}"><b>${phaseLetter[k]}</b><div><strong>${esc(p.label)}</strong><p>${esc(p.text)}</p><div>${p.keyIds.map(rid=>{const n=byId.get(rid);return n?`<button type="button" data-record="${esc(rid)}"><span>${esc(n.label)}</span><small>${esc(summary(n))}</small></button>`:'';}).join('')}</div></div></article>`;}).join('')}</div><div class="record-next"><small>Why next</small><strong>${esc(scene.primary.whyNext)}</strong></div></section>
      ${scene.secondary.length?`<section class="record-secondary"><h4>Secondary loops</h4>${scene.secondary.map(s=>`<button type="button" data-mini-loop="${esc(s.id)}"><strong>${esc(s.title)}</strong><small>${esc(s.outcome)}</small></button>`).join('')}</section>`:''}
      ${remaining.length?`<details class="supporting-records"><summary>Other attempts in this scene <span>${remaining.length}</span></summary><div>${remaining.map(rid=>{const n=byId.get(rid);return n?`<button type="button" data-record="${esc(rid)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`:'';}).join('')}</div></details>`:''}`;
  }

  function openSceneRecords(id){const scene=data.scenes.find(s=>s.id===id);if(scene)openDialog(sceneRecordHtml(scene),true);}

  function bindDialogButtons(){detailBody.querySelectorAll('[data-record]').forEach(b=>b.addEventListener('click',()=>openRecord(b.dataset.record)));detailBody.querySelectorAll('[data-mini-loop]').forEach(b=>b.addEventListener('click',()=>openMiniLoop(b.dataset.miniLoop)));}

  function buildNav(){sceneNav.innerHTML=data.scenes.map(s=>`<button type="button" data-target="scene-${esc(s.id)}">${esc(s.number)}</button>`).join('')+'<button type="button" data-target="architecture">BRIDGE</button>';sceneNav.querySelectorAll('[data-target]').forEach(b=>b.addEventListener('click',()=>{const t=document.getElementById(b.dataset.target);if(t)t.scrollIntoView({behavior:'smooth',block:'start'});}));}

  function bindMain(){document.querySelectorAll('[data-loop-detail]').forEach(b=>b.addEventListener('click',()=>openPrimaryLoop(b.dataset.loopDetail)));document.querySelectorAll('[data-mini-loop]').forEach(b=>b.addEventListener('click',()=>openMiniLoop(b.dataset.miniLoop)));document.querySelectorAll('[data-scene-records]').forEach(b=>b.addEventListener('click',()=>openSceneRecords(b.dataset.sceneRecords)));document.querySelectorAll('[data-track]').forEach(b=>b.addEventListener('click',()=>openTrack(b.dataset.track)));}

  function bindObserver(){const buttons=new Map([...sceneNav.querySelectorAll('[data-target]')].map(b=>[b.dataset.target,b]));const sections=[...data.scenes.map(s=>document.getElementById(`scene-${s.id}`)),document.getElementById('architecture')].filter(Boolean);const obs=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(!visible)return;buttons.forEach(b=>b.classList.remove('active'));const a=buttons.get(visible.target.id);if(a)a.classList.add('active');},{rootMargin:'-28% 0px -58% 0px',threshold:[0,.1,.25,.5]});sections.forEach(s=>obs.observe(s));}

  function locateRecord(id){const sid=recordToScene.get(id);const el=sid?document.getElementById(`scene-${sid}`):null;if(el)el.scrollIntoView({behavior:'smooth',block:'center'});setTimeout(()=>openRecord(id),el?240:0);}
  function updateSearch(){const q=searchInput.value.trim().toLowerCase();if(!q){searchResults.hidden=true;searchResults.innerHTML='';return;}const matches=data.nodes.filter(n=>`${n.label} ${n.why} ${n.result} ${n.legacy}`.toLowerCase().includes(q)).slice(0,12);searchResults.innerHTML=matches.map(n=>`<button type="button" data-result="${esc(n.id)}"><strong>${esc(n.label)}</strong><small>${esc(summary(n))}</small></button>`).join('');searchResults.hidden=!matches.length;searchResults.querySelectorAll('[data-result]').forEach(b=>b.addEventListener('click',()=>{searchInput.value='';searchResults.hidden=true;locateRecord(b.dataset.result);}));}

  storyRoot.innerHTML=data.scenes.map(sceneMarkup).join('');
  architectureRoot.innerHTML=architectureMarkup();
  buildNav();bindMain();bindObserver();
  dialog.querySelector('.dialog-close').addEventListener('click',closeDialog);dialog.addEventListener('click',e=>{if(e.target===dialog)closeDialog();});
  searchInput.addEventListener('input',updateSearch);searchInput.addEventListener('keydown',e=>{if(e.key==='Escape')searchResults.hidden=true;if(e.key==='Enter'){const f=searchResults.querySelector('[data-result]');if(f){e.preventDefault();f.click();}}});document.addEventListener('click',e=>{if(!e.target.closest('.search-box'))searchResults.hidden=true;});
})();
