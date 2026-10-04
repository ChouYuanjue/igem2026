(() => {
  const data = window.LINEAGE_DATA;
  if (!data || !data.cycles) throw new Error('Engineering cycle data is not loaded');

  const cycles = data.cycles;
  const tracks = new Map(data.tracks.map(track => [track.id, track]));
  const byId = new Map(data.nodes.map(node => [node.id, node]));
  const cycleSpine = document.getElementById('cycleSpine');
  const architectureRoot = document.getElementById('architectureRoot');
  const cycleNav = document.getElementById('cycleNav');
  const searchInput = document.getElementById('cycleSearch');
  const searchResults = document.getElementById('searchResults');
  const recordDialog = document.getElementById('recordDialog');
  const recordDialogBody = document.getElementById('recordDialogBody');

  const phaseOrder = ['design','build','test','learn'];
  const phaseLetters = {design:'D',build:'B',test:'T',learn:'L'};
  const statusLabels = {root:'origin',keep:'kept',turn:'turn',local:'local',reject:'rejected',historical:'historical',considered:'considered'};
  const statusColors = {root:'#252720',keep:'#315f49',turn:'#a1712c',local:'#536f80',reject:'#a14d3e',historical:'#73587b',considered:'#85867f'};
  const recordToCycle = new Map();
  cycles.forEach(cycle => cycle.recordIds.forEach(id => { if (!recordToCycle.has(id)) recordToCycle.set(id, cycle.id); }));

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  }

  function nodeSummary(node) {
    const result = String(node.result || '').trim();
    const generic = /^(Rejected|Historical|Implemented|Insufficient|No improvement|No gain)\.?$/i.test(result);
    return generic ? node.why : result;
  }

  function groupedRecords(cycle) {
    const groups = new Map();
    cycle.recordIds.forEach(id => {
      const node = byId.get(id);
      if (!node) return;
      const label = data.families[node.family] || node.family;
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(node);
    });
    return [...groups.entries()];
  }

  function renderEvidence(cycle) {
    return `<div class="cycle-evidence">${cycle.evidence.map(item => `<div><span>${esc(item.label)}</span><strong>${esc(item.value)}</strong></div>`).join('')}</div>`;
  }

  function renderTrack(trackId) {
    const track = tracks.get(trackId);
    if (!track) return '';
    return `<aside class="parallel-track ${esc(track.id)}">
      <span class="parallel-label">${esc(track.label)}</span>
      <div class="parallel-flow">
        ${track.recordIds.map((id,index) => {
          const node = byId.get(id);
          return node ? `${index?'<i>→</i>':''}<button type="button" data-record="${esc(id)}">${esc(node.label)}</button>` : '';
        }).join('')}
      </div>
    </aside>`;
  }

  function renderRecordAccordion(cycle) {
    const groups = groupedRecords(cycle);
    return `<details class="cycle-records">
      <summary><span>Full experimental record</span><small>All attempts, failures and retained ideas</small></summary>
      <div class="record-groups">
        ${groups.map(([label,nodes]) => `<section class="record-group">
          <h4>${esc(label)}</h4>
          ${nodes.map(node => `<button class="record-row" type="button" data-record="${esc(node.id)}" style="--status:${statusColors[node.status]||'#85867f'}">
            <span class="record-dot"></span><span class="record-main"><strong>${esc(node.label)}</strong><small>${esc(nodeSummary(node))}</small></span><em>${esc(statusLabels[node.status]||node.status)}</em>
          </button>`).join('')}
        </section>`).join('')}
      </div>
    </details>`;
  }

  function renderCycle(cycle, index) {
    const phase = cycle.phases.learn;
    return `<article class="cycle-step" id="cycle-${esc(cycle.id)}" data-cycle="${esc(cycle.id)}">
      <div class="cycle-side cycle-side-left">
        <span class="cycle-number">Cycle ${esc(cycle.number)}</span>
        <h2>${esc(cycle.title)}</h2>
        <p class="cycle-change">${esc(cycle.change)}</p>
        ${renderEvidence(cycle)}
      </div>

      <div class="cycle-path">
        <div class="path-line"></div>
        <div class="dbtl-wheel" data-active="learn">
          <div class="wheel-ring"></div>
          ${phaseOrder.map(key => `<button class="phase-button phase-${key}${key==='learn'?' active':''}" type="button" data-phase="${key}" aria-label="${esc(cycle.phases[key].title)}"><b>${phaseLetters[key]}</b><span>${esc(cycle.phases[key].title)}</span></button>`).join('')}
          <div class="wheel-center"><small>Cycle ${esc(cycle.number)}</small><strong>${esc(cycle.outcome)}</strong></div>
        </div>
        ${index < cycles.length-1 ? '<div class="next-label">Learn feeds next Design ↓</div>' : ''}
      </div>

      <div class="cycle-side cycle-side-right">
        <div class="phase-detail" data-phase-detail>
          <span class="phase-kicker learn">Learn</span>
          <h3>${esc(phase.summary)}</h3>
          <div class="phase-records">${phase.keyIds.map(id => {
            const node = byId.get(id);
            return node ? `<button type="button" data-record="${esc(id)}">${esc(node.label)}</button>` : '';
          }).join('')}</div>
        </div>
        ${cycle.trackIds.map(renderTrack).join('')}
        ${renderRecordAccordion(cycle)}
      </div>
    </article>`;
  }

  function renderArchitecture() {
    const a = data.architecture;
    return `<div class="architecture-wrap">
      <header class="architecture-head">
        <span>After Cycle 07</span>
        <h2>BRIDGE today</h2>
        <p>History ends here. The elements below are components of the final system, not additional engineering generations.</p>
      </header>
      <div class="architecture-flow">
        <div class="arch-node base"><small>global base</small><strong>${esc(a.base.title)}</strong></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-node control"><small>query control</small><strong>${esc(a.control.title)}</strong></div>
        <div class="arch-arrow">↓</div>
        <div class="expert-hub">
          <div class="hub-center"><span>g<sub>k</sub>(q)</span><strong>permission</strong></div>
          ${a.experts.map((expert,index) => `<button class="expert-node expert-${index+1}" type="button" data-expert="${esc(expert.id)}"><strong>${esc(expert.title)}</strong><small>${expert.members.map(esc).join(' · ')}</small></button>`).join('')}
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-node correction"><small>ranking action</small><strong>${esc(a.correction.title)}</strong></div>
      </div>
      <div class="arch-formula">${esc(a.formula)}</div>
      <div class="arch-note" id="archNote">Select an expert to inspect its role. Missing or inapplicable evidence contributes zero.</div>
    </div>`;
  }

  function selectPhase(button) {
    const cycleEl = button.closest('.cycle-step');
    const cycle = cycles.find(item => item.id === cycleEl.dataset.cycle);
    if (!cycle) return;
    const key = button.dataset.phase;
    const phase = cycle.phases[key];
    cycleEl.querySelectorAll('.phase-button').forEach(el => el.classList.toggle('active', el === button));
    const wheel = cycleEl.querySelector('.dbtl-wheel');
    wheel.dataset.active = key;
    const detail = cycleEl.querySelector('[data-phase-detail]');
    detail.innerHTML = `<span class="phase-kicker ${key}">${esc(phase.title)}</span><h3>${esc(phase.summary)}</h3><div class="phase-records">${phase.keyIds.map(id => {
      const node = byId.get(id);
      return node ? `<button type="button" data-record="${esc(id)}">${esc(node.label)}</button>` : '';
    }).join('')}</div>`;
    bindRecordButtons(detail);
  }

  function openRecord(id) {
    const node = byId.get(id);
    if (!node) return;
    recordDialogBody.innerHTML = `<span class="dialog-kicker">${esc(data.families[node.family]||node.family)} · ${esc(statusLabels[node.status]||node.status)}</span>
      <h3>${esc(node.label)}</h3>
      <div class="dialog-section"><small>Why</small><p>${esc(node.why)}</p></div>
      <div class="dialog-section"><small>What happened</small><p>${esc(node.result)}</p></div>
      <div class="dialog-section"><small>What survived</small><p>${esc(node.legacy)}</p></div>`;
    if (typeof recordDialog.showModal === 'function') recordDialog.showModal();
    else recordDialog.setAttribute('open','');
  }

  function bindRecordButtons(root) {
    root.querySelectorAll('[data-record]').forEach(button => {
      if (button.dataset.bound) return;
      button.dataset.bound = '1';
      button.addEventListener('click', () => openRecord(button.dataset.record));
    });
  }

  function locateRecord(id) {
    const cycleId = recordToCycle.get(id);
    const cycleEl = cycleId ? document.getElementById(`cycle-${cycleId}`) : null;
    if (!cycleEl) { openRecord(id); return; }
    cycleEl.scrollIntoView({behavior:'smooth',block:'center'});
    const details = cycleEl.querySelector('.cycle-records');
    if (details) details.open = true;
    setTimeout(() => {
      const row = cycleEl.querySelector(`[data-record="${CSS.escape(id)}"]`);
      if (row) {
        row.classList.add('flash');
        row.scrollIntoView({behavior:'smooth',block:'center'});
        setTimeout(() => row.classList.remove('flash'), 1200);
      }
      openRecord(id);
    }, 300);
  }

  function updateSearch() {
    const q = searchInput.value.trim().toLowerCase();
    if (!q) { searchResults.hidden = true; searchResults.innerHTML = ''; return; }
    const matches = data.nodes.filter(node => `${node.label} ${node.why} ${node.result} ${node.legacy}`.toLowerCase().includes(q)).slice(0,14);
    searchResults.innerHTML = matches.map(node => `<button type="button" data-result="${esc(node.id)}"><strong>${esc(node.label)}</strong><small>${esc(nodeSummary(node))}</small></button>`).join('');
    searchResults.hidden = matches.length === 0;
    searchResults.querySelectorAll('[data-result]').forEach(button => button.addEventListener('click', () => {
      searchInput.value = '';
      searchResults.hidden = true;
      locateRecord(button.dataset.result);
    }));
  }

  function buildNav() {
    cycleNav.innerHTML = cycles.map(cycle => `<button type="button" data-target="cycle-${esc(cycle.id)}">${esc(cycle.number)}</button>`).join('') + '<button type="button" data-target="architecture">A</button>';
    cycleNav.querySelectorAll('[data-target]').forEach(button => button.addEventListener('click', () => {
      const target = document.getElementById(button.dataset.target);
      if (target) target.scrollIntoView({behavior:'smooth',block:'start'});
    }));
  }

  function bindObserver() {
    const buttons = new Map([...cycleNav.querySelectorAll('[data-target]')].map(button => [button.dataset.target,button]));
    const sections = [...cycles.map(cycle => document.getElementById(`cycle-${cycle.id}`)),document.getElementById('architecture')].filter(Boolean);
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a,b) => b.intersectionRatio-a.intersectionRatio)[0];
      if (!visible) return;
      buttons.forEach(button => button.classList.remove('active'));
      const active = buttons.get(visible.target.id);
      if (active) active.classList.add('active');
    }, {rootMargin:'-28% 0px -58% 0px',threshold:[0,.1,.25,.5]});
    sections.forEach(section => observer.observe(section));
  }

  cycleSpine.innerHTML = cycles.map(renderCycle).join('');
  architectureRoot.innerHTML = renderArchitecture();
  buildNav();
  bindObserver();
  document.querySelectorAll('.phase-button').forEach(button => button.addEventListener('click', () => selectPhase(button)));
  bindRecordButtons(document);

  document.querySelectorAll('[data-expert]').forEach(button => button.addEventListener('click', () => {
    const expert = data.architecture.experts.find(item => item.id === button.dataset.expert);
    if (!expert) return;
    document.querySelectorAll('[data-expert].selected').forEach(el => el.classList.remove('selected'));
    button.classList.add('selected');
    document.getElementById('archNote').innerHTML = `<strong>${esc(expert.title)}</strong><span>${esc(expert.body)}</span>`;
  }));

  recordDialog.querySelector('.dialog-close').addEventListener('click', () => recordDialog.close());
  recordDialog.addEventListener('click', event => { if (event.target === recordDialog) recordDialog.close(); });
  searchInput.addEventListener('input', updateSearch);
  searchInput.addEventListener('keydown', event => {
    if (event.key === 'Escape') searchResults.hidden = true;
    if (event.key === 'Enter') {
      const first = searchResults.querySelector('[data-result]');
      if (first) { event.preventDefault(); first.click(); }
    }
  });
  document.addEventListener('click', event => { if (!event.target.closest('.search-box')) searchResults.hidden = true; });
})();
