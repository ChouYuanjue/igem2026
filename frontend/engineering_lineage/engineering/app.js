(() => {
  const data = window.LINEAGE_DATA;
  if (!data || !data.story) throw new Error('Engineering story data is not loaded');

  const story = data.story;
  const byId = new Map(data.nodes.map(node => [node.id, node]));
  const storyRoot = document.getElementById('storyRoot');
  const architectureRoot = document.getElementById('architectureRoot');
  const chapterLinks = document.getElementById('chapterLinks');
  const searchInput = document.getElementById('storySearch');
  const searchResults = document.getElementById('searchResults');

  const statusColors = {
    root:'#242620',keep:'#315f49',turn:'#a1712c',local:'#536f80',reject:'#a14d3e',historical:'#73587b',considered:'#85867f'
  };
  const statusLabels = {
    root:'origin',keep:'kept',turn:'turn',local:'local',reject:'rejected',historical:'historical',considered:'considered'
  };

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>'"]/g, char => ({
      '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'
    }[char]));
  }

  function nodeSummary(node) {
    const result = String(node.result || '').trim();
    const generic = /^(Rejected|Historical|Implemented|Insufficient|No improvement|No gain)\.?$/i.test(result);
    return generic ? node.why : result;
  }

  function recordClass(node, evidenceIds) {
    const classes = ['record', node.kind];
    if (evidenceIds && evidenceIds.includes(node.id)) classes.push('evidence');
    return classes.join(' ');
  }

  function renderRecord(nodeId, evidenceIds) {
    const node = byId.get(nodeId);
    if (!node) return '';
    const color = statusColors[node.status] || '#85867f';
    return `
      <div class="${recordClass(node, evidenceIds)}" data-record-id="${esc(node.id)}">
        <button class="record-toggle" type="button" aria-expanded="false">
          <span class="status-dot" style="--status:${color}"></span>
          <span class="record-copy">
            <strong>${esc(node.label)}</strong>
            <small>${esc(nodeSummary(node))}</small>
          </span>
          <span class="record-status">${esc(statusLabels[node.status] || node.status)}</span>
        </button>
        <div class="record-detail">
          <b>Why:</b> ${esc(node.why)}<br />
          <b>What survived:</b> ${esc(node.legacy)}
        </div>
      </div>`;
  }

  function renderEvidenceBand(chapter) {
    const ids = chapter.evidenceGroup ? chapter.evidenceGroup.recordIds : chapter.evidenceIds;
    if (!ids || !ids.length) return '';
    const title = chapter.evidenceGroup ? chapter.evidenceGroup.title : 'Evidence that changed the decision';
    return `
      <section class="evidence-band">
        <h4>${esc(title)}</h4>
        <div class="evidence-grid">
          ${ids.map(id => {
            const node = byId.get(id);
            if (!node) return '';
            return `<article class="evidence-card" data-record-id="${esc(id)}"><strong>${esc(node.label)}</strong><p>${esc(node.result)}</p></article>`;
          }).join('')}
        </div>
      </section>`;
  }

  function renderTrack(track) {
    const cssClass = track.id === 'wetlab' ? 'parallel-track wetlab' : 'parallel-track';
    return `
      <aside class="${cssClass}" id="track-${esc(track.id)}">
        <div class="parallel-track-head">
          <div><span class="track-label">${esc(track.label)}</span><h4>${esc(track.title)}</h4></div>
          <p>${esc(track.summary)}</p>
        </div>
        <div class="track-steps">
          ${track.recordIds.map(id => {
            const node = byId.get(id);
            return node ? `<article class="track-step" data-record-id="${esc(id)}"><strong>${esc(node.label)}</strong><small>${esc(nodeSummary(node))}</small></article>` : '';
          }).join('')}
        </div>
      </aside>`;
  }

  function renderChapter(chapter) {
    const tracks = story.tracks.filter(track => track.atChapter === chapter.id);
    return `
      <section class="chapter ${esc(chapter.id)}" id="chapter-${esc(chapter.id)}" data-chapter="${esc(chapter.id)}">
        <aside class="chapter-rail">
          <span class="chapter-number">${esc(chapter.number)}</span>
          <span class="chapter-eyebrow">${esc(chapter.eyebrow)}</span>
          <h2>${esc(chapter.title)}</h2>
          <p class="rail-summary">The page gives visual priority to the decision. The complete experiment record stays attached below it.</p>
        </aside>

        <div class="chapter-body">
          <div class="narrative-grid">
            <article class="narrative-cell"><span>Problem</span><p>${esc(chapter.problem)}</p></article>
            <article class="narrative-cell"><span>What we learned</span><p>${esc(chapter.learn)}</p></article>
            <article class="narrative-cell decision"><span>Decision</span><p>${esc(chapter.decision)}</p></article>
          </div>

          <div class="decision-path-wrap">
            <div class="decision-path-title">The decision path</div>
            <div class="decision-path">
              ${chapter.pathIds.map(id => {
                const node = byId.get(id);
                if (!node) return '';
                return `<article class="path-step" data-record-id="${esc(id)}"><strong>${esc(node.label)}</strong><small>${esc(nodeSummary(node))}</small></article>`;
              }).join('')}
            </div>
          </div>

          <section class="research-ledger">
            <div class="ledger-title-row">
              <h3>Research ledger</h3>
              <p>Every listed route remains visible. Select one row only when you want its motivation and surviving lesson; the chapter story never depends on opening it.</p>
            </div>
            <div class="research-grid">
              ${chapter.groups.map(group => `
                <section class="research-group">
                  <header class="research-group-head">
                    <h4>${esc(group.title)}</h4>
                    <p>${esc(group.summary)}</p>
                  </header>
                  <div class="record-list">${group.recordIds.map(id => renderRecord(id, chapter.evidenceIds)).join('')}</div>
                </section>`).join('')}
            </div>
          </section>

          ${renderEvidenceBand(chapter)}
          ${tracks.map(renderTrack).join('')}

          <div class="chapter-conclusion">
            <span>What changed</span>
            <p>${esc(chapter.decision)}</p>
          </div>
        </div>
      </section>`;
  }

  function renderArchitecture() {
    const arch = story.architecture;
    return `
      <div class="architecture-wrap">
        <div class="architecture-head">
          <div><span class="section-kicker">Current system · composition, not chronology</span><h2>${esc(arch.title)}</h2></div>
          <p>${esc(arch.subtitle)} This is where expert families, gates and correction rules belong: inside BRIDGE.</p>
        </div>

        <div class="architecture-flow">
          <article class="arch-card accent">
            <span>Base order</span><h3>${esc(arch.base.title)}</h3><p>${esc(arch.base.body)}</p>
          </article>
          <div class="arch-arrow">→</div>
          <article class="arch-card">
            <span>Control</span><h3>${esc(arch.control.title)}</h3><p>${esc(arch.control.body)}</p>
          </article>
          <div class="arch-arrow">→</div>
          <div class="expert-grid">
            ${arch.experts.map(expert => `
              <article class="expert-card">
                <h4>${esc(expert.title)}</h4>
                <ul>${expert.members.map(member => `<li>${esc(member)}</li>`).join('')}</ul>
                <p>${esc(expert.body)}</p>
              </article>`).join('')}
          </div>
          <div class="arch-arrow">→</div>
          <article class="arch-card">
            <span>Ranking action</span><h3>${esc(arch.correction.title)}</h3><p>${esc(arch.correction.body)}</p>
          </article>
        </div>

        <div class="formula">${esc(arch.formula)}</div>

        <div class="architecture-evidence">
          ${arch.evidence.map(metric => `<article class="metric"><span>${esc(metric.label)}</span><strong>${esc(metric.value)}</strong></article>`).join('')}
        </div>

        <div class="policy-notes">
          ${arch.policyNotes.map(note => `<div class="policy-note">${esc(note)}</div>`).join('')}
        </div>
      </div>`;
  }

  function renderNavigation() {
    chapterLinks.innerHTML = story.chapters.map(chapter => `
      <button class="chapter-link" type="button" data-target="chapter-${esc(chapter.id)}">${esc(chapter.number)} ${esc(chapter.id === 'closed' ? 'EnzymeCAGE' : chapter.id === 'open' ? 'Open retrieval' : chapter.id === 'broad' ? 'Broad → BiME' : chapter.id === 'bime' ? 'BiME' : chapter.id === 'fibre' ? 'FIBRE' : 'BRIDGE')}</button>`).join('') +
      '<button class="chapter-link" type="button" data-target="architecture">Architecture</button>';
    chapterLinks.querySelectorAll('[data-target]').forEach(button => {
      button.addEventListener('click', () => { const target = document.getElementById(button.dataset.target); if (target) target.scrollIntoView({behavior:'smooth',block:'start'}); });
    });
  }

  function bindRecordToggles() {
    document.querySelectorAll('.record-toggle').forEach(button => {
      button.addEventListener('click', () => {
        const record = button.closest('.record');
        const open = !record.classList.contains('open');
        record.classList.toggle('open', open);
        button.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
    });
  }

  function searchableText(node) {
    return `${node.label} ${node.why} ${node.result} ${node.legacy} ${data.families[node.family] || ''}`.toLowerCase();
  }

  function locateRecord(id) {
    const candidates = [...document.querySelectorAll(`[data-record-id="${CSS.escape(id)}"]`)];
    const target = candidates[0];
    if (!target) return;
    target.scrollIntoView({behavior:'smooth',block:'center'});
    target.classList.remove('flash');
    void target.offsetWidth;
    target.classList.add('flash');
    if (target.classList.contains('record')) {
      target.classList.add('open');
      const toggle = target.querySelector('.record-toggle'); if (toggle) toggle.setAttribute('aria-expanded','true');
    }
  }

  function updateSearch() {
    const query = searchInput.value.trim().toLowerCase();
    if (!query) {
      searchResults.hidden = true;
      searchResults.innerHTML = '';
      return;
    }
    const matches = data.nodes.filter(node => searchableText(node).includes(query)).slice(0,16);
    searchResults.innerHTML = matches.map(node => `
      <button type="button" data-result="${esc(node.id)}"><strong>${esc(node.label)}</strong><small>${esc(nodeSummary(node))}</small></button>`).join('');
    searchResults.hidden = matches.length === 0;
    searchResults.querySelectorAll('[data-result]').forEach(button => {
      button.addEventListener('click', () => {
        const id = button.dataset.result;
        searchInput.value = '';
        searchResults.hidden = true;
        locateRecord(id);
      });
    });
  }

  function bindNavigationObserver() {
    const links = new Map([...chapterLinks.querySelectorAll('[data-target]')].map(button => [button.dataset.target, button]));
    const sections = [...story.chapters.map(chapter => document.getElementById(`chapter-${chapter.id}`)), document.getElementById('architecture')].filter(Boolean);
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a,b) => b.intersectionRatio-a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach(button => button.classList.remove('active'));
      const active = links.get(visible.target.id); if (active) active.classList.add('active');
    }, {rootMargin:'-28% 0px -58% 0px', threshold:[0,.1,.25,.5]});
    sections.forEach(section => observer.observe(section));
  }

  storyRoot.innerHTML = story.chapters.map(renderChapter).join('');
  architectureRoot.innerHTML = renderArchitecture();
  renderNavigation();
  bindRecordToggles();
  bindNavigationObserver();

  searchInput.addEventListener('input', updateSearch);
  searchInput.addEventListener('keydown', event => {
    if (event.key === 'Escape') searchResults.hidden = true;
    if (event.key === 'Enter') {
      const first = searchResults.querySelector('[data-result]');
      if (first) { event.preventDefault(); first.click(); }
    }
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.search-box')) searchResults.hidden = true;
  });
})();
