import { parseIdentifierList } from '../api'
import type { QueryForm, RunState, TaxonomyScopeSummary } from '../types'

type PresetKind = 'r2e-zero' | 'e2r-zero' | 'r2e-few' | 'e2r-few'

type Props = {
  form: QueryForm
  runState: RunState
  error: string | null
  onChange: (next: QueryForm) => void
  onRun: () => void
  onLoadPreset: (kind: PresetKind) => void
  taxonomySummary: TaxonomyScopeSummary | null
}

export function QueryPanel({ form, runState, error, onChange, onRun, onLoadPreset, taxonomySummary }: Props) {
  const isR2E = form.direction === 'reaction_to_enzyme'
  const isFewShot = form.shotMode === 'few_shot'
  const rawLabel = isR2E ? 'Reaction SMILES' : 'Protein sequence'
  const idLabel = isR2E ? 'Rhea or project reaction ID' : 'UniProt or project protein ID'
  const seedIds = parseIdentifierList(form.seedIdsText)
  const maskIds = parseIdentifierList(form.maskIdsText)
  const canRun = Boolean(form.queryValue.trim()) && (!isFewShot || seedIds.length > 0)

  const changeDirection = (direction: QueryForm['direction']) => {
    const nextR2E = direction === 'reaction_to_enzyme'
    onChange({
      ...form,
      direction,
      entityMode: 'id',
      queryValue: nextR2E ? 'RHEA:54512' : 'A0A023J8Z5',
      seedIdsText: '',
      maskIdsText: '',
      enzymeTaxonomyScope: 'all',
    })
  }

  const changeShotMode = (shotMode: QueryForm['shotMode']) => {
    onChange({
      ...form,
      shotMode,
      seedIdsText: shotMode === 'few_shot' ? form.seedIdsText : '',
      conformalMode: shotMode === 'few_shot' ? 'disabled' : form.conformalMode,
    })
  }

  const changeTaxonomyScope = (enzymeTaxonomyScope: QueryForm['enzymeTaxonomyScope']) => {
    onChange({
      ...form,
      enzymeTaxonomyScope,
      conformalMode: enzymeTaxonomyScope === 'all' || isFewShot ? form.conformalMode : 'disabled',
    })
  }

  return (
    <aside className="query-panel glass-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">BUILD A SEARCH</span>
          <h2>What would you like the system to predict?</h2>
        </div>
        <span className={`run-indicator ${runState}`}>{runState}</span>
      </div>

      <div className="composer-step">
        <span className="composer-step-label">01 · Scientific question</span>
        <div className="segmented direction-switch" role="group" aria-label="Choose the scientific prediction direction">
          <button className={isR2E ? 'active' : ''} onClick={() => changeDirection('reaction_to_enzyme')}>
            <span>R2E</span>
            Find enzymes for a reaction
          </button>
          <button className={!isR2E ? 'active' : ''} onClick={() => changeDirection('enzyme_to_reaction')}>
            <span>E2R</span>
            Find reactions for an enzyme
          </button>
        </div>
      </div>

      <div className="composer-step protocol-step">
        <span className="composer-step-label">02 · Do you have known positive examples?</span>
        <div className="protocol-switch" role="group" aria-label="Choose whether to use known positive examples">
          <button className={!isFewShot ? 'active zero' : ''} onClick={() => changeShotMode('zero_shot')}>
            <i className="protocol-icon zero"><span /></i>
            <span><strong>Query only</strong><small>Zero-shot: no positive examples supplied</small></span>
          </button>
          <button className={isFewShot ? 'active few' : ''} onClick={() => changeShotMode('few_shot')}>
            <i className="protocol-icon few"><span /><span /><span /></i>
            <span><strong>Use examples</strong><small>Few-shot: guide the search with known positives</small></span>
          </button>
        </div>
        <div className={`protocol-explainer ${isFewShot ? 'few' : 'zero'}`}>
          <strong>{isFewShot ? 'Example-guided search' : 'Query-only search'}</strong>
          <span>{isFewShot
            ? isR2E
              ? 'The system looks for enzymes whose learned sequence representation resembles the supplied working catalysts. The supplied examples are not returned as new candidates.'
              : 'The system looks for reactions whose learned representation resembles the supplied known reactions. The supplied examples are not returned as new candidates.'
            : 'The system chooses a route from the query type, whether it is already represented in the reference data and how many results you request. New queries may also receive benchmark-based uncertainty estimates.'}</span>
        </div>
      </div>

      {isR2E && (
        <div className="composer-step taxonomy-scope-step">
          <span className="composer-step-label">03 · Which enzyme sources may participate?</span>
          <div className="taxonomy-scope-switch" role="group" aria-label="Choose the enzyme candidate taxonomy scope">
            {([
              ['all', 'All enzymes', taxonomySummary?.total, 'Use the full deployed candidate universe.'],
              ['eukaryote', 'Eukaryotes only', taxonomySummary?.scope_counts?.eukaryote, 'Plants, fungi, animals and amoebozoa only.'],
              ['prokaryote', 'Prokaryotes only', taxonomySummary?.scope_counts?.prokaryote, 'Bacteria, archaea and cyanobacteria only.'],
            ] as const).map(([scope, title, count, detail]) => (
              <button
                key={scope}
                className={form.enzymeTaxonomyScope === scope ? `active ${scope}` : scope}
                onClick={() => changeTaxonomyScope(scope)}
              >
                <i className={`taxonomy-icon ${scope}`}><span /><span /><span /></i>
                <span>
                  <strong>{title}</strong>
                  <small>{count != null ? `${Number(count).toLocaleString()} candidates · ` : ''}{detail}</small>
                </span>
              </button>
            ))}
          </div>
          <div className={`taxonomy-scope-note ${form.enzymeTaxonomyScope}`}>
            <strong>{form.enzymeTaxonomyScope === 'all' ? 'No taxonomy restriction' : `${form.enzymeTaxonomyScope === 'eukaryote' ? 'Eukaryotic' : 'Prokaryotic'} candidate filter`}</strong>
            <span>{form.enzymeTaxonomyScope === 'all'
              ? 'All 2,085 deployed enzyme candidates remain eligible, including records whose taxonomy is not yet resolved locally.'
              : `The filter is applied before model scoring, not after ranking. ${taxonomySummary?.scope_counts?.unknown?.toLocaleString() ?? 559} locally unresolved proteins and all candidates outside the selected biological domain are excluded rather than guessed.`}</span>
          </div>
        </div>
      )}

      <div className="composer-step">
        <span className="composer-step-label">{isR2E ? '04' : '03'} · Enter the reaction or enzyme</span>
        <label className="field-label compact-label">
          Input mode
          <div className="segmented compact">
            <button className={form.entityMode === 'id' ? 'active' : ''} onClick={() => onChange({ ...form, entityMode: 'id' })}>Database ID</button>
            <button className={form.entityMode === 'raw' ? 'active' : ''} onClick={() => onChange({ ...form, entityMode: 'raw', queryValue: '' })}>{rawLabel}</button>
          </div>
        </label>

        <label className="field-label">
          {form.entityMode === 'id' ? idLabel : rawLabel}
          {form.entityMode === 'raw' ? (
            <textarea
              rows={6}
              value={form.queryValue}
              onChange={(event) => onChange({ ...form, queryValue: event.target.value })}
              placeholder={isR2E ? 'CCO>>CC=O' : 'MSTNPKPQRKTKRNTNRRPQDVKFPGG...'}
            />
          ) : (
            <input
              value={form.queryValue}
              onChange={(event) => onChange({ ...form, queryValue: event.target.value })}
              placeholder={isR2E ? 'RHEA:54512' : 'A0A023J8Z5'}
            />
          )}
        </label>
      </div>

      {isFewShot && (
        <div className="composer-step seed-input-step">
          <span className="composer-step-label">{isR2E ? '05' : '04'} · Add known positive examples</span>
          <label className="field-label">
            {isR2E ? 'Known working enzyme IDs' : 'Known reaction IDs'}
            <textarea
              rows={4}
              value={form.seedIdsText}
              onChange={(event) => onChange({ ...form, seedIdsText: event.target.value })}
              placeholder={isR2E ? 'A0A075FBG7\nQ9ZSY2' : 'RHEA:54512\nRHEA:31807'}
            />
          </label>
          <div className="seed-token-preview">
            <span>{seedIds.length} positive example{seedIds.length === 1 ? '' : 's'}</span>
            {seedIds.slice(0, 5).map((id) => <em key={id}>{id}</em>)}
            {seedIds.length > 5 && <em>+{seedIds.length - 5}</em>}
          </div>
          <div className="seed-boundary-note">
            Example-guided searches answer a different question from query-only searches, so the query-only benchmark uncertainty estimates are not applied.
          </div>
        </div>
      )}

      {!isR2E && (
        <details className="mask-only-details" open={Boolean(form.maskIdsText.trim())}>
          <summary>Exclude reactions from the results <span>hide them without treating them as positive examples</span></summary>
          <label className="field-label">
            Reaction IDs to hide from the returned list
            <textarea
              rows={3}
              value={form.maskIdsText}
              onChange={(event) => onChange({ ...form, maskIdsText: event.target.value })}
              placeholder="RHEA:12345, RHEA:67890"
            />
          </label>
          <small>{maskIds.length} reaction ID{maskIds.length === 1 ? '' : 's'} will be hidden. Hidden reactions do not act as positive examples and do not change the underlying scoring method.</small>
        </details>
      )}

      <div className="query-grid">
        <label className="field-label">
          Number of results to return
          <div className="segmented compact three">
            {([3, 10, 20] as const).map((topK) => (
              <button key={topK} className={form.topK === topK ? 'active' : ''} onClick={() => onChange({ ...form, topK })}>Top-{topK}</button>
            ))}
          </div>
        </label>
        <label className={`field-label ${isFewShot || (isR2E && form.enzymeTaxonomyScope !== 'all') ? 'disabled-field' : ''}`}>
          Recall-set display
          <select
            disabled={isFewShot || (isR2E && form.enzymeTaxonomyScope !== 'all')}
            value={isFewShot || (isR2E && form.enzymeTaxonomyScope !== 'all') ? 'disabled' : form.conformalMode}
            onChange={(event) => onChange({ ...form, conformalMode: event.target.value as QueryForm['conformalMode'] })}
          >
            <option value="annotate">Show the estimated review depth</option>
            <option value="expand">Return the full recall-controlled prefix</option>
            <option value="disabled">Do not calculate</option>
          </select>
        </label>
      </div>

      {!isFewShot && !(isR2E && form.enzymeTaxonomyScope !== 'all') && (
        <label className="field-label">
          Desired benchmark recall target
          <div className="segmented compact three">
            {([0.2, 0.1, 0.05] as const).map((alpha) => (
              <button key={alpha} className={form.conformalAlpha === alpha ? 'active' : ''} onClick={() => onChange({ ...form, conformalAlpha: alpha })}>
                {Math.round((1 - alpha) * 100)}%
              </button>
            ))}
          </div>
        </label>
      )}

      {!isFewShot && !(isR2E && form.enzymeTaxonomyScope !== 'all') && form.conformalMode !== 'disabled' && (
        <div className="seed-boundary-note">
          This does not change candidate scores. It estimates how far down the ranked list you would need to review to reduce the chance of missing every known positive on comparable benchmark queries.
        </div>
      )}

      {isR2E && form.enzymeTaxonomyScope !== 'all' && (
        <div className="seed-boundary-note taxonomy-calibration-note">
          Taxonomy-restricted rankings use the same trained models but a different candidate universe. Existing unrestricted reliability and conformal recall calibrations are therefore intentionally not applied.
        </div>
      )}

      <div className="query-note">
        <strong>How to interpret the output</strong>
        <span>The system ranks hypotheses for follow-up experiments. A high rank or evidence score does not mean a candidate has that probability of being active.</span>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <button className="run-button" disabled={runState === 'running' || !canRun} onClick={onRun}>
        <span className="run-pulse" />
        {runState === 'running' ? 'Running the selected search strategy…' : `Run ${isFewShot ? 'example-guided' : 'query-only'} search`}
      </button>

      <div className="demo-row route-presets">
        <span>Try an example</span>
        <button onClick={() => onLoadPreset('r2e-zero')}>Reaction → enzyme · query only</button>
        <button onClick={() => onLoadPreset('e2r-zero')}>Enzyme → reaction · query only</button>
        <button onClick={() => onLoadPreset('r2e-few')}>Reaction → enzyme · with examples</button>
        <button onClick={() => onLoadPreset('e2r-few')}>Enzyme → reaction · with examples</button>
      </div>
    </aside>
  )
}
