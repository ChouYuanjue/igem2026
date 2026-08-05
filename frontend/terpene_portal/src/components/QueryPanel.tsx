import type { QueryForm, RunState } from '../types'

type Props = {
  form: QueryForm
  runState: RunState
  error: string | null
  onChange: (next: QueryForm) => void
  onRun: () => void
  onLoadDemo: (kind: 'r2e' | 'e2r') => void
}

export function QueryPanel({ form, runState, error, onChange, onRun, onLoadDemo }: Props) {
  const isR2E = form.direction === 'reaction_to_enzyme'
  const rawLabel = isR2E ? 'Reaction SMILES' : 'Protein sequence'
  const idLabel = isR2E ? 'Rhea / registry reaction ID' : 'UniProt / registry enzyme ID'

  return (
    <aside className="query-panel glass-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">QUERY COMPOSER</span>
          <h2>Start from a reaction or an enzyme</h2>
        </div>
        <span className={`run-indicator ${runState}`}>{runState}</span>
      </div>

      <div className="segmented direction-switch" role="group" aria-label="Retrieval direction">
        <button
          className={isR2E ? 'active' : ''}
          onClick={() => onChange({ ...form, direction: 'reaction_to_enzyme', entityMode: 'id', queryValue: 'RHEA:54512' })}
        >
          <span>R2E</span>
          Reaction → Enzyme
        </button>
        <button
          className={!isR2E ? 'active' : ''}
          onClick={() => onChange({ ...form, direction: 'enzyme_to_reaction', entityMode: 'id', queryValue: 'A0A023J8Z5' })}
        >
          <span>E2R</span>
          Enzyme → Reaction
        </button>
      </div>

      <label className="field-label">
        Input mode
        <div className="segmented compact">
          <button className={form.entityMode === 'id' ? 'active' : ''} onClick={() => onChange({ ...form, entityMode: 'id' })}>Stable ID</button>
          <button className={form.entityMode === 'raw' ? 'active' : ''} onClick={() => onChange({ ...form, entityMode: 'raw', queryValue: '' })}>{rawLabel}</button>
        </div>
      </label>

      <label className="field-label">
        {form.entityMode === 'id' ? idLabel : rawLabel}
        {form.entityMode === 'raw' ? (
          <textarea
            rows={7}
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

      <div className="query-grid">
        <label className="field-label">
          Ranking budget
          <div className="segmented compact three">
            {([3, 10, 20] as const).map((topK) => (
              <button key={topK} className={form.topK === topK ? 'active' : ''} onClick={() => onChange({ ...form, topK })}>Top-{topK}</button>
            ))}
          </div>
        </label>
        <label className="field-label">
          Conformal mode
          <select value={form.conformalMode} onChange={(event) => onChange({ ...form, conformalMode: event.target.value as QueryForm['conformalMode'] })}>
            <option value="annotate">Annotate</option>
            <option value="expand">Expand set</option>
            <option value="disabled">Disabled</option>
          </select>
        </label>
      </div>

      <label className="field-label">
        Target retrieval coverage
        <div className="segmented compact three">
          {([0.2, 0.1, 0.05] as const).map((alpha) => (
            <button key={alpha} className={form.conformalAlpha === alpha ? 'active' : ''} onClick={() => onChange({ ...form, conformalAlpha: alpha })}>
              {Math.round((1 - alpha) * 100)}%
            </button>
          ))}
        </div>
      </label>

      <div className="query-note">
        <strong>Scientific boundary</strong>
        <span>Scores are ranking evidence, not catalytic activity probabilities.</span>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <button className="run-button" disabled={runState === 'running' || !form.queryValue.trim()} onClick={onRun}>
        <span className="run-pulse" />
        {runState === 'running' ? 'Running production route…' : 'Run model workflow'}
      </button>

      <div className="demo-row">
        <span>Production-backed demos</span>
        <button onClick={() => onLoadDemo('r2e')}>Current R2E</button>
        <button onClick={() => onLoadDemo('e2r')}>External E2R · RRF</button>
      </div>
    </aside>
  )
}
