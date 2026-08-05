import { formatPercent, humanize } from '../flow'
import type { QueryMetadata } from '../types'

type Props = { query: QueryMetadata }

export function EvidencePanel({ query }: Props) {
  const evidence = query.evidence_passport || {}
  const conformal = query.conformal_retrieval_set || {}
  const components = Object.entries(evidence.components || {})
  const reliability = query.empirical_reliability_score

  return (
    <section className="evidence-section">
      <div className="evidence-grid">
        <article className="evidence-card glass-panel applicability-card">
          <div className="card-topline">
            <span className="section-kicker">QUERY APPLICABILITY</span>
            <span className={`tier-badge tier-${evidence.applicability_tier || 'unknown'}`}>{humanize(evidence.applicability_tier)}</span>
          </div>
          <div className="radial-meter" style={{ '--meter': `${Math.round((evidence.applicability_score || 0) * 100)}%` } as React.CSSProperties}>
            <div>
              <strong>{formatPercent(evidence.applicability_score)}</strong>
              <span>representation support</span>
            </div>
          </div>
          <p>{humanize(evidence.recommendation)}</p>
          <small>Diagnostic applicability, not catalytic activity probability.</small>
        </article>

        <article className="evidence-card glass-panel component-card">
          <div className="card-topline">
            <span className="section-kicker">WHY THE QUERY IS SUPPORTED</span>
            <span>{components.length} diagnostics</span>
          </div>
          <div className="component-bars">
            {components.map(([name, value]) => (
              <div className="component-row" key={name}>
                <span>{humanize(name)}</span>
                <div><i style={{ width: `${Math.max(2, value * 100)}%` }} /></div>
                <strong>{formatPercent(value, 0)}</strong>
              </div>
            ))}
          </div>
        </article>

        <article className="evidence-card glass-panel reliability-card">
          <div className="card-topline">
            <span className="section-kicker">EMPIRICAL RANKING RELIABILITY</span>
            <span className="binding-state">{humanize(query.empirical_reliability_binding_status)}</span>
          </div>
          <div className="reliability-value">
            <strong>{reliability == null ? 'N/A' : formatPercent(reliability)}</strong>
            <span>{humanize(query.empirical_reliability_tier)}</span>
          </div>
          <dl>
            <div><dt>Status</dt><dd>{humanize(query.empirical_reliability_status)}</dd></div>
            <div><dt>Recommendation</dt><dd>{humanize(query.reliability_recommendation)}</dd></div>
            <div><dt>Nearest reference</dt><dd>{query.query_nearest_library_id || '—'} · {formatPercent(query.query_nearest_library_similarity)}</dd></div>
          </dl>
          <small>Empirical retrieval evidence, not biochemical success probability.</small>
        </article>

        <article className="evidence-card glass-panel conformal-card">
          <div className="card-topline">
            <span className="section-kicker">CONFORMAL RETRIEVAL SET</span>
            <span className={`binding-state ${conformal.binding_status === 'compatible' ? 'ok' : ''}`}>{humanize(conformal.binding_status)}</span>
          </div>
          <div className="conformal-scale">
            <div className="requested-prefix" style={{ width: `${conformal.set_size ? Math.max(4, ((conformal.requested_top_k || 0) / conformal.set_size) * 100) : 38}%` }}>
              <span>requested Top-{conformal.requested_top_k || 'K'}</span>
            </div>
            <div className="set-prefix" style={{ width: `${Math.max(0, (conformal.set_fraction || 0) * 100)}%` }} />
          </div>
          <div className="conformal-metrics">
            <div><span>Target coverage</span><strong>{formatPercent(conformal.target_coverage)}</strong></div>
            <div><span>Set size</span><strong>{conformal.set_size ?? 'N/A'}</strong></div>
            <div><span>Universe share</span><strong>{formatPercent(conformal.set_fraction)}</strong></div>
            <div><span>Validation coverage</span><strong>{formatPercent(conformal.validation_coverage)}</strong></div>
          </div>
          <p>{humanize(conformal.status)}</p>
          <small>Coverage of at least one known positive under the locked protocol; not per-candidate probability.</small>
        </article>
      </div>
    </section>
  )
}
