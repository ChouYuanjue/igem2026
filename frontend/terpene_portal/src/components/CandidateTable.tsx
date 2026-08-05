import { formatPercent, humanize } from '../flow'
import type { Candidate, RankingResponse } from '../types'

type Props = {
  response: RankingResponse
  selected: Candidate | null
  onSelect: (candidate: Candidate) => void
}

export function CandidateTable({ response, selected, onSelect }: Props) {
  const maxScore = Math.max(...response.candidates.map((candidate) => Math.abs(candidate.score || 0)), 0.00001)

  return (
    <section className="candidate-section glass-panel">
      <div className="candidate-header">
        <div>
          <span className="section-kicker">LOCKED PRODUCTION RANKING</span>
          <h2>{response.candidates.length} returned candidates</h2>
          <p>Bars are normalized only within this query. Raw score scales are not compared across routes.</p>
        </div>
        <div className="candidate-summary">
          <span>{response.candidates.filter((candidate) => candidate.evidence_passport?.tier === 'priority_candidate').length} priority</span>
          <span>{response.candidates.filter((candidate) => candidate.is_external_candidate).length} external</span>
          <span>{response.candidates.filter((candidate) => candidate.conformal_set_member).length} set members shown</span>
        </div>
      </div>

      <div className="candidate-layout">
        <div className="candidate-list" role="table" aria-label="Ranked candidates">
          <div className="candidate-table-head" role="row">
            <span>Rank</span><span>Candidate</span><span>Query-relative score</span><span>Consensus</span><span>Evidence</span><span>Source</span>
          </div>
          {response.candidates.map((candidate) => (
            <button
              key={`${candidate.rank}-${candidate.candidate_id}`}
              className={`candidate-row ${selected?.candidate_id === candidate.candidate_id ? 'selected' : ''}`}
              onClick={() => onSelect(candidate)}
              role="row"
            >
              <span className="rank-cell">#{candidate.rank}</span>
              <span className="candidate-id-cell">
                <strong>{candidate.candidate_id}</strong>
                <small>{candidate.is_external_candidate ? 'external candidate' : 'library candidate'}</small>
              </span>
              <span className="score-cell">
                <i style={{ width: `${Math.max(3, (Math.abs(candidate.score || 0) / maxScore) * 100)}%` }} />
                <em>{formatNumber(candidate.score)}</em>
              </span>
              <span className="consensus-cell">
                <strong>{formatPercent(candidate.ensemble_topk_vote_fraction, 0)}</strong>
                <small>rank σ {formatNumber(candidate.ensemble_rank_std, 2)}</small>
              </span>
              <span><EvidenceBadge candidate={candidate} /></span>
              <span className="source-cell">{humanize(candidate.selection_source)}</span>
            </button>
          ))}
        </div>

        <CandidateDetail candidate={selected || response.candidates[0] || null} />
      </div>
    </section>
  )
}

function EvidenceBadge({ candidate }: { candidate: Candidate }) {
  const tier = candidate.evidence_passport?.tier || 'unrated'
  return (
    <span className={`evidence-badge ${tier}`}>
      <i />
      {humanize(tier)}
    </span>
  )
}

function CandidateDetail({ candidate }: { candidate: Candidate | null }) {
  if (!candidate) return <aside className="candidate-detail empty">Select a candidate.</aside>
  const evidence = candidate.evidence_passport || {}
  return (
    <aside className="candidate-detail">
      <div className="candidate-detail-rank">#{candidate.rank}</div>
      <span className="section-kicker">CANDIDATE EVIDENCE PASSPORT</span>
      <h3>{candidate.candidate_id}</h3>
      <p>{humanize(evidence.tier)}</p>

      <div className="passport-score">
        <span>Evidence strength</span>
        <strong>{formatPercent(evidence.score)}</strong>
        <div><i style={{ width: `${Math.max(2, (evidence.score || 0) * 100)}%` }} /></div>
      </div>

      <dl className="detail-metrics">
        <div><dt>Raw route score</dt><dd>{formatNumber(candidate.score, 6)}</dd></div>
        <div><dt>Top-K vote</dt><dd>{formatPercent(candidate.ensemble_topk_vote_fraction)}</dd></div>
        <div><dt>Rank dispersion</dt><dd>{formatNumber(candidate.ensemble_rank_std, 3)}</dd></div>
        <div><dt>Conformal member</dt><dd>{candidate.conformal_set_member ? 'yes' : 'not in returned annotation'}</dd></div>
      </dl>

      <div className="passport-block">
        <span>Evidence paths</span>
        <div className="chip-cloud">
          {(evidence.paths || []).map((path) => <em key={path}>{humanize(path)}</em>)}
        </div>
      </div>

      <div className="passport-block warnings">
        <span>Warnings</span>
        {(evidence.warnings || []).length ? (
          <ul>{evidence.warnings?.map((warning) => <li key={warning}>{humanize(warning)}</li>)}</ul>
        ) : <p>No evidence-layer warnings for this returned candidate.</p>}
      </div>

      <small>{humanize(evidence.interpretation)}</small>
    </aside>
  )
}

function formatNumber(value: number | null | undefined, digits = 4) {
  if (value == null || Number.isNaN(value)) return '—'
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2)
  return value.toFixed(digits)
}
