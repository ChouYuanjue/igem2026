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
          <span className="section-kicker">RANKED CANDIDATES</span>
          <h2>{response.candidates.length} candidates returned for this search</h2>
          <p>This list is a model-generated priority order for follow-up. Bar lengths compare candidates only within this search; scores from different routes should not be compared directly.</p>
        </div>
        <div className="candidate-summary">
          <span>{response.candidates.filter((candidate) => candidate.evidence_passport?.tier === 'priority_candidate').length} priority</span>
          <span>{response.candidates.filter((candidate) => candidate.is_external_candidate).length} newly added candidates</span>
          <span>{recallSummary(response)}</span>
        </div>
      </div>

      <div className="candidate-layout">
        <div className="candidate-list" role="table" aria-label="Ranked candidates">
          <div className="candidate-table-head" role="row">
            <span>Rank</span><span>Candidate</span><span>Relative model score</span><span>Model agreement</span><span>Supporting evidence</span><span>How it entered the list</span>
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
                <small>{candidate.candidate_taxonomy_scope
                  ? `${candidate.candidate_kingdom || humanize(candidate.candidate_taxonomy_scope)} · ${candidate.is_external_candidate ? 'registered candidate' : 'reference candidate'}`
                  : candidate.is_external_candidate ? 'added beyond the original reference set' : 'present in the reference data'}</small>
              </span>
              <span className="score-cell">
                <i style={{ width: `${Math.max(3, (Math.abs(candidate.score || 0) / maxScore) * 100)}%` }} />
                <em>{formatNumber(candidate.score)}</em>
              </span>
              <span className="consensus-cell">
                <strong>{formatPercent(candidate.ensemble_topk_vote_fraction, 0)}</strong>
                <small>rank variation {formatNumber(candidate.ensemble_rank_std, 2)}</small>
              </span>
              <span><EvidenceBadge candidate={candidate} /></span>
              <span className="source-cell">{selectionSourceLabel(candidate.selection_source)}</span>
            </button>
          ))}
        </div>

        <CandidateDetail candidate={selected || response.candidates[0] || null} recallBoundary={response.query.conformal_retrieval_set?.set_size || null} />
      </div>
    </section>
  )
}

function EvidenceBadge({ candidate }: { candidate: Candidate }) {
  const tier = candidate.evidence_passport?.tier || 'unrated'
  return (
    <span className={`evidence-badge ${tier}`}>
      <i />
      {evidenceTierLabel(tier)}
    </span>
  )
}

function CandidateDetail({ candidate, recallBoundary }: { candidate: Candidate | null; recallBoundary: number | null }) {
  if (!candidate) return <aside className="candidate-detail empty">Select a candidate.</aside>
  const evidence = candidate.evidence_passport || {}
  return (
    <aside className="candidate-detail">
      <div className="candidate-detail-rank">#{candidate.rank}</div>
      <span className="section-kicker">WHY THIS CANDIDATE IS SUPPORTED</span>
      <h3>{candidate.candidate_id}</h3>
      <p>{evidenceTierLabel(evidence.tier)}</p>

      <div className="passport-score">
        <span>Combined evidence score</span>
        <strong>{formatPercent(evidence.score)}</strong>
        <div><i style={{ width: `${Math.max(2, (evidence.score || 0) * 100)}%` }} /></div>
      </div>

      <dl className="detail-metrics">
        <div><dt>Model score on this route</dt><dd>{formatNumber(candidate.score, 6)}</dd></div>
        <div><dt>Agreement across model runs</dt><dd>{formatPercent(candidate.ensemble_topk_vote_fraction)}</dd></div>
        <div><dt>Variation in rank</dt><dd>{formatNumber(candidate.ensemble_rank_std, 3)}</dd></div>
        {candidate.candidate_taxonomy_scope && <div><dt>Biological domain</dt><dd>{candidate.candidate_kingdom || humanize(candidate.candidate_taxonomy_scope)} · {humanize(candidate.candidate_taxonomy_scope)}</dd></div>}
        <div><dt>Inside recall-controlled prefix?</dt><dd>{candidate.conformal_set_member ? `Yes — rank ${candidate.rank} is within the estimated review depth${recallBoundary ? ` of ${recallBoundary}` : ''}.` : 'No, or no recall-depth estimate is available.'}</dd></div>
      </dl>

      {candidate.conformal_set_member && <small className="candidate-recall-note">Being inside the recall-controlled prefix does not make this candidate individually more likely to work; it only describes how deep the overall ranked list is reviewed.</small>}

      <div className="passport-block">
        <span>Sources of support</span>
        <div className="chip-cloud">
          {(evidence.paths || []).map((path) => <em key={path}>{evidencePathLabel(path)}</em>)}
        </div>
      </div>

      <div className="passport-block warnings">
        <span>Warnings</span>
        {(evidence.warnings || []).length ? (
          <ul>{evidence.warnings?.map((warning) => <li key={warning}>{warningLabel(warning)}</li>)}</ul>
        ) : <p>No additional interpretation warnings were attached to this candidate.</p>}
      </div>

      <small>{interpretationLabel(evidence.interpretation)}</small>
    </aside>
  )
}

function evidenceTierLabel(value: string | null | undefined) {
  const labels: Record<string, string> = {
    priority_candidate: 'Strongest combined support',
    supported_candidate: 'Supported for review',
    review_candidate: 'Needs additional review',
    exploratory_candidate: 'Exploratory hypothesis',
    unrated: 'Not yet rated',
  }
  return labels[value || 'unrated'] || humanize(value)
}

function selectionSourceLabel(value: string | null | undefined) {
  const labels: Record<string, string> = {
    primary: 'main model ranking',
    cage_rescue: 'added by structure-based evidence',
    canonical_primary: 'main registered collection',
    uniprot_rescue: 'additional database rescue candidate',
  }
  return labels[value || ''] || humanize(value)
}

function evidencePathLabel(value: string) {
  const labels: Record<string, string> = {
    production_retrieval: 'model ranking',
    empirical_reliability: 'benchmark ranking stability',
    applicability_domain: 'similarity to known queries',
    conformal_retrieval_set: 'recall-controlled review depth',
    ensemble_consensus: 'agreement across model runs',
    registered_candidate: 'registered candidate record',
    external_candidate: 'candidate added beyond the reference set',
  }
  return labels[value] || humanize(value)
}

function warningLabel(value: string) {
  const labels: Record<string, string> = {
    external_candidate_requires_identity_and_input_audit: 'This candidate was added beyond the original reference set; verify its identity and input record before experimental use.',
    low_query_applicability: 'The query is unlike the reference data, so treat this ranking as exploratory.',
    low_ensemble_consensus: 'Different model runs disagree on this candidate’s position.',
    high_rank_variance: 'This candidate’s rank changes substantially between model runs.',
    reliability_not_available: 'No route-matched benchmark stability estimate is available for this candidate.',
  }
  return labels[value] || humanize(value)
}

function interpretationLabel(value: string | null | undefined) {
  if (!value) return 'Use this evidence summary to prioritize review; it is not an activity probability.'
  if (value.includes('not_activity_probability')) return 'This evidence score summarizes support for prioritization and is not the probability of catalytic activity.'
  if (value.includes('ranking_evidence')) return 'This is ranking evidence for follow-up, not a biochemical success probability.'
  return humanize(value)
}

function recallSummary(response: RankingResponse) {
  const recall = response.query.conformal_retrieval_set
  if (!recall?.set_size) return 'Recall depth not estimated'
  const shown = response.candidates.filter((candidate) => candidate.conformal_set_member).length
  if (recall.set_size > response.candidates.length) return `${shown} shown of Top ${recall.set_size} recall-controlled prefix`
  return `Returned list reaches Top ${recall.set_size} recall boundary`
}

function formatNumber(value: number | null | undefined, digits = 4) {
  if (value == null || Number.isNaN(value)) return '—'
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2)
  return value.toFixed(digits)
}
