import { formatPercent, humanize } from '../flow'
import type { QueryMetadata } from '../types'

type Props = { query: QueryMetadata }

export function EvidencePanel({ query }: Props) {
  const evidence = query.evidence_passport || {}
  const conformal = query.conformal_retrieval_set || {}
  const components = Object.entries(evidence.components || {})
  const reliability = query.empirical_reliability_score
  const universeSize = query.candidate_universe_size || 0
  const setSize = conformal.set_size || 0
  const requestedTopK = conformal.requested_top_k || Number(query.ranking_objective?.replace('top', '')) || 0
  const hasRecallSet = conformal.binding_status === 'compatible' && setSize > 0 && universeSize > 0
  const setPercent = hasRecallSet ? Math.min(100, (setSize / universeSize) * 100) : 0
  const topKPercent = universeSize > 0 ? Math.min(100, (requestedTopK / universeSize) * 100) : 0
  const taxonomyRestricted = query.direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all'

  return (
    <section className="evidence-section" aria-label="How to interpret this search">
      {taxonomyRestricted && (
        <div className="taxonomy-evidence-banner glass-panel">
          <div><span className="section-kicker">ENZYME CANDIDATE UNIVERSE CONSTRAINT</span><strong>{query.enzyme_taxonomy_scope === 'eukaryote' ? 'Eukaryotes only' : 'Prokaryotes only'}</strong></div>
          <div className="taxonomy-contraction"><b>{query.candidate_universe_pre_taxonomy_size?.toLocaleString() || '—'}</b><i>→</i><b>{query.candidate_universe_post_taxonomy_size?.toLocaleString() || query.candidate_universe_size?.toLocaleString() || '—'}</b><small>eligible proteins before model scoring</small></div>
          <p>{query.taxonomy_excluded_count?.toLocaleString() || '—'} proteins were excluded by biological scope, including {query.taxonomy_unknown_count?.toLocaleString() || '—'} locally unresolved records. Unresolved proteins are excluded rather than guessed into a domain. Existing unrestricted reliability and conformal calibrations are not reused.</p>
        </div>
      )}
      <div className="evidence-grid">
        <article className="evidence-card glass-panel applicability-card">
          <div className="card-topline">
            <span className="section-kicker">IS THIS QUERY FAMILIAR TO THE MODEL?</span>
            <span className={`tier-badge tier-${evidence.applicability_tier || 'unknown'}`}>{applicabilityLabel(evidence.applicability_tier)}</span>
          </div>
          <div className="radial-meter" style={{ '--meter': `${Math.round((evidence.applicability_score || 0) * 100)}%` } as React.CSSProperties}>
            <div>
              <strong>{formatPercent(evidence.applicability_score)}</strong>
              <span>support from known data</span>
            </div>
          </div>
          <p>{applicabilityMeaning(evidence.applicability_tier)}</p>
          <small>This measures how similar the query is to cases the system has seen before. It is not the probability that any candidate will work in the laboratory.</small>
        </article>

        <article className="evidence-card glass-panel component-card">
          <div className="card-topline">
            <span className="section-kicker">WHAT SUPPORTS THAT ASSESSMENT?</span>
            <span>{components.length} signals</span>
          </div>
          <div className="component-bars">
            {components.map(([name, value]) => (
              <div className="component-row" key={name}>
                <span>{componentLabel(name)}</span>
                <div><i style={{ width: `${Math.max(2, value * 100)}%` }} /></div>
                <strong>{formatPercent(value, 0)}</strong>
              </div>
            ))}
          </div>
          <small>These signals summarize similarity to reference data, agreement between model runs and how clearly the shortlist separates from lower-ranked candidates.</small>
        </article>

        <article className="evidence-card glass-panel reliability-card">
          <div className="card-topline">
            <span className="section-kicker">HOW STABLE IS THE RANKING?</span>
            <span className={`binding-state ${reliability != null ? 'ok' : ''}`}>{reliability == null ? 'Not estimated' : 'Benchmark estimate'}</span>
          </div>
          <div className="reliability-value">
            <strong>{reliability == null ? 'N/A' : formatPercent(reliability)}</strong>
            <span>{reliabilityTierLabel(query.empirical_reliability_tier)}</span>
          </div>
          <dl>
            <div><dt>What it means</dt><dd>{reliabilityMeaning(query.empirical_reliability_status)}</dd></div>
            <div><dt>Suggested use</dt><dd>{recommendationLabel(query.reliability_recommendation)}</dd></div>
            <div><dt>Closest known query</dt><dd>{query.query_nearest_library_id || '—'} · {formatPercent(query.query_nearest_library_similarity)}</dd></div>
          </dl>
          <small>This is an empirical estimate of ranking stability on comparable benchmark queries, not a biochemical success rate.</small>
        </article>

        <article className="evidence-card glass-panel conformal-card">
          <div className="card-topline">
            <span className="section-kicker">HOW FAR DOWN THE RANKING SHOULD WE LOOK?</span>
            <span className={`binding-state ${hasRecallSet ? 'ok' : ''}`}>{hasRecallSet ? 'Review-depth estimate available' : 'Not available for this search'}</span>
          </div>

          {hasRecallSet ? <>
            <p className="conformal-purpose">
              This <strong>conformal retrieval set</strong> is a benchmark-based review-depth estimate. Across comparable queries, the calibration is designed so that the first <strong>{setSize.toLocaleString()}</strong> of {universeSize.toLocaleString()} ranked candidates include <strong>at least one known positive</strong> about <strong>{formatPercent(conformal.target_coverage, 0)}</strong> of the time.
            </p>
            <div className="conformal-universe-scale" aria-label={`Requested Top-${requestedTopK}; recall-controlled boundary at rank ${setSize} in a universe of ${universeSize}`}>
              <div className="conformal-recall-prefix" style={{ width: `${setPercent}%` }} />
              <div className="conformal-returned-prefix" style={{ width: `${topKPercent}%` }} />
              <div className="conformal-topk-marker" style={{ left: `${topKPercent}%` }}><i /><span>Returned list<br />Top-{requestedTopK}</span></div>
              <div className="conformal-boundary-marker" style={{ left: `${setPercent}%` }}><i /><span>Recall boundary<br />rank {setSize.toLocaleString()}</span></div>
              <b className="conformal-scale-start">rank 1</b>
              <b className="conformal-scale-end">{universeSize.toLocaleString()} candidates</b>
            </div>
            <div className="conformal-legend">
              <span><i className="returned" /> results currently returned</span>
              <span><i className="recall" /> broader prefix suggested by benchmark calibration</span>
            </div>
            <div className="conformal-metrics">
              <div><span>Designed recall target</span><strong>{formatPercent(conformal.target_coverage)}</strong></div>
              <div><span>Review depth</span><strong>Top {setSize.toLocaleString()}</strong></div>
              <div><span>Share of all candidates</span><strong>{formatPercent(conformal.set_fraction)}</strong></div>
              <div><span>Observed on separate test queries</span><strong>{formatPercent(conformal.validation_coverage)}{conformal.validation_n ? ` across ${Math.round(conformal.validation_n)} queries` : ''}</strong></div>
            </div>
            <div className="conformal-meaning">
              <strong>Why this matters</strong>
              <span>{setSize > requestedTopK
                ? `The practical Top-${requestedTopK} list is much smaller than the recall-controlled prefix. That means the model can prioritize candidates, but it cannot support a ${formatPercent(conformal.target_coverage, 0)} recall claim for the short list alone. ${setSizeInterpretation(conformal.set_fraction)}`
                : `The returned Top-${requestedTopK} already reaches the recall-controlled boundary for this route and target.`}</span>
            </div>
            <small>“Conformal” here controls the depth of the ranked list, not the confidence of an individual candidate. This version is calibrated from rank positions: queries using the same route and familiarity group receive the same review depth, and a large score gap in this one query does not automatically shrink the set. The target is marginal across comparable queries under the benchmark assumptions; it is not a guarantee for this specific query or experiment.</small>
          </> : <div className="conformal-unavailable">
            <strong>No recall-controlled depth is reported for this search.</strong>
            <p>{conformalUnavailableReason(conformal.status, query.shot_mode)}</p>
            <small>The ranked candidates are still available. Only the benchmark-based statement about how deep to review is omitted.</small>
          </div>}
        </article>
      </div>
    </section>
  )
}

function componentLabel(name: string) {
  const labels: Record<string, string> = {
    nearest_library_similarity: 'Similarity to known queries',
    similarity_to_known_queries: 'Similarity to known queries',
    query_nearest_library_similarity: 'Similarity to known queries',
    ensemble_top1_vote_fraction: 'Agreement on the top candidate',
    ensemble_top1_consensus: 'Agreement on the top candidate',
    ensemble_top1_rank_stability: 'Top-candidate rank stability',
    ensemble_top1_rank_std: 'Top-candidate rank stability',
    ensemble_topk_jaccard: 'Shortlist overlap between models',
    ensemble_topk_set_stability: 'Shortlist overlap between model runs',
    ensemble_topk_vote_mean: 'Average shortlist agreement',
    ensemble_topk_membership_support: 'Support for candidates staying in the shortlist',
    ensemble_boundary_margin: 'Separation at the shortlist boundary',
    ensemble_boundary_margin_z: 'Separation at the shortlist boundary',
    topk_boundary_separation: 'Separation at the shortlist boundary',
    top1_rank_stability: 'Top-candidate rank stability',
  }
  return labels[name] || humanize(name)
}

function applicabilityLabel(tier: string | null | undefined) {
  const labels: Record<string, string> = {
    reference_library: 'Already represented',
    in_domain: 'Well supported',
    near_domain: 'Related to known data',
    weakly_supported: 'Limited support',
    far_out_of_domain: 'Far from known data',
  }
  return labels[tier || ''] || 'Not assessed'
}

function applicabilityMeaning(tier: string | null | undefined) {
  const meanings: Record<string, string> = {
    reference_library: 'The query is already present in the reference data, so the system can use its stored representation.',
    in_domain: 'The query closely resembles examples used to build and test the system.',
    near_domain: 'The query is not identical to known examples, but it still lies near familiar regions of the data.',
    weakly_supported: 'Only part of the query resembles known examples, so the ranking should be treated as exploratory.',
    far_out_of_domain: 'The query is unlike the available reference data; use the ranking mainly to generate hypotheses.',
  }
  return meanings[tier || ''] || 'The system could not determine how close this query is to its reference data.'
}

function reliabilityTierLabel(tier: string | null | undefined) {
  const labels: Record<string, string> = {
    high: 'more stable',
    moderate: 'moderately stable',
    low: 'less stable',
    unavailable: 'not available',
  }
  return labels[tier || ''] || humanize(tier)
}

function reliabilityMeaning(status: string | null | undefined) {
  if (!status) return 'No route-matched benchmark estimate is available.'
  if (status.includes('validated') || status.includes('deployed')) return 'Comparable benchmark queries were used to estimate how often the leading ranks remain dependable.'
  if (status.includes('not_applicable')) return 'This search type is outside the scope of the available ranking-stability calibration.'
  if (status.includes('incompatible')) return 'The available calibration does not match the route or candidate collection used here.'
  return humanize(status)
}

function recommendationLabel(value: string | null | undefined) {
  const labels: Record<string, string> = {
    proceed_with_standard_shortlist: 'Use the shortlist for normal experimental triage.',
    proceed_with_review: 'Review the shortlist together with evidence and diversity.',
    broaden_candidate_review: 'Inspect a broader set and add orthogonal evidence before experiments.',
    manual_review_required: 'Treat the ranking as exploratory and review supporting evidence.',
  }
  return labels[value || ''] || humanize(value)
}

function setSizeInterpretation(fraction: number | null | undefined) {
  if (fraction == null) return ''
  if (fraction >= 0.6) return 'Because the required prefix covers most of the candidate collection, comparable benchmark positives often appeared deep in the ranking; treat this as a clear warning that high-recall narrowing is weak for this route.'
  if (fraction >= 0.3) return 'The review depth is still substantial, indicating meaningful uncertainty remains beyond the short experimental list.'
  return 'The model narrows the candidate collection relatively well at this recall target, although the estimate is still population-level rather than query-specific.'
}

function conformalUnavailableReason(status: string | null | undefined, shotMode: string | null | undefined) {
  if (shotMode === 'few_shot') return 'Few-shot searches use supplied examples and have not been calibrated with the zero-shot recall protocol.'
  if (status?.includes('current_entity')) return 'The recall calibration was built for new, zero-shot queries rather than entities already present in the reference data.'
  if (status?.includes('masked')) return 'Removing selected candidates changes the ranking population, so the existing recall calibration is not applied.'
  if (status?.includes('taxonomy_restricted')) return 'The enzyme candidate universe was restricted by biological domain, so it no longer matches the unrestricted population used for calibration.'
  if (status?.includes('temporary')) return 'The candidate collection was extended, so it no longer matches the collection used for calibration.'
  if (status?.includes('manual')) return 'A manually selected route is outside the validated calibration contract.'
  if (status === 'disabled') return 'Recall-set calculation was turned off for this search.'
  return 'A route-matched recall calibration is not available for this query configuration.'
}
