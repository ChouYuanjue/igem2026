import type { FlowLane, FlowStage, QueryMetadata } from './types'

function routeIsExternal(query: QueryMetadata) {
  return Boolean(query.route_id?.includes('-external-'))
}

function scoreHas(query: QueryMetadata, token: string) {
  return Boolean(query.score_source?.toLowerCase().includes(token))
}

export function buildFlowLanes(query: QueryMetadata): FlowLane[] {
  const direction = query.direction
  const lanes: FlowLane[] = []
  const source = query.score_source || 'production route'

  lanes.push({
    id: 'primary',
    title: direction === 'enzyme_to_reaction' ? 'Primary E2R neural lane' : 'Primary R2E neural lane',
    subtitle: scoreHas(query, 'neighbor') ? 'direct score + nearest-neighbor transfer' : 'production ensemble direct retrieval',
    tone: 'primary',
    active: true,
    weight: scoreHas(query, 'top20') && scoreHas(query, 'rrf') ? 0.7 : scoreHas(query, 'top10') && scoreHas(query, 'rrf') ? 0.35 : undefined,
  })

  if (query.secondary_model_directory || source.includes('secondary')) {
    lanes.push({
      id: 'secondary',
      title: 'Hard-negative secondary lane',
      subtitle: 'independent ranking view trained against difficult negatives',
      tone: 'secondary',
      active: true,
      weight: 0.65,
    })
  }

  if (query.auxiliary_score_directory || scoreHas(query, 'dual_kernel')) {
    lanes.push({
      id: 'auxiliary',
      title: 'Dual-kernel collaborative lane',
      subtitle: 'auxiliary protein–reaction evidence over the same candidate universe',
      tone: 'auxiliary',
      active: true,
      weight: 0.3,
    })
  }

  if (lanes.length === 1) {
    lanes.push({
      id: 'not-executed',
      title: 'No auxiliary lane executed',
      subtitle: 'the selected objective uses the validated direct route only',
      tone: 'neutral',
      active: false,
    })
  }
  return lanes
}

export function buildFlowStages(query: QueryMetadata, activeIndex = 99): FlowStage[] {
  const direction = query.direction || 'enzyme_to_reaction'
  const external = routeIsExternal(query)
  const audit = query.input_audit || {}
  const inputStatus = String(
    audit[direction === 'reaction_to_enzyme' ? 'reaction_input_status' : 'protein_input_status'] || 'validated',
  )
  const representation = direction === 'reaction_to_enzyme'
    ? String(audit.reaction_input_drfp_status || 'DRFP multi-view representation')
    : 'ESM-C 1152-dimensional protein representation'
  const fusion = scoreHas(query, 'rrf') ? 'reciprocal-rank fusion' : scoreHas(query, 'hybrid') ? 'direct + neighbor hybrid' : 'direct production ranking'
  const applicability = query.evidence_passport?.applicability_tier || 'pending'
  const conformal = query.conformal_retrieval_set

  const definitions = [
    ['INPUT', 'Parse and validate', `${inputStatus}; stable ID or raw scientific input accepted`, query.query_id || 'query'],
    ['IDENTITY', external ? 'External open-world query' : 'Current library query', external ? 'query is outside the current reference library' : 'query resolved to a precomputed reference entity', external ? 'external' : 'current'],
    ['REPRESENT', 'Scientific representation', representation, direction === 'reaction_to_enzyme' ? 'reaction vector' : 'protein vector'],
    ['ROUTE', 'Automatic objective router', query.route_id || 'route pending', query.ranking_objective || 'Top-K'],
    ['UNIVERSE', 'Candidate universe assembly', query.candidate_universe_version || 'candidate registry', `${query.candidate_universe_size ?? '—'} candidates`],
    ['RETRIEVE', 'Parallel retrieval lanes', 'only lanes actually executed by the selected route are illuminated', query.score_source || 'score source'],
    ['FUSE', 'Rank fusion and constraints', fusion, scoreHas(query, 'rrf') ? 'RRF' : 'single route'],
    ['RANK', 'Locked production ranking', 'raw route score and rank are preserved before evidence annotation', query.ranking_objective || 'rank'],
    ['TRUST', 'Reliability and applicability', `${query.empirical_reliability_tier || 'uncalibrated'} reliability; ${applicability} applicability`, formatPercent(query.evidence_passport?.applicability_score)],
    ['SET', 'Conformal retrieval set', conformal?.status || 'not available for this query type', conformal?.set_size ? `${conformal.set_size} candidates` : 'annotate only'],
    ['EVIDENCE', 'Candidate Evidence Passports', 'evidence paths and warnings are appended without changing production rank', query.evidence_passport?.version || 'passport'],
    ['OUTPUT', 'Decision handoff', 'review candidates, export results, or open the database atlas for known records', `${query.conformal_retrieval_set?.requested_top_k || query.ranking_objective || 'Top-K'}`],
  ] as const

  return definitions.map(([eyebrow, title, detail, metric], index) => ({
    id: eyebrow.toLowerCase(),
    index,
    eyebrow,
    title,
    detail,
    metric,
    state: index < activeIndex ? 'complete' : index === activeIndex ? 'active' : 'queued',
  }))
}

export function formatPercent(value: number | null | undefined, digits = 1) {
  return value == null || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`
}

export function compactRouteName(query: QueryMetadata) {
  if (!query.route_id) return 'Route pending'
  return query.route_id
    .replace(/^e2r-/, 'E2R · ')
    .replace(/^r2e-/, 'R2E · ')
    .replaceAll('-', ' ')
    .replace(/\bv1\b$/, 'v1')
}

export function humanize(value: string | null | undefined) {
  if (!value) return '—'
  return value.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}
