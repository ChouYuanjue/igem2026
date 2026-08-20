import type { FlowLane, FlowStage, QueryMetadata } from './types'

function routeIsExternal(query: QueryMetadata) {
  return Boolean(query.route_id?.includes('-external-'))
}

function scoreHas(query: QueryMetadata, token: string) {
  return Boolean(query.score_source?.toLowerCase().includes(token))
}

export function buildFlowLanes(query: QueryMetadata): FlowLane[] {
  const lanes: FlowLane[] = []
  lanes.push({
    id: 'primary',
    title: query.direction === 'enzyme_to_reaction' ? 'Main enzyme-to-reaction model' : 'Main reaction-to-enzyme model',
    subtitle: scoreHas(query, 'neighbor') ? 'direct prediction combined with evidence from related proteins' : 'direct ranking from the selected neural model',
    tone: 'primary',
    active: true,
    weight: scoreHas(query, 'top20') && scoreHas(query, 'rrf') ? 0.7 : scoreHas(query, 'top10') && scoreHas(query, 'rrf') ? 0.35 : undefined,
  })
  if (query.secondary_model_directory || query.score_source?.includes('secondary')) {
    lanes.push({ id: 'secondary', title: 'Independent second neural ranking', subtitle: 'a model trained to distinguish difficult alternatives', tone: 'secondary', active: true, weight: 0.65 })
  }
  if (query.auxiliary_score_directory || scoreHas(query, 'dual_kernel')) {
    lanes.push({ id: 'auxiliary', title: 'Similarity-network evidence', subtitle: 'support from protein similarity, reaction similarity and known links', tone: 'auxiliary', active: true, weight: 0.3 })
  }
  if (lanes.length === 1) lanes.push({ id: 'not-executed', title: 'No second ranking needed', subtitle: 'this search uses one validated direct strategy', tone: 'neutral', active: false })
  return lanes
}

export function buildFlowStages(query: QueryMetadata, activeIndex = 99): FlowStage[] {
  const direction = query.direction || 'enzyme_to_reaction'
  const external = routeIsExternal(query)
  const audit = query.input_audit || {}
  const inputStatus = String(audit[direction === 'reaction_to_enzyme' ? 'reaction_input_status' : 'protein_input_status'] || 'input accepted')
  const representation = direction === 'reaction_to_enzyme'
    ? 'the chemical transformation is converted into a reaction fingerprint'
    : 'the amino-acid sequence is converted into learned protein features'
  const fusion = scoreHas(query, 'rrf')
    ? 'two independently ordered lists are combined by rank position'
    : scoreHas(query, 'hybrid')
      ? 'direct prediction is combined with transfer from related proteins'
      : 'one model produces the candidate order directly'
  const recall = query.conformal_retrieval_set
  const definitions = [
    ['INPUT', 'Read the scientific query', `${inputStatus}; accepts a database ID or raw scientific input`, query.query_id || 'query'],
    ['REFERENCE', external ? 'Treat this as a new query' : 'Use an existing reference record', external ? 'the query is not already represented in the original reference collection' : 'a stored representation is available for this entity', external ? 'new query' : 'known query'],
    ['FEATURES', 'Convert the query into model-readable features', representation, direction === 'reaction_to_enzyme' ? 'reaction features' : 'protein features'],
    ['STRATEGY', 'Choose the search strategy', 'direction, known examples, query status and requested list size determine the route', query.route_id || 'selecting route'],
    ['CANDIDATES', direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all' ? 'Load and biologically filter possible enzymes' : 'Load possible answers', direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all'
      ? `load ${query.candidate_universe_pre_taxonomy_size ?? '—'} deployed proteins, then retain only ${query.enzyme_taxonomy_scope === 'eukaryote' ? 'eukaryotic' : 'prokaryotic'} candidates before scoring (${query.candidate_universe_post_taxonomy_size ?? query.candidate_universe_size ?? '—'} remain; unresolved records are excluded rather than guessed)`
      : `search across ${query.candidate_universe_size ?? '—'} registered ${direction === 'reaction_to_enzyme' ? 'proteins' : 'reactions'}`,
      direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all'
        ? `${query.candidate_universe_pre_taxonomy_size ?? '—'} → ${query.candidate_universe_post_taxonomy_size ?? query.candidate_universe_size ?? '—'}`
        : `${query.candidate_universe_size ?? '—'} candidates`],
    ['RANK', 'Score candidate matches', 'only the model paths selected for this query are run', query.score_source || 'ranking method'],
    ['COMBINE', 'Combine evidence when needed', fusion, scoreHas(query, 'rrf') ? 'rank fusion' : 'single ranking'],
    ['ORDER', 'Create the final priority order', 'candidate scores are ordered before interpretation fields are added', query.ranking_objective || 'ranked list'],
    ['INTERPRET', 'Estimate how cautiously to use the result', 'compare the query with known data and measure agreement between model runs', formatPercent(query.evidence_passport?.applicability_score)],
    ['REVIEW DEPTH', 'Estimate how far down the list to review', recall?.set_size ? `Top ${recall.set_size} is the benchmark-based recall-controlled prefix` : 'no route-matched recall-depth estimate is available', recall?.set_size ? `Top ${recall.set_size}` : 'not estimated'],
    ['EXPLAIN', 'Attach candidate-level explanations', 'show supporting signals and warnings without changing the candidate order', 'evidence summary'],
    ['RESULT', 'Prepare candidates for review or experiments', 'inspect the shortlist, download results or open related data records', `${query.conformal_retrieval_set?.requested_top_k || query.ranking_objective || 'Top-K'}`],
  ] as const
  return definitions.map(([eyebrow, title, detail, metric], index) => ({ id: eyebrow.toLowerCase().replaceAll(' ', '-'), index, eyebrow, title, detail, metric, state: index < activeIndex ? 'complete' : index === activeIndex ? 'active' : 'queued' }))
}

export function formatPercent(value: number | null | undefined, digits = 1) {
  return value == null || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`
}

export function compactRouteName(query: QueryMetadata) {
  if (!query.route_id) return 'Search strategy is being selected'
  const direction = query.direction === 'reaction_to_enzyme' ? 'Reaction → enzyme' : 'Enzyme → reaction'
  const scope = query.scope === 'current' ? 'known query' : 'new query'
  const examples = query.shot_mode === 'few_shot' ? 'with known examples' : 'query only'
  const taxonomy = query.direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all'
    ? query.enzyme_taxonomy_scope === 'eukaryote' ? 'eukaryotes only' : 'prokaryotes only'
    : ''
  const depth = query.ranking_objective?.replace('top', 'Top ') || 'selected depth'
  return [direction, scope, examples, taxonomy, depth].filter(Boolean).join(' · ')
}

export function humanize(value: string | null | undefined) {
  if (!value) return '—'
  return value.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}
