import type { QueryMetadata, RouteCatalogEntry } from './types'

export function queryRouteId(query: QueryMetadata): string {
  return query.route_id || (query.direction === 'reaction_to_enzyme'
    ? 'r2e-route-pending'
    : 'e2r-route-pending')
}

export function queryRouteMeta(query: QueryMetadata, shown?: number): string {
  const parts = [
    directionLabel(query.direction),
    scopeLabel(query.scope || (query.query_is_current_entity ? 'current' : 'external')),
    shotLabel(query.shot_mode),
  ]
  if (query.direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all') {
    parts.push(query.enzyme_taxonomy_scope === 'eukaryote' ? 'eukaryotic enzymes only' : 'prokaryotic enzymes only')
  }
  parts.push(
    objectiveLabel(query.ranking_objective),
    compactRetrieval(query.score_source),
  )
  if (shown != null) parts.push(`${shown} results shown`)
  return parts.join(' · ')
}

export function entryRouteId(entry: RouteCatalogEntry): string {
  return entry.route_id || entry.route_id_pattern || entry.key
}

export function entryRouteMeta(entry: RouteCatalogEntry): string {
  return [
    directionLabel(entry.direction),
    entry.scope === 'any' ? 'known or new query' : scopeLabel(entry.scope),
    shotLabel(entry.shot_mode),
    objectiveLabel(entry.objective),
  ].join(' · ')
}

export function compactRetrieval(value: string | null | undefined): string {
  if (!value) return 'strategy not selected yet'
  const normalized = value.toLowerCase()
  if (normalized === 'seed') return 'similarity to known positive examples'
  if (normalized.includes('taxonomy_candidate_filter')) return 'candidate-universe taxonomy filter'
  if (normalized.includes('dual_kernel')) return 'neural ranking combined with similarity-graph evidence'
  if (normalized.includes('top10') && normalized.includes('rrf')) return 'two neural rankings combined by rank fusion'
  if (normalized.includes('neighbor_hybrid')) return 'direct prediction plus transfer from related proteins'
  if (normalized.includes('residual')) return 'reaction fingerprint with a learned correction'
  if (normalized.includes('direct')) return 'direct model ranking'
  return value.replaceAll('_', ' ')
}

export function directionLabel(direction: string | null | undefined) {
  return direction === 'reaction_to_enzyme' ? 'reaction → candidate enzymes' : 'enzyme → possible reactions'
}

export function scopeLabel(scope: string | null | undefined) {
  return scope === 'current' ? 'query already represented' : 'new or unregistered query'
}

export function shotLabel(mode: string | null | undefined) {
  return mode === 'few_shot' ? 'guided by known positives' : 'query only'
}

export function objectiveLabel(objective: string | null | undefined) {
  const number = objective?.match(/\d+/)?.[0]
  return number ? `return ${number} candidates` : 'selected result depth'
}
