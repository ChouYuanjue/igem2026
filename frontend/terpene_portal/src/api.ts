import type { ModelDataGraph, ModelDataSearchResponse, ModelDataSummary, PortalStatus, QueryForm, RankingResponse, RouteCatalog } from './types'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

export async function loadPortalStatus(): Promise<PortalStatus> {
  const response = await fetch('/api/portal/status')
  if (!response.ok) throw new Error(`Portal status returned ${response.status}`)
  return response.json() as Promise<PortalStatus>
}

export async function loadRouteCatalog(): Promise<RouteCatalog> {
  const response = await fetch('/api/model/routes')
  if (!response.ok) throw new Error(`Route catalog returned ${response.status}`)
  return response.json() as Promise<RouteCatalog>
}

export async function loadModelDataSummary(): Promise<ModelDataSummary> {
  const response = await fetch('/api/model-data/summary')
  if (!response.ok) throw new Error(`Model data summary returned ${response.status}`)
  return response.json() as Promise<ModelDataSummary>
}

export async function loadModelDataGraph(query = '', focusId = '', limit = 36): Promise<ModelDataGraph> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (query.trim()) params.set('q', query.trim())
  if (focusId) params.set('focus_id', focusId)
  const response = await fetch(`/api/model-data/graph?${params}`)
  if (!response.ok) throw new Error(`Model data graph returned ${response.status}`)
  return response.json() as Promise<ModelDataGraph>
}

export async function searchModelData(query: string, kind = 'all', limit = 40): Promise<ModelDataSearchResponse> {
  const params = new URLSearchParams({ q: query.trim(), kind, limit: String(limit) })
  const response = await fetch(`/api/model-data/search?${params}`)
  if (!response.ok) throw new Error(`Model data search returned ${response.status}`)
  return response.json() as Promise<ModelDataSearchResponse>
}

export async function runRanking(form: QueryForm): Promise<RankingResponse> {
  const isR2E = form.direction === 'reaction_to_enzyme'
  const payload: Record<string, unknown> = {
    top_k: form.topK,
    ranking_objective: `top${form.topK}`,
    conformal_mode: form.conformalMode,
    conformal_alpha: form.conformalAlpha,
  }
  if (isR2E) payload.enzyme_taxonomy_scope = form.enzymeTaxonomyScope

  const seedIds = parseIdentifierList(form.seedIdsText)
  const maskIds = parseIdentifierList(form.maskIdsText)

  if (isR2E) {
    payload[form.entityMode === 'id' ? 'reaction_id' : 'reaction_smiles'] = form.queryValue.trim()
    if (form.shotMode === 'few_shot') payload.known_enzyme_ids = seedIds
  } else {
    payload[form.entityMode === 'id' ? 'enzyme_id' : 'enzyme_sequence'] = form.queryValue.trim()
    if (form.shotMode === 'few_shot') payload.known_reaction_ids = seedIds
    if (maskIds.length) payload.mask_reaction_ids = maskIds
  }

  const endpoint = isR2E ? '/api/model/rank/enzymes' : '/api/model/rank/reactions'
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  })
  const body = await response.json().catch(() => ({})) as RankingResponse & { error?: string; message?: string }
  if (!response.ok) {
    throw new Error(body.message || body.error || `Ranking request returned ${response.status}`)
  }
  return body
}

export function parseIdentifierList(value: string) {
  return [...new Set(value.split(/[\s,;]+/).map((item) => item.trim()).filter(Boolean))]
}

export function downloadJson(payload: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' })
  const href = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = href
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(href)
}

export function downloadCsv(response: RankingResponse) {
  const candidates = response.candidates
  const columns = [
    'rank',
    'candidate_id',
    'score',
    'selection_source',
    'ensemble_topk_vote_fraction',
    'ensemble_rank_std',
    'is_external_candidate',
    'conformal_set_member',
    'candidate_taxonomy_scope',
    'candidate_kingdom',
    'candidate_taxonomy_source',
    'evidence_tier',
    'evidence_score',
    'evidence_paths',
    'evidence_warnings',
  ]
  const rows: Array<Record<string, unknown>> = candidates.map((candidate) => ({
    ...candidate,
    evidence_tier: candidate.evidence_passport?.tier ?? '',
    evidence_score: candidate.evidence_passport?.score ?? '',
    evidence_paths: candidate.evidence_passport?.paths?.join('|') ?? '',
    evidence_warnings: candidate.evidence_passport?.warnings?.join('|') ?? '',
  }))
  const escape = (value: unknown) => {
    const text = value == null ? '' : String(value)
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
  }
  const csv = [columns.join(','), ...rows.map((row) => columns.map((column) => escape(row[column])).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const href = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = href
  anchor.download = `${response.query.query_id || 'terpene-ranking'}-${response.query.ranking_objective || 'ranking'}.csv`
  anchor.click()
  URL.revokeObjectURL(href)
}
