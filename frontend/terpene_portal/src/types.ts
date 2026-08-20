export type Direction = 'reaction_to_enzyme' | 'enzyme_to_reaction'
export type ShotMode = 'zero_shot' | 'few_shot'
export type EnzymeTaxonomyScope = 'all' | 'eukaryote' | 'prokaryote'
export type PortalView = 'navigator' | 'atlas'
export type RunState = 'idle' | 'running' | 'success' | 'error'

export type EvidencePassport = {
  version?: string | null
  applicability_model_version?: string | null
  applicability_score?: number | null
  applicability_tier?: string | null
  recommendation?: string | null
  components?: Record<string, number>
  interpretation?: string | null
}

export type ConformalRetrievalSet = {
  version?: string | null
  method?: string | null
  mode?: string | null
  alpha?: number | null
  target_coverage?: number | null
  calibrator?: string | null
  binding_status?: string | null
  status?: string | null
  applicability_group?: string | null
  group_source?: string | null
  qhat?: number | null
  set_size?: number | null
  set_fraction?: number | null
  truncated?: boolean | null
  validation_coverage?: number | null
  validation_n?: number | null
  guarantee_scope?: string | null
  interpretation?: string | null
  recommendation?: string | null
  requested_top_k?: number | null
  expanded_output?: boolean | null
}

export type QueryMetadata = {
  query_id?: string
  direction?: Direction
  ranking_objective?: string
  route_id?: string
  route_version?: string
  candidate_universe_version?: string
  candidate_universe_hash?: string
  candidate_universe_size?: number
  model_bundle_version?: string
  registry_version?: string
  score_source?: string
  model_directory?: string
  secondary_model_directory?: string
  auxiliary_score_directory?: string
  query_nearest_library_id?: string
  query_nearest_library_similarity?: number | null
  query_is_current_entity?: boolean
  scope?: 'current' | 'external'
  requested_shot_mode?: ShotMode
  shot_mode?: ShotMode
  seed_ids?: string[]
  seed_count?: number
  mask_ids?: string[]
  mask_count?: number
  empirical_reliability_score?: number | null
  empirical_reliability_tier?: string
  empirical_reliability_status?: string
  empirical_reliability_binding_status?: string
  reliability_recommendation?: string
  input_audit?: Record<string, unknown>
  evidence_passport?: EvidencePassport
  conformal_retrieval_set?: ConformalRetrievalSet
  taxonomy_scope_version?: string
  enzyme_taxonomy_scope?: EnzymeTaxonomyScope
  taxonomy_scope_mode?: 'unrestricted' | 'candidate_filter'
  candidate_universe_pre_taxonomy_size?: number
  candidate_universe_post_taxonomy_size?: number
  taxonomy_eukaryote_count?: number
  taxonomy_prokaryote_count?: number
  taxonomy_other_count?: number
  taxonomy_unknown_count?: number
  taxonomy_excluded_count?: number
}

export type CandidateEvidence = {
  score?: number | null
  tier?: string | null
  paths?: string[]
  warnings?: string[]
  interpretation?: string | null
}

export type Candidate = {
  query_is_current_entity?: boolean
  rank: number
  candidate_id: string
  score?: number | null
  selection_source?: string
  ensemble_score_mean?: number | null
  ensemble_score_std?: number | null
  ensemble_rank_mean?: number | null
  ensemble_rank_std?: number | null
  ensemble_topk_vote_fraction?: number | null
  ensemble_top1_vote_fraction?: number | null
  ensemble_topk_jaccard?: number | null
  is_external_candidate?: boolean
  empirical_reliability_calibrator?: string
  conformal_set_member?: boolean
  candidate_taxonomy_scope?: string
  candidate_kingdom?: string
  candidate_taxonomy_source?: string
  evidence_passport?: CandidateEvidence
  [key: string]: unknown
}

export type RankingResponse = {
  query: QueryMetadata
  candidates: Candidate[]
}

export type QueryForm = {
  direction: Direction
  shotMode: ShotMode
  entityMode: 'id' | 'raw'
  queryValue: string
  seedIdsText: string
  maskIdsText: string
  enzymeTaxonomyScope: EnzymeTaxonomyScope
  topK: 3 | 10 | 20
  conformalMode: 'annotate' | 'expand' | 'disabled'
  conformalAlpha: 0.05 | 0.1 | 0.2
}

export type FlowLane = {
  id: string
  title: string
  subtitle: string
  weight?: number
  tone: 'primary' | 'secondary' | 'auxiliary' | 'neutral'
  active: boolean
}

export type FlowStage = {
  id: string
  index: number
  eyebrow: string
  title: string
  detail: string
  metric?: string
  state: 'complete' | 'active' | 'queued' | 'skipped'
}

export type RouteCatalogEntry = {
  key: string
  route_id?: string
  route_id_pattern?: string
  direction: Direction
  scope: 'current' | 'external' | 'any'
  shot_mode: ShotMode
  objective: string
  retrieval: string
  family: string
  modules: string[]
  deployment_key?: string
  deployment?: string
  secondary_deployment?: string | null
  auxiliary_deployment?: string | null
  settings?: Record<string, number | string | boolean>
  description: string
  use_case: string
  reliability_scope: string
  conformal_scope: string
  modifier_suffix?: string | null
  availability?: 'portal' | 'conditional' | 'cli_only' | 'batch_only'
  category?: 'manifest_route' | 'execution_path' | 'modifier' | 'conditional_path'
}

export type RouteCoverage = {
  manifest_route_ids: string[]
  manifest_route_count: number
  missing_manifest_routes: string[]
  unexpected_manifest_routes: string[]
  runtime_modifier_suffixes: string[]
  represented_modifier_suffixes: string[]
  missing_runtime_modifiers: string[]
  conditional_paths: string[]
  complete: boolean
}

export type TaxonomyScopeSummary = {
  version: string
  total: number
  scope_counts: Record<string, number>
  taxonomy_source_counts: Record<string, number>
  genus_conflicts_excluded?: string[]
  policy?: Record<string, string>
}

export type RouteCatalog = {
  manifest_version: number
  route_version: string
  candidate_universe_version: string
  model_bundle_version: string
  routes: RouteCatalogEntry[]
  overlays: RouteCatalogEntry[]
  route_count: number
  overlay_count: number
  display_path_count: number
  coverage: RouteCoverage
  taxonomy_scope?: TaxonomyScopeSummary | null
  read_only: boolean
}

export type PortalStatus = {
  status: string
  model_api: string
  database_mode: 'proxy' | 'compatibility_snapshot' | 'unavailable'
  database_commit: string
  database_origin: string
  route_version?: string
  registry_version?: string
}

export type CountBucket = { label: string; count: number }

export type ModelDataSummary = {
  proteins: number
  reactions: number
  associations: number
  registered_proteins: number
  registered_reactions: number
  mechanism_reactions: number
  seen_proteins: number
  seen_reactions: number
  terpene_types: CountBucket[]
  open_world_categories: CountBucket[]
  source_files: string[]
  read_only: boolean
}

export type ModelDataNode = {
  id: string
  kind: 'protein' | 'reaction'
  name: string
  degree?: number
  species?: string | null
  kingdom?: string | null
  terpene_type?: string | null
  tps_class?: string | null
  sequence_length?: number | null
  substrate_name?: string | null
  product_name?: string | null
  has_mechanism?: boolean
  seen?: boolean
  registered?: boolean
  source_file?: string
  [key: string]: unknown
}

export type ModelDataEdge = {
  id: string
  protein_id: string
  reaction_id: string
  protein_name: string
  reaction_name: string
  terpene_type?: string | null
  open_world_category?: string | null
  protein_seen?: boolean
  reaction_seen?: boolean
  has_mechanism?: boolean
  publication?: string | null
  [key: string]: unknown
}

export type ModelDataGraph = {
  query: string
  focus_id?: string | null
  nodes: ModelDataNode[]
  edges: ModelDataEdge[]
  node_count: number
  edge_count: number
  total_associations: number
  truncated: boolean
  read_only: boolean
}

export type ModelDataSearchItem = ModelDataNode & {
  protein_id?: string
  reaction_id?: string
  protein_name?: string
  reaction_name?: string
  open_world_category?: string | null
}

export type ModelDataSearchResponse = {
  query: string
  kind: string
  items: ModelDataSearchItem[]
  limit: number
  total_returned: number
}
