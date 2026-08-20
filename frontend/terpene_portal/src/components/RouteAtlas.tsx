import { useEffect, useMemo, useState } from 'react'
import {
  Background,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  Handle,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  getNodesBounds,
  getSmoothStepPath,
  useReactFlow,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import { formatPercent, humanize } from '../flow'
import { compactRetrieval, directionLabel, entryRouteId, entryRouteMeta, objectiveLabel, queryRouteId, queryRouteMeta, scopeLabel, shotLabel } from '../routePresentation'
import type { Candidate, QueryMetadata, RouteCatalog, RouteCatalogEntry, RunState } from '../types'

type Props = {
  query: QueryMetadata
  candidates: Candidate[]
  routeCatalog: RouteCatalog | null
  runState: RunState
  activeStage: number
}

type ModuleState = 'complete' | 'active' | 'queued' | 'standby'
type ModuleKind =
  | 'section'
  | 'reaction-input'
  | 'protein-input'
  | 'shot-gate'
  | 'scope-gate'
  | 'reaction-encoder'
  | 'protein-encoder'
  | 'universe'
  | 'taxonomy-filter'
  | 'router'
  | 'dual-tower'
  | 'loss075'
  | 'residual'
  | 'seed-protein'
  | 'seed-reaction'
  | 'seed-mask'
  | 'neighbor'
  | 'hard-negative'
  | 'dual-kernel'
  | 'rrf'
  | 'cage'
  | 'mask-only'
  | 'rank-lock'
  | 'trust'
  | 'output'

type ModuleSpec = {
  id: string
  position: { x: number; y: number }
  kind: ModuleKind
  title: string
  subtitle: string
  metric: string
  phase: number
  width?: number
  height?: number
  direction?: 'r2e' | 'e2r'
  detail: string
}

type ModuleData = ModuleSpec & {
  state: ModuleState
  onRoute: boolean
  onActualRoute: boolean
  metric: string
  selected: boolean
}

type ModuleNode = Node<ModuleData, 'routeModule'>
type RailData = {
  routes: string[]
  label?: string
  phase: number
  tone: 'r2e' | 'e2r' | 'few' | 'modifier'
  state: ModuleState
  onRoute: boolean
  actual: boolean
  optional?: 'cage' | 'mask'
}
type RailEdge = Edge<RailData, 'routeRail'>

type RailSpec = {
  id: string
  source: string
  target: string
  routes: string[]
  label?: string
  phase: number
  tone: RailData['tone']
  optional?: RailData['optional']
}

const R2E_CURRENT = ['r2e-current-top3-v1', 'r2e-current-top10-v1', 'r2e-current-top20-v1']
const R2E_EXTERNAL_TOP3 = ['r2e-external-top3-v1']
const R2E_EXTERNAL_RESIDUAL = ['r2e-external-top10-v1', 'r2e-external-top20-v1']
const R2E_FEW = ['r2e-fewshot-seed']
const R2E_ALL = [...R2E_CURRENT, ...R2E_EXTERNAL_TOP3, ...R2E_EXTERNAL_RESIDUAL, ...R2E_FEW]

const E2R_CURRENT = ['e2r-current-top3-v1', 'e2r-current-top10-v1', 'e2r-current-top20-v1']
const E2R_EXTERNAL_TOP3 = ['e2r-external-top3-neighbor-v1']
const E2R_EXTERNAL_TOP10 = ['e2r-external-top10-neural-rrf-v1']
const E2R_EXTERNAL_TOP20 = ['e2r-external-top20-dual-kernel-rrf-v1']
const E2R_FEW = ['e2r-fewshot-seed']
const E2R_ALL = [...E2R_CURRENT, ...E2R_EXTERNAL_TOP3, ...E2R_EXTERNAL_TOP10, ...E2R_EXTERNAL_TOP20, ...E2R_FEW]

const ROUTE_PATHS: Record<string, string[]> = {
  'r2e-current-top3-v1': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-shared', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'r2e-current-top10-v1': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-shared', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'r2e-current-top20-v1': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-shared', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'r2e-external-top3-v1': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-loss075', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'r2e-external-top10-v1': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-residual', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'r2e-external-top20-v1': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-residual', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'r2e-fewshot-seed': ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router', 'r2e-seed', 'r2e-seed-mask', 'r2e-rank', 'r2e-trust', 'r2e-output'],
  'e2r-current-top3-v1': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-current', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'e2r-current-top10-v1': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-current', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'e2r-current-top20-v1': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-current', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'e2r-external-top3-neighbor-v1': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-neighbor', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'e2r-external-top10-neural-rrf-v1': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-neighbor', 'e2r-hardneg', 'e2r-rrf10', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'e2r-external-top20-dual-kernel-rrf-v1': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-neighbor', 'e2r-dualkernel', 'e2r-rrf20', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'e2r-fewshot-seed': ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router', 'e2r-seed', 'e2r-seed-mask', 'e2r-rank', 'e2r-trust', 'e2r-output'],
  'r2e-known-association-mask-overlay': ['r2e-known-mask'],
  'e2r-zero-shot-mask-overlay': ['e2r-query', 'e2r-shot', 'e2r-mask-only', 'e2r-rank'],
}

const MODULE_SPECS: ModuleSpec[] = [
  { id: 'r2e-section', position: { x: -10, y: -115 }, kind: 'section', title: 'REACTION → ENZYME', subtitle: 'find candidate catalysts', metric: 'query-only, example-guided and rescue paths', phase: -1, width: 330, height: 54, direction: 'r2e', detail: 'Search strategies that start from a terpene-forming reaction and rank possible enzyme catalysts.' },
  { id: 'r2e-query', position: { x: 0, y: 35 }, kind: 'reaction-input', title: 'Reaction to explain', subtitle: 'database ID or reaction string', metric: 'substrate → product', phase: 0, direction: 'r2e', detail: 'The starting point can be a known reaction record or a reaction written as substrate and product structures.' },
  { id: 'r2e-shot', position: { x: 225, y: 35 }, kind: 'shot-gate', title: 'Known examples?', subtitle: 'query only / example guided', metric: 'no seed', phase: 1, direction: 'r2e', detail: 'Chooses between searching from the reaction alone and searching with known working catalysts as examples.' },
  { id: 'r2e-scope', position: { x: 450, y: 35 }, kind: 'scope-gate', title: 'Is the query already known?', subtitle: 'reference / new query', metric: 'resolve identity', phase: 2, direction: 'r2e', detail: 'Checks whether the query is already represented in the reference data or must be encoded as a new case.' },
  { id: 'r2e-encoder', position: { x: 675, y: 35 }, kind: 'reaction-encoder', title: 'Reaction fingerprint', subtitle: 'changed bonds + chemical context', metric: '2,115 dimensions', phase: 3, direction: 'r2e', detail: 'Converts the chemical transformation into numbers that capture changed atom environments, precursor type and product skeleton.' },
  { id: 'r2e-universe', position: { x: 900, y: 35 }, kind: 'universe', title: 'Candidate enzyme collection', subtitle: 'reference + newly added proteins', metric: '2,085 proteins', phase: 4, direction: 'r2e', detail: 'Loads the deployed enzyme candidate universe: 1,391 reference proteins plus 694 registered proteins.' },
  { id: 'r2e-taxonomy', position: { x: 1125, y: 35 }, kind: 'taxonomy-filter', title: 'Taxonomy sieve', subtitle: 'all / eukaryote / prokaryote', metric: 'pass all candidates', phase: 5, direction: 'r2e', detail: 'Optionally restricts the enzyme matrix before scoring. Eukaryote-only retains locally classified plants, fungi, animals and amoebozoa; prokaryote-only retains bacteria, archaea and cyanobacteria. Unresolved records are never guessed into a restricted group.' },
  { id: 'r2e-router', position: { x: 1350, y: 35 }, kind: 'router', title: 'Choose enzyme-search strategy', subtitle: 'query status × examples × list size', metric: 'Top-3 / 10 / 20', phase: 6, direction: 'r2e', detail: 'Selects the scoring strategy that matches whether the reaction is known, whether examples were supplied and how many candidates are requested.' },
  { id: 'r2e-shared', position: { x: 1590, y: -80 }, kind: 'dual-tower', title: 'Shared reaction–enzyme model', subtitle: 'known-reaction search', metric: 'direct · all budgets', phase: 7, direction: 'r2e', detail: 'A paired neural model scores how well each eligible enzyme representation matches the known reaction representation.' },
  { id: 'r2e-loss075', position: { x: 1590, y: 55 }, kind: 'loss075', title: 'Focused new-reaction model', subtitle: 'new reaction · Top-3', metric: 'reaction loss 0.75', phase: 7, direction: 'r2e', detail: 'A model tuned to produce a very small, focused enzyme shortlist for a reaction outside the reference set.' },
  { id: 'r2e-residual', position: { x: 1590, y: 190 }, kind: 'residual', title: 'Residual reaction model', subtitle: 'new reaction · Top-10 / 20', metric: 'base + learned residual', phase: 7, direction: 'r2e', detail: 'Combines an exact reaction fingerprint with a learned correction to broaden enzyme retrieval for new reactions.' },
  { id: 'r2e-seed', position: { x: 1590, y: 325 }, kind: 'seed-protein', title: 'Compare with known catalysts', subtitle: 'example-guided enzyme search', metric: 'max similarity to seeds', phase: 7, direction: 'r2e', detail: 'Ranks eligible proteins by how closely their learned sequence representations resemble the supplied working catalysts.' },
  { id: 'r2e-cage', position: { x: 1825, y: -80 }, kind: 'cage', title: 'Structure-informed rescue', subtitle: 'known reaction · broad screen', metric: 'up to 5 rescue slots', phase: 8, direction: 'r2e', detail: 'Can add a small number of independently validated structure-informed candidates when they are still inside the selected taxonomy universe.' },
  { id: 'r2e-seed-mask', position: { x: 1825, y: 325 }, kind: 'seed-mask', title: 'Remove supplied examples', subtitle: 'return only new candidates', metric: 'remove supplied catalysts', phase: 8, direction: 'r2e', detail: 'Supplied known catalyst IDs are excluded so the result contains new candidates only.' },
  { id: 'r2e-known-mask', position: { x: 1825, y: 460 }, kind: 'mask-only', title: 'Exclude known enzyme links', subtitle: 'registry batch discovery', metric: 'known associations hidden', phase: 8, direction: 'r2e', detail: 'Registry-wide discovery batches remove enzyme IDs already linked to the query reaction before selecting the final Top-K, so known associations are not re-reported as discoveries.' },
  { id: 'r2e-rank', position: { x: 2060, y: 85 }, kind: 'rank-lock', title: 'Rank candidate enzymes', subtitle: 'model priority order', metric: 'Top-K prefix', phase: 9, direction: 'r2e', detail: 'Fixes the model-generated order before uncertainty and explanation fields are added.' },
  { id: 'r2e-trust', position: { x: 2295, y: 85 }, kind: 'trust', title: 'Interpret the search', subtitle: 'familiarity · stability · review depth', metric: 'scope-aware', phase: 10, direction: 'r2e', detail: 'For eligible unrestricted new query-only searches, the system estimates familiarity, ranking stability and review depth. Taxonomy-restricted runs are explicitly marked outside those calibrations.' },
  { id: 'r2e-output', position: { x: 2530, y: 85 }, kind: 'output', title: 'Candidate enzymes for testing', subtitle: 'prioritized experimental hypotheses', metric: 'ranked catalysts', phase: 11, direction: 'r2e', detail: 'Returns enzyme hypotheses with rank, taxonomy provenance, supporting signals, warnings and suggested interpretation.' },

  { id: 'e2r-section', position: { x: -10, y: 535 }, kind: 'section', title: 'ENZYME → REACTION', subtitle: 'predict possible activities', metric: 'query-only, example-guided and filtered paths', phase: -1, width: 330, height: 54, direction: 'e2r', detail: 'Search strategies that start from a terpene synthase and rank possible reactions it may catalyze.' },
  { id: 'e2r-query', position: { x: 0, y: 685 }, kind: 'protein-input', title: 'Enzyme to annotate', subtitle: 'database ID or protein sequence', metric: 'terpene synthase sequence', phase: 0, direction: 'e2r', detail: 'The starting point can be a known protein record or a new amino-acid sequence.' },
  { id: 'e2r-shot', position: { x: 225, y: 685 }, kind: 'shot-gate', title: 'Known examples?', subtitle: 'query only / example guided', metric: 'no seed', phase: 1, direction: 'e2r', detail: 'Chooses between predicting from the enzyme alone and expanding from reactions already known for related activity.' },
  { id: 'e2r-scope', position: { x: 450, y: 685 }, kind: 'scope-gate', title: 'Is the query already known?', subtitle: 'reference / new query', metric: 'resolve identity', phase: 2, direction: 'e2r', detail: 'Checks whether the protein already has a stored representation or must be encoded from its sequence.' },
  { id: 'e2r-encoder', position: { x: 675, y: 685 }, kind: 'protein-encoder', title: 'Protein sequence embedding', subtitle: 'learned sequence features', metric: '1,152 dimensions', phase: 3, direction: 'e2r', detail: 'Uses a protein language model to convert the amino-acid sequence into a numerical representation of sequence patterns.' },
  { id: 'e2r-universe', position: { x: 900, y: 685 }, kind: 'universe', title: 'Candidate reaction collection', subtitle: 'reference + newly added reactions', metric: '753 reactions', phase: 4, direction: 'e2r', detail: 'Searches across 753 reactions: 513 from the reference collection and 240 added for broader activity discovery.' },
  { id: 'e2r-router', position: { x: 1125, y: 685 }, kind: 'router', title: 'Choose reaction-search strategy', subtitle: 'query status × examples × list size', metric: 'Top-3 / 10 / 20', phase: 5, direction: 'e2r', detail: 'Selects direct prediction, transfer from related proteins, rank fusion, similarity-graph support or example-guided search.' },
  { id: 'e2r-current', position: { x: 1360, y: 550 }, kind: 'dual-tower', title: 'Dedicated enzyme-activity model', subtitle: 'known-enzyme search', metric: 'direct · all budgets', phase: 6, direction: 'e2r', detail: 'A neural model specialized for assigning reactions to enzymes already represented in the reference data.' },
  { id: 'e2r-neighbor', position: { x: 1360, y: 685 }, kind: 'neighbor', title: 'Direct model + related proteins', subtitle: 'new-protein primary search', metric: 'direct + 5 neighbours', phase: 6, direction: 'e2r', detail: 'Combines the new protein’s direct prediction with activity evidence transferred from five related reference proteins.' },
  { id: 'e2r-hardneg', position: { x: 1595, y: 815 }, kind: 'hard-negative', title: 'Contrastive secondary model', subtitle: 'new protein · second ranking view', metric: '90% direct · 3 neighbours', phase: 6, direction: 'e2r', detail: 'A second neural model trained to separate difficult alternatives provides an independent ranking for Top-10 fusion.' },
  { id: 'e2r-dualkernel', position: { x: 1595, y: 950 }, kind: 'dual-kernel', title: 'Similarity-graph support', subtitle: 'new protein · broad-search support', metric: 'reaction × graph × protein', phase: 6, direction: 'e2r', detail: 'Adds evidence from reaction similarity, protein similarity and the known protein–reaction network.' },
  { id: 'e2r-seed', position: { x: 1360, y: 1085 }, kind: 'seed-reaction', title: 'Compare with known reactions', subtitle: 'example-guided activity search', metric: 'max similarity to seeds', phase: 6, direction: 'e2r', detail: 'Ranks reactions by how closely their learned representations resemble the supplied known activities.' },
  { id: 'e2r-rrf10', position: { x: 1830, y: 815 }, kind: 'rrf', title: 'Combine two Top-10 rankings', subtitle: 'merge by rank position', metric: '35% primary · 65% secondary', phase: 7, direction: 'e2r', detail: 'Combines two independently ordered lists using reciprocal-rank fusion, so rank positions rather than incompatible raw scores are merged.' },
  { id: 'e2r-rrf20', position: { x: 1830, y: 950 }, kind: 'rrf', title: 'Combine neural and graph rankings', subtitle: 'merge by rank position', metric: '70% neural · 30% dual-kernel', phase: 7, direction: 'e2r', detail: 'Combines the neural activity ranking with similarity-graph evidence to support broader Top-20 exploration.' },
  { id: 'e2r-seed-mask', position: { x: 1595, y: 1085 }, kind: 'seed-mask', title: 'Remove supplied examples', subtitle: 'return only new candidates', metric: 'remove known reactions', phase: 7, direction: 'e2r', detail: 'Known example reactions guide the search but are removed from the final list of new possibilities.' },
  { id: 'e2r-mask-only', position: { x: 1830, y: 1085 }, kind: 'mask-only', title: 'Exclude selected reactions', subtitle: 'filter only · no positive signal', metric: 'route preserved', phase: 7, direction: 'e2r', detail: 'Hides specified reactions from the results without treating them as evidence that similar reactions should rank higher.' },
  { id: 'e2r-rank', position: { x: 2065, y: 745 }, kind: 'rank-lock', title: 'Rank possible reactions', subtitle: 'model priority order', metric: 'Top-K prefix', phase: 8, direction: 'e2r', detail: 'Fixes the model-generated order before uncertainty and explanation fields are added.' },
  { id: 'e2r-trust', position: { x: 2300, y: 745 }, kind: 'trust', title: 'Interpret the search', subtitle: 'familiarity · stability · review depth', metric: 'scope-aware', phase: 9, direction: 'e2r', detail: 'For eligible new query-only searches, the system estimates query familiarity, ranking stability and the review depth needed for a recall target.' },
  { id: 'e2r-output', position: { x: 2535, y: 745 }, kind: 'output', title: 'Possible reactions to investigate', subtitle: 'prioritized experimental hypotheses', metric: 'prioritized activity hypotheses', phase: 10, direction: 'e2r', detail: 'Returns possible reactions with rank, supporting signals, warnings and suggested interpretation.' },
]

const RAIL_SPECS: RailSpec[] = [
  ...chainRails('r2e-prefix', ['r2e-query', 'r2e-shot', 'r2e-scope', 'r2e-encoder', 'r2e-universe', 'r2e-taxonomy', 'r2e-router'], R2E_ALL, 'r2e'),
  { id: 'r2e-route-shared', source: 'r2e-router', target: 'r2e-shared', routes: R2E_CURRENT, label: 'KNOWN QUERY · TOP-3 / 10 / 20', phase: 7, tone: 'r2e' },
  { id: 'r2e-route-top3', source: 'r2e-router', target: 'r2e-loss075', routes: R2E_EXTERNAL_TOP3, label: 'NEW QUERY · TOP-3', phase: 7, tone: 'r2e' },
  { id: 'r2e-route-residual', source: 'r2e-router', target: 'r2e-residual', routes: R2E_EXTERNAL_RESIDUAL, label: 'NEW QUERY · TOP-10 / 20', phase: 7, tone: 'r2e' },
  { id: 'r2e-route-few', source: 'r2e-router', target: 'r2e-seed', routes: R2E_FEW, label: 'KNOWN EXAMPLES · ANY LIST SIZE', phase: 7, tone: 'few' },
  { id: 'r2e-shared-rank', source: 'r2e-shared', target: 'r2e-rank', routes: R2E_CURRENT, label: 'ONE MODEL', phase: 9, tone: 'r2e' },
  { id: 'r2e-shared-cage', source: 'r2e-shared', target: 'r2e-cage', routes: ['r2e-current-top20-v1'], label: 'STRUCTURE EVIDENCE AVAILABLE', phase: 8, tone: 'modifier', optional: 'cage' },
  { id: 'r2e-cage-rank', source: 'r2e-cage', target: 'r2e-rank', routes: ['r2e-current-top20-v1'], label: 'ADD SUPPORTED CANDIDATES', phase: 9, tone: 'modifier', optional: 'cage' },
  { id: 'r2e-top3-rank', source: 'r2e-loss075', target: 'r2e-rank', routes: R2E_EXTERNAL_TOP3, label: 'ONE MODEL', phase: 9, tone: 'r2e' },
  { id: 'r2e-residual-rank', source: 'r2e-residual', target: 'r2e-rank', routes: R2E_EXTERNAL_RESIDUAL, label: 'ONE MODEL', phase: 9, tone: 'r2e' },
  { id: 'r2e-seed-mask', source: 'r2e-seed', target: 'r2e-seed-mask', routes: R2E_FEW, label: 'REMOVE SUPPLIED EXAMPLES', phase: 8, tone: 'few' },
  { id: 'r2e-seed-rank', source: 'r2e-seed-mask', target: 'r2e-rank', routes: R2E_FEW, label: 'RETURN NEW CANDIDATES', phase: 9, tone: 'few' },
  ...chainRails('r2e-output', ['r2e-rank', 'r2e-trust', 'r2e-output'], R2E_ALL, 'r2e', 10),

  ...chainRails('e2r-prefix', ['e2r-query', 'e2r-shot', 'e2r-scope', 'e2r-encoder', 'e2r-universe', 'e2r-router'], E2R_ALL, 'e2r'),
  { id: 'e2r-route-current', source: 'e2r-router', target: 'e2r-current', routes: E2R_CURRENT, label: 'KNOWN QUERY · TOP-3 / 10 / 20', phase: 6, tone: 'e2r' },
  { id: 'e2r-route-neighbor', source: 'e2r-router', target: 'e2r-neighbor', routes: [...E2R_EXTERNAL_TOP3, ...E2R_EXTERNAL_TOP10, ...E2R_EXTERNAL_TOP20], label: 'NEW QUERY · MAIN MODEL', phase: 6, tone: 'e2r' },
  { id: 'e2r-route-hardneg', source: 'e2r-router', target: 'e2r-hardneg', routes: E2R_EXTERNAL_TOP10, label: 'TOP-10 · SECOND MODEL', phase: 6, tone: 'e2r' },
  { id: 'e2r-route-dualkernel', source: 'e2r-router', target: 'e2r-dualkernel', routes: E2R_EXTERNAL_TOP20, label: 'TOP-20 · SIMILARITY NETWORK', phase: 6, tone: 'e2r' },
  { id: 'e2r-route-few', source: 'e2r-router', target: 'e2r-seed', routes: E2R_FEW, label: 'KNOWN EXAMPLES · ANY LIST SIZE', phase: 6, tone: 'few' },
  { id: 'e2r-current-rank', source: 'e2r-current', target: 'e2r-rank', routes: E2R_CURRENT, label: 'ONE MODEL', phase: 8, tone: 'e2r' },
  { id: 'e2r-top3-rank', source: 'e2r-neighbor', target: 'e2r-rank', routes: E2R_EXTERNAL_TOP3, label: 'DIRECT + RELATED PROTEINS', phase: 8, tone: 'e2r' },
  { id: 'e2r-primary-rrf10', source: 'e2r-neighbor', target: 'e2r-rrf10', routes: E2R_EXTERNAL_TOP10, label: 'MAIN RANKING 35%', phase: 7, tone: 'e2r' },
  { id: 'e2r-hardneg-rrf10', source: 'e2r-hardneg', target: 'e2r-rrf10', routes: E2R_EXTERNAL_TOP10, label: 'SECOND RANKING 65%', phase: 7, tone: 'e2r' },
  { id: 'e2r-rrf10-rank', source: 'e2r-rrf10', target: 'e2r-rank', routes: E2R_EXTERNAL_TOP10, label: 'MERGED BY RANK POSITION', phase: 8, tone: 'e2r' },
  { id: 'e2r-primary-rrf20', source: 'e2r-neighbor', target: 'e2r-rrf20', routes: E2R_EXTERNAL_TOP20, label: 'NEURAL RANKING 70%', phase: 7, tone: 'e2r' },
  { id: 'e2r-dual-rrf20', source: 'e2r-dualkernel', target: 'e2r-rrf20', routes: E2R_EXTERNAL_TOP20, label: 'SIMILARITY NETWORK 30%', phase: 7, tone: 'e2r' },
  { id: 'e2r-rrf20-rank', source: 'e2r-rrf20', target: 'e2r-rank', routes: E2R_EXTERNAL_TOP20, label: 'MERGED BY RANK POSITION', phase: 8, tone: 'e2r' },
  { id: 'e2r-seed-mask', source: 'e2r-seed', target: 'e2r-seed-mask', routes: E2R_FEW, label: 'REMOVE SUPPLIED EXAMPLES', phase: 7, tone: 'few' },
  { id: 'e2r-seed-rank', source: 'e2r-seed-mask', target: 'e2r-rank', routes: E2R_FEW, label: 'RETURN NEW REACTIONS', phase: 8, tone: 'few' },
  { id: 'e2r-mask-overlay', source: 'e2r-mask-only', target: 'e2r-rank', routes: ['e2r-zero-shot-mask-overlay'], label: 'HIDE FROM RESULTS', phase: 8, tone: 'modifier', optional: 'mask' },
  ...chainRails('e2r-output', ['e2r-rank', 'e2r-trust', 'e2r-output'], E2R_ALL, 'e2r', 9),
]

const MODULE_INFO = new Map(MODULE_SPECS.map((module) => [module.id, module]))
const nodeTypes = { routeModule: RouteModuleNode }
const edgeTypes = { routeRail: RouteRailEdge }

export function RouteAtlas(props: Props) {
  return <ReactFlowProvider><RouteAtlasInner {...props} /></ReactFlowProvider>
}

function RouteAtlasInner({ query, candidates, routeCatalog, runState, activeStage }: Props) {
  const { fitView, getNode, setCenter } = useReactFlow()
  const [previewRoute, setPreviewRoute] = useState<string | null>(null)
  const [selectedModule, setSelectedModule] = useState<string | null>(null)
  const [replayStage, setReplayStage] = useState<number | null>(null)
  const actualRouteId = queryRouteId(query)
  const actualBaseRouteId = actualRouteId.split('+')[0]
  const actualPathKey = actualRouteKey(query)
  const displayRouteKey = previewRoute || actualPathKey
  const previewing = Boolean(previewRoute)
  const hasCageRescue = candidates.some((candidate) => candidate.selection_source === 'cage_rescue')
  const hasE2RMaskOnly = query.direction === 'enzyme_to_reaction' && (Boolean(query.mask_count) || actualRouteId.includes('+masked'))
  const hasR2EKnownMask = query.direction === 'reaction_to_enzyme' && actualRouteId.includes('+masked')
  const hasTemporaryUniverse = actualRouteId.includes('+temporary-universe')
  const hasManualOverride = actualRouteId.includes('+manual')
  const hasEukaryoteOnly = actualRouteId.includes('+eukaryote-only') || query.enzyme_taxonomy_scope === 'eukaryote'
  const hasProkaryoteOnly = actualRouteId.includes('+prokaryote-only') || query.enzyme_taxonomy_scope === 'prokaryote'

  useEffect(() => {
    if (replayStage == null) return
    if (replayStage >= 11) {
      const done = window.setTimeout(() => setReplayStage(null), 800)
      return () => window.clearTimeout(done)
    }
    const timer = window.setTimeout(() => setReplayStage((value) => value == null ? null : value + 1), 560)
    return () => window.clearTimeout(timer)
  }, [replayStage])

  const playhead = previewing ? 99 : runState === 'running' ? activeStage : replayStage ?? 99
  const routeEntries = useMemo(() => {
    if (!routeCatalog) return []
    return [...routeCatalog.routes, ...routeCatalog.overlays]
  }, [routeCatalog])
  const selectedEntry = routeEntries.find((entry) => entry.key === displayRouteKey) || null
  const actualBaseEntry = routeCatalog?.routes.find((entry) => entry.route_id === actualBaseRouteId) || null
  const actualPathEntry = routeEntries.find((entry) => entry.key === actualPathKey) || actualBaseEntry
  const actualPath = useMemo(() => {
    const path = [...(actualPathEntry?.modules || ROUTE_PATHS[actualPathKey] || [])]
    if (hasE2RMaskOnly && !path.includes('e2r-mask-only')) path.push('e2r-mask-only')
    if (hasR2EKnownMask && !path.includes('r2e-known-mask')) path.push('r2e-known-mask')
    if (hasCageRescue && !path.includes('r2e-cage')) path.push('r2e-cage')
    return path
  }, [actualPathEntry, actualPathKey, hasCageRescue, hasE2RMaskOnly, hasR2EKnownMask])
  const displayPath = previewing
    ? selectedEntry?.modules || ROUTE_PATHS[displayRouteKey] || []
    : actualPath
  const activeOverlayKeys = useMemo(() => {
    const keys = new Set<string>()
    if (query.shot_mode === 'few_shot' || actualRouteId.includes('+fewshot')) {
      keys.add(query.direction === 'reaction_to_enzyme' ? 'r2e-fewshot-seed' : 'e2r-fewshot-seed')
    }
    if (hasE2RMaskOnly) keys.add('e2r-zero-shot-mask-overlay')
    if (hasR2EKnownMask) keys.add('r2e-known-association-mask-overlay')
    if (hasTemporaryUniverse) keys.add(query.direction === 'reaction_to_enzyme' ? 'r2e-temporary-universe-overlay' : 'e2r-temporary-universe-overlay')
    if (hasManualOverride) keys.add(query.direction === 'reaction_to_enzyme' ? 'r2e-manual-override-overlay' : 'e2r-manual-override-overlay')
    if (hasEukaryoteOnly) keys.add('r2e-eukaryote-only-overlay')
    if (hasProkaryoteOnly) keys.add('r2e-prokaryote-only-overlay')
    if (hasCageRescue) keys.add('r2e-cage-rescue-overlay')
    return keys
  }, [actualRouteId, hasCageRescue, hasE2RMaskOnly, hasEukaryoteOnly, hasManualOverride, hasProkaryoteOnly, hasR2EKnownMask, hasTemporaryUniverse, query.direction, query.shot_mode])
  const displayRouteId = previewing && selectedEntry ? entryRouteId(selectedEntry) : actualRouteId
  const displayRouteMeta = previewing && selectedEntry ? entryRouteMeta(selectedEntry) : queryRouteMeta(query, candidates.length)

  useEffect(() => {
    if (!displayPath.length) return
    const timer = window.setTimeout(() => {
      const focusNodes = displayPath.map((id) => getNode(id)).filter(Boolean) as ModuleNode[]
      if (!focusNodes.length) return
      const pathZoom = window.innerWidth < 760 ? 0.30 : 0.48
      const bounds = getNodesBounds(focusNodes)
      void setCenter(
        bounds.x + bounds.width / 2,
        bounds.y + bounds.height / 2,
        { zoom: pathZoom, duration: 700 },
      )
    }, 120)
    return () => window.clearTimeout(timer)
  }, [displayRouteKey, displayPath, getNode, setCenter])

  const nodes = useMemo<ModuleNode[]>(() => MODULE_SPECS.map((spec) => {
    const onRoute = displayPath.includes(spec.id)
      || (spec.id === 'r2e-cage' && (displayRouteKey === 'r2e-current-top20-v1' || displayRouteKey === 'r2e-cage-rescue-overlay'))
      || (spec.id === 'e2r-mask-only' && (displayRouteKey === 'e2r-zero-shot-mask-overlay' || (!previewing && hasE2RMaskOnly)))
      || (spec.id === 'r2e-known-mask' && (displayRouteKey === 'r2e-known-association-mask-overlay' || (!previewing && hasR2EKnownMask)))
    const onActualRoute = actualPath.includes(spec.id)
      || (spec.id === 'r2e-cage' && actualBaseRouteId === 'r2e-current-top20-v1' && hasCageRescue)
      || (spec.id === 'e2r-mask-only' && hasE2RMaskOnly)
      || (spec.id === 'r2e-known-mask' && hasR2EKnownMask)
    const state = moduleState(spec.phase, onRoute, playhead)
    return {
      id: spec.id,
      type: 'routeModule',
      position: spec.position,
      data: {
        ...spec,
        metric: dynamicMetric(spec, query, candidates, onActualRoute),
        state,
        onRoute,
        onActualRoute,
        selected: selectedModule === spec.id,
      },
      draggable: false,
      selectable: true,
      focusable: true,
      width: spec.width || 198,
      height: spec.height || (spec.kind === 'section' ? 54 : 112),
      initialWidth: spec.width || 198,
      initialHeight: spec.height || (spec.kind === 'section' ? 54 : 112),
      style: { width: spec.width || 198, height: spec.height || (spec.kind === 'section' ? 54 : 112) },
    }
  }), [actualBaseRouteId, actualPath, candidates, displayPath, displayRouteKey, hasCageRescue, hasE2RMaskOnly, hasR2EKnownMask, playhead, previewing, query, selectedModule])

  const edges = useMemo<RailEdge[]>(() => RAIL_SPECS.map((spec) => {
    const optionalEnabled = spec.optional === 'cage' ? hasCageRescue : spec.optional === 'mask' ? hasE2RMaskOnly : true
    const routeMatch = spec.routes.includes(displayRouteKey)
    const actualMatch = spec.routes.includes(actualPathKey)
    const previewModifier = previewing && (
      (spec.optional === 'mask' && displayRouteKey === 'e2r-zero-shot-mask-overlay')
      || (spec.optional === 'cage' && displayRouteKey === 'r2e-cage-rescue-overlay')
    )
    const actualModifier = (spec.optional === 'mask' && hasE2RMaskOnly) || (spec.optional === 'cage' && hasCageRescue)
    const onRoute = (routeMatch && (spec.optional ? optionalEnabled || previewing : true)) || previewModifier || (!previewing && actualModifier)
    const actual = (actualMatch && (spec.optional ? optionalEnabled : true)) || actualModifier
    return {
      id: spec.id,
      source: spec.source,
      target: spec.target,
      type: 'routeRail',
      data: {
        routes: spec.routes,
        label: spec.label,
        phase: spec.phase,
        tone: spec.tone,
        state: moduleState(spec.phase, onRoute, playhead),
        onRoute,
        actual,
        optional: spec.optional,
      },
      animated: false,
      focusable: false,
      selectable: false,
    }
  }), [actualPathKey, displayRouteKey, hasCageRescue, hasE2RMaskOnly, playhead, previewing])

  const focus = (mode: 'all' | 'r2e' | 'e2r' | 'current') => {
    const ids = mode === 'all'
      ? MODULE_SPECS.map((node) => node.id)
      : mode === 'current'
        ? actualPath
        : MODULE_SPECS.filter((node) => node.direction === mode || node.kind === 'section' && node.direction === mode).map((node) => node.id)
    const focusNodes = ids.map((id) => getNode(id)).filter(Boolean) as ModuleNode[]
    if (mode === 'current') {
      const pathZoom = window.innerWidth < 760 ? 0.30 : 0.48
      const bounds = getNodesBounds(focusNodes)
      void setCenter(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2, { zoom: pathZoom, duration: 650 })
      return
    }
    void fitView({ nodes: focusNodes, padding: mode === 'all' ? 0.08 : 0.18, duration: 650, minZoom: 0.28, maxZoom: 1.15 })
  }

  return (
    <div className="route-atlas-shell">
      <div className="route-atlas-toolbar">
        <div className="route-atlas-state">
          <span className={`route-live-dot ${previewing ? 'preview' : ''}`} />
          <div><small>{previewing ? 'VIEWING ANOTHER SEARCH PATH' : 'PATH USED FOR THIS SEARCH'}</small><strong>{displayRouteId}</strong><em>{displayRouteMeta}</em></div>
        </div>
        <div className="route-atlas-actions">
          <button onClick={() => focus('all')}>Show whole system</button>
          <button onClick={() => focus('r2e')}>Reaction → enzyme paths</button>
          <button onClick={() => focus('e2r')}>Enzyme → reaction paths</button>
          <button onClick={() => focus('current')}>Focus selected path</button>
          <button className="replay-flow-button" onClick={() => { setPreviewRoute(null); setReplayStage(0); focus('current') }} disabled={runState === 'running'}>▶ Replay this search</button>
        </div>
      </div>

      <RouteBoard entries={routeEntries} actualRouteId={actualRouteId} actualBaseRouteId={actualBaseRouteId} actualPathKey={actualPathKey} activeOverlayKeys={activeOverlayKeys} previewRoute={previewRoute} onPreview={setPreviewRoute} />

      <div className="route-atlas-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          panOnScroll
          zoomOnScroll
          minZoom={0.28}
          maxZoom={1.45}
          fitView
          fitViewOptions={{ padding: 0.08, maxZoom: 0.72 }}
          onNodeClick={(_, node) => setSelectedModule(node.id)}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={34} size={1} color="rgba(103, 151, 181, 0.10)" />
          <Controls showInteractive={false} position="bottom-left" />
          <Panel position="top-left" className="route-map-legend">
            <span><i className="r2e" /> reaction → enzyme path</span>
            <span><i className="e2r" /> enzyme → reaction path</span>
            <span><i className="few" /> example-guided path</span>
            <span><i className="standby" /> available but not used</span>
          </Panel>
          {previewing && <Panel position="top-right" className="route-preview-banner"><span>Viewing another search strategy</span><button onClick={() => setPreviewRoute(null)}>Return to this search</button></Panel>}
        </ReactFlow>
      </div>

      <div className="route-atlas-readout">
        <span><small>Search style</small><strong>{shotLabel(query.shot_mode)}</strong><em>{query.seed_count ? `${query.seed_count} known positive example${query.seed_count === 1 ? '' : 's'}` : 'no examples supplied'}</em></span>
        <span><small>Query familiarity</small><strong>{scopeLabel(query.scope || (query.query_is_current_entity ? 'current' : 'external'))}</strong><em>{query.query_nearest_library_id ? `closest known record: ${query.query_nearest_library_id}` : 'identity checked'}</em></span>
        <span><small>Candidates searched</small><strong>{query.direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all'
          ? `${query.candidate_universe_pre_taxonomy_size ?? '—'} → ${query.candidate_universe_size ?? '—'}`
          : query.candidate_universe_size ?? '—'}</strong><em>{query.direction === 'reaction_to_enzyme'
            ? query.enzyme_taxonomy_scope === 'eukaryote' ? 'eukaryotic proteins only' : query.enzyme_taxonomy_scope === 'prokaryote' ? 'prokaryotic proteins only' : 'proteins · unrestricted'
            : 'reactions'}</em></span>
        <span><small>Ranking method</small><strong>{compactRetrieval(query.score_source)}</strong><em>{query.ranking_objective || 'Top-K'}</em></span>
        <span><small>Uncertainty summary</small><strong>{query.shot_mode === 'few_shot' ? 'example-guided; query-only calibration not used' : reliabilitySummary(query.empirical_reliability_status)}</strong><em>{query.conformal_retrieval_set?.set_size ? `review depth: Top ${query.conformal_retrieval_set.set_size}` : 'recall depth not estimated'}</em></span>
      </div>

      <div className="route-atlas-lower">
        <RouteInspector entry={selectedEntry} query={query} actualRouteId={actualRouteId} previewing={previewing} />
        <ModuleInspector moduleId={selectedModule} query={query} />
      </div>
    </div>
  )
}

function RouteModuleNode({ data }: NodeProps<ModuleNode>) {
  const isSection = data.kind === 'section'
  return (
    <article className={`route-module-node kind-${data.kind} state-${data.state} ${data.onActualRoute ? 'actual-route' : ''} ${data.selected ? 'selected' : ''}`} title={data.detail}>
      {!isSection && <Handle type="target" position={Position.Left} className="route-handle" />}
      <div className="route-module-top"><span>{moduleEyebrow(data.kind)}</span><i /></div>
      <div className="route-module-main">
        <ModuleVisual kind={data.kind} state={data.state} />
        <div className="route-module-copy"><strong>{data.title}</strong><small>{data.subtitle}</small></div>
      </div>
      <div className="route-module-metric">{data.metric}</div>
      {!isSection && <Handle type="source" position={Position.Right} className="route-handle" />}
    </article>
  )
}

function RouteRailEdge(props: EdgeProps<RailEdge>) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
    borderRadius: 22,
    offset: 18,
  })
  const data = props.data!
  const className = `route-rail route-rail-${data.tone} state-${data.state} ${data.actual ? 'actual' : ''} ${data.optional ? 'optional' : ''}`
  return (
    <>
      <BaseEdge id={`${props.id}-bed`} path={edgePath} className={`${className} rail-bed`} />
      <BaseEdge id={props.id} path={edgePath} className={`${className} rail-line`} />
      {data.onRoute && data.state !== 'standby' && (
        <g className={`${className} route-train`}>
          <rect width="17" height="7" rx="3.5" fill="currentColor">
            <animateMotion dur={data.state === 'active' ? '1.25s' : '3.2s'} repeatCount="indefinite" path={edgePath} />
          </rect>
        </g>
      )}
      {data.label && (
        <EdgeLabelRenderer>
          <span className={`${className} route-rail-label`} style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}>{data.label}</span>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

function ModuleVisual({ kind, state }: { kind: ModuleKind; state: ModuleState }) {
  const pulse = state === 'active' ? ' visual-active' : ''
  if (kind === 'section') return <div className="visual-section"><i /><i /><i /></div>
  if (kind === 'reaction-input') return <div className={`visual-reaction${pulse}`}><i className="molecule left" /><b>→</b><i className="molecule right" /></div>
  if (kind === 'protein-input') return <div className={`visual-sequence${pulse}`}>{['M', 'E', 'S', 'M', 'C', '…'].map((token, index) => <i key={`${token}-${index}`}>{token}</i>)}</div>
  if (kind === 'shot-gate') return <div className={`visual-shot${pulse}`}><i className="empty-orbit" /><i className="seed a" /><i className="seed b" /><i className="seed c" /></div>
  if (kind === 'scope-gate') return <div className={`visual-scope${pulse}`}><i className="library-ring" /><i className="external-orbit" /><b /></div>
  if (kind === 'reaction-encoder') return <div className={`visual-fingerprint${pulse}`}>{Array.from({ length: 16 }, (_, index) => <i key={index} style={{ height: `${20 + ((index * 17) % 65)}%` }} />)}</div>
  if (kind === 'protein-encoder') return <div className={`visual-embedding${pulse}`}>{Array.from({ length: 24 }, (_, index) => <i key={index} style={{ opacity: 0.22 + ((index * 7) % 10) / 13 }} />)}</div>
  if (kind === 'universe') return <div className={`visual-universe${pulse}`}><i /><i /><i /><i /><i /><b /></div>
  if (kind === 'taxonomy-filter') return <div className={`visual-taxonomy${pulse}`}><span className="tax-source"><i /><i /><i /><i /><i /></span><b className="tax-sieve" /><span className="tax-pass"><i /><i /><i /></span></div>
  if (kind === 'router') return <div className={`visual-switch${pulse}`}><i className="stem" /><i className="branch one" /><i className="branch two" /><i className="branch three" /><b /></div>
  if (kind === 'dual-tower') return <div className={`visual-towers${pulse}`}><i className="tower a"><b /><b /><b /></i><span>shared space</span><i className="tower b"><b /><b /><b /></i></div>
  if (kind === 'loss075') return <div className={`visual-loss${pulse}`}><i style={{ width: '75%' }}>R→E 75%</i><b style={{ width: '25%' }}>E→R 25%</b></div>
  if (kind === 'residual') return <div className={`visual-residual${pulse}`}><i /><svg viewBox="0 0 70 24"><path d="M1 15 C12 2, 22 23, 34 10 S55 4, 69 14" /></svg><b>+</b></div>
  if (kind === 'seed-protein' || kind === 'seed-reaction') return <div className={`visual-seeds ${kind}${pulse}`}><b /><i className="one" /><i className="two" /><i className="three" /><span>closest example</span></div>
  if (kind === 'seed-mask') return <div className={`visual-mask${pulse}`}><i className="seeded" /><i /><i /><i /><b>×</b></div>
  if (kind === 'neighbor') return <div className={`visual-neighbor${pulse}`}><b /><i className="n1" /><i className="n2" /><i className="n3" /><i className="n4" /><i className="n5" /></div>
  if (kind === 'hard-negative') return <div className={`visual-hardneg${pulse}`}>{[72, 58, 83, 41, 66].map((value, index) => <i key={index} style={{ width: `${value}%` }}><b className={index === 2 || index === 4 ? 'hard' : ''} /></i>)}</div>
  if (kind === 'dual-kernel') return <div className={`visual-kernels${pulse}`}><KernelGrid /><span>×</span><KernelGrid /><b className="graph-link" /></div>
  if (kind === 'rrf') return <div className={`visual-rrf${pulse}`}><i><b>1</b><b>3</b><b>2</b></i><span>merge ranks</span><i><b>2</b><b>1</b><b>3</b></i></div>
  if (kind === 'cage') return <div className={`visual-cage${pulse}`}><i /><i /><i /><i /><i /><b>+</b></div>
  if (kind === 'mask-only') return <div className={`visual-mask-only${pulse}`}><i /><i className="blocked" /><i /><b>filter only</b></div>
  if (kind === 'rank-lock') return <div className={`visual-rank${pulse}`}><i style={{ width: '92%' }} /><i style={{ width: '72%' }} /><i style={{ width: '54%' }} /><b>⌑</b></div>
  if (kind === 'trust') return <div className={`visual-trust${pulse}`}><i className="gauge" /><i className="bracket" /><b className="passport">P</b></div>
  return <div className={`visual-output${pulse}`}><i /><i /><i /><b>→</b></div>
}

function KernelGrid() {
  return <i className="kernel-grid">{Array.from({ length: 9 }, (_, index) => <b key={index} style={{ opacity: 0.2 + ((index * 5) % 8) / 10 }} />)}</i>
}

function RouteBoard({
  entries,
  actualRouteId,
  actualBaseRouteId,
  actualPathKey,
  activeOverlayKeys,
  previewRoute,
  onPreview,
}: {
  entries: RouteCatalogEntry[]
  actualRouteId: string
  actualBaseRouteId: string
  actualPathKey: string
  activeOverlayKeys: Set<string>
  previewRoute: string | null
  onPreview: (route: string | null) => void
}) {
  const groups = [
    {
      key: 'r2e-manifest',
      title: 'Start with a reaction: find candidate enzymes',
      note: 'Choose a smaller list for focused testing or a broader list for discovery.',
      entries: entries.filter((entry) => entry.category === 'manifest_route' && entry.direction === 'reaction_to_enzyme'),
    },
    {
      key: 'e2r-manifest',
      title: 'Start with an enzyme: predict possible reactions',
      note: 'Use concise routes for annotation or broader routes to explore catalytic promiscuity.',
      entries: entries.filter((entry) => entry.category === 'manifest_route' && entry.direction === 'enzyme_to_reaction'),
    },
    {
      key: 'seed-paths',
      title: 'Guide the search with known positive examples',
      note: 'Useful when at least one working enzyme or reaction is already known.',
      entries: entries.filter((entry) => entry.category === 'execution_path'),
    },
    {
      key: 'taxonomy',
      title: 'Restrict which enzymes enter reaction → enzyme scoring',
      note: 'These paths shrink the protein candidate universe before any model score is calculated.',
      entries: entries.filter((entry) => entry.key === 'r2e-eukaryote-only-overlay' || entry.key === 'r2e-prokaryote-only-overlay'),
    },
    {
      key: 'modifiers',
      title: 'Other optional filters and specialist search modes',
      note: 'Hide known results, extend the candidate collection or run research-only variants.',
      entries: entries.filter((entry) => (entry.category === 'modifier' || entry.category === 'conditional_path') && !entry.key.includes('eukaryote-only') && !entry.key.includes('prokaryote-only')),
    },
  ]

  return (
    <section className="route-board" aria-label="Available search strategies">
      <div className="route-board-heading">
        <div>
          <span>CHOOSE A SEARCH STRATEGY</span>
          <strong>Each path answers a different experimental question</strong>
          <small>Select a card to see when it is useful and which computational modules it uses.</small>
        </div>
        <div className="route-board-intro-badge">
          <span>Two directions</span>
          <small>reaction → enzyme · enzyme → reaction</small>
        </div>
      </div>
      <div className="route-board-groups">
        {groups.map((group) => (
          <div className={`route-board-group group-${group.key}`} key={group.key}>
            <div className="route-board-group-title"><span>{group.title}</span><small>{group.note}</small></div>
            <div className="route-board-card-row">
              {group.entries.map((entry) => {
                const primaryLive = entry.key === actualPathKey
                const overlayLive = activeOverlayKeys.has(entry.key)
                const live = primaryLive || overlayLive
                const context = entry.route_id === actualBaseRouteId && !primaryLive
                const preview = entry.key === previewRoute
                const displayId = primaryLive ? actualRouteId : entryRouteId(entry)
                return (
                  <button
                    key={entry.key}
                    className={`route-board-card ${live ? 'live' : ''} ${context ? 'context' : ''} ${preview ? 'preview' : ''} ${entry.shot_mode === 'few_shot' ? 'few' : ''} availability-${entry.availability || 'portal'}`}
                    onClick={() => onPreview(preview ? null : entry.key)}
                    aria-pressed={preview}
                    title={`${entry.use_case} ${entry.description}`}
                  >
                    <div className="route-board-card-top">
                      <span>{routeCategoryLabel(entry)}</span>
                      <div>
                        {context && <em>BASE STRATEGY</em>}
                        {live && <em className="live-badge">ACTIVE</em>}
                        {!live && entry.availability !== 'portal' && <em>{entry.availability === 'cli_only' ? 'SPECIALIST OPTION' : entry.availability === 'batch_only' ? 'BATCH WORKFLOW' : 'WHEN AVAILABLE'}</em>}
                      </div>
                    </div>
                    <code>{displayId}</code>
                    <RouteMiniPath modules={entry.modules} direction={entry.direction} few={entry.shot_mode === 'few_shot'} />
                    <small>{entryRouteMeta(entry)}</small>
                    <p>{entry.use_case}</p>
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function routeCategoryLabel(entry: RouteCatalogEntry) {
  if (entry.category === 'execution_path') return 'EXAMPLE-GUIDED SEARCH'
  if (entry.category === 'conditional_path') return 'OPTIONAL RESCUE STEP'
  if (entry.category === 'modifier') return 'OPTIONAL SEARCH MODIFIER'
  return entry.scope === 'current' ? 'QUERY ALREADY REPRESENTED' : 'NEW OR UNREGISTERED QUERY'
}

function RouteMiniPath({ modules, direction, few }: { modules: string[]; direction: RouteCatalogEntry['direction']; few: boolean }) {
  const compact = modules.length > 8
    ? [...modules.slice(0, 3), 'ellipsis', ...modules.slice(-4)]
    : modules
  return (
    <span className={`route-mini-path ${direction === 'reaction_to_enzyme' ? 'r2e' : 'e2r'} ${few ? 'few' : ''}`} aria-hidden="true">
      {compact.map((module, index) => (
        <span className="route-mini-segment" key={`${module}-${index}`}>
          {index > 0 && <i className="route-mini-rail" />}
          <b className={`route-mini-station station-${miniStationKind(module)}`} />
        </span>
      ))}
    </span>
  )
}

function miniStationKind(module: string) {
  if (module === 'ellipsis') return 'ellipsis'
  if (module.includes('query')) return 'input'
  if (module.includes('encoder')) return 'encoder'
  if (module.includes('router')) return 'router'
  if (module.includes('rrf')) return 'fusion'
  if (module.includes('seed')) return 'seed'
  if (module.includes('trust')) return 'trust'
  if (module.includes('rank')) return 'rank'
  if (module.includes('output')) return 'output'
  if (module.includes('kernel') || module.includes('hardneg') || module.includes('residual') || module.includes('loss075') || module.includes('shared') || module.includes('neighbor') || module.includes('current')) return 'model'
  return 'control'
}

function RouteInspector({ entry, query, actualRouteId, previewing }: { entry: RouteCatalogEntry | null; query: QueryMetadata; actualRouteId: string; previewing: boolean }) {
  const routeId = previewing && entry ? entryRouteId(entry) : actualRouteId
  const settings = entry?.settings ? Object.entries(entry.settings) : []
  const direction = entry?.direction || query.direction
  const scope = entry?.scope === 'any' ? query.scope : entry?.scope || query.scope
  const shotMode = entry?.shot_mode || query.shot_mode
  const objective = entry?.objective || query.ranking_objective
  const useCase = entry?.use_case || actualRouteUseCase(query)
  const method = entry?.description || actualRouteExplanation(query)
  return (
    <section className="route-inspector">
      <div className="route-lower-heading"><span>{previewing ? 'SEARCH STRATEGY PREVIEW' : 'WHY THIS STRATEGY WAS USED'}</span><strong>{compactRetrieval(entry?.retrieval || query.score_source)}</strong></div>
      <h3>{routeId}</h3>
      <div className="route-use-case"><small>Best used when</small><p>{useCase}</p></div>
      <div className="route-method"><small>How it works</small><p>{method}</p></div>
      <div className="route-inspector-grid">
        <span><small>Scientific direction</small><strong>{directionLabel(direction)}</strong></span>
        <span><small>Query status</small><strong>{scopeLabel(scope)}</strong></span>
        <span><small>Starting knowledge</small><strong>{shotLabel(shotMode)}</strong></span>
        <span><small>Results requested</small><strong>{objectiveLabel(objective)}</strong></span>
      </div>
      <div className="route-uncertainty-note">{routeUncertaintyMessage(entry, query)}</div>
      {(settings.length > 0 || entry?.availability !== 'portal') && (
        <details className="route-technical-details">
          <summary>Technical route details</summary>
          {settings.length > 0 && <div className="route-setting-chips">{settings.map(([key, value]) => <em key={key}>{humanize(key)} {String(value)}</em>)}</div>}
          <small>{entry?.availability === 'cli_only' ? 'This option is intended for specialist command-line analysis.' : entry?.availability === 'batch_only' ? 'This path is used by registry-wide batch discovery rather than the interactive single-query form.' : entry?.availability === 'conditional' ? 'This step appears only when its supporting evidence is available.' : 'Available in the interactive search.'}</small>
        </details>
      )}
    </section>
  )
}

function ModuleInspector({ moduleId, query }: { moduleId: string | null; query: QueryMetadata }) {
  const module = moduleId ? MODULE_INFO.get(moduleId) : null
  return (
    <section className="module-inspector">
      <div className="route-lower-heading"><span>WHAT THIS MODULE DOES</span><strong>{module ? moduleEyebrow(module.kind) : 'select a module'}</strong></div>
      {module ? <>
        <h3>{module.title}</h3>
        <p>{module.detail}</p>
        <dl>
          <div><dt>Receives</dt><dd>{moduleInput(module.kind)}</dd></div>
          <div><dt>Produces</dt><dd>{moduleOutput(module.kind)}</dd></div>
          <div><dt>Used in this search?</dt><dd>{ROUTE_PATHS[actualRouteKey(query)]?.includes(module.id) ? 'Yes — this query passed through this module.' : 'No — it belongs to another available strategy.'}</dd></div>
        </dl>
      </> : <p>Select any module in the diagram to see what information enters it, what calculation it performs and what it passes to the next step.</p>}
    </section>
  )
}

function chainRails(prefix: string, nodes: string[], routes: string[], tone: RailData['tone'], phaseOffset = 1): RailSpec[] {
  return nodes.slice(0, -1).map((source, index) => ({ id: `${prefix}-${index}`, source, target: nodes[index + 1], routes, phase: phaseOffset + index, tone }))
}

function moduleState(phase: number, onRoute: boolean, playhead: number): ModuleState {
  if (!onRoute || phase < 0) return 'standby'
  if (phase < playhead) return 'complete'
  if (phase === playhead) return 'active'
  return 'queued'
}

function actualRouteKey(query: QueryMetadata) {
  if (query.shot_mode === 'few_shot' || query.score_source === 'seed' || query.route_id?.includes('+fewshot')) {
    return query.direction === 'reaction_to_enzyme' ? 'r2e-fewshot-seed' : 'e2r-fewshot-seed'
  }
  return query.route_id?.split('+')[0] || (query.direction === 'reaction_to_enzyme' ? 'r2e-current-top10-v1' : 'e2r-external-top20-dual-kernel-rrf-v1')
}

function dynamicMetric(spec: ModuleSpec, query: QueryMetadata, candidates: Candidate[], onActualRoute: boolean) {
  if (!onActualRoute) return spec.metric
  if (spec.id.endsWith('-query')) return query.query_id || spec.metric
  if (spec.kind === 'shot-gate') return query.shot_mode === 'few_shot' ? `${query.seed_count || 0} known positive example${query.seed_count === 1 ? '' : 's'}` : 'query only · no examples'
  if (spec.kind === 'scope-gate') return query.query_is_current_entity ? 'already represented in reference data' : 'new or unregistered query'
  if (spec.kind === 'universe') {
    const before = query.candidate_universe_pre_taxonomy_size ?? query.candidate_universe_size
    return `${before ?? '—'} candidates loaded`
  }
  if (spec.kind === 'taxonomy-filter') {
    const before = query.candidate_universe_pre_taxonomy_size ?? query.candidate_universe_size
    const after = query.candidate_universe_post_taxonomy_size ?? query.candidate_universe_size
    const scope = query.enzyme_taxonomy_scope || 'all'
    return scope === 'all' ? `${before ?? '—'} → ${after ?? '—'} · unrestricted` : `${before ?? '—'} → ${after ?? '—'} · ${scope} only`
  }
  if (spec.kind === 'router') return `${query.ranking_objective || 'Top-K'} · ${query.route_id || 'route pending'}`
  if (spec.kind === 'seed-protein' || spec.kind === 'seed-reaction') return `${query.seed_count || 0} example${query.seed_count === 1 ? '' : 's'} · closest match used`
  if (spec.kind === 'seed-mask') return `${query.seed_count || 0} supplied example${query.seed_count === 1 ? '' : 's'} removed`
  if (spec.kind === 'mask-only') return spec.id === 'r2e-known-mask' ? spec.metric : `${query.mask_count || 0} reactions hidden`
  if (spec.kind === 'rank-lock') return `${candidates.length} returned · ${query.ranking_objective || 'Top-K'}`
  if (spec.kind === 'trust') {
    if (query.shot_mode === 'few_shot') return 'example-guided · query-only uncertainty estimate not used'
    const reliability = query.empirical_reliability_score == null ? humanize(query.empirical_reliability_status) : formatPercent(query.empirical_reliability_score)
    return `${reliability} · review depth ${query.conformal_retrieval_set?.set_size ?? 'not estimated'}`
  }
  if (spec.kind === 'output') return `${candidates.length} candidates handed off`
  return spec.metric
}

function moduleEyebrow(kind: ModuleKind) {
  const labels: Record<ModuleKind, string> = {
    section: 'SEARCH DIRECTION', 'reaction-input': 'CHEMICAL INPUT', 'protein-input': 'SEQUENCE INPUT', 'shot-gate': 'EXAMPLE CHECK', 'scope-gate': 'REFERENCE CHECK', 'reaction-encoder': 'REACTION FEATURES', 'protein-encoder': 'PROTEIN FEATURES', universe: 'SEARCH COLLECTION', 'taxonomy-filter': 'BIOLOGICAL SCOPE', router: 'STRATEGY SELECTION', 'dual-tower': 'PAIRED NEURAL MODEL', loss075: 'FOCUSED MODEL', residual: 'CORRECTED FINGERPRINT MODEL', 'seed-protein': 'EXAMPLE SIMILARITY', 'seed-reaction': 'EXAMPLE SIMILARITY', 'seed-mask': 'EXAMPLE REMOVAL', neighbor: 'TRANSFER FROM RELATED PROTEINS', 'hard-negative': 'SECOND NEURAL VIEW', 'dual-kernel': 'SIMILARITY NETWORK', rrf: 'RANK-LIST COMBINATION', cage: 'STRUCTURE-INFORMED ADDITION', 'mask-only': 'RESULT FILTER', 'rank-lock': 'CANDIDATE ORDER', trust: 'UNCERTAINTY INTERPRETATION', output: 'EXPERIMENTAL SHORTLIST',
  }
  return labels[kind]
}

function moduleInput(kind: ModuleKind) {
  if (kind === 'reaction-input') return 'a reaction database ID or a substrate-to-product reaction string'
  if (kind === 'protein-input') return 'a protein database ID or an amino-acid sequence'
  if (kind === 'reaction-encoder') return 'a validated chemical transformation'
  if (kind === 'protein-encoder') return 'a validated amino-acid sequence'
  if (kind === 'universe') return 'the full registered collection of possible answers'
  if (kind === 'taxonomy-filter') return 'the full enzyme candidate matrix plus the selected all/eukaryote/prokaryote scope'
  if (kind === 'router') return 'query direction, query status, known examples and requested list size'
  if (kind === 'dual-kernel') return 'protein similarity, reaction similarity and the known association network'
  if (kind === 'rrf') return 'two independently ordered candidate lists'
  if (kind === 'seed-protein' || kind === 'seed-reaction') return 'the query and representations of supplied known positives'
  if (kind === 'trust') return 'the final ranking plus query- and model-agreement diagnostics'
  return 'the output of the preceding search step'
}

function moduleOutput(kind: ModuleKind) {
  if (kind.includes('encoder')) return 'a numerical representation that models can compare'
  if (kind === 'taxonomy-filter') return 'the eligible enzyme matrix that is allowed to enter model scoring'
  if (kind === 'router') return 'the search strategy selected for this query'
  if (kind === 'rrf') return 'one combined ordering built from two rank lists'
  if (kind === 'trust') return 'plain-language estimates of familiarity, ranking stability and review depth'
  if (kind === 'output') return 'a prioritized list of candidates for interpretation or experiments'
  if (kind === 'rank-lock') return 'the final candidate order before explanations are attached'
  if (kind === 'seed-mask' || kind === 'mask-only') return 'a list with supplied or excluded records removed'
  return 'route-specific candidate scores or a control decision'
}

function actualRouteExplanation(query: QueryMetadata) {
  const taxonomy = query.direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all'
    ? ` Before scoring, the enzyme universe was restricted to ${query.enzyme_taxonomy_scope === 'eukaryote' ? 'eukaryotic' : 'prokaryotic'} proteins (${query.candidate_universe_pre_taxonomy_size ?? '—'} → ${query.candidate_universe_size ?? '—'} candidates).`
    : ''
  return `The system selected ${compactRetrieval(query.score_source)} for a ${scopeLabel(query.scope)} ${directionLabel(query.direction)} search with ${shotLabel(query.shot_mode)} and ${objectiveLabel(query.ranking_objective)}.${taxonomy}`
}

function actualRouteUseCase(query: QueryMetadata) {
  const direction = query.direction === 'reaction_to_enzyme'
  const broad = query.ranking_objective === 'top20'
  if (query.shot_mode === 'few_shot') {
    return direction
      ? 'Known working catalysts are available and the goal is to find additional enzymes with related sequence-level evidence.'
      : 'Known reactions are available and the goal is to explore additional activities related to those examples.'
  }
  if (direction) {
    if (query.enzyme_taxonomy_scope === 'eukaryote') return 'A reaction needs candidate enzymes specifically from eukaryotic organisms, with all other and unresolved proteins excluded before scoring.'
    if (query.enzyme_taxonomy_scope === 'prokaryote') return 'A reaction needs candidate enzymes specifically from bacterial, archaeal or cyanobacterial sources, with all other and unresolved proteins excluded before scoring.'
    return broad
      ? 'A reaction needs a broad enzyme screening panel for discovery and experimental design.'
      : 'A reaction needs a focused list of candidate enzymes for initial testing.'
  }
  return broad
    ? 'An enzyme needs a broad map of possible reactions or catalytic promiscuity.'
    : 'An enzyme needs a focused functional annotation or first assay panel.'
}

function routeUncertaintyMessage(entry: RouteCatalogEntry | null, query: QueryMetadata) {
  const shotMode = entry?.shot_mode || query.shot_mode
  const scope = entry?.scope === 'any' ? query.scope : entry?.scope || query.scope
  if (shotMode === 'few_shot') return 'Uncertainty note: this path is guided by supplied examples, so the query-only benchmark estimates of ranking reliability and recall depth are not applied.'
  if (query.direction === 'reaction_to_enzyme' && query.enzyme_taxonomy_scope && query.enzyme_taxonomy_scope !== 'all') return 'Uncertainty note: this path changes the enzyme candidate population before scoring, so unrestricted reliability and conformal recall calibrations are intentionally not applied.'
  if (entry?.availability === 'cli_only') return 'Uncertainty note: specialist route changes are outside the standard benchmark calibration and should be interpreted as research analyses.'
  if (entry?.category === 'modifier') return 'Uncertainty note: filtering or extending the candidate collection can invalidate route-matched recall estimates.'
  if (scope === 'current') return 'Uncertainty note: this query is already represented in the reference data; the external-query recall calibration is not needed or applied.'
  return 'Uncertainty note: eligible new query-only searches can include benchmark-based ranking stability and recall-depth estimates.'
}

function reliabilitySummary(status: string | null | undefined) {
  if (!status) return 'ranking stability not estimated'
  if (status.includes('validated') || status.includes('deployed')) return 'ranking stability estimated from comparable benchmark queries'
  if (status.includes('not_applicable')) return 'benchmark stability estimate not applicable to this search style'
  if (status.includes('incompatible')) return 'available estimate does not match this route'
  return humanize(status)
}
