import { useEffect, useMemo, useState } from 'react'
import { buildFlowLanes, formatPercent, humanize } from '../flow'
import type { FlowLane, QueryMetadata, RunState } from '../types'

type Props = {
  query: QueryMetadata
  runState: RunState
  activeStage: number
}

type NodeState = 'complete' | 'active' | 'queued' | 'skipped'
type GraphNode = {
  id: string
  x: number
  y: number
  width: number
  height: number
  stage: number
  eyebrow: string
  title: string
  metric: string
  tone: string
  state?: NodeState
  detail?: string
}
type GraphEdge = {
  id: string
  from: string
  to: string
  stage: number
  label?: string
  active: boolean
  tone?: string
  dashed?: boolean
}

const VIEW_WIDTH = 1700
const VIEW_HEIGHT = 470

export function ExecutionGraph({ query, runState, activeStage }: Props) {
  const [replayStage, setReplayStage] = useState<number | null>(null)
  useEffect(() => {
    if (replayStage == null) return
    if (replayStage >= 11) {
      const done = window.setTimeout(() => setReplayStage(null), 750)
      return () => window.clearTimeout(done)
    }
    const timer = window.setTimeout(() => setReplayStage((stage) => (stage == null ? null : stage + 1)), 520)
    return () => window.clearTimeout(timer)
  }, [replayStage])

  const playhead = runState === 'running' ? activeStage : replayStage ?? 99
  const lanes = useMemo(() => buildFlowLanes(query), [query])
  const graph = useMemo(() => buildGraph(query, lanes, playhead), [query, lanes, playhead])

  return (
    <div className="execution-graph-shell">
      <div className="execution-graph-toolbar">
        <div className="graph-legend">
          <span><i className="legend-dot complete" /> executed</span>
          <span><i className="legend-dot active" /> flowing now</span>
          <span><i className="legend-dot skipped" /> not executed</span>
        </div>
        <button className="replay-flow-button" onClick={() => setReplayStage(0)} disabled={runState === 'running'}>
          <span>▶</span> Replay data flow
        </button>
      </div>
      <div className="execution-graph-scroll">
        <svg className="execution-graph" viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`} role="img" aria-label="Production model execution graph">
          <defs>
            <marker id="arrow-cyan" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" /></marker>
            <marker id="arrow-muted" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" /></marker>
            <filter id="edge-glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3.2" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          </defs>

          <g className="graph-zone retrieval-zone"><rect x="810" y="26" width="215" height="340" rx="24" /><text x="830" y="52">PARALLEL RETRIEVAL LANES</text></g>
          <g className="graph-zone evidence-zone"><rect x="1335" y="26" width="190" height="340" rx="24" /><text x="1355" y="52">TRUST & EVIDENCE</text></g>
          <g className="graph-zone optional-zone"><rect x="1075" y="390" width="450" height="62" rx="18" /><text x="1095" y="414">ON-DEMAND AUDITS · VISIBLE BUT NOT CLAIMED AS EXECUTED</text></g>

          <g className="graph-edges">
            {graph.edges.map((edge) => {
              const from = graph.nodeMap.get(edge.from)!
              const to = graph.nodeMap.get(edge.to)!
              const path = curveBetween(from, to)
              const state = edgeState(edge, playhead)
              return (
                <g key={edge.id} className={`data-edge-group ${state} ${edge.tone || ''} ${edge.dashed ? 'dashed' : ''}`}>
                  <path id={`path-${edge.id}`} d={path} className="data-edge-glow" />
                  <path d={path} className="data-edge-line" markerEnd={`url(#${state === 'queued' || state === 'skipped' ? 'arrow-muted' : 'arrow-cyan'})`} />
                  {edge.label && <text className="data-edge-label" x={(from.x + from.width + to.x) / 2} y={(from.y + from.height / 2 + to.y + to.height / 2) / 2 - 7}>{edge.label}</text>}
                  {(state === 'active' || state === 'complete') && !edge.dashed && (
                    <circle className={`data-packet ${state}`} r={state === 'active' ? 5 : 3.5}>
                      <animateMotion dur={state === 'active' ? '1.2s' : '2.8s'} repeatCount="indefinite" path={path} />
                    </circle>
                  )}
                </g>
              )
            })}
          </g>

          <g className="graph-nodes">
            {graph.nodes.map((node) => <ExecutionNode key={node.id} node={node} />)}
          </g>
        </svg>
      </div>
      <div className="execution-readout">
        <span><strong>{query.candidate_universe_size ?? '—'}</strong> candidates assembled</span>
        <span><strong>{query.ranking_objective || 'Top-K'}</strong> locked objective</span>
        <span><strong>{formatPercent(query.evidence_passport?.applicability_score)}</strong> applicability</span>
        <span><strong>{query.conformal_retrieval_set?.set_size ?? '—'}</strong> conformal prefix</span>
      </div>
    </div>
  )
}

function ExecutionNode({ node }: { node: GraphNode }) {
  return (
    <foreignObject x={node.x} y={node.y} width={node.width} height={node.height}>
      <article className={`execution-node ${node.tone} ${node.state || 'queued'}`} title={node.detail || node.title}>
        <div className="execution-node-top"><span>{node.eyebrow}</span><i /></div>
        <strong>{node.title}</strong>
        <small>{node.metric}</small>
      </article>
    </foreignObject>
  )
}

function buildGraph(query: QueryMetadata, lanes: FlowLane[], playhead: number) {
  const hasFusion = lanes.filter((lane) => lane.active).length > 1 || Boolean(query.score_source?.includes('rrf'))
  const conformal = query.conformal_retrieval_set
  const nodeState = (stage: number, forced?: NodeState): NodeState => forced || (stage < playhead ? 'complete' : stage === playhead ? 'active' : 'queued')
  const nodes: GraphNode[] = [
    { id: 'input', x: 18, y: 145, width: 130, height: 82, stage: 0, eyebrow: '01 INPUT', title: query.query_id || 'Scientific query', metric: query.direction === 'reaction_to_enzyme' ? 'reaction' : 'protein', tone: 'input', state: nodeState(0) },
    { id: 'identity', x: 180, y: 145, width: 140, height: 82, stage: 1, eyebrow: '02 RESOLVE', title: query.route_id?.includes('external') ? 'Open-world entity' : 'Library entity', metric: query.query_nearest_library_id || 'identity audit', tone: 'identity', state: nodeState(1) },
    { id: 'represent', x: 352, y: 145, width: 145, height: 82, stage: 2, eyebrow: '03 ENCODE', title: query.direction === 'reaction_to_enzyme' ? 'Reaction views' : 'Protein embedding', metric: query.direction === 'reaction_to_enzyme' ? 'canonical + DRFP' : 'ESM-C · 1152d', tone: 'represent', state: nodeState(2) },
    { id: 'router', x: 530, y: 145, width: 145, height: 82, stage: 3, eyebrow: '04 ROUTE', title: 'Objective router', metric: query.ranking_objective || 'Top-K', tone: 'router', state: nodeState(3), detail: query.route_id },
    { id: 'universe', x: 708, y: 145, width: 145, height: 82, stage: 4, eyebrow: '05 ASSEMBLE', title: 'Candidate universe', metric: `${query.candidate_universe_size ?? '—'} entities`, tone: 'universe', state: nodeState(4), detail: query.candidate_universe_version },
  ]

  const laneY = [64, 158, 252]
  lanes.slice(0, 3).forEach((lane, index) => nodes.push({
    id: `lane-${lane.id}`, x: 835, y: laneY[index], width: 165, height: 76, stage: 5,
    eyebrow: `${String(index + 1).padStart(2, '0')} ${lane.active ? 'EXECUTED' : 'SKIPPED'}`,
    title: lane.title.replace(' lane', ''), metric: lane.active ? `${lane.weight == null ? 'validated route' : `${Math.round(lane.weight * 100)}% fusion weight`}` : 'not executed',
    tone: lane.tone, state: lane.active ? nodeState(5) : 'skipped', detail: lane.subtitle,
  }))

  nodes.push(
    { id: 'fusion', x: 1060, y: 145, width: 125, height: 82, stage: 6, eyebrow: '06 MERGE', title: hasFusion ? 'RRF chamber' : 'Direct pass', metric: hasFusion ? 'rank fusion' : 'single route', tone: 'fusion', state: nodeState(6) },
    { id: 'rank', x: 1218, y: 145, width: 125, height: 82, stage: 7, eyebrow: '07 LOCK', title: 'Production rank', metric: `${query.ranking_objective || 'Top-K'} preserved`, tone: 'rank', state: nodeState(7) },
    { id: 'applicability', x: 1360, y: 64, width: 140, height: 76, stage: 8, eyebrow: '08 DOMAIN', title: humanize(query.evidence_passport?.applicability_tier), metric: formatPercent(query.evidence_passport?.applicability_score), tone: 'applicability', state: nodeState(8) },
    { id: 'conformal', x: 1360, y: 158, width: 140, height: 76, stage: 9, eyebrow: '09 COVERAGE', title: 'Conformal set', metric: conformal?.set_size ? `${conformal.set_size} candidates` : 'not available', tone: 'conformal', state: conformal?.set_size ? nodeState(9) : 'skipped' },
    { id: 'passport', x: 1360, y: 252, width: 140, height: 76, stage: 10, eyebrow: '10 EVIDENCE', title: 'Evidence passport', metric: query.evidence_passport?.version ? 'candidate-level audit' : 'not available', tone: 'passport', state: query.evidence_passport?.version ? nodeState(10) : 'skipped' },
    { id: 'output', x: 1542, y: 145, width: 140, height: 82, stage: 11, eyebrow: '11 HANDOFF', title: 'Experimental shortlist', metric: `${conformal?.requested_top_k || query.ranking_objective || 'Top-K'} returned`, tone: 'output', state: nodeState(11) },
    { id: 'cycle', x: 1100, y: 405, width: 170, height: 38, stage: 10, eyebrow: 'OPTIONAL', title: 'Cycle consistency', metric: 'on-demand · evidence only', tone: 'optional', state: 'skipped' },
    { id: 'mechanism', x: 1300, y: 405, width: 190, height: 38, stage: 10, eyebrow: 'OPTIONAL', title: 'Mechanism trace', metric: 'on-demand · coverage aware', tone: 'optional', state: 'skipped' },
  )

  const activeLanes = lanes.slice(0, 3)
  const edges: GraphEdge[] = [
    { id: 'input-identity', from: 'input', to: 'identity', stage: 1, label: 'validate', active: true },
    { id: 'identity-represent', from: 'identity', to: 'represent', stage: 2, label: 'resolve', active: true },
    { id: 'represent-router', from: 'represent', to: 'router', stage: 3, label: 'vector', active: true },
    { id: 'router-universe', from: 'router', to: 'universe', stage: 4, label: query.route_id?.includes('external') ? 'open world' : 'current', active: true },
  ]
  activeLanes.forEach((lane) => {
    edges.push({ id: `universe-${lane.id}`, from: 'universe', to: `lane-${lane.id}`, stage: 5, label: lane.active && lane.weight != null ? `${Math.round(lane.weight * 100)}%` : undefined, active: lane.active, tone: lane.tone, dashed: !lane.active })
    edges.push({ id: `${lane.id}-fusion`, from: `lane-${lane.id}`, to: 'fusion', stage: 6, active: lane.active, tone: lane.tone, dashed: !lane.active })
  })
  edges.push(
    { id: 'fusion-rank', from: 'fusion', to: 'rank', stage: 7, label: hasFusion ? 'RRF' : 'direct', active: true },
    { id: 'rank-applicability', from: 'rank', to: 'applicability', stage: 8, active: true },
    { id: 'rank-conformal', from: 'rank', to: 'conformal', stage: 9, active: Boolean(conformal?.set_size), dashed: !conformal?.set_size },
    { id: 'rank-passport', from: 'rank', to: 'passport', stage: 10, active: Boolean(query.evidence_passport?.version), dashed: !query.evidence_passport?.version },
    { id: 'applicability-output', from: 'applicability', to: 'output', stage: 11, active: true },
    { id: 'conformal-output', from: 'conformal', to: 'output', stage: 11, active: Boolean(conformal?.set_size), dashed: !conformal?.set_size },
    { id: 'passport-output', from: 'passport', to: 'output', stage: 11, active: Boolean(query.evidence_passport?.version), dashed: !query.evidence_passport?.version },
    { id: 'rank-cycle', from: 'rank', to: 'cycle', stage: 10, active: false, dashed: true },
    { id: 'rank-mechanism', from: 'rank', to: 'mechanism', stage: 10, active: false, dashed: true },
  )
  return { nodes, edges, nodeMap: new Map(nodes.map((node) => [node.id, node])) }
}

function curveBetween(from: GraphNode, to: GraphNode) {
  const x1 = from.x + from.width
  const y1 = from.y + from.height / 2
  const x2 = to.x
  const y2 = to.y + to.height / 2
  const bend = Math.max(36, Math.abs(x2 - x1) * 0.45)
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`
}

function edgeState(edge: GraphEdge, playhead: number): NodeState {
  if (!edge.active) return 'skipped'
  if (edge.stage < playhead) return 'complete'
  if (edge.stage === playhead) return 'active'
  return 'queued'
}
