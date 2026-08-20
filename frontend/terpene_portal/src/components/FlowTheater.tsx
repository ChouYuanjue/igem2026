import { buildFlowStages, compactRouteName } from '../flow'
import { directionLabel, objectiveLabel, shotLabel } from '../routePresentation'
import type { Candidate, QueryMetadata, RouteCatalog, RunState } from '../types'
import { RouteAtlas } from './RouteAtlas'

type Props = {
  query: QueryMetadata
  candidates: Candidate[]
  routeCatalog: RouteCatalog | null
  runState: RunState
  activeStage: number
}

export function FlowTheater({ query, candidates, routeCatalog, runState, activeStage }: Props) {
  const stages = buildFlowStages(query, runState === 'running' ? activeStage : 99)

  return (
    <section className="flow-theater glass-panel">
      <div className="flow-header">
        <div>
          <span className="section-kicker">HOW THIS SEARCH WORKS</span>
          <h2>{compactRouteName(query)}</h2>
          <p>The system chooses a search path from the query direction, whether the query is already known, whether examples are supplied and how many results are requested. The highlighted path below shows exactly which modules were used.</p>
        </div>
        <div className="route-chip-stack">
          <span className="route-chip">{directionLabel(query.direction)}</span>
          <span className={`route-chip ${query.shot_mode === 'few_shot' ? 'few-shot-chip' : 'accent'}`}>{shotLabel(query.shot_mode)}</span>
          <span className="route-chip accent">{objectiveLabel(query.ranking_objective)}</span>
        </div>
      </div>

      <RouteAtlas
        query={query}
        candidates={candidates}
        routeCatalog={routeCatalog}
        runState={runState}
        activeStage={activeStage}
      />

      <details className="stage-detail-drawer">
        <summary><span>Step-by-step details</span><small>Open to see the input, transformation and output at each stage</small></summary>
        <div className="stage-strip" aria-label="Detailed workflow stages">
          {stages.map((stage) => (
            <article key={stage.id} className={`stage-card ${stage.state}`}>
              <div className="stage-number">{String(stage.index + 1).padStart(2, '0')}</div>
              <div className="stage-card-copy">
                <span>{stage.eyebrow}</span>
                <strong>{stage.title}</strong>
                <p>{stage.detail}</p>
              </div>
              <em>{stage.metric}</em>
            </article>
          ))}
        </div>
      </details>
    </section>
  )
}
