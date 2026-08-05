import { buildFlowStages, compactRouteName, humanize } from '../flow'
import type { QueryMetadata, RunState } from '../types'
import { ExecutionGraph } from './ExecutionGraph'

type Props = {
  query: QueryMetadata
  runState: RunState
  activeStage: number
}

export function FlowTheater({ query, runState, activeStage }: Props) {
  const stages = buildFlowStages(query, runState === 'running' ? activeStage : 99)

  return (
    <section className="flow-theater glass-panel">
      <div className="flow-header">
        <div>
          <span className="section-kicker">LIVE MODEL DATAFLOW</span>
          <h2>{compactRouteName(query)}</h2>
          <p>The graph illuminates only the production stages and model lanes that were actually executed.</p>
        </div>
        <div className="route-chip-stack">
          <span className="route-chip">{query.direction === 'reaction_to_enzyme' ? 'Reaction → Enzyme' : 'Enzyme → Reaction'}</span>
          <span className="route-chip accent">{humanize(query.ranking_objective)}</span>
        </div>
      </div>

      <ExecutionGraph query={query} runState={runState} activeStage={activeStage} />

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
    </section>
  )
}
