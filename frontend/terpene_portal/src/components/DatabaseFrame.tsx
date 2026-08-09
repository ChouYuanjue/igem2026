import { useState } from 'react'
import type { PortalStatus } from '../types'
import { ModelDataHub } from './ModelDataHub'

type Props = { status: PortalStatus | null }
type DatabaseView = 'model-data' | 'terpene-map'

export function DatabaseFrame({ status }: Props) {
  const [view, setView] = useState<DatabaseView>('model-data')
  return (
    <section className="database-shell">
      <div className="database-toolbar glass-panel">
        <div>
          <span className="section-kicker">EXPLORE THE DATA BEHIND THE SEARCH</span>
          <h2>{view === 'model-data' ? 'Proteins, reactions and known links' : 'Interactive terpene relationship map'}</h2>
          <p>{view === 'model-data'
            ? 'Browse the protein and reaction collections used by the retrieval system, together with known enzyme–reaction associations and mechanism links.'
            : 'Explore curated compounds, enzymes and reaction connections as an interactive network. Select nodes and edges to inspect the available records.'}</p>
        </div>
        <div className="database-toolbar-actions">
          <div className="database-view-switch" role="tablist" aria-label="Choose a data view">
            <button className={view === 'model-data' ? 'active' : ''} onClick={() => setView('model-data')}><span>01</span> Model datasets</button>
            <button className={view === 'terpene-map' ? 'active' : ''} onClick={() => setView('terpene-map')}><span>02</span> Terpene map</button>
          </div>
          <div className="database-statuses">
            <span><i className="status-dot" /> {view === 'model-data' ? 'Data available' : status?.database_mode === 'unavailable' ? 'Map unavailable' : 'Interactive map ready'}</span>
            {view === 'terpene-map' && <a href="/database/" target="_blank" rel="noreferrer">Open map full screen ↗</a>}
          </div>
        </div>
      </div>

      {view === 'model-data' ? <ModelDataHub /> : (
        <div className="database-frame-wrap">
          <iframe
            title="Interactive terpene relationship map"
            src="/database/"
            className="database-frame"
            sandbox="allow-scripts allow-same-origin allow-forms allow-downloads allow-popups"
          />
        </div>
      )}
    </section>
  )
}
