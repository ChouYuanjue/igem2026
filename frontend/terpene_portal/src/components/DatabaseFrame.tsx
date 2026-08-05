import { useState } from 'react'
import type { PortalStatus } from '../types'
import { ModelDataHub } from './ModelDataHub'

type Props = { status: PortalStatus | null }
type DatabaseView = 'model-data' | 'upstream-atlas'

export function DatabaseFrame({ status }: Props) {
  const [view, setView] = useState<DatabaseView>('model-data')
  return (
    <section className="database-shell">
      <div className="database-toolbar glass-panel">
        <div>
          <span className="section-kicker">TWO READ-ONLY DATA SURFACES</span>
          <h2>{view === 'model-data' ? 'Model Data Hub' : 'Terpene Atlas database frontend'}</h2>
          <p>{view === 'model-data'
            ? 'Our production model datasets are exposed through an external read-only adapter; the database branch remains untouched.'
            : 'The original built interface is displayed unchanged. This portal does not modify or complete the database team’s branch.'}</p>
        </div>
        <div className="database-toolbar-actions">
          <div className="database-view-switch" role="tablist" aria-label="Database views">
            <button className={view === 'model-data' ? 'active' : ''} onClick={() => setView('model-data')}><span>01</span> Model Data Hub</button>
            <button className={view === 'upstream-atlas' ? 'active' : ''} onClick={() => setView('upstream-atlas')}><span>02</span> Upstream Atlas</button>
          </div>
          <div className="database-statuses">
            <span><i className="status-dot" /> {view === 'model-data' ? 'Live model files' : status?.database_mode === 'proxy' ? 'Live database API proxy' : 'Compatibility snapshot'}</span>
            <span>{view === 'model-data' ? 'read-only adapter' : `commit ${status?.database_commit?.slice(0, 8) || '87b50790'}`}</span>
            {view === 'upstream-atlas' && <a href="/database/" target="_blank" rel="noreferrer">Open full screen ↗</a>}
          </div>
        </div>
      </div>

      {view === 'model-data' ? <ModelDataHub /> : <>
        <div className="database-frame-wrap">
          <iframe
            title="Terpene Atlas database frontend"
            src="/database/"
            className="database-frame"
            sandbox="allow-scripts allow-same-origin allow-forms allow-downloads allow-popups"
          />
        </div>
        <div className="database-boundary-note">
          <strong>Boundary:</strong> map, search, edge expansion and detail views are shown because they already exist upstream. Features not completed by the database team are not reimplemented here.
        </div>
      </>}
    </section>
  )
}
