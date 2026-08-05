import { useEffect, useMemo, useState } from 'react'
import { downloadCsv, downloadJson, loadPortalStatus, runRanking } from './api'
import { CandidateTable } from './components/CandidateTable'
import { DatabaseFrame } from './components/DatabaseFrame'
import { EvidencePanel } from './components/EvidencePanel'
import { FlowTheater } from './components/FlowTheater'
import { QueryPanel } from './components/QueryPanel'
import e2rDemo from './demo/e2r_external_top20.json'
import r2eDemo from './demo/r2e_current_top10.json'
import { humanize } from './flow'
import type { Candidate, PortalStatus, PortalView, QueryForm, RankingResponse, RunState } from './types'

const INITIAL_FORM: QueryForm = {
  direction: 'enzyme_to_reaction',
  entityMode: 'id',
  queryValue: 'A0A023J8Z5',
  topK: 20,
  conformalMode: 'annotate',
  conformalAlpha: 0.1,
}

function App() {
  const [view, setView] = useState<PortalView>('navigator')
  const [form, setForm] = useState<QueryForm>(INITIAL_FORM)
  const [response, setResponse] = useState<RankingResponse>(e2rDemo as RankingResponse)
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>((e2rDemo as RankingResponse).candidates[0] || null)
  const [runState, setRunState] = useState<RunState>('success')
  const [activeStage, setActiveStage] = useState(99)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<PortalStatus | null>(null)

  useEffect(() => {
    loadPortalStatus().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    if (runState !== 'running') return
    const timer = window.setInterval(() => setActiveStage((stage) => Math.min(stage + 1, 11)), 420)
    return () => window.clearInterval(timer)
  }, [runState])

  const resultLabel = useMemo(() => {
    const query = response.query
    return `${query.direction === 'reaction_to_enzyme' ? 'R2E' : 'E2R'} · ${humanize(query.ranking_objective)} · ${response.candidates.length} shown`
  }, [response])

  const handleRun = async () => {
    setRunState('running')
    setActiveStage(0)
    setError(null)
    try {
      const next = await runRanking(form)
      setResponse(next)
      setSelectedCandidate(next.candidates[0] || null)
      setRunState('success')
      setActiveStage(99)
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : 'Unable to run ranking')
      setRunState('error')
      setActiveStage(99)
    }
  }

  const loadDemo = (kind: 'r2e' | 'e2r') => {
    const next = (kind === 'r2e' ? r2eDemo : e2rDemo) as RankingResponse
    setResponse(next)
    setSelectedCandidate(next.candidates[0] || null)
    setRunState('success')
    setError(null)
    setForm(kind === 'r2e'
      ? { ...INITIAL_FORM, direction: 'reaction_to_enzyme', queryValue: 'RHEA:54512', topK: 10 }
      : INITIAL_FORM)
  }

  return (
    <div className="portal-app">
      <header className="portal-header">
        <div className="brand-block">
          <div className="brand-symbol"><span /><span /><span /></div>
          <div>
            <strong>TerpeneNavigator</strong>
            <small>Atlas-compatible model discovery portal</small>
          </div>
        </div>

        <nav className="portal-nav" aria-label="Primary portal navigation">
          <button className={view === 'navigator' ? 'active' : ''} onClick={() => setView('navigator')}>
            <span>01</span> Model Navigator
          </button>
          <button className={view === 'atlas' ? 'active' : ''} onClick={() => setView('atlas')}>
            <span>02</span> Database Atlas
          </button>
        </nav>

        <div className="header-status">
          <span><i className="status-dot" /> {status?.status || 'portal ready'}</span>
          <small>{status?.route_version || response.query.route_version}</small>
        </div>
      </header>

      {view === 'navigator' ? (
        <main className="navigator-page">
          <section className="hero-band">
            <div>
              <span className="hero-eyebrow">DATABASE → MODEL → EVIDENCE → EXPERIMENT</span>
              <h1>Watch a terpene query become an auditable experimental shortlist.</h1>
              <p>Every illuminated lane is a route the production system actually executed. Unused routes remain visibly inactive.</p>
            </div>
            <div className="hero-summary glass-panel">
              <span>Current run</span>
              <strong>{resultLabel}</strong>
              <small>{response.query.route_id}</small>
              <div className="hero-actions">
                <button onClick={() => downloadCsv(response)}>Export CSV</button>
                <button onClick={() => downloadJson(response, `${response.query.query_id || 'query'}.json`)}>Audit JSON</button>
              </div>
            </div>
          </section>

          <section className="navigator-grid">
            <QueryPanel
              form={form}
              runState={runState}
              error={error}
              onChange={setForm}
              onRun={handleRun}
              onLoadDemo={loadDemo}
            />
            <div className="navigator-main">
              <FlowTheater query={response.query} runState={runState} activeStage={activeStage} />
              <EvidencePanel query={response.query} />
              <CandidateTable response={response} selected={selectedCandidate} onSelect={setSelectedCandidate} />
            </div>
          </section>
        </main>
      ) : <DatabaseFrame status={status} />}

      <footer className="portal-footer">
        <span>TerpeneNavigator evidence fields annotate — never overwrite — the locked production ranking.</span>
        <span>Database source remains read-only at {status?.database_commit?.slice(0, 12) || '87b507908441'}.</span>
      </footer>
    </div>
  )
}

export default App
