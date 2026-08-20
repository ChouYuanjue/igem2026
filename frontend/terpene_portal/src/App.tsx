import { useEffect, useMemo, useState } from 'react'
import { downloadCsv, downloadJson, loadPortalStatus, loadRouteCatalog, runRanking } from './api'
import { CandidateTable } from './components/CandidateTable'
import { DatabaseFrame } from './components/DatabaseFrame'
import { EvidencePanel } from './components/EvidencePanel'
import { FlowTheater } from './components/FlowTheater'
import { QueryPanel } from './components/QueryPanel'
import e2rDemo from './demo/e2r_external_top20.json'
import r2eDemo from './demo/r2e_current_top10.json'
import { queryRouteId, queryRouteMeta } from './routePresentation'
import type { Candidate, PortalStatus, PortalView, QueryForm, RankingResponse, RouteCatalog, RunState } from './types'

const INITIAL_FORM: QueryForm = {
  direction: 'enzyme_to_reaction',
  shotMode: 'zero_shot',
  entityMode: 'id',
  queryValue: 'A0A023J8Z5',
  seedIdsText: '',
  maskIdsText: '',
  enzymeTaxonomyScope: 'all',
  topK: 20,
  conformalMode: 'annotate',
  conformalAlpha: 0.1,
}

const E2R_DEMO = normalizeDemo(e2rDemo as RankingResponse)
const R2E_DEMO = normalizeDemo(r2eDemo as RankingResponse)

type PresetKind = 'r2e-zero' | 'e2r-zero' | 'r2e-few' | 'e2r-few'

function App() {
  const [view, setView] = useState<PortalView>('navigator')
  const [form, setForm] = useState<QueryForm>(INITIAL_FORM)
  const [response, setResponse] = useState<RankingResponse>(E2R_DEMO)
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(E2R_DEMO.candidates[0] || null)
  const [runState, setRunState] = useState<RunState>('success')
  const [activeStage, setActiveStage] = useState(99)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<PortalStatus | null>(null)
  const [routeCatalog, setRouteCatalog] = useState<RouteCatalog | null>(null)

  useEffect(() => {
    Promise.all([loadPortalStatus(), loadRouteCatalog()])
      .then(([nextStatus, nextRoutes]) => {
        setStatus(nextStatus)
        setRouteCatalog(nextRoutes)
      })
      .catch(() => {
        setStatus(null)
        setRouteCatalog(null)
      })
  }, [])

  useEffect(() => {
    if (runState !== 'running') return
    const timer = window.setInterval(() => setActiveStage((stage) => Math.min(stage + 1, 11)), 480)
    return () => window.clearInterval(timer)
  }, [runState])

  const currentRouteId = useMemo(() => queryRouteId(response.query), [response.query])
  const currentRouteMeta = useMemo(
    () => queryRouteMeta(response.query, response.candidates.length),
    [response],
  )

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

  const loadPreset = (kind: PresetKind) => {
    setRunState('success')
    setError(null)
    if (kind === 'r2e-zero') {
      setForm({ ...INITIAL_FORM, direction: 'reaction_to_enzyme', queryValue: 'RHEA:54512', topK: 10 })
      setResponse(R2E_DEMO)
      setSelectedCandidate(R2E_DEMO.candidates[0] || null)
      return
    }
    if (kind === 'e2r-zero') {
      setForm(INITIAL_FORM)
      setResponse(E2R_DEMO)
      setSelectedCandidate(E2R_DEMO.candidates[0] || null)
      return
    }
    if (kind === 'r2e-few') {
      setForm({
        ...INITIAL_FORM,
        direction: 'reaction_to_enzyme',
        shotMode: 'few_shot',
        queryValue: 'RHEA:54512',
        seedIdsText: 'A0A075FBG7',
        topK: 10,
        conformalMode: 'disabled',
      })
      return
    }
    setForm({
      ...INITIAL_FORM,
      direction: 'enzyme_to_reaction',
      shotMode: 'few_shot',
      queryValue: 'A0A075FBG7',
      seedIdsText: 'RHEA:54512',
      topK: 10,
      conformalMode: 'disabled',
    })
  }

  return (
    <div className="portal-app">
      <header className="portal-header">
        <div className="brand-block">
          <div className="brand-symbol"><span /><span /><span /></div>
          <div>
            <strong>TerpeneNavigator</strong>
            <small>Bidirectional discovery for terpene enzymes and reactions</small>
          </div>
        </div>

        <nav className="portal-nav" aria-label="Primary portal navigation">
          <button className={view === 'navigator' ? 'active' : ''} onClick={() => setView('navigator')}>
            <span>01</span> Search & explain
          </button>
          <button className={view === 'atlas' ? 'active' : ''} onClick={() => setView('atlas')}>
            <span>02</span> Explore data
          </button>
        </nav>

        <div className="header-status">
          <span><i className="status-dot" /> {status?.status === 'ready' ? 'system ready' : status?.status || 'system ready'}</span>
          <small>reaction ↔ enzyme discovery</small>
        </div>
      </header>

      {view === 'navigator' ? (
        <main className="navigator-page">
          <section className="hero-band">
            <div>
              <span className="hero-eyebrow">FROM A REACTION TO AN ENZYME — OR BACK AGAIN</span>
              <h1>Find terpene synthases for reactions, and possible reactions for terpene synthases.</h1>
              <p>Enter a reaction or protein, choose whether known examples are available, and follow the exact search strategy used to rank candidates. Each result includes an explanation of model support, ranking stability and remaining uncertainty.</p>
            </div>
            <div className="hero-summary glass-panel">
              <span>Active search</span>
              <strong>{currentRouteId}</strong>
              <small>{currentRouteMeta}</small>
              <div className="hero-actions">
                <button onClick={() => downloadCsv(response)}>Export CSV</button>
                <button onClick={() => downloadJson(response, `${response.query.query_id || 'query'}.json`)}>Download full result JSON</button>
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
              onLoadPreset={loadPreset}
              taxonomySummary={routeCatalog?.taxonomy_scope || null}
            />
            <div className="navigator-main">
              <FlowTheater
                query={response.query}
                candidates={response.candidates}
                routeCatalog={routeCatalog}
                runState={runState}
                activeStage={activeStage}
              />
              <EvidencePanel query={response.query} />
              <CandidateTable response={response} selected={selectedCandidate} onSelect={setSelectedCandidate} />
            </div>
          </section>
        </main>
      ) : <DatabaseFrame status={status} />}

      <footer className="portal-footer">
        <span>Ranked candidates are hypotheses for experimental follow-up, not probabilities of catalytic success.</span>
        <span>Browse the protein, reaction and association data that support each search.</span>
      </footer>
    </div>
  )
}

function normalizeDemo(response: RankingResponse): RankingResponse {
  const current = Boolean(response.query.route_id?.includes('-current-'))
  return {
    ...response,
    query: {
      ...response.query,
      query_is_current_entity: current,
      scope: current ? 'current' : 'external',
      requested_shot_mode: 'zero_shot',
      shot_mode: 'zero_shot',
      seed_ids: [],
      seed_count: 0,
      mask_ids: [],
      mask_count: 0,
    },
  }
}

export default App
