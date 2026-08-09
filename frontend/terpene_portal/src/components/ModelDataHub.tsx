import { FormEvent, useEffect, useMemo, useState } from 'react'
import { loadModelDataGraph, loadModelDataSummary, searchModelData } from '../api'
import type { ModelDataGraph, ModelDataNode, ModelDataSearchItem, ModelDataSummary } from '../types'

type PositionedNode = ModelDataNode & { x: number; y: number; width: number; height: number }

export function ModelDataHub() {
  const [summary, setSummary] = useState<ModelDataSummary | null>(null)
  const [graph, setGraph] = useState<ModelDataGraph | null>(null)
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState('all')
  const [results, setResults] = useState<ModelDataSearchItem[]>([])
  const [selected, setSelected] = useState<ModelDataNode | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([loadModelDataSummary(), loadModelDataGraph('', '', 28)])
      .then(([nextSummary, nextGraph]) => {
        setSummary(nextSummary)
        setGraph(nextGraph)
        setSelected(nextGraph.nodes[0] || null)
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : 'Unable to load model data'))
      .finally(() => setLoading(false))
  }, [])

  const submitSearch = async (event: FormEvent) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const [search, nextGraph] = await Promise.all([
        searchModelData(query, kind, 40),
        loadModelDataGraph(query, '', 32),
      ])
      setResults(search.items)
      setGraph(nextGraph)
      setSelected(nextGraph.nodes[0] || null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const focusNode = async (node: ModelDataNode | ModelDataSearchItem) => {
    if (node.kind !== 'protein' && node.kind !== 'reaction') return
    setSelected(node as ModelDataNode)
    setLoading(true)
    try {
      const nextGraph = await loadModelDataGraph('', node.id, 40)
      setGraph(nextGraph)
      const focused = nextGraph.nodes.find((item) => item.id === node.id)
      if (focused) setSelected(focused)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to focus entity')
    } finally {
      setLoading(false)
    }
  }

  const resetGraph = async () => {
    setQuery('')
    setResults([])
    setLoading(true)
    try {
      const nextGraph = await loadModelDataGraph('', '', 28)
      setGraph(nextGraph)
      setSelected(nextGraph.nodes[0] || null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="model-data-hub">
      <div className="model-data-intro">
        <div>
          <span className="section-kicker">DATA USED BY THE SEARCH SYSTEM</span>
          <h2>Protein–reaction knowledge space</h2>
          <p>Explore the proteins, reactions and known enzyme–reaction links that the model searches. Use the graph to see which records are connected and which candidates were added to support searches beyond the original reference set.</p>
        </div>
        <div className="model-data-badges">
          <span><i className="status-dot" /> current data snapshot</span>
          <span>exploration view</span>
          <span>source records available</span>
        </div>
      </div>

      <div className="catalog-metrics">
        <Metric label="Proteins" value={summary?.proteins} sub={`${summary?.registered_proteins ?? '—'} added for search`} />
        <Metric label="Reactions" value={summary?.reactions} sub={`${summary?.registered_reactions ?? '—'} added for search`} />
        <Metric label="Known links" value={summary?.associations} sub="enzyme–reaction pairs" />
        <Metric label="Mechanism records" value={summary?.mechanism_reactions} sub="reactions with step-level information" />
        <Metric label="Reference proteins" value={summary?.seen_proteins} sub="present in the original reference data" />
        <Metric label="Reference reactions" value={summary?.seen_reactions} sub="present in the original reference data" />
      </div>

      <div className="model-data-layout">
        <aside className="catalog-sidebar glass-panel">
          <form className="catalog-search" onSubmit={submitSearch}>
            <label>
              <span>Search proteins, reactions and links</span>
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="enzyme, species, reaction, product…" />
            </label>
            <div className="catalog-kind-switch">
              {['all', 'protein', 'reaction', 'association'].map((value) => (
                <button type="button" key={value} className={kind === value ? 'active' : ''} onClick={() => setKind(value)}>{dataKindLabel(value)}</button>
              ))}
            </div>
            <div className="catalog-search-actions">
              <button type="submit" disabled={loading}>{loading ? 'Loading…' : 'Search and show connections'}</button>
              <button type="button" onClick={resetGraph}>Reset</button>
            </div>
          </form>

          {error && <div className="error-banner">{error}</div>}

          <div className="catalog-result-list">
            <div className="catalog-sidebar-heading"><span>Search results</span><strong>{results.length || graph?.edge_count || 0}</strong></div>
            {(results.length ? results : graph?.nodes.slice(0, 14) || []).map((item) => (
              <button key={`${item.kind}-${item.id}`} className={selected?.id === item.id ? 'selected' : ''} onClick={() => focusNode(item)}>
                <span className={`catalog-kind-dot ${item.kind}`} />
                <span><strong>{trim(item.name, 48)}</strong><small>{item.id}</small></span>
                <em>{item.terpene_type || item.kind}</em>
              </button>
            ))}
          </div>

          <div className="catalog-source-block">
            <span>Source tables</span>
            {summary?.source_files.map((file) => <code key={file}>{file}</code>)}
          </div>
        </aside>

        <div className="catalog-graph-panel glass-panel">
          <div className="catalog-graph-header">
            <div>
              <span className="section-kicker">HOW PROTEINS AND REACTIONS ARE CONNECTED</span>
              <h3>{graph?.query ? `Matches for “${graph.query}”` : graph?.focus_id ? `Neighborhood of ${graph.focus_id}` : 'A representative sample of known links'}</h3>
            </div>
            <div><span>{graph?.node_count ?? '—'} records</span><span>{graph?.edge_count ?? '—'} links</span><span>{graph?.total_associations ?? '—'} known links in full dataset</span></div>
          </div>
          <CatalogGraph graph={graph} selectedId={selected?.id || null} hoveredId={hovered} onHover={setHovered} onSelect={focusNode} />
          <div className="catalog-graph-legend">
            <span><i className="protein" /> protein / terpene synthase</span>
            <span><i className="reaction" /> reaction</span>
            <span><i className="seen" /> present in the reference data</span>
            <span><i className="external" /> added for broader searches</span>
          </div>
        </div>

        <EntityInspector node={selected} graph={graph} />
      </div>

      <div className="catalog-distributions">
        <Distribution title="Terpene types" buckets={summary?.terpene_types || []} />
        <Distribution title="Reference coverage" buckets={summary?.open_world_categories || []} />
      </div>
    </section>
  )
}

function CatalogGraph({ graph, selectedId, hoveredId, onHover, onSelect }: { graph: ModelDataGraph | null; selectedId: string | null; hoveredId: string | null; onHover: (id: string | null) => void; onSelect: (node: ModelDataNode) => void }) {
  const positioned = useMemo(() => layoutNodes(graph?.nodes || []), [graph])
  const nodeMap = useMemo(() => new Map(positioned.map((node) => [node.id, node])), [positioned])
  if (!graph) return <div className="catalog-graph-empty">Loading protein–reaction connections…</div>
  if (!graph.nodes.length) return <div className="catalog-graph-empty">No protein–reaction links match this search.</div>

  return (
    <div className="catalog-graph-scroll">
      <svg className="catalog-graph" viewBox="0 0 1100 650" role="img" aria-label="Graph of known links between terpene synthases and reactions">
        <defs>
          <filter id="catalog-glow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <g className="catalog-edge-layer">
          {graph.edges.map((edge) => {
            const source = nodeMap.get(edge.protein_id)
            const target = nodeMap.get(edge.reaction_id)
            if (!source || !target) return null
            const active = [selectedId, hoveredId].filter(Boolean).some((id) => id === edge.protein_id || id === edge.reaction_id)
            const path = `M ${source.x + source.width} ${source.y + source.height / 2} C 430 ${source.y + source.height / 2}, 670 ${target.y + target.height / 2}, ${target.x} ${target.y + target.height / 2}`
            return <path key={edge.id} d={path} className={`catalog-edge ${active ? 'active' : ''} ${edge.protein_seen && edge.reaction_seen ? 'seen' : 'external'}`} />
          })}
        </g>
        <g className="catalog-node-layer">
          {positioned.map((node) => (
            <foreignObject key={node.id} x={node.x} y={node.y} width={node.width} height={node.height}>
              <button
                className={`catalog-node ${node.kind} ${selectedId === node.id ? 'selected' : ''} ${hoveredId === node.id ? 'hovered' : ''}`}
                onMouseEnter={() => onHover(node.id)} onMouseLeave={() => onHover(null)} onClick={() => onSelect(node)}
              >
                <span><i />{node.kind === 'protein' ? 'PROTEIN' : 'REACTION'}</span>
                <strong>{trim(node.name, 34)}</strong>
                <small>{trim(node.id, 28)} · {node.degree || 1} link{node.degree === 1 ? '' : 's'}</small>
              </button>
            </foreignObject>
          ))}
        </g>
        <text className="catalog-axis-label" x="70" y="30">TERPENE SYNTHASE PROTEINS</text>
        <text className="catalog-axis-label" x="815" y="30">TERPENE-FORMING REACTIONS</text>
      </svg>
    </div>
  )
}

function layoutNodes(nodes: ModelDataNode[]): PositionedNode[] {
  const proteins = nodes.filter((node) => node.kind === 'protein').slice(0, 26)
  const reactions = nodes.filter((node) => node.kind === 'reaction').slice(0, 26)
  const place = (items: ModelDataNode[], side: 'left' | 'right') => {
    const columns = items.length > 13 ? 2 : 1
    const rows = Math.ceil(items.length / columns)
    const gap = Math.min(48, 560 / Math.max(rows, 1))
    return items.map((node, index) => {
      const column = Math.floor(index / rows)
      const row = index % rows
      const baseX = side === 'left' ? 55 : 710
      return { ...node, x: baseX + column * 190, y: 48 + row * gap, width: 180, height: 40 }
    })
  }
  return [...place(proteins, 'left'), ...place(reactions, 'right')]
}

function EntityInspector({ node, graph }: { node: ModelDataNode | null; graph: ModelDataGraph | null }) {
  const linked = node && graph ? graph.edges.filter((edge) => edge.protein_id === node.id || edge.reaction_id === node.id) : []
  return (
    <aside className="catalog-inspector glass-panel">
      <span className="section-kicker">SELECTED RECORD</span>
      {node ? <>
        <div className="catalog-inspector-title"><i className={node.kind} /><div><small>{node.kind}</small><h3>{node.name}</h3><code>{node.id}</code></div></div>
        <dl>
          <div><dt>Terpene type</dt><dd>{node.terpene_type || '—'}</dd></div>
          <div><dt>Status</dt><dd>{node.seen ? 'present in reference data' : 'added for broader search'}</dd></div>
          <div><dt>Added to search collection</dt><dd>{node.registered ? 'yes' : 'no'}</dd></div>
          {node.kind === 'protein' && <><div><dt>Species</dt><dd>{node.species || '—'}</dd></div><div><dt>TPS class</dt><dd>{node.tps_class || '—'}</dd></div><div><dt>Length</dt><dd>{node.sequence_length ? `${node.sequence_length} aa` : '—'}</dd></div></>}
          {node.kind === 'reaction' && <><div><dt>Substrate</dt><dd>{node.substrate_name || '—'}</dd></div><div><dt>Product</dt><dd>{node.product_name || '—'}</dd></div><div><dt>Mechanism</dt><dd>{node.has_mechanism ? 'step-level mechanism available' : 'no step-level mechanism linked'}</dd></div></>}
          <div><dt>Connections shown here</dt><dd>{linked.length}</dd></div>
        </dl>
        <div className="catalog-linked-list"><span>Connected records shown here</span>{linked.slice(0, 8).map((edge) => <p key={edge.id}>{node.kind === 'protein' ? edge.reaction_name : edge.protein_name}</p>)}</div>
        <small>Record source: {node.source_file || 'enzyme–reaction association table'}</small>
      </> : <p className="catalog-empty-copy">Select a protein or reaction to see its available metadata and connected records.</p>}
    </aside>
  )
}

function Metric({ label, value, sub }: { label: string; value?: number; sub: string }) {
  return <article><span>{label}</span><strong>{value?.toLocaleString() ?? '—'}</strong><small>{sub}</small></article>
}

function Distribution({ title, buckets }: { title: string; buckets: Array<{ label: string; count: number }> }) {
  const max = Math.max(...buckets.map((bucket) => bucket.count), 1)
  return <article className="catalog-distribution glass-panel"><div><span className="section-kicker">DATA OVERVIEW</span><h3>{title}</h3></div><div>{buckets.slice(0, 9).map((bucket) => <p key={bucket.label}><span>{bucketLabel(bucket.label)}</span><i><b style={{ width: `${(bucket.count / max) * 100}%` }} /></i><strong>{bucket.count}</strong></p>)}</div></article>
}

function dataKindLabel(value: string) {
  const labels: Record<string, string> = { all: 'all records', protein: 'proteins', reaction: 'reactions', association: 'known links' }
  return labels[value] || value
}

function bucketLabel(value: string) {
  const labels: Record<string, string> = {
    seen_seen: 'protein and reaction both in reference data',
    seen_unseen: 'reference protein with newly added reaction',
    unseen_seen: 'newly added protein with reference reaction',
    unseen_unseen: 'protein and reaction both added beyond reference data',
    current_current: 'protein and reaction both in reference data',
    current_registered: 'reference protein with newly added reaction',
    registered_current: 'newly added protein with reference reaction',
    registered_registered: 'both records added for broader search',
  }
  return labels[value] || value.replaceAll('_', ' ')
}

function trim(value: string, length: number) {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value
}
