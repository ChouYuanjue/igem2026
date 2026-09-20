from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Protocol

from .publication_sources import normalize_doi, normalized_title


def normalize_seed_value(kind: str, value: str) -> str:
    kind=str(kind).strip().lower()
    value=str(value or '').strip()
    if kind=='doi':
        return normalize_doi(value)
    if kind=='pmcid':
        return value.upper()
    if kind in {'uniprot','rhea'}:
        return value.upper()
    if kind in {'title','literature_query'}:
        return normalized_title(value)
    if kind=='url':
        return value.split('#',1)[0]
    return value


@dataclass(frozen=True)
class EnrichmentSeed:
    kind: str
    value: str
    origin: str='user'
    parent_resource_id: str | None=None
    context: dict[str,Any]=field(default_factory=dict)

    @property
    def normalized_value(self) -> str:
        return normalize_seed_value(self.kind,self.value)

    @property
    def seed_id(self) -> str:
        payload=f'{self.kind.lower()}\x1f{self.normalized_value}'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]

    def to_dict(self) -> dict[str,Any]:
        row=asdict(self)
        row['seed_id']=self.seed_id
        row['normalized_value']=self.normalized_value
        return row


@dataclass(frozen=True)
class DiscoveredResource:
    source: str
    kind: str
    canonical_key: str
    payload: dict[str,Any]
    source_uri: str | None=None
    blob_text: str | None=None
    blob_suffix: str='.txt'

    @property
    def resource_id(self) -> str:
        payload=f'{self.source}\x1f{self.kind}\x1f{self.canonical_key}'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


@dataclass(frozen=True)
class AdapterResult:
    resources: tuple[DiscoveredResource,...]=()
    next_seeds: tuple[EnrichmentSeed,...]=()
    status: str='ok'
    message: str=''


class SourceAdapter(Protocol):
    name: str
    accepted_kinds: frozenset[str]

    def discover(
        self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False
    ) -> AdapterResult: ...


class EnrichmentStore:
    def __init__(self, root: str | Path):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.blobs=self.root/'blobs'
        self.blobs.mkdir(exist_ok=True)
        self.db_path=self.root/'discovery.sqlite'
        self.db=sqlite3.connect(self.db_path)
        self.db.row_factory=sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            '''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS seeds(
              seed_id TEXT PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL,
              normalized_value TEXT NOT NULL, origin TEXT NOT NULL,
              parent_resource_id TEXT, context_json TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS resources(
              resource_id TEXT PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL,
              canonical_key TEXT NOT NULL, source_uri TEXT, payload_json TEXT NOT NULL,
              blob_path TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges(
              seed_id TEXT NOT NULL, adapter TEXT NOT NULL, resource_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY(seed_id,adapter,resource_id)
            );
            CREATE TABLE IF NOT EXISTS seed_edges(
              parent_seed_id TEXT NOT NULL, adapter TEXT NOT NULL, child_seed_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY(parent_seed_id,adapter,child_seed_id)
            );
            CREATE TABLE IF NOT EXISTS runs(
              seed_id TEXT NOT NULL, adapter TEXT NOT NULL, status TEXT NOT NULL,
              message TEXT NOT NULL, resource_count INTEGER NOT NULL,
              next_seed_count INTEGER NOT NULL, updated_at REAL NOT NULL,
              PRIMARY KEY(seed_id,adapter)
            );
            CREATE INDEX IF NOT EXISTS idx_resources_source_kind ON resources(source,kind);
            CREATE INDEX IF NOT EXISTS idx_seeds_kind ON seeds(kind);
            '''
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def add_seed(self, seed: EnrichmentSeed) -> None:
        self.db.execute(
            '''INSERT OR IGNORE INTO seeds(
               seed_id,kind,value,normalized_value,origin,parent_resource_id,context_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?)''',
            (seed.seed_id,seed.kind.lower(),seed.value,seed.normalized_value,seed.origin,
             seed.parent_resource_id,json.dumps(seed.context,sort_keys=True),time.time()),
        )
        self.db.commit()

    def resource_exists(self, resource_id: str) -> bool:
        return self.db.execute(
            'SELECT 1 FROM resources WHERE resource_id=?',(resource_id,)
        ).fetchone() is not None

    def run_status(self, seed: EnrichmentSeed, adapter: str) -> str | None:
        row=self.db.execute(
            'SELECT status FROM runs WHERE seed_id=? AND adapter=?',
            (seed.seed_id,adapter),
        ).fetchone()
        return None if row is None else str(row['status'])

    def record_run(
        self,seed: EnrichmentSeed,adapter: str,result: AdapterResult
    ) -> None:
        self.db.execute(
            '''INSERT INTO runs(seed_id,adapter,status,message,resource_count,next_seed_count,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(seed_id,adapter) DO UPDATE SET
                 status=excluded.status,message=excluded.message,
                 resource_count=excluded.resource_count,next_seed_count=excluded.next_seed_count,
                 updated_at=excluded.updated_at''',
            (seed.seed_id,adapter,result.status,result.message,
             len(result.resources),len(result.next_seeds),time.time()),
        )
        self.db.commit()

    def _write_blob(self, resource: DiscoveredResource) -> str | None:
        if resource.blob_text is None:
            return None
        digest=hashlib.sha256(resource.blob_text.encode('utf-8')).hexdigest()
        suffix=resource.blob_suffix if resource.blob_suffix.startswith('.') else '.'+resource.blob_suffix
        path=self.blobs/f'{digest}{suffix}'
        if not path.exists():
            path.write_text(resource.blob_text,encoding='utf-8')
        return str(path.relative_to(self.root))

    def add_resource(
        self,seed: EnrichmentSeed,adapter: str,resource: DiscoveredResource
    ) -> bool:
        existing=self.db.execute(
            'SELECT 1 FROM resources WHERE resource_id=?',(resource.resource_id,)
        ).fetchone() is not None
        now=time.time(); blob=self._write_blob(resource)
        self.db.execute(
            '''INSERT INTO resources(resource_id,source,kind,canonical_key,source_uri,payload_json,blob_path,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(resource_id) DO UPDATE SET
                 source_uri=excluded.source_uri,payload_json=excluded.payload_json,
                 blob_path=COALESCE(excluded.blob_path,resources.blob_path),updated_at=excluded.updated_at''',
            (resource.resource_id,resource.source,resource.kind,resource.canonical_key,
             resource.source_uri,json.dumps(resource.payload,sort_keys=True,ensure_ascii=False),
             blob,now,now),
        )
        self.db.execute(
            'INSERT OR IGNORE INTO edges(seed_id,adapter,resource_id,created_at) VALUES(?,?,?,?)',
            (seed.seed_id,adapter,resource.resource_id,now),
        )
        self.db.commit()
        return not existing

    def link_child_seed(
        self,parent: EnrichmentSeed,adapter: str,child: EnrichmentSeed
    ) -> None:
        self.add_seed(child)
        self.db.execute(
            'INSERT OR IGNORE INTO seed_edges(parent_seed_id,adapter,child_seed_id,created_at) VALUES(?,?,?,?)',
            (parent.seed_id,adapter,child.seed_id,time.time()),
        )
        self.db.commit()

    def child_seeds(self,parent: EnrichmentSeed,adapter: str) -> list[EnrichmentSeed]:
        rows=self.db.execute(
            '''SELECT s.kind,s.value,s.origin,s.parent_resource_id,s.context_json
               FROM seed_edges e JOIN seeds s ON s.seed_id=e.child_seed_id
               WHERE e.parent_seed_id=? AND e.adapter=?
               ORDER BY s.seed_id''',
            (parent.seed_id,adapter),
        ).fetchall()
        return [
            EnrichmentSeed(
                kind=str(r['kind']),value=str(r['value']),origin=str(r['origin']),
                parent_resource_id=r['parent_resource_id'],
                context=json.loads(str(r['context_json']) or '{}'),
            )
            for r in rows
        ]

    def summary(self) -> dict[str,Any]:
        def scalar(sql: str) -> int:
            return int(self.db.execute(sql).fetchone()[0])
        sources={
            str(r[0]):int(r[1])
            for r in self.db.execute(
                'SELECT source,COUNT(*) FROM resources GROUP BY source ORDER BY source'
            )
        }
        kinds={
            str(r[0]):int(r[1])
            for r in self.db.execute(
                'SELECT kind,COUNT(*) FROM seeds GROUP BY kind ORDER BY kind'
            )
        }
        return {
            'schema':'fibre-enrichment-store-v1',
            'seed_count':scalar('SELECT COUNT(*) FROM seeds'),
            'resource_count':scalar('SELECT COUNT(*) FROM resources'),
            'edge_count':scalar('SELECT COUNT(*) FROM edges'),
            'seed_edge_count':scalar('SELECT COUNT(*) FROM seed_edges'),
            'run_count':scalar('SELECT COUNT(*) FROM runs'),
            'resource_counts_by_source':sources,
            'seed_counts_by_kind':kinds,
        }


def run_discovery(
    initial_seeds: Iterable[EnrichmentSeed],
    adapters: Iterable[SourceAdapter],
    store: EnrichmentStore,
    *,
    max_depth: int=2,
    max_resources: int=100,
    per_adapter_limit: int=10,
    materialize: bool=False,
    refresh: bool=False,
) -> dict[str,Any]:
    adapters=tuple(adapters)
    queue=[(0,s) for s in initial_seeds]
    seen=set()
    new_resources=0
    adapter_calls=0
    failures=[]

    while queue and new_resources < max_resources:
        depth,seed=queue.pop(0)
        marker=(seed.kind.lower(),seed.normalized_value)
        if not seed.normalized_value or marker in seen:
            continue
        seen.add(marker)
        store.add_seed(seed)
        if depth>max_depth:
            continue
        for adapter in adapters:
            if seed.kind.lower() not in adapter.accepted_kinds:
                continue
            run_key=(
                f'{adapter.name}:materialize' if materialize else adapter.name
            )
            status=store.run_status(seed,run_key)
            if status in {'ok','not_found','invalid'} and not refresh:
                if depth<max_depth:
                    for child in store.child_seeds(seed,run_key):
                        queue.append((depth+1,child))
                continue
            try:
                result=adapter.discover(
                    seed,limit=per_adapter_limit,materialize=materialize
                )
            except Exception as exc:
                result=AdapterResult(
                    status='failed',message=f'{type(exc).__name__}: {exc}'
                )
            adapter_calls+=1
            processed_resource_ids=set()
            truncated=False
            for resource in result.resources:
                if new_resources>=max_resources and not store.resource_exists(resource.resource_id):
                    truncated=True
                    continue
                if store.add_resource(seed,run_key,resource):
                    new_resources+=1
                processed_resource_ids.add(resource.resource_id)
            accepted_children=[]
            for child in result.next_seeds:
                parent_id=child.parent_resource_id
                if parent_id and parent_id not in processed_resource_ids and not store.resource_exists(parent_id):
                    truncated=True
                    continue
                store.link_child_seed(seed,run_key,child)
                accepted_children.append(child)
            stored_result=result
            if truncated and result.status=='ok':
                stored_result=AdapterResult(
                    resources=result.resources,
                    next_seeds=result.next_seeds,
                    status='partial_budget',
                    message='adapter result truncated by max_resources; rerun resumes remaining resources',
                )
            store.record_run(seed,run_key,stored_result)
            if stored_result.status not in {'ok','not_found','invalid','partial_budget'}:
                failures.append({
                    'seed':seed.to_dict(),'adapter':adapter.name,
                    'status':stored_result.status,'message':stored_result.message,
                })
            if depth<max_depth:
                for child in accepted_children:
                    queue.append((depth+1,child))
            if new_resources>=max_resources:
                break

    return {
        **store.summary(),
        'new_resource_events':new_resources,
        'adapter_calls':adapter_calls,
        'visited_seed_count':len(seen),
        'max_depth':int(max_depth),
        'max_resources':int(max_resources),
        'failure_count':len(failures),
        'failures':failures[:50],
    }
