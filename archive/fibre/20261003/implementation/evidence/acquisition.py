from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import quote

import requests

from .source_registry import SOURCE_BY_ID, SourceSpec, sources_for


@dataclass(frozen=True)
class AcquisitionQuery:
    observation_families: tuple[str, ...]
    identifiers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    text: str = ''
    limit: int = 20

    @classmethod
    def build(
        cls,
        *,
        observation_families: Iterable[str],
        identifiers: dict[str, Iterable[str]] | None = None,
        text: str = '',
        limit: int = 20,
    ) -> 'AcquisitionQuery':
        clean={}
        for kind,values in (identifiers or {}).items():
            seen=[]
            for value in values:
                value=str(value or '').strip()
                if value and value not in seen:
                    seen.append(value)
            if seen:
                clean[str(kind)]=tuple(seen)
        return cls(
            observation_families=tuple(dict.fromkeys(str(x) for x in observation_families if str(x))),
            identifiers=clean,
            text=str(text or '').strip(),
            limit=max(1,min(200,int(limit or 20))),
        )

    def values(self, kind: str) -> tuple[str, ...]:
        return tuple(self.identifiers.get(kind) or ())


@dataclass(frozen=True)
class AcquiredRecord:
    source_id: str
    record_id: str
    observation_families: tuple[str, ...]
    source_uri: str
    subject_ids: dict[str, tuple[str, ...]]
    payload_format: str
    payload: Any
    cache_state: str

    def to_dict(self) -> dict[str,Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceFetchStatus:
    source_id: str
    status: str
    detail: str = ''
    record_count: int = 0

    def to_dict(self) -> dict[str,Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionResult:
    query: AcquisitionQuery
    records: tuple[AcquiredRecord, ...]
    statuses: tuple[SourceFetchStatus, ...]

    def to_dict(self) -> dict[str,Any]:
        return {
            'query':asdict(self.query),
            'records':[x.to_dict() for x in self.records],
            'statuses':[x.to_dict() for x in self.statuses],
        }


class AcquisitionCache:
    def __init__(self, root: Path) -> None:
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)

    def _path(self, source_id: str, fingerprint: str) -> Path:
        d=self.root/source_id
        d.mkdir(parents=True,exist_ok=True)
        return d/f'{fingerprint}.json'

    @staticmethod
    def fingerprint(method: str, url: str, params: Any = None, data: Any = None) -> str:
        raw=json.dumps(
            {'method':method.upper(),'url':url,'params':params,'data':data},
            sort_keys=True,ensure_ascii=False,default=str,separators=(',',':'),
        )
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def read(
        self, source_id: str, fingerprint: str, *, max_age_seconds: int | None
    ) -> tuple[dict[str,Any],int] | None:
        path=self._path(source_id,fingerprint)
        if not path.is_file():
            return None
        try:
            env=json.loads(path.read_text(encoding='utf-8'))
            age=max(0,int(time.time()-float(env['cached_at_unix'])))
            if max_age_seconds is not None and age>int(max_age_seconds):
                return None
            return dict(env),age
        except Exception:
            return None

    def write(self, source_id: str, fingerprint: str, envelope: dict[str,Any]) -> None:
        path=self._path(source_id,fingerprint)
        payload={'cached_at_unix':time.time(),**envelope}
        with tempfile.NamedTemporaryFile(
            'w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False
        ) as fh:
            tmp=Path(fh.name)
            json.dump(payload,fh,ensure_ascii=False,default=str)
        tmp.replace(path)


class CachedHTTPClient:
    def __init__(self, cache: AcquisitionCache, *, user_agent: str, timeout: int = 15) -> None:
        self.cache=cache
        self.timeout=int(timeout)
        self.session=requests.Session()
        self.session.headers.update({'User-Agent':user_agent})

    @staticmethod
    def _transient(exc: Exception) -> bool:
        if isinstance(exc,(requests.Timeout,requests.ConnectionError)):
            return True
        if isinstance(exc,requests.HTTPError):
            code=int(getattr(getattr(exc,'response',None),'status_code',0) or 0)
            return code==429 or 500<=code<600
        return False

    def request(
        self, source: SourceSpec, method: str, url: str, *, params: Any = None,
        data: Any = None, expect: str = 'json', timeout: int | None = None,
    ) -> tuple[Any,str,str]:
        fp=self.cache.fingerprint(method,url,params,data)
        fresh=self.cache.read(source.source_id,fp,max_age_seconds=source.cache_ttl_seconds)
        if fresh is not None:
            env,_age=fresh
            return env['payload'],'fresh_cache',str(env.get('final_url') or url)
        try:
            response=self.session.request(
                method,url,params=params,data=data,timeout=timeout or self.timeout,
            )
            response.raise_for_status()
            payload=response.json() if expect=='json' else response.text
            final_url=str(response.url)
            self.cache.write(source.source_id,fp,{
                'status_code':response.status_code,
                'content_type':response.headers.get('content-type'),
                'final_url':final_url,
                'payload':payload,
            })
            return payload,'live',final_url
        except Exception as exc:
            if self._transient(exc):
                stale=self.cache.read(source.source_id,fp,max_age_seconds=None)
                if stale is not None:
                    env,_age=stale
                    return env['payload'],'stale_cache',str(env.get('final_url') or url)
            raise


class SourceAdapter:
    source_id: str

    @property
    def spec(self) -> SourceSpec:
        return SOURCE_BY_ID[self.source_id]

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        raise NotImplementedError


def _families(query: AcquisitionQuery, spec: SourceSpec) -> tuple[str,...]:
    requested=set(query.observation_families)
    return tuple(x for x in spec.observation_families if x in requested)


class UniProtAdapter(SourceAdapter):
    source_id='uniprot'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        out=[]
        fam=_families(query,self.spec)
        for accession in query.values('uniprot')[:query.limit]:
            url=f'https://rest.uniprot.org/uniprotkb/{quote(accession,safe="")}.json'
            payload,state,final=http.request(self.spec,'GET',url)
            primary=str(payload.get('primaryAccession') or accession)
            out.append(AcquiredRecord(
                self.source_id,primary,fam,final,
                {'uniprot':tuple(dict.fromkeys([primary,accession]))},
                'json',payload,state,
            ))
        return out


class RheaAdapter(SourceAdapter):
    source_id='rhea'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        terms=[]
        terms += [('rhea',x,f'id:{x.split(":")[-1]}') for x in query.values('rhea')]
        terms += [('ec',x,f'ec:{x}') for x in query.values('ec')]
        terms += [('chebi',x,f'chebi:{x.split(":")[-1]}') for x in query.values('chebi')]
        if query.text:
            terms.append(('text',query.text,query.text))
        out=[]; fam=_families(query,self.spec)
        for kind,value,term in terms[:query.limit]:
            payload,state,final=http.request(
                self.spec,'GET','https://www.rhea-db.org/rhea',
                params={'query':term,'columns':'rhea-id,equation,chebi','format':'tsv','limit':query.limit},
                expect='text',
            )
            out.append(AcquiredRecord(
                self.source_id,f'{kind}:{value}',fam,final,{kind:(value,)},
                'tsv',payload,state,
            ))
        return out


class EuropePMCAdapter(SourceAdapter):
    source_id='europe_pmc'

    @staticmethod
    def _search_term(query: AcquisitionQuery) -> str:
        terms=[]
        terms += [f'EXT_ID:{x} AND SRC:MED' for x in query.values('pmid')]
        terms += [f'PMCID:{x}' for x in query.values('pmcid')]
        terms += [f'DOI:{x}' for x in query.values('doi')]
        terms += [f'ACCESSION_ID:{x}' for x in query.values('uniprot')]
        if query.text:
            terms.append(f'({query.text})')
        return ' OR '.join(f'({x})' for x in terms)

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        out=[]; fam=_families(query,self.spec)
        term=self._search_term(query)
        if term:
            payload,state,final=http.request(
                self.spec,'GET','https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                params={'query':term,'format':'json','pageSize':query.limit},
            )
            out.append(AcquiredRecord(
                self.source_id,'search:'+hashlib.sha256(term.encode()).hexdigest()[:16],
                tuple(x for x in fam if x!='full_text'),final,{},'json',payload,state,
            ))
        if 'full_text' in fam:
            for pmcid in query.values('pmcid')[:query.limit]:
                url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{quote(pmcid,safe="")}/fullTextXML'
                payload,state,final=http.request(self.spec,'GET',url,expect='text')
                out.append(AcquiredRecord(
                    self.source_id,pmcid,('full_text','assay_context') if 'assay_context' in fam else ('full_text',),
                    final,{'pmcid':(pmcid,)},'xml',payload,state,
                ))
        return out


class MCSAAdapter(SourceAdapter):
    source_id='mcsa'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        filters=[]
        for x in query.values('uniprot'): filters.append(('uniprot',x,'entries.proteins.sequences.uniprot_ids'))
        for x in query.values('ec'): filters.append(('ec',x,'entries.reactions.ecs.codes'))
        for x in query.values('mcsa'): filters.append(('mcsa',x,'entries.mcsa_ids'))
        out=[]; fam=_families(query,self.spec)
        for kind,value,key in filters[:query.limit]:
            params={'format':'json',key:value}
            entries,state,url=http.request(
                self.spec,'GET','https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/',params=params
            )
            out.append(AcquiredRecord(
                self.source_id,f'entries:{kind}:{value}',tuple(x for x in fam if x!='catalytic_residue'),
                url,{kind:(value,)},'json',entries,state,
            ))
            if 'catalytic_residue' in fam:
                residues,rstate,rurl=http.request(
                    self.spec,'GET','https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/',params=params
                )
                out.append(AcquiredRecord(
                    self.source_id,f'residues:{kind}:{value}',('catalytic_residue',),
                    rurl,{kind:(value,)},'json',residues,rstate,
                ))
        return out


class PDBeAdapter(SourceAdapter):
    source_id='pdbe'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        out=[]; fam=_families(query,self.spec)
        for pdb in query.values('pdb')[:query.limit]:
            pid=pdb.lower()
            for name,path,record_families in (
                ('summary',f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pid}',('experimental_structure','structure_annotation')),
                ('ligands',f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/ligand_monomers/{pid}',('ligand',)),
            ):
                needed=tuple(x for x in record_families if x in fam)
                if not needed: continue
                payload,state,url=http.request(self.spec,'GET',path)
                out.append(AcquiredRecord(
                    self.source_id,f'{pid}:{name}',needed,url,{'pdb':(pdb.upper(),)},'json',payload,state,
                ))
        return out


class AlphaFoldAdapter(SourceAdapter):
    source_id='alphafold_db'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        out=[]; fam=_families(query,self.spec)
        for accession in query.values('uniprot')[:query.limit]:
            url=f'https://alphafold.ebi.ac.uk/api/prediction/{quote(accession,safe="")}'
            payload,state,final=http.request(self.spec,'GET',url)
            out.append(AcquiredRecord(
                self.source_id,accession,fam,final,{'uniprot':(accession,)},'json',payload,state,
            ))
        return out


class ChEBIAdapter(SourceAdapter):
    source_id='chebi'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        out=[]; fam=_families(query,self.spec)
        for chebi in query.values('chebi')[:query.limit]:
            value=chebi.split(':')[-1]
            url=f'https://www.ebi.ac.uk/chebi/backend/api/public/compound/{quote(value,safe="")}/'
            payload,state,final=http.request(self.spec,'GET',url)
            out.append(AcquiredRecord(
                self.source_id,chebi,fam,final,{'chebi':(chebi,)},'json',payload,state,
            ))
        if query.text and not query.values('chebi'):
            payload,state,final=http.request(
                self.spec,'GET','https://www.ebi.ac.uk/chebi/backend/api/public/es_search/',
                params={'q':query.text,'size':query.limit},
            )
            out.append(AcquiredRecord(
                self.source_id,'search:'+hashlib.sha256(query.text.encode()).hexdigest()[:16],
                fam,final,{'compound_name':(query.text,)},'json',payload,state,
            ))
        return out


class PubChemAdapter(SourceAdapter):
    source_id='pubchem'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        out=[]; fam=_families(query,self.spec)
        for cid in query.values('pubchem_cid')[:query.limit]:
            url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{quote(cid,safe="")}/property/CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula/JSON'
            payload,state,final=http.request(self.spec,'GET',url)
            out.append(AcquiredRecord(
                self.source_id,cid,tuple(x for x in fam if x!='chemical_annotation'),final,
                {'pubchem_cid':(cid,)},'json',payload,state,
            ))
            if 'chemical_annotation' in fam:
                view,state2,url2=http.request(
                    self.spec,'GET',f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{quote(cid,safe="")}/JSON'
                )
                out.append(AcquiredRecord(
                    self.source_id,f'{cid}:annotations',('chemical_annotation',),url2,
                    {'pubchem_cid':(cid,)},'json',view,state2,
                ))
        return out


class SabioRKAdapter(SourceAdapter):
    source_id='sabio_rk'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        # SABIO-RK still documents the legacy REST web service, but live requests
        # currently redirect to the SPA 404 page. Fail closed: endpoint failure
        # must never be interpreted as zero kinetic observations.
        url='https://sabiork.h-its.org/sabioRestWebServices/status'
        payload,state,final=http.request(self.spec,'GET',url,expect='text')
        if final.rstrip('/').endswith('/ui/404') or '<html' in str(payload).lower():
            raise RuntimeError('documented SABIO-RK REST endpoint currently redirects to SPA 404')
        return []


class BrendaAdapter(SourceAdapter):
    source_id='brenda'

    def fetch(self, query: AcquisitionQuery, http: CachedHTTPClient) -> list[AcquiredRecord]:
        # Do not silently scrape the website. BRENDA SOAP requires registration;
        # bulk JSON requires explicit license acceptance. Either can be wired by
        # supplying a local licensed JSON snapshot or registered credentials.
        bulk=os.environ.get('BRENDA_JSON_PATH','').strip()
        if bulk and Path(bulk).is_file():
            return [AcquiredRecord(
                self.source_id,'licensed_bulk_snapshot',_families(query,self.spec),str(Path(bulk).resolve()),
                {},'local_json_bulk',{'path':str(Path(bulk).resolve())},'local',
            )]
        if os.environ.get('BRENDA_EMAIL','').strip() and os.environ.get('BRENDA_PASSWORD','').strip():
            raise RuntimeError('BRENDA credentials configured but SOAP adapter is not initialized in this runtime')
        raise RuntimeError('BRENDA requires registered SOAP credentials or an explicitly licensed local JSON snapshot')


DEFAULT_ADAPTERS: tuple[SourceAdapter,...] = (
    UniProtAdapter(),RheaAdapter(),EuropePMCAdapter(),MCSAAdapter(),PDBeAdapter(),
    AlphaFoldAdapter(),ChEBIAdapter(),PubChemAdapter(),SabioRKAdapter(),BrendaAdapter(),
)


class AcquisitionManager:
    def __init__(
        self, cache_root: Path, *, user_agent: str = 'FIBRE on-demand acquisition',
        adapters: Iterable[SourceAdapter] = DEFAULT_ADAPTERS,
    ) -> None:
        self.http=CachedHTTPClient(AcquisitionCache(cache_root),user_agent=user_agent)
        self.adapters={x.source_id:x for x in adapters}

    def plan(self, query: AcquisitionQuery, *, include_registered: bool = True) -> tuple[SourceSpec,...]:
        kinds=set(query.identifiers)
        if query.text:
            kinds.add('text')
            kinds.add('compound_name')
        return sources_for(
            query.observation_families,kinds,include_registered=include_registered
        )

    def acquire(
        self, query: AcquisitionQuery, *, source_ids: Iterable[str] | None = None,
        include_registered: bool = True,
    ) -> AcquisitionResult:
        planned=self.plan(query,include_registered=include_registered)
        selected=set(source_ids) if source_ids is not None else {x.source_id for x in planned}
        records=[]; statuses=[]
        for spec in planned:
            if spec.source_id not in selected:
                continue
            adapter=self.adapters.get(spec.source_id)
            if adapter is None:
                statuses.append(SourceFetchStatus(spec.source_id,'adapter_missing'))
                continue
            try:
                rows=adapter.fetch(query,self.http)
                records.extend(rows)
                statuses.append(SourceFetchStatus(spec.source_id,'ok',record_count=len(rows)))
            except requests.HTTPError as exc:
                code=int(getattr(exc.response,'status_code',0) or 0)
                statuses.append(SourceFetchStatus(spec.source_id,'endpoint_error',f'HTTP {code}: {exc}'))
            except Exception as exc:
                statuses.append(SourceFetchStatus(spec.source_id,'unavailable',f'{type(exc).__name__}: {exc}'))
        return AcquisitionResult(query,tuple(records),tuple(statuses))
