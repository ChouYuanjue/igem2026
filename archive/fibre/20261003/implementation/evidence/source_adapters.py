from __future__ import annotations

import io
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from .enrichment import (
    AdapterResult, DiscoveredResource, EnrichmentSeed,
)
from .publication_sources import (
    extract_citation_metadata, normalize_doi, normalized_title, looks_like_url,
)


USER_AGENT='FIBRE/1.0 biological-evidence-discovery'


def _child(
    kind: str,value: str,parent: DiscoveredResource,origin: str,
    seed: EnrichmentSeed,
) -> EnrichmentSeed:
    return EnrichmentSeed(
        kind=kind,value=value,origin=origin,parent_resource_id=parent.resource_id,
        context=dict(seed.context),
    )


class EuropePMCAdapter:
    name='europe_pmc'
    accepted_kinds=frozenset({'pmid','pmcid','doi','title','literature_query'})

    def __init__(self,session: requests.Session | None=None):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)

    def _query(self,seed: EnrichmentSeed) -> str:
        kind=seed.kind.lower(); value=seed.value.strip()
        if kind=='pmid': return f'EXT_ID:{value} AND SRC:MED'
        if kind=='pmcid': return f'PMCID:{value.upper()}'
        if kind=='doi': return f'DOI:"{normalize_doi(value)}"'
        if kind=='title': return f'TITLE:"{value}"'
        return value

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        params={
            'query':self._query(seed),'format':'json','resultType':'core',
            'pageSize':max(1,min(int(limit),1000)),
        }
        r=self.session.get(
            'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
            params=params,timeout=25,
        )
        r.raise_for_status()
        results=r.json().get('resultList',{}).get('result',[])
        resources=[]; children=[]
        for row in results[:limit]:
            pmid=str(row.get('pmid') or '')
            pmcid=str(row.get('pmcid') or '')
            doi=normalize_doi(str(row.get('doi') or ''))
            title=' '.join(str(row.get('title') or '').split())
            key=pmcid or pmid or doi or normalized_title(title)
            if not key: continue
            source_uri=(
                f'https://europepmc.org/article/PMC/{pmcid[3:]}'
                if pmcid.startswith('PMC') else
                (f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else
                 (f'https://doi.org/{doi}' if doi else None))
            )
            payload={
                'pmid':pmid,'pmcid':pmcid,'doi':doi,'title':title,
                'author_string':str(row.get('authorString') or ''),
                'journal':str(row.get('journalTitle') or ''),
                'publication_year':str(row.get('pubYear') or ''),
                'open_access':str(row.get('isOpenAccess') or ''),
                'in_pmc':str(row.get('inPMC') or ''),
                'cited_by_count':row.get('citedByCount'),
                'has_references':str(row.get('hasReferences') or ''),
            }
            article=DiscoveredResource(
                source=self.name,kind='publication_metadata',canonical_key=key,
                payload=payload,source_uri=source_uri,
            )
            resources.append(article)
            for kind,value in [('pmid',pmid),('pmcid',pmcid),('doi',doi)]:
                if value:
                    children.append(_child(kind,value,article,'europe_pmc_metadata',seed))
            if materialize and pmcid:
                ft_url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
                ft=self.session.get(ft_url,timeout=30)
                if ft.status_code==200 and '<article' in ft.text[:2000]:
                    resources.append(DiscoveredResource(
                        source=self.name,kind='publication_fulltext_xml',
                        canonical_key=pmcid,payload={'pmcid':pmcid,'pmid':pmid,'doi':doi},
                        source_uri=ft_url,blob_text=ft.text,blob_suffix='.xml',
                    ))
        return AdapterResult(tuple(resources),tuple(children))


class CrossrefAdapter:
    name='crossref'
    accepted_kinds=frozenset({'doi','title','literature_query'})

    def __init__(self,session: requests.Session | None=None):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)

    def _messages(self,seed: EnrichmentSeed,limit: int) -> list[dict[str,Any]]:
        if seed.kind.lower()=='doi':
            doi=normalize_doi(seed.value)
            if not doi: return []
            r=self.session.get(
                'https://api.crossref.org/works/'+quote(doi,safe=''),timeout=25
            )
            if r.status_code==404: return []
            r.raise_for_status()
            return [r.json().get('message',{})]
        params={
            'query.bibliographic':seed.value,
            'rows':max(1,min(int(limit),1000)),
        }
        r=self.session.get('https://api.crossref.org/works',params=params,timeout=25)
        r.raise_for_status()
        return list(r.json().get('message',{}).get('items',[])[:limit])

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        resources=[]; children=[]
        for row in self._messages(seed,limit):
            doi=normalize_doi(str(row.get('DOI') or ''))
            titles=row.get('title') or []
            title=' '.join(str(titles[0] if titles else '').split())
            key=doi or normalized_title(title)
            if not key: continue
            url=str(row.get('URL') or (f'https://doi.org/{doi}' if doi else ''))
            payload={
                'doi':doi,'title':title,'publisher':str(row.get('publisher') or ''),
                'type':str(row.get('type') or ''),'url':url,
                'reference_count':row.get('reference-count'),
                'is_referenced_by_count':row.get('is-referenced-by-count'),
                'published':row.get('published'),
                'authors':row.get('author') or [],
            }
            resource=DiscoveredResource(
                source=self.name,kind='publication_metadata',canonical_key=key,
                payload=payload,source_uri=url or None,
            )
            resources.append(resource)
            if doi: children.append(_child('doi',doi,resource,'crossref_metadata',seed))
            if url and looks_like_url(url):
                children.append(_child('url',url,resource,'crossref_metadata',seed))
        return AdapterResult(tuple(resources),tuple(children))


class PublicationURLAdapter:
    name='publication_url'
    accepted_kinds=frozenset({'url'})

    def __init__(self,session: requests.Session | None=None,max_bytes: int=2_000_000):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)
        self.max_bytes=int(max_bytes)

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        url=seed.value.strip()
        r=self.session.get(url,timeout=25,allow_redirects=True,stream=True)
        r.raise_for_status()
        ctype=str(r.headers.get('Content-Type') or '').lower()
        body=r.raw.read(self.max_bytes,decode_content=True)
        final=str(r.url)
        if 'pdf' in ctype or body[:4]==b'%PDF':
            resource=DiscoveredResource(
                source=self.name,kind='publication_pdf_url',canonical_key=final,
                payload={'requested_url':url,'final_url':final,'content_type':ctype},
                source_uri=final,
            )
            return AdapterResult((resource,),())
        text=body.decode(r.encoding or 'utf-8',errors='replace')
        meta=extract_citation_metadata(text)
        key=meta.get('doi') or meta.get('pmcid') or meta.get('pmid') or normalized_title(meta.get('title','')) or final
        resource=DiscoveredResource(
            source=self.name,kind='publication_landing_page',canonical_key=key,
            payload={**meta,'requested_url':url,'final_url':final,'content_type':ctype},
            source_uri=final,blob_text=text if materialize else None,blob_suffix='.html',
        )
        children=[]
        for kind in ('doi','pmid','pmcid'):
            value=str(meta.get(kind) or '')
            if value: children.append(_child(kind,value,resource,'citation_meta',seed))
        if meta.get('title'):
            children.append(_child('title',str(meta['title']),resource,'citation_meta',seed))
        return AdapterResult((resource,),tuple(children))


class UniProtAdapter:
    name='uniprot'
    accepted_kinds=frozenset({'uniprot'})
    fields=(
        'accession,id,reviewed,protein_name,organism_name,ec,'
        'cc_catalytic_activity,cc_cofactor,ft_act_site,ft_binding,xref_pdb'
    )

    def __init__(self,session: requests.Session | None=None):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        acc=seed.value.strip()
        r=self.session.get(
            f'https://rest.uniprot.org/uniprotkb/{quote(acc,safe="")}.tsv',
            params={'fields':self.fields},timeout=25,
        )
        if r.status_code==404:
            return AdapterResult(status='not_found',message=f'UniProtKB {acc} not found')
        r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text),sep='\t',dtype=str).fillna('')
        if df.empty: return AdapterResult(status='not_found',message=f'UniProtKB {acc} empty')
        row=df.iloc[0].to_dict()
        entry=str(row.get('Entry') or acc)
        payload={str(k):str(v) for k,v in row.items()}
        resource=DiscoveredResource(
            source=self.name,kind='protein_annotation',canonical_key=entry,
            payload=payload,source_uri=f'https://www.uniprot.org/uniprotkb/{entry}/entry',
        )
        text='\n'.join(payload.values())
        children=[]
        for rid in sorted(set(re.findall(r'Rhea:(RHEA:\d+)',text))):
            children.append(_child('rhea',rid,resource,'uniprot_annotation',seed))
        for pmid in sorted(set(re.findall(r'PubMed:(\d+)',text))):
            children.append(_child('pmid',pmid,resource,'uniprot_annotation',seed))
        for other in sorted(set(re.findall(r'UniProtKB:([A-Z0-9]+)',text))):
            if other!=entry:
                children.append(_child('uniprot',other,resource,'uniprot_annotation_provenance',seed))
        pdb_field=str(row.get('PDB') or '')
        for pdb_id in [x.strip() for x in pdb_field.split(';') if x.strip()]:
            children.append(_child('pdb',pdb_id.upper(),resource,'uniprot_pdb_crossref',seed))
        return AdapterResult((resource,),tuple(children[:max(1,limit*6)]))


class LocalRheaAdapter:
    name='rhea_local'
    accepted_kinds=frozenset({'rhea'})

    def __init__(self,root: str | Path='results/fibre_rhea_mapping_v1'):
        self.root=Path(root)
        self._loaded=False
        self.smiles: dict[str,str]={}
        self.master_to_directed: dict[str,list[tuple[str,str]]]={}

    def _load(self) -> None:
        if self._loaded: return
        s=self.root/'rhea-reaction-smiles.tsv'; d=self.root/'rhea-directions.tsv'
        if not s.is_file() or not d.is_file():
            self._loaded=True; return
        sf=pd.read_csv(s,sep='\t',header=None,names=['id','smiles'],dtype=str).fillna('')
        self.smiles={str(x.id):str(x.smiles) for x in sf.itertuples(index=False)}
        df=pd.read_csv(d,sep='\t',dtype=str).fillna('')
        for row in df.itertuples(index=False):
            master=str(row.RHEA_ID_MASTER)
            self.master_to_directed.setdefault(master,[]).extend([
                (str(row.RHEA_ID_LR),'left_to_right'),
                (str(row.RHEA_ID_RL),'right_to_left'),
            ])
        self._loaded=True

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        self._load()
        raw=seed.value.upper().replace('RHEA:','')
        if not raw.isdigit(): return AdapterResult(status='invalid',message='invalid Rhea id')
        candidates=[(raw,'direct')]
        if raw in self.master_to_directed:
            candidates=self.master_to_directed[raw][:limit]
        resources=[]; children=[]
        for rid,direction in candidates:
            smi=self.smiles.get(str(rid))
            if smi is None: continue
            resource=DiscoveredResource(
                source=self.name,kind='reaction_annotation',canonical_key=f'RHEA:{rid}',
                payload={'rhea_directed_id':f'RHEA:{rid}','reaction_smiles':smi,'direction':direction},
                source_uri=f'https://www.rhea-db.org/rhea/{rid}',
            )
            resources.append(resource)
            if raw in self.master_to_directed:
                children.append(_child('rhea',f'RHEA:{rid}',resource,'rhea_master_direction',seed))
        if not resources:
            return AdapterResult(status='not_found',message=f'Rhea {seed.value} not in pinned local snapshot')
        return AdapterResult(tuple(resources),tuple(children))



class AlphaFoldDBAdapter:
    name='alphafold_db'
    accepted_kinds=frozenset({'uniprot'})

    def __init__(self,session: requests.Session | None=None):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        acc=seed.value.strip().upper()
        r=self.session.get(f'https://alphafold.ebi.ac.uk/api/prediction/{quote(acc,safe="")}',timeout=25)
        if r.status_code==404:
            return AdapterResult(status='not_found',message=f'AlphaFoldDB {acc} not found')
        r.raise_for_status()
        rows=r.json()
        if not isinstance(rows,list) or not rows:
            return AdapterResult(status='not_found',message=f'AlphaFoldDB {acc} empty')
        resources=[]
        for row in rows[:max(1,limit)]:
            model_id=str(row.get('modelEntityId') or row.get('entryId') or acc)
            payload={
                'uniprot':acc,
                'model_id':model_id,
                'tool_used':row.get('toolUsed'),
                'provider_id':row.get('providerId'),
                'global_metric_value':row.get('globalMetricValue'),
                'fraction_plddt_very_low':row.get('fractionPlddtVeryLow'),
                'fraction_plddt_low':row.get('fractionPlddtLow'),
                'fraction_plddt_confident':row.get('fractionPlddtConfident'),
                'fraction_plddt_very_high':row.get('fractionPlddtVeryHigh'),
                'latest_version':row.get('latestVersion'),
                'model_created_date':row.get('modelCreatedDate'),
                'sequence_version_date':row.get('sequenceVersionDate'),
                'sequence_start':row.get('sequenceStart'),
                'sequence_end':row.get('sequenceEnd'),
                'pdb_url':row.get('pdbUrl'),
                'cif_url':row.get('cifUrl'),
                'pae_image_url':row.get('paeImageUrl'),
                'pae_doc_url':row.get('paeDocUrl'),
            }
            resources.append(DiscoveredResource(
                source=self.name,kind='predicted_structure',canonical_key=model_id,
                payload=payload,
                source_uri=str(row.get('pdbUrl') or row.get('cifUrl') or f'https://alphafold.ebi.ac.uk/entry/{acc}'),
            ))
            if materialize:
                structure_url=str(row.get('cifUrl') or row.get('pdbUrl') or '')
                if structure_url:
                    sr=self.session.get(structure_url,timeout=30)
                    if sr.status_code==200 and sr.text.strip():
                        suffix='.cif' if '.cif' in structure_url.lower() else '.pdb'
                        resources.append(DiscoveredResource(
                            source=self.name,kind='predicted_structure_file',
                            canonical_key=f'{model_id}:{suffix[1:]}',
                            payload={
                                'uniprot':acc,'model_id':model_id,
                                'format':suffix[1:],'parent_model_key':model_id,
                            },
                            source_uri=structure_url,blob_text=sr.text,blob_suffix=suffix,
                        ))
        return AdapterResult(tuple(resources),())


class RCSBPDBAdapter:
    name='rcsb_pdb'
    accepted_kinds=frozenset({'pdb'})

    def __init__(self,session: requests.Session | None=None):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        pdb_id=seed.value.strip().upper()
        r=self.session.get(f'https://data.rcsb.org/rest/v1/core/entry/{quote(pdb_id,safe="")}',timeout=25)
        if r.status_code==404:
            return AdapterResult(status='not_found',message=f'PDB {pdb_id} not found')
        r.raise_for_status()
        row=r.json()
        citations=row.get('citation') or []
        primary=next((x for x in citations if str(x.get('rcsb_is_primary') or '').upper()=='Y'),citations[0] if citations else {})
        doi=normalize_doi(str(primary.get('pdbx_database_id_DOI') or ''))
        pmid=str(primary.get('pdbx_database_id_PubMed') or '')
        methods=[str(x.get('method') or '') for x in (row.get('exptl') or []) if x.get('method')]
        payload={
            'pdb_id':pdb_id,
            'title':str((row.get('struct') or {}).get('title') or ''),
            'experimental_methods':methods,
            'initial_release_date':(row.get('rcsb_accession_info') or {}).get('initial_release_date'),
            'resolution_combined':(row.get('rcsb_entry_info') or {}).get('resolution_combined'),
            'polymer_entity_count':(row.get('rcsb_entry_info') or {}).get('polymer_entity_count'),
            'primary_citation_title':str(primary.get('title') or ''),
            'primary_citation_doi':doi,
            'primary_citation_pmid':pmid,
        }
        resource=DiscoveredResource(
            source=self.name,kind='experimental_structure',canonical_key=pdb_id,
            payload=payload,source_uri=f'https://www.rcsb.org/structure/{pdb_id}',
        )
        resources=[resource]
        if materialize:
            cif_url=f'https://files.rcsb.org/download/{pdb_id}.cif'
            sr=self.session.get(cif_url,timeout=30)
            if sr.status_code==200 and sr.text.strip():
                resources.append(DiscoveredResource(
                    source=self.name,kind='experimental_structure_file',
                    canonical_key=f'{pdb_id}:cif',
                    payload={
                        'pdb_id':pdb_id,'format':'cif',
                        'parent_structure_key':pdb_id,
                    },
                    source_uri=cif_url,blob_text=sr.text,blob_suffix='.cif',
                ))
        children=[]
        if doi: children.append(_child('doi',doi,resource,'pdb_primary_citation',seed))
        if pmid: children.append(_child('pmid',pmid,resource,'pdb_primary_citation',seed))
        citation_title=str(primary.get('title') or '').strip()
        if citation_title:
            children.append(_child('title',citation_title,resource,'pdb_primary_citation',seed))
        elif payload['title']:
            children.append(_child('title',payload['title'],resource,'pdb_structure_title',seed))
        return AdapterResult(tuple(resources),tuple(children))


class SabioRKAdapter:
    name='sabio_rk'
    accepted_kinds=frozenset({'uniprot','rhea','literature_query'})

    def __init__(self,session: requests.Session | None=None):
        self.session=session or requests.Session()
        self.session.headers.setdefault('User-Agent',USER_AGENT)

    def _query(self,seed: EnrichmentSeed) -> str:
        if seed.kind.lower()=='uniprot':
            return f'UniProtKB_AC:"{seed.value.strip()}"'
        if seed.kind.lower()=='rhea':
            return f'RheaReactionID:"{seed.value.strip()}"'
        return seed.value.strip()

    def discover(self,seed: EnrichmentSeed,*,limit: int=10,materialize: bool=False) -> AdapterResult:
        endpoint='https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/entryIDs'
        r=self.session.get(endpoint,params={'q':self._query(seed),'format':'txt'},timeout=25,headers={'Accept':'text/plain'})
        ctype=str(r.headers.get('Content-Type') or '').lower()
        text=r.text.strip()
        if r.status_code>=400:
            return AdapterResult(status='unavailable',message=f'HTTP {r.status_code}')
        if 'text/html' in ctype or text.lower().startswith('<!doctype html'):
            return AdapterResult(status='unavailable',message='legacy SABIO-RK REST route currently resolves to web UI on this host')
        ids=[]
        for token in re.findall(r'\b\d+\b',text):
            if token not in ids: ids.append(token)
        resources=[]
        for entry_id in ids[:max(1,limit)]:
            resources.append(DiscoveredResource(
                source=self.name,kind='kinetic_entry_reference',canonical_key=entry_id,
                payload={'entry_id':entry_id,'query':self._query(seed)},
                source_uri=f'https://sabiork.h-its.org/newSearch?q=EntryID:{entry_id}',
            ))
        return AdapterResult(tuple(resources),(),status='ok' if resources else 'not_found')


def default_adapters(rhea_root: str | Path='results/fibre_rhea_mapping_v1'):
    return (
        UniProtAdapter(),
        AlphaFoldDBAdapter(),
        LocalRheaAdapter(rhea_root),
        RCSBPDBAdapter(),
        EuropePMCAdapter(),
        CrossrefAdapter(),
        PublicationURLAdapter(),
    )


def optional_adapters():
    return (SabioRKAdapter(),)
