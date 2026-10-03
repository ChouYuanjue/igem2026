from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.active.fibre.evidence.enrichment import (
    AdapterResult, DiscoveredResource, EnrichmentSeed, EnrichmentStore,
)
from projects.active.fibre.evidence.publication_context import (
    extract_candidate_blocks, explicit_quantities,
)
from projects.active.fibre.evidence.source_adapters import EuropePMCAdapter


def _seed_from_row(row) -> EnrichmentSeed:
    return EnrichmentSeed(
        kind=str(row['kind']),value=str(row['value']),origin=str(row['origin']),
        parent_resource_id=row['parent_resource_id'],
        context=json.loads(str(row['context_json']) or '{}'),
    )


def _resource_contexts(store: EnrichmentStore, resource_id: str) -> list[dict]:
    rows=store.db.execute(
        '''SELECT DISTINCT s.context_json
           FROM edges e JOIN seeds s ON s.seed_id=e.seed_id
           WHERE e.resource_id=?''',
        (resource_id,),
    ).fetchall()
    out=[]; seen=set()
    for row in rows:
        ctx=json.loads(str(row['context_json']) or '{}')
        key=json.dumps(ctx,sort_keys=True,ensure_ascii=False)
        if key in seen: continue
        seen.add(key); out.append(ctx)
    return out


def materialize_missing_fulltext(store: EnrichmentStore,limit: int,refresh: bool) -> dict:
    rows=store.db.execute(
        '''SELECT kind,value,origin,parent_resource_id,context_json
           FROM seeds WHERE kind='pmcid' ORDER BY seed_id LIMIT ?''',
        (int(limit),),
    ).fetchall()
    adapter=EuropePMCAdapter()
    attempted=0; fulltext_added=0; failures=[]
    for row in rows:
        seed=_seed_from_row(row)
        if not refresh:
            existing=store.db.execute(
                '''SELECT 1 FROM resources r JOIN edges e ON e.resource_id=r.resource_id
                   WHERE e.seed_id=? AND r.kind='publication_fulltext_xml' LIMIT 1''',
                (seed.seed_id,),
            ).fetchone()
            if existing is not None:
                continue
        attempted+=1
        try:
            result=adapter.discover(seed,limit=1,materialize=True)
        except Exception as exc:
            result=AdapterResult(status='failed',message=f'{type(exc).__name__}: {exc}')
        for resource in result.resources:
            if resource.kind=='publication_fulltext_xml':
                if store.add_resource(seed,'europe_pmc_fulltext',resource):
                    fulltext_added+=1
        for child in result.next_seeds:
            store.link_child_seed(seed,'europe_pmc_fulltext',child)
        store.record_run(seed,'europe_pmc_fulltext',result)
        if result.status!='ok':
            failures.append({'pmcid':seed.value,'status':result.status,'message':result.message})
    return {
        'attempted':attempted,'fulltext_added':fulltext_added,
        'failure_count':len(failures),'failures':failures[:30],
    }


def extract_graph_candidates(
    store: EnrichmentStore,limit: int,families: set[str] | None,refresh: bool
) -> dict:
    rows=store.db.execute(
        '''SELECT resource_id,canonical_key,payload_json,blob_path,source_uri
           FROM resources
           WHERE kind='publication_fulltext_xml' AND blob_path IS NOT NULL
           ORDER BY resource_id LIMIT ?''',
        (int(limit),),
    ).fetchall()
    fulltexts=0; candidate_count=0; new_count=0
    family_counts={}
    for row in rows:
        resource_id=str(row['resource_id'])
        blob=store.root/str(row['blob_path'])
        if not blob.is_file():
            continue
        fulltexts+=1
        payload=json.loads(str(row['payload_json']) or '{}')
        contexts=_resource_contexts(store,resource_id)
        xml=blob.read_text(encoding='utf-8')
        blocks=extract_candidate_blocks(xml,families=families)
        candidate_count+=len(blocks)
        synthetic=EnrichmentSeed(
            kind='fulltext_resource',value=resource_id,
            origin='deterministic_context_extractor',
            parent_resource_id=resource_id,
            context={'biological_contexts':contexts},
        )
        store.add_seed(synthetic)
        for block in blocks:
            for fam in block.cue_families:
                family_counts[fam]=family_counts.get(fam,0)+1
            candidate=DiscoveredResource(
                source='deterministic_context_extractor',
                kind='publication_context_candidate',
                canonical_key=f'{resource_id}:{block.block_id}',
                source_uri=str(row['source_uri'] or ''),
                payload={
                    'source_fulltext_resource_id':resource_id,
                    'publication_key':str(row['canonical_key']),
                    'publication_identifiers':payload,
                    'block_id':block.block_id,
                    'block_type':block.block_type,
                    'source_index':block.source_index,
                    'section_path':list(block.section_path),
                    'cue_families':list(block.cue_families),
                    'text':block.text,
                    'explicit_quantities':explicit_quantities(block.text),
                    'biological_contexts':contexts,
                    'promotion_status':'candidate_source_block_only',
                },
            )
            if refresh:
                store.db.execute('DELETE FROM resources WHERE resource_id=?',(candidate.resource_id,))
                store.db.commit()
            if store.add_resource(synthetic,'deterministic_context_extractor',candidate):
                new_count+=1
    return {
        'fulltexts_processed':fulltexts,
        'candidate_blocks':candidate_count,
        'new_candidate_resources':new_count,
        'cue_family_counts':dict(sorted(family_counts.items())),
    }


def main() -> None:
    p=argparse.ArgumentParser(description='Materialize and deterministically extract publication context from a FIBRE enrichment graph.')
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--materialize-missing',action='store_true')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--families',default='',help='Comma-separated cue families; empty means all')
    args=p.parse_args()
    families={x.strip() for x in args.families.split(',') if x.strip()} or None
    store=EnrichmentStore(args.graph)
    try:
        materialized={}
        if args.materialize_missing:
            materialized=materialize_missing_fulltext(
                store,max(1,args.limit),bool(args.refresh)
            )
        extraction=extract_graph_candidates(
            store,max(1,args.limit),families,bool(args.refresh)
        )
        summary={
            'schema':'fibre-publication-context-graph-v1',
            'graph':str(args.graph),
            'materialization':materialized,
            'extraction':extraction,
            'store':store.summary(),
            'policy':(
                'Candidate blocks are deterministic source-linked observations only; '
                'they are not assigned to enzyme-reaction observations or promoted into ranking.'
            ),
        }
        (args.graph/'context_extraction_summary.json').write_text(
            json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'
        )
        print(json.dumps(summary,indent=2,ensure_ascii=False))
    finally:
        store.close()


if __name__=='__main__':
    main()
