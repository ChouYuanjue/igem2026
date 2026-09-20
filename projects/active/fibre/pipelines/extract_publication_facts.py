from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from projects.active.fibre.evidence.enrichment import (
    AdapterResult, DiscoveredResource, EnrichmentSeed, EnrichmentStore,
)
from projects.active.fibre.evidence.publication_context import explicit_quantities
from scripts.starase_navigator.routing.language import DeepSeekResolver


def _target_id(context: dict) -> str:
    enzyme=str(context.get('enzyme_id') or context.get('uniprot_id') or '')
    reaction=str(context.get('reaction_signature') or '')
    digest=hashlib.sha256(reaction.encode('utf-8')).hexdigest()[:12]
    return f'marts:{enzyme}:{digest}'


def _targets(contexts: list[dict]) -> list[dict]:
    out=[]; seen=set()
    for ctx in contexts:
        if not isinstance(ctx,dict):
            continue
        target={
            'target_id':_target_id(ctx),
            'enzyme_name':str(ctx.get('enzyme_name') or ''),
            'species':str(ctx.get('species') or ''),
            'tps_class':str(ctx.get('tps_class') or ''),
            'substrate_name':str(ctx.get('substrate_name') or ''),
            'product_name':str(ctx.get('product_name') or ''),
        }
        if target['target_id'] in seen:
            continue
        if not any(target[k] for k in ('enzyme_name','substrate_name','product_name')):
            continue
        seen.add(target['target_id']); out.append(target)
    return out[:96]


def main() -> None:
    p=argparse.ArgumentParser(
        description='Conservative source-bound semantic fact extraction from any FIBRE discovery graph.'
    )
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--allowed-types',default='',help='Comma-separated DeepSeek fact types; empty uses resolver defaults')
    args=p.parse_args()

    resolver=DeepSeekResolver()
    store=EnrichmentStore(args.graph)
    allowed=[x.strip() for x in args.allowed_types.split(',') if x.strip()] or None
    processed=0; succeeded=0; failed=0; facts_added=0; unresolved_target_facts=0
    role_counts={}; type_counts={}
    try:
        rows=store.db.execute(
            '''SELECT resource_id,canonical_key,source_uri,payload_json
               FROM resources WHERE kind='publication_context_candidate'
               ORDER BY resource_id LIMIT ?''',
            (max(1,args.limit),),
        ).fetchall()
        for row in rows:
            payload=json.loads(str(row['payload_json']) or '{}')
            candidate_id=str(row['resource_id'])
            contexts=list(payload.get('biological_contexts') or [])
            seed=EnrichmentSeed(
                kind='context_candidate',value=candidate_id,
                origin='source_bound_fact_extractor',
                parent_resource_id=candidate_id,
                context={'biological_contexts':contexts},
            )
            store.add_seed(seed)
            status=store.run_status(seed,'deepseek_context_extractor')
            if status=='ok' and not args.refresh:
                continue
            processed+=1
            text=str(payload.get('text') or '')
            source_context={
                'publication_key':str(payload.get('publication_key') or ''),
                'publication_identifiers':payload.get('publication_identifiers') or {},
                'section_path':payload.get('section_path') or [],
                'block_type':str(payload.get('block_type') or ''),
                'source_index':payload.get('source_index'),
                'cue_families':payload.get('cue_families') or [],
                'source_uri':str(row['source_uri'] or ''),
            }
            try:
                result=resolver.extract_source_bound_facts(
                    text,allowed_types=allowed,source_context=source_context,
                    target_context=_targets(contexts),
                )
            except Exception as exc:
                result={'status':'failed','facts':[],'error':f'{type(exc).__name__}: {exc}'}

            result_status=str(result.get('status') or 'failed')
            if result_status!='ok':
                failed+=1
                store.record_run(
                    seed,'deepseek_context_extractor',AdapterResult(
                        status=result_status,message=str(result.get('error') or result_status)
                    )
                )
                continue

            succeeded+=1
            role=str(result.get('paragraph_role') or 'unclear')
            role_counts[role]=role_counts.get(role,0)+1
            created=[]
            for fact in result.get('facts') or []:
                kind=str(fact.get('type') or '')
                type_counts[kind]=type_counts.get(kind,0)+1
                if not fact.get('target_ids'):
                    unresolved_target_facts+=1
                key_payload=json.dumps({
                    'candidate':candidate_id,
                    'type':kind,
                    'evidence':fact.get('evidence_text'),
                    'scope':fact.get('scope_text'),
                    'targets':fact.get('target_ids') or [],
                },sort_keys=True,ensure_ascii=False)
                key=hashlib.sha256(key_payload.encode('utf-8')).hexdigest()[:24]
                resource=DiscoveredResource(
                    source='deepseek_context_extractor',
                    kind='publication_fact_candidate',
                    canonical_key=f'{candidate_id}:{key}',
                    source_uri=str(row['source_uri'] or ''),
                    payload={
                        'source_candidate_resource_id':candidate_id,
                        'type':kind,
                        'evidence_text':str(fact.get('evidence_text') or ''),
                        'scope_text':str(fact.get('scope_text') or ''),
                        'scope_resolved':bool(fact.get('scope_resolved')),
                        'target_ids':list(fact.get('target_ids') or []),
                        'target_assignment_status':str(fact.get('target_assignment_status') or 'unresolved'),
                        'source_span_verified':bool(fact.get('source_span_verified')),
                        'explicit_quantities':explicit_quantities(str(fact.get('evidence_text') or '')),
                        'paragraph_role':role,
                        'biocatalyst_application':str(result.get('biocatalyst_application') or 'unspecified'),
                        'biocatalyst_evidence_text':str(result.get('biocatalyst_evidence_text') or ''),
                        'operation_mode':str(result.get('operation_mode') or 'unspecified'),
                        'operation_mode_evidence_text':str(result.get('operation_mode_evidence_text') or ''),
                        'model':result.get('model'),
                        'response_id':result.get('response_id'),
                        'biological_contexts':contexts,
                        'promotion_status':'candidate_fact_only_not_canonical_observation',
                    },
                )
                if store.add_resource(seed,'deepseek_context_extractor',resource):
                    facts_added+=1
                created.append(resource)
            store.record_run(
                seed,'deepseek_context_extractor',
                AdapterResult(resources=tuple(created),status='ok')
            )

        summary={
            'schema':'fibre-source-bound-fact-extraction-v1',
            'graph':str(args.graph),
            'resolver_configured':resolver.configured,
            'processed_candidate_blocks':processed,
            'successful_blocks':succeeded,
            'failed_or_unconfigured_blocks':failed,
            'new_fact_resources':facts_added,
            'unresolved_target_facts':unresolved_target_facts,
            'paragraph_role_counts':dict(sorted(role_counts.items())),
            'fact_type_counts':dict(sorted(type_counts.items())),
            'store':store.summary(),
            'policy':(
                'The language model is only a source-span selector/classifier. '
                'Facts remain candidates until target scope, endpoint and units are independently valid.'
            ),
        }
        (args.graph/'fact_extraction_summary.json').write_text(
            json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'
        )
        print(json.dumps(summary,indent=2,ensure_ascii=False))
    finally:
        store.close()


if __name__=='__main__':
    main()
