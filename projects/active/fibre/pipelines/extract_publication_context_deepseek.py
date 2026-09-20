from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import requests

from scripts.starase_navigator.routing.language import DeepSeekResolver

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'results/fibre_publication_context_pilot_v1/candidate_paragraphs.csv'
MARTS=ROOT/'data/terpene_marts/marts_reaction_pairs.tsv'
OUT=ROOT/'results/fibre_publication_context_deepseek_v1'


def _key(row: pd.Series) -> str:
    return f"{str(row.pmcid)}:{int(row.source_paragraph_index)}"


def _explicit_quantities(text: str) -> list[dict[str,str]]:
    patterns=(
        (r'(?P<value>[−–-]?\d+(?:\.\d+)?)\s*(?P<unit>°C|° C|degC|degrees? C)',re.I),
        (r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mM|µM|μM|uM|nM|M)\b',0),
        (r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg\s*(?:/|\s+)m[lL](?:[−-]?1)?|g\s*(?:/|\s+)L(?:[−-]?1)?)',re.I),
        (r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|h|minutes?|mins?|min|seconds?|secs?|sec|s)\b',re.I),
    )
    out=[]
    seen=set()
    for pattern,flags in patterns:
        for match in re.finditer(pattern,text,flags):
            raw=match.group(0)
            key=(raw,match.start())
            if key in seen:
                continue
            seen.add(key)
            out.append({
                'raw':raw,
                'value':match.group('value'),
                'unit':match.group('unit'),
            })
    for match in re.finditer(r'\bpH\s*(?P<value>\d+(?:\.\d+)?)',text,re.I):
        raw=match.group(0)
        key=(raw,match.start())
        if key in seen:
            continue
        seen.add(key)
        out.append({'raw':raw,'value':match.group('value'),'unit':'pH'})
    return out


def _load_existing(path: Path) -> dict[str,dict]:
    if not path.is_file():
        return {}
    out={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        out[str(row['source_key'])]=row
    return out


def _write_checkpoint(path: Path, rows: dict[str,dict]) -> None:
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as fh:
        for key in sorted(rows):
            fh.write(json.dumps(rows[key],ensure_ascii=False,sort_keys=True)+'\n')
    tmp.replace(path)


def _target_id(enzyme_id: str, reaction_signature: str) -> str:
    digest=hashlib.sha256(str(reaction_signature).encode('utf-8')).hexdigest()[:12]
    return f'marts:{enzyme_id}:{digest}'


def _targets_for_article(marts: pd.DataFrame, pmid: str, pmcid: str) -> list[dict]:
    mask=pd.Series(False,index=marts.index)
    if pmid:
        mask |= marts.publication.astype(str).str.contains(str(pmid),regex=False)
    if pmcid:
        mask |= marts.publication.astype(str).str.contains(str(pmcid),regex=False)
    rows=marts[mask].drop_duplicates(['enzyme_id','reaction_signature'])
    targets=[]
    for row in rows.itertuples(index=False):
        targets.append({
            'target_id':_target_id(str(row.enzyme_id),str(row.reaction_signature)),
            'enzyme_name':str(row.enzyme_name),
            'species':str(row.species),
            'tps_class':str(row.tps_class),
            'substrate_name':str(row.substrate_name),
            'product_name':str(row.product_name),
        })
    return targets[:96]


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(SOURCE,dtype=str).fillna('')
    marts=pd.read_csv(MARTS,sep='\t',dtype=str).fillna('')
    source['source_paragraph_index']=source.source_paragraph_index.astype(int)
    checkpoint=OUT/'extractions.jsonl'
    rows=_load_existing(checkpoint)
    resolver=DeepSeekResolver()
    if not resolver.configured:
        raise RuntimeError('DeepSeek is not configured in this process')

    for _,row in source.iterrows():
        key=_key(row)
        if key in rows and rows[key].get('status')=='ok':
            print('cached',key,len(rows[key].get('facts') or []),flush=True)
            continue
        targets=_targets_for_article(marts,str(row.pmid),str(row.pmcid))
        source_context={
            'pmcid':str(row.pmcid),
            'pmid':str(row.pmid),
            'section_title':str(row.get('section_title') or ''),
            'keyword_families':str(row.attribute_families),
        }
        result=None
        error=None
        for attempt in range(3):
            try:
                result=resolver.extract_source_bound_facts(
                    str(row.text),
                    source_context=source_context,
                    target_context=targets,
                )
                if result.get('status')=='ok':
                    break
                error=f"status={result.get('status')}"
            except (requests.RequestException,KeyError,IndexError,json.JSONDecodeError,TypeError,ValueError) as exc:
                error=f'{type(exc).__name__}: {exc}'
            print('retry',key,attempt+1,error,flush=True)
            time.sleep(2.0*(attempt+1))

        if result is None or result.get('status')!='ok':
            rows[key]={
                'source_key':key,
                'pmcid':str(row.pmcid),
                'pmid':str(row.pmid),
                'source_paragraph_index':int(row.source_paragraph_index),
                'source_url':str(row.source_url),
                'status':'failed',
                'error':error or 'unknown',
                'facts':[],
            }
            _write_checkpoint(checkpoint,rows)
            continue

        facts=[]
        for fact in result.get('facts') or []:
            evidence=str(fact['evidence_text'])
            facts.append({
                **fact,
                'explicit_quantities':_explicit_quantities(evidence),
            })
        rows[key]={
            'source_key':key,
            'pmcid':str(row.pmcid),
            'pmid':str(row.pmid),
            'source_paragraph_index':int(row.source_paragraph_index),
            'source_url':str(row.source_url),
            'source_attribute_families':str(row.attribute_families),
            'section_title':str(row.get('section_title') or ''),
            'target_context_count':len(targets),
            'status':'ok',
            'paragraph_role':str(result.get('paragraph_role') or 'unclear'),
            'biocatalyst_application':str(result.get('biocatalyst_application') or 'unspecified'),
            'biocatalyst_evidence_text':str(result.get('biocatalyst_evidence_text') or ''),
            'operation_mode':str(result.get('operation_mode') or 'unspecified'),
            'operation_mode_evidence_text':str(result.get('operation_mode_evidence_text') or ''),
            'explicit_target_id_count':int(result.get('explicit_target_id_count') or 0),
            'model':result.get('model'),
            'response_id':result.get('response_id'),
            'raw_fact_count':int(result.get('raw_fact_count') or 0),
            'rejected_fact_count':int(result.get('rejected_fact_count') or 0),
            'facts':facts,
        }
        _write_checkpoint(checkpoint,rows)
        print('ok',key,'facts',len(facts),'rejected',rows[key]['rejected_fact_count'],flush=True)

    all_rows=[rows[k] for k in sorted(rows)]
    fact_types=Counter(
        f['type']
        for row in all_rows if row.get('status')=='ok'
        for f in row.get('facts') or []
    )
    summary={
        'schema':'fibre-publication-context-deepseek-v1',
        'source_paragraphs':int(len(source)),
        'completed_paragraphs':sum(r.get('status')=='ok' for r in all_rows),
        'failed_paragraphs':sum(r.get('status')!='ok' for r in all_rows),
        'fact_count':sum(len(r.get('facts') or []) for r in all_rows),
        'fact_type_counts':dict(sorted(fact_types.items())),
        'paragraph_role_counts':dict(Counter(str(r.get('paragraph_role') or 'unknown') for r in all_rows)),
        'biocatalyst_application_counts':dict(Counter(str(r.get('biocatalyst_application') or 'unspecified') for r in all_rows)),
        'operation_mode_counts':dict(Counter(str(r.get('operation_mode') or 'unspecified') for r in all_rows)),
        'paragraphs_with_explicit_target_descriptor':sum(int(r.get('explicit_target_id_count') or 0)>0 for r in all_rows),
        'target_linked_facts':sum(
            bool(f.get('target_ids'))
            for r in all_rows for f in r.get('facts') or []
        ),
        'scope_resolved_facts':sum(
            bool(f.get('scope_resolved'))
            for r in all_rows for f in r.get('facts') or []
        ),
        'all_evidence_spans_programmatically_verified':all(
            bool(f.get('source_span_verified'))
            for r in all_rows if r.get('status')=='ok'
            for f in r.get('facts') or []
        ),
        'promotion_status':'candidate_observations_only_not_in_canonical_graph',
        'policy':(
            'DeepSeek only selects exact source spans and semantic fact types. '
            'Numeric/unit parsing is deterministic from accepted evidence spans. '
            'No extracted fact is linked to an enzyme-reaction observation until '
            'experiment scope is independently resolved.'
        ),
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
