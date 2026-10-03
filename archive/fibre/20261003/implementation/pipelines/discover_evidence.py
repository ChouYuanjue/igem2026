from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from projects.active.fibre.evidence.enrichment import (
    EnrichmentSeed, EnrichmentStore, run_discovery,
)
from projects.active.fibre.evidence.publication_sources import publication_seed_values
from projects.active.fibre.evidence.source_adapters import (
    default_adapters, optional_adapters,
)

ROOT=Path(__file__).resolve().parents[4]
MARTS=ROOT/'data/terpene_marts/marts_reaction_pairs.tsv'


def parse_seed(text: str) -> EnrichmentSeed:
    if ':' not in text:
        raise ValueError('seed must be KIND:VALUE')
    kind,value=text.split(':',1)
    return EnrichmentSeed(kind=kind.strip(),value=value.strip(),origin='cli')


def load_seed_file(path: Path) -> list[EnrichmentSeed]:
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        out.append(EnrichmentSeed(
            kind=str(row['kind']),value=str(row['value']),
            origin=str(row.get('origin') or f'file:{path.name}'),
            context=dict(row.get('context') or {}),
        ))
    return out


def marts_row_seeds(row: pd.Series, *, row_index: int) -> list[EnrichmentSeed]:
    out=[]
    enzyme=' '.join(str(row.get('enzyme_name') or '').split())
    substrate=' '.join(str(row.get('substrate_name') or '').split())
    product=' '.join(str(row.get('product_name') or '').split())
    species=' '.join(str(row.get('species') or '').split())
    context={
        'marts_row':int(row_index),
        'enzyme_id':str(row.get('enzyme_id') or ''),
        'reaction_signature':str(row.get('reaction_signature') or ''),
        'uniprot_id':str(row.get('uniprot_id') or ''),
        'enzyme_name':enzyme,
        'substrate_name':substrate,
        'product_name':product,
        'species':species,
        'tps_class':str(row.get('tps_class') or ''),
    }
    accession=str(row.get('uniprot_id') or '').strip()
    if accession:
        out.append(EnrichmentSeed(
            kind='uniprot',value=accession,origin='marts_row',context=context,
        ))
    publication=str(row.get('publication') or '').strip()
    for kind,value in publication_seed_values(publication):
        out.append(EnrichmentSeed(
            kind=kind,value=value,origin='marts_publication',context=context,
        ))
    terms=[x for x in (enzyme,substrate,product,species) if x]
    if terms:
        query=' '.join(dict.fromkeys(terms))
        out.append(EnrichmentSeed(
            kind='literature_query',value=query,origin='marts_biological_query',
            context=context,
        ))
    return out


def selected_adapters(names: set[str] | None):
    adapters=list(default_adapters())
    if not names: return adapters
    known={a.name:a for a in [*adapters,*optional_adapters()]}
    missing=sorted(names-set(known))
    if missing:
        raise ValueError(f'unknown adapters: {missing}; known={sorted(known)}')
    return [known[n] for n in sorted(names)]


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(
        description='Lazy, budgeted biological evidence discovery for FIBRE.'
    )
    p.add_argument('--seed',action='append',default=[],help='KIND:VALUE; repeatable')
    p.add_argument('--seed-file',type=Path,action='append',default=[],help='JSONL with kind/value')
    p.add_argument('--marts-row',type=int,action='append',default=[],help='Seed from one raw MARTS row')
    p.add_argument('--marts-all',action='store_true',help='Explicitly allow sequential MARTS row seeding')
    p.add_argument('--marts-limit',type=int,default=0,help='Required positive cap with --marts-all')
    p.add_argument('--sources',default='',help='Comma-separated adapter names; empty means all')
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--budget',type=int,default=100,help='Maximum new resource events')
    p.add_argument('--per-adapter-limit',type=int,default=10)
    p.add_argument('--materialize',action='store_true',help='Cache open full text / HTML blobs when adapters support it')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'results/fibre_enrichment_graph_v1')
    return p


def main() -> None:
    args=build_parser().parse_args()
    seeds=[parse_seed(x) for x in args.seed]
    for path in args.seed_file:
        seeds.extend(load_seed_file(path))

    raw=None
    if args.marts_row or args.marts_all:
        raw=pd.read_csv(MARTS,sep='\t',dtype=str).fillna('')
    for idx in args.marts_row:
        if idx<0 or idx>=len(raw):
            raise IndexError(f'MARTS row out of range: {idx}')
        seeds.extend(marts_row_seeds(raw.iloc[idx],row_index=idx))
    if args.marts_all:
        if args.marts_limit<=0:
            raise ValueError('--marts-all requires a positive --marts-limit')
        for idx,row in raw.head(args.marts_limit).iterrows():
            seeds.extend(marts_row_seeds(row,row_index=int(idx)))

    if not seeds:
        raise ValueError('no seeds supplied; use --seed, --seed-file or --marts-row')

    names={x.strip() for x in args.sources.split(',') if x.strip()} or None
    store=EnrichmentStore(args.out)
    try:
        summary=run_discovery(
            seeds,selected_adapters(names),store,
            max_depth=max(0,args.depth),
            max_resources=max(1,args.budget),
            per_adapter_limit=max(1,args.per_adapter_limit),
            materialize=bool(args.materialize),
            refresh=bool(args.refresh),
        )
        summary['initial_seed_count']=len(seeds)
        summary['output_dir']=str(args.out)
        (args.out/'last_run_summary.json').write_text(
            json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'
        )
        print(json.dumps(summary,indent=2,ensure_ascii=False))
    finally:
        store.close()


if __name__=='__main__':
    main()
