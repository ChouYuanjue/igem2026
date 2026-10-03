from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.bime_rank.scripts import run_bime_r2e_clipzyme_expert_v1 as clip
from reproducibility.bime_rank.scripts import run_r2e_lambdarank_fusion_v1 as base
from projects.active.fibre.evidence.biological_witness import (
    PROTEIN_EMBEDDINGS,
    PROTEIN_ENTRIES,
    _protein_local_energy,
    _reaction_blocks,
    _reaction_local_energy,
)
from projects.active.fibre.runtime.cli import (
    load_feature_schema,
    load_registered_reaction_feature_library,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=(0,1,2))
    ap.add_argument('--diagnostics', type=Path, required=True)
    ap.add_argument('--query-metrics', type=Path, required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    fold=args.fold; device=torch.device(args.device)
    diag=pd.read_csv(args.diagnostics,dtype={'query_id':str})
    qm=pd.read_csv(args.query_metrics,dtype={'query_id':str})
    metrics=qm.set_index('query_id')

    entries=pd.read_csv(PROTEIN_ENTRIES,dtype=str)
    ids=entries.Entry.astype(str).tolist(); pindex={p:i for i,p in enumerate(ids)}
    esmc=np.load(PROTEIN_EMBEDDINGS,mmap_mode='r')
    clip_pt,_,clip_lookup,_,_=clip._load_clip_assets(ids,device)

    schema=load_feature_schema(base.PRIMARY_MODELS/f'fold{fold}')
    reaction_features,reaction_ids=load_registered_reaction_feature_library(base.REACTIONS,schema)
    rindex={r:i for i,r in enumerate(reaction_ids)}; views=_reaction_blocks(reaction_features)
    train=pd.read_csv(base.DEV_ROOT/'baseline_base'/f'fold{fold}'/'training_pairs.csv',dtype=str).fillna('')
    train=train[train.reaction_id.isin(rindex)&train.protein_id.isin(pindex)].copy()
    train_reactions=sorted(train.reaction_id.unique())
    train_rows=np.asarray([rindex[r] for r in train_reactions],dtype=np.int64)

    records=[]
    for row in diag.itertuples(index=False):
        q=str(row.query_id)
        top=json.loads(row.top10_predicted_proteins)
        if not top: continue
        candidate=str(top[0]); crow=pindex[candidate]
        renergy,_=_reaction_local_energy(rindex[q],train_rows,views)
        finite=np.flatnonzero(np.isfinite(renergy))
        rk=min(len(finite),max(1,int(math.ceil(math.sqrt(len(finite))))))
        chosen=finite[np.argpartition(renergy[finite],rk-1)[:rk]]
        r_boundary=float(np.max(renergy[chosen])); r_nearest=float(np.min(renergy[finite]))
        neighbour_ids=[train_reactions[int(i)] for i in chosen]
        local_pairs=train[train.reaction_id.isin(neighbour_ids)].drop_duplicates(['reaction_id','protein_id'])
        source_ids=sorted(local_pairs.protein_id.unique())
        source_rows=np.asarray([pindex[x] for x in source_ids],dtype=np.int64)
        penergy,_=_protein_local_energy(crow,source_rows,esmc,clip_pt=clip_pt,clip_lookup=clip_lookup,device=device)
        pf=penergy[np.isfinite(penergy)]
        pk=min(len(pf),max(1,int(math.ceil(math.sqrt(len(pf))))))
        p_boundary=float(np.partition(pf,pk-1)[pk-1]); p_nearest=float(np.min(pf))
        nearest_idx=int(np.nanargmin(penergy)); nearest_source=source_ids[nearest_idx]
        m=metrics.loc[q]
        records.append({
            'fold':fold,'query_id':q,'top1_protein':candidate,
            'reaction_locality_ratio':r_nearest/max(r_boundary,1e-12),
            'protein_locality_ratio':p_nearest/max(p_boundary,1e-12),
            'reaction_nearest_energy':r_nearest,'protein_nearest_energy':p_nearest,
            'nearest_support_protein':nearest_source,
            'same_protein_precedent':bool(nearest_source==candidate),
            'reciprocal_rank':float(m.reciprocal_rank),'average_precision':float(m.average_precision),
            'ndcg_at_10':float(m.ndcg_at_10),'hit_at_10':float(m.hit_at_10),
            'best_positive_rank':float(m.best_positive_rank),
        })
        if len(records)%16==0: print(f'applicability {len(records)}/{len(diag)}',flush=True)

    out=pd.DataFrame(records)
    args.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    def spearman(a,b):
        return float(pd.Series(a).corr(pd.Series(b),method='spearman'))
    summary={
        'schema':'product-manifold-applicability-audit-v1',
        'fold':fold,'queries':int(len(out)),
        'selection':'actual canonical-v8 Top-1; no held-out label used to construct locality coordinates',
        'ranking_modified':False,'threshold_selected':False,
        'median_reaction_locality_ratio':float(out.reaction_locality_ratio.median()),
        'median_protein_locality_ratio':float(out.protein_locality_ratio.median()),
        'same_protein_precedent_fraction':float(out.same_protein_precedent.mean()),
        'spearman':{
            'reaction_locality_vs_reciprocal_rank':spearman(out.reaction_locality_ratio,out.reciprocal_rank),
            'protein_locality_vs_reciprocal_rank':spearman(out.protein_locality_ratio,out.reciprocal_rank),
            'sum_locality_vs_reciprocal_rank':spearman(out.reaction_locality_ratio+out.protein_locality_ratio,out.reciprocal_rank),
            'reaction_locality_vs_best_positive_rank':spearman(out.reaction_locality_ratio,out.best_positive_rank),
            'protein_locality_vs_best_positive_rank':spearman(out.protein_locality_ratio,out.best_positive_rank),
        },
        'descriptive_hit10_means':{
            'hit10_reaction_locality_mean':float(out.loc[out.hit_at_10>0,'reaction_locality_ratio'].mean()),
            'miss10_reaction_locality_mean':float(out.loc[out.hit_at_10<=0,'reaction_locality_ratio'].mean()),
            'hit10_protein_locality_mean':float(out.loc[out.hit_at_10>0,'protein_locality_ratio'].mean()),
            'miss10_protein_locality_mean':float(out.loc[out.hit_at_10<=0,'protein_locality_ratio'].mean()),
        },
    }
    sp=args.output.with_suffix('.summary.json'); sp.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
