from __future__ import annotations

import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TRAIN = ROOT / 'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
BENCH = ROOT / 'results/broad_rhea_fair_benchmarks_v1'
POOL_CELL = BENCH / 'broad_pair_hash_holdout_both_seen'
OUT = ROOT / 'results/bridge_relation_seed_strength_v2'
ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 50.0, 100.0)


def z(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    return (x-x.mean())/max(float(x.std()),1e-8)


def rank_of_best(scores: np.ndarray, target_rows: np.ndarray, lexical: np.ndarray) -> int:
    vals=scores[target_rows]
    best=float(vals.max())
    best_targets=target_rows[vals==best]
    best_row=int(best_targets[np.argmin(lexical[best_targets])])
    return int(np.count_nonzero(scores>best)+np.count_nonzero((scores==best)&(lexical<lexical[best_row]))+1)


def metrics(ranks: list[int]) -> dict[str,float]:
    r=np.asarray(ranks,dtype=np.int64)
    return {'queries':int(len(r)),'mrr':float((1/r).mean()),'hit10':float((r<=10).mean()),'hit100':float((r<=100).mean()),'median_rank':float(np.median(r))}


def load_relation_delta():
    train=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id'])
    newer=pd.concat([
        pd.read_csv(POOL_CELL/'train_pairs.csv',dtype=str),
        pd.read_csv(POOL_CELL/'test_pairs.csv',dtype=str),
    ],ignore_index=True).fillna('').drop_duplicates(['protein_id','reaction_id'])
    old=set(zip(train.protein_id,train.reaction_id))
    outer=set()
    for d in BENCH.iterdir():
        p=d/'test_pairs.csv'
        if p.exists():
            x=pd.read_csv(p,dtype=str).fillna('')
            outer |= set(zip(x.protein_id.astype(str),x.reaction_id.astype(str)))
    # Relation development surface is the source-expansion delta, with every
    # frozen outer-test relation removed before any strength selection.
    new=(set(zip(newer.protein_id,newer.reaction_id))-old)-outer
    by_r=defaultdict(list); by_p=defaultdict(list)
    for p,r in train[['protein_id','reaction_id']].itertuples(index=False): by_r[str(r)].append(str(p)); by_p[str(p)].append(str(r))
    trg_r=defaultdict(list); trg_p=defaultdict(list)
    for p,r in new:
        if r in by_r: trg_r[r].append(p)
        if p in by_p: trg_p[p].append(r)
    return train,by_r,by_p,trg_r,trg_p,new


def evaluate(direction:str, index:FibreCandidateIndex):
    train,by_r,by_p,trg_r,trg_p,new=load_relation_delta()
    if direction=='r2e':
        candidate_ids=index.protein_ids; cindex=index.protein_index; C=index.protein_embeddings
        query_ids=sorted(trg_r); qindex=index.reaction_index; Q=index.reaction_embeddings; seed_map=by_r; target_map=trg_r
    else:
        candidate_ids=index.reaction_ids; cindex=index.reaction_index; C=index.reaction_embeddings
        query_ids=sorted(trg_p); qindex=index.protein_index; Q=index.protein_embeddings; seed_map=by_p; target_map=trg_p
    lexical=np.empty(len(candidate_ids),dtype=np.int64); order=np.argsort(np.asarray(candidate_ids,dtype=object),kind='stable'); lexical[order]=np.arange(len(order))
    ranks={a:[] for a in ALPHAS}; audit=[]
    for qi,q in enumerate(query_ids):
        if q not in qindex: continue
        seeds=[cindex[x] for x in seed_map[q] if x in cindex]
        targets=[cindex[x] for x in target_map[q] if x in cindex]
        if not seeds or not targets: continue
        with torch.no_grad():
            base=(C @ Q[qindex[q]]).float().cpu().numpy().astype(np.float64)
            srows=torch.as_tensor(seeds,dtype=torch.long,device=index.device)
            seed=(C @ C.index_select(0,srows).T).max(dim=1).values.float().cpu().numpy().astype(np.float64)
        bz=z(base); sz=z(seed); sidx=np.asarray(seeds,dtype=np.int64); tidx=np.asarray(targets,dtype=np.int64)
        for a in ALPHAS:
            score=bz+float(a)*sz
            score[sidx]=-np.inf
            ranks[a].append(rank_of_best(score,tidx,lexical))
        audit.append({'query_id':q,'seed_count':len(seeds),'target_count':len(targets)})
        if (qi+1)%100==0: print(direction,qi+1,'/',len(query_ids),flush=True)
    table=[]
    for a in ALPHAS: table.append({'alpha':a,**metrics(ranks[a])})
    zero=table[0]
    positive=[x for x in table[1:] if x['mrr']>zero['mrr'] and x['hit10']>=zero['hit10']]
    selected=max(positive,key=lambda x:(x['mrr'],x['hit10'],-x['alpha'])) if positive else zero
    return {'direction':direction,'protocol':{'train_relations':'clean2023 only','targets':'source-expansion relations absent from clean2023 and absent from every frozen seven-cell outer test; query entity must be seen in clean2023','seed':'all clean2023 positives for the query','target_leakage':False,'outer_test_relation_firewall':True,'selection':'max MRR among nonzero alpha with Hit@10 non-regression vs alpha=0; otherwise alpha=0'},'alpha_grid':table,'selected':selected,'audit':{'eligible_queries':len(audit),'mean_seed_count':float(np.mean([x['seed_count'] for x in audit])),'median_seed_count':float(np.median([x['seed_count'] for x in audit])),'target_relations':int(sum(x['target_count'] for x in audit))}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);index=FibreCandidateIndex(device=args.device)
    result={'schema':'bridge-relation-seed-strength-v2','status':'development_relation_delta_only','r2e':evaluate('r2e',index),'e2r':evaluate('e2r',index),'outer_test_metrics_used_for_alpha_selection':False}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
