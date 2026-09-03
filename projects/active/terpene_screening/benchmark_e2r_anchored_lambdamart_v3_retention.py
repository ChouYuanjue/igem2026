from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import pandas as pd
from projects.active.terpene_screening.core.engine import RetrievalEngine
from projects.active.terpene_screening.e2r_anchored_lambdamart_runtime import AnchoredE2RRuntime
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3/retention_gate'
ASSOC=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
OLD_SCHEMA=ROOT/'results/terpene_production_models/marts_adapted_drfp_pu_e2r/feature_schema.json'
SALT='e2r-v3-external-known-retention-v1'; N=48

def h(x): return hashlib.blake2b(f'{SALT}|{x}'.encode(),digest_size=16).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assoc=pd.read_csv(ASSOC,dtype=str).drop_duplicates()
    old=json.loads(OLD_SCHEMA.read_text()); old_ids=set(pd.read_csv(Path(old['protein_ids_file']),dtype={'Entry':str}).Entry.astype(str))
    byp={p:set(g.reaction_id.astype(str)) for p,g in assoc.groupby('protein_id')}
    rt=AnchoredE2RRuntime(device='cuda'); support=set(rt.candidate_ids)
    eligible=[p for p,pos in byp.items() if p not in old_ids and p in rt._protein_libraries['enzgfm'].row_by_id and (pos & support)]
    qids=sorted(eligible,key=lambda x:(h(x),x))[:N]
    if len(qids)!=N: raise RuntimeError('insufficient fixed retention queries')
    engine=RetrievalEngine(); rows=[]
    for i,p in enumerate(qids):
        positives=byp[p]&support
        cand=rt.rank_registered(p); rank={cand.candidate_ids[int(r)]:j+1 for j,r in enumerate(cand.order)}
        best=min(rank[x] for x in positives); cand_top={k:set(cand.top_ids(k)) for k in (3,10,20)}
        row={'protein_id':p,'n_positive':len(positives),'candidate_best_rank':best,'candidate_mrr':1.0/best}
        for k in (3,10,20): row[f'candidate_hit{k}']=float(bool(cand_top[k]&positives)); row[f'candidate_recall{k}']=len(cand_top[k]&positives)/len(positives)
        for obj,k in [('top3',3),('top10',10),('top20',20)]:
            f=engine.rank_frame('rank-reactions',{'enzyme_id':p,'candidate_universe':'general_merged','top_k':k,'ranking_objective':obj,'conformal_mode':'disabled','device':'cuda'})
            ids=set(f.candidate_id.astype(str)); row[f'old_{obj}_hit']=float(bool(ids&positives)); row[f'old_{obj}_recall']=len(ids&positives)/len(positives)
        rows.append(row); print(i+1,p,flush=True)
    d=pd.DataFrame(rows); d.to_csv(OUT/'query_metrics.csv',index=False)
    metrics={}
    for obj,k in [('top3',3),('top10',10),('top20',20)]:
        oldh=d[f'old_{obj}_hit'].mean(); candh=d[f'candidate_hit{k}'].mean(); oldr=d[f'old_{obj}_recall'].mean(); candr=d[f'candidate_recall{k}'].mean()
        metrics[obj]={'old_hit':oldh,'candidate_hit':candh,'delta_hit':candh-oldh,'old_recall':oldr,'candidate_recall':candr,'delta_recall':candr-oldr,'pass':candh>=oldh-0.01 and candr>=oldr-0.01}
    result={'status':'pass' if all(v['pass'] for v in metrics.values()) else 'fail','salt':SALT,'query_count':N,'query_rule':'48 smallest blake2b(salt|protein_id) among clean2023 known proteins absent from old current protein IDs, registered in V3, with supported positives','scope':'retention/memorization only; not cold-start generalization','metrics':metrics,'candidate_mrr':d.candidate_mrr.mean(),'external_labels_used_for_model_selection':False,'retention_labels_used_only_for_production_gate':True}
    (OUT/'result.json').write_text(json.dumps(result,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o))+'\n'); print(json.dumps(result,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
if __name__=='__main__': main()
