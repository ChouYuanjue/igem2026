from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT=Path('/home/s241850073/igem2026')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.e2r_anchored_lambdamart_runtime import AnchoredE2RRuntime
from projects.active.terpene_screening.run_e2r_clipzyme_anchored_lambdamart_v4 import anchored_order_v4

OUT=ROOT/'results/clipzyme_native_extension_v1/e2r_strict650_clipzyme_v4_fair_v1'
BASE=ROOT/'results/clipzyme_native_extension_v1/e2r_strict650_current_system_vs_clipzyme_v1'
CLIP_BASE=ROOT/'results/clipzyme_native_extension_v1/e2r_strict650_mutual_cold_10131_v2_lexical'
QDIR=ROOT/'results/clipzyme_native_extension_v1/strict650_e2r_query_embeddings_v1'
RDIR=ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1'
V4=ROOT/'results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected'

def norm(x):
    x=np.asarray(x,dtype=np.float32); n=np.linalg.norm(x,axis=1,keepdims=True); n[n==0]=1.; return x/n

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def summarize(df):
    out={'query_count':int(len(df)),'candidate_count':int(df.candidate_count.iloc[0]),'positive_rows':int(df.positive_count.sum()),
         'mrr':float(df.mrr.mean()),'map':float(df['map'].mean()),'ndcg_at_10':float(df.ndcg_at_10.mean()),
         'median_best_positive_rank':float(df.best_positive_rank.median()),
         'mean_positive_rank':float((df.mean_positive_rank*df.positive_count).sum()/df.positive_count.sum())}
    for b in (1,5,10,20,50): out[f'hit_at_{b}']=float(df[f'hit_at_{b}'].mean())
    return out

def bootstrap(a,b,seed=20260904,nboot=50000):
    rng=np.random.default_rng(seed); d=np.asarray(b,float)-np.asarray(a,float); vals=[]
    for _ in range(nboot//1000):
        idx=rng.integers(0,len(d),size=(1000,len(d))); vals.append(d[idx].mean(1))
    z=np.concatenate(vals)
    return {'delta':float(d.mean()),'ci95':[float(np.quantile(z,.025)),float(np.quantile(z,.975))]}

clip_qmetrics=pd.read_csv(CLIP_BASE/'official_clipzyme_query_metrics.csv',dtype={'query_id':str})
v3_qmetrics=pd.read_csv(BASE/'catalyst_current_query_metrics.csv',dtype={'query_id':str})
query_ids=clip_qmetrics.query_id.astype(str).tolist()
assert query_ids==v3_qmetrics.query_id.astype(str).tolist() and len(query_ids)==248

rentries=pd.read_csv(RDIR/'entries.csv',dtype=str).fillna('')
supported=set(rentries.loc[rentries.clipzyme_supported.str.lower().eq('true'),'reaction_id'].astype(str))
common_ids=sorted(supported); common_set=set(common_ids); assert len(common_ids)==10131
pairs=pd.read_csv(ROOT/'results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv',dtype=str).fillna('')
pos=pairs[pairs.protein_id.isin(query_ids)&pairs.reaction_id.isin(common_set)].groupby('protein_id').reaction_id.agg(lambda s:set(s.astype(str))).to_dict()
assert set(query_ids)==set(pos) and sum(map(len,pos.values()))==348

# Frozen V4 query embeddings from the same official CLIPZyme encoder used to define this strict set.
qentries=pd.read_csv(QDIR/'entries.csv',dtype=str).fillna('').sort_values('row')
qmat=np.load(QDIR/'embeddings.npy',mmap_mode='r'); qidx={q:int(r) for q,r in qentries[['protein_id','row']].itertuples(index=False)}
assert set(query_ids)<=set(qidx)

runtime=AnchoredE2RRuntime(device='cuda')
full_ids=list(runtime.candidate_ids); full_index={r:i for i,r in enumerate(full_ids)}; assert len(full_ids)==11081
# Build full-universe CLIPZyme candidate matrix with explicit unsupported mask.
rmat=np.load(RDIR/'embeddings.npy',mmap_mode='r')
ridx={r:int(row) for r,row in rentries.loc[rentries.clipzyme_supported.str.lower().eq('true'),['reaction_id','row']].itertuples(index=False)}
clip_support=np.asarray([r in ridx for r in full_ids],dtype=bool); assert int(clip_support.sum())==10131
clip_c=np.zeros((len(full_ids),rmat.shape[1]),dtype=np.float32)
rows=[ridx[r] for r in full_ids if r in ridx]; clip_c[clip_support]=norm(np.asarray(rmat[rows],dtype=np.float32))
clip_ct=torch.as_tensor(clip_c,dtype=torch.float32,device=runtime.device)

cfg=json.loads((V4/'config.json').read_text()); assert sha(V4/'ranker.json')==cfg['ranker_sha256']
sel=cfg['selected_config']; ranker=xgb.Booster(); ranker.load_model(V4/'ranker.json')

records=[]; audit=[]; full_arr=np.asarray(full_ids,dtype=object)
for i,q in enumerate(query_ids,1):
    features=runtime.registered_query_features(q)
    S=runtime._expert_scores(features)
    qv=norm(np.asarray(qmat[[qidx[q]]],dtype=np.float32))[0]
    with torch.no_grad(): cscore=(clip_ct@torch.as_tensor(qv,dtype=torch.float32,device=runtime.device)).detach().cpu().numpy().astype(np.float32)
    order,info=anchored_order_v4(S,cscore,clip_support,True,ranker,protected_prefix=int(sel['protected_prefix']),pool_k=int(sel['pool_k']),prefix_k=int(sel['prefix_k']))
    filtered=[str(x) for x in full_arr[order] if str(x) in common_set]
    assert len(filtered)==10131 and len(set(filtered))==10131
    rankmap={r:j+1 for j,r in enumerate(filtered)}
    ranks=sorted(rankmap[r] for r in pos[q]); p=len(ranks)
    rr=1./ranks[0]; ap=float(np.mean(np.arange(1,p+1,dtype=float)/np.asarray(ranks,float)))
    gains=np.asarray([1. if r in pos[q] else 0. for r in filtered[:10]]); discounts=1./np.log2(np.arange(2,12,dtype=float)); dcg=float((gains*discounts).sum()); ideal=float(discounts[:min(p,10)].sum()); ndcg=dcg/ideal if ideal else 0.
    records.append({'query_id':q,'positive_count':p,'candidate_count':10131,'mrr':rr,'map':ap,'ndcg_at_10':ndcg,'best_positive_rank':ranks[0],'mean_positive_rank':float(np.mean(ranks)),**{f'hit_at_{b}':float(ranks[0]<=b) for b in (1,5,10,20,50)}})
    audit.append({'query_id':q,**info})
    if i%25==0 or i==len(query_ids): print(f'v4 strict {i}/{len(query_ids)}',flush=True)

v4=pd.DataFrame(records); v4.to_csv(OUT/'catalyst_v4_query_metrics.csv',index=False); pd.DataFrame(audit).to_csv(OUT/'routing_audit.csv',index=False)
metrics=['mrr','map','ndcg_at_10','hit_at_1','hit_at_5','hit_at_10','hit_at_20','hit_at_50']
res={'protocol':'Rhea128->141 strict double-cold; 248 protein queries mutual-cold to CLIPZyme and Catalyst training; <=650 aa; identical lexical 10,131 CLIPZyme-supported reaction candidates. V3/V4 route on full 11,081 Catalyst reaction universe first, then filter to common support. V4 ranker/config frozen on clean-dev only before this evaluation.',
     'fairness':{'same_queries':True,'same_candidate_support':True,'mutual_train_cold':True,'selection_uses_test_labels':False,'v4_ranker_sha256':cfg['ranker_sha256'],'caveat':'The 248-query benchmark had prior baseline visibility in the project, but V4 parameters/configuration were not selected on its labels.'},
     'queries':248,'reaction_candidates':10131,'positive_pairs':348,
     'models':{'official_clipzyme':{k:float(clip_qmetrics[k].mean()) for k in metrics},'current_four_expert_v3':summarize(v3_qmetrics),'clipzyme_five_expert_v4':summarize(v4)},
     'paired_bootstrap_50000':{}}
for comp,df in [('v4_minus_clipzyme',clip_qmetrics),('v4_minus_current_v3',v3_qmetrics)]:
    res['paired_bootstrap_50000'][comp]={}
    for metric in metrics:
        b=bootstrap(df[metric].to_numpy(float),v4[metric].to_numpy(float),seed=20260904+metrics.index(metric))
        res['paired_bootstrap_50000'][comp][metric]={df.columns.name or 'baseline':float(df[metric].mean()),'v4':float(v4[metric].mean()),**b}
(OUT/'summary.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
