from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, xgboost as xgb
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening import run_bime_r2e_seed_context_v1 as r2
from projects.active.terpene_screening import run_bime_e2r_seed_context_v1 as e2
from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as r2base
from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as r2clip
from projects.active.terpene_screening import run_unified_safe_system_e2r_anchored_lambdamart_v3 as e2base
from projects.active.terpene_screening import run_e2r_clipzyme_anchored_lambdamart_v4 as e2clip
from projects.active.terpene_screening.rank_open_world import load_protein_library
from projects.active.terpene_screening.bime_rank_r2e_seed_runtime import masked_order, context_features, lexical_rank

OUT=ROOT/'results/bime_rank_unified_v1/multiseed_scaling_v1'; OUT.mkdir(parents=True,exist_ok=True)
SEED_COUNTS=(1,2,3,5); TARGET_REPS=1

def _metrics(df:pd.DataFrame)->dict:
 return {'trials':int(len(df)),'queries':int(df.query_id.nunique()),'mrr':float((1.0/df.target_rank).mean()),'hit_at_10':float((df.target_rank<=10).mean()),'hit_at_20':float((df.target_rank<=20).mean()),'hit_at_50':float((df.target_rank<=50).mean()),'median_target_rank':float(df.target_rank.median())}

def _finish(rows:list[dict],direction:str,cohort_min_positives:int)->dict:
 d=pd.DataFrame(rows); d.to_csv(OUT/f'{direction.lower()}_target_trials.csv',index=False)
 res={'direction':direction,'design':'same held-out query and same hidden target across nested 1/2/3/5 positive seed sets; all provided seeds masked; frozen one-seed-trained cross-fit context ranker; no multi-seed retuning','cohort_rule':f'at least {cohort_min_positives} positives so 5 seeds plus one fixed hidden target are possible','seed_counts':list(SEED_COUNTS),'target_reps':TARGET_REPS,'external_metrics_used':False,'ranker_training_seed_count':1,'metrics':{}}
 for method in ['context','seed_only','zero_shot']:
  res['metrics'][method]={str(k):_metrics(d[(d.method==method)&(d.seed_count==k)]) for k in SEED_COUNTS}
 # paired per-trial scaling within context
 piv=d[d.method=='context'].pivot_table(index=['fold','query_id','rep'],columns='seed_count',values='target_rank',aggfunc='first').dropna()
 res['paired_context_scaling']={
  'pairs':int(len(piv)),
  'rank_nonworse_1_to_5_fraction':float((piv[5]<=piv[1]).mean()),
  'rank_strictly_better_1_to_5_fraction':float((piv[5]<piv[1]).mean()),
  'median_rank_change_5_minus_1':float((piv[5]-piv[1]).median()),
 }
 (OUT/f'{direction.lower()}_summary.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2)); return res

def run_r2e(device_name='cuda'):
 device=torch.device(device_name); rows=[]
 raw_features,raw_ids=load_protein_library(r2.PROTEIN_ROOT); raw=np.asarray(raw_features,dtype=np.float32); raw_t=torch.as_tensor(raw,dtype=torch.float32,device=device)
 structural_model=r2.load_structural_model()
 for fold in r2.FOLDS:
  context_model=r2.train([r2.load_cache(f) for f in r2.FOLDS if f!=fold],r2base._stable_seed(f'seedctx-fold{fold}'))
  ids,_,reaction_ids,pe0,pe1,re0,re1=r2base._load_fold_embeddings(fold,device); assert ids==raw_ids
  pidx={p:i for i,p in enumerate(ids)}; ridx={x:i for i,x in enumerate(reaction_ids)}; lex=r2base._lexical_rank(ids)
  clip_pt,clip_rows,clip_lookup,clip_rmat,clip_ridx=r2clip._load_clip_assets(ids,device); clip_lex=lex[clip_rows]
  dev=pd.read_csv(r2base.DEV_ROOT/'baseline_base'/f'fold{fold}'/'dev_pairs.csv',dtype=str).fillna(''); pos=dev.groupby('reaction_id').protein_id.apply(lambda s:sorted(set(map(str,s)))).to_dict(); qids=sorted(q for q,ps in pos.items() if len(ps)>=6)
  cell=f'clean2023_internal_double_cold_salted_comprehensive_enzgfm_center_top1_v1_dev_20260901_fold{fold}'; diff=pd.read_csv(r2base.DEV_ROOT/'difficulty'/cell/'reaction_slices.csv',dtype={'reaction_id':str}); smap=dict(zip(diff.reaction_id.astype(str),diff.max_train_drfp_tanimoto.astype(float)))
  for qi,q in enumerate(qids):
   rr=ridx[q]
   with torch.no_grad(): s0=(re0[rr]@pe0.T).float().cpu().numpy().astype(np.float32); s1=(re1[rr]@pe1.T).float().cpu().numpy().astype(np.float32)
   sim=float(smap[q]); use_secondary=sim<r2base.ROUTER_THRESHOLD
   _,inv0=masked_order(s0,lex,set()); _,inv1=masked_order(s1,lex,set())
   cscore=None; cinv=None; ctop=np.empty(0,dtype=np.int32); qsup=q in clip_ridx
   if qsup:
    qt=torch.as_tensor(np.asarray(clip_rmat[clip_ridx[q]],dtype=np.float32),device=device)
    with torch.no_grad(): cscore=(clip_pt@qt).float().cpu().numpy().astype(np.float32)
    co=np.lexsort((clip_lex,-cscore)).astype(np.int32); cinv=np.full(len(cscore),len(co)+1,dtype=np.int32);cinv[co]=np.arange(1,len(co)+1,dtype=np.int32);ctop=clip_rows[co[:r2.POOL_K]]
   base_full=r2.structural_order(structural_model,s0=s0,s1=s1,inv0=inv0,inv1=inv1,use_secondary=use_secondary,similarity=sim,lex=lex,clip_scores=cscore,clip_inv=cinv,clip_lookup=clip_lookup,query_supported=qsup,clip_top=ctop,masked_row=-1)
   prows=np.asarray([pidx[p] for p in pos[q]],dtype=np.int32); rng=np.random.default_rng(r2base._stable_seed(f'bime-r2e-multiseed-scale|{q}')); perm=rng.permutation(prows); target=int(perm[0]); pool=np.asarray(perm[1:6],dtype=np.int32)
   with torch.no_grad(): sims=(raw_t@raw_t[torch.as_tensor(pool,dtype=torch.long,device=device)].T).float().cpu().numpy().astype(np.float32)
   for k in SEED_COUNTS:
    seeds=pool[:k]; seed_scores=sims[:,:k].max(axis=1); masked=set(map(int,seeds)); base_order=base_full[np.fromiter((int(x) not in masked for x in base_full),dtype=bool,count=len(base_full))]; binv=np.full(len(ids),len(base_order)+1,dtype=np.int32);binv[base_order]=np.arange(1,len(base_order)+1,dtype=np.int32)
    sorder,sinv=masked_order(seed_scores,lex,masked); union=np.unique(np.concatenate([base_order[:r2.POOL_K],sorder[:r2.POOL_K]])).astype(np.int32); union=union[np.fromiter((int(x) not in masked for x in union),dtype=bool,count=len(union))]; X=context_features(union,binv,seed_scores,sinv,len(ids)-k); pred=context_model.predict(xgb.DMatrix(X)); take=np.lexsort((lex[union],-pred))[:min(r2.PREFIX_K,len(union))]; sel=union[take]; chosen=np.zeros(len(ids),dtype=bool);chosen[sel]=True; full=np.concatenate([sel,base_order[~chosen[base_order]]]); inv=np.full(len(ids),len(full)+1,dtype=np.int32);inv[full]=np.arange(1,len(full)+1,dtype=np.int32)
    for method,order in [('context',full),('seed_only',sorder),('zero_shot',base_order)]:
     if method=='context': rank=int(inv[target])
     else:
      iv=np.full(len(ids),len(order)+1,dtype=np.int32);iv[order]=np.arange(1,len(order)+1,dtype=np.int32);rank=int(iv[target])
     rows.append({'fold':fold,'query_id':q,'rep':0,'seed_count':k,'method':method,'target_id':ids[target],'target_rank':rank,'positive_count':len(prows)})
   if (qi+1)%50==0: print('R2E multiseed fold',fold,qi+1,'/',len(qids),flush=True)
 return _finish(rows,'R2E',6)

def run_e2r(device_name='cuda'):
 device=torch.device(device_name); rows=[]; v4ranker,vcfg=e2.load_v4_ranker(); scfg=vcfg['selected_config']
 for fold in e2.FOLDS:
  context_model=e2.train([e2.load_cache(f) for f in e2.FOLDS if f!=fold],e2.stable(f'e2r-seedctx-fold{fold}'))
  dev,cids,qids,emb=e2base.load_fold_embeddings(fold); positives=dev.groupby('protein_id').reaction_id.apply(lambda x:sorted(set(map(str,x)))).to_dict(); eligible=[q for q in qids if len(positives[q])>=6]; ridx={r:i for i,r in enumerate(cids)}; lex=lexical_rank(cids); clip_q,clip_q_support,clip_c,clip_c_support=e2clip._clip_assets(cids,qids);qpos={q:i for i,q in enumerate(qids)};rt={n:torch.as_tensor(emb[n][1],dtype=torch.float32,device=device) for n in e2base.NAMES};clip_ct=torch.as_tensor(clip_c,dtype=torch.float32,device=device)
  with torch.no_grad():
   for qi,q in enumerate(eligible):
    qrow=qids.index(q); S=[]
    for n in e2base.NAMES: S.append((torch.as_tensor(emb[n][0][qrow],dtype=torch.float32,device=device)@rt[n].T).cpu().numpy())
    S=np.stack(S).astype(np.float32); csc=(torch.as_tensor(clip_q[qrow],dtype=torch.float32,device=device)@clip_ct.T).cpu().numpy().astype(np.float32); base_full,_=e2clip.anchored_order_v4(S,csc,clip_c_support,bool(clip_q_support[qpos[q]]),v4ranker,protected_prefix=int(scfg['protected_prefix']),pool_k=int(scfg['pool_k']),prefix_k=int(scfg['prefix_k']))
    prows=np.asarray([ridx[r] for r in positives[q]],dtype=np.int32);rng=np.random.default_rng(e2.stable(f'bime-e2r-multiseed-scale|{q}'));perm=rng.permutation(prows);target=int(perm[0]);pool=np.asarray(perm[1:6],dtype=np.int32)
    # per expert candidate x 5 similarities
    sims=[]
    for n in e2base.NAMES:
     M=rt[n]; sims.append((M@M[torch.as_tensor(pool,dtype=torch.long,device=device)].T).cpu().numpy().astype(np.float32))
    for k in SEED_COUNTS:
     seeds=pool[:k]; seed_scores=np.zeros(len(cids),dtype=np.float32)
     for Z in sims: seed_scores += Z[:,:k].max(axis=1)
     seed_scores/=float(len(sims));masked=set(map(int,seeds));base_order=base_full[np.fromiter((int(x) not in masked for x in base_full),dtype=bool,count=len(base_full))];binv=np.full(len(cids),len(base_order)+1,dtype=np.int32);binv[base_order]=np.arange(1,len(base_order)+1,dtype=np.int32);sorder,sinv=masked_order(seed_scores,lex,masked);union=np.unique(np.concatenate([base_order[:e2.POOL_K],sorder[:e2.POOL_K]])).astype(np.int32);union=union[np.fromiter((int(x) not in masked for x in union),dtype=bool,count=len(union))];X=context_features(union,binv,seed_scores,sinv,len(cids)-k);pred=context_model.predict(xgb.DMatrix(X));take=np.lexsort((lex[union],-pred))[:min(e2.PREFIX_K,len(union))];sel=union[take];chosen=np.zeros(len(cids),dtype=bool);chosen[sel]=True;full=np.concatenate([sel,base_order[~chosen[base_order]]]);inv=np.full(len(cids),len(full)+1,dtype=np.int32);inv[full]=np.arange(1,len(full)+1,dtype=np.int32)
     for method,order in [('context',full),('seed_only',sorder),('zero_shot',base_order)]:
      if method=='context':rank=int(inv[target])
      else:iv=np.full(len(cids),len(order)+1,dtype=np.int32);iv[order]=np.arange(1,len(order)+1,dtype=np.int32);rank=int(iv[target])
      rows.append({'fold':fold,'query_id':q,'rep':0,'seed_count':k,'method':method,'target_id':cids[target],'target_rank':rank,'positive_count':len(prows)})
    if (qi+1)%25==0: print('E2R multiseed fold',fold,qi+1,'/',len(eligible),flush=True)
 return _finish(rows,'E2R',6)

def main():
 import argparse;ap=argparse.ArgumentParser();ap.add_argument('--direction',choices=['r2e','e2r','both'],default='both');ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args();summary={'status':'complete','protocol':'frozen one-seed-trained cross-fit context ranker evaluated on same-target nested 1/2/3/5 seed sets; no retuning'}
 if a.direction in ('r2e','both'):summary['r2e']=run_r2e(a.device)
 if a.direction in ('e2r','both'):summary['e2r']=run_e2r(a.device)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
