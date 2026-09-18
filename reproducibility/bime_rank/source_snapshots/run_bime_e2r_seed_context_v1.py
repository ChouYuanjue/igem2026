from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening import run_unified_safe_system_e2r_anchored_lambdamart_v3 as v3
from projects.active.terpene_screening import run_e2r_clipzyme_anchored_lambdamart_v4 as v4
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
from projects.active.terpene_screening.bime_rank_r2e_seed_runtime import FEATURE_NAMES, context_features, lexical_rank, masked_order

OUT=ROOT/'results/bime_rank_unified_v1/e2r_seed_context_v1'
V4_BUNDLE=ROOT/'results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected'
FOLDS=(0,1,2); POOL_K=100; PREFIX_K=100; SEED_REPS=3
FIXED_CONFIG={'pool_k':POOL_K,'prefix_k':PREFIX_K,'max_depth':2,'eta':0.12,'min_child_weight':5.0,'reg_lambda':1.0,'rounds':80,'lambdarank_pairs':20}


def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(text:str)->int: return int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8).digest(),'big')%(2**32)

def seed_rows(q:str, positives:np.ndarray)->list[int]:
 vals=list(map(int,positives)); out=[]
 if len(vals)<2:return out
 for rep in range(min(SEED_REPS,len(vals))):
  rng=np.random.default_rng(stable(f'bime-e2r-seed-context|{q}|{rep}')); pool=[v for v in vals if v not in out]
  if not pool:break
  out.append(int(pool[int(rng.integers(0,len(pool)))]))
 return out

def load_v4_ranker():
 cfg=json.loads((V4_BUNDLE/'config.json').read_text()); assert sha(V4_BUNDLE/'ranker.json')==cfg['ranker_sha256']; assert cfg['external_evaluation_metrics_used'] is False
 m=xgb.Booster();m.load_model(V4_BUNDLE/'ranker.json');return m,cfg

def rank_metrics(order,hidden,n):
 inv=np.full(n,len(order)+1,dtype=np.int32);inv[order]=np.arange(1,len(order)+1,dtype=np.int32)
 return evaluate_full_candidate_ranks(inv[hidden],len(order))

def prepare_fold(fold:int,device_name:str):
 dev,cids,qids,emb=v3.load_fold_embeddings(fold); positives=dev.groupby('protein_id').reaction_id.apply(lambda x:sorted(set(map(str,x)))).to_dict(); eligible=[q for q in qids if len(positives[q])>=2]
 ridx={r:i for i,r in enumerate(cids)}; lex=lexical_rank(cids); device=torch.device(device_name); base_ranker,vcfg=load_v4_ranker(); scfg=vcfg['selected_config']
 clip_q,clip_q_support,clip_c,clip_c_support=v4._clip_assets(cids,qids); qpos={q:i for i,q in enumerate(qids)}
 rt={n:torch.as_tensor(emb[n][1],dtype=torch.float32,device=device) for n in v3.NAMES}; clip_ct=torch.as_tensor(clip_c,dtype=torch.float32,device=device)
 Xs=[];ys=[];rows_all=[];fallback_all=[];ptr=[0];pos_all=[];posfb_all=[];pptr=[0];trials=[];aud=[];bm=[];sm=[]
 with torch.no_grad():
  for st in range(0,len(eligible),64):
   qs=eligible[st:st+64]; qi=[qids.index(q) for q in qs]
   scores={n:(torch.as_tensor(emb[n][0][qi],dtype=torch.float32,device=device)@rt[n].T).cpu().numpy() for n in v3.NAMES}
   csc=(torch.as_tensor(clip_q[qi],dtype=torch.float32,device=device)@clip_ct.T).cpu().numpy()
   for j,q in enumerate(qs):
    S=np.stack([scores[n][j] for n in v3.NAMES]).astype(np.float32)
    order,_=v4.anchored_order_v4(S,np.asarray(csc[j],dtype=np.float32),clip_c_support,bool(clip_q_support[qpos[q]]),base_ranker,protected_prefix=int(scfg['protected_prefix']),pool_k=int(scfg['pool_k']),prefix_k=int(scfg['prefix_k']))
    prows=np.asarray([ridx[r] for r in positives[q]],dtype=np.int32)
    for rep,srow in enumerate(seed_rows(q,prows)):
     hidden=prows[prows!=srow]; base_order=order[order!=srow]; binv=np.full(len(cids),len(base_order)+1,dtype=np.int32);binv[base_order]=np.arange(1,len(base_order)+1,dtype=np.int32)
     seed=np.zeros(len(cids),dtype=np.float32)
     for n in v3.NAMES:
      M=np.asarray(emb[n][1],dtype=np.float32); seed += M@M[srow]
     seed/=float(len(v3.NAMES)); sorder,sinv=masked_order(seed,lex,{int(srow)})
     union=np.unique(np.concatenate([base_order[:POOL_K],sorder[:POOL_K]])).astype(np.int32);union=union[union!=srow]
     x=context_features(union,binv,seed,sinv,len(cids)-1);y=np.isin(union,hidden).astype(np.uint8)
     Xs.append(x);ys.append(y);rows_all.append(union);fallback_all.append(binv[union]);ptr.append(ptr[-1]+len(union));pos_all.append(hidden);posfb_all.append(binv[hidden]);pptr.append(pptr[-1]+len(hidden))
     tid=f'{q}|seed={cids[srow]}|rep={rep}';trials.append(tid);bm.append({'fold':fold,'trial_id':tid,'query_id':q,'seed_id':cids[srow],**rank_metrics(base_order,hidden,len(cids))});sm.append({'fold':fold,'trial_id':tid,'query_id':q,'seed_id':cids[srow],**rank_metrics(sorder,hidden,len(cids))});aud.append({'fold':fold,'trial_id':tid,'query_id':q,'seed_id':cids[srow],'hidden_positives':len(hidden),'union_size':len(union),'hidden_in_union':int(y.sum()),'clip_query_supported':bool(clip_q_support[qpos[q]])})
   print(f'prepare e2r seed fold={fold} {min(st+64,len(eligible))}/{len(eligible)}',flush=True)
 out=OUT/'prepared'/f'fold{fold}';out.mkdir(parents=True,exist_ok=True);np.savez(out/'cache.npz',X=np.concatenate(Xs),labels=np.concatenate(ys),candidate_rows=np.concatenate(rows_all),fallback_ranks=np.concatenate(fallback_all),query_ptr=np.asarray(ptr,dtype=np.int64),positive_rows=np.concatenate(pos_all),positive_fallback_ranks=np.concatenate(posfb_all),pos_ptr=np.asarray(pptr,dtype=np.int64),lexical_rank=lex)
 pd.DataFrame({'trial_id':trials}).to_csv(out/'trials.csv',index=False);pd.DataFrame(aud).to_csv(out/'audit.csv',index=False);pd.DataFrame(bm).to_csv(out/'base_query_metrics.csv',index=False);pd.DataFrame(sm).to_csv(out/'seed_query_metrics.csv',index=False);(out/'candidate_reactions.txt').write_text('\n'.join(cids)+'\n')
 summary={'fold':fold,'eligible_queries':len(eligible),'trials':len(trials),'candidate_count':len(cids),'mean_union_size':float(pd.DataFrame(aud).union_size.mean()),'queries_hidden_in_union_fraction':float((pd.DataFrame(aud).hidden_in_union>0).mean()),'seed_representation':'mean cosine across four fold-safe non-structural BiME reaction embedding experts','external_metrics_used':False};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

def load_cache(f):
 p=OUT/'prepared'/f'fold{f}';z=np.load(p/'cache.npz');return {'z':{k:z[k] for k in z.files},'trials':pd.read_csv(p/'trials.csv',dtype=str).trial_id.astype(str).tolist()}
def trainmat(caches):
 xs=[];ys=[];groups=[]
 for c in caches:
  z=c['z'];ptr=z['query_ptr']
  for i,t in enumerate(c['trials']):
   a,b=map(int,ptr[i:i+2]);X=z['X'][a:b];y=z['labels'][a:b];rows=z['candidate_rows'][a:b];pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0)
   if not len(pos):continue
   hardness=np.minimum(X[neg,0],X[neg,7]);hard=neg[np.lexsort((rows[neg],hardness))][:128];rem=np.setdiff1d(neg,hard,assume_unique=False);rng=np.random.default_rng(stable('e2r-seedctx|'+t));rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=np.int64);keep=np.concatenate([pos,hard,rnd]);xs.append(X[keep]);ys.append(y[keep].astype(np.float32));groups.append(len(keep))
 return np.concatenate(xs),np.concatenate(ys),groups
def train(caches,seed):
 X,y,g=trainmat(caches);d=xgb.DMatrix(X,label=y);d.set_group(g);p={'objective':'rank:ndcg','eval_metric':'ndcg@10','tree_method':'hist','device':'cuda' if torch.cuda.is_available() else 'cpu','max_depth':2,'eta':.12,'min_child_weight':5.,'lambda':1.,'subsample':.9,'colsample_bytree':.9,'lambdarank_pair_method':'topk','lambdarank_num_pair_per_sample':20,'seed':seed,'verbosity':0};return xgb.train(p,d,num_boost_round=80)
def evaluate(model,c,fold):
 z=c['z'];ptr=z['query_ptr'];pp=z['pos_ptr'];pred=model.predict(xgb.DMatrix(z['X']));lex=z['lexical_rank'];out=[]
 for i,t in enumerate(c['trials']):
  a,b=map(int,ptr[i:i+2]);rows=z['candidate_rows'][a:b];local=pred[a:b];fb=z['fallback_ranks'][a:b];take=np.lexsort((lex[rows],-local))[:min(PREFIX_K,len(rows))];sel=rows[take];selfb=fb[take];sp={int(r):j+1 for j,r in enumerate(sel)};pa,pb=map(int,pp[i:i+2]);prows=z['positive_rows'][pa:pb];pfb=z['positive_fallback_ranks'][pa:pb];ranks=[]
  for r,fr in zip(prows,pfb,strict=True):
   r=int(r);fr=int(fr);ranks.append(sp[r] if r in sp else len(sel)+fr-int(np.count_nonzero(selfb<fr)))
  q=t.split('|seed=',1)[0];sid=t.split('|seed=',1)[1].split('|rep=',1)[0];out.append({'fold':fold,'trial_id':t,'query_id':q,'seed_id':sid,**evaluate_full_candidate_ranks(np.asarray(ranks),len(lex)-1)})
 return pd.DataFrame(out)
def mm(d):
 return {'mrr':float(d.reciprocal_rank.mean()),'map':float(d.average_precision.mean()),'macro_roc_auc':float(d.roc_auc.mean()),'ndcg_at_10':float(d.ndcg_at_10.mean()),'hit_at_10':float(d.hit_at_10.mean()),'hit_at_20':float(d.hit_at_20.mean()),'hit_at_50':float(d.hit_at_50.mean()),'median_best_positive_rank':float(d.best_positive_rank.median())}
def crossfit():
 cs={f:load_cache(f) for f in FOLDS};frames=[]
 for h in FOLDS:
  m=train([cs[f] for f in FOLDS if f!=h],stable(f'e2r-seedctx-fold{h}'));frames.append(evaluate(m,cs[h],h));print('crossfit',h,flush=True)
 new=pd.concat(frames,ignore_index=True);new.to_csv(OUT/'development_oof_query_metrics.csv',index=False);base=pd.concat([pd.read_csv(OUT/'prepared'/f'fold{f}'/'base_query_metrics.csv') for f in FOLDS],ignore_index=True);seed=pd.concat([pd.read_csv(OUT/'prepared'/f'fold{f}'/'seed_query_metrics.csv') for f in FOLDS],ignore_index=True);nm,bm,sm=mm(new),mm(base),mm(seed);keys=['mrr','map','macro_roc_auc','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50'];delta={k:nm[k]-bm[k] for k in keys};folds={}
 for f in FOLDS:
  a=mm(new[new.fold.eq(f)]);b=mm(base[base.fold.eq(f)]);folds[str(f)]={k:a[k]-b[k] for k in keys}
 safe=all(v['mrr']>=-.001 and v['map']>=-.001 and v['hit_at_20']>=-.005 and v['hit_at_50']>=-.005 for v in folds.values());pooled=delta['mrr']>=-.001 and delta['map']>=-.001 and delta['hit_at_20']>=-.005 and delta['hit_at_50']>=-.005;material=delta['mrr']>=.003 or delta['map']>=.003 or delta['hit_at_20']>=.01 or delta['hit_at_50']>=.01;admit=bool(safe and pooled and material);res={'status':'admitted_internal' if admit else 'not_admitted','scenario':'one known positive reaction; seed masked; hidden reactions evaluated in 11081-reaction universe','trials':len(new),'eligible_queries':int(new.query_id.nunique()),'context_bime':nm,'zero_shot_bime':bm,'seed_only':sm,'delta_vs_zero_shot':delta,'fold_delta_vs_zero_shot':folds,'fold_safe':safe,'pooled_safe':pooled,'material':material,'admitted':admit,'external_metrics_used':False,'next_if_admitted':'fit final on all internal folds then frozen strict temporal retention'};OUT.mkdir(parents=True,exist_ok=True);(OUT/'development_result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2));return admit
def fit_final():
 r=json.loads((OUT/'development_result.json').read_text());assert r['admitted'];m=train([load_cache(f) for f in FOLDS],stable('e2r-seedctx-final'));o=OUT/'selected';o.mkdir(parents=True,exist_ok=True);m.save_model(o/'ranker.json');cfg={'feature_names':FEATURE_NAMES,'fixed_config':FIXED_CONFIG,'training_folds':list(FOLDS),'seed_reps':SEED_REPS,'seed_representation':'mean cosine across four fold-safe/non-structural BiME reaction embedding experts','external_metrics_used':False,'frozen_e2r_v4_ranker_sha256':sha(V4_BUNDLE/'ranker.json')};cfg['ranker_sha256']=sha(o/'ranker.json');(o/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(cfg,indent=2))
def main():
 import argparse;ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','crossfit','fit-final']);ap.add_argument('--fold',type=int,choices=FOLDS);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args()
 if a.stage=='prepare':assert a.fold is not None;prepare_fold(a.fold,a.device)
 elif a.stage=='crossfit':crossfit()
 else:fit_final()
if __name__=='__main__':main()
