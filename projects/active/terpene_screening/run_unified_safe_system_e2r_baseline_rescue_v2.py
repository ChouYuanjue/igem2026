from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,xgboost as xgb
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.evaluate_lambdarank_stacking_double_cold import train_ranker
from projects.active.terpene_screening.evaluate_unified_safe_system_e2r_full_reaction_v1 import load_bundle,encode_rows
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
ER=ROOT/'results/unified_safe_system_v1/e2r_baseline_rescue_v2_dev/experts'; OUT=ROOT/'results/unified_safe_system_v1/e2r_baseline_rescue_v2_dev/rescue'
NAMES=['enzgfm','esmc','equalblock','rdkitplus']
FEATURE_NAMES=[*(f'raw_{n}' for n in NAMES),*(f'z_{n}' for n in NAMES),*(f'top50_rank_{n}' for n in NAMES),'z_mean','z_std','z_max','top10_votes','top50_votes','enzgfm_gap_to_rank10','is_enzgfm_rank10']
assert len(FEATURE_NAMES)==19

def top_rows(v,k):
 idx=np.argpartition(-v,k-1)[:k]; return idx[np.argsort(-v[idx],kind='stable')]
def context(S):
 t10=[top_rows(S[e],10) for e in range(4)]; t50=[top_rows(S[e],50) for e in range(4)]; prefix=set(map(int,t10[0][:9])); base=int(t10[0][9]); slot={base}
 for e in range(1,4): slot.update(int(x) for x in t50[e] if int(x) not in prefix)
 return t10,t50,base,np.asarray(sorted(slot),dtype=np.int64)
def features(S,rows,t10,t50,base):
 z=(S-S.mean(1,keepdims=True))/np.maximum(S.std(1,keepdims=True),1e-6); raw=S[:,rows].T; zr=z[:,rows].T; blocks=[]
 s10=[set(map(int,a)) for a in t10]; s50=[set(map(int,a)) for a in t50]
 for a in t50:
  r=np.full(S.shape[1],51,dtype=np.float32); r[a]=np.arange(1,51,dtype=np.float32); blocks.append(r[rows,None])
 v10=np.asarray([sum(int(int(x) in s) for s in s10) for x in rows],dtype=np.float32)[:,None]; v50=np.asarray([sum(int(int(x) in s) for s in s50) for x in rows],dtype=np.float32)[:,None]
 return np.concatenate([raw,zr,*blocks,zr.mean(1,keepdims=True),zr.std(1,keepdims=True),zr.max(1,keepdims=True),v10,v50,(S[0,rows]-S[0,base])[:,None],(rows==base).astype(np.float32)[:,None]],axis=1).astype(np.float32)
def load_fold(f):
 dev=common=qids=None; emb={}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 for n in NAMES:
  d,_,m,pf,pids,rf,rids=load_bundle(ER/n,f,device); local=pd.read_csv(d/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates()
  if dev is None: dev=local
  else: assert dev.equals(local)
  ids=sorted(rids)
  if common is None: common=ids
  else: assert common==ids
  if qids is None: qids=sorted(dev.protein_id.unique())
  pi={x:i for i,x in enumerate(pids)}; ri={x:i for i,x in enumerate(rids)}; emb[n]=(encode_rows(m,'encode_proteins',pf,[pi[x] for x in qids],device),encode_rows(m,'encode_reactions',rf,[ri[x] for x in common],device))
 assert len(common)==11081; return dev,common,qids,emb
def batches(qids,emb,device,bs=64):
 rt={n:torch.from_numpy(emb[n][1]).to(device) for n in NAMES}
 with torch.no_grad():
  for st in range(0,len(qids),bs): yield st,{n:(torch.from_numpy(emb[n][0][st:st+bs]).to(device)@rt[n].T).cpu().numpy() for n in NAMES}
def prepare(f):
 dev,common,qids,emb=load_fold(f); pos=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); ri={r:i for i,r in enumerate(common)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); X=[];Y=[];G=[];A=[]
 for st,sc in batches(qids,emb,device):
  nloc=len(next(iter(sc.values())))
  for j,q in enumerate(qids[st:st+nloc]):
   S=np.stack([sc[n][j] for n in NAMES]); t10,t50,base,rows=context(S); p={ri[r] for r in pos[q]}; y=np.asarray([int(int(x) in p) for x in rows],dtype=np.float32); X.append(features(S,rows,t10,t50,base)); Y.append(y); G.append(len(rows)); A.append({'fold':f,'query_id':q,'slot_candidates':len(rows),'slot_positive_count':int(y.sum()),'baseline_rank10_positive':int(base in p)})
 out=OUT/'prepared'/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); np.save(out/'X.npy',np.concatenate(X)); np.save(out/'y.npy',np.concatenate(Y)); np.save(out/'groups.npy',np.asarray(G,dtype=np.int32)); pd.DataFrame(A).to_csv(out/'audit.csv',index=False); (out/'feature_names.json').write_text(json.dumps(FEATURE_NAMES)); print(json.dumps({'fold':f,'queries':len(G),'rows':int(sum(G)),'mean_slot_candidates':float(np.mean(G))},indent=2))
def train_holdout(f):
 X=[];Y=[];G=[]
 for g in [0,1,2]:
  if g==f: continue
  p=OUT/'prepared'/f'fold{g}'; X.append(np.load(p/'X.npy')); Y.append(np.load(p/'y.npy')); G.extend(np.load(p/'groups.npy').astype(int).tolist())
 return train_ranker(np.concatenate(X),np.concatenate(Y),G,'rank:pairwise',80,2,.05,5.,10.,8,20260901+f)
def transform(base_order,chosen,positives):
 inv=np.empty(len(base_order),dtype=np.int64); inv[base_order]=np.arange(1,len(base_order)+1); slot=int(base_order[9]); cr=int(inv[chosen]); out=[]
 for p in positives:
  br=int(inv[p])
  if chosen==slot or br<=9: nr=br
  elif p==chosen: nr=10
  elif 10<=br<cr: nr=br+1
  else: nr=br
  out.append(nr)
 return np.sort(np.asarray(out,dtype=np.int64))
def evaluate(f):
 model=train_holdout(f); dev,common,qids,emb=load_fold(f); pos=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); ri={r:i for i,r in enumerate(common)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rec=[]
 for st,sc in batches(qids,emb,device):
  nloc=len(next(iter(sc.values())))
  for j,q in enumerate(qids[st:st+nloc]):
   S=np.stack([sc[n][j] for n in NAMES]); t10,t50,base,rows=context(S); pred=model.predict(xgb.DMatrix(features(S,rows,t10,t50,base))); chosen=int(rows[np.lexsort((rows,-pred))[0]]); order=np.argsort(-S[0],kind='stable'); inv=np.empty(len(order),dtype=np.int64); inv[order]=np.arange(1,len(order)+1); p={ri[r] for r in pos[q]}; br=np.sort(np.asarray([inv[x] for x in p],dtype=np.int64)); rr=transform(order,chosen,p); bm=evaluate_full_candidate_ranks(br,len(common)); rm=evaluate_full_candidate_ranks(rr,len(common)); rec.append({'query_id':q,'slot_candidates':len(rows),'changed_slot':int(chosen!=base),'promoted_baseline_rank':int(inv[chosen]),**{f'base_{k}':v for k,v in bm.items()},**{f'rescue_{k}':v for k,v in rm.items()}})
 out=OUT/'oof'/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rec).to_csv(out/'query_metrics.csv',index=False); model.save_model(out/'slot_ranker.json'); print(json.dumps({'fold':f,'queries':len(rec),'changed_slot_fraction':float(np.mean([x['changed_slot'] for x in rec]))},indent=2))
def aggregate():
 q=pd.concat([pd.read_csv(OUT/'oof'/f'fold{f}/query_metrics.csv') for f in [0,1,2]],ignore_index=True); ms=['reciprocal_rank','average_precision','roc_auc','ndcg_at_10','hit_at_1','hit_at_3','hit_at_5','hit_at_10','hit_at_20','hit_at_50']; b={m:float(q[f'base_{m}'].mean()) for m in ms}; r={m:float(q[f'rescue_{m}'].mean()) for m in ms}; d={m:r[m]-b[m] for m in ms}; exact={f'hit{k}_exact':abs(d[f'hit_at_{k}'])<1e-15 for k in [1,3,5]}; checks={'mrr':d['reciprocal_rank']>=0,'map':d['average_precision']>=0,'auc':d['roc_auc']>=0,'ndcg10':d['ndcg_at_10']>=0,**exact,'hit20_floor':d['hit_at_20']>=-.005,'hit50_floor':d['hit_at_50']>=-.005}; passed=all(checks.values()) and d['hit_at_10']>=.05; result={'status':'passed_development_freeze_confirmation' if passed else 'rejected_v2_no_same_dev_retuning','baseline':b,'rescue':r,'delta':d,'checks':checks,'material_breakthrough_hit10':d['hit_at_10']>=.05,'mean_slot_candidates':float(q.slot_candidates.mean()),'changed_slot_fraction':float(q.changed_slot.mean()),'external_test_metrics_used':False,'same_dev_retuning_allowed':False}; (OUT/'development_result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','evaluate','aggregate']); ap.add_argument('--fold',type=int,choices=[0,1,2]); a=ap.parse_args()
 if a.stage=='prepare': assert a.fold is not None; prepare(a.fold)
 elif a.stage=='evaluate': assert a.fold is not None; evaluate(a.fold)
 else: aggregate()
if __name__=='__main__': main()
