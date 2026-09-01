from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, xgboost as xgb
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.evaluate_lambdarank_stacking_double_cold import train_ranker
from projects.active.terpene_screening.evaluate_unified_safe_system_e2r_full_reaction_v1 import load_bundle,encode_rows
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
EXPERTS={
 'enzgfm':ROOT/'results/enzgfm_gate_a_bidirectional_v1/enzgfm',
 'esmc':ROOT/'results/enzgfm_gate_a_bidirectional_v1/esmc',
 'equalblock':ROOT/'results/enzgfm_gate_a_bidirectional_v1/esmc_enzgfm_equalblock',
 'rdkitplus':ROOT/'results/enzgfm_gate_b_rdkitplus_v1/rdkitplus',
}
NAMES=list(EXPERTS); OUT=ROOT/'results/unified_safe_system_v1/e2r_lambdarank_stack_v1'; BASE_MET=ROOT/'results/unified_safe_system_v1/e2r_full_reaction_contract_v1'
FEATURE_NAMES=[*(f'raw_{n}' for n in NAMES),*(f'z_{n}' for n in NAMES),'z_mean','z_std','z_min','z_max','top10_votes','top50_votes']
def stable_seed(text:str)->int: return int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8).digest(),'big')%(2**32)
def feature_matrix(scores:np.ndarray,rows:np.ndarray,top10_sets:list[set[int]],top50_sets:list[set[int]])->np.ndarray:
 means=scores.mean(1,keepdims=True); std=scores.std(1,keepdims=True); std[std<1e-6]=1.; z=(scores-means)/std; zr=z[:,rows].T; raw=scores[:,rows].T
 votes10=np.asarray([sum(int(int(r) in s) for s in top10_sets) for r in rows],dtype=np.float32)[:,None]; votes50=np.asarray([sum(int(int(r) in s) for s in top50_sets) for r in rows],dtype=np.float32)[:,None]
 return np.concatenate([raw,zr,zr.mean(1,keepdims=True),zr.std(1,keepdims=True),zr.min(1,keepdims=True),zr.max(1,keepdims=True),votes10,votes50],axis=1).astype(np.float32)
def top_rows(values:np.ndarray,k:int)->np.ndarray:
 # Candidate IDs are globally lexical-sorted, so stable score sort gives exact lexical tie break.
 if k>=len(values): return np.argsort(-values,kind='stable')
 idx=np.argpartition(-values,k-1)[:k]; return idx[np.argsort(-values[idx],kind='stable')]
def load_fold_embeddings(f:int):
 dev=None; common=None; qids=None; emb={}
 for name,root in EXPERTS.items():
  d,s,m,pf,pids,rf,rids=load_bundle(root,f,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  local=pd.read_csv(d/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates()
  if dev is None: dev=local
  else: assert dev.equals(local)
  if common is None: common=sorted(rids)
  else: assert common==sorted(rids)
  if qids is None: qids=sorted(dev.protein_id.unique())
  pi={x:i for i,x in enumerate(pids)}; ri={x:i for i,x in enumerate(rids)}; device=next(m.parameters()).device
  pe=encode_rows(m,'encode_proteins',pf,[pi[x] for x in qids],device); re=encode_rows(m,'encode_reactions',rf,[ri[x] for x in common],device); emb[name]=(pe,re)
 assert dev is not None and common is not None and qids is not None and len(common)==11081
 return dev,common,qids,emb
def prepare_fold(f:int):
 dev,common,qids,emb=load_fold_embeddings(f); positives=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); ridx={r:i for i,r in enumerate(common)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rt={n:torch.from_numpy(emb[n][1]).to(device) for n in NAMES}; X=[];Y=[];groups=[];audit=[]
 batch=64
 with torch.no_grad():
  for st in range(0,len(qids),batch):
   scores={n:(torch.from_numpy(emb[n][0][st:st+batch]).to(device)@rt[n].T).cpu().numpy() for n in NAMES}
   for j,q in enumerate(qids[st:st+batch]):
    S=np.stack([scores[n][j] for n in NAMES]); t10=[set(map(int,top_rows(S[e],10))) for e in range(4)]; t50=[set(map(int,top_rows(S[e],50))) for e in range(4)]; pos=np.asarray(sorted({ridx[r] for r in positives[q]}),dtype=np.int64); mask=np.ones(len(common),dtype=bool); mask[pos]=False
    z=(S-S.mean(1,keepdims=True))/np.maximum(S.std(1,keepdims=True),1e-6); hardness=z.max(0); eligible=np.flatnonzero(mask); order=eligible[np.lexsort((eligible,-hardness[eligible]))]; hard=order[:64]; rem=np.setdiff1d(eligible,hard,assume_unique=False); rng=np.random.default_rng(stable_seed(f'{20260901}|{f}|{q}')); rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=np.int64); rows=np.concatenate([pos,hard,rnd]); y=np.concatenate([np.ones(len(pos),dtype=np.float32),np.zeros(len(hard)+len(rnd),dtype=np.float32)]); X.append(feature_matrix(S,rows,t10,t50)); Y.append(y); groups.append(len(rows)); audit.append({'fold':f,'query_id':q,'positives':len(pos),'hard_negatives':len(hard),'random_negatives':len(rnd),'group_size':len(rows)})
 out=OUT/'prepared'/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); np.save(out/'X.npy',np.concatenate(X)); np.save(out/'y.npy',np.concatenate(Y)); np.save(out/'groups.npy',np.asarray(groups,dtype=np.int32)); pd.DataFrame(audit).to_csv(out/'audit.csv',index=False); (out/'feature_names.json').write_text(json.dumps(FEATURE_NAMES)); print(json.dumps({'fold':f,'queries':len(groups),'rows':int(sum(groups)),'features':len(FEATURE_NAMES)},indent=2))
def train_for_holdout(f:int):
 xs=[];ys=[];groups=[]
 for g in range(3):
  if g==f: continue
  p=OUT/'prepared'/f'fold{g}'; xs.append(np.load(p/'X.npy')); ys.append(np.load(p/'y.npy')); groups.extend(np.load(p/'groups.npy').astype(int).tolist())
 model=train_ranker(np.concatenate(xs),np.concatenate(ys),groups,'rank:ndcg',80,3,.05,1.,10.,8,20260901+f); return model
def evaluate_holdout(f:int):
 booster=train_for_holdout(f); dev,common,qids,emb=load_fold_embeddings(f); positives=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); ridx={r:i for i,r in enumerate(common)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rt={n:torch.from_numpy(emb[n][1]).to(device) for n in NAMES}; rec=[]; batch=64
 with torch.no_grad():
  for st in range(0,len(qids),batch):
   scores={n:(torch.from_numpy(emb[n][0][st:st+batch]).to(device)@rt[n].T).cpu().numpy() for n in NAMES}
   for j,q in enumerate(qids[st:st+batch]):
    S=np.stack([scores[n][j] for n in NAMES]); top50=[top_rows(S[e],50) for e in range(4)]; top10=[set(map(int,top_rows(S[e],10))) for e in range(4)]; top50sets=[set(map(int,x)) for x in top50]; union=np.asarray(sorted(set().union(*top50sets)),dtype=np.int64); feat=feature_matrix(S,union,top10,top50sets); pred=booster.predict(xgb.DMatrix(feat)); order=union[np.lexsort((union,-pred))]
    # Full output = reranked union, then exact EnzGFM baseline order excluding union.
    b_order=np.argsort(-S[0],kind='stable'); b_inv=np.empty(len(common),dtype=np.int64); b_inv[b_order]=np.arange(1,len(common)+1); union_set=set(map(int,union)); stack_pos={int(r):i+1 for i,r in enumerate(order)}; pos_ranks=[]
    for rid in positives[q]:
     r=ridx[rid]
     if r in stack_pos: rank=stack_pos[r]
     else:
      br=int(b_inv[r]); before=sum(int(b_inv[u]<br) for u in union_set); rank=len(union_set)+br-before
     pos_ranks.append(rank)
    m=evaluate_full_candidate_ranks(np.asarray(pos_ranks,dtype=np.int64),len(common)); rec.append({'query_id':q,'shortlist_size':len(union),**m})
 q=pd.DataFrame(rec); out=OUT/'oof'/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); q.to_csv(out/'query_metrics.csv',index=False); booster.save_model(out/'ranker.json'); print(json.dumps({'fold':f,'queries':len(q),'mean_shortlist':q.shortlist_size.mean()},indent=2))
def aggregate():
 qs=[];bs=[]
 for f in range(3): qs.append(pd.read_csv(OUT/'oof'/f'fold{f}/query_metrics.csv')); bs.append(pd.read_csv(BASE_MET/f'fold{f}/baseline_query_metrics.csv'))
 q=pd.concat(qs,ignore_index=True); b=pd.concat(bs,ignore_index=True); assert q.query_id.tolist()==b.query_id.tolist()
 cols={'mrr':'reciprocal_rank','map':'average_precision','auc':'roc_auc','ndcg10':'ndcg_at_10','hit10':'hit_at_10','hit20':'hit_at_20','hit50':'hit_at_50'}; sm={k:float(q[c].mean()) for k,c in cols.items()}; bm={k:float(b[c].mean()) for k,c in cols.items()}; d={k:sm[k]-bm[k] for k in cols}; checks={'mrr':d['mrr']>=0,'map':d['map']>=0,'auc':d['auc']>=0,'ndcg10':d['ndcg10']>=0,'hit20_floor':d['hit20']>=-.005,'hit50_floor':d['hit50']>=-.005}; result={'status':'selected' if all(checks.values()) and d['hit10']>=.05 else 'pass_floor_no_material' if all(checks.values()) else 'reject_floor','baseline':bm,'stack':sm,'delta':d,'checks':checks,'material_breakthrough_hit10':d['hit10']>=.05,'mean_shortlist_size':float(q.shortlist_size.mean()),'external_test_metrics_used':False,'retuning_v1_allowed':False}; (OUT/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','evaluate','aggregate']); ap.add_argument('--fold',type=int,choices=[0,1,2]); a=ap.parse_args();
 if a.stage=='prepare': assert a.fold is not None; prepare_fold(a.fold)
 elif a.stage=='evaluate': assert a.fold is not None; evaluate_holdout(a.fold)
 else: aggregate()
if __name__=='__main__': main()
