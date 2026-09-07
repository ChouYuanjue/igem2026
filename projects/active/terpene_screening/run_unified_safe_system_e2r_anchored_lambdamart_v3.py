from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,xgboost as xgb
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.evaluate_unified_safe_system_e2r_full_reaction_v1 import load_bundle,encode_rows
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks

PROTOCOL=ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3.json'
ER=ROOT/'results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_dev/experts'
OUT=ROOT/'results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_dev/anchored'
NAMES=['enzgfm','esmc','equalblock','rdkitplus']; N_CANDIDATES=11081; MAX_POOL=100
FEATURE_NAMES=[*(f'raw_{n}' for n in NAMES),*(f'z_{n}' for n in NAMES),*(f'logrank_{n}' for n in NAMES),*(f'rr_{n}' for n in NAMES),'z_mean','z_std','z_min','z_max','top10_votes','top50_votes','top100_votes','baseline_anchor','best_other_z_minus_base','rdkit_z_minus_base']
assert len(FEATURE_NAMES)==26
BASELINE_ANCHOR_INDEX=FEATURE_NAMES.index('baseline_anchor')
METRICS={'mrr':'reciprocal_rank','map':'average_precision','auc':'roc_auc','ndcg10':'ndcg_at_10','hit10':'hit_at_10','hit20':'hit_at_20','hit50':'hit_at_50'}

def protocol(): return json.loads(PROTOCOL.read_text())
def stable_seed(text:str)->int: return int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8).digest(),'big')%(2**32)
def top_rows(v:np.ndarray,k:int)->np.ndarray:
 if k>=len(v): return np.argsort(-v,kind='stable')
 idx=np.argpartition(-v,k-1)[:k]; return idx[np.argsort(-v[idx],kind='stable')]
def full_ranks(v:np.ndarray)->np.ndarray:
 order=np.argsort(-v,kind='stable'); inv=np.empty(len(v),dtype=np.int32); inv[order]=np.arange(1,len(v)+1,dtype=np.int32); return inv

def feature_matrix(S:np.ndarray,rows:np.ndarray,ranks:np.ndarray)->np.ndarray:
 means=S.mean(1,keepdims=True); std=np.maximum(S.std(1,keepdims=True),1e-6); z=(S-means)/std
 raw=S[:,rows].T.astype(np.float32); zr=z[:,rows].T.astype(np.float32); rr=ranks[rows].astype(np.float32)
 logrank=np.log1p(rr)/math.log1p(S.shape[1]); recip=1.0/rr
 v10=(rr<=10).sum(1,dtype=np.int16).astype(np.float32)[:,None]; v50=(rr<=50).sum(1,dtype=np.int16).astype(np.float32)[:,None]; v100=(rr<=100).sum(1,dtype=np.int16).astype(np.float32)[:,None]
 anchor=(1.0-logrank[:,0:1]).astype(np.float32); other_minus=(zr[:,1:].max(1)-zr[:,0])[:,None]; rdkit_minus=(zr[:,3]-zr[:,0])[:,None]
 return np.concatenate([raw,zr,logrank,recip,zr.mean(1,keepdims=True),zr.std(1,keepdims=True),zr.min(1,keepdims=True),zr.max(1,keepdims=True),v10,v50,v100,anchor,other_minus,rdkit_minus],axis=1).astype(np.float32,copy=False)

def load_fold_embeddings(fold:int):
 dev=common=qids=None; emb={}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 for name in NAMES:
  d,_,m,pf,pids,rf,rids=load_bundle(ER/name,fold,device); local=pd.read_csv(d/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates().sort_values(['protein_id','reaction_id']).reset_index(drop=True)
  if dev is None: dev=local
  else: assert dev.equals(local),f'dev pairs differ for {name} fold{fold}'
  ids=sorted(map(str,rids));
  if common is None: common=ids
  else: assert common==ids
  if qids is None: qids=sorted(dev.protein_id.unique())
  pi={x:i for i,x in enumerate(pids)}; ri={x:i for i,x in enumerate(rids)}
  emb[name]=(encode_rows(m,'encode_proteins',pf,[pi[x] for x in qids],device),encode_rows(m,'encode_reactions',rf,[ri[x] for x in common],device))
 assert dev is not None and common is not None and qids is not None and len(common)==N_CANDIDATES
 return dev,common,qids,emb

def prepare_fold(fold:int):
 dev,common,qids,emb=load_fold_embeddings(fold); positives=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); ridx={r:i for i,r in enumerate(common)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rt={n:torch.from_numpy(emb[n][1]).to(device) for n in NAMES}
 X=[]; rows_all=[]; ranks_all=[]; labels=[]; offsets=[0]; pos_rows=[]; pos_base_ranks=[]; pos_offsets=[0]; baseline=[]; audits=[]
 with torch.no_grad():
  for st in range(0,len(qids),64):
   sc={n:(torch.from_numpy(emb[n][0][st:st+64]).to(device)@rt[n].T).cpu().numpy() for n in NAMES}; nloc=len(next(iter(sc.values())))
   for j,q in enumerate(qids[st:st+nloc]):
    S=np.stack([sc[n][j] for n in NAMES]).astype(np.float32); ranks_full=np.stack([full_ranks(S[e]) for e in range(4)],axis=1); union=np.asarray(sorted(set().union(*(set(map(int,top_rows(S[e],MAX_POOL))) for e in range(4)))),dtype=np.int32); uranks=ranks_full[union].astype(np.int16); feat=feature_matrix(S,union,ranks_full)
    p=np.asarray(sorted({ridx[r] for r in positives[q]}),dtype=np.int32); pset=set(map(int,p)); y=np.asarray([int(int(r) in pset) for r in union],dtype=np.uint8); br=ranks_full[p,0].astype(np.int32)
    X.append(feat); rows_all.append(union); ranks_all.append(uranks); labels.append(y); offsets.append(offsets[-1]+len(union)); pos_rows.append(p); pos_base_ranks.append(br); pos_offsets.append(pos_offsets[-1]+len(p)); baseline.append({'query_id':q,**evaluate_full_candidate_ranks(br,N_CANDIDATES)}); audits.append({'fold':fold,'query_id':q,'union_size':len(union),'positive_count':len(p),'positive_in_union':int(y.sum()),'baseline_best_rank':int(br.min())})
 out=OUT/'prepared'/f'fold{fold}'; out.mkdir(parents=True,exist_ok=True); np.save(out/'X.npy',np.concatenate(X)); np.save(out/'rows.npy',np.concatenate(rows_all)); np.save(out/'ranks.npy',np.concatenate(ranks_all)); np.save(out/'labels.npy',np.concatenate(labels)); np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.int64)); np.save(out/'positive_rows.npy',np.concatenate(pos_rows)); np.save(out/'positive_base_ranks.npy',np.concatenate(pos_base_ranks)); np.save(out/'positive_offsets.npy',np.asarray(pos_offsets,dtype=np.int64)); pd.DataFrame({'query_id':qids}).to_csv(out/'queries.csv',index=False); pd.DataFrame(baseline).to_csv(out/'baseline_query_metrics.csv',index=False); pd.DataFrame(audits).to_csv(out/'audit.csv',index=False); (out/'candidate_reactions.txt').write_text('\n'.join(common)+'\n'); (out/'feature_names.json').write_text(json.dumps(FEATURE_NAMES,indent=2)+'\n')
 summary={'fold':fold,'queries':len(qids),'candidate_count':len(common),'rows':int(offsets[-1]),'mean_union_size':float(np.mean([a['union_size'] for a in audits])),'query_positive_in_union_fraction':float(np.mean([a['positive_in_union']>0 for a in audits])),'dev_pairs':len(dev),'split_salt':protocol()['development_split']['split_salt'],'external_metrics_used':False}; (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)

def load_cache(fold:int):
 p=OUT/'prepared'/f'fold{fold}'; return {'X':np.load(p/'X.npy',mmap_mode='r'),'rows':np.load(p/'rows.npy',mmap_mode='r'),'ranks':np.load(p/'ranks.npy',mmap_mode='r'),'labels':np.load(p/'labels.npy',mmap_mode='r'),'offsets':np.load(p/'offsets.npy'),'pos_rows':np.load(p/'positive_rows.npy',mmap_mode='r'),'pos_base_ranks':np.load(p/'positive_base_ranks.npy',mmap_mode='r'),'pos_offsets':np.load(p/'positive_offsets.npy'),'queries':pd.read_csv(p/'queries.csv',dtype=str).query_id.astype(str).tolist(),'baseline':pd.read_csv(p/'baseline_query_metrics.csv')}

def sampled_training(cache:dict,fold:int):
 xs=[];ys=[];groups=[]
 for qi,q in enumerate(cache['queries']):
  a,b=map(int,cache['offsets'][qi:qi+2]); y=np.asarray(cache['labels'][a:b],dtype=np.uint8); pos=np.flatnonzero(y>0)
  if not len(pos): continue
  neg=np.flatnonzero(y==0); X=np.asarray(cache['X'][a:b],dtype=np.float32); hard=neg[np.lexsort((np.asarray(cache['rows'][a:b])[neg],-X[neg,FEATURE_NAMES.index('z_max')]))][:96]; rem=np.setdiff1d(neg,hard,assume_unique=False); rng=np.random.default_rng(stable_seed(f'e2r-v3|{fold}|{q}')); rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=np.int64); keep=np.concatenate([pos,hard,rnd]); xs.append(X[keep]); ys.append(y[keep].astype(np.float32)); groups.append(len(keep))
 return np.concatenate(xs),np.concatenate(ys),groups

def ranker_configs(): return protocol()['ranker']['ranker_configs']
def structure_configs():
 spec=protocol()['anchored_runtime_family']; out=[]
 for K in spec['expert_pool_topk_candidates']:
  for P in spec['learned_prefix_candidates']:
   if P>K: continue
   for A in spec['protected_prefix_candidates']:
    if A<P: out.append({'protected_prefix':int(A),'pool_k':int(K),'prefix_k':int(P)})
 assert len(out)==32; return out

def train_model(train_X,train_y,groups,cfg,seed:int):
 d=xgb.DMatrix(train_X,label=train_y); d.set_group(groups); mono=[0]*len(FEATURE_NAMES); mono[BASELINE_ANCHOR_INDEX]=1
 params={'objective':cfg['objective'],'tree_method':'hist','device':'cuda' if torch.cuda.is_available() else 'cpu','max_depth':int(cfg['max_depth']),'eta':float(cfg['learning_rate']),'min_child_weight':5.0,'lambda':5.0,'subsample':0.9,'colsample_bytree':0.9,'lambdarank_pair_method':'topk','lambdarank_num_pair_per_sample':int(cfg['lambdarank_pairs']),'monotone_constraints':'('+','.join(map(str,mono))+')','seed':int(seed),'nthread':16,'verbosity':0}; return xgb.train(params,d,num_boost_round=int(cfg['rounds']))

def candidate_query_metrics(cache:dict,pred:np.ndarray,structure:dict)->pd.DataFrame:
 A,K,P=structure['protected_prefix'],structure['pool_k'],structure['prefix_k']; rec=[]
 for qi,q in enumerate(cache['queries']):
  a,b=map(int,cache['offsets'][qi:qi+2]); rows=np.asarray(cache['rows'][a:b],dtype=np.int32); ranks=np.asarray(cache['ranks'][a:b],dtype=np.int32); local_pred=pred[a:b]; pool=(ranks.min(1)<=K) & (ranks[:,0]>A); cand=np.flatnonzero(pool); order=cand[np.lexsort((rows[cand],-local_pred[cand]))]; L=min(P-A,len(order)); selected_local=order[:L]; selected_rows=rows[selected_local]; selected_base=ranks[selected_local,0]; selected_pos={int(r):A+i+1 for i,r in enumerate(selected_rows)}
  pa,pb=map(int,cache['pos_offsets'][qi:qi+2]); prows=np.asarray(cache['pos_rows'][pa:pb],dtype=np.int32); pbase=np.asarray(cache['pos_base_ranks'][pa:pb],dtype=np.int32); new=[]
  for r,br in zip(prows,pbase,strict=True):
   r=int(r); br=int(br)
   if br<=A: nr=br
   elif r in selected_pos: nr=selected_pos[r]
   else: nr=br+L-int(np.count_nonzero(selected_base<br))
   new.append(nr)
  rec.append({'query_id':q,'selected_prefix_size':L,'pool_candidates':int(pool.sum()),**evaluate_full_candidate_ranks(np.asarray(new,dtype=np.int64),N_CANDIDATES)})
 return pd.DataFrame(rec)

def metric_map(frame:pd.DataFrame): return {k:float(frame[c].mean()) for k,c in METRICS.items()}
def gm_ratio(c,b): return float(math.exp(np.mean([math.log(max(c[k],1e-12)/max(b[k],1e-12)) for k in ['mrr','map','ndcg10','hit10']])))

def run_search():
 caches={f:load_cache(f) for f in [0,1,2]}; sampled={f:sampled_training(caches[f],f) for f in [0,1,2]}; models=OUT/'search_models'; models.mkdir(parents=True,exist_ok=True); rows=[]
 for ci,cfg in enumerate(ranker_configs()):
  preds={}
  for hold in [0,1,2]:
   xs=[];ys=[];groups=[]
   for f in [0,1,2]:
    if f==hold: continue
    X,y,g=sampled[f]; xs.append(X); ys.append(y); groups.extend(g)
   booster=train_model(np.concatenate(xs),np.concatenate(ys),groups,cfg,20260902+ci*10+hold); d=xgb.DMatrix(np.asarray(caches[hold]['X'],dtype=np.float32)); preds[hold]=booster.predict(d); md=models/cfg['id']; md.mkdir(parents=True,exist_ok=True); booster.save_model(md/f'fold{hold}.json')
  for structure in structure_configs():
   fold_deltas={}; cand_frames=[]; base_frames=[]
   for f in [0,1,2]:
    cand=candidate_query_metrics(caches[f],preds[f],structure); base=caches[f]['baseline']; assert cand.query_id.tolist()==base.query_id.tolist(); cm=metric_map(cand); bm=metric_map(base); fold_deltas[str(f)]={k:cm[k]-bm[k] for k in METRICS}; cand_frames.append(cand); base_frames.append(base)
   C=pd.concat(cand_frames,ignore_index=True); B=pd.concat(base_frames,ignore_index=True); cm=metric_map(C); bm=metric_map(B); delta={k:cm[k]-bm[k] for k in METRICS}; pooled=all(delta[k]>=-1e-12 for k in METRICS); safe=all(fd['mrr']>=-.001 and fd['map']>=-.001 and fd['auc']>=-.001 and fd['ndcg10']>=-.002 and fd['hit10']>=-.005 and fd['hit20']>=-.005 and fd['hit50']>=-.005 for fd in fold_deltas.values()); two=all(sum(fd[k]>0 for fd in fold_deltas.values())>=2 for k in ['mrr','map','hit10']); material=delta['mrr']>=.003 or delta['map']>=.003 or delta['hit10']>=.01; feasible=pooled and safe and two and material; rows.append({'ranker_id':cfg['id'],'ranker_max_depth':int(cfg['max_depth']),'ranker_rounds':int(cfg['rounds']),**structure,**{f'baseline_{k}':v for k,v in bm.items()},**{f'candidate_{k}':v for k,v in cm.items()},**{f'delta_{k}':v for k,v in delta.items()},'primary_geomean_ratio':gm_ratio(cm,bm),'pooled_no_regression':pooled,'fold_safe':safe,'two_of_three_positive':two,'material_gate':material,'feasible':feasible,'fold_deltas_json':json.dumps(fold_deltas,sort_keys=True)})
  print('searched ranker',cfg['id'],flush=True)
 frame=pd.DataFrame(rows); frame.to_csv(OUT/'search_results.csv',index=False); feasible=frame[frame.feasible.astype(bool)].copy()
 if feasible.empty:
  result={'status':'rejected_no_feasible_material_config','configuration_count':len(frame),'feasible_count':0,'external_metrics_used':False,'same_dev_retuning_allowed':False}; (OUT/'selection_result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return
 feasible=feasible.sort_values(['primary_geomean_ratio','delta_hit20','delta_hit50','protected_prefix','prefix_k','pool_k','ranker_max_depth','ranker_rounds','ranker_id'],ascending=[False,False,False,False,True,True,True,True,True],kind='stable'); best=feasible.iloc[0].to_dict(); selected={'ranker_id':best['ranker_id'],'protected_prefix':int(best['protected_prefix']),'pool_k':int(best['pool_k']),'prefix_k':int(best['prefix_k'])}; cfg=next(x for x in ranker_configs() if x['id']==selected['ranker_id']); selected['ranker_config']=cfg
 # Save selected per-query OOF metrics from frozen cross-fit models.
 oof=[]
 for f in [0,1,2]:
  booster=xgb.Booster(); booster.load_model(models/selected['ranker_id']/f'fold{f}.json'); pred=booster.predict(xgb.DMatrix(np.asarray(caches[f]['X'],dtype=np.float32))); q=candidate_query_metrics(caches[f],pred,selected); q['fold']=f; oof.append(q)
 pd.concat(oof,ignore_index=True).to_csv(OUT/'selected_oof_query_metrics.csv',index=False)
 result={'status':'selected_development_pass_confirmation_authorized','configuration_count':len(frame),'feasible_count':len(feasible),'selected_config':selected,'selected_summary':{k:(bool(v) if isinstance(v,(np.bool_,bool)) else float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for k,v in best.items() if k!='fold_deltas_json'},'fold_deltas':json.loads(best['fold_deltas_json']),'external_metrics_used':False,'same_dev_retuning_allowed':False,'confirmation':protocol()['future_confirmation_split']}; (OUT/'selection_result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','search']); ap.add_argument('--fold',type=int,choices=[0,1,2]); a=ap.parse_args()
 if a.stage=='prepare': assert a.fold is not None; prepare_fold(a.fold)
 else: run_search()
if __name__=='__main__': main()
