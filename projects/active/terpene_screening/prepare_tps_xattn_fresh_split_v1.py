from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[3]
PAIRS=ROOT/'data/terpene_marts_adaptation/marts_pair_folds.csv'
PROTEINS=ROOT/'data/terpene_marts_adaptation/protein_entities.csv'
REACTIONS=ROOT/'data/terpene_marts_adaptation/reaction_entities.csv'
ESMC=ROOT/'data/terpene_tps_foundation_v1/esmc'
ENZGFM=ROOT/'data/terpene_tps_foundation_v1/enzgfm'
RFEAT=ROOT/'data/terpene_tps_foundation_v1/reaction_2115'
RMETA=ROOT/'data/terpene_cold_splits/reaction_cluster_folds.csv'
OUT=ROOT/'results/tps_active_site_xattn_v1/fresh_split'
SALT='tps_active_site_xattn_v1_dev_20260902'

def stable_fold(cluster:str)->int: return int.from_bytes(hashlib.blake2b((SALT+'::'+str(cluster)).encode(),digest_size=8).digest(),'big')%5

def load_matrix(d:Path,idcol:str)->tuple[np.ndarray,list[str]]:
 e=pd.read_csv(d/'entries.csv',dtype=str).fillna(''); e['row']=pd.to_numeric(e.row).astype(int); candidates=[d/'embeddings.npy',d/'reaction_feature_matrix.npy',d/'features.npy']; matrix=next((q for q in candidates if q.exists()),None);
 if matrix is None: raise FileNotFoundError(f'no supported feature matrix under {d}');
 x=np.load(matrix).astype(np.float32); e=e.sort_values('row'); return x[e.row.to_numpy()],e[idcol].astype(str).tolist()

def prepare_splits(out:Path)->dict:
 out.mkdir(parents=True,exist_ok=True); p=pd.read_csv(PAIRS,dtype=str).fillna('')[['Entry','rhea_id','protein_cluster']].drop_duplicates(); p['fresh_fold']=p.protein_cluster.map(stable_fold)
 audits=[]
 for f in range(5):
  train=p[p.fresh_fold!=f][['Entry','rhea_id','protein_cluster']].copy(); dev=p[p.fresh_fold==f][['Entry','rhea_id','protein_cluster']].copy(); train.to_csv(out/f'fold{f}_train_pairs.csv',index=False); dev.to_csv(out/f'fold{f}_dev_pairs.csv',index=False)
  ov=set(train.protein_cluster)&set(dev.protein_cluster)
  q=[]
  for rid,g in dev.groupby('rhea_id'):
   known=set(train.loc[train.rhea_id.eq(rid),'Entry']); held=set(g.Entry); q.append({'query_id':str(rid),'heldout_positive_count':len(held),'train_known_positive_count':len(known)})
  pd.DataFrame(q).to_csv(out/f'fold{f}_query_support.csv',index=False)
  audits.append({'fold':f,'train_pairs':len(train),'dev_pairs':len(dev),'train_clusters':train.protein_cluster.nunique(),'dev_clusters':dev.protein_cluster.nunique(),'cluster_overlap':len(ov),'query_count':len(q),'heldout_positive_rows':len(dev),'queries_with_train_known_positives':sum(x['train_known_positive_count']>0 for x in q)})
 pd.DataFrame(audits).to_csv(out/'split_audit.csv',index=False)
 result={'status':'ready_no_model_scores','salt':SALT,'folds':5,'hpo_folds':[0,1,2],'confirmation_folds':[3,4],'all_cluster_overlap_zero':all(x['cluster_overlap']==0 for x in audits),'support':audits,'performance_metrics_materialized':False}
 (out/'split_manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def normalize(x):
 n=np.linalg.norm(x,axis=1,keepdims=True); return x/np.maximum(n,1e-12)

def binary_tanimoto_matrix(x:np.ndarray)->np.ndarray:
 b=(x[:,:2048]>0).astype(np.float32); inter=b@b.T; sums=b.sum(1); union=sums[:,None]+sums[None,:]-inter; return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def grouped_transfer_scores(protein_cos:np.ndarray, reaction_sim:np.ndarray, train:pd.DataFrame, query_rows:list[int], pidx:dict[str,int], ridx:dict[str,int], chunk:int=16)->np.ndarray:
 groups=[]; train_reactions=[]
 for rid,g in train.groupby('rhea_id',sort=True):
  if rid not in ridx: continue
  rows=[pidx[x] for x in g.Entry.astype(str) if x in pidx]
  if not rows: continue
  groups.append(rows); train_reactions.append(ridx[str(rid)])
 # exact max cosine(candidate, train positives) for each train reaction
 best=np.stack([np.maximum(protein_cos[:,rows],0).max(axis=1) for rows in groups],axis=0).astype(np.float32) # [Rtrain,C]
 result=np.empty((len(query_rows),protein_cos.shape[0]),dtype=np.float32)
 for start in range(0,len(query_rows),chunk):
  q=query_rows[start:start+chunk]; weights=reaction_sim[np.ix_(q,train_reactions)] # [Q,Rtrain]
  result[start:start+len(q)]=(weights[:,:,None]*best[None,:,:]).max(axis=1)
 return result

def top_indices(score:np.ndarray,k:int,ids:list[str])->np.ndarray:
 lexical=np.argsort(np.asarray(ids,dtype=object),kind='stable'); # deterministic id rank
 return np.lexsort((lexical,-score))[:k]

def materialize_shortlists(out:Path)->dict:
 if not (out/'split_manifest.json').exists(): prepare_splits(out)
 ex,eids=load_matrix(ESMC,'Entry'); zx,zids=load_matrix(ENZGFM,'Entry');
 if eids!=zids: raise RuntimeError('protein feature entry orders differ')
 rx,rids=load_matrix(RFEAT,'reaction_id'); pidx={x:i for i,x in enumerate(eids)}; ridx={x:i for i,x in enumerate(rids)}
 ecos=normalize(ex)@normalize(ex).T; zcos=normalize(zx)@normalize(zx).T; rsim=binary_tanimoto_matrix(rx)
 audits=[]
 for f in range(5):
  train=pd.read_csv(out/f'fold{f}_train_pairs.csv',dtype=str).fillna(''); dev=pd.read_csv(out/f'fold{f}_dev_pairs.csv',dtype=str).fillna(''); queries=sorted(dev.rhea_id.unique()); qrows=[ridx[q] for q in queries]
  es=grouped_transfer_scores(ecos,rsim,train,qrows,pidx,ridx); zs=grouped_transfer_scores(zcos,rsim,train,qrows,pidx,ridx); base=(es+zs)/2.0
  rows=[]; candidate_counts=[]; known_counts=[]
  for qi,q in enumerate(queries):
   known=set(train.loc[train.rhea_id.eq(q),'Entry'].astype(str)); known_rows=[pidx[x] for x in known if x in pidx]
   for a in (es,zs,base): a[qi,known_rows]=-np.inf
   ie=top_indices(es[qi],160,eids); iz=top_indices(zs[qi],160,eids); union=set(map(int,ie))|set(map(int,iz)); ordered=sorted(union,key=lambda j:(-float(base[qi,j]),eids[j]))
   candidate_counts.append(len(ordered)); known_counts.append(len(known_rows))
   se=set(map(int,ie)); sz=set(map(int,iz))
   for rank,j in enumerate(ordered,1): rows.append({'query_id':q,'candidate_id':eids[j],'shortlist_rank':rank,'baseline_score':float(base[qi,j]),'esmc_transfer_score':float(es[qi,j]),'enzgfm_transfer_score':float(zs[qi,j]),'in_esmc_top160':j in se,'in_enzgfm_top160':j in sz})
  pd.DataFrame(rows).to_csv(out/f'fold{f}_shortlist.csv',index=False)
  np.savez_compressed(out/f'fold{f}_transfer_scores.npz',query_ids=np.asarray(queries,dtype='U64'),protein_ids=np.asarray(eids,dtype='U64'),baseline=base,esmc=es,enzgfm=zs)
  audits.append({'fold':f,'query_count':len(queries),'mean_shortlist_size':float(np.mean(candidate_counts)),'min_shortlist_size':int(min(candidate_counts)),'max_shortlist_size':int(max(candidate_counts)),'mean_masked_train_known_positives':float(np.mean(known_counts)),'heldout_labels_used_for_membership':False,'performance_metrics_materialized':False})
 pd.DataFrame(audits).to_csv(out/'shortlist_audit.csv',index=False)
 result={'status':'ready_without_heldout_performance_evaluation','per_representation_topk':160,'baseline_fusion':'equal_mean','transfer_definition':'exact max over train association pairs of max(cos,0)*binary-DRFP-Tanimoto via equivalent reaction-grouped max','folds':audits,'heldout_labels_used_for_membership':False,'performance_metrics_materialized':False}
 (out/'shortlist_manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def reaction_skeleton_map()->dict[str,str]:
 from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature
 a=pd.read_csv(REACTIONS,dtype=str).fillna(''); b=pd.read_csv(RMETA,dtype=str).fillna(''); b['sig']=b.reaction_smiles.map(reaction_signature)
 by={}
 for sig,g in b.groupby('sig'):
  sk=sorted(set(g.product_skeleton_class.astype(str))-{''})
  if len(sk)==1: by[str(sig)]=sk[0]
 return {str(r.reaction_id):by.get(str(r.reaction_signature),'') for r in a.itertuples(index=False)}

def materialize_hard_pools(out:Path)->dict:
 if not (out/'shortlist_manifest.json').exists(): materialize_shortlists(out)
 ex,eids=load_matrix(ESMC,'Entry'); zx,zids=load_matrix(ENZGFM,'Entry'); rx,rids=load_matrix(RFEAT,'reaction_id')
 if eids!=zids: raise RuntimeError('protein feature entry orders differ')
 pidx={x:i for i,x in enumerate(eids)}; ridx={x:i for i,x in enumerate(rids)}; ecos=normalize(ex)@normalize(ex).T; zcos=normalize(zx)@normalize(zx).T; rsim=binary_tanimoto_matrix(rx); skel=reaction_skeleton_map(); audits=[]
 for f in range(5):
  train=pd.read_csv(out/f'fold{f}_train_pairs.csv',dtype=str).fillna(''); train_proteins=sorted(set(train.Entry)); train_rows=set(pidx[x] for x in train_proteins); queries=sorted(train.rhea_id.unique()); qrows=[ridx[q] for q in queries]
  es=grouped_transfer_scores(ecos,rsim,train,qrows,pidx,ridx); zs=grouped_transfer_scores(zcos,rsim,train,qrows,pidx,ridx); base=(es+zs)/2.0
  cand_skeletons={pid:set(skel.get(r,'') for r in g.rhea_id.astype(str))- {''} for pid,g in train.groupby('Entry')}
  rows=[]; min_pool=10**9; preferred_queries=0
  for qi,q in enumerate(queries):
   known=set(train.loc[train.rhea_id.eq(q),'Entry'].astype(str)); qr=skel.get(q,''); candidates=[]
   for j,pid in enumerate(eids):
    if j not in train_rows or pid in known: continue
    pref=bool(qr and cand_skeletons.get(pid) and qr not in cand_skeletons[pid]); candidates.append((0 if pref else 1,-float(base[qi,j]),pid,j,pref))
   candidates.sort(); pool=candidates[:32]; min_pool=min(min_pool,len(pool)); preferred_queries+=int(any(x[4] for x in pool))
   for rank,x in enumerate(pool,1): rows.append({'query_id':q,'candidate_id':x[2],'pool_rank':rank,'baseline_score':-x[1],'preferred_different_known_skeleton':x[4]})
  pd.DataFrame(rows).to_csv(out/f'fold{f}_hard_pool_max32.csv',index=False); audits.append({'fold':f,'train_query_count':len(queries),'min_pool_size':int(min_pool),'queries_with_preferred_skeleton_negative':int(preferred_queries),'reaction_skeleton_metadata_coverage':int(sum(bool(skel.get(q,'')) for q in queries))})
 result={'status':'fixed_train_only_hard_pools_ready','max_pool_size':32,'folds':audits,'heldout_association_labels_used':False}; (out/'hard_pool_manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['splits','shortlists','hard-pools','all']); ap.add_argument('--output-dir',type=Path,default=OUT); a=ap.parse_args(); o=a.output_dir.resolve()
 if a.action in {'splits','all'}: print(json.dumps(prepare_splits(o),indent=2))
 if a.action in {'shortlists','all'}: print(json.dumps(materialize_shortlists(o),indent=2))
 if a.action in {'hard-pools','all'}: print(json.dumps(materialize_hard_pools(o),indent=2))
if __name__=='__main__': main()
