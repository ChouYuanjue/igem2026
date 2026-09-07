from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, torch
from scipy import sparse
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import load_protein_library, load_registered_reaction_feature_library, load_feature_schema

DEV=ROOT/'results/comprehensive_enzgfm_center_top1_v1/dev'
PROT=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'
RXN=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'
BASE_OOF=ROOT/'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_oof_query_metrics.csv'
OUT=ROOT/'results/bime_rank_unified_v1/r2e_homology_admission_v1'
FOLDS=(0,1,2);TOP_RXN=5;TOP_PROT=100
CELL=lambda f:f'clean2023_internal_double_cold_salted_comprehensive_enzgfm_center_top1_v1_dev_20260901_fold{f}'

def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def top_reaction_neighbors(qids:list[str],train_ids:list[str],rxn_ids:list[str],features:np.ndarray,drfp_dim:int):
 idx={r:i for i,r in enumerate(rxn_ids)}; qrows=np.asarray([idx[q] for q in qids],dtype=np.int64); trows=np.asarray([idx[r] for r in train_ids],dtype=np.int64)
 q=sparse.csr_matrix((np.asarray(features[qrows,:drfp_dim])>0).astype(np.float32));t=sparse.csr_matrix((np.asarray(features[trows,:drfp_dim])>0).astype(np.float32))
 inter=(q@t.T).toarray().astype(np.float32);qc=np.asarray(q.sum(1)).ravel();tc=np.asarray(t.sum(1)).ravel();union=qc[:,None]+tc[None,:]-inter;sim=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
 out={}
 tids=np.asarray(train_ids,dtype=object)
 for i,qid in enumerate(qids):
  # deterministic similarity desc, reaction ID asc
  order=np.lexsort((tids,-sim[i]))[:TOP_RXN];out[qid]=[(str(tids[j]),float(sim[i,j])) for j in order]
 return out

def exact_seed_top100(candidate:torch.Tensor, seed_rows:list[int],cache_path:Path,batch:int=96):
 if cache_path.is_file():
  z=np.load(cache_path);rows=z['seed_rows'].astype(np.int64);inds=z['indices'].astype(np.int32);vals=z['values'].astype(np.float32);assert rows.tolist()==list(seed_rows);print('reuse seed-neighbor cache',cache_path,flush=True);return {int(r):(inds[i],vals[i]) for i,r in enumerate(rows)}
 res={}
 all_i=[];all_v=[]
 for st in range(0,len(seed_rows),batch):
  rows=seed_rows[st:st+batch];s=candidate[torch.as_tensor(rows,dtype=torch.long,device=candidate.device)]
  with torch.no_grad(): scores=s@candidate.T; vals,idx=torch.topk(scores,k=TOP_PROT,dim=1,largest=True,sorted=True)
  vals=vals.cpu().numpy();idx=idx.cpu().numpy()
  for j,r in enumerate(rows):res[int(r)]=(idx[j].astype(np.int32),vals[j].astype(np.float32))
  all_i.append(idx.astype(np.int32));all_v.append(vals.astype(np.float32))
  if st%960==0:print('seed-neighbor',min(st+batch,len(seed_rows)),'/',len(seed_rows),flush=True)
 cache_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache_path,seed_rows=np.asarray(seed_rows,dtype=np.int64),indices=np.concatenate(all_i),values=np.concatenate(all_v));return res

def main():
 OUT.mkdir(parents=True,exist_ok=True);device=torch.device('cuda')
 pf,pids=load_protein_library(PROT);pidx={p:i for i,p in enumerate(pids)};pt=torch.as_tensor(np.asarray(pf,dtype=np.float32),device=device)
 schema=load_feature_schema(RXN);rf,rids=load_registered_reaction_feature_library(RXN,schema);drfp=int(schema.get('drfp_dimension',2048))
 old=pd.read_csv(BASE_OOF,dtype={'query_id':str});records=[];fold_summaries={}
 for f in FOLDS:
  cell=CELL(f);train=pd.read_csv(DEV/'benchmarks'/cell/'train_pairs.csv',dtype=str).fillna('');dev=pd.read_csv(DEV/'benchmarks'/cell/'test_pairs.csv',dtype=str).fillna('')
  qids=sorted(dev.reaction_id.astype(str).unique());train_rids=sorted(set(train.reaction_id.astype(str)) & set(rids));neigh=top_reaction_neighbors(qids,train_rids,rids,rf,drfp)
  enz=train.groupby('reaction_id').protein_id.agg(lambda s:sorted(set(s.astype(str)))).to_dict(); needed=set()
  for q in qids:
   for r,w in neigh[q]:
    if w>0:needed.update(p for p in enz.get(r,[]) if p in pidx)
  seed_rows=sorted(pidx[p] for p in needed); print('fold',f,'queries',len(qids),'unique seed proteins',len(seed_rows),flush=True);top=exact_seed_top100(pt,seed_rows,OUT/f'fold{f}_seed_top100.npz');row_to_id=np.asarray(pids,dtype=object)
  base=old[old.fold.eq(f)].set_index('query_id');pos=dev.groupby('reaction_id').protein_id.agg(lambda s:sorted(set(s.astype(str)))).to_dict();foldrec=[]
  for qi,q in enumerate(qids):
   cand_score={}
   seed_count=0
   for r,rw in neigh[q]:
    if rw<=0:continue
    for p in enz.get(r,[]):
     if p not in pidx:continue
     seed_count+=1;inds,vals=top[pidx[p]]
     weighted=vals*float(rw)
     for rr,v in zip(inds,weighted,strict=True):
      rr=int(rr); vv=float(v)
      if vv>cand_score.get(rr,-1e9):cand_score[rr]=vv
   order=sorted(cand_score,key=lambda rr:(-cand_score[rr],pids[rr]))[:TOP_PROT];rank={pids[rr]:i+1 for i,rr in enumerate(order)}
   positives=pos[q];pr=[rank.get(p,TOP_PROT+1) for p in positives];rec={'fold':f,'query_id':q,'seed_proteins':seed_count,'neighbor_reaction_max_similarity':neigh[q][0][1] if neigh[q] else 0.,'positive_count':len(positives),'homology_positive_in_top100':sum(r<=100 for r in pr),'homology_hit_at_20':float(any(r<=20 for r in pr)),'homology_hit_at_50':float(any(r<=50 for r in pr)),'homology_hit_at_100':float(any(r<=100 for r in pr)),'homology_best_positive_rank_top100':min(pr),'bime_best_positive_rank':int(base.loc[q,'best_positive_rank']),'bime_hit_at_20':float(base.loc[q,'hit_at_20']),'bime_hit_at_50':float(base.loc[q,'hit_at_50'])}
   rec['rescues_bime_miss20']=float(rec['bime_hit_at_20']==0 and rec['homology_hit_at_20']==1);rec['rescues_bime_miss50']=float(rec['bime_hit_at_50']==0 and rec['homology_hit_at_50']==1);foldrec.append(rec);records.append(rec)
  d=pd.DataFrame(foldrec);fold_summaries[str(f)]={'queries':len(d),'homology_hit20':float(d.homology_hit_at_20.mean()),'homology_hit50':float(d.homology_hit_at_50.mean()),'homology_hit100':float(d.homology_hit_at_100.mean()),'bime_hit20':float(d.bime_hit_at_20.mean()),'bime_hit50':float(d.bime_hit_at_50.mean()),'rescue_miss20_fraction_all_queries':float(d.rescues_bime_miss20.mean()),'rescue_miss50_fraction_all_queries':float(d.rescues_bime_miss50.mean()),'rescue_among_bime_miss20':float(d.rescues_bime_miss20.sum()/max((d.bime_hit_at_20==0).sum(),1)),'rescue_among_bime_miss50':float(d.rescues_bime_miss50.sum()/max((d.bime_hit_at_50==0).sum(),1))}
  print('fold summary',fold_summaries[str(f)],flush=True)
 d=pd.DataFrame(records);d.to_csv(OUT/'query_audit.csv',index=False);summary={'status':'candidate_recall_diagnostic_complete','protocol':'Fold-safe production-semantic R2E homology diagnostic: top5 training reaction neighbors by binary-DRFP Tanimoto; their train-only positive enzymes seed ESM-C cosine transfer; exact global homology Top100 recovered from union of exact per-seed Top100 neighbors. Dev labels used only after ranking to measure complementarity.','topk_neighbor_reactions':TOP_RXN,'topk_homology_candidates':TOP_PROT,'external_metrics_used':False,'folds':fold_summaries,'pooled':{'queries':len(d),'homology_hit20':float(d.homology_hit_at_20.mean()),'homology_hit50':float(d.homology_hit_at_50.mean()),'homology_hit100':float(d.homology_hit_at_100.mean()),'bime_hit20':float(d.bime_hit_at_20.mean()),'bime_hit50':float(d.bime_hit_at_50.mean()),'rescue_among_bime_miss20':float(d.rescues_bime_miss20.sum()/max((d.bime_hit_at_20==0).sum(),1)),'rescue_among_bime_miss50':float(d.rescues_bime_miss50.sum()/max((d.bime_hit_at_50==0).sum(),1))},'next_rule':'Only if homology Top100 shows material complementary rescue on every fold, train a same-capacity BiME prefix ranker with homology features; otherwise reject homology as a ranking expert.'};(OUT/'candidate_recall_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
