from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
TARGETS=ROOT/'results/bridge_relation_unseen_max_v3/targets.csv'
OUT=ROOT/'results/bridge_relation_unseen_edgewise_broad_v1'

def metrics(df,col='rank'):
 r=df[col].to_numpy(np.int64);return {'edges':int(len(r)),'mrr':float((1/r).mean()),'hit10':float((r<=10).mean()),'hit100':float((r<=100).mean()),'hit1000':float((r<=1000).mean()),'median_rank':float(np.median(r))}

def evaluate(direction,index,train,target,batch):
 if direction=='r2e':
  qcol,ccol='reaction_id','protein_id';Q=index.reaction_embeddings;C=index.protein_embeddings;qidx=index.reaction_index;cidx=index.protein_index;cids=index.protein_ids
 else:
  qcol,ccol='protein_id','reaction_id';Q=index.protein_embeddings;C=index.reaction_embeddings;qidx=index.protein_index;cidx=index.reaction_index;cids=index.reaction_ids
 # all known true candidates for filtered evaluation = train + every target edge in the frozen relation-unseen set
 alltrue=pd.concat([train[[qcol,ccol]],target[[qcol,ccol]]],ignore_index=True).drop_duplicates()
 true_map=alltrue.groupby(qcol)[ccol].apply(lambda s:set(map(str,s))).to_dict(); target_map=target.groupby(qcol)[ccol].apply(lambda s:list(map(str,s))).to_dict()
 lex=np.empty(len(cids),np.int64);o=np.argsort(np.asarray(cids,dtype=object),kind='stable');lex[o]=np.arange(len(o));rows=[];qs=sorted(target_map)
 for st in range(0,len(qs),batch):
  qb=[q for q in qs[st:st+batch] if q in qidx]
  qi=torch.as_tensor([qidx[q] for q in qb],dtype=torch.long,device=index.device)
  with torch.no_grad():S=(Q.index_select(0,qi)@C.T).float().cpu().numpy()
  for j,q in enumerate(qb):
   score=S[j].astype(np.float64);targets=[x for x in target_map[q] if x in cidx];all_rows=[cidx[x] for x in true_map.get(q,set()) if x in cidx]
   all_set=set(all_rows)
   for t in targets:
    tr=int(cidx[t]);tv=float(score[tr]);# filter every other true positive, but keep target itself
    # exact rank among admissible candidates after filtering other positives
    greater=(score>tv);ties=(score==tv)&(lex<lex[tr]);
    for rr in all_set:
     if rr!=tr: greater[rr]=False;ties[rr]=False
    rank=int(greater.sum()+ties.sum()+1);rows.append({'direction':direction,'query_id':q,'target_id':t,'rank':rank,'candidate_count_filtered':int(len(cids)-max(len(all_set)-1,0))})
  print(direction,min(st+batch,len(qs)),'/',len(qs),flush=True)
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);idx=FibreCandidateIndex(device=a.device);tr=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);ta=pd.read_csv(TARGETS,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);sp=set(tr.protein_id);sr=set(tr.reaction_id);ta['protein_seen']=ta.protein_id.isin(sp);ta['reaction_seen']=ta.reaction_id.isin(sr)
 r=evaluate('r2e',idx,tr,ta,32);e=evaluate('e2r',idx,tr,ta,256);key=ta[['protein_id','reaction_id','protein_seen','reaction_seen']];r=r.merge(key,left_on=['target_id','query_id'],right_on=['protein_id','reaction_id']);e=e.merge(key,left_on=['query_id','target_id'],right_on=['protein_id','reaction_id'])
 r.to_csv(OUT/'r2e_edges.csv.gz',index=False);e.to_csv(OUT/'e2r_edges.csv.gz',index=False)
 def group(df):
  out={}
  masks={'protein_cold_only':(~df.protein_seen)&df.reaction_seen,'reaction_cold_only':df.protein_seen&(~df.reaction_seen),'double_cold':(~df.protein_seen)&(~df.reaction_seen),'both_seen_edge_unseen':df.protein_seen&df.reaction_seen}
  for n,m in masks.items():out[n]=metrics(df[m]) if m.any() else None
  out['all']=metrics(df);return out
 result={'schema':'bridge-relation-unseen-edgewise-broad-v1','protocol':{'unit':'one held-out relation edge','filtered':'all other true relations for the same query in train plus frozen target set are removed','candidate_universe':'full 185918 proteins for R2E / full 11081 reactions for E2R before true-positive filtering','target_relation_unseen':True},'r2e':group(r),'e2r':group(e)};(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
