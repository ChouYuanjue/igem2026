from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
TARGETS=ROOT/'results/bridge_relation_unseen_max_v3/targets.csv'
DEV=ROOT/'results/bridge_relation_seed_strength_v2/summary.json'
OUT=ROOT/'results/bridge_relation_seed_max_v3'


def z(x):
 x=np.asarray(x,dtype=np.float64);return (x-x.mean())/max(float(x.std()),1e-8)

def best_rank(score,target_rows,lex):
 tr=np.asarray(target_rows,dtype=np.int64);vals=score[tr];best=float(vals.max());br=tr[vals==best];row=int(br[np.argmin(lex[br])]);return int(np.count_nonzero(score>best)+np.count_nonzero((score==best)&(lex<lex[row]))+1)

def metrics(df,col):
 r=df[col].to_numpy(np.int64);return {'instances':int(len(r)),'mrr':float((1/r).mean()),'hit10':float((r<=10).mean()),'hit100':float((r<=100).mean()),'hit1000':float((r<=1000).mean()),'median_rank':float(np.median(r))}

def relation_data(direction):
 tr=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id'])
 if direction=='r2e': known=tr.groupby('reaction_id').protein_id.apply(lambda x:set(x.astype(str))).to_dict(); qcol='reaction_id';ccol='protein_id'
 else: known=tr.groupby('protein_id').reaction_id.apply(lambda x:set(x.astype(str))).to_dict(); qcol='protein_id';ccol='reaction_id'
 x=pd.read_csv(TARGETS,dtype=str).fillna('')[[qcol,ccol]].drop_duplicates()
 targets=x.groupby(qcol)[ccol].apply(lambda g:set(g.astype(str))).to_dict()
 return known,targets


def evaluate(direction,index,alpha):
 known,targets=relation_data(direction)
 if direction=='r2e':
  C=index.protein_embeddings;Q=index.reaction_embeddings;cids=index.protein_ids;cidx=index.protein_index;qidx=index.reaction_index
 else:
  C=index.reaction_embeddings;Q=index.protein_embeddings;cids=index.reaction_ids;cidx=index.reaction_index;qidx=index.protein_index
 lex=np.empty(len(cids),dtype=np.int64);o=np.argsort(np.asarray(cids,dtype=object),kind='stable');lex[o]=np.arange(len(o))
 rows=[];audit=[]
 for qi,q in enumerate(sorted(targets)):
  if q not in qidx:continue
  seeds=[cidx[x] for x in known.get(q,set()) if x in cidx]
  with torch.no_grad(): base=(C@Q[qidx[q]]).float().cpu().numpy().astype(np.float64)
  base_score=z(base); ctx_score=base_score.copy()
  if seeds:
   sr=torch.as_tensor(seeds,dtype=torch.long,device=index.device)
   with torch.no_grad(): seed=(C@C.index_select(0,sr).T).max(dim=1).values.float().cpu().numpy().astype(np.float64)
   ctx_score=base_score+float(alpha)*z(seed)
  srows=np.asarray(seeds,dtype=np.int64)
  if len(srows):base_score[srows]=-np.inf;ctx_score[srows]=-np.inf
  audit.append({'query_id':q,'seed_count':len(seeds),'context_active':int(bool(seeds))})
  pos=targets[q]
  t=[cidx[x] for x in pos if x in cidx]
  if t: rows.append({'query_id':q,'seed_count':len(seeds),'target_relations':len(t),'base_rank':best_rank(base_score,t,lex),'context_rank':best_rank(ctx_score,t,lex)})
  if (qi+1)%100==0:print(direction,qi+1,'/',len(targets),flush=True)
 df=pd.DataFrame(rows);ad=pd.DataFrame(audit)
 out={'direction':direction,'alpha':float(alpha),'unique_queries':int(df.query_id.nunique()),'relation_unseen_queries_evaluated':int(len(df)),'seed_active_unique_fraction':float(ad.context_active.mean()),'overall':{'broad':metrics(df,'base_rank'),'broad_plus_train_seed_context':metrics(df,'context_rank')},'seed_active_subset':{}}
 g=df[df.seed_count>0]
 if len(g):out['seed_active_subset']={'broad':metrics(g,'base_rank'),'broad_plus_train_seed_context':metrics(g,'context_rank'),'instances':int(len(g))}
 df.to_csv(OUT/f'{direction}_query_metrics.csv',index=False);ad.to_csv(OUT/f'{direction}_query_audit.csv',index=False)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 dev=json.loads(DEV.read_text());ar=float(dev['r2e']['selected']['alpha']);ae=float(dev['e2r']['selected']['alpha'])
 idx=FibreCandidateIndex(device=a.device)
 result={'schema':'bridge-relation-seed-max-v3','status':'maximal_relation_unseen_confirmation','protocol':'all maximal fair-pool relations absent clean2023; no enzyme/reaction cold restriction; every train-known relation for query is a masked seed; no seed means exact Broad fallback; alpha frozen before this benchmark','r2e':evaluate('r2e',idx,ar),'e2r':evaluate('e2r',idx,ae),'outer_metrics_used_for_alpha_selection':False}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
