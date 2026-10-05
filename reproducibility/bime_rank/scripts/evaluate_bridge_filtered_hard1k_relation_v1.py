from __future__ import annotations
import argparse,json
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as v4

OUT=ROOT/'results/bridge_filtered_hard1k_relation_v1';K=1000;RAW_ALPHA=10.0;BOUNDED_ALPHA=20.0;PREFIX=5;HEAD_END=100

def metrics(r):
 a=np.asarray(r,np.int64);return {'edges':int(len(a)),'mrr':float((1/a).mean()),'hit1':float((a<=1).mean()),'hit10':float((a<=10).mean()),'hit100':float((a<=100).mean()),'median_rank':float(np.median(a))}
def rank_in_set(score,rows,target,lex):
 vals=score[rows];tv=float(score[target]);tl=int(lex[target]);return int(np.count_nonzero(vals>tv)+np.count_nonzero((vals==tv)&(lex[rows]<tl))+1)
def hard_negs(base,lex,forbidden,k):
 s=base.copy()
 if forbidden:s[np.asarray(sorted(forbidden),np.int64)]=-np.inf
 n=min(k,len(s)-len(forbidden));u=np.argpartition(-s,n-1)[:n];return u[np.lexsort((lex[u],-s[u]))]
def bounded_rank(base,ctx,neg,target,lex):
 rows=np.concatenate([[target],neg]);order=rows[np.lexsort((lex[rows],-base[rows]))];rankmap={int(r):i+1 for i,r in enumerate(order)};band=order[PREFIX:min(HEAD_END,len(order))]
 if len(band):
  mix=base[band]+BOUNDED_ALPHA*ctx[band];o=band[np.lexsort((lex[band],-mix))]
  for rank,row in zip(range(PREFIX+1,PREFIX+1+len(o)),o,strict=True):rankmap[int(row)]=int(rank)
 return rankmap[int(target)]

def evaluate(direction,index,frame,pp,pm,rp,rm,batch):
 P=index.protein_embeddings;R=index.reaction_embeddings;pdct=index.protein_index;rdct=index.reaction_index
 train=pd.read_csv(v4.TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id'])
 if direction=='r2e':targets=frame.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();known=train.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();qids=sorted(targets);cids=index.protein_ids;cidx=pdct;qidx=rdct
 else:targets=frame.groupby('protein_id').reaction_id.apply(lambda s:set(map(str,s))).to_dict();known=train.groupby('protein_id').reaction_id.apply(lambda s:set(map(str,s))).to_dict();qids=sorted(targets);cids=index.reaction_ids;cidx=rdct;qidx=pdct
 lex=np.empty(len(cids),np.int64);o=np.argsort(np.asarray(cids,dtype=object),kind='stable');lex[o]=np.arange(len(o));rb=[];rr=[];rbd=[];rowsout=[]
 for st in range(0,len(qids),batch):
  qs=[q for q in qids[st:st+batch] if q in qidx]
  if not qs:continue
  qi=torch.as_tensor([qidx[q] for q in qs],dtype=torch.long,device=index.device)
  if direction=='r2e':
   qemb=R.index_select(0,qi);base=qemb@P.T;qa=rp.index_select(0,qi)@P.T;qa_av=rm.index_select(0,qi)[:,None].expand_as(qa);qb=qemb@pp.T;qb_av=pm[None,:].expand_as(qb)
  else:
   qemb=P.index_select(0,qi);base=qemb@R.T;qa=pp.index_select(0,qi)@R.T;qa_av=pm.index_select(0,qi)[:,None].expand_as(qa);qb=qemb@rp.T;qb_av=rm[None,:].expand_as(qb)
  bz=v4.z_masked(base,torch.ones_like(base,dtype=torch.bool));za=v4.z_masked(qa,qa_av);zb=v4.z_masked(qb,qb_av);den=qa_av.float()+qb_av.float();ctx=(za+zb)/den.clamp_min(1.0);ctx[den==0]=0
  B=bz.cpu().numpy();C=ctx.cpu().numpy()
  for j,q in enumerate(qs):
   b=B[j];c=C[j];pos=[cidx[x] for x in targets[q] if x in cidx];trainpos={cidx[x] for x in known.get(q,set()) if x in cidx};allpos=set(pos);forbidden=trainpos|allpos;neg=hard_negs(b,lex,forbidden,K-1);qdeg=len(trainpos)
   for t in pos:
    candidates=np.concatenate([[t],neg]);br=rank_in_set(b,candidates,t,lex);raw=rank_in_set(b+RAW_ALPHA*c,candidates,t,lex);bd=bounded_rank(b,c,neg,t,lex);rb.append(br);rr.append(raw);rbd.append(bd);rowsout.append({'direction':direction,'query_id':q,'target_id':cids[t],'query_train_degree':qdeg,'broad_rank':br,'reciprocal_raw_rank':raw,'reciprocal_bounded_rank':bd})
  print(direction,min(st+batch,len(qids)),'/',len(qids),flush=True)
 df=pd.DataFrame(rowsout)
 buckets={}
 for name,g in [('deg0',df[df.query_train_degree.eq(0)]),('deg1',df[df.query_train_degree.eq(1)]),('deg2_4',df[df.query_train_degree.between(2,4)]),('deg5plus',df[df.query_train_degree.ge(5)])]:
  if len(g):buckets[name]={'edges':int(len(g)),'broad':metrics(g.broad_rank),'reciprocal_bounded':metrics(g.reciprocal_bounded_rank)}
 return {'broad':metrics(rb),'reciprocal_raw':metrics(rr),'reciprocal_bounded':metrics(rbd),'degree_buckets':buckets},df

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);idx=FibreCandidateIndex(device=a.device);train=pd.read_csv(v4.TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);pp,pm,rp,rm=v4.build_prototypes(idx,train);frame=pd.read_csv(v4.TARGETS,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates();r,rd=evaluate('r2e',idx,frame,pp,pm,rp,rm,v4.BATCH_R2E);e,ed=evaluate('e2r',idx,frame,pp,pm,rp,rm,v4.BATCH_E2R);pd.concat([rd,ed],ignore_index=True).to_csv(OUT/'edge_metrics.csv.gz',index=False)
 result={'schema':'bridge-filtered-hard1k-relation-v1','protocol':{'targets':'same 23,773 exact relation-unseen edges for both directions','candidate_budget':K,'negative_policy':'999 highest-Broad-scoring candidates after filtering every clean2023 train-positive and every frozen relation-unseen positive for the same query; deterministic lexical tie-break','context':'symmetric reciprocal train-neighborhood prototypes only','raw_alpha':RAW_ALPHA,'bounded':{'alpha':BOUNDED_ALPHA,'protected_prefix':PREFIX,'rerank_end':HEAD_END},'target_relations_used_as_context':False,'purpose':'candidate-budget-matched filtered hard-negative relation ranking; not an end-to-end full-universe recall metric'},'r2e':r,'e2r':e};(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
