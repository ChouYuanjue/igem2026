from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
TARGETS=ROOT/'results/bridge_relation_unseen_max_v3/targets.csv'
BENCH=ROOT/'results/broad_rhea_fair_benchmarks_v1'
POOL_CELL=BENCH/'broad_pair_hash_holdout_both_seen'
OUT=ROOT/'results/bridge_reciprocal_relation_context_v4'
ALPHAS=(0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,10.0)
BATCH_R2E=32;BATCH_E2R=256

def normalize_rows(x):
 return x/x.norm(dim=1,keepdim=True).clamp_min(1e-8)
def z_masked(x,mask):
 # x BxN, mask BxN bool; missing entries become neutral 0.
 out=torch.zeros_like(x);count=mask.sum(1,keepdim=True);valid=(count[:,0]>1)
 if valid.any():
  xv=x[valid];mv=mask[valid];n=mv.sum(1,keepdim=True).float();mean=(xv*mv).sum(1,keepdim=True)/n;var=(((xv-mean)*mv)**2).sum(1,keepdim=True)/n;std=var.sqrt().clamp_min(1e-6);tmp=(xv-mean)/std;tmp[~mv]=0;out[valid]=tmp
 return out

def metrics(r):
 a=np.asarray(r,np.int64);return {'queries':int(len(a)),'mrr':float((1/a).mean()),'hit10':float((a<=10).mean()),'hit100':float((a<=100).mean()),'hit1000':float((a<=1000).mean()),'median_rank':float(np.median(a))}
def ranks(score,targets,lex):
 vals=score[targets];best=float(vals.max());rows=targets[vals==best];row=int(rows[np.argmin(lex[rows])]);return int(np.count_nonzero(score>best)+np.count_nonzero((score==best)&(lex<lex[row]))+1)

def build_prototypes(index,train):
 device=index.device;P=index.protein_embeddings;R=index.reaction_embeddings;pdct=index.protein_index;rdct=index.reaction_index
 psum=torch.zeros_like(P);pcnt=torch.zeros((len(index.protein_ids),1),device=device);rsum=torch.zeros_like(R);rcnt=torch.zeros((len(index.reaction_ids),1),device=device)
 prow=[];rrow=[]
 for p,r in train[['protein_id','reaction_id']].itertuples(index=False):
  if p in pdct and r in rdct:prow.append(pdct[p]);rrow.append(rdct[r])
 pi=torch.as_tensor(prow,dtype=torch.long,device=device);ri=torch.as_tensor(rrow,dtype=torch.long,device=device)
 # protein -> mean known reaction embedding
 psum.index_add_(0,pi,R.index_select(0,ri));pcnt.index_add_(0,pi,torch.ones((len(pi),1),device=device))
 # reaction -> mean known protein embedding
 rsum.index_add_(0,ri,P.index_select(0,pi));rcnt.index_add_(0,ri,torch.ones((len(ri),1),device=device))
 pproto=torch.zeros_like(P);rproto=torch.zeros_like(R);pm=(pcnt[:,0]>0);rm=(rcnt[:,0]>0);pproto[pm]=normalize_rows(psum[pm]/pcnt[pm]);rproto[rm]=normalize_rows(rsum[rm]/rcnt[rm]);return pproto,pm,rproto,rm

def load_dev(train):
 newer=pd.concat([pd.read_csv(POOL_CELL/'train_pairs.csv',dtype=str),pd.read_csv(POOL_CELL/'test_pairs.csv',dtype=str)],ignore_index=True).fillna('').drop_duplicates(['protein_id','reaction_id'])
 old=set(zip(train.protein_id,train.reaction_id));outer=set()
 for d in BENCH.iterdir():
  p=d/'test_pairs.csv'
  if p.exists():
   x=pd.read_csv(p,dtype=str).fillna('');outer|=set(zip(x.protein_id.astype(str),x.reaction_id.astype(str)))
 new=(set(zip(newer.protein_id,newer.reaction_id))-old)-outer
 return pd.DataFrame(sorted(new),columns=['protein_id','reaction_id'])

def evaluate(direction,index,frame,pproto,pmask,rproto,rmask,alphas,batch):
 P=index.protein_embeddings;R=index.reaction_embeddings;pdct=index.protein_index;rdct=index.reaction_index
 known_p=frame.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict() if direction=='r2e' else None
 known_r=frame.groupby('protein_id').reaction_id.apply(lambda s:set(map(str,s))).to_dict() if direction=='e2r' else None
 if direction=='r2e':qids=sorted(known_p);cids=index.protein_ids;cidx=pdct;qidx=rdct;C=P
 else:qids=sorted(known_r);cids=index.reaction_ids;cidx=rdct;qidx=pdct;C=R
 lex=np.empty(len(cids),np.int64);o=np.argsort(np.asarray(cids,dtype=object),kind='stable');lex[o]=np.arange(len(o));out={a:[] for a in alphas};baseout=[];coverage=[]
 train=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);train_q=(train.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict() if direction=='r2e' else train.groupby('protein_id').reaction_id.apply(lambda s:set(map(str,s))).to_dict())
 for st in range(0,len(qids),batch):
  qs=[q for q in qids[st:st+batch] if q in qidx]
  if not qs:continue
  qi=torch.as_tensor([qidx[q] for q in qs],dtype=torch.long,device=index.device)
  if direction=='r2e':
   qemb=R.index_select(0,qi);base=qemb@P.T
   # term A: candidate protein vs query reaction's train-known protein prototype
   qa=rproto.index_select(0,qi)@P.T; qa_av=rmask.index_select(0,qi)[:,None].expand_as(qa)
   # term B: query reaction vs candidate protein's train-known reaction prototype
   qb=qemb@pproto.T; qb_av=pmask[None,:].expand_as(qb)
  else:
   qemb=P.index_select(0,qi);base=qemb@R.T
   # term B: candidate reaction vs query protein's train-known reaction prototype
   qa=pproto.index_select(0,qi)@R.T; qa_av=pmask.index_select(0,qi)[:,None].expand_as(qa)
   # term A: query protein vs candidate reaction's train-known protein prototype
   qb=qemb@rproto.T; qb_av=rmask[None,:].expand_as(qb)
  bz=z_masked(base,torch.ones_like(base,dtype=torch.bool));za=z_masked(qa,qa_av);zb=z_masked(qb,qb_av);den=qa_av.float()+qb_av.float();ctx=(za+zb)/den.clamp_min(1.0);ctx[den==0]=0
  base_np=bz.cpu().numpy();ctx_np=ctx.cpu().numpy()
  for j,q in enumerate(qs):
   seeds=[cidx[x] for x in train_q.get(q,set()) if x in cidx]
   targets=(known_p[q] if direction=='r2e' else known_r[q]);t=np.asarray([cidx[x] for x in targets if x in cidx],np.int64)
   if not len(t):continue
   b=base_np[j].copy();c=ctx_np[j].copy();
   if seeds:b[np.asarray(seeds,np.int64)]=-np.inf
   br=ranks(b,t,lex);baseout.append(br)
   for a in alphas:
    s=b+float(a)*c
    if seeds:s[np.asarray(seeds,np.int64)]=-np.inf
    out[a].append(ranks(s,t,lex))
   coverage.append({'query_id':q,'query_neighbor_available':int(bool(seeds)),'candidate_neighbor_fraction':float((pmask.float().mean() if direction=='r2e' else rmask.float().mean()).item())})
  print(direction,min(st+batch,len(qids)),'/',len(qids),flush=True)
 return {'base':metrics(baseout),'grid':[{'alpha':a,**metrics(out[a])} for a in alphas],'coverage':{'queries':len(coverage),'query_neighbor_fraction':float(np.mean([x['query_neighbor_available'] for x in coverage])) if coverage else 0.0}}

def choose_common(dev):
 bR=dev['r2e']['base'];bE=dev['e2r']['base'];c=[]
 for rr,ee in zip(dev['r2e']['grid'],dev['e2r']['grid']):
  if rr['alpha']!=ee['alpha']:continue
  if rr['mrr']>=bR['mrr'] and ee['mrr']>=bE['mrr'] and rr['hit10']>=bR['hit10'] and ee['hit10']>=bE['hit10']:
   gain=((rr['mrr']/max(bR['mrr'],1e-12))*(ee['mrr']/max(bE['mrr'],1e-12)))**0.5;c.append((gain,rr['alpha']))
 return max(c)[1] if c else 0.0

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);idx=FibreCandidateIndex(device=a.device);train=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);pp,pm,rp,rm=build_prototypes(idx,train);devf=load_dev(train);outer=pd.read_csv(TARGETS,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates()
 dev={'r2e':evaluate('r2e',idx,devf,pp,pm,rp,rm,ALPHAS,BATCH_R2E),'e2r':evaluate('e2r',idx,devf,pp,pm,rp,rm,ALPHAS,BATCH_E2R)};alpha=choose_common(dev);print('SELECTED COMMON ALPHA',alpha,flush=True)
 outerres={'r2e':evaluate('r2e',idx,outer,pp,pm,rp,rm,(alpha,),BATCH_R2E),'e2r':evaluate('e2r',idx,outer,pp,pm,rp,rm,(alpha,),BATCH_E2R)}
 result={'schema':'bridge-reciprocal-relation-context-v4','protocol':{'train_graph':'clean2023 only','target':'exact relation unseen','pair_context':'mean of enzyme-to-reaction-neighborhood prototype similarity and reaction-to-enzyme-neighborhood prototype similarity; missing side neutral','direction_invariant_pair_score':True,'alpha_selection':'one common alpha selected on firewalled source-expansion development relations; both R2E and E2R MRR/Hit10 must not regress','outer_used_for_selection':False},'development':dev,'selected_common_alpha':alpha,'outer':outerres};(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
