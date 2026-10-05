from __future__ import annotations
import argparse,json
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as v4

OUT=ROOT/'results/bridge_reciprocal_relation_context_headband_v7'
ALPHAS=(0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,10.0,15.0,20.0)
PREFIXES=(5,10,20)
HEAD_END=100

def metric(r):
 a=np.asarray(r,np.int64);return {'queries':int(len(a)),'mrr':float((1/a).mean()),'hit10':float((a<=10).mean()),'hit100':float((a<=100).mean()),'hit1000':float((a<=1000).mean()),'median_rank':float(np.median(a))}
def rank_one(score,row,lex):
 v=float(score[row]);return int(np.count_nonzero(score>v)+np.count_nonzero((score==v)&(lex<lex[row]))+1)
def best_rank(score,t,lex):return min(rank_one(score,int(r),lex) for r in t)
def top100(score,lex):
 u=np.argpartition(-score,99)[:100];return u[np.lexsort((lex[u],-score[u]))]
def rerank_head(base,ctx,lex,alpha,prefix):
 top=top100(base,lex);rankmap={int(row):i+1 for i,row in enumerate(top)};rows=top[prefix:HEAD_END]
 if len(rows):
  mix=base[rows]+float(alpha)*ctx[rows];order=rows[np.lexsort((lex[rows],-mix))]
  for rank,row in zip(range(prefix+1,HEAD_END+1),order,strict=True):rankmap[int(row)]=int(rank)
 return rankmap,set(map(int,top))

def evaluate(direction,index,frame,pp,pm,rp,rm,configs,batch):
 P=index.protein_embeddings;R=index.reaction_embeddings;pdct=index.protein_index;rdct=index.reaction_index
 if direction=='r2e':targets=frame.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();qids=sorted(targets);cids=index.protein_ids;cidx=pdct;qidx=rdct
 else:targets=frame.groupby('protein_id').reaction_id.apply(lambda s:set(map(str,s))).to_dict();qids=sorted(targets);cids=index.reaction_ids;cidx=rdct;qidx=pdct
 train=pd.read_csv(v4.TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);known=(train.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict() if direction=='r2e' else train.groupby('protein_id').reaction_id.apply(lambda s:set(map(str,s))).to_dict())
 lex=np.empty(len(cids),np.int64);o=np.argsort(np.asarray(cids,dtype=object),kind='stable');lex[o]=np.arange(len(o));base_r=[];out={cfg:[] for cfg in configs}
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
   b=B[j].copy();c=C[j];seeds=[cidx[x] for x in known.get(q,set()) if x in cidx]
   if seeds:b[np.asarray(seeds,np.int64)]=-np.inf
   t=np.asarray([cidx[x] for x in targets[q] if x in cidx],np.int64)
   if not len(t):continue
   br=best_rank(b,t,lex);base_r.append(br)
   for cfg in configs:
    alpha,prefix=cfg
    if float(alpha)==0:out[cfg].append(br);continue
    rankmap,topset=rerank_head(b,c,lex,alpha,prefix);rr=[]
    for row in t:
     row=int(row);rr.append(rankmap[row] if row in topset else rank_one(b,row,lex))
    out[cfg].append(min(rr))
  print(direction,min(st+batch,len(qids)),'/',len(qids),flush=True)
 return {'base':metric(base_r),'grid':[{'alpha':a,'protected_prefix':p,**metric(out[(a,p)])} for a,p in configs]}

def choose(dev):
 br=dev['r2e']['base'];be=dev['e2r']['base'];cand=[]
 rmap={(x['alpha'],x['protected_prefix']):x for x in dev['r2e']['grid']};emap={(x['alpha'],x['protected_prefix']):x for x in dev['e2r']['grid']}
 for cfg,r in rmap.items():
  e=emap[cfg]
  safe=(r['mrr']>=br['mrr'] and e['mrr']>=be['mrr'] and r['hit10']>=br['hit10'] and e['hit10']>=be['hit10'] and r['hit100']>=br['hit100'] and e['hit100']>=be['hit100'] and r['hit1000']>=br['hit1000'] and e['hit1000']>=be['hit1000'] and r['median_rank']<=br['median_rank'] and e['median_rank']<=be['median_rank'])
  if safe:
   gain=((r['mrr']/max(br['mrr'],1e-12))*(e['mrr']/max(be['mrr'],1e-12)))**0.5;cand.append((gain,cfg))
 return max(cand)[1] if cand else (0.0,10)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);idx=FibreCandidateIndex(device=a.device);train=pd.read_csv(v4.TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);pp,pm,rp,rm=v4.build_prototypes(idx,train);devf=v4.load_dev(train);outer=pd.read_csv(v4.TARGETS,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates();configs=tuple((a,p) for p in PREFIXES for a in ALPHAS)
 dev={'r2e':evaluate('r2e',idx,devf,pp,pm,rp,rm,configs,v4.BATCH_R2E),'e2r':evaluate('e2r',idx,devf,pp,pm,rp,rm,configs,v4.BATCH_E2R)};selected=choose(dev);print('SELECTED',selected,flush=True);outerres={'r2e':evaluate('r2e',idx,outer,pp,pm,rp,rm,(selected,),v4.BATCH_R2E),'e2r':evaluate('e2r',idx,outer,pp,pm,rp,rm,(selected,),v4.BATCH_E2R)}
 result={'schema':'bridge-reciprocal-relation-context-headband-v7','protocol':{'train_graph':'clean2023 only','target':'exact relation unseen','pair_context':'symmetric reciprocal train-neighborhood prototypes','permission':'protect Top-P; rerank only P+1..100; rank>100 exact Broad','selection':'one shared P and alpha chosen on firewalled development; both directions require MRR non-regression, Hit@10/100/1000 non-regression, median non-worsening','outer_used_for_selection':False},'development':dev,'selected':{'alpha':selected[0],'protected_prefix':selected[1]},'outer':outerres};(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
