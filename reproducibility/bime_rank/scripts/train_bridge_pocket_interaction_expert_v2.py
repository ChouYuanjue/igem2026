from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TRAIN_BASE=ROOT/'results/bridge_pocket_interaction_train_v2'
EVAL_BASE=ROOT/'results/bridge_pocket_interaction_repr_v1'
FOLDS=ROOT/'results/cleanroom_internal_full_candidate_benchmarks_v1'
TARGETS=ROOT/'results/bridge_relation_unseen_max_v3/targets.csv'
TRAIN_REL=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
OUT=ROOT/'results/bridge_pocket_interaction_expert_v2'
ALPHAS=(0.01,0.02,0.05,0.1,0.2,0.35,0.5)
PREFIXES=(0,5,10,20)
NEG_PER_POS=16;EPOCHS=12;BATCH=8192;WD=1e-4;LR=3e-3;SEED=20261005

def load_train():
 d=pd.read_csv(TRAIN_BASE/'pairs_scored.csv.gz',dtype={'reaction_id':str,'protein_id':str,'heldout_folds':str}).fillna('');d['Label']=pd.to_numeric(d.Label).astype(int);x=np.load(TRAIN_BASE/'interaction_fused_f32.npy').astype(np.float32);return d,x

def load_eval(tag):
 d=pd.read_csv(EVAL_BASE/tag/'pairs.csv.gz',dtype={'reaction_id':str,'protein_id':str}).fillna('');x=np.load(EVAL_BASE/tag/'interaction_fused_f32.npy').astype(np.float32);return d,x

def heldout(mask_value:str,fold:int)->bool:
 return str(fold) in [z for z in str(mask_value).split(',') if z!='']

def build_pairs(meta,holdout_fold=None):
 pos_i=[];neg_i=[];qcount=0;pcount=0
 for q,idx in meta.groupby('reaction_id',sort=False).groups.items():
  rows=np.asarray(list(idx),dtype=np.int64);g=meta.loc[rows];lab=g.Label.to_numpy(np.int8);pos=rows[lab>0];neg=rows[lab<=0]
  if holdout_fold is not None and len(pos):
   keep=np.asarray([not heldout(meta.at[int(r),'heldout_folds'],holdout_fold) for r in pos],dtype=bool);pos=pos[keep]
  if not len(pos) or not len(neg):continue
  neg=neg[np.argsort(pd.to_numeric(meta.loc[neg,'broad_rank_top1000']).to_numpy(),kind='stable')]
  chosen=neg[:min(NEG_PER_POS,len(neg))]
  for p in pos:
   pos_i.extend([int(p)]*len(chosen));neg_i.extend(map(int,chosen));pcount+=1
  qcount+=1
 return np.asarray(pos_i,np.int64),np.asarray(neg_i,np.int64),{'queries':qcount,'positive_rows':pcount,'pairwise_pairs':len(pos_i)}

def train_linear(meta,x,device,holdout_fold=None):
 pi,ni,audit=build_pairs(meta,holdout_fold);used=np.unique(np.concatenate([pi,ni]));mean=x[used].mean(0).astype(np.float32);std=x[used].std(0).astype(np.float32);std=np.maximum(std,1e-5)
 xt=torch.as_tensor((x-mean)/std,dtype=torch.float32,device=device);p=torch.as_tensor(pi,dtype=torch.long,device=device);n=torch.as_tensor(ni,dtype=torch.long,device=device);torch.manual_seed(SEED+(holdout_fold if holdout_fold is not None else 9));w=torch.nn.Parameter(torch.zeros(x.shape[1],device=device));opt=torch.optim.AdamW([w],lr=LR,weight_decay=WD);gen=torch.Generator(device='cpu').manual_seed(SEED+17+(holdout_fold or 0))
 losses=[]
 for ep in range(EPOCHS):
  perm=torch.randperm(len(pi),generator=gen);total=0.0
  for st in range(0,len(pi),BATCH):
   sel=perm[st:st+BATCH].to(device);delta=(xt.index_select(0,p.index_select(0,sel))-xt.index_select(0,n.index_select(0,sel)))@w;loss=torch.nn.functional.softplus(-delta).mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss.item())*len(sel)
  losses.append(total/len(pi))
 score=((x-mean)/std)@w.detach().cpu().numpy().astype(np.float32);model={'mean':mean,'std':std,'weight':w.detach().cpu().numpy().astype(np.float32)};audit['loss_first']=losses[0];audit['loss_last']=losses[-1];audit['epochs']=EPOCHS;return model,score.astype(np.float64),audit

def apply(model,x):return (((x-model['mean'])/model['std'])@model['weight']).astype(np.float64)
def z(v):
 v=np.asarray(v,float);return (v-v.mean())/max(float(v.std()),1e-8)
def metrics(r):
 a=np.asarray(r,np.int64);return {'queries':int(len(a)),'mrr':float((1/a).mean()),'hit10':float((a<=10).mean()),'hit20':float((a<=20).mean()),'hit100':float((a<=100).mean()),'hit1000':float((a<=1000).mean())}
def exact_rank(full,target,prows):
 t=torch.as_tensor(target,dtype=torch.long,device=full.device);v=full.index_select(0,t);b=v.max();br=t[v==b].min();return int(((full>b).sum()+((full==b)&(prows<br)).sum()+1).item())
def local_rank(g,score,alpha,prefix):
 orig=pd.to_numeric(g.broad_rank_top1000).to_numpy(np.int64);mix=z(pd.to_numeric(g.broad_score).to_numpy(float))+float(alpha)*z(score);new=orig.copy();mov=np.flatnonzero(orig>prefix)
 if len(mov):
  order=mov[np.argsort(-mix[mov],kind='stable')];slots=np.sort(orig[mov]);new[order]=slots
 return new

def eval_fold(fold,meta,score,index,selected=None):
 test=pd.read_csv(FOLDS/f'clean2023_internal_double_cold_fold{fold}/test_pairs.csv',dtype=str).fillna('');train=pd.read_csv(FOLDS/f'clean2023_internal_double_cold_fold{fold}/train_pairs.csv',dtype=str).fillna('');pos=test.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();known=train.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();m=meta.copy();m['expert_score']=score;byq={q:g.sort_values(['broad_rank_top1000','protein_id'],kind='stable') for q,g in m.groupby('reaction_id',sort=False)};pid=index.protein_index;prows=torch.arange(len(index.protein_ids),device=index.device);base=[];keys=[selected] if selected else [(a,p) for a in ALPHAS for p in PREFIXES];vals={k:[] for k in keys}
 queries=sorted(pos)
 for st in range(0,len(queries),128):
  qs=[q for q in queries[st:st+128] if q in index.reaction_index];qr=torch.as_tensor([index.reaction_index[q] for q in qs],dtype=torch.long,device=index.device)
  with torch.no_grad():scores=(index.reaction_embeddings.index_select(0,qr)@index.protein_embeddings.T).float()
  for j,q in enumerate(qs):
   full=scores[j].clone();kr=[pid[x] for x in known.get(q,set()) if x in pid]
   if kr:full[torch.as_tensor(kr,dtype=torch.long,device=index.device)]=-torch.inf
   targ=[pid[x] for x in pos[q] if x in pid]
   if not targ:continue
   br=exact_rank(full,targ,prows);base.append(br);g=byq.get(q)
   if g is None or len(g)<8:
    for k in vals:vals[k].append(br)
    continue
   lookup={p:i for i,p in enumerate(g.protein_id.astype(str))};tn=[index.protein_ids[x] for x in targ]
   for k in vals:
    nr=local_rank(g,g.expert_score.to_numpy(float),*k);rr=[int(nr[lookup[p]]) if p in lookup else br for p in tn];vals[k].append(min(rr))
 return metrics(base),{k:metrics(v) for k,v in vals.items()}

def eval_outer(meta,score,index,selected):
 target=pd.read_csv(TARGETS,dtype=str).fillna('');pos=target.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();train=pd.read_csv(TRAIN_REL,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);known=train.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();m=meta.copy();m['expert_score']=score;byq={q:g.sort_values(['broad_rank_top1000','protein_id'],kind='stable') for q,g in m.groupby('reaction_id',sort=False)};pid=index.protein_index;prows=torch.arange(len(index.protein_ids),device=index.device);base=[];expert=[];active=[]
 qsall=sorted(pos)
 for st in range(0,len(qsall),128):
  qs=[q for q in qsall[st:st+128] if q in index.reaction_index];qr=torch.as_tensor([index.reaction_index[q] for q in qs],dtype=torch.long,device=index.device)
  with torch.no_grad():scores=(index.reaction_embeddings.index_select(0,qr)@index.protein_embeddings.T).float()
  for j,q in enumerate(qs):
   full=scores[j].clone();kr=[pid[x] for x in known.get(q,set()) if x in pid]
   if kr:full[torch.as_tensor(kr,dtype=torch.long,device=index.device)]=-torch.inf
   targ=[pid[x] for x in pos[q] if x in pid]
   if not targ:continue
   br=exact_rank(full,targ,prows);base.append(br);g=byq.get(q)
   if g is None or len(g)<8:expert.append(br);active.append(0);continue
   nr=local_rank(g,g.expert_score.to_numpy(float),*selected);lookup={p:i for i,p in enumerate(g.protein_id.astype(str))};tn=[index.protein_ids[x] for x in targ];rr=[int(nr[lookup[p]]) if p in lookup else br for p in tn];expert.append(min(rr));active.append(1)
 return metrics(base),metrics(expert),np.asarray(active,bool)

def save_model(model,path):np.savez(path,mean=model['mean'],std=model['std'],weight=model['weight'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();device=torch.device(a.device);OUT.mkdir(parents=True,exist_ok=True);train_meta,train_x=load_train();evaldata={f:load_eval(f'fold{f}') for f in (0,1,2)};index=FibreCandidateIndex(device=a.device)
 models={};audits={};scores={};results={}
 for f in (0,1):
  model,_,audit=train_linear(train_meta,train_x,device,holdout_fold=f);models[f]=model;audits[f]=audit;s=apply(model,evaldata[f][1]);scores[f]=s;results[f]=eval_fold(f,evaldata[f][0],s,index)
 b0,v0=results[0];b1,v1=results[1];cand=[]
 for k in v0:
  g0=v0[k]['mrr']-b0['mrr'];g1=v1[k]['mrr']-b1['mrr'];ok=g0>=0 and g1>=0 and (g0+g1)>0 and v0[k]['hit10']>=b0['hit10'] and v1[k]['hit10']>=b1['hit10']
  if ok:cand.append(((g0+g1)/2,k))
 selected=max(cand,key=lambda t:t[0])[1] if cand else None
 model2,_,audit2=train_linear(train_meta,train_x,device,holdout_fold=2);s2=apply(model2,evaldata[2][1]);b2,v2=eval_fold(2,evaldata[2][0],s2,index,selected) if selected else (None,{})
 confirmed=bool(selected and v2[selected]['mrr']>b2['mrr'] and v2[selected]['hit10']>=b2['hit10'])
 result={'schema':'bridge-pocket-interaction-expert-v2','representation':'256-d pocket-substrate bidirectional cross-attention fused vector before global ESM/DRFP','training':'all clean2023 pocket-supported positives + Broad-hard negatives; exact held-out relation removed for each validation fold; entities may remain seen','ranker':{'type':'linear pairwise logistic','epochs':EPOCHS,'negative_per_positive':NEG_PER_POS,'weight_decay':WD,'learning_rate':LR},'selection':'fold0/1 relation-heldout; require nonnegative MRR and Hit10 in each; fold2 untouched confirmation; outer unused for selection','selected':{'alpha':selected[0],'protected_prefix':selected[1]} if selected else None,'fold0':{'train':audits[0],'broad':b0,'selected':v0.get(selected) if selected else None},'fold1':{'train':audits[1],'broad':b1,'selected':v1.get(selected) if selected else None},'fold2':{'train':audit2,'broad':b2,'selected':v2.get(selected) if selected else None},'confirmed':confirmed}
 if confirmed:
  final_model,_,af=train_linear(train_meta,train_x,device,holdout_fold=None);save_model(final_model,OUT/'model.npz');om,ox=load_eval('outer_max');ob,oe,act=eval_outer(om,apply(final_model,ox),index,selected);result['final_train']=af;result['outer']={'broad':ob,'broad_plus_pocket_interaction':oe,'delta_mrr':oe['mrr']-ob['mrr'],'delta_hit10_pp':100*(oe['hit10']-ob['hit10']),'delta_hit20_pp':100*(oe['hit20']-ob['hit20']),'delta_hit100_pp':100*(oe['hit100']-ob['hit100']),'active_queries':int(act.sum()),'active_fraction':float(act.mean())}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
