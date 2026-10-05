from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import train_bridge_pocket_interaction_expert_v2 as v2
from reproducibility.bime_rank.scripts import train_bridge_pocket_interaction_expert_v3 as v3

OUT=ROOT/'results/bridge_pocket_loo_expert_v6'
MODES=('raw','directed_product','agreement_plus_directed')
EPOCHS=18;BATCH=8192;LR=3e-3;WD=1e-4;SEED=20261005

def heldout(s,fold):return str(fold) in [z for z in str(s).split(',') if z]
def transform(x,mode):return x.astype(np.float32) if mode=='raw' else v3.transform(x,mode)

def build_loo_pairs(meta,holdout_fold=None):
 pos_i=[];neg_i=[];episodes=0;queries=set()
 for q,idx in meta.groupby('reaction_id',sort=False).groups.items():
  rows=np.asarray(list(idx),dtype=np.int64);g=meta.loc[rows];lab=g.Label.to_numpy(np.int8);pos=rows[lab>0];neg=rows[lab<=0]
  if not len(pos) or not len(neg):continue
  neg=neg[np.argsort(-pd.to_numeric(meta.loc[neg,'broad_score']).to_numpy(),kind='stable')]
  for p in pos:
   if holdout_fold is not None and heldout(meta.at[int(p),'heldout_folds'],holdout_fold):continue
   pool=np.concatenate([[p],neg]);scores=pd.to_numeric(meta.loc[pool,'broad_score']).to_numpy(float);order=np.argsort(-scores,kind='stable')[:32];local=pool[order]
   if int(p) not in set(map(int,local)):continue
   local_neg=[int(r) for r in local if int(r)!=int(p) and int(meta.at[int(r),'Label'])==0]
   if not local_neg:continue
   pos_i.extend([int(p)]*len(local_neg));neg_i.extend(local_neg);episodes+=1;queries.add(q)
 return np.asarray(pos_i,np.int64),np.asarray(neg_i,np.int64),{'episodes':episodes,'queries':len(queries),'pairwise_pairs':len(pos_i)}

def train(meta,x,device,holdout_fold=None):
 pi,ni,audit=build_loo_pairs(meta,holdout_fold);used=np.unique(np.concatenate([pi,ni]));mean=x[used].mean(0).astype(np.float32);std=np.maximum(x[used].std(0).astype(np.float32),1e-5);xt=torch.as_tensor((x-mean)/std,dtype=torch.float32,device=device);p=torch.as_tensor(pi,dtype=torch.long,device=device);n=torch.as_tensor(ni,dtype=torch.long,device=device);torch.manual_seed(SEED+(holdout_fold or 0));w=torch.nn.Parameter(torch.zeros(x.shape[1],device=device));opt=torch.optim.AdamW([w],lr=LR,weight_decay=WD);gen=torch.Generator(device='cpu').manual_seed(SEED+73+(holdout_fold or 0));losses=[]
 for ep in range(EPOCHS):
  perm=torch.randperm(len(pi),generator=gen);tot=0.0
  for st in range(0,len(pi),BATCH):
   sel=perm[st:st+BATCH].to(device);margin=(xt.index_select(0,p.index_select(0,sel))-xt.index_select(0,n.index_select(0,sel)))@w;loss=torch.nn.functional.softplus(-margin).mean();opt.zero_grad();loss.backward();opt.step();tot+=float(loss.item())*len(sel)
  losses.append(tot/len(pi))
 model={'mean':mean,'std':std,'weight':w.detach().cpu().numpy().astype(np.float32)};audit.update({'loss_first':losses[0],'loss_last':losses[-1],'epochs':EPOCHS});return model,audit

def apply(model,x):return (((x-model['mean'])/model['std'])@model['weight']).astype(np.float64)
def save(model,path):np.savez(path,mean=model['mean'],std=model['std'],weight=model['weight'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();device=torch.device(a.device);OUT.mkdir(parents=True,exist_ok=True);train_meta,raw_train=v2.load_train();evalraw={f:v2.load_eval(f'fold{f}') for f in (0,1,2)};index=FibreCandidateIndex(device=a.device);dev={};cand=[]
 for mode in MODES:
  tx=transform(raw_train,mode);m0,a0=train(train_meta,tx,device,0);m1,a1=train(train_meta,tx,device,1);e0,x0=evalraw[0];e1,x1=evalraw[1];b0,v0=v2.eval_fold(0,e0,apply(m0,transform(x0,mode)),index);b1,v1=v2.eval_fold(1,e1,apply(m1,transform(x1,mode)),index);dev[mode]=(a0,a1,b0,v0,b1,v1)
  for k in v0:
   g0=v0[k]['mrr']-b0['mrr'];g1=v1[k]['mrr']-b1['mrr'];ok=g0>=0 and g1>=0 and (g0+g1)>0 and v0[k]['hit10']>=b0['hit10'] and v1[k]['hit10']>=b1['hit10']
   if ok:cand.append(((g0+g1)/2,mode,k))
 sel=max(cand,key=lambda t:t[0]) if cand else None;result={'schema':'bridge-pocket-loo-expert-v6','training':'deterministic leave-one-positive-out local episodes: target positive plus Broad-hard scoreable negatives, top32 after masking all other known positives','modes':list(MODES),'selection':'fold0/1 only; exact heldout relations removed from episode training; fold2 untouched; outer unused','selected':None,'confirmed':False}
 if sel:
  _,mode,k=sel;a0,a1,b0,v0,b1,v1=dev[mode];result['selected']={'mode':mode,'alpha':k[0],'protected_prefix':k[1]};result['fold0']={'train':a0,'broad':b0,'selected':v0[k]};result['fold1']={'train':a1,'broad':b1,'selected':v1[k]};tx=transform(raw_train,mode);m2,a2=train(train_meta,tx,device,2);e2,x2=evalraw[2];b2,v2m=v2.eval_fold(2,e2,apply(m2,transform(x2,mode)),index,k);result['fold2']={'train':a2,'broad':b2,'selected':v2m[k]};confirmed=v2m[k]['mrr']>b2['mrr'] and v2m[k]['hit10']>=b2['hit10'];result['confirmed']=bool(confirmed)
  if confirmed:
   mf,af=train(train_meta,tx,device,None);save(mf,OUT/'model.npz');om,ox=v2.load_eval('outer_max');ob,oe,act=v2.eval_outer(om,apply(mf,transform(ox,mode)),index,k);result['final_train']=af;result['outer']={'broad':ob,'broad_plus_pocket_loo':oe,'delta_mrr':oe['mrr']-ob['mrr'],'delta_hit10_pp':100*(oe['hit10']-ob['hit10']),'delta_hit20_pp':100*(oe['hit20']-ob['hit20']),'delta_hit100_pp':100*(oe['hit100']-ob['hit100']),'active_queries':int(act.sum()),'active_fraction':float(act.mean())}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
