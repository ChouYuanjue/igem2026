from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import train_bridge_pocket_interaction_expert_v2 as v2

OUT=ROOT/'results/bridge_pocket_interaction_expert_v3'
MODES=('agreement','agreement_plus_directed','directed_product')
EPOCHS=16;BATCH=8192;LR=3e-3;WD=1e-4;SEED=20261005

def transform(x,mode):
 x=np.asarray(x,np.float32);u=x[:,:128];v=x[:,128:256]
 un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-6);vn=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-6)
 prod=un*vn;diff=np.abs(un-vn);cos=np.sum(prod,axis=1,keepdims=True);nr=np.log(np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-6)/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-6))
 if mode=='agreement':return np.concatenate([prod,diff,cos,nr],axis=1).astype(np.float32)
 if mode=='agreement_plus_directed':return np.concatenate([un,vn,prod,diff,cos,nr],axis=1).astype(np.float32)
 return np.concatenate([prod,un-vn,cos,nr],axis=1).astype(np.float32)

def train(meta,x,device,holdout_fold=None):
 pi,ni,audit=v2.build_pairs(meta,holdout_fold);used=np.unique(np.concatenate([pi,ni]));mean=x[used].mean(0).astype(np.float32);std=np.maximum(x[used].std(0).astype(np.float32),1e-5);xt=torch.as_tensor((x-mean)/std,dtype=torch.float32,device=device);p=torch.as_tensor(pi,dtype=torch.long,device=device);n=torch.as_tensor(ni,dtype=torch.long,device=device);torch.manual_seed(SEED+(holdout_fold or 0));w=torch.nn.Parameter(torch.zeros(x.shape[1],device=device));opt=torch.optim.AdamW([w],lr=LR,weight_decay=WD);gen=torch.Generator(device='cpu').manual_seed(SEED+31+(holdout_fold or 0));losses=[]
 for ep in range(EPOCHS):
  perm=torch.randperm(len(pi),generator=gen);tot=0.0
  for st in range(0,len(pi),BATCH):
   sel=perm[st:st+BATCH].to(device);margin=(xt.index_select(0,p.index_select(0,sel))-xt.index_select(0,n.index_select(0,sel)))@w;loss=torch.nn.functional.softplus(-margin).mean();opt.zero_grad();loss.backward();opt.step();tot+=float(loss.item())*len(sel)
  losses.append(tot/len(pi))
 model={'mean':mean,'std':std,'weight':w.detach().cpu().numpy().astype(np.float32)};audit.update({'loss_first':losses[0],'loss_last':losses[-1],'epochs':EPOCHS});return model,audit

def apply(model,x):return (((x-model['mean'])/model['std'])@model['weight']).astype(np.float64)
def save(model,path):np.savez(path,mean=model['mean'],std=model['std'],weight=model['weight'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();device=torch.device(a.device);OUT.mkdir(parents=True,exist_ok=True);train_meta,raw_train=v2.load_train();evalraw={f:v2.load_eval(f'fold{f}') for f in (0,1,2)};index=FibreCandidateIndex(device=a.device)
 dev={};candidates=[]
 for mode in MODES:
  tx=transform(raw_train,mode);m0,a0=train(train_meta,tx,device,0);m1,a1=train(train_meta,tx,device,1);e0,x0=evalraw[0];e1,x1=evalraw[1];b0,v0=v2.eval_fold(0,e0,apply(m0,transform(x0,mode)),index);b1,v1=v2.eval_fold(1,e1,apply(m1,transform(x1,mode)),index);dev[mode]={'a0':a0,'a1':a1,'b0':b0,'v0':v0,'b1':b1,'v1':v1}
  for k in v0:
   g0=v0[k]['mrr']-b0['mrr'];g1=v1[k]['mrr']-b1['mrr'];ok=g0>=0 and g1>=0 and (g0+g1)>0 and v0[k]['hit10']>=b0['hit10'] and v1[k]['hit10']>=b1['hit10']
   if ok:candidates.append(((g0+g1)/2,mode,k))
 selected=max(candidates,key=lambda t:t[0]) if candidates else None
 result={'schema':'bridge-pocket-interaction-expert-v3','representation':'split bidirectional 128+128 pocket-substrate cross-attention; explicit agreement features','modes':list(MODES),'selection':'relation-heldout fold0/1 only; fold2 untouched; outer unused','selected':None,'confirmed':False}
 if selected:
  _,mode,k=selected;result['selected']={'mode':mode,'alpha':k[0],'protected_prefix':k[1]};d=dev[mode];result['fold0']={'train':d['a0'],'broad':d['b0'],'selected':d['v0'][k]};result['fold1']={'train':d['a1'],'broad':d['b1'],'selected':d['v1'][k]}
  tx=transform(raw_train,mode);m2,a2=train(train_meta,tx,device,2);e2,x2=evalraw[2];b2,v2m=v2.eval_fold(2,e2,apply(m2,transform(x2,mode)),index,k);result['fold2']={'train':a2,'broad':b2,'selected':v2m[k]};confirmed=v2m[k]['mrr']>b2['mrr'] and v2m[k]['hit10']>=b2['hit10'];result['confirmed']=bool(confirmed)
  if confirmed:
   mf,af=train(train_meta,tx,device,None);save(mf,OUT/'model.npz');om,ox=v2.load_eval('outer_max');ob,oe,act=v2.eval_outer(om,apply(mf,transform(ox,mode)),index,k);result['final_train']=af;result['outer']={'broad':ob,'broad_plus_pocket_interaction':oe,'delta_mrr':oe['mrr']-ob['mrr'],'delta_hit10_pp':100*(oe['hit10']-ob['hit10']),'delta_hit20_pp':100*(oe['hit20']-ob['hit20']),'delta_hit100_pp':100*(oe['hit100']-ob['hit100']),'active_queries':int(act.sum()),'active_fraction':float(act.mean())}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
