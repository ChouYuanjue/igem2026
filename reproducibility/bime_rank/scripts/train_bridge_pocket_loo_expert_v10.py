from __future__ import annotations
import argparse,json
import numpy as np, torch
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import train_bridge_pocket_loo_expert_v6 as v6
from reproducibility.bime_rank.scripts import train_bridge_pocket_interaction_expert_v2 as v2

OUT=ROOT/'results/bridge_pocket_loo_expert_v10'
MODES=('raw','directed_product','agreement_plus_directed')
ALPHAS=(0.01,0.02,0.05,0.1,0.2,0.35,0.5)
PREFIX=20

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();device=torch.device(a.device);OUT.mkdir(parents=True,exist_ok=True)
 train_meta,raw_train=v2.load_train();evalraw={f:v2.load_eval(f'fold{f}') for f in (0,1,2)};index=FibreCandidateIndex(device=a.device)
 dev={};cand=[]
 for mode in MODES:
  tx=v6.transform(raw_train,mode);m0,a0=v6.train(train_meta,tx,device,0);m1,a1=v6.train(train_meta,tx,device,1)
  e0,x0=evalraw[0];e1,x1=evalraw[1]
  b0,v0=v2.eval_fold(0,e0,v6.apply(m0,v6.transform(x0,mode)),index);b1,v1=v2.eval_fold(1,e1,v6.apply(m1,v6.transform(x1,mode)),index)
  dev[mode]=(a0,a1,b0,v0,b1,v1)
  for alpha in ALPHAS:
   k=(alpha,PREFIX)
   g0=v0[k]['mrr']-b0['mrr'];g1=v1[k]['mrr']-b1['mrr']
   # Top10 is invariant by construction. Require MRR non-regression on each dev fold.
   if g0>=0 and g1>=0 and (g0+g1)>0:cand.append(((g0+g1)/2,mode,k))
 sel=max(cand,key=lambda t:t[0]) if cand else None
 result={'schema':'bridge-pocket-loo-expert-v10','training':'same relation-level leave-one-positive-out interaction expert as v6','permission':'BRIDGE Top20 protection: pocket expert may only reorder Broad ranks 21-1000','selection':'mode/alpha selected on fold0/1 only with fixed prefix=20; fold2 untouched; outer unused','selected':None,'confirmed':False}
 if sel:
  _,mode,k=sel;a0,a1,b0,v0,b1,v1=dev[mode];result['selected']={'mode':mode,'alpha':k[0],'protected_prefix':PREFIX};result['fold0']={'train':a0,'broad':b0,'selected':v0[k]};result['fold1']={'train':a1,'broad':b1,'selected':v1[k]}
  tx=v6.transform(raw_train,mode);m2,a2=v6.train(train_meta,tx,device,2);e2,x2=evalraw[2];b2,v2m=v2.eval_fold(2,e2,v6.apply(m2,v6.transform(x2,mode)),index,k);result['fold2']={'train':a2,'broad':b2,'selected':v2m[k]}
  confirmed=v2m[k]['mrr']>b2['mrr'] and v2m[k]['hit10']>=b2['hit10'];result['confirmed']=bool(confirmed)
  if confirmed:
   mf,af=v6.train(train_meta,tx,device,None);v6.save(mf,OUT/'model.npz');om,ox=v2.load_eval('outer_max');ob,oe,act=v2.eval_outer(om,v6.apply(mf,v6.transform(ox,mode)),index,k);result['final_train']=af;result['outer']={'broad':ob,'broad_plus_pocket_loo':oe,'delta_mrr':oe['mrr']-ob['mrr'],'delta_hit10_pp':100*(oe['hit10']-ob['hit10']),'delta_hit20_pp':100*(oe['hit20']-ob['hit20']),'delta_hit100_pp':100*(oe['hit100']-ob['hit100']),'active_queries':int(act.sum()),'active_fraction':float(act.mean())}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
