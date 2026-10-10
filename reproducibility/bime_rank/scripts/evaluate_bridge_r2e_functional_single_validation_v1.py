"""R2E functional-expert reliability study on ONE shared 5,216-positive validation.

All frozen neural evidence and 2023 relations unchanged. The R2E relation
authority is the newly 5,216-only refitted exploratory gate; both the old
and updated relationship policies are identifiable. No final test target
labels are read in this script. All functional multipliers prespecified.
Quantify Direct and degree-novelty Balanced MRR/Hit@10/Hit@100 and group
query loss. Invalid/absent functional features imply exactly 0 contribution.
"""
from __future__ import annotations
import json,math,pickle,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime,_z
from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import balanced
from reproducibility.bime_rank.scripts import evaluate_bridge_r2e_four_group_ablation_v1 as old
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

VALID=ROOT/'results/bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz'
OUT=ROOT/'results/bridge_single_validation_v1'
GATE=OUT/'r2e_relation_gate.exploratory.pkl'
RANKS=OUT/'r2e_functional_validation_edges.csv.gz'
AUDIT=OUT/'r2e_functional_query_audit.csv.gz'
SUMMARY=OUT/'r2e_functional_validation_summary.json'
BETAS=(0.,.25,.5,.75,1.,1.25)
FEATURES=('availability','router_functional_weight','router_geometry_weight',
          'functional_core_corr','functional_geometry_corr','functional_top10_overlap',
          'query_degree','relation_weight','functional_top10_spread')

def stats(a,b):
 if len(a)<3 or a.std()<1e-8 or b.std()<1e-8:return 0.
 return float(np.corrcoef(a,b)[0,1])

def filtered(arr):
 x=np.asarray(arr,dtype=np.int64)
 return x-np.asarray([np.count_nonzero(x<x[i]) for i in range(len(x))],dtype=np.int64)

def query_features(broad,func,fa,geom,ga,fw,gw,rw,q_degree):
 indices=np.flatnonzero(fa)
 if len(indices):
  best_func=indices[np.argsort(-func[indices],kind='stable')[:10]]
  best_broad=set(np.argsort(-broad,kind='stable')[:10])
  overlap=len(best_broad.intersection(best_func))/10.
  s=np.sort(func[indices])
  spread=float(s[-1]-s[max(0,len(s)-10)])
 else:
  overlap=0.;spread=0.
 both=fa&ga
 return np.array([float(np.mean(fa)),float(fw),float(gw),
          stats(broad[fa],func[fa]) if fa.any() else 0.,
          stats(func[both],geom[both]) if both.any() else 0.,
          overlap,math.log1p(q_degree),float(rw),spread],dtype=np.float64)

@torch.no_grad()
def extraction():
 pairs=pd.read_csv(VALID,dtype={'protein_id':str,'reaction_id':str})
 assert len(pairs)==5216
 qgroups={q:g.copy() for q,g in pairs.groupby('reaction_id',sort=True)}
 rt=FinalBridgeRuntime(device='cuda')
 with GATE.open('rb') as fp:asset=pickle.load(fp)
 rt.relation_gate['directions']['r2e']=asset
 print('R2E_UNIFIED_SINGLE_VALID_FUNCTIONAL_BEGIN',len(qgroups),'queries',len(pairs),'relations',flush=True)
 allrows=[];qmeta=[];missing=0;t=time.time()
 for ii,(q,g) in enumerate(qgroups.items(),1):
  pi=rt.index.reaction_index.get(q)
  if pi is None:
   raise RuntimeError('validation reaction not in released Broad space '+q)
  target=list(g.protein_id)
  tid=np.array([rt.index.protein_index.get(p,-1) for p in target],np.int64)
  if (tid<0).any():raise RuntimeError('missing validation protein '+q+' '+str([target[i] for i in np.flatnonzero(tid<0)]))
  broad=(rt.index.reaction_embeddings[pi]@rt.index.protein_embeddings.T).float().cpu().numpy().astype(np.float64)
  bstd=max(float(broad.std()),1e-8)
  bz=(broad-float(broad.mean()))/bstd
  ctx,ca0,cb0,available,both=rt._relation_components('r2e',q)
  rw,_,_=rt._relation_authority('r2e',q,bz,ctx,ca0,cb0,available,both)
  retrieval=broad+bstd*rw*ctx
  masked=np.array([rt.index.protein_index[p] for p in rt.known_by_reaction.get(q,set()) if p in rt.index.protein_index],np.int64)
  if len(masked):retrieval[masked]=-np.inf
  k=min(1000,int(np.isfinite(retrieval).sum()))
  score_t=torch.as_tensor(retrieval,dtype=torch.float32,device=rt.index.device)
  top=torch.topk(score_t,k=k,largest=True,sorted=True).indices.cpu().numpy()
  candidates=[rt.index.protein_ids[int(row)] for row in top]
  core=broad[top]
  core_std=max(float(core.std()),1e-6)
  core_z=(core-float(core.mean()))/core_std
  relation_local=rw*(bstd/core_std)*ctx[top]

  fr=rt.functional_r2e.r_index.get(q,-1); frows=rt._r2e_functional_p_row[top]
  fa=frows>=0
  fraw=np.zeros(k,np.float64)
  if fr<0:fa[:]=False
  elif fa.any():
   ft=torch.as_tensor(frows[fa],dtype=torch.long,device=rt.index.device)
   fraw[fa]=(rt.functional_r2e.p.index_select(0,ft)@rt.functional_r2e.r[fr]).float().cpu().numpy()
  fz=_z(fraw,fa)

  crows=rt._clip_p_row[top];cr=rt.clip.r_index.get(q,-1);ca=crows>=0
  if cr>=0 and bool(rt.clip.r_supported[cr]):ca &= rt.clip.p_supported[np.maximum(crows,0)]
  else:ca[:]=False
  craw=np.zeros(k,np.float64)
  if ca.any():
   ct=torch.as_tensor(crows[ca],dtype=torch.long,device=rt.index.device)
   craw[ca]=(rt.clip.p_device.index_select(0,ct)@rt.clip.r_device[cr]).float().cpu().numpy()
  cz=_z(craw,ca)

  mrows=rt._mechanism_p_row[top];mr=rt.mechanism.r_index.get(q,-1);ma=mrows>=0
  mraw=np.zeros(k,np.float64)
  if mr<0:ma[:]=False
  elif ma.any():
   mt=torch.as_tensor(mrows[ma],dtype=torch.long,device=rt.index.device)
   mraw[ma]=(rt.mechanism.p.index_select(0,mt)@rt.mechanism.r[mr]).float().cpu().numpy()
  mz=_z(mraw,ma)
  ga=ca|ma
  geom=np.zeros(k,np.float64)
  div=ca.astype(int)+ma.astype(int)
  geom[ca]+=cz[ca];geom[ma]+=mz[ma]
  geom[ga]/=div[ga]
  fw,gw=rt._r2e_general_weights(q,core,fz,fa,geom,ga)
  # Maintain the uncalibrated evidence grid as the frozen reference.
  from projects.active.bridge.runtime.final_system import R2E_FUNCTIONAL_VALIDATION_SCALE
  fw/=R2E_FUNCTIONAL_VALIDATION_SCALE
  special,_=rt._r2e_specialists(q,candidates)
  base=core_z+relation_local+gw*geom+special
  selected={int(x):i for i,x in enumerate(top)}
  broad_target_ranks=np.zeros(len(tid),np.int64)
  for j,tr in enumerate(tid):
   if int(tr) not in selected:
    broad_target_ranks[j]=old.rank_full(retrieval,int(tr),rt._protein_lex)
  t_ranks={}
  for beta in BETAS:
   score=base+float(beta)*fw*fz
   order=rt._pocket_reorder(q,candidates,score)
   lr={int(top[pos]):r+1 for r,pos in enumerate(order)}
   raw=[lr.get(int(tr),int(broad_target_ranks[j])) for j,tr in enumerate(tid)]
   t_ranks[beta]=filtered(raw)
  qfeat=query_features(core_z,fz,fa,geom,ga,fw,gw,rw,
                       len(rt.known_by_reaction.get(q,set())))
  meta={'query_id':q,'target_count':len(g)}
  for key,value in zip(FEATURES,qfeat):meta[key]=float(value)
  meta['main_novelty']=str(g.novelty.mode().iloc[0])
  qmeta.append(meta)
  for j,item in enumerate(g.to_dict('records')):
   record={'protein_id':item['protein_id'],'reaction_id':q,
           'novelty':item['novelty'],'difficulty_stratum':item['difficulty_stratum']}
   for beta in BETAS:
    record['rank_beta_'+str(beta).replace('.','p')]=int(t_ranks[beta][j])
   allrows.append(record)
  if ii%100==0 or ii==len(qgroups):
   frame=pd.DataFrame(allrows)
   temp=RANKS.with_name('r2e_functional_validation_edges.tmp.csv.gz')
   frame.to_csv(temp,index=False);temp.replace(RANKS)
   pd.DataFrame(qmeta).to_csv(AUDIT,index=False)
   print('R2E_UNIFIED_FUN_VALID_PROGRESS',ii,len(qgroups),'pos',len(frame),'elapsed',round(time.time()-t,1),flush=True)
 assert len(allrows)==len(pairs)
 return pd.DataFrame(allrows),pd.DataFrame(qmeta)

def audit():
 df=pd.read_csv(RANKS,dtype={'protein_id':str,'reaction_id':str})
 meta=pd.read_csv(AUDIT,dtype={'query_id':str})
 assert len(df)==5216 and meta.query_id.nunique()==990
 result={'schema':'BRIDGE_R2E_SINGLE_5216_VALIDATION_FUNCTIONAL_RELIABILITY_V1',
 'validation_relations':5216,'validation_reaction_queries':990,
 'model_change':'no weights promoted; frozen new 5216-only relation gate in simulation',
 'fixed_functional_scalars':list(BETAS),'per_beta':{},'fivefold_oof':{}}
 for beta in BETAS:
  name='rank_beta_'+str(beta).replace('.','p')
  result['per_beta'][str(beta)]=balanced(df,name)
  print('FUNC_BETA',beta,
        'direct', {k:round(v,5) for k,v in result['per_beta'][str(beta)]['direct'].items()},
        'balanced',{k:round(v,5) for k,v in result['per_beta'][str(beta)]['balanced'].items()},flush=True)
 # Analyze frozen target rank improvements per query; OOF classifier uses
 # only known positives within the 5,216 relation development source.
 rows={}
 for q,g in df.groupby('reaction_id'):
  row={'query_id':q}
  for b in BETAS:
   rank=g['rank_beta_'+str(b).replace('.','p')].to_numpy(int)
   row['u_'+str(b)]=float(np.mean(1/rank+0.2*(rank<=10)+0.04*(rank<=100)))
  rows[q]=row
 y=pd.DataFrame(rows.values()).merge(meta,on='query_id',validate='one_to_one')
 feat=y[list(FEATURES)].to_numpy(float)
 target=(y['u_1.0'].to_numpy()>y['u_0.0'].to_numpy()).astype(int)
 label=y['main_novelty'].astype(str).to_numpy()
 skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261011)
 oof_beta=np.empty(len(y),float)
 oof_class=np.empty(len(y),float)
 for tr,va in skf.split(feat,label):
  classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=500))
  classifier.fit(feat[tr],target[tr])
  probability=classifier.predict_proba(feat[va])[:,1]
  oof_class[va]=probability
  oof_beta[va]=np.asarray(BETAS)[np.abs(np.asarray(BETAS)[None,:]-probability[:,None]).argmin(axis=1)]
 map_beta=dict(zip(y.query_id,oof_beta))
 picks=[]
 for r in df.itertuples(index=False):
  p=map_beta[r.reaction_id]
  picks.append(getattr(r,'rank_beta_'+str(float(p)).replace('.','p')))
 df['functional_oof_rank']=np.asarray(picks,int)
 result['fivefold_oof']['logistic_p_help']=balanced(df,'functional_oof_rank')
 result['fivefold_oof']['label_help_fraction']=float(target.mean())
 result['fivefold_oof']['predicted_mean_beta']=float(oof_beta.mean())
 result['fivefold_oof']['fixed_beta1']=result['per_beta']['1.0']
 print('FUNCTIONAL_OOF_GATE',
       json.dumps({k:v for k,v in result['fivefold_oof'].items() if k!='fixed_beta1'},
                 ensure_ascii=False),flush=True)
 SUMMARY.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('R2E_UNIFIED_FUNCTIONAL_STUDY_FINISHED',flush=True)

if __name__=='__main__':
 if not RANKS.exists() or not AUDIT.exists():extraction()
 audit()
