from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from projects.active.bridge.evidence.pair_scores import ClipzymePairEvidence, enzgfm_pair_evidence

TARGETS=ROOT/'results/bridge_relation_unseen_max_v3/targets.csv'
TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
BUNDLE=ROOT/'projects/active/bridge/release/manifests/score_evidence_v1/e2r_bundle.json'
CORE_CAL=ROOT/'projects/active/bridge/release/manifests/score_evidence_v1/e2r_core_calibration.json'
OUT=ROOT/'results/bridge_e2r_relation_unseen_max_v3'


def metrics(frame:pd.DataFrame,col:str)->dict:
 r=frame[col].to_numpy(np.int64)
 return {'mrr':float((1.0/r).mean()),'hit10':float((r<=10).mean()),'hit100':float((r<=100).mean()),'hit1000':float((r<=1000).mean())}


def load_targets():
 train=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id'])
 known=defaultdict(set)
 for p,r in train[['protein_id','reaction_id']].itertuples(index=False): known[str(p)].add(str(r))
 frame=pd.read_csv(TARGETS,dtype=str).fillna('')
 positives=frame.groupby('protein_id').reaction_id.apply(lambda x:set(x.astype(str))).to_dict()
 return positives,known


def calibrated(raw:np.ndarray,available:np.ndarray,center:float,scale:float)->np.ndarray:
 out=np.zeros(len(raw),dtype=np.float64)
 if available.any(): out[available]=(raw[available]-float(center))/max(float(scale),1e-8)
 return out


def rank_best(score:np.ndarray,target_rows:list[int],lex:np.ndarray)->int:
 tr=np.asarray(target_rows,dtype=np.int64); vals=score[tr]; best=float(vals.max()); br=tr[vals==best]; row=int(br[np.argmin(lex[br])])
 return int(np.count_nonzero(score>best)+np.count_nonzero((score==best)&(lex<lex[row]))+1)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 bundle=json.loads(BUNDLE.read_text()); corecal=json.loads(CORE_CAL.read_text())
 members={m['descriptor']['name']:m for m in bundle['members']}
 clipm=members['clipzyme_structure']; funcm=members['enzgfm_e2r']
 positives,known=load_targets(); queries=sorted(positives)
 index=FibreCandidateIndex(device=a.device); functional=enzgfm_pair_evidence('e2r',device=a.device); clip=ClipzymePairEvidence(device=a.device)
 rids=index.reaction_ids; ridx=index.reaction_index; lex=np.empty(len(rids),dtype=np.int64); o=np.argsort(np.asarray(rids,dtype=object),kind='stable');lex[o]=np.arange(len(o))
 f_r=np.asarray([functional.r_index.get(r,-1) for r in rids],dtype=np.int64); c_r=np.asarray([clip.r_index.get(r,-1) for r in rids],dtype=np.int64)
 rows=[]; qa=[]
 for st in range(0,len(queries),a.batch_size):
  qs=queries[st:st+a.batch_size]; valid=[q for q in qs if q in index.protein_index]
  if not valid:continue
  qrows=torch.as_tensor([index.protein_index[q] for q in valid],dtype=torch.long,device=index.device)
  with torch.no_grad(): broad=(index.protein_embeddings.index_select(0,qrows)@index.reaction_embeddings.T).float().cpu().numpy()
  for j,q in enumerate(valid):
   core=(broad[j].astype(np.float64)-float(corecal['center']))/float(corecal['scale'])
   # EnzGFM E2R evidence.
   fp=functional.p_index.get(q,-1); fa=f_r>=0; fraw=np.zeros(len(rids),dtype=np.float64)
   if fp<0: fa[:]=False
   elif fa.any():
    rt=torch.as_tensor(f_r[fa],dtype=torch.long,device=index.device)
    with torch.no_grad(): fraw[fa]=(functional.r.index_select(0,rt)@functional.p[fp]).float().cpu().numpy()
   fcal=calibrated(fraw,fa,funcm['calibration']['score_center'],funcm['calibration']['score_scale'])
   # CLIPZyme E2R evidence.
   cp=clip.p_index.get(q,-1); ca=c_r>=0; craw=np.zeros(len(rids),dtype=np.float64)
   if cp<0 or not bool(clip.p_supported[cp]): ca[:]=False
   else: ca &= clip.r_supported[np.maximum(c_r,0)]
   if ca.any():
    rt=torch.as_tensor(c_r[ca],dtype=torch.long,device=index.device)
    with torch.no_grad(): craw[ca]=(clip.r_device.index_select(0,rt)@clip.p_device[cp]).float().cpu().numpy()
   ccal=calibrated(craw,ca,clipm['calibration']['score_center'],clipm['calibration']['score_scale'])
   full=core+float(funcm['strength'])*fcal+float(clipm['strength'])*ccal
   variants={'full':full,'minus_functional_homology':core+float(clipm['strength'])*ccal,'minus_structural_mechanistic':core+float(funcm['strength'])*fcal}
   known_rows=np.asarray([ridx[r] for r in known.get(q,set()) if r in ridx],dtype=np.int64)
   for score in variants.values():
    if len(known_rows): score[known_rows]=-np.inf
   qa.append({'query_id':q,'known_relations_masked':int(len(known_rows)),'functional_support':float(fa.mean()),'structural_support':float(ca.mean())})
   pos=positives[q]
   target=[ridx[r] for r in pos if r in ridx]
   if target:
    rec={'query_id':q,'target_relations':len(target)}
    for name,score in variants.items(): rec[name+'_best_rank']=rank_best(score,target,lex)
    rows.append(rec)
  print('e2r',min(st+a.batch_size,len(queries)),'/',len(queries),flush=True)
 frame=pd.DataFrame(rows); qf=pd.DataFrame(qa); frame.to_csv(OUT/'query_metrics.csv',index=False);qf.to_csv(OUT/'query_audit.csv',index=False)
 cols={'full_system':'full_best_rank','minus_functional_homology':'minus_functional_homology_best_rank','minus_structural_mechanistic':'minus_structural_mechanistic_best_rank'}
 overall={k:metrics(frame,v) for k,v in cols.items()};  summary={'schema':'bridge-e2r-relation-unseen-max-v3','status':'completed','direction':'e2r','target_policy':'all maximal fair-pool relations absent from clean2023; no enzyme-seen or reaction-seen restriction; all train-known relations for a query masked from candidate output','expert_surface':{'functional_homology':'EnzGFM zero-shot member; seed/homology evaluated separately in relation-context experiment','structural_mechanistic':'CLIPZyme structural evidence','domain_specialists':'not ablated per requested scope'},'queries':int(frame.query_id.nunique()),'relation_unseen_queries_evaluated':int(len(frame)),'overall':overall,'source_bundle':str(BUNDLE.relative_to(ROOT))}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
