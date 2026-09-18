from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.runtime.cli import load_models,ensemble_similarity
from projects.active.terpene_screening.runtime.base_model import rank_metrics
C=ROOT/'data/terpene_marts_adaptation';BASE=ROOT/'results/terpene_production_models/drfp_categorical';OUT=ROOT/'results/terpene_clean_prior_field_dev_v1';BUD=(3,10,20)

def b(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def agg(q):
 rows=[]
 for d,g in q.groupby('direction'):
  x={'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:x[f'hit{k}']=g[f'hit_at_{k}'].mean();x[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  rows.append(x)
 return pd.DataFrame(rows)
def main():
 p=pd.read_csv(C/'protein_entities.csv',dtype=str).fillna('');r=pd.read_csv(C/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(C/'marts_pair_folds.csv',dtype=str).fillna('')
 pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=b(pairs.protein_seen);pairs['reaction_seen']=b(pairs.reaction_seen)
 pids=p.protein_id.tolist();rids=r.reaction_id.tolist();P=np.load(C/'protein_features.npy').astype(np.float32);R=np.load(C/'reaction_features.npy').astype(np.float32)
 assert P.shape==(len(pids),1152) and R.shape==(len(rids),2115)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');models=load_models(BASE/'models','production',device);F=ensemble_similarity(models,P,R,device)
 assert F.shape==(len(rids),len(pids))
 np.save(OUT/'score_matrix.npy',F) if OUT.exists() else None
 rows=[];sp=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry'])
   sp.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   ri={x:i for i,x in enumerate(rids)};pi={x:i for i,x in enumerate(pids)}
   for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
   for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(sp);met=agg(q);OUT.mkdir(parents=True,exist_ok=True);np.save(OUT/'score_matrix.npy',F);q.to_csv(OUT/'query_metrics.csv',index=False);sp.to_csv(OUT/'split_summary.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 basepairs=pd.read_csv(ROOT/'data/terpene_cold_splits/positive_pair_fold_assignments.csv',dtype=str).fillna('');B=set(zip(basepairs.rhea_id,basepairs.Entry))
 dev=pairs[(pairs.protein_fold.eq(4)|pairs.reaction_fold.eq(4))&~pairs.protein_seen&~pairs.reaction_seen]
 base_rxn=set(basepairs.rhea_id.astype(str)); base_prot=set(basepairs.Entry.astype(str))
 dev_rxn=set(dev.rhea_id.astype(str)); dev_prot=set(dev.Entry.astype(str))
 summary={'version':'terpene-clean-prior-field-dev-v1','prior':'pre-MARTS drfp_categorical three-seed production ensemble','training_pairs':len(basepairs),'dev_exact_pair_overlap_with_prior_training':len(set(zip(dev.rhea_id,dev.Entry))&B),'all_marts_pair_overlap_with_prior_training':len(set(zip(pairs.rhea_id,pairs.Entry))&B),'dev_reaction_entity_overlap_with_prior_training':len(dev_rxn&base_rxn),'dev_protein_entity_overlap_with_prior_training':len(dev_prot&base_prot),'all_development_entity_overlaps_with_prior_training':bool((not (dev_rxn&base_rxn)) and (not (dev_prot&base_prot))),'score_shape':list(F.shape),'score_min':float(F.min()),'score_max':float(F.max()),'device':str(device),'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
