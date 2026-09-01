from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.broad_rhea_metrics import candidate_ranking_context,evaluate_full_candidate_scores
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,load_protein_library,load_registered_reaction_feature_library
BASE=ROOT/'results/enzgfm_gate_a_bidirectional_v1/enzgfm'
CAND=ROOT/'results/enzgfm_gate_b_rdkitplus_v1/rdkitplus'
OUT=ROOT/'results/unified_safe_system_v1/e2r_full_reaction_contract_v1'

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def encode_rows(model,fn,features,rows,device,batch=4096):
 out=[]
 with torch.no_grad():
  for s in range(0,len(rows),batch):
   x=torch.from_numpy(features[np.asarray(rows[s:s+batch],dtype=np.int64)]).to(device)
   out.append(getattr(model,fn)(x).cpu().numpy().astype(np.float32))
 return np.concatenate(out,axis=0)
def aggregate(q:pd.DataFrame)->dict[str,float|int]:
 return {'n_queries':int(len(q)),'mrr':float(q.reciprocal_rank.mean()),'map':float(q.average_precision.mean()),'macro_roc_auc':float(q.roc_auc.mean()),'ndcg_at_10':float(q.ndcg_at_10.mean()),'hit_at_1':float(q.hit_at_1.mean()),'hit_at_3':float(q.hit_at_3.mean()),'hit_at_5':float(q.hit_at_5.mean()),'hit_at_10':float(q.hit_at_10.mean()),'hit_at_20':float(q.hit_at_20.mean()),'hit_at_50':float(q.hit_at_50.mean()),'median_best_positive_rank':float(q.best_positive_rank.median())}
def load_bundle(root:Path,fold:int,device):
 d=root/f'fold{fold}'; summary=json.loads((d/'summary.json').read_text()); model=load_models(d/'models','production',device)[0]; model.eval()
 pdir=Path(summary['protein_feature_dir']); rdir=Path(summary['reaction_feature_dir']); pf,pids=load_protein_library(pdir); schema=load_feature_schema(d); rf,rids=load_registered_reaction_feature_library(rdir,schema)
 return d,summary,model,pf,pids,rf,rids
def evaluate_one(root:Path,fold:int,common_rids:list[str],dev:pd.DataFrame,device)->tuple[pd.DataFrame,dict]:
 d,summary,model,pf,pids,rf,rids=load_bundle(root,fold,device); pidx={x:i for i,x in enumerate(pids)}; ridx={x:i for i,x in enumerate(rids)}
 assert all(x in ridx for x in common_rids); assert all(x in pidx for x in dev.protein_id.unique())
 rrows=[ridx[x] for x in common_rids]; remb=encode_rows(model,'encode_reactions',rf,rrows,device)
 query_ids=sorted(dev.protein_id.unique()); qrows=[pidx[x] for x in query_ids]; pemb=encode_rows(model,'encode_proteins',pf,qrows,device)
 positives=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); cidx,lex=candidate_ranking_context(common_rids); records=[]
 batch=256
 with torch.no_grad():
  rt=torch.from_numpy(remb).to(device)
  for s in range(0,len(query_ids),batch):
   qt=torch.from_numpy(pemb[s:s+batch]).to(device); scores=(qt@rt.T).cpu().numpy()
   for j,qid in enumerate(query_ids[s:s+batch]):
    m=evaluate_full_candidate_scores(scores[j],common_rids,positives[qid],candidate_index=cidx,lexical_order=lex)
    records.append({'query_id':qid,**m})
 q=pd.DataFrame(records); meta={'model_root':str(root),'fold':fold,'checkpoint':summary['checkpoint'],'checkpoint_sha256':sha(Path(summary['checkpoint'])),'protein_feature_dir':summary['protein_feature_dir'],'reaction_feature_dir':summary['reaction_feature_dir'],'dev_pairs_sha256':sha(d/'dev_pairs.csv'),'candidate_count':len(common_rids),'query_count':len(q)}
 return q,meta
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True,choices=[0,1,2]); a=ap.parse_args(); f=a.fold; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 bd=BASE/f'fold{f}'; cd=CAND/f'fold{f}'; bdev=pd.read_csv(bd/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates(); cdev=pd.read_csv(cd/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates(); assert bdev.equals(cdev), 'baseline/candidate dev pairs differ'
 # Load only schemas/libraries to freeze exact common support before model score generation.
 bs=json.loads((bd/'summary.json').read_text()); cs=json.loads((cd/'summary.json').read_text()); _,br=load_registered_reaction_feature_library(Path(bs['reaction_feature_dir']),load_feature_schema(bd)); _,cr=load_registered_reaction_feature_library(Path(cs['reaction_feature_dir']),load_feature_schema(cd)); common=sorted(set(br)&set(cr)); assert len(common)==11081,(len(common),len(br),len(cr)); assert set(bdev.reaction_id).issubset(common)
 out=OUT/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); (out/'candidate_reactions.txt').write_text('\n'.join(common)+'\n');
 bq,bmeta=evaluate_one(BASE,f,common,bdev,device); cq,cmeta=evaluate_one(CAND,f,common,cdev,device); assert bq.query_id.tolist()==cq.query_id.tolist(); bq.to_csv(out/'baseline_query_metrics.csv',index=False); cq.to_csv(out/'candidate_query_metrics.csv',index=False)
 bm,cm=aggregate(bq),aggregate(cq); delta={k:float(cm[k]-bm[k]) for k in ['mrr','map','macro_roc_auc','ndcg_at_10','hit_at_1','hit_at_3','hit_at_5','hit_at_10','hit_at_20','hit_at_50']}; result={'fold':f,'candidate_count':len(common),'dev_pairs_sha256':sha(bd/'dev_pairs.csv'),'baseline':bm,'candidate':cm,'delta':delta,'baseline_provenance':bmeta,'candidate_provenance':cmeta,'external_test_labels_used':False,'model_training_performed':False}; (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
