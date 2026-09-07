from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks, summarize_query_metrics
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,load_protein_library,load_registered_reaction_feature_library

PROTOCOL=ROOT/'projects/active/terpene_screening/CATALYST_TPS_FOUNDATION_R2E_V1.json'
CACHE=ROOT/'data/terpene_marts_adaptation'
FEATURES=ROOT/'data/terpene_tps_foundation_v1'
OUT=ROOT/'results/tps_foundation_r2e_v1'
TRAINER=ROOT/'projects/active/terpene_screening/train_cleanroom_rhea_retriever.py'

PROTEIN_DIRS={'esmc':FEATURES/'esmc','enzgfm':FEATURES/'enzgfm','equalblock':FEATURES/'equalblock'}
REACTION_DIRS={'base2115':FEATURES/'reaction_2115','rdkitplus3139':FEATURES/'reaction_3139_rdkitplus'}


def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()


def load_protocol()->dict:
 p=json.loads(PROTOCOL.read_text())
 if p['status']!='frozen_before_new_foundation_feature_development_scores': raise RuntimeError('unexpected protocol status')
 if len(p['candidates'])!=7: raise RuntimeError('candidate grid changed')
 return p


def bool_col(s:pd.Series)->pd.Series:
 return s.astype(str).str.lower().eq('true')


def development_cells()->list[tuple[int,int]]:
 return [(p,r) for p in range(5) for r in range(5) if p==4 or r==4]


def frozen_cells()->list[tuple[int,int]]:
 return [(p,r) for p in range(4) for r in range(4)]


def prepare_partitions(out:Path)->dict:
 p=load_protocol(); out.mkdir(parents=True,exist_ok=True); part=out/'partitions'; part.mkdir(exist_ok=True)
 pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
 pairs['protein_fold']=pd.to_numeric(pairs.protein_fold).astype(int); pairs['reaction_fold']=pd.to_numeric(pairs.reaction_fold).astype(int)
 pairs['protein_seen_bool']=bool_col(pairs.protein_seen); pairs['reaction_seen_bool']=bool_col(pairs.reaction_seen)
 canonical=pairs.rename(columns={'Entry':'protein_id','rhea_id':'reaction_id'})
 canonical[['protein_id','reaction_id']].drop_duplicates().to_csv(part/'all_pairs.csv',index=False)
 audit=[]
 for pf,rf in development_cells():
  sid=f'p{pf}_r{rf}'
  train=canonical[(canonical.protein_fold!=pf)&(canonical.reaction_fold!=rf)][['protein_id','reaction_id']].drop_duplicates()
  cell=canonical[(canonical.protein_fold==pf)&(canonical.reaction_fold==rf)]
  # Match the strongest historical TPS R2E reference: both entities are outside the old current-library support.
  dev=cell[(~cell.protein_seen_bool)&(~cell.reaction_seen_bool)][['protein_id','reaction_id']].drop_duplicates()
  if dev.empty: raise RuntimeError(f'empty development cell {sid}')
  po=len(set(train.protein_id)&set(dev.protein_id)); ro=len(set(train.reaction_id)&set(dev.reaction_id))
  if po or ro: raise RuntimeError(f'{sid} is not strict double cold: protein={po} reaction={ro}')
  train.to_csv(part/f'{sid}_train.csv',index=False); dev.to_csv(part/f'{sid}_dev.csv',index=False)
  audit.append({'split_id':sid,'train_pairs':len(train),'dev_pairs':len(dev),'dev_queries':dev.reaction_id.nunique(),'dev_positive_proteins':dev.protein_id.nunique(),'protein_overlap':po,'reaction_overlap':ro})
 pd.DataFrame(audit).to_csv(part/'development_partition_audit.csv',index=False)
 result={'status':'ready','protocol_sha256':sha256_file(PROTOCOL),'split_source_sha256':sha256_file(CACHE/'marts_pair_folds.csv'),'development_cells':len(audit),'development_query_cells':int(sum(x['dev_queries'] for x in audit)),'all_strict_double_cold':all(x['protein_overlap']==0 and x['reaction_overlap']==0 for x in audit)}
 (part/'manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result


def candidate_spec(cid:str)->dict:
 p=load_protocol(); rows=[x for x in p['candidates'] if x['id']==cid]
 if len(rows)!=1: raise KeyError(cid)
 return rows[0]


def trainer_command(cid:str,sid:str,out:Path,device:str)->list[str]:
 p=load_protocol(); c=candidate_spec(cid); t=p['trainer']; part=out/'partitions'; model_out=out/'models'/cid/sid
 return [str(ROOT/'.venv/bin/python'),str(TRAINER),'--associations-csv',str(part/'all_pairs.csv'),'--fixed-training-pairs',str(part/f'{sid}_train.csv'),'--fixed-dev-pairs',str(part/f'{sid}_dev.csv'),'--universe-dir',str(FEATURES),'--schema-dir',str(REACTION_DIRS[c['reaction']]),'--protein-feature-dir',str(PROTEIN_DIRS[c['protein']]),'--reaction-feature-dir',str(REACTION_DIRS[c['reaction']]),'--output-dir',str(model_out),'--epochs',str(t['epochs']),'--steps-per-epoch',str(t['steps_per_epoch']),'--reaction-batch-size',str(t['reaction_batch_size']),'--protein-batch-size',str(t['protein_batch_size']),'--hard-negatives',str(t['hard_negatives']),'--random-negatives',str(t['random_negatives']),'--hidden-dim',str(t['hidden_dim']),'--embedding-dim',str(t['embedding_dim']),'--dropout',str(t['dropout']),'--learning-rate',str(t['learning_rate']),'--weight-decay',str(t['weight_decay']),'--temperature',str(t['temperature']),'--topk',str(t['topk']),'--topk-weight',str(t['topk_weight']),'--margin',str(t['margin']),'--r2e-weight',str(t['r2e_weight']),'--reaction-novelty-threshold',str(t['reaction_novelty_threshold']),'--reaction-novelty-repeat',str(c['replay']),'--seed',str(t['seed']),'--device',device]


def full_eval(cid:str,sid:str,out:Path,device_name:str)->dict:
 c=candidate_spec(cid); model_out=out/'models'/cid/sid; eval_out=out/'development_eval'/cid/sid; eval_out.mkdir(parents=True,exist_ok=True)
 dev=pd.read_csv(out/'partitions'/f'{sid}_dev.csv',dtype=str).fillna('')
 protein_features,protein_ids=load_protein_library(PROTEIN_DIRS[c['protein']]); reaction_features,reaction_ids=load_registered_reaction_feature_library(REACTION_DIRS[c['reaction']],load_feature_schema(model_out))
 pidx={x:i for i,x in enumerate(protein_ids)}; ridx={x:i for i,x in enumerate(reaction_ids)}
 missing_p=sorted(set(dev.protein_id)-set(pidx)); missing_r=sorted(set(dev.reaction_id)-set(ridx))
 if missing_p or missing_r: raise RuntimeError(f'missing eval entities p={missing_p[:3]} r={missing_r[:3]}')
 device=torch.device(device_name); models=load_models(model_out/'models','production',device)
 if len(models)!=1: raise RuntimeError('expected one model')
 model=models[0]; pe=encode_chunks(model,protein_features,kind='protein',device=device,chunk_size=4096); re=encode_chunks(model,reaction_features,kind='reaction',device=device,chunk_size=4096)
 lexical=np.argsort(np.argsort(np.asarray(protein_ids,dtype=object),kind='stable'),kind='stable')
 pos=dev.groupby('reaction_id').protein_id.apply(lambda x:set(map(str,x))).to_dict(); rows=[]
 for q in sorted(pos):
  with torch.no_grad(): scores=(re[ridx[q]]@pe.T).float().cpu().numpy()
  order=np.lexsort((lexical,-scores)); inv=np.empty(len(order),dtype=np.int64); inv[order]=np.arange(1,len(order)+1)
  ranks=np.asarray(sorted(int(inv[pidx[x]]) for x in pos[q]),dtype=np.int64)
  m=evaluate_full_candidate_ranks(ranks,len(protein_ids)); rows.append({'split_id':sid,'query_id':q,'n_positives':len(ranks),**m})
 frame=pd.DataFrame(rows); frame.to_csv(eval_out/'query_metrics.csv',index=False)
 s=summarize_query_metrics(frame); summary={'candidate_id':cid,'split_id':sid,'query_count':len(frame),'candidate_count':len(protein_ids),'mrr':float(s['mrr']),'map':float(s['map']),'ndcg_at_10':float(s['ndcg_at_10']),'hit_at_3':float(s['hit_at_3']),'hit_at_10':float(s['hit_at_10']),'hit_at_20':float(s['hit_at_20']),'median_best_positive_rank':float(s['median_best_positive_rank'])}
 (eval_out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); return summary


def run_development(out:Path,device:str)->None:
 if not (out/'partitions/manifest.json').is_file(): print(json.dumps(prepare_partitions(out),indent=2),flush=True)
 p=load_protocol()
 for c in p['candidates']:
  cid=c['id']
  for pf,rf in development_cells():
   sid=f'p{pf}_r{rf}'; model_out=out/'models'/cid/sid; eval_summary=out/'development_eval'/cid/sid/'summary.json'
   if eval_summary.is_file(): print(f'SKIP {cid} {sid}',flush=True); continue
   if not (model_out/'summary.json').is_file():
    print(f'TRAIN {cid} {sid}',flush=True); subprocess.run(trainer_command(cid,sid,out,device),cwd=ROOT,check=True)
   print(f'EVAL {cid} {sid}',flush=True); print(json.dumps(full_eval(cid,sid,out,device),sort_keys=True),flush=True)
 summarize_development(out)


def summarize_development(out:Path)->dict:
 p=load_protocol(); records=[]; candidate_frames={}
 for c in p['candidates']:
  cid=c['id']; frames=[]
  for pf,rf in development_cells():
   sid=f'p{pf}_r{rf}'; q=out/'development_eval'/cid/sid/'query_metrics.csv'
   if not q.is_file(): raise RuntimeError(f'missing development eval {cid}/{sid}')
   frames.append(pd.read_csv(q))
  f=pd.concat(frames,ignore_index=True); candidate_frames[cid]=f; s=summarize_query_metrics(f)
  records.append({'candidate_id':cid,'query_cells':len(f),'mrr':float(s['mrr']),'map':float(s['map']),'ndcg_at_10':float(s['ndcg_at_10']),'hit_at_3':float(s['hit_at_3']),'hit_at_10':float(s['hit_at_10']),'hit_at_20':float(s['hit_at_20']),'median_best_positive_rank':float(s['median_best_positive_rank'])})
 table=pd.DataFrame(records).sort_values(['hit_at_10','mrr','ndcg_at_10','candidate_id'],ascending=[False,False,False,True]).reset_index(drop=True); table.to_csv(out/'development_candidates.csv',index=False)
 gate=float(p['development_evaluation']['existing_reference_hit10'])+float(p['development_evaluation']['minimum_hit10_absolute_gain_to_open_legacy_frozen']); best=table.iloc[0].to_dict(); passed=bool(float(best['hit_at_10'])>=gate-1e-12)
 result={'status':'passed_development_gate_legacy_frozen_allowed' if passed else 'failed_development_gate','pass':passed,'selected_candidate':str(best['candidate_id']) if passed else None,'best_candidate':str(best['candidate_id']),'reference_hit_at_10':float(p['development_evaluation']['existing_reference_hit10']),'required_hit_at_10':gate,'best_metrics':{k:(int(v) if k=='query_cells' else float(v)) for k,v in best.items() if k!='candidate_id'},'candidate_table':'results/tps_foundation_r2e_v1/development_candidates.csv','legacy_frozen_scores_read':False,'v2_1_scores_read':False,'retuning_allowed':False}
 (out/'development_result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return result


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','development','summarize']); ap.add_argument('--output-dir',type=Path,default=OUT); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); out=a.output_dir.resolve()
 if a.action=='prepare': print(json.dumps(prepare_partitions(out),indent=2))
 elif a.action=='development': run_development(out,a.device)
 else: summarize_development(out)
if __name__=='__main__': main()
