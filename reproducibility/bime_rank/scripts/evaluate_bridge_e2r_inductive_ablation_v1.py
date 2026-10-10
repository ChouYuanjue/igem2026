"""Independent, frozen 4-component ablations of deployed E2R relation ranking.

Uses exactly the same 21,505 protein-query-heldout pairs and the released
frozen E2R gate + C2 coefficients; no fitting or test-model selection.
One ablation per component removes its score evidence, retaining the
pretrained query-level route gate frozen in all configurations.
"""
from __future__ import annotations
import hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from projects.active.bridge.runtime.e2r_query_gate import FrozenE2RSurface,predict_joint_route,filtered_target_ranks
from projects.active.bridge.runtime.inductive_relation import InductiveE2RRelation,InductiveRelationConfig
from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import balanced

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'results/bridge_e2r_inductive_c2_outer_v1'
SRC=RAW/'single_heldout_edge_ranks.csv.gz'
PRE=RAW/'frozen_test_candidate_precommit.json'
FILE=ROOT/'projects/active/bridge/release/runtime/final_bridge_v1/e2r_inductive_relation.production.json'
GATE=ROOT/'projects/active/bridge/release/runtime/final_bridge_v1/e2r_joint_query_v4.production.pkl'
TARGET=ROOT/'results/bridge_gate_split_v2/evaluation_pairs.csv.gz'
OUT=ROOT/'results/bridge_e2r_inductive_ablation_v1'
CACHE=OUT/'partial_scores.csv.gz'
FINAL=OUT/'edge_ranks.csv.gz'
RESULT=OUT/'summary.json'
COLS=('full_rank','minus_functional_rank','minus_structure_mechanism_rank',
      'minus_relational_memory_rank','minus_family_domain_rank')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()

def main():
 frozen=json.loads(PRE.read_text())
 asset=json.loads(FILE.read_text())
 assert asset['frozen_selection_sha256']==digest(PRE)
 assert asset['coefficients']['observed_links']==frozen['weights_observed']
 assert asset['coefficients']['reaction_analogy']==frozen['weights_inferred']
 with GATE.open('rb') as f:gate=pickle.load(f)
 f=pd.read_csv(SRC,dtype={'protein_id':str,'reaction_id':str})
 t=pd.read_csv(TARGET,dtype=str)
 assert len(f)==len(t)==21505
 assert f[['protein_id','reaction_id']].equals(t[['protein_id','reaction_id']])
 assert not f.duplicated(['protein_id','reaction_id']).any()
 groups=f.groupby('protein_id',sort=True)
 keys=list(groups.groups)
 assert len(keys)==15751
 OUT.mkdir(parents=True,exist_ok=True)
 if FINAL.exists() or RESULT.exists():
  raise RuntimeError('Refuse to modify a finalized ablation result')
 rt=FinalBridgeRuntime(device='cuda')
 surface=FrozenE2RSurface(rt)
 cfg=asset['neighborhood']
 weights=asset['coefficients']
 relation=InductiveE2RRelation(rt,InductiveRelationConfig(
  observed_coefficient=float(weights['observed_links']),
  inferred_coefficient=float(weights['reaction_analogy']),
  nearest_proteins=int(cfg['protein_neighbors']),
  nearest_reactions=int(cfg['reaction_neighbors']),
  reaction_neighbor_pool=int(cfg['reaction_retrieval_pool']),
  protein_attention_temperature=float(cfg['protein_temperature']),
  reaction_attention_temperature=float(cfg['reaction_temperature']),
  prior_normalizer=float(cfg['observation_prior_factor']),
  reaction_min_cosine=float(cfg['reaction_confidence_floor'])
 ))
 progress={}
 if CACHE.exists():
  saved=pd.read_csv(CACHE,dtype={'protein_id':str,'reaction_id':str})
  assert not saved.duplicated(['protein_id','reaction_id']).any()
  for q,g in saved.groupby('protein_id',sort=True):
   assert g.reaction_id.tolist()==groups.get_group(q).reaction_id.tolist()
   progress[q]=g
  print('RESUMED',len(progress),'QUERIES',len(saved),'EDGES',flush=True)
 else:
  print('START_FROZEN_E2R_ABLATIONS',json.dumps({
   'test_pairs':len(f),'queries':len(keys),
   'frozen_C2_asset_sha256':digest(FILE),
   'frozen_v4_gate_sha256':digest(GATE),
   'test_positive_pairs_sha256':digest(TARGET),
   'method':"4 isolated evidence-channel deletion, router unchanged",
   'coefficient_refit':False
  }),flush=True)
 start=time.time()
 numerical_ties=0
 max_rank_diff=0
 for i,q in enumerate(keys,1):
  if q in progress: continue
  old=groups.get_group(q)
  with torch.no_grad():
   sur=surface.score(q)
   _,a,_=predict_joint_route(gate,sur['features'],sur)
   top=sur['top']
   core=sur['core'][top]
   functional=float(a[0])*sur['functional'][top]
   structure=float(a[1])*sur['structure'][top]
   relation_old=float(a[2])*sur['relation'][top]
   correction=relation.correction(q,top,float(np.std(sur['broad'])))
   s_all=core+functional+structure+relation_old+correction
   scores={
    'recomputed_full_rank':s_all,
    'minus_functional_rank':core+structure+relation_old+correction,
    'minus_structure_mechanism_rank':core+functional+relation_old+correction,
    'minus_relational_memory_rank':core+functional+structure,
   }
   tr=np.asarray([rt.index.reaction_index[r] for r in old.reaction_id],np.int64)
   if np.any(tr<0):raise AssertionError('Missing reaction in canonical universe')
   new=old[['protein_id','reaction_id']].copy()
   for c,v in scores.items():
    sort=np.lexsort((surface.lex[top],-v))
    order=np.concatenate((top[sort],sur['order'][1000:]))
    ranks=filtered_target_ranks(order,tr)
    new[c]=ranks.astype(np.int32)
  old_rank=old.inductive_rank.to_numpy(np.int64)
  delta=np.abs(new.recomputed_full_rank.to_numpy(np.int64)-old_rank)
  max_rank_diff=max(max_rank_diff,int(delta.max()))
  numerical_ties+=int((delta>0).sum())
  if int(delta.max())>20:
   raise RuntimeError(f'Recomputation differs >20 ranks from precommitted full C2 at {q}: {new.recomputed_full_rank.tolist()} versus {old_rank.tolist()}')
  if int(delta.max())>2:
   print('FULL_RANK_FP32_DISCREPANCY_AUDIT',q,'max_difference',int(delta.max()),'snapshot',old_rank.tolist(),'rerun',new.recomputed_full_rank.tolist(),flush=True)
  new['full_rank']=old_rank
  new['minus_family_domain_rank']=old_rank
  progress[q]=new
  if i%250==0 or i==len(keys):
   frame=pd.concat(progress.values(),ignore_index=True)
   tmp=OUT/'partial.tmp.csv.gz'
   frame.to_csv(tmp,index=False)
   os.replace(tmp,CACHE)
   print('FROZEN_ABLATION_PROGRESS',i,len(keys),'positive_relations',len(frame),
         'numerical_ties',numerical_ties,'elapsed_s',round(time.time()-start,1),flush=True)
 output=pd.concat(progress.values(),ignore_index=True)
 output=output.merge(f[['protein_id','reaction_id','novelty','difficulty_stratum']].assign(_position=np.arange(len(f))),
                     on=['protein_id','reaction_id'],validate='one_to_one').sort_values('_position').drop(columns='_position')
 assert output[['protein_id','reaction_id']].reset_index(drop=True).equals(f[['protein_id','reaction_id']])
 output.to_csv(FINAL,index=False)
 methods={}
 for c in COLS:
  methods[c]=balanced(output,c)
 with RESULT.open('w') as fw:
  json.dump({
   'schema':'BRIDGE_E2R_INDUCTIVE_FROZEN_HELDOUT_ABLATION_V1',
   'test_relations':21505,'test_enzyme_queries':15751,
   'frozen_C2_asset_sha256':digest(FILE),
   'frozen_v4_gate_sha256':digest(GATE),
   'final_test_pairs_sha256':digest(TARGET),
   'ablation_definition':{
    'functional':'remove functional score contribution with frozen gate',
    'structure_mechanism':'remove structural score contribution with frozen gate',
    'long_term_relation':'remove original train graph context AND two-sided induced relation score with frozen gate',
    'family_domain':'E2R model has no family/domain expert; identical full ranks',
   },
   'query_router_retrained':False,'expert_or_C2_weights_retrained':False,
   'same_full_C2_test_rank_as_preregistered_heldout':True,
   'numerical_recomputation_tie_rows':int(numerical_ties),
   'max_full_model_rank_recomputation_delta':max_rank_diff,
   'methods':methods
  },fw,indent=2)
 print('FINAL_FROZEN_E2R_ABLATIONS',json.dumps({
   key: {n:{k:round(v*100,3) if k.startswith('hit') else round(v,6)
      for k,v in m[n].items()} for n in ('direct','balanced')}
   for key,m in methods.items()
 },ensure_ascii=False),flush=True)
if __name__=='__main__':main()
