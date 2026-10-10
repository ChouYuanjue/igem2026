"""Train both episodic-memory gates using the *single* BRIDGE validation set.

Frozen clean2023 graph; 5,216 annotated development relations (3,454
proteins and 990 reactions); no positive relation shared with the 21,505
final test. One deterministic multi-positive support/target split per
query; all query folds are disjoint. The two long-term graph authority
gates were independently refit from the same 5,216 relations already.
Prior 16,108 relation episodic gate is never loaded.
"""
from __future__ import annotations
import hashlib,json,pickle
from pathlib import Path
import pandas as pd
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from projects.active.bridge.runtime.final_system import RELATION_GATE
from reproducibility.bime_rank.scripts import evaluate_bridge_episodic_memory_v1 as epi
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as graph

VALID=ROOT/'results/bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz'
COLUMNS=['protein_id','reaction_id','partition','novelty','difficulty_stratum','source']
TEST=ROOT/'results/bridge_gate_split_v2/evaluation_pairs.csv.gz'
OUT=ROOT/'results/bridge_single_validation_v1'
OUTPUT=OUT/'episodic_memory_gate.single_validation.pkl'
RECORD=OUT/'episodic_memory_gate_summary.json'
def main():
 dev=pd.read_csv(VALID,dtype=str)
 test=pd.read_csv(TEST,dtype=str)
 train=pd.read_csv(graph.TRAIN,dtype=str)
 assert len(dev)==5216 and len(test)==21505 and len(train)==218537
 pairs=lambda a:set(zip(a.protein_id,a.reaction_id))
 assert not (pairs(dev)&pairs(test) or pairs(dev)&pairs(train))
 assert not (set(dev.protein_id)&set(test.protein_id))
 with RELATION_GATE.open('rb') as ff:long_gate=pickle.load(ff)
 assert set(long_gate['directions'])=={'r2e','e2r'}
 assert long_gate['directions']['r2e']['permission_power']==1.0
 assert long_gate['directions']['e2r']['permission_power']==2.0
 cache={}
 if (OUT/'episodic_r2e_query_features.csv.gz').exists() and (OUT/'episodic_e2r_query_features.csv.gz').exists():
  for d in ('r2e','e2r'):
   cache[d]=pd.read_csv(OUT/f'episodic_{d}_query_features.csv.gz')
  print('REUSE_SHARED_5216_EPISODIC_QUERY_FEATURES',flush=True)
 else:
  index=FibreCandidateIndex(device='cuda')
  prot=graph.build_prototypes(index,train)
  for direction,batch in (('r2e',16),('e2r',128)):
   f=epi.collect_direction(direction,index,dev,train,*prot,long_gate,batch)
   assert len(f)>10 and f.query_id.is_unique and set(f.query_id).issubset(
         set(dev.reaction_id if direction=='r2e' else dev.protein_id))
   p=OUT/f'episodic_{direction}_query_features.csv.gz'
   tmp=OUT/f'episodic_{direction}_query_features.tmp.csv.gz'
   f.to_csv(tmp,index=False);tmp.replace(p)
   cache[direction]=f
   print('UNIFIED_EPISODIC_COLLECTED',direction,len(f),'features',len(f.columns),flush=True)
 reports={};assets={}
 for direction in ('r2e','e2r'):
  frame=cache[direction]
  assert frame.query_id.is_unique
  report,gate=epi.fit_direction(frame)
  assert gate['schema']=='bridge-episodic-memory-gate-v1'
  assert gate['direction']==direction
  assert report['selected']['safe'],f'unsafe crossfold episodic {direction}'
  gate['fit_surface']='single frozen 5216 positive relation development corpus, grouped by query: distinct support/target within query, no overlap with final 21505 exact positives'
  gate['external_metrics_used']=False
  assets[direction]=gate
  reports[direction]={'episodes':len(frame),'selected_oof':report['selected'],
       'oracle_gate_distribution':report['oracle_gate_distribution']}
  print('EPISODIC_5216_FOLDS',direction,json.dumps(reports[direction],ensure_ascii=False),flush=True)
 bundle={
  'schema':'bridge-episodic-memory-gate-bundle-v1',
  'training_graph':'clean2023 218537 positives',
  'validation_positive_edges':5216,
  'support_semantics':'user-confirmed positive associations disjoint from clean2023 query training graph',
  'attention':'frozen Broad embedding query-to-support attention, parameter-free',
  'adapter':'Broad context correction with query-specific continuous gate in [0,1]',
  'outer_test_metrics_used_for_gate_fit':False,
  'posthoc_exploratory_after_formal_test_review':True,
  'directions':assets
 }
 tmp=OUTPUT.with_name('episodic_memory_gate.single_validation.tmp.pkl')
 with tmp.open('wb') as ff:pickle.dump(bundle,ff,pickle.HIGHEST_PROTOCOL)
 tmp.replace(OUTPUT)
 h=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
 metadata={
  'schema':'BRIDGE_SINGLE_5216_EPISODIC_SUPPORT_MEMORY_VALIDATION_V1',
  'training_graph_positive_edges':218537,
  'unique_development_positive_pairs':5216,
  'main_final_test_positive_pairs':21505,
  'dev_final_test_exact_positive_overlap':0,
  'dev_final_test_protein_query_overlap':0,
  'long_term_relation_gate_sha256':hashlib.sha256(RELATION_GATE.read_bytes()).hexdigest(),
  'validity':'posthoc exploratory: earlier formal test observed before method revisions',
  'source_validation_sha256':hashlib.sha256(dev[COLUMNS].sort_values(['protein_id','reaction_id']).reset_index(drop=True).to_csv(index=False,lineterminator=chr(10)).encode('utf-8')).hexdigest(),
  'episode_rule':'deterministic support among multi-positive query relations, residual positives targets, query-stratified 5fold',
  'r2e':reports['r2e'],'e2r':reports['e2r'],
  'candidate_release_file_sha256':h}
 RECORD.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
 print('SHARED_5216_EPISODIC_BOTH_DIRECTIONS_PASSED',h,flush=True)
if __name__=='__main__':main()
