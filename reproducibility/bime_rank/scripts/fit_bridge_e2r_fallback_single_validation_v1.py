"""Refit the E2R compatible reciprocal-relation gate on the sole shared 5,216 dataset.

The main E2R v4 query route and two-sided induction from 5,216 development
associations remain unchanged. This refits only the *fallback* reciprocal
training-graph relation authority, to exclude historical old-16,108 labels
which overlap the parent heldout relations.
"""
from __future__ import annotations
import json,pickle
from pathlib import Path
import pandas as pd, numpy as np
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import evaluate_bridge_adaptive_relation_context_v8 as adap
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as base
from reproducibility.bime_rank.scripts.fit_bridge_r2e_single_validation_v1 import inspect,OUT

OUT_FRAME=OUT/'e2r_fallback_relation_query_features.csv.gz'
ASSET=OUT/'e2r_fallback_relation_gate.exploratory.pkl'
SUMMARY=OUT/'e2r_fallback_relation_gate_summary.json'
def main():
 m,development,train=inspect()
 if OUT_FRAME.exists():
  f=pd.read_csv(OUT_FRAME)
  assert len(f)==m['validation_enzyme_queries']
  print('REUSE_UNIFIED_E2R_FALLBACK_RELATION_FEATURES',len(f),flush=True)
 else:
  index=FibreCandidateIndex(device='cuda')
  prot=base.build_prototypes(index,train)
  f=adap._collect_direction('e2r',index,development,train,*prot,128)
  assert len(f)==m['validation_enzyme_queries'] and f.query_id.is_unique
  temp=OUT_FRAME.with_name('e2r_fallback_relation_query_features.tmp.csv.gz')
  f.to_csv(temp,index=False);temp.replace(OUT_FRAME)
  print('UNIFIED_E2R_FALLBACK_GATE_FEATURES',f.shape,flush=True)
 cv,gate=adap._cv_fit(f)
 assert gate['direction']=='e2r'
 with ASSET.open('wb') as fp:pickle.dump(gate,fp)
 result={'schema':'bridge-single-5216-e2r-fallback-adaptive-gate-v1',
  'validation_source':'shared 5216 source positive relations',
  'E2R_query_count':len(f),'train_graph_positive_edges':218537,
  'main_E2R_v4_gate_and_C2_unchanged':True,
  'out_of_fold_baseline':adap._metric(f.base_rank.to_numpy(np.int64)),
  'out_of_fold_gate':{k:v for k,v in cv.items() if k!='cv_candidates'},
  'posthoc_correction':'Historical 16108 validation had exact positives in final test; only 5216 clean positives used',
  'final_test_labels_used_for_fitting':False}
 SUMMARY.write_text(json.dumps(result,indent=2)+'\n')
 print('E2R_SINGLE_5216_FALLBACK_GATE_OOF_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
