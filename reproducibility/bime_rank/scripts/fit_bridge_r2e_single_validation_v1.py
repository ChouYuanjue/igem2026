"""One frozen enzyme-reaction development/validation inventory for both directions.

Historical 16,108-relation source-expansion gate fit is not used for this
method. R2E and E2R validation uses the *same* 5,216 annotated relations.
R2E fit uses reaction-query out-of-fold partitions inside this cohort;
E2R joint router's original 3,900+1,316 protein-query split is a subset
of the same inventory, not an additional source of labels.

This is a post-hoc exploratory refit after initial final test review.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle
from pathlib import Path
import pandas as pd, numpy as np
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import evaluate_bridge_adaptive_relation_context_v8 as adap
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as base
from reproducibility.bime_rank.scripts.analyze_bridge_difficulty_standardized_v3 import annotate

SRC=ROOT/'results/bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz'
COLUMNS=['protein_id','reaction_id','partition','novelty','difficulty_stratum','source']
TEST=ROOT/'results/bridge_gate_split_v2/evaluation_pairs.csv.gz'
TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
OUT=ROOT/'results/bridge_single_validation_v1'
RECORD=ROOT/'reproducibility/bime_rank/records/BRIDGE_SINGLE_VALIDATION_V1_RESULT.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def inspect():
 pairs=pd.read_csv(SRC,dtype=str)
 test=pd.read_csv(TEST,dtype=str)
 train=pd.read_csv(TRAIN,dtype=str)
 keys=['protein_id','reaction_id']
 assert len(pairs)==5216 and len(test)==21505 and len(train)==218537
 assert not pairs.duplicated(keys).any() and not test.duplicated(keys).any()
 assert not set(zip(pairs.protein_id,pairs.reaction_id))&set(zip(train.protein_id,train.reaction_id))
 assert not set(zip(pairs.protein_id,pairs.reaction_id))&set(zip(test.protein_id,test.reaction_id))
 assert not set(pairs.protein_id)&set(test.protein_id)
 summary={
  'schema':'bridge-shared-validation-single-source-v1',
  'training':'clean2023 association graph 218537 positives',
  'validation_source':'frozen bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz',
  'validation_edges':5216,
  'validation_enzyme_queries':int(pairs.protein_id.nunique()),
  'validation_reaction_queries':int(pairs.reaction_id.nunique()),
  'validation_novelty_counts':pairs.novelty.value_counts().to_dict(),
  'test_edges':21505,
  'test_enzyme_queries':int(test.protein_id.nunique()),
  'test_reaction_queries':int(test.reaction_id.nunique()),
  'validation_test_shared_protein_queries':0,
  'validation_test_shared_reaction_queries':int(len(set(pairs.reaction_id)&set(test.reaction_id))),
  'validation_test_shared_positive_edges':0,
  'validation_protein_query_fold_split_counts':pairs.partition.value_counts().to_dict(),
  'source_validation_sha256':hashlib.sha256(pairs[COLUMNS].sort_values(['protein_id','reaction_id']).reset_index(drop=True).to_csv(index=False,lineterminator=chr(10)).encode('utf-8')).hexdigest(),
  'source_validation_hash_semantics':'sha256 of sorted CSV plaintext: protein_id,reaction_id,partition,novelty,difficulty_stratum,source',
  'source_test_sha256':sha(TEST),
  'source_training_sha256':sha(TRAIN),
  'experimental_status':'post-hoc exploratory refitting after original final test read; formal independent confirmation requires fresh external untouched final labels',
  'r2e_old_16108_gate_replaced_only_if_shared_validation_r2e_oof_safe':True,
 }
 OUT.mkdir(exist_ok=True,parents=True)
 (OUT/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 return summary,pairs,train

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--prepare-only',action='store_true')
 ap.add_argument('--device',default='cuda')
 args=ap.parse_args()
 sm,pairs,train=inspect()
 print('ONE_VALIDATION_SOURCE',json.dumps(sm,ensure_ascii=False),flush=True)
 if args.prepare_only:return
 out=OUT/'r2e_relation_query_features.csv.gz'
 if out.exists():
  f=pd.read_csv(out)
  assert len(f)==sm['validation_reaction_queries']
  print('REUSE_VALIDATION_R2E_FEATURES',len(f),flush=True)
 else:
  index=FibreCandidateIndex(device=args.device)
  prot=base.build_prototypes(index,train)
  f=adap._collect_direction('r2e',index,pairs,train,*prot,12)
  assert len(f)==sm['validation_reaction_queries'] and f.query_id.is_unique
  temp=out.with_suffix('.tmp.gz');f.to_csv(temp,index=False);temp.replace(out)
  print('UNIFIED_R2E_RELATION_FEATURES_COLLECTED',f.shape,flush=True)
 cv,asset=adap._cv_fit(f)
 print('UNIFIED_R2E_RELATION_GATE_QUERY_OOF',
       json.dumps({'queries':len(f),'base':adap._metric(f.base_rank.to_numpy(np.int64)),
          'selected':{k:v for k,v in cv.items() if k!='cv_candidates'}},ensure_ascii=False),flush=True)
 assert asset['direction']=='r2e'
 with (OUT/'r2e_relation_gate.exploratory.pkl').open('wb') as fp:pickle.dump(asset,fp)
 (OUT/'r2e_relation_gate_summary.json').write_text(json.dumps({
  'shared_validation_edges':5216,
  'r2e_reaction_queries':len(f),
  'baseline_query_mrr':adap._metric(f.base_rank.to_numpy(np.int64)),
  'crossvalidation':cv,
  'train_and_validation_only_no_test_fit':True,
  'exploratory_recalibration':True,
 },ensure_ascii=False,indent=2)+'\n')
 print('UNIFIED_R2E_RELATION_GATE_FIT_DONE')
if __name__=='__main__':main()
