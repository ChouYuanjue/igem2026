"""Frozen TPS semantic-eligibility slice of the shared 21,505 test cohort.

No new R2E inference. Gate is the released TPS production semantic threshold.
Full vs minus-family/domain ranks are reused from original 23,773 cached
per-pair R2E score file, joined by exact pair key to canonical test only.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from projects.active.bridge.model.assets import ROOT

TGT=ROOT/'results/bridge_gate_split_v2/evaluation_pairs.csv.gz'
AB=ROOT/'results/bridge_r2e_single_validation_release_v1/edge_metrics.csv.gz'
FEATURES=ROOT/'results/fibre_tps_specialist_response_v2/production/query_features.csv'
GATE=ROOT/'results/fibre_tps_specialist_gate_v2/summary.json'
OUT=ROOT/'reproducibility/bime_rank/records/BRIDGE_TPS_CANONICAL_QUERY_SLICE_V1_RESULT.json'
def main():
 if OUT.exists():
  previous=json.loads(OUT.read_text())
  assert previous["schema"]=="BRIDGE_TPS_SEMANTIC_DOMAIN_CANONICAL_21505_V1"
 gate=json.loads(GATE.read_text())
 threshold=float(gate['semantic_gate']['threshold'])
 feat=pd.read_csv(FEATURES,dtype={'query_id':str})
 activated=set(feat.query_id[feat.tps_ref_max_cosine>=threshold])
 target=pd.read_csv(TGT,dtype=str)
 rows=pd.read_csv(AB,dtype={'protein_id':str,'reaction_id':str})
 selected=target.merge(rows[['protein_id','reaction_id','full_rank','minus_family_domain_rank']],
                       on=['protein_id','reaction_id'],how='left',validate='one_to_one')
 assert len(selected)==21505 and selected[['full_rank','minus_family_domain_rank']].notna().all().all()
 subset=selected[selected.reaction_id.isin(activated)]
 count=int(subset.reaction_id.nunique())
 assert count>0
 out={}
 for c,name in (('minus_family_domain_rank','without_family_domain'),('full_rank','bridge')):
  best=subset.groupby('reaction_id')[c].min().to_numpy(np.int64)
  assert len(best)==count
  out[name]={'query_mrr':float(np.mean(1.0/best)),
             'query_hit10':float(np.mean(best<=10)),
             'query_hit100':float(np.mean(best<=100))}
 result={
  'schema':'BRIDGE_TPS_SEMANTIC_DOMAIN_CANONICAL_21505_V1',
  'selection':'pre-frozen full-production TPS semantic applicability only',
  'source_frozen_semantic_feature':'tps_ref_max_cosine',
  'source_frozen_threshold':threshold,
  'evaluation':'same frozen canonical 21505 relation test cohort, best filtered true-pair rank per reaction query',
  'formal_main_test_pairs':21505,
  'tps_semantic_active_reaction_queries':count,
  'tps_semantic_active_positive_edges':len(subset),
  'method_ablation_definition':'full R2E ranking versus removal of all family/domain specialists within the TPS-semantic query slice',
  'all_r2e_ranks_reused_from_frozen_cache':True,
  'r2e_relation_gate_trained_on_shared_validation_edges':5216,
  'functional_score_validated_multiplier':1.25,
  'no_test_cohort_parameter_training':True,
  'reaction_ids':sorted(subset.reaction_id.unique().tolist()),
  'measurements':out,
 }
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
