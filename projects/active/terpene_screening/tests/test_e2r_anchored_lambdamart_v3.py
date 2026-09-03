import json
from pathlib import Path
import numpy as np
from projects.active.terpene_screening.run_unified_safe_system_e2r_anchored_lambdamart_v3 import FEATURE_NAMES,BASELINE_ANCHOR_INDEX,feature_matrix,structure_configs
ROOT=Path(__file__).resolve().parents[4]

def test_protocol_is_fresh_and_frozen_before_development():
 p=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3.json').read_text())
 assert p['status']=='frozen_before_any_v3_salted_development_performance'
 assert p['development_split']['split_salt']=='e2r_anchored_lambdamart_v3_dev_20260902_c'
 assert p['future_confirmation_split']['split_salt']=='e2r_anchored_lambdamart_v3_confirm_20260902_c'
 assert p['development_split']['development_folds']==[0,1,2]
 assert p['future_confirmation_split']['dev_fold']==6 and p['future_confirmation_split']['folds']==7
 assert p['ranker']['automatic_search_count']==256 and len(p['ranker']['ranker_configs'])==8
 assert p['ranker']['no_manual_post_result_hyperparameter_choice'] is True
 assert 'V1 folds 0/1/2 or V2 salted development metrics for V3 selection' in p['forbidden']

def test_structure_grid_is_exactly_frozen_32_configs():
 s=structure_configs(); assert len(s)==32
 assert all(x['protected_prefix']<x['prefix_k']<=x['pool_k'] for x in s)
 assert {x['protected_prefix'] for x in s}=={1,3,5,9}
 assert {x['pool_k'] for x in s}=={20,50,100}
 assert {x['prefix_k'] for x in s}=={10,20,50}

def test_feature_contract_has_single_monotone_baseline_anchor():
 assert len(FEATURE_NAMES)==26
 assert FEATURE_NAMES[BASELINE_ANCHOR_INDEX]=='baseline_anchor'
 S=np.asarray([[.9,.8,.1],[.1,.2,.3],[.2,.4,.1],[.7,.1,.5]],dtype=np.float32)
 ranks=np.asarray([[1,3,2,1],[2,2,1,3],[3,1,3,2]],dtype=np.int32)
 X=feature_matrix(S,np.asarray([0,1,2]),ranks)
 assert X.shape==(3,26)
 assert np.all(np.diff(X[:,BASELINE_ANCHOR_INDEX])<0)

def test_material_gate_is_multi_metric_not_old_five_point_hit10_only():
 p=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3.json').read_text())
 assert p['development_gate']['material_gain']=='pooled MRR delta >= 0.003 OR MAP delta >= 0.003 OR Hit@10 delta >= 0.01'
 assert p['development_gate']['confirmation_authorized_only_if_selected'] is True


def test_automatic_selection_implements_frozen_complexity_tiebreak():
 s=(ROOT/'projects/active/terpene_screening/run_unified_safe_system_e2r_anchored_lambdamart_v3.py').read_text()
 assert "'protected_prefix','prefix_k','pool_k','ranker_max_depth','ranker_rounds','ranker_id'" in s
 assert "ascending=[False,False,False,False,True,True,True,True,True]" in s

def test_confirmation_protocol_freezes_selected_config_and_strict_gate_before_reveal():
 p=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3_CONFIRMATION.json').read_text())
 assert p['status']=='frozen_after_development_selection_before_confirmation_materialization'
 assert p['confirmation_split']['split_salt']=='e2r_anchored_lambdamart_v3_confirm_20260902_c'
 assert p['confirmation_split']['folds']==7 and p['confirmation_split']['dev_fold']==6
 assert p['development_selection']['selected_config']['ranker_id']=='ndcg_d3_e010'
 assert p['development_selection']['selected_config']['protected_prefix']==1
 assert p['development_selection']['selected_config']['pool_k']==20
 assert p['development_selection']['selected_config']['prefix_k']==20
 assert p['final_ranker']['seed']==20260902 and p['final_ranker']['confirmation_labels_used_for_training'] is False
 assert p['confirmation_gate']['material_gain']=='MRR delta >= 0.003 OR MAP delta >= 0.003 OR Hit@10 delta >= 0.01'
 assert len(p['confirmation_gate']['required_non_regression'])==7


def test_confirmation_gate_requires_no_regression_and_material_gain():
 from projects.active.terpene_screening.run_unified_safe_system_e2r_anchored_v3_confirmation import gate
 good={k:0.0 for k in ['mrr','map','auc','ndcg10','hit10','hit20','hit50']}; good['mrr']=0.003
 assert gate(good)['pass'] is True
 bad=dict(good); bad['hit50']=-1e-4
 assert gate(bad)['pass'] is False
 weak={k:0.0 for k in good}; weak['mrr']=0.0029; weak['map']=0.0029; weak['hit10']=0.0099
 assert gate(weak)['pass'] is False

def test_confirmation_result_passes_and_forbids_same_fold_retuning():
 r=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3_CONFIRMATION_RESULT.json').read_text())
 assert r['status']=='passed_fresh_internal_salted_confirmation'
 assert r['decision']=='authorize_production_packaging_subject_to_runtime_and_retention_gates'
 assert r['query_count']==3786 and r['candidate_count']==11081
 assert r['split_audit']['protein_overlap']==0 and r['split_audit']['reaction_overlap']==0
 assert r['split_audit']['all_four_experts_byte_identical_train_dev'] is True
 assert r['same_confirmation_retuning_allowed'] is False and r['external_metrics_used'] is False
 assert all(r['gate']['checks'].values()) and r['gate']['material_gain'] is True and r['gate']['pass'] is True
 assert r['delta']['mrr']>0.003 and r['delta']['map']>0.003 and r['delta']['hit10']>0.01
 assert r['delta']['hit20']>0 and r['delta']['hit50']>0 and r['delta']['ndcg10']>0 and r['delta']['auc']>0
 assert r['final_ranker_sha256']=='2f860391751aa0c420054f0b30c70b878d451a9340d992f8bd5a379d5714b7ae'

def test_production_protocol_preserves_scope_and_runtime_gates():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_E2R_ANCHORED_LAMBDAMART_V3_PRODUCTION.json').read_text())
 assert p['status']=='frozen_before_full_clean_packaging_and_runtime_measurement'
 assert p['ranker']['sha256']=='2f860391751aa0c420054f0b30c70b878d451a9340d992f8bd5a379d5714b7ae'
 assert p['full_clean_experts']['expected_training_pairs']==218537 and p['full_clean_experts']['dev_fold']==-1
 assert p['selected_config']['protected_prefix']==1 and p['selected_config']['pool_k']==20 and p['selected_config']['prefix_k']==20
 assert p['runtime_scope_gate']['raw_sequence_may_promote_only_if_feature_parity_and_runtime_gate_pass'] is True
 assert 'preserve existing E2R production route' in p['runtime_scope_gate']['out_of_scope_fallback']
 assert p['production_gates']['latency']['registered_query_warm_median_ratio_vs_existing_route_max']==3.0
 assert p['external_benchmark_selection_allowed'] is False


def test_runtime_anchored_order_matches_frozen_positive_rank_transform():
 from projects.active.terpene_screening.e2r_anchored_lambdamart_runtime import AnchoredE2RRuntime
 rng=np.random.default_rng(7); n=61
 S=rng.normal(size=(4,n)).astype(np.float32)
 ranks=np.stack([__import__('projects.active.terpene_screening.run_unified_safe_system_e2r_anchored_lambdamart_v3',fromlist=['full_ranks']).full_ranks(S[e]) for e in range(4)],axis=1)
 union=np.asarray(sorted(set().union(*(set(np.argsort(-S[e],kind='stable')[:40].tolist()) for e in range(4)))),dtype=np.int32)
 pred=rng.normal(size=len(union)).astype(np.float32)
 order,base,selected=AnchoredE2RRuntime.anchored_order(S,pred,union,protected_prefix=3,pool_k=20,prefix_k=20)
 inv=np.empty(n,dtype=np.int64); inv[order]=np.arange(1,n+1)
 # Reproduce the exact algebra used by frozen candidate_query_metrics for every row.
 union_ranks=ranks[union]; pool=(union_ranks.min(1)<=20)&(union_ranks[:,0]>3); local=np.flatnonzero(pool); local=local[np.lexsort((union[local],-pred[local]))]; chosen=union[local[:17]]; chosen_base=ranks[chosen,0]; chosen_pos={int(r):3+i+1 for i,r in enumerate(chosen)}
 expected=[]
 for r in range(n):
  br=int(ranks[r,0])
  if br<=3: nr=br
  elif r in chosen_pos: nr=chosen_pos[r]
  else: nr=br+len(chosen)-int(np.count_nonzero(chosen_base<br))
  expected.append(nr)
 assert inv.tolist()==expected
 assert base.tolist()==np.argsort(-S[0],kind='stable').tolist()
 assert selected.tolist()==chosen.tolist()

def test_runtime_gate_query_selection_is_label_free_and_deterministic():
 from projects.active.terpene_screening.benchmark_e2r_anchored_lambdamart_v3_runtime import fixed_query_ids
 a=fixed_query_ids(); b=fixed_query_ids()
 assert a==b and len(a)==12 and len(set(a))==12


def test_live_manifest_declares_confirmed_e2r_v3_fast_route_with_old_fallbacks():
 import yaml
 live=yaml.safe_load((ROOT/'configs/production_routes/terpene_v1.yaml').read_text())
 assert live['route_version']=='terpene-production-routes-v5'
 assert live['deployments']['e2r_clean_anchored_v3']=='results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3'
 for objective in ('top3','top10','top20'):
  spec=live['routes']['enzyme_to_reaction']['external'][objective]
  learned=spec['anchored_lambdamart_v3']
  assert learned['enabled'] is True
  assert learned['route_id']=='e2r-external-anchored-lambdamart-v3'
  assert learned['model_bundle_version']=='catalyst-e2r-anchored-lambdamart-v3'
  assert 'preserve existing objective-specific E2R production route' in learned['ineligible_behavior']
 # Historical objective-specific fallbacks remain declared rather than overwritten.
 assert live['routes']['enzyme_to_reaction']['external']['top3']['route_id']=='e2r-external-top3-neighbor-v1'
 assert live['routes']['enzyme_to_reaction']['external']['top10']['route_id']=='e2r-external-top10-neural-rrf-v1'
 assert live['routes']['enzyme_to_reaction']['external']['top20']['route_id']=='e2r-external-top20-dual-kernel-rrf-v1'


def test_e2r_v3_fast_path_scope_is_exact_and_special_requests_fall_back():
 from projects.active.terpene_screening.core.engine import payload_to_argv
 from projects.active.terpene_screening import rank_open_world as rw
 def args(extra=None):
  payload={'enzyme_id':'EXTERNAL_TEST','candidate_universe':'general_merged','top_k':10}
  payload.update(extra or {})
  return rw.build_parser().parse_args(payload_to_argv('rank-reactions',payload))
 a=args()
 assert rw.should_use_e2r_anchored_lambdamart_v3(a,dual_tower_dir=a.dual_tower_dir.resolve(),is_current_enzyme=False)
 assert not rw.should_use_e2r_anchored_lambdamart_v3(a,dual_tower_dir=a.dual_tower_dir.resolve(),is_current_enzyme=True)
 for extra in (
  {'known_reaction_ids':['RHEA:1']},
  {'mask_reaction_ids':['RHEA:1']},
  {'candidate_ids':['RHEA:1']},
  {'retrieval_mode':'direct'},
 ):
  b=args(extra)
  assert not rw.should_use_e2r_anchored_lambdamart_v3(b,dual_tower_dir=b.dual_tower_dir.resolve(),is_current_enzyme=False)


def test_e2r_v3_runtime_and_retention_gates_are_passed():
 runtime=json.loads((ROOT/'results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3/runtime_gate/result.json').read_text())
 retention=json.loads((ROOT/'results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3/retention_gate/result.json').read_text())
 assert runtime['status']=='pass' and all(runtime['checks'].values())
 assert runtime['ratios_vs_fastest_existing_objective']['median'] < .1
 assert retention['status']=='pass' and retention['query_count']==48
 assert all(v['pass'] for v in retention['metrics'].values())
 assert retention['metrics']['top10']['candidate_hit'] > .9
 assert retention['external_labels_used_for_model_selection'] is False
