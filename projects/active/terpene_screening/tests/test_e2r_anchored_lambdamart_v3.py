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
