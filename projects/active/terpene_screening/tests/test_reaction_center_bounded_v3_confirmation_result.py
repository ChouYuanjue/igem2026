import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3_CONFIRMATION_RESULT.json'
def test_v3_confirmation_result_is_clean_pass_and_mainline():
 r=json.loads(PATH.read_text())
 assert r['status']=='passed_fresh_internal_salted_confirmation'
 assert r['decision']=='promote_bounded_residual_v3_cap0p1_as_internal_r2e_mainline'
 assert r['selected_max_residual_ratio']==0.1 and r['outer_labels_used'] is False and r['model_selection_allowed_after_confirmation'] is False
 assert r['split_audit']['exact_pair_overlap']==0 and r['split_audit']['protein_overlap_fraction']==0 and r['split_audit']['reaction_overlap_fraction']==0
 assert r['candidate_identity_audit']['max_abs_diff']==0.0 and r['trainable_parameter_names']==['aux_to_hidden.weight']
 assert all(r['checks'].values())
def test_v3_confirmation_improves_registered_all_and_hard_metrics():
 r=json.loads(PATH.read_text())
 for scope in ['all','lt0p3']:
  assert r[scope]['delta']['mrr']>0 and r[scope]['delta']['map']>0 and r[scope]['delta']['ndcg_at_10']>0
  assert r[scope]['delta']['hit_at_10']>0 and r[scope]['delta']['hit_at_50']>0
  assert r[scope]['delta']['median_best_positive_rank']<0
 assert r['posthoc_paired_bootstrap']['descriptive_only'] is True
