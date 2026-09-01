import json
from pathlib import Path
import numpy as np
from projects.active.terpene_screening.tiger_reactzyme_reaction_similarity_native_v1_common import TIGER, CombinedProteinFeatures, paper_common_metrics

ROOT=Path(__file__).resolve().parents[4]
CONTRACT=ROOT/'projects/active/terpene_screening/TIGER_REACTZYME_REACTION_SIMILARITY_BASELINE_CONTRACT_V1.json'

def test_contract_is_historical_methodology_reference_after_clipzyme_switch():
 d=json.loads(CONTRACT.read_text()); assert d['authoritative_external_baseline']['name']=='TIGER'; assert d['authoritative_external_baseline']['official_executable_code_available_as_of_2026_09_01'] is False; assert d['authoritative_external_baseline']['formal_baseline'] is False; assert d['executable_reference_not_baseline']['name']=='CLIPZyme'; assert d['executable_reference_not_baseline']['external_baseline'] is True; assert d['superseded_by'].endswith('CLIPZYME_REACTZYME_REACTION_SIMILARITY_BASELINE_CONTRACT_V1.json')
 r=d['benchmark_reveal_state']; assert r['already_revealed_in_this_repository'] is True; assert r['model_selection_allowed'] is False; assert r['promotion_evidence_allowed'] is False

def test_official_native_support_and_reaction_novel_identity_are_frozen():
 d=json.loads(CONTRACT.read_text())['official_split']; assert d['archive_md5']=='2d9f4e6c78d8daf5752cc2a5ae2bef0d'; assert d['normalized_train_reaction_bags']==7340; assert d['normalized_test_reaction_bags']==386; assert d['normalized_train_test_reaction_bag_overlap']==0; assert d['native_score_shape']==[14688,386]; assert d['normalized_test_unique_positive_pairs']==14689

def test_tiger_mrr_semantics_never_alias_standard_best_mrr():
 d=json.loads(CONTRACT.read_text()); assert 'not best-positive standard MRR' in d['tiger_reaction_similarity_metrics']['metric_semantics']['paper_MRR']; assert set(TIGER['r2e'])=={'hit_at_1','hit_at_5','hit_at_10','hit_at_20','author_avg_positive_rr'}; assert TIGER['r2e']['author_avg_positive_rr'] < TIGER['r2e']['hit_at_1']
 m={'hit_at_1':.1,'hit_at_5':.2,'hit_at_10':.3,'hit_at_20':.4,'author_avg_positive_rr':.05,'best_mrr':.9,'map':.8}; assert 'best_mrr' not in paper_common_metrics(m)

def test_combined_feature_overlay_keeps_base_rows_unchanged():
 base=np.arange(12,dtype=np.float32).reshape(3,4); over=np.arange(8,dtype=np.float32).reshape(2,4)+100; x=CombinedProteinFeatures(base,over); assert np.array_equal(x[[0,2]],base[[0,2]]); assert np.array_equal(x[[3,4]],over); assert np.array_equal(x[[2,3]],np.vstack([base[2],over[0]]))

def test_single_recipe_is_fixed_before_native_scoring():
 d=json.loads(CONTRACT.read_text())['catalyst_alignment_recipe']; assert d['selection']=='none_single_prefrozen_recipe'; assert d['model']=='DualTowerNative'; assert d['epochs']==8 and d['batch_size']==512 and d['temperature']==.05; assert d['candidate_or_hyperparameter_addition_after_freeze'] is False
