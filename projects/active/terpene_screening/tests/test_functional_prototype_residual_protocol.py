import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_FUNCTIONAL_PROTOTYPE_RESIDUAL_V1.json'
def test_external_blind_and_frozen():
 d=json.loads(P.read_text()); assert d['status'].startswith('frozen_before_any'); assert d['development']['folds']==[0,1,2]; assert d['development']['outer_labels_used'] is False; assert d['expert']['inference_uses_ec_metadata'] is False; assert d['expert']['inference_uses_dataset_identity'] is False; assert d['future_confirmation']['salt_and_fold_frozen_now_before_development_performance'] is True
def test_single_ec_abstraction_and_fallback():
 d=json.loads(P.read_text()); assert d['ec_metadata']['prefix_level']==2; assert d['ec_metadata']['min_train_proteins_per_class']==50; assert d['ec_metadata']['min_train_reactions_per_class']==10; assert d['grid']['exact_coarse_fallback'] is True; assert d['grid']['no_other_level_scale_margin_or_support_threshold_after_reveal'] is True
