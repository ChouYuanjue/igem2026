import json
import numpy as np
from pathlib import Path
from projects.active.terpene_screening.train_reactzyme_native_bag_adapter_v1 import bag_feature,BagAdapter
ROOT=Path(__file__).resolve().parents[4]
def test_bag_feature_is_direction_and_side_invariant():
 a,na,ia=bag_feature('CCO.CC','O=C=O')
 b,nb,ib=bag_feature('O=C=O','CC.CCO')
 assert a.shape==(4096,) and np.allclose(a,b)
 assert na==nb==3 and ia==ib==0
def test_adapter_dimensions():
 import torch
 m=BagAdapter(); y=m(torch.zeros(2,4096)); assert y.shape==(2,320); assert torch.isfinite(y).all()
def test_protocol_is_frozen_and_forbids_dev_training():
 d=json.loads((ROOT/'projects/active/terpene_screening/REACTZYME_NATIVE_BAG_ADAPTER_V1.json').read_text())
 assert d['status']=='frozen_before_performance'; assert d['input']['dimension']==4096; assert d['adapter']['hyperparameter_sweep'] is False; assert d['split']['protein_or_pair_labels_used_by_adapter_training'] is False
 assert any('dev reaction IDs' in x for x in d['forbidden'])

def test_selector_thresholds_are_frozen_in_source():
 source=(ROOT/'projects/active/terpene_screening/select_reactzyme_native_bag_adapter_v1.py').read_text()
 assert "teacher_cosine_mean_ge_0p75" in source and ">=0.75" in source
 assert "at_least_2_folds_mrr_retention_ge_0p65" in source and ">=0.65" in source
 assert "external_reactzyme_metrics_used':False" in source

def test_evaluator_uses_repository_standard_normalized_protein_loader():
 source=(ROOT/'projects/active/terpene_screening/evaluate_reactzyme_native_bag_adapter_v1.py').read_text()
 assert 'load_protein_library(PROT)' in source
 assert 'teacher baseline reproduction failed' in source
 assert "cleanroom_internal_full_candidate_rdkitplus_v1" in source
