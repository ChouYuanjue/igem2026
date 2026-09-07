from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / 'projects/active/terpene_screening/CLEANROOM_R2E_FUNCTIONAL_PROTOTYPE_RESIDUAL_V1_RESULT.json'


def test_functional_prototype_v1_is_frozen_rejection():
    d=json.loads(RESULT.read_text())
    assert d['status'] == 'rejected_no_candidate_passed'
    assert d['candidate_count'] == 9
    assert d['passing_candidate_count'] == 0
    assert d['selected'] is None
    assert d['pooled_primary_query_count'] == 204
    assert d['pooled_all_query_count'] == 3226
    assert d['target_outer_labels_used'] is False
    assert d['target_benchmark_identity_used'] is False
    assert d['future_confirmation']['materialized'] is False
    assert d['future_confirmation']['reason'] == 'development_gate_failed'
    assert d['evidence']['selector_commit'] == 'b634043'


def test_no_v1_candidate_is_promoted_and_universal_failures_are_explicit():
    d=json.loads(RESULT.read_text())
    assert len(d['candidate_deltas']) == 9
    assert all(c['passed'] is False for c in d['candidate_deltas'])
    universal=set(d['universal_failure_reasons'])
    assert 'primary_macro_roc_auc_regressed' in universal
    assert 'primary_median_best_positive_rank_not_decreased' in universal
    assert 'all_map_regressed' in universal
    assert 'all_hit_at_50_regressed' in universal


def test_rejection_forbids_v1_retuning():
    d=json.loads(RESULT.read_text())
    assert 'Do not change EC level' in d['retuning_policy']
    assert d['metric_crosscheck']['status'].startswith('passed_against_')
