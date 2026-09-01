import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'projects/active/terpene_screening/CLIPZYME_DIRECTED_REACTION_NOVEL_FALLBACK_CONTRACT_V1.json'


def test_fallback_is_selected_by_native_support_not_model_performance():
    d = json.loads(P.read_text())
    p = d['selection_policy']
    assert p['uses_model_scores'] is False
    assert p['uses_test_performance'] is False
    assert p['uses_target_labels_for_cell_selection'] is False
    assert p['minimum_native_reaction_queries'] >= 50
    assert p['priority_order_is_frozen'] is True
    assert [c['cell'] for c in d['priority_cells']] == [
        'reactzyme_reaction_projected_double_cold',
        'temporal_post2020_double_cold',
        'broad_reaction_hash_cold_protein_seen',
    ]


def test_official_clipzyme_is_not_rewritten_to_gain_support():
    d = json.loads(P.read_text())
    b = d['authoritative_external_baseline']
    assert b['name'] == 'CLIPZyme'
    assert b['encoder_substitution_allowed'] is False
    assert b['fine_tuning_on_fallback_labels_allowed'] is False
    rejected = ' | '.join(d['rejected_paths'])
    assert 'MAT/UniMol/RDKit' in rejected
    assert 'largest delta' in rejected
    assert 'Catalyst representations' in rejected


def test_existing_fallback_cells_are_descriptive_not_fresh_promotion_evidence():
    d = json.loads(P.read_text())
    r = d['reveal_and_promotion_boundary']
    assert r['these_existing_cells_have_been_previously_inspected'] is True
    assert r['model_selection_on_their_scores_allowed'] is False
    assert r['hyperparameter_or_router_selection_allowed'] is False
    assert 'fresh promotion claims' in r['role']
