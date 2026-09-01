import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'projects/active/terpene_screening/CATALYST_FINAL_CAPABILITY_CONTRACT_MATRIX_V1.json'


def load():
    return json.loads(P.read_text())


def test_global_baseline_selection_rules_are_nonnegotiable():
    g = load()['global_rules']
    assert g['one_authoritative_external_baseline_per_capability'] is True
    assert g['internal_catalyst_model_is_never_the_external_baseline'] is True
    assert g['minimal_scope_adapter_only'] is True
    assert g['same_support_for_direct_delta'] is True
    assert g['unsupported_metrics_are_na'] is True
    assert g['revealed_test_metrics_never_select_models_or_routes'] is True
    assert g['external_baseline_may_be_retained_as_expert_or_backstop'] is True


def test_completed_contracts_have_one_external_baseline_and_no_metric_mixing():
    d = load()['capabilities']
    for cid in ('e2r_sequence_divergent_known_reaction_native', 'r2e_sequence_divergent_known_reaction_native', 'r2e_structure_available', 'r2e_native_molecule_bag'):
        c = d[cid]
        assert isinstance(c['authoritative_external_baseline'], str) and c['authoritative_external_baseline']
        assert c['primary_common_metrics']
        assert c['allowed_adaptation']
        assert c['forbidden_adaptation']


def test_known_reaction_and_true_reaction_novel_are_not_conflated():
    d = load()['capabilities']
    for cid in ('e2r_sequence_divergent_known_reaction_native','r2e_sequence_divergent_known_reaction_native'):
        assert 'known-reaction' in d[cid]['claim_boundary'].lower() or 'reaction-novel' in d[cid]['claim_boundary'].lower()
    for cid in ('r2e_true_reaction_novel','e2r_true_reaction_novel'):
        assert d[cid]['authoritative_external_baseline'] == 'CLIPZyme'
        assert 'MAT2D-ESM' in d[cid]['baseline_status']
        assert 'frozen before training' in d[cid]['baseline_status']
        assert 'no promotion' in d[cid]['next_action'].lower()
        assert 'TIGER' in ' | '.join(d[cid]['candidate_baselines_to_research'])


def test_material_gain_status_is_metric_specific():
    d = load()['capabilities']
    e2r=d['e2r_sequence_divergent_known_reaction_native']
    r2e=d['r2e_sequence_divergent_known_reaction_native']
    assert set(e2r['material_gain_gt5pp_metrics']) == {'MAP','NDCG@1','NDCG@5','Top1','Top5'}
    assert set(r2e['material_gain_gt5pp_metrics']) == {'MAP','NDCG@1','NDCG@5','Top1'}
    assert 'Top5' not in r2e['material_gain_gt5pp_metrics']
