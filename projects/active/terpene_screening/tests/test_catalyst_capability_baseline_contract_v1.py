import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'projects/active/terpene_screening/CATALYST_CAPABILITY_BASELINE_CONTRACT_V1.json'


def load():
    return json.loads(P.read_text())


def test_one_external_baseline_per_contract_and_never_internal():
    d = load()
    assert d['global_rules']['exactly_one_authoritative_external_baseline_per_contract'] is True
    assert d['global_rules']['own_historical_model_may_be_external_baseline'] is False
    assert len(d['contracts']) >= 4
    forbidden_joiners = ('/', ' or ', ' OR ', ',')
    for cid, c in d['contracts'].items():
        b = c['authoritative_external_baseline']
        assert isinstance(b, str) and b.strip()
        assert not any(x in b for x in forbidden_joiners), (cid, b)
        assert c['baseline_origin'] == 'external_author_method'
        floor = c['internal_safety_floor']
        assert floor['external_baseline'] is False
        assert floor['name'] != b


def test_support_and_adapter_policy_is_fair_and_metric_complete():
    d = load()
    g = d['global_rules']
    assert g['same_support_required_for_direct_delta'] is True
    assert g['support_may_be_restricted_using_target_performance'] is False
    assert g['baseline_may_receive_catalyst_extra_training_signal'] is False
    assert g['baseline_may_receive_catalyst_extra_expert_or_router'] is False
    assert g['unsupported_metric_is_explicit_na_never_imputed'] is True
    assert g['material_gain_target_hit_rate_pp'] >= 5.0
    banned = ('catalyst expert fusion', 'catalyst router', 'extra training labels', 'target-performance support filtering')
    for cid, c in d['contracts'].items():
        assert c['support_scope']
        assert c['primary_metrics']
        assert c['material_gain_target_hit_rate_pp'] >= 5.0
        forbidden = ' | '.join(c['forbidden_adaptation']).lower()
        if cid != 'r2e_structure_available':
            assert 'catalyst router' in forbidden
        assert 'target-performance support filtering' in forbidden
        allowed = ' | '.join(c['allowed_adaptation']).lower()
        assert 'target-performance' not in allowed
        assert 'external test label' not in allowed


def test_revealed_external_evidence_never_selects_models():
    d = load()
    assert d['global_rules']['revealed_external_benchmark_may_be_used_for_model_selection'] is False
    assert d['contracts']['r2e_structure_available']['model_selection_allowed_on_current_direct_external_evidence'] is False
    assert d['contracts']['r2e_native_molecule_bag']['model_selection_allowed_on_revealed_external_reactzyme_metrics'] is False
    for cid in ('r2e_sequence_reaction','e2r_sequence_reaction'):
        assert d['contracts'][cid]['model_selection_allowed_on_revealed_outer_sets'] is False


def test_literature_frontiers_are_not_silently_mixed_into_operational_baseline():
    d=load()
    frontier=d['literature_frontier']['sequence_reaction_bidirectional']
    assert frontier['method']=='TIGER'
    assert frontier['role']=='stronger_literature_ceiling_not_current_operational_baseline'
    for cid in ('r2e_sequence_reaction','e2r_sequence_reaction'):
        assert d['contracts'][cid]['authoritative_external_baseline']=='EnzGFM-1.5B'
