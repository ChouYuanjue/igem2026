import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'projects/active/terpene_screening/CLIPZYME_REACTZYME_REACTION_SIMILARITY_BASELINE_CONTRACT_V1.json'


def load():
    return json.loads(P.read_text())


def test_formal_clipzyme_baseline_keeps_official_native_encoder():
    d = load()
    b = d['authoritative_external_baseline']
    assert b['name'] == 'CLIPZyme'
    assert b['formal_baseline'] is True
    assert b['official_code_available'] is True
    assert b['official_checkpoint_available_locally'] is True
    assert 'released CLIPZyme model and native author encoders' in b['execution_rule']
    assert 'MAT' in b['execution_rule']
    assert 'never the authoritative CLIPZyme' in b['execution_rule']


def test_unordered_reactzyme_bags_do_not_silently_gain_direction():
    d = load()
    boundary = d['official_reactzyme_split']['input_semantics_boundary']
    assert 'unordered dot-separated molecule bags' in boundary
    assert 'not automatically a valid native CLIPZyme input domain' in boundary
    audit = d['reactzyme_compatibility_audit']
    assert audit['performance_blind'] is True
    assert audit['minimum_query_count_for_direct_reporting'] >= 50
    forbidden = ' | '.join(audit['forbidden_direction_recovery']).lower()
    assert 'molecule bag by position' in forbidden
    assert 'target labels' in forbidden
    assert 'replacement reaction encoder' in forbidden


def test_incompatible_full_matrix_causes_subset_or_benchmark_switch_not_baseline_rewrite():
    d = load()
    rule = d['reactzyme_compatibility_audit']['decision_rule']
    assert 'same subset' in rule
    assert 'fewer than 50' in rule
    assert 'do not modify CLIPZyme' in rule
    assert 'separately frozen directed-reaction benchmark' in rule
    forbidden = ' | '.join(d['forbidden_adaptation'])
    assert 'MAT/UniMol/RDKit/ReactZyme reaction encoder substituted' in forbidden
    assert 'inventing substrate/product direction' in forbidden


def test_tiger_inspired_mat_esm_is_only_secondary_methodology_ablation():
    d = load()
    tiger = d['tiger_reference']
    assert tiger['formal_baseline'] is False
    assert tiger['allowed_role'] == 'methodology and literature-frontier context only'
    a = d['secondary_methodology_ablation']
    assert a['authoritative_external_baseline'] is False
    assert a['allowed'] is True
    assert 'must never be labeled as the official CLIPZyme checkpoint result' in a['role']
