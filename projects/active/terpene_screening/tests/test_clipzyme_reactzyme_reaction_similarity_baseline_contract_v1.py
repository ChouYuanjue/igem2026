import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'projects/active/terpene_screening/CLIPZYME_REACTZYME_REACTION_SIMILARITY_BASELINE_CONTRACT_V1.json'


def load():
    return json.loads(P.read_text())


def test_clipzyme_is_formal_reproducible_baseline_and_tiger_is_context_only():
    d = load()
    b = d['authoritative_external_baseline']
    assert b['name'] == 'CLIPZyme'
    assert b['formal_baseline'] is True
    assert b['official_code_available'] is True
    assert b['official_checkpoint_available_locally'] is True
    assert b['checkpoint_sha256'] == '536257d84126342105bd96046d98f68f58de7ceaa063331bb5b240e72c29bc98'
    assert d['tiger_reference']['formal_baseline'] is False
    assert 'methodology' in d['tiger_reference']['allowed_role']


def test_same_support_is_performance_blind_and_never_uses_catalyst_features():
    d = load()
    s = d['support_freeze_rule']
    assert s['performance_blind'] is True
    assert s['zero_imputation_for_unsupported'] is False
    assert s['repair_or_catalyst_fallback'] is False
    assert s['minimum_query_count_for_direct_reporting'] >= 50
    forbidden = ' | '.join(d['forbidden_adaptation'])
    assert 'Catalyst RDKit+' in forbidden
    assert 'performance-based support filtering' in forbidden
    assert 'TIGER result substitution' in forbidden


def test_native_reaction_novel_support_and_reveal_boundary_are_locked():
    d = load()
    s = d['official_split']
    assert s['archive_md5'] == '2d9f4e6c78d8daf5752cc2a5ae2bef0d'
    assert s['normalized_test_reaction_bags'] == 386
    assert s['normalized_train_test_reaction_bag_overlap'] == 0
    assert s['native_score_shape'] == [14688, 386]
    assert s['normalized_test_unique_positive_pairs'] == 14689
    r = d['benchmark_reveal_state']
    assert r['already_revealed_in_repository'] is True
    assert r['model_selection_allowed'] is False
    assert r['promotion_evidence_allowed'] is False


def test_metric_contract_uses_standard_query_metrics_and_keeps_optional_metrics_explicit():
    d = load()['metrics']
    assert d['primary'] == ['MRR', 'Hit@1', 'Hit@5', 'Hit@10', 'Hit@20']
    assert set(d['secondary_if_raw_scores_retained']) == {'MAP', 'NDCG@10', 'AUROC', 'median_rank'}
    assert 'standard best-positive reciprocal rank' in d['query_semantics']
