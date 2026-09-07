import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / 'projects/active/terpene_screening/ENZGFM_NATIVE_SAME_SUPPORT_CATALYST_V1_RESULT.json'


def load():
    return json.loads(RESULT.read_text())


def test_native_result_is_frozen_and_not_reused_for_selection():
    d = load()
    assert d['status'] == 'revealed_external_frozen_posthoc_descriptive_only'
    assert d['no_post_reveal_model_selection'] is True
    assert d['selected_candidate'] == 'dual_tower'
    assert d['alternative_candidate_native_test_scored'] is False
    assert d['interpretation']['future_tuning_on_this_native_test_forbidden'] is True


def test_official_archive_and_split_boundary_are_explicit():
    d = load()
    prov = d['official_split_provenance']
    assert prov['archive_md5'] == 'e351fdb85830968fc9abe933c39f9eda'
    assert prov['matches_official_zenodo_md5'] is True
    split = d['split_structure']
    assert split['test_unique_reactions'] == 1573
    assert split['test_reaction_seen_in_train_count'] == 1573
    assert split['test_reaction_seen_in_train_fraction'] == 1.0
    assert split['raw_exact_train_test_sequence_overlap']['unique_exact_sequences'] == 3
    assert d['interpretation']['native_split_should_not_support_reaction_novel_discovery_claim'] is True


def test_paper_metric_set_and_values_are_exactly_aligned():
    d = load()
    a = d['paper_metric_alignment']
    assert a['common_metrics'] == ['map', 'ndcg@1', 'ndcg@5', 'top1', 'top5']
    assert a['map_definition_matches_standard_average_precision'] is True
    assert a['single_catalyst_run_vs_paper_five_run_mean'] is True
    assert a['paired_statistical_superiority_claim_allowed'] is False
    p = a['paper']
    assert p['e2r'] == {'map': .5156, 'ndcg@1': .4233, 'ndcg@5': .5152, 'top1': .4233, 'top5': .6636}
    assert p['r2e'] == {'map': .8211, 'ndcg@1': .721, 'ndcg@5': .8484, 'top1': .721, 'top5': .9425}


def test_large_result_survives_public_split_exact_overlap_removal():
    d = load()
    raw = d['catalyst_frozen_paper_aligned_metrics']['e2r']['map']
    strict = d['strict_remove_three_exact_overlap_posthoc']['e2r']['map']
    assert raw > .95 and strict > .95
    assert abs(raw - strict) < .001


def test_train_only_prototype_is_diagnostic_not_baseline():
    d = load()
    q = d['train_only_650m_prototype_diagnostic']
    assert q['status'] == 'post_reveal_descriptive_only_no_model_selection'
    assert q['score_construction_reads_test_labels'] is False
    assert q['e2r']['map'] > .85
    assert q['r2e']['map'] > .82
    assert 'unique authoritative external baseline' in d['baseline_policy']
