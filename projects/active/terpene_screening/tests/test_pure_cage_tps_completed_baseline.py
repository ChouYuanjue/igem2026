import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_pure_cage_tps_result_is_external_same_support_and_separates_applicability():
    d = json.loads((ROOT / 'projects/active/terpene_screening/CATALYST_TPS_PURE_CAGE_APPLICABILITY_BASELINE_V1_RESULT.json').read_text())
    main = d['main_external_comparison']
    assert main['support'] == {'reactions': 459, 'proteins': 1379, 'why_459': main['support']['why_459']}
    cage = main['enzymecage_official_algorithm_reproduction']
    cat = main['catalyst_locked_practical_route']
    assert cat['hit10'] > cage['hit10'] and main['delta']['hit10_pp'] > 15
    assert cat['hit20'] > cage['hit20'] and main['delta']['hit20_pp'] > 15
    assert main['catalyst_reselection_on_this_support'] is False
    app = d['applicability']
    assert app['enzymecage']['fully_evaluable_reactions'] == 462
    assert app['enzymecage']['archived_gate_common_evaluable_reactions'] == 459
    assert app['enzymecage']['canonicalization_failures'] == 44
    assert app['catalyst']['reaction_coverage'] == 1.0
    assert d['neural_scorer_full_matrix_diagnostic']['headline_baseline'] is False
    assert d['claim_boundaries']['rf_hgb_cage_is_external_baseline'] is False
    assert d['claim_boundaries']['author_original_tps_gate_claimed'] is False
    assert main['baseline_provenance']['author_original_tps_gate_file'] is False
    assert main['paired_stability']['hit10']['bootstrap95_delta_pp'][0] > 0
    assert main['paired_stability']['hit20']['bootstrap95_delta_pp'][0] > 0
