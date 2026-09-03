from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def test_current_retrieval_status_is_the_single_human_entrypoint():
 text=(ROOT/'projects/active/terpene_screening/CURRENT_RETRIEVAL_STATUS.md').read_text()
 assert 'terpene-production-routes-v5' in text
 assert 'R2E LambdaRank' in text and 'E2R Anchored LambdaMART V3' in text
 assert 'Rhea release128→141' in text and '185,918 proteins' in text and '11,081 reactions' in text
 assert '0.02988' in text and '0.02746' in text
 assert '92.90%' in text and '此前跑出的 E2R MRR≈0.71 / Hit@10≈93% 已删除' in text
 assert '大规模新外部编码' in text and '默认不做' in text

def test_old_strongest_doc_is_only_a_pointer():
 text=(ROOT/'projects/active/terpene_screening/CURRENT_STRONGEST_VS_ENZYMECAGE.md').read_text()
 assert '不再是当前状态文档' in text
 assert 'CURRENT_RETRIEVAL_STATUS.md' in text
 assert len(text.splitlines()) < 20


def test_budgeted_best_seed_policy_is_explicit():
 import json
 policy=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_EXTERNAL_EVALUATION_POLICY_V2.json').read_text())
 b=policy['budgeted_presentation']
 assert b['seed_count']==8 and b['reverse_seed_search_allowed'] is False and b['all_seed_results_required'] is True
 assert b['r2e']['primary_seed']==2025598660
 assert b['e2r']['primary_seed']==4254708239
 assert b['tps_r2e']['primary_seed']==3734383874
 assert b['tps_e2r']['primary_seed']==2327310358
 text=(ROOT/'projects/active/terpene_screening/CURRENT_RETRIEVAL_STATUS.md').read_text()
 for value in ('2025598660','4254708239'):
  assert value in text
 assert 'best-of-8' in text and 'Open-world temporal' in text and 'Production' in text


def test_legacy_style_hitk_presentation_is_distinct_from_external_headline():
 import json
 d=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_LEGACY_STYLE_HITK_PRESENTATION_V1.json').read_text())
 assert d['selection_allowed'] is False and d['model_or_hyperparameter_changed'] is False
 assert d['r2e']['best_of_8']['hit50_current'] > 0.50
 assert d['e2r']['best_of_8']['hit50_current'] > 0.50
 text=(ROOT/'projects/active/terpene_screening/CURRENT_RETRIEVAL_STATUS.md').read_text()
 assert '旧式 Hit@K 口径复测' in text and '52.73%' in text and '51.56%' in text


def test_retrieval_capability_scorecard_separates_tps_and_general_metrics():
 import json
 d=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_RETRIEVAL_CAPABILITY_SCORECARD_V1.json').read_text())
 batches={x['id']:x for x in d['batches']}
 tps=batches['A_tps_exploitation']
 assert tps['candidate_universe']=={'reactions':513,'proteins':1391}
 assert tps['current_best']['hit10'] > tps['internal_historical_comparator']['hit10']
 assert tps['current_best']['hit20'] > 0.55
 general=batches['E_general_open_retrieval']
 assert general['candidate_universe']=={'R2E_proteins':185918,'E2R_reactions':11081}
 assert general['R2E']['current_success_at_0p1pct'] > general['R2E']['baseline_success_at_0p1pct']
 assert general['E2R']['current_success_at_0p2pct'] > general['E2R']['baseline_success_at_0p2pct']
 text=(ROOT/'projects/active/terpene_screening/RETRIEVAL_CAPABILITY_SCORECARD.md').read_text()
 assert 'TPS 专项：数据库补全' in text and '通用能力：大候选宇宙' in text
 assert 'Success@0.1%' in text and 'Success@0.2%' in text


def test_evidence_ledger_keeps_internal_rf_cage_separate_from_completed_pure_cage_baseline():
 import json
 ledger=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_RETRIEVAL_EVIDENCE_LEDGER_V1.json').read_text())
 entries={x['id']:x for x in ledger['entries']}
 full=entries['tps_current_library_r2e_full513']
 assert full['class']=='E_capability_slice'
 assert full['internal_history']['external_baseline'] is False
 assert full['external_baseline_status']=='SEPARATE_COMMON_SUPPORT_ENTRY'
 ext=entries['tps_practical_pure_enzymecage_common459']
 assert ext['class']=='A_local_external_same_support'
 assert ext['external_baseline']['author_original_tps_gate_file'] is False
 assert ext['delta']['hit10_pp'] > 15
 assert ext['delta']['hit20_pp'] > 15
 assert ext['delta']['hit10_bootstrap95_pp'][0] > 0
 assert ext['applicability']['catalyst_reaction_coverage'] == 1.0
 assert ext['applicability']['enzymecage_gate_common_evaluable_reactions'] == 459
 marts=entries['tps_current_marts_1421x453_bidirectional']
 assert marts['r2e']['finalization_status']=='finalized_locked_route_confirmation'
 assert marts['e2r']['finalization_status']=='finalized_independent_route_confirmation'
 assert marts['r2e']['confirmed_mrr'] > marts['r2e']['previous_mrr']
 assert marts['e2r']['confirmed_fused_hit20'] > marts['e2r']['previous_production_hit20']
 assert 'enzgfm_native_reactzyme' in ledger['paper_not_locally_reproduced']
 score=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_RETRIEVAL_CAPABILITY_SCORECARD_V1.json').read_text())
 a={x['id']:x for x in score['batches']}['A_tps_exploitation']
 assert a['external_baseline']['identity'].startswith('pure EnzymeCAGE')
 assert a['external_baseline']['delta_pp']['hit10'] > 15
 assert a['external_baseline']['delta_pp']['hit20'] > 15
