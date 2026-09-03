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
 for value in ('2025598660','4254708239','3734383874','2327310358'):
  assert value in text
 assert 'best-of-8' in text and 'Open-world temporal' in text and 'Production' in text
