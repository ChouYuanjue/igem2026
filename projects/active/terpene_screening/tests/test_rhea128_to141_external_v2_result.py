import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2_RESULT.json'
def test_fresh_external_result_is_powered_failed_and_frozen():
 r=json.loads(PATH.read_text())
 assert r['status']=='failed_fresh_external_snapshot_no_retuning' and r['pass'] is False
 assert r['support']['met'] is True and r['support']['observed_query_reactions']==208 and r['support']['observed_test_pairs']==1122
 assert r['model_selection_allowed_after_external_reveal'] is False
 assert r['benchmark_audit']['exact_train_test_pair_overlap']==0
 assert r['benchmark_audit']['train_test_protein_overlap']==0
 assert r['benchmark_audit']['train_test_reaction_overlap']==0
 assert r['delta']['mrr']<0 and r['delta']['map']<0 and r['delta']['ndcg_at_10']<0 and r['delta']['hit_at_10']<0
 assert r['delta']['macro_roc_auc']>0 and r['delta']['hit_at_50']>0
 assert sum(bool(v) for v in r['checks'].values())==2

def test_result_does_not_reopen_external_selection():
 r=json.loads(PATH.read_text())
 assert 'Do not tune' in r['post_reveal_policy']
 assert 'frozen full-clean2023 RDKit+ base is therefore the stronger model' in r['interpretation']
