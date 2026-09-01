import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RESULT=ROOT/'projects/active/terpene_screening/REACTZYME_NATIVE_BAG_ADAPTER_V1_CONFIRMATION_RESULT.json'
def test_confirmation_result_promotes_only_internal_capability():
 d=json.loads(RESULT.read_text())
 assert d['status']=='confirmed'
 assert d['decision']=='promote_native_molecule_bag_capability_for_production_packaging'
 assert d['external_reactzyme_metrics_used_for_selection'] is False
 assert d['confirmation_retraining_allowed'] is False
 assert d['split']['dev_reactions']==675
 assert d['split']['protein_overlap']==d['split']['reaction_overlap']==0
 assert all(d['metrics']['checks'].values())
def test_confirmation_retains_ranking_strongly():
 d=json.loads(RESULT.read_text()); r=d['metrics']['retention']
 assert r['mrr']>=0.90 and r['map']>=0.90 and r['hit_at_10']>=0.88 and r['hit_at_50']>=0.90
 assert d['metrics']['teacher_cosine_mean']>=0.80
 assert d['metrics']['median_best_positive_rank_ratio']<=1.50
