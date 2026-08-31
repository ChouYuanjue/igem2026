from projects.active.terpene_screening.evaluate_reaction_center_bounded_v3_confirmation import gate

def report(delta=0.01):
 b={'mrr':.1,'map':.08,'macro_roc_auc':.9,'ndcg_at_10':.1,'hit_at_10':.2,'hit_at_20':.3,'hit_at_50':.4,'median_best_positive_rank':100.}
 c={k:(v+delta if k!='median_best_positive_rank' else v-10) for k,v in b.items()}
 return {'all':{'baseline':b,'candidate':c},'lt0p3':{'baseline':b,'candidate':c}}
def test_confirmation_gate_passes_strict_positive_candidate(): assert gate(report())['pass'] is True
def test_confirmation_gate_rejects_all_query_regression():
 r=report(); r['all']['candidate']['hit_at_20']=r['all']['baseline']['hit_at_20']-.001
 g=gate(r); assert g['checks']['all_hit20_no_regress'] is False and g['pass'] is False
