from projects.active.terpene_screening.run_rhea128_to141_external_v2 import support_decision

def test_support_gate_scores_only_when_both_frozen_minima_are_met():
 p={'minimum_support_rule':{'min_query_reactions':50,'min_test_pairs':200}}
 good=support_decision({'audit':{'test_query_reactions':50,'test_pairs':200}},p)
 assert good['minimum_support_met'] is True and good['action']=='score_once_with_frozen_models'
 for audit in ({'test_query_reactions':49,'test_pairs':1000},{'test_query_reactions':100,'test_pairs':199}):
  bad=support_decision({'audit':audit},p)
  assert bad['minimum_support_met'] is False and bad['action']=='underpowered_stop_without_model_scoring'
