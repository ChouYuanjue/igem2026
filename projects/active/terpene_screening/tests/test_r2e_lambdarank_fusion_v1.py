from pathlib import Path
import json
import numpy as np
from projects.active.terpene_screening.run_r2e_lambdarank_fusion_v1 import FEATURE_NAMES,_build_features,_configs,_full_order
ROOT=Path(__file__).resolve().parents[4]

def test_protocol_is_frozen_and_search_is_automatic():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_R2E_LAMBDARANK_FUSION_V1.json').read_text())
 assert p['status']=='frozen_before_learned_reranker_development_results'
 assert p['automatic_search']['generator']=='sklearn.model_selection.ParameterSampler'
 assert p['development']['outer_or_external_labels_used'] is False
 assert p['confirmation']['single_run_no_retuning_after_reveal'] is True

def test_full_order_uses_score_then_lexical_tie_break():
 scores=np.array([.5,.5,.2],dtype=np.float32); lex=np.array([1,0,2],dtype=np.int32)
 order,inv=_full_order(scores,lex)
 assert order.tolist()==[1,0,2] and inv.tolist()==[2,1,3]

def test_feature_contract_has_no_candidate_identity_and_routes_fallback_columns():
 s0=np.array([.9,.8,.1],dtype=np.float32); s1=np.array([.1,.7,.8],dtype=np.float32); rows=np.array([0,1,2]); r0=np.array([1,2,3]); r1=np.array([3,2,1])
 x=_build_features(s0,s1,rows,r0,r1,True,.4)
 assert x.shape==(3,len(FEATURE_NAMES))
 assert 'candidate_id' not in FEATURE_NAMES
 assert np.allclose(x[:,-2],.4) and np.allclose(x[:,-1],1.)
 # secondary is fallback, so fallback rank/log column equals secondary log-rank column
 assert np.allclose(x[:,12],x[:,5]) and np.allclose(x[:,13],x[:,4])

def test_search_configurations_are_deterministic_and_include_anchor():
 a=_configs(); b=_configs(); assert [x.__dict__ for x in a]==[x.__dict__ for x in b]
 assert len(a)>=19
 assert any(x.pool_k==50 and x.prefix_k==50 and x.max_depth==3 and x.learning_rate==.05 and x.rounds==80 for x in a)
