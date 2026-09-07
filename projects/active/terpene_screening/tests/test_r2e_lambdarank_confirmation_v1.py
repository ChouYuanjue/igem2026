import json
from pathlib import Path
import numpy as np
from projects.active.terpene_screening.evaluate_r2e_lambdarank_fusion_v1_confirmation import apply_prefix_rerank_positive_ranks
from projects.active.terpene_screening.run_r2e_lambdarank_fusion_v1 import Config
ROOT=Path(__file__).resolve().parents[4]

class FakeDMatrix:
    def __init__(self,x): self.x=x
class FakeBooster:
    def predict(self,dm):
        # Prefer larger primary raw score (feature 0), deterministically.
        return dm.x[:,0]

def test_frozen_confirmation_contract_is_new_salt_and_single_run():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_R2E_LAMBDARANK_FUSION_V1.json').read_text())
 assert p['confirmation']['salt']=='r2e_lambdarank_fusion_v1_confirm_20260902'
 assert p['confirmation']['fold']==6 and p['confirmation']['folds']==7
 assert p['confirmation']['single_run_no_retuning_after_reveal'] is True

def test_prefix_rerank_preserves_tail_relative_order(monkeypatch):
 import projects.active.terpene_screening.evaluate_r2e_lambdarank_fusion_v1_confirmation as mod
 monkeypatch.setattr(mod.xgb,'DMatrix',FakeDMatrix)
 cfg=Config('x',3,2,2,.1,1.,1.,10,8)
 p=np.array([.9,.8,.7,.6,.5],dtype=np.float32)
 s=np.array([.1,.2,.3,.4,.5],dtype=np.float32)
 lex=np.arange(5,dtype=np.int32)
 positives=np.array([0,2,4],dtype=np.int32)
 base,cand,a=apply_prefix_rerank_positive_ranks(primary_scores=p,secondary_scores=s,positive_rows=positives,lexical_rank=lex,similarity=.95,booster=FakeBooster(),config=cfg)
 assert base.tolist()==[1,3,5]
 # Candidate 0 stays first; candidate 2 is not selected after candidate1 is promoted and keeps tail-relative ordering.
 assert cand.tolist()==[1,3,5]
 assert a['selected_prefix_size']==2

def test_low_similarity_uses_secondary_as_fallback(monkeypatch):
 import projects.active.terpene_screening.evaluate_r2e_lambdarank_fusion_v1_confirmation as mod
 monkeypatch.setattr(mod.xgb,'DMatrix',FakeDMatrix)
 cfg=Config('x',2,1,2,.1,1.,1.,10,8)
 p=np.array([.9,.8,.1],dtype=np.float32); s=np.array([.1,.8,.9],dtype=np.float32); lex=np.arange(3,dtype=np.int32)
 base,_,a=apply_prefix_rerank_positive_ranks(primary_scores=p,secondary_scores=s,positive_rows=np.array([2],dtype=np.int32),lexical_rank=lex,similarity=.2,booster=FakeBooster(),config=cfg)
 assert base.tolist()==[1] and a['use_secondary_fallback'] is True
