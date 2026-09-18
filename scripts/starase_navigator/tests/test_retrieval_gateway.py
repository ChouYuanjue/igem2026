from __future__ import annotations

from types import SimpleNamespace

import scripts.starase_navigator.retrieval.gateway as gateway_module
from projects.active.fibre.core.candidate_universes import DEFAULT_CANDIDATE_UNIVERSE, MARTS_CORRESPONDENCE_UNIVERSE
from scripts.starase_navigator.routing.enzyme_to_reaction import E2RRoutePlanner
from scripts.starase_navigator.retrieval.gateway import ModelGateway
from scripts.starase_navigator.routing.reaction_to_enzyme import RoutePlanner

class FakeCorrespondence:
    def __init__(self): self.calls=[]
    def rank(self,command,payload):
        self.calls.append((command,dict(payload)))
        return {'query':{'score_source':'correspondence_geometry'},'candidates':[{'candidate_id':'X','score':0.0}]}
    def contains_protein(self,value): return value=='P-MARTS'
    def contains_reaction(self,value): return value=='R-MARTS'

class FakeEngine:
    def __init__(self): self.calls=[]
    def rank(self,command,payload):
        self.calls.append((command,dict(payload)))
        return {'query':{},'candidates':[]}

def test_gateway_marts_scope_bypasses_legacy_engine(monkeypatch):
    gateway=ModelGateway(); correspondence=FakeCorrespondence(); engine=FakeEngine()
    gateway._correspondence_service=correspondence; gateway._engine=engine
    monkeypatch.setattr(gateway_module,'route_payload',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('legacy router must not run')))
    result=gateway.rank('rank-enzymes',{'candidate_universe':MARTS_CORRESPONDENCE_UNIVERSE,'reaction_id':'R-MARTS'})
    assert len(correspondence.calls)==1 and engine.calls==[]
    assert result['query']['model_expert']=='fibre'
    assert result['query']['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE

def test_gateway_general_scope_keeps_existing_router_and_engine(monkeypatch):
    gateway=ModelGateway(); correspondence=FakeCorrespondence(); engine=FakeEngine()
    gateway._correspondence_service=correspondence; gateway._engine=engine
    decision=SimpleNamespace(expert='old',reason='existing',ranking_objective='top10')
    monkeypatch.setattr(gateway_module,'route_payload',lambda command,payload:({**payload,'routed':True},decision))
    result=gateway.rank('rank-enzymes',{'candidate_universe':DEFAULT_CANDIDATE_UNIVERSE,'reaction_id':'R'})
    assert correspondence.calls==[] and engine.calls[0][1]['routed'] is True
    assert result['query']['model_expert']=='old'

def test_gateway_exposes_correspondence_membership_without_engine():
    gateway=ModelGateway(); gateway._correspondence_service=FakeCorrespondence(); engine=FakeEngine(); gateway._engine=engine
    assert gateway.correspondence_contains_protein('P-MARTS') is True
    assert gateway.correspondence_contains_reaction('R-MARTS') is True
    assert engine.calls==[]

def test_r2e_semantic_planner_can_choose_best_application_route_without_architecture_terms():
    planner=RoutePlanner(proposal_fn=lambda *_a,**_k:{'_semantic_source':'deepseek','top_k':10,'retrieval_scope':'application_domain','analysis_depth':'deep','seed_mode':'none','known_association_policy':'separate_known','reason':'verified target fits the focused application domain'},protein_ids={'P1'})
    plan=planner.plan(user_text='Find the most plausible catalysts for this terpene cyclization and use the strongest in-domain evidence available.',reaction_equation='geranylgeranyl diphosphate -> diterpene product',route_mode='intelligent',is_current=False,orientation='forward')
    assert plan['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE
    assert plan['candidate_universe_source']=='deepseek_semantic_scope'
    assert plan['retrieval_scope']=='application_domain'
    assert plan['analysis_depth']=='deep'

def test_e2r_semantic_planner_can_choose_best_application_route_without_architecture_terms():
    planner=E2RRoutePlanner(proposal_fn=lambda *_a,**_k:{'_semantic_source':'deepseek','top_k':10,'retrieval_scope':'application_domain','analysis_depth':'deep','seed_mode':'none','known_association_policy':'separate_known','reason':'verified target fits the focused application domain'})
    plan=planner.plan(user_text='What terpene-forming reactions might this synthase catalyze? Use the best evidence available.',route_mode='intelligent',is_current=False,target_context={'protein':{'name':'terpene synthase'}})
    assert plan['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE
    assert plan['candidate_universe_source']=='deepseek_semantic_scope'
    assert plan['retrieval_scope']=='application_domain'
    assert plan['analysis_depth']=='deep'

def test_nonsemantic_proposal_cannot_narrow_to_marts_scope():
    planner=RoutePlanner(proposal_fn=lambda *_a,**_k:{'retrieval_scope':'application_domain','analysis_depth':'deep'},protein_ids={'P1'})
    plan=planner.plan(user_text='terpene',reaction_equation='A=B',route_mode='intelligent',is_current=False,orientation='forward')
    assert plan['candidate_universe']==DEFAULT_CANDIDATE_UNIVERSE
    assert plan['candidate_universe_source']=='semantic_scope_default'
    assert plan['retrieval_scope']=='broad'
    assert plan['analysis_depth']=='standard'
