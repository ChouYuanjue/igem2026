from __future__ import annotations

from types import SimpleNamespace

import scripts.starase_navigator.retrieval.gateway as gateway_module
from projects.active.bridge.core.candidate_universes import (
    DEFAULT_CANDIDATE_UNIVERSE,
    MARTS_CORRESPONDENCE_UNIVERSE,
)
from scripts.starase_navigator.retrieval.gateway import ModelGateway
from scripts.starase_navigator.routing.enzyme_to_reaction import E2RRoutePlanner
from scripts.starase_navigator.routing.reaction_to_enzyme import RoutePlanner


class FakeCorrespondence:
    def __init__(self):
        self.calls = []

    def rank(self, command, payload):
        self.calls.append((command, dict(payload)))
        return {
            "query": {"score_source": "correspondence_geometry"},
            "candidates": [{"candidate_id": "X", "score": 0.0}],
        }

    def contains_protein(self, value):
        return value == "P-MARTS"

    def contains_reaction(self, value):
        return value == "R-MARTS"


class FakeFinalBridge:
    def __init__(self, known=()):
        self.known = set(known)
        self.calls = []

    def contains(self, command, query_id):
        return query_id in self.known

    def rank(self, command, payload):
        self.calls.append((command, dict(payload)))
        return {
            "query": {
                "route_id": "bridge-final-r2e-v1",
                "model_bundle_version": "bridge-final-v1",
            },
            "candidates": [{"candidate_id": "P1", "score": 1.0}],
        }

    def status(self):
        return {
            "status": "ready",
            "version": "bridge-final-v1",
            "protein_candidates": 185918,
            "reaction_candidates": 11081,
        }


class FakeEngine:
    def __init__(self):
        self.calls = []

    def rank(self, command, payload):
        self.calls.append((command, dict(payload)))
        return {"query": {}, "candidates": []}


def test_gateway_marts_scope_is_explicit_compatibility_only(monkeypatch):
    gateway = ModelGateway()
    correspondence = FakeCorrespondence()
    final = FakeFinalBridge({"R-MARTS"})
    engine = FakeEngine()
    gateway._correspondence_service = correspondence
    gateway._final_bridge = final
    gateway._engine = engine
    monkeypatch.setattr(
        gateway_module,
        "route_payload",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("general router must not run")
        ),
    )
    result = gateway.rank(
        "rank-enzymes",
        {
            "candidate_universe": MARTS_CORRESPONDENCE_UNIVERSE,
            "reaction_id": "R-MARTS",
        },
    )
    assert len(correspondence.calls) == 1
    assert final.calls == []
    assert engine.calls == []
    assert result["query"]["candidate_universe"] == MARTS_CORRESPONDENCE_UNIVERSE


def test_gateway_registered_general_entity_uses_final_bridge(monkeypatch):
    gateway = ModelGateway()
    correspondence = FakeCorrespondence()
    final = FakeFinalBridge({"R-BRIDGE"})
    engine = FakeEngine()
    gateway._correspondence_service = correspondence
    gateway._final_bridge = final
    gateway._engine = engine
    monkeypatch.setattr(
        gateway_module,
        "route_payload",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("fallback router must not run")
        ),
    )
    result = gateway.rank(
        "rank-enzymes",
        {
            "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
            "reaction_id": "R-BRIDGE",
        },
    )
    assert len(final.calls) == 1
    assert correspondence.calls == []
    assert engine.calls == []
    assert result["query"]["model_expert"] == "bridge_final"
    assert result["query"]["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE
    assert result["query"]["candidate_universe_specialized"] is False


def test_gateway_unregistered_general_entity_uses_current_production_manifest(monkeypatch):
    gateway = ModelGateway()
    gateway._correspondence_service = FakeCorrespondence()
    gateway._final_bridge = FakeFinalBridge()
    engine = FakeEngine()
    gateway._engine = engine
    decision = SimpleNamespace(
        expert="production_manifest",
        reason="current",
        ranking_objective="top10",
    )
    monkeypatch.setattr(
        gateway_module,
        "route_payload",
        lambda command, payload: ({**payload, "routed": True}, decision),
    )
    result = gateway.rank(
        "rank-enzymes",
        {
            "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
            "reaction_id": "R-EXTERNAL",
        },
    )
    assert engine.calls[0][1]["routed"] is True
    assert result["query"]["model_expert"] == "production_manifest"


def test_gateway_final_bridge_status_exposes_authoritative_full_universe():
    gateway = ModelGateway()
    gateway._final_bridge = FakeFinalBridge()
    status = gateway.final_bridge_status()
    assert status["status"] == "ready"
    assert status["integrity_verified"] is True
    assert status["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE
    assert status["routing_role"] == "authoritative_registered_general_retrieval"


def test_gateway_exposes_correspondence_membership_without_engine():
    gateway = ModelGateway()
    gateway._correspondence_service = FakeCorrespondence()
    engine = FakeEngine()
    gateway._engine = engine
    assert gateway.correspondence_contains_protein("P-MARTS") is True
    assert gateway.correspondence_contains_reaction("R-MARTS") is True
    assert engine.calls == []


def test_r2e_application_domain_keeps_full_bridge_candidate_universe():
    planner = RoutePlanner(
        proposal_fn=lambda *_a, **_k: {
            "_semantic_source": "deepseek",
            "top_k": 10,
            "retrieval_scope": "application_domain",
            "analysis_depth": "deep",
            "seed_mode": "none",
            "known_association_policy": "separate_known",
            "reason": "verified target fits the application domain",
        },
        protein_ids={"P1"},
    )
    plan = planner.plan(
        user_text="Find the most plausible catalysts for this terpene cyclization.",
        reaction_equation="geranylgeranyl diphosphate -> diterpene product",
        route_mode="intelligent",
        is_current=False,
        orientation="forward",
    )
    assert plan["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE
    assert plan["candidate_universe_source"] == "deepseek_semantic_scope"
    assert plan["retrieval_scope"] == "application_domain"
    assert plan["analysis_depth"] == "deep"


def test_e2r_application_domain_keeps_full_bridge_candidate_universe():
    planner = E2RRoutePlanner(
        proposal_fn=lambda *_a, **_k: {
            "_semantic_source": "deepseek",
            "top_k": 10,
            "retrieval_scope": "application_domain",
            "analysis_depth": "deep",
            "seed_mode": "none",
            "known_association_policy": "separate_known",
            "reason": "verified target fits the application domain",
        }
    )
    plan = planner.plan(
        user_text="What terpene-forming reactions might this synthase catalyze?",
        route_mode="intelligent",
        is_current=False,
        target_context={"protein": {"name": "terpene synthase"}},
    )
    assert plan["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE
    assert plan["candidate_universe_source"] == "deepseek_semantic_scope"
    assert plan["retrieval_scope"] == "application_domain"
    assert plan["analysis_depth"] == "deep"


def test_current_application_defaults_do_not_shrink_candidate_universe():
    r2e = RoutePlanner(
        proposal_fn=lambda *_a, **_k: {
            "_semantic_source": "deepseek",
            "top_k": 10,
            "analysis_depth": "standard",
            "seed_mode": "none",
            "known_association_policy": "separate_known",
        },
        protein_ids={"P1"},
    )
    rplan = r2e.plan(
        user_text="Find candidates.",
        reaction_equation="A = B",
        route_mode="intelligent",
        is_current=True,
        orientation="forward",
    )
    assert rplan["retrieval_scope"] == "application_domain"
    assert rplan["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE

    e2r = E2RRoutePlanner(
        proposal_fn=lambda *_a, **_k: {"retrieval_scope": "broad", "analysis_depth": "deep"}
    )
    eplan = e2r.plan(
        user_text="What else might this enzyme catalyze?",
        route_mode="intelligent",
        is_current=True,
        catalog_known_reactions=[],
    )
    assert eplan["retrieval_scope"] == "application_domain"
    assert eplan["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE


def test_nonsemantic_proposal_cannot_change_verified_scope_or_pool():
    planner = RoutePlanner(
        proposal_fn=lambda *_a, **_k: {
            "retrieval_scope": "application_domain",
            "analysis_depth": "deep",
        },
        protein_ids={"P1"},
    )
    plan = planner.plan(
        user_text="terpene",
        reaction_equation="A=B",
        route_mode="intelligent",
        is_current=False,
        orientation="forward",
    )
    assert plan["candidate_universe"] == DEFAULT_CANDIDATE_UNIVERSE
    assert plan["candidate_universe_source"] == "semantic_scope_default"
    assert plan["retrieval_scope"] == "broad"
    assert plan["analysis_depth"] == "standard"
