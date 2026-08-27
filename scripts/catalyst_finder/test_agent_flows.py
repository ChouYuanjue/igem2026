from __future__ import annotations

import unittest

from projects.active.terpene_screening.core.candidate_universes import (
    DEFAULT_CANDIDATE_UNIVERSE,
    TPS_SPECIALIZED_UNIVERSE,
)
from scripts.catalyst_finder.e2r_routing_graph import E2RRoutePlanner
from scripts.catalyst_finder.routing_graph import RoutePlanner
from scripts.catalyst_finder.route_view import build_e2r_route_view


class ConfirmedPositivePlannerTests(unittest.TestCase):
    def test_confirmed_description_match_can_authorize_seed_without_literal_id_in_text(self) -> None:
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "seed_mode": "explicit",
                "known_enzyme_ids": ["C8XPS0"],
                "homology_policy": "allow",
                "reason": "使用用户确认的阳性酶进行扩展。",
            },
            protein_ids={"C8XPS0", "A0A1W6QDI7"},
        )
        plan = planner.plan(
            user_text="已知阳性酶是丹参里的 miltiradiene synthase KSL1，请基于它扩展。",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            confirmed_known_ids=["C8XPS0"],
        )
        self.assertEqual(plan["known_enzyme_ids"], ["C8XPS0"])
        self.assertEqual(plan["seed_source"], "user_confirmed")
        self.assertEqual(plan["planned_route_id"], "r2e-current-top10-v1+fewshot")

    def test_user_confirmed_external_sequence_id_can_authorize_seed(self) -> None:
        external_id = "EXT-PROT-0123456789ABCDEF"
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "seed_mode": "explicit",
                "known_enzyme_ids": [external_id],
                "homology_policy": "allow",
                "reason": "Use the confirmed external sequence as a seed.",
            },
            protein_ids={"P12345"},
        )
        plan = planner.plan(
            user_text="Use the protein sequence I supplied as a known-active reference.",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=False,
            orientation="forward",
            confirmed_known_ids=[external_id],
        )
        self.assertEqual(plan["known_enzyme_ids"], [external_id])
        self.assertEqual(plan["seed_source"], "user_confirmed")
        self.assertEqual(plan["planned_route_id"], "r2e-external-top10-v1+fewshot")

    def test_unconfirmed_external_id_proposed_by_language_model_is_rejected(self) -> None:
        invented = "EXT-PROT-FFFFFFFFFFFFFFFF"
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "seed_mode": "explicit",
                "known_enzyme_ids": [invented],
                "homology_policy": "allow",
                "reason": "seed",
            },
            protein_ids={"P12345"},
        )
        plan = planner.plan(
            user_text="Find enzymes for this reaction.",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=False,
            orientation="forward",
            confirmed_known_ids=[],
        )
        self.assertEqual(plan["known_enzyme_ids"], [])
        self.assertEqual(plan["seed_mode"], "none")
        self.assertTrue(any("未确认酶 ID" in warning for warning in plan["warnings"]))

    def test_semantic_catalog_seed_does_not_require_keyword_reconfirmation(self) -> None:
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "_semantic_source": "deepseek",
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "seed_mode": "catalog_known",
                "known_enzyme_ids": [],
                "homology_policy": "allow",
                "known_association_policy": "allow_known",
                "reason": "Continue from the recorded active enzyme.",
            },
            protein_ids={"P12345"},
        )
        plan = planner.plan(
            user_text="Use that evidence and continue the search.",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["P12345"],
        )
        self.assertEqual(plan["known_enzyme_ids"], ["P12345"])
        self.assertEqual(plan["seed_source"], "catalog_known_associations")

    def test_semantic_remote_family_request_does_not_require_regex_match(self) -> None:
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "_semantic_source": "deepseek",
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "seed_mode": "none",
                "known_enzyme_ids": [],
                "homology_policy": "cross_cluster",
                "known_association_policy": "allow_known",
                "reason": "Search a more distant family than before.",
            },
            protein_ids={"P12345"},
        )
        plan = planner.plan(
            user_text="Try a substantially different family this time.",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["P12345"],
        )
        self.assertEqual(plan["homology_policy"], "cross_cluster")
        self.assertTrue(plan["homology_filter_applied"])

    def test_semantic_request_can_select_tps_specialized_candidate_universe(self) -> None:
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "_semantic_source": "deepseek",
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "seed_mode": "none",
                "known_enzyme_ids": [],
                "homology_policy": "allow",
                "known_association_policy": "allow_known",
                "candidate_universe": TPS_SPECIALIZED_UNIVERSE,
                "reason": "The user explicitly requested the project TPS-specialized library.",
            },
            protein_ids={"P12345"},
        )
        plan = planner.plan(
            user_text="Restrict this search to the project TPS-specialized candidate library.",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=False,
            orientation="forward",
        )
        self.assertEqual(plan["candidate_universe"], TPS_SPECIALIZED_UNIVERSE)

    def test_nonsemantic_proposal_cannot_narrow_candidate_universe(self) -> None:
        planner = RoutePlanner(
            proposal_fn=lambda *_: {
                "top_k": 10,
                "candidate_universe": TPS_SPECIALIZED_UNIVERSE,
                "reason": "narrow",
            },
            protein_ids={"P12345"},
        )
        plan = planner.plan(
            user_text="Find candidate enzymes.",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=False,
            orientation="forward",
        )
        self.assertEqual(plan["candidate_universe"], DEFAULT_CANDIDATE_UNIVERSE)


class E2RPlannerTests(unittest.TestCase):
    def planner(self, proposal):
        return E2RRoutePlanner(proposal_fn=lambda *_: proposal)

    def test_default_is_top10_and_keeps_known_activities(self) -> None:
        plan = self.planner({"top_k": 20}).plan(
            user_text="看看这个酶可能催化什么",
            route_mode="default",
            is_current=True,
            catalog_known_reactions=["RHEA:33983"],
        )
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["known_activity_policy"], "none")
        self.assertEqual(plan["known_association_policy"], "allow_known")
        self.assertEqual(plan["mask_reaction_ids"], [])
        self.assertFalse(plan["discovery_default_applied"])
        self.assertEqual(plan["candidate_universe"], DEFAULT_CANDIDATE_UNIVERSE)
        self.assertEqual(plan["planned_route_id"], "e2r-current-top10-v1")

    def test_default_without_known_activity_remains_plain_zero_shot(self) -> None:
        plan = self.planner({"top_k": 20}).plan(
            user_text="看看这个酶可能催化什么",
            route_mode="default",
            is_current=True,
            catalog_known_reactions=[],
        )
        self.assertEqual(plan["known_activity_policy"], "none")
        self.assertEqual(plan["planned_route_id"], "e2r-current-top10-v1")

    def test_natural_language_exclusion_forces_mask_even_if_ai_says_none(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "known_activity_policy": "none",
            "reason": "普通排序。",
        }).plan(
            user_text="请排除已经记录的反应，只返回未记录的可能功能",
            route_mode="intelligent",
            is_current=True,
            catalog_known_reactions=["RHEA:33983"],
        )
        self.assertEqual(plan["known_association_policy"], "exclude_known")
        self.assertEqual(plan["mask_reaction_ids"], ["RHEA:33983"])
        self.assertIn("+masked", plan["planned_route_id"])

    def test_explicit_keep_known_overrides_ai_mask(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "known_activity_policy": "mask_known",
            "reason": "mask",
        }).plan(
            user_text="请保留已知反应一起排序，不要排除已经记录的活性",
            route_mode="intelligent",
            is_current=True,
            catalog_known_reactions=["RHEA:33983"],
        )
        self.assertEqual(plan["known_association_policy"], "allow_known")
        self.assertEqual(plan["mask_reaction_ids"], [])
        self.assertEqual(plan["planned_route_id"], "e2r-current-top10-v1")

    def test_natural_language_can_request_known_only_reactions(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "known_activity_policy": "none",
            "known_association_policy": "allow_known",
            "reason": "普通排序。",
        }).plan(
            user_text="只看这个酶已经记录的反应，按模型分数排序",
            route_mode="intelligent",
            is_current=True,
            catalog_known_reactions=["RHEA:33983", "RHEA:54512"],
        )
        self.assertEqual(plan["known_association_policy"], "known_only")
        self.assertEqual(plan["mask_reaction_ids"], [])
        self.assertNotIn("+masked", plan["planned_route_id"])
        self.assertEqual(plan["known_association_policy_source"], "natural_language")

    def test_natural_language_known_only_scope_works_with_default_route(self) -> None:
        plan = self.planner({"top_k": 20}).plan(
            user_text="只看这个酶已经记录的反应",
            route_mode="default",
            is_current=True,
            catalog_known_reactions=["RHEA:33983"],
        )
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["known_association_policy"], "known_only")
        self.assertEqual(plan["mask_reaction_ids"], [])
        self.assertEqual(plan["known_association_policy_source"], "natural_language")

    def test_known_activity_seed_requires_explicit_intent(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "known_activity_policy": "seed_known",
            "reason": "从已有活性扩展。",
        }).plan(
            user_text="基于这个酶已有的已知反应继续扩展可能活性，Top 10",
            route_mode="intelligent",
            is_current=True,
            catalog_known_reactions=["RHEA:33983", "RHEA:54512"],
        )
        self.assertEqual(plan["known_reaction_ids"], ["RHEA:33983", "RHEA:54512"])
        self.assertEqual(plan["planned_route_id"], "e2r-current-top10-v1+fewshot")

    def test_semantic_known_activity_seed_does_not_require_keyword_reconfirmation(self) -> None:
        plan = self.planner({
            "_semantic_source": "deepseek",
            "top_k": 10,
            "known_activity_policy": "seed_known",
            "known_association_policy": "allow_known",
            "reason": "Continue from the previously discussed recorded activities.",
        }).plan(
            user_text="Do that expansion now.",
            route_mode="intelligent",
            is_current=True,
            catalog_known_reactions=["RHEA:33983"],
        )
        self.assertEqual(plan["known_reaction_ids"], ["RHEA:33983"])
        self.assertEqual(plan["known_activity_policy"], "seed_known")

    def test_e2r_semantic_request_can_select_tps_specialized_candidate_universe(self) -> None:
        plan = self.planner({
            "_semantic_source": "deepseek",
            "top_k": 10,
            "known_activity_policy": "none",
            "known_association_policy": "allow_known",
            "candidate_universe": TPS_SPECIALIZED_UNIVERSE,
            "reason": "Use the explicitly requested TPS-specialized library.",
        }).plan(
            user_text="Use only the project TPS-specialized reaction library for this query.",
            route_mode="intelligent",
            is_current=False,
            catalog_known_reactions=[],
        )
        self.assertEqual(plan["candidate_universe"], TPS_SPECIALIZED_UNIVERSE)

    def test_mask_known_is_filter_only(self) -> None:
        plan = self.planner({
            "top_k": 20,
            "known_activity_policy": "mask_known",
            "reason": "排除已有活性，探索新功能。",
        }).plan(
            user_text="排除这个酶已经知道的反应，只找新功能，Top 20",
            route_mode="intelligent",
            is_current=False,
            catalog_known_reactions=["RHEA:33983"],
        )
        self.assertEqual(plan["known_reaction_ids"], [])
        self.assertEqual(plan["mask_reaction_ids"], ["RHEA:33983"])
        self.assertIn("+masked", plan["planned_route_id"])

    def test_e2r_route_view_exposes_actual_model_chain(self) -> None:
        view = build_e2r_route_view(
            protein={"id": "EXT1", "name": "Protein", "organism": "Organism"},
            query={
                "route_id": "e2r-external-top10-neural-rrf-v1",
                "scope": "external",
                "shot_mode": "zero_shot",
                "ranking_objective": "top10",
                "candidate_universe_size": 753,
                "score_source": "rrf_e2r_top10_primary0.35_secondary0.65_c60",
            },
            routing={"top_k": 10, "known_activity_policy": "none", "reason": "balanced"},
            candidates=[{"candidate_id": "RHEA:33983"}],
        )
        ids = [row["id"] for row in view["nodes"]]
        self.assertIn("e2r-neighbor", ids)
        self.assertIn("e2r-hardneg", ids)
        self.assertIn("e2r-rrf10", ids)
        self.assertEqual(view["route_id"], "e2r-external-top10-neural-rrf-v1")


if __name__ == "__main__":
    unittest.main()
