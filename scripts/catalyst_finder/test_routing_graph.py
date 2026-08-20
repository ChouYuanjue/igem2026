from __future__ import annotations

import unittest

from scripts.catalyst_finder.routing_graph import RoutePlanner


class RoutePlannerTests(unittest.TestCase):
    proteins = {"A0A1W6QDI7", "A0A075FBG7", "G9MAN7"}

    def planner(self, proposal):
        return RoutePlanner(proposal_fn=lambda *_: proposal, protein_ids=self.proteins)

    def test_default_route_is_deterministic(self) -> None:
        plan = self.planner({"top_k": 20}).plan(
            user_text="给我候选酶",
            reaction_equation="A = B",
            route_mode="default",
            is_current=True,
            orientation="forward",
        )
        self.assertEqual(plan["selected_by"], "default")
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["enzyme_taxonomy_scope"], "all")
        self.assertEqual(plan["shot_mode"], "zero_shot")
        self.assertEqual(plan["homology_policy"], "allow")
        self.assertEqual(plan["known_association_policy"], "allow_known")
        self.assertEqual(plan["planned_route_id"], "r2e-current-top10-v1")


    def test_explicit_natural_language_can_exclude_known_associations(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "none",
            "homology_policy": "allow",
            "known_association_policy": "allow_known",
            "reason": "普通排序。",
        }).plan(
            user_text="请排除数据库里已经记录的催化酶，只看未记录候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["known_association_policy"], "exclude_known")

    def test_explicit_natural_language_can_request_known_only(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "none",
            "homology_policy": "allow",
            "known_association_policy": "allow_known",
            "reason": "普通排序。",
        }).plan(
            user_text="只看数据库里已经记录的催化酶，按模型分数排序",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7", "G9MAN7"],
        )
        self.assertEqual(plan["known_association_policy"], "known_only")
        self.assertEqual(plan["known_association_policy_source"], "natural_language")

    def test_natural_language_result_scope_also_applies_to_default_route(self) -> None:
        plan = self.planner({"top_k": 20}).plan(
            user_text="只看已经记录的催化酶，按模型分数排序",
            reaction_equation="A = B",
            route_mode="default",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["known_association_policy"], "known_only")
        self.assertEqual(plan["known_association_policy_source"], "natural_language")

    def test_ai_cannot_exclude_known_without_explicit_user_request(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "none",
            "homology_policy": "allow",
            "known_association_policy": "exclude_known",
            "reason": "try discovery",
        }).plan(
            user_text="给我总体排名最高的候选酶",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["known_association_policy"], "allow_known")
        self.assertTrue(any("保留已知关联" in warning for warning in plan["warnings"]))

    def test_ai_can_choose_supported_budget_and_taxonomy(self) -> None:
        plan = self.planner({
            "top_k": 20,
            "enzyme_taxonomy_scope": "eukaryote",
            "seed_mode": "none",
            "homology_policy": "allow",
            "reason": "需要更广的真核候选。",
        }).plan(
            user_text="只看真核，给我更广的候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=False,
            orientation="forward",
        )
        self.assertEqual(plan["selected_by"], "ai")
        self.assertEqual(plan["top_k"], 20)
        self.assertEqual(plan["enzyme_taxonomy_scope"], "eukaryote")
        self.assertEqual(plan["planned_route_id"], "r2e-external-top20-v1+eukaryote-only")

    def test_ai_cannot_invent_known_positive_ids(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "explicit",
            "known_enzyme_ids": ["A0A1W6QDI7", "FAKE123"],
            "homology_policy": "allow",
            "reason": "使用显式阳性酶。",
        }).plan(
            user_text="已知阳性酶 A0A1W6QDI7，请据此扩展",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
        )
        self.assertEqual(plan["known_enzyme_ids"], ["A0A1W6QDI7"])
        self.assertEqual(plan["seed_source"], "user_explicit")
        self.assertEqual(plan["shot_mode"], "few_shot")
        self.assertIn("+fewshot", plan["planned_route_id"])
        self.assertTrue(any("拒绝" in warning for warning in plan["warnings"]))

    def test_catalog_known_positives_require_explicit_user_intent(self) -> None:
        proposal = {
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "catalog_known",
            "known_enzyme_ids": [],
            "homology_policy": "allow",
            "reason": "use catalog seeds",
        }
        accepted = self.planner(proposal).plan(
            user_text="使用已有的已知阳性酶作为参考来扩展候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7", "G9MAN7"],
        )
        self.assertEqual(accepted["known_enzyme_ids"], ["A0A075FBG7", "G9MAN7"])
        self.assertEqual(accepted["seed_source"], "catalog_known_associations")
        self.assertEqual(accepted["shot_mode"], "few_shot")

        rejected = self.planner(proposal).plan(
            user_text="给我普通 Top 10",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(rejected["known_enzyme_ids"], [])
        self.assertEqual(rejected["shot_mode"], "zero_shot")

    def test_top5_uses_top10_route_family(self) -> None:
        plan = self.planner({
            "top_k": 5,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "none",
            "homology_policy": "allow",
            "reason": "用户要求五个候选。",
        }).plan(
            user_text="只要 Top 5",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
        )
        self.assertEqual(plan["top_k"], 5)
        self.assertEqual(plan["ranking_objective"], "top10")
        self.assertEqual(plan["planned_route_id"], "r2e-current-top10-v1")

    def test_remote_request_uses_catalog_positives_as_filter_only_anchors(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "none",
            "known_enzyme_ids": [],
            "homology_policy": "cross_cluster",
            "reason": "用户要求跨家族远缘候选。",
        }).plan(
            user_text="请主动排除近缘同源酶，我想找跨家族的远缘候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A1W6QDI7", "A0A075FBG7"],
        )
        self.assertEqual(plan["shot_mode"], "zero_shot")
        self.assertEqual(plan["homology_policy"], "cross_cluster")
        self.assertTrue(plan["homology_filter_requested"])
        self.assertTrue(plan["homology_filter_applied"])
        self.assertEqual(plan["homology_anchor_source"], "catalog_known_associations_filter_only")
        self.assertEqual(plan["homology_anchor_ids"], ["A0A1W6QDI7", "A0A075FBG7"])
        self.assertTrue(any("不会作为 Zero-shot" in warning for warning in plan["warnings"]))

    def test_cross_cluster_with_explicit_seed_is_seeded_remote_expansion(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "explicit",
            "known_enzyme_ids": ["A0A1W6QDI7"],
            "homology_policy": "cross_cluster",
            "reason": "使用 seed 但只返回跨簇候选。",
        }).plan(
            user_text="已知阳性 A0A1W6QDI7，请用它引导，但排除近缘同源，找远缘酶",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
        )
        self.assertEqual(plan["shot_mode"], "few_shot")
        self.assertTrue(plan["homology_filter_applied"])
        self.assertEqual(plan["homology_anchor_ids"], ["A0A1W6QDI7"])
        self.assertIn("+fewshot", plan["planned_route_id"])

    def test_remote_request_without_any_anchor_falls_back_to_homolog_allowed(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "none",
            "homology_policy": "cross_cluster",
            "reason": "remote",
        }).plan(
            user_text="只要远缘跨簇候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=False,
            orientation="forward",
        )
        self.assertEqual(plan["homology_policy"], "allow")
        self.assertFalse(plan["homology_filter_applied"])
        self.assertTrue(any("无法定义" in warning for warning in plan["warnings"]))

    def test_explicit_exclude_known_survives_ai_route_failure(self) -> None:
        def fail(*_):
            raise RuntimeError("offline")
        planner = RoutePlanner(proposal_fn=fail, protein_ids=self.proteins)
        plan = planner.plan(
            user_text="排除已经记录的催化酶，只看未记录候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["known_association_policy"], "exclude_known")
        self.assertEqual(plan["fallback_reason"], "ai_route_failed")

    def test_ai_failure_falls_back_without_blocking(self) -> None:
        def fail(*_):
            raise RuntimeError("offline")
        planner = RoutePlanner(proposal_fn=fail, protein_ids=self.proteins)
        plan = planner.plan(
            user_text="Top 20",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
        )
        self.assertEqual(plan["selected_by"], "default")
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["fallback_reason"], "ai_route_failed")


if __name__ == "__main__":
    unittest.main()
