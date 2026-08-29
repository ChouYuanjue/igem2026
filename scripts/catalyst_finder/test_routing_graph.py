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
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["selected_by"], "default")
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["enzyme_taxonomy_scope"], "all")
        self.assertEqual(plan["known_enzyme_ids"], ["A0A075FBG7"])
        self.assertEqual(plan["shot_mode"], "few_shot")
        self.assertEqual(plan["homology_policy"], "allow")
        self.assertEqual(plan["known_association_policy"], "separate_known")
        self.assertEqual(plan["planned_route_id"], "r2e-current-top10-v1+fewshot")


    def test_explicit_mixed_ranking_forces_zero_shot_even_with_catalog_positives(self) -> None:
        plan = self.planner({
            "_semantic_source": "deepseek",
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "catalog_known",
            "known_enzyme_ids": [],
            "homology_policy": "allow",
            "known_association_policy": "rank_with_known",
            "reason": "Retrospective mixed ranking.",
        }).plan(
            user_text="把数据库已知酶和未知候选放到同一个榜单里一起排序",
            reaction_equation="A = B", route_mode="intelligent", is_current=True, orientation="forward",
            known_association_ids=["A0A075FBG7", "G9MAN7"],
        )
        self.assertEqual(plan["known_association_policy"], "rank_with_known")
        self.assertEqual(plan["known_enzyme_ids"], [])
        self.assertEqual(plan["seed_mode"], "none")
        self.assertEqual(plan["seed_source"], "mixed_ranking_forces_zero_shot")
        self.assertEqual(plan["shot_mode"], "zero_shot")
        self.assertNotIn("+fewshot", plan["planned_route_id"])

    def test_show_both_restores_layered_default_not_mixed_ranking(self) -> None:
        plan = self.planner({"_semantic_source": "deepseek", "known_association_policy": "separate_known"}).plan(
            user_text="恢复默认，把已知证据和新关联候选都显示出来",
            reaction_equation="A = B", route_mode="intelligent", is_current=True, orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["known_association_policy"], "separate_known")
        self.assertEqual(plan["shot_mode"], "few_shot")
        self.assertEqual(plan["known_enzyme_ids"], ["A0A075FBG7"])

    def test_semantic_policy_can_exclude_known_associations(self) -> None:
        plan = self.planner({
            "_semantic_source": "deepseek",
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "catalog_known",
            "homology_policy": "allow",
            "known_association_policy": "exclude_known",
            "reason": "用户只要未记录候选。",
        }).plan(
            user_text="请排除数据库里已经记录的催化酶，只看未记录候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["known_association_policy"], "exclude_known")

    def test_semantic_policy_can_request_known_only(self) -> None:
        plan = self.planner({
            "_semantic_source": "deepseek",
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "catalog_known",
            "homology_policy": "allow",
            "known_association_policy": "known_only",
            "reason": "用户只看已记录催化酶。",
        }).plan(
            user_text="只看数据库里已经记录的催化酶，按模型分数排序",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7", "G9MAN7"],
        )
        self.assertEqual(plan["known_association_policy"], "known_only")
        self.assertEqual(plan["known_association_policy_source"], "deepseek_semantic")
        self.assertEqual(plan["known_enzyme_ids"], [])
        self.assertEqual(plan["seed_mode"], "none")
        self.assertEqual(plan["seed_source"], "known_only_scoring_forces_zero_shot")
        self.assertEqual(plan["shot_mode"], "zero_shot")

    def test_default_route_does_not_infer_result_scope_from_language(self) -> None:
        plan = self.planner({"top_k": 20}).plan(
            user_text="只看已经记录的催化酶，按模型分数排序",
            reaction_equation="A = B",
            route_mode="default",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["top_k"], 10)
        self.assertEqual(plan["known_association_policy"], "separate_known")
        self.assertEqual(plan["known_association_policy_source"], "default_fallback")

    def test_nonsemantic_proposal_cannot_change_association_scope(self) -> None:
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
        self.assertEqual(plan["known_association_policy"], "separate_known")
        self.assertTrue(any("语义策略" in warning for warning in plan["warnings"]))

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

    def test_catalog_known_positives_are_default_few_shot_context(self) -> None:
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

        ordinary = self.planner({**proposal, "seed_mode": "none"}).plan(
            user_text="给我普通 Top 10",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(ordinary["known_enzyme_ids"], ["A0A075FBG7"])
        self.assertEqual(ordinary["seed_source"], "catalog_known_associations")
        self.assertEqual(ordinary["shot_mode"], "few_shot")

        zero_shot = self.planner({**proposal, "_semantic_source": "deepseek", "seed_mode": "none"}).plan(
            user_text="给我普通 Top 10，但这次 zero-shot，不要用数据库阳性酶引导",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(zero_shot["known_enzyme_ids"], [])
        self.assertEqual(zero_shot["seed_source"], "user_explicit_zero_shot")
        self.assertEqual(zero_shot["shot_mode"], "zero_shot")

    def test_explicit_positive_extends_and_deduplicates_database_seeds(self) -> None:
        plan = self.planner({
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "explicit",
            "known_enzyme_ids": ["A0A1W6QDI7", "A0A075FBG7"],
            "homology_policy": "allow",
            "known_association_policy": "separate_known",
            "reason": "用户补充一个阳性酶。",
        }).plan(
            user_text="数据库阳性保留，另外 A0A1W6QDI7 也是我确认的阳性酶，请一起作为参考",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7", "G9MAN7"],
        )
        self.assertEqual(plan["known_enzyme_ids"], ["A0A075FBG7", "G9MAN7", "A0A1W6QDI7"])
        self.assertEqual(plan["seed_source"], "catalog_known_plus_user_explicit")
        self.assertEqual(plan["shot_mode"], "few_shot")

    def test_default_route_is_fixed_and_does_not_guess_zero_shot_from_text(self) -> None:
        plan = self.planner({}).plan(
            user_text="这次用 zero-shot，不要用数据库已知阳性酶引导",
            reaction_equation="A = B",
            route_mode="default",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(plan["known_enzyme_ids"], ["A0A075FBG7"])
        self.assertEqual(plan["seed_source"], "catalog_known_associations")
        self.assertEqual(plan["shot_mode"], "few_shot")

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
            "_semantic_source": "deepseek",
            "top_k": 10,
            "enzyme_taxonomy_scope": "all",
            "seed_mode": "catalog_known",
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
        self.assertEqual(plan["shot_mode"], "few_shot")
        self.assertEqual(plan["homology_policy"], "cross_cluster")
        self.assertTrue(plan["homology_filter_requested"])
        self.assertTrue(plan["homology_filter_applied"])
        self.assertEqual(plan["homology_anchor_source"], "catalog_known_associations")
        self.assertEqual(plan["homology_anchor_ids"], ["A0A1W6QDI7", "A0A075FBG7"])

    def test_cross_cluster_with_explicit_seed_is_seeded_remote_expansion(self) -> None:
        plan = self.planner({
            "_semantic_source": "deepseek",
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
            "_semantic_source": "deepseek",
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

    def test_semantic_router_failure_uses_safe_default_without_keyword_guessing(self) -> None:
        def fail(*_):
            raise RuntimeError("offline")
        planner = RoutePlanner(proposal_fn=fail, protein_ids=self.proteins)
        unrecorded = planner.plan(
            user_text="只找尚未记录的新关联候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(unrecorded["known_association_policy"], "separate_known")
        restored = planner.plan(
            user_text="恢复默认结果范围，保留数据库已知关联，同时展示尚未记录的新关联候选",
            reaction_equation="A = B",
            route_mode="intelligent",
            is_current=True,
            orientation="forward",
            known_association_ids=["A0A075FBG7"],
        )
        self.assertEqual(restored["known_association_policy"], "separate_known")

    def test_ai_route_failure_does_not_recover_intent_with_keywords(self) -> None:
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
        self.assertEqual(plan["known_association_policy"], "separate_known")
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
