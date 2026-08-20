from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.catalyst_finder.route_design import RheaRouteDesigner, _connectivity_key
from scripts.catalyst_finder.serve import PATHWAY_INTENT_RE, ROUTE_DESIGN_INTENT_RE, classify_task_intent


class RouteDesignTests(unittest.TestCase):
    def _designer(self, tmp: str) -> RheaRouteDesigner:
        d = RheaRouteDesigner(Path(tmp), user_agent="test", cache_root=Path(tmp) / "cache")
        names = {
            "A": "start",
            "B": "enzyme-rich intermediate",
            "C": "alternative intermediate",
            "D": "target",
            "E": "predicted bridge",
        }
        smiles = {"A": "CCCC", "B": "CCCO", "C": "CCCN", "D": "CCCCO", "E": "CCCOC"}

        def edge(src: str, dst: str, rid: str, enzymes: int, transform: float, direction: float = 1.0):
            return {
                "source": src,
                "target": dst,
                "rhea_id": rid,
                "directed_rhea_id": rid,
                "orientation": "forward",
                "direction_code": "LR",
                "transformation_score": transform,
                "swissprot_count": enzymes,
                "direction_swissprot_count": enzymes,
                "direction_support": direction,
            }

        adjacency = {
            "A": [
                edge("A", "D", "RHEA:10000", 0, 0.50, 0.5),
                edge("A", "B", "RHEA:10001", 20, 0.90),
                edge("A", "C", "RHEA:10002", 5, 0.70),
            ],
            "B": [edge("B", "D", "RHEA:10003", 20, 0.90)],
            "C": [edge("C", "D", "RHEA:10004", 5, 0.70)],
            "E": [edge("E", "D", "RHEA:10005", 4, 0.65)],
        }
        d._index = {
            "names": names,
            "name_to_ids": {name.casefold(): [cid] for cid, name in names.items()},
            "chebi_smiles": smiles,
            "adjacency": adjacency,
            "reverse": {},
            "enzyme_counts": {},
            "stats": {"route_nodes": 5, "route_edges": 6},
        }
        return d

    def test_route_design_intent_is_distinct_from_fixed_pathway_evaluation(self) -> None:
        self.assertIsNotNone(ROUTE_DESIGN_INTENT_RE.search("推荐从 GPP 到 beta-myrcene 的候选合成路线并排序"))
        self.assertIsNotNone(ROUTE_DESIGN_INTENT_RE.search("推荐从 GPP 到 beta-myrcene 的 5 条路线，并按可实现性排序，只用数据库已知反应。"))
        self.assertIsNone(ROUTE_DESIGN_INTENT_RE.search("评估 GPP → linalool → myrcene 这条路径的酶冲突"))
        self.assertIsNotNone(PATHWAY_INTENT_RE.search("评估 GPP → linalool → myrcene 这条路径的酶冲突"))

    def test_substrate_product_and_route_endpoints_have_disjoint_intent_contracts(self) -> None:
        single_reaction_cases = [
            "我想把【底物】转化为【产物】，请帮我找候选酶。",
            "底物是 GPP，产物是 beta-myrcene，请找催化酶。",
            "目标反应是 GPP → beta-myrcene，请推荐 10 个候选酶。",
            "请找能把 GPP 转化为 beta-myrcene 的酶。",
        ]
        route_cases = [
            "起始前体是 GPP，目标产物是 beta-myrcene。",
            "请推荐从 GPP 到 beta-myrcene 的几条生物合成路线。",
            "以 GPP 为起始前体，beta-myrcene 为目标产物，帮我设计路线。",
        ]
        for text in single_reaction_cases:
            with self.subTest(text=text):
                self.assertEqual(classify_task_intent(text), "reaction_to_enzyme")
        for text in route_cases:
            with self.subTest(text=text):
                self.assertEqual(classify_task_intent(text), "route_design")
        self.assertEqual(
            classify_task_intent("评估 GPP → linalool → beta-myrcene 这条路径的酶冲突"),
            "pathway_compatibility",
        )

    def test_hidden_starter_hint_cannot_force_an_obviously_rewritten_task(self) -> None:
        self.assertEqual(
            classify_task_intent("我想把 GPP 转化为 beta-myrcene，请找候选酶。", "route_design"),
            "reaction_to_enzyme",
        )
        self.assertEqual(
            classify_task_intent("请推荐从 GPP 到 beta-myrcene 的路线。", "pathway_compatibility"),
            "route_design",
        )

    def test_priority_changes_route_order_without_changing_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = self._designer(tmp)
            short = d.design(source_terms=["start"], target_terms=["target"], max_steps=4, limit=3, priority="short")
            enzyme = d.design(source_terms=["start"], target_terms=["target"], max_steps=4, limit=3, priority="enzyme_available")
        self.assertEqual(short["routes"][0]["compound_ids"], ["A", "D"])
        self.assertEqual(enzyme["routes"][0]["compound_ids"], ["A", "B", "D"])
        self.assertEqual(short["routes"][0]["thermodynamics"]["status"], "not_computed")
        self.assertTrue(all(step["rhea_id"].startswith("RHEA:") for route in short["routes"] for step in route["steps"]))

    def test_predicted_bridge_is_separate_and_known_direct_prediction_is_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = self._designer(tmp)
            # Keep A->B as a known direct edge; C remains a novel predicted bridge.
            d._run_pickaxe = lambda _smiles: {
                "engine": "MINE/Pickaxe", "generation": 1, "operators": 10,
                "generated_compounds": 2, "generated_reactions": 2,
                "predictions": [
                    {"product_smiles": "CCCO", "rules": ["known-rule"], "reaction_smiles": "A>>B"},
                    {"product_smiles": "CCCOC", "rules": ["novel-rule"], "reaction_smiles": "A>>E"},
                ],
            }
            mapping = {
                _connectivity_key("CCCO"): ["B"],
                _connectivity_key("CCCOC"): ["E"],
                _connectivity_key("CCCCO"): ["D"],
            }
            d._connectivity_to_chebi = lambda: mapping
            result = d.explore_predicted_bridges(
                source_chebi_id="A", target_chebi_id="D", max_steps=3, limit=5,
                priority="balanced", local_reaction_ids=[],
            )
        self.assertGreaterEqual(result["known_duplicate_count"], 1)
        self.assertEqual(len(result["routes"]), 1)
        route = result["routes"][0]
        self.assertEqual(route["compound_ids"], ["A", "E", "D"])
        self.assertEqual(route["steps"][0]["evidence_type"], "predicted_pickaxe")
        self.assertEqual(route["steps"][0]["prediction_rules"], ["novel-rule"])
        self.assertEqual(route["steps"][1]["evidence_type"], "known_rhea")


if __name__ == "__main__":
    unittest.main()
