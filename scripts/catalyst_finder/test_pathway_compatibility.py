from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.catalyst_finder.pathway_compatibility import (
    PathwayCompatibilityAnalyzer,
    pairwise_compatibility,
    target_condition_compatibility,
)
from scripts.catalyst_finder.serve import PATHWAY_ARROW_RE, PATHWAY_INTENT_RE


class PathwayCompatibilityTests(unittest.TestCase):
    def test_pathway_intent_is_natural_language_driven(self) -> None:
        self.assertIsNotNone(PATHWAY_INTENT_RE.search("这条完整反应路径里的酶会不会条件冲突？"))
        self.assertIsNotNone(PATHWAY_INTENT_RE.search("帮我检查多步反应能不能 one-pot"))
        self.assertIsNone(PATHWAY_INTENT_RE.search("帮我找这个反应的 10 个候选酶"))
        self.assertIsNotNone(PATHWAY_ARROW_RE.search("GGPP -> CPP -> miltiradiene"))
        self.assertIsNone(PATHWAY_ARROW_RE.search("CPP -> miltiradiene"))

    def test_temperature_conflict_is_explicit(self) -> None:
        a = {"condition_profile": {"temperature_optimum_c": [80.0, 80.0]}}
        b = {"condition_profile": {"temperature_optimum_c": [37.0, 37.0]}}
        score, issues = pairwise_compatibility(a, b, mode="one_pot")
        self.assertLess(score, 0)
        self.assertTrue(any(row["type"] == "temperature" and row["severity"] == "high" for row in issues))

    def test_explicit_target_condition_can_flag_candidate(self) -> None:
        candidate = {"condition_profile": {"ph_optimum": [6.0, 6.5], "temperature_optimum_c": [70.0, 70.0]}}
        score, issues = target_condition_compatibility(candidate, {"ph": 8.0, "temperature_c": 30.0, "cofactors": []})
        self.assertLess(score, 0)
        self.assertTrue(any(row["type"] == "target_ph" for row in issues))
        self.assertTrue(any(row["type"] == "target_temperature" for row in issues))

    def test_natural_language_mg2_matches_uniprot_mg_parenthesized_charge(self) -> None:
        candidate = {"condition_profile": {"cofactors": ["Mg(2+)"]}}
        score, issues = target_condition_compatibility(candidate, {"ph": None, "temperature_c": None, "cofactors": ["Mg2+"]})
        self.assertGreater(score, 0.0)
        self.assertEqual(issues, [])

    def test_sequential_mode_allows_different_step_conditions(self) -> None:
        a = {"condition_profile": {"temperature_optimum_c": [80.0, 80.0], "ph_optimum": [9.0, 9.0]}}
        b = {"condition_profile": {"temperature_optimum_c": [30.0, 30.0], "ph_optimum": [6.0, 6.0]}}
        score, issues = pairwise_compatibility(a, b, mode="sequential")
        self.assertEqual(score, 0.0)
        self.assertEqual(issues, [])

    def test_global_rerank_can_trade_small_local_rank_loss_for_compatibility(self) -> None:
        def rank_reaction(rhea_id: str, **_: object) -> dict:
            if rhea_id == "RHEA:10000":
                rows = [
                    {"rank": 1, "candidate_id": "A", "uniprot_id": "AAAAAA", "score": 1.0, "score_fraction": 1.0},
                    {"rank": 2, "candidate_id": "B", "uniprot_id": "BBBBBB", "score": 0.9, "score_fraction": 0.9},
                ]
            else:
                rows = [{"rank": 1, "candidate_id": "C", "uniprot_id": "CCCCCC", "score": 1.0, "score_fraction": 1.0}]
            return {"candidates": rows, "ranking": {"route_id": "fake-r2e"}}

        profiles = {
            "AAAAAA": {"available": True, "temperature_optimum_c": [80.0, 80.0], "ph_optimum": [7.0, 7.0]},
            "BBBBBB": {"available": True, "temperature_optimum_c": [35.0, 35.0], "ph_optimum": [7.0, 7.0]},
            "CCCCCC": {"available": True, "temperature_optimum_c": [37.0, 37.0], "ph_optimum": [7.0, 7.0]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = PathwayCompatibilityAnalyzer(
                root=Path(tmp),
                catalog=SimpleNamespace(protein_by_id={}),
                rank_reaction=rank_reaction,
                user_agent="test",
                cache_root=Path(tmp),
            )
            analyzer.conditions.profile = lambda accession: {"accession": accession, "source": "fixture", **profiles[accession]}
            result = analyzer.analyze(
                steps=[{"rhea_id": "RHEA:10000"}, {"rhea_id": "RHEA:20000"}],
                execution_mode="one_pot",
            )
        self.assertEqual(result["steps"][0]["local_best_id"], "A")
        self.assertEqual(result["steps"][0]["selected_enzyme"]["candidate_id"], "B")
        self.assertTrue(result["steps"][0]["changed_for_pathway_compatibility"])
        self.assertIsNone(result["shared_conditions"]["temperature_c"])

    def test_model_only_pathway_selection_does_not_fetch_uniprot_conditions(self) -> None:
        def rank_reaction(rhea_id: str, **_: object) -> dict:
            accession = "AAAAAA" if rhea_id == "RHEA:10000" else "BBBBBB"
            return {"candidates": [{"rank": 1, "candidate_id": accession, "uniprot_id": accession, "score": 1.0, "score_fraction": 1.0}], "ranking": {"route_id": "fake"}}
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = PathwayCompatibilityAnalyzer(
                root=Path(tmp), catalog=SimpleNamespace(protein_by_id={}, pairs_by_protein={}), rank_reaction=rank_reaction,
                user_agent="test", cache_root=Path(tmp),
            )
            analyzer.conditions.profile = lambda _accession: (_ for _ in ()).throw(AssertionError("UniProt condition lookup should not run"))
            result = analyzer.analyze(
                steps=[{"rhea_id": "RHEA:10000"}, {"rhea_id": "RHEA:20000"}],
                evidence_dimensions=[],
            )
        self.assertEqual(result["evidence_dimensions"], [])
        self.assertEqual(result["verdict"], "model_joint_selection")
        self.assertEqual([row["name"] for row in result["evidence_sources"]], ["Catalyst Finder R2E"])
        self.assertEqual(result["route_view"]["decision"]["evidence_dimensions"], [])
        self.assertNotIn("pathway-uniprot-conditions", [row["id"] for row in result["route_view"]["nodes"]])

    def test_cofactor_only_pathway_ignores_ph_temperature_and_localization(self) -> None:
        def rank_reaction(rhea_id: str, **_: object) -> dict:
            accession = "AAAAAA" if rhea_id == "RHEA:10000" else "BBBBBB"
            return {"candidates": [{"rank": 1, "candidate_id": accession, "uniprot_id": accession, "score": 1.0, "score_fraction": 1.0}], "ranking": {"route_id": "fake"}}
        profiles = {
            "AAAAAA": {"available": True, "ph_optimum": [5.0, 5.0], "temperature_optimum_c": [80.0, 80.0], "cofactors": ["Mg(2+)"]},
            "BBBBBB": {"available": True, "ph_optimum": [10.0, 10.0], "temperature_optimum_c": [20.0, 20.0], "cofactors": ["Mg(2+)"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = PathwayCompatibilityAnalyzer(
                root=Path(tmp), catalog=SimpleNamespace(protein_by_id={}, pairs_by_protein={}), rank_reaction=rank_reaction,
                user_agent="test", cache_root=Path(tmp),
            )
            analyzer.conditions.profile = lambda accession: {"accession": accession, "source": "fixture", **profiles[accession]}
            result = analyzer.analyze(
                steps=[{"rhea_id": "RHEA:10000"}, {"rhea_id": "RHEA:20000"}],
                execution_mode="one_pot", evidence_dimensions=["cofactors"],
            )
        self.assertEqual(result["evidence_dimensions"], ["cofactors"])
        self.assertIsNone(result["shared_conditions"]["ph"])
        self.assertIsNone(result["shared_conditions"]["temperature_c"])
        self.assertEqual(result["shared_conditions"]["cofactors"], ["Mg(2+)"])
        self.assertFalse(any(row.get("type") in {"ph", "temperature", "target_ph", "target_temperature", "localization"} for row in result["conflicts"]))

    def test_shared_window_requires_evidence_for_every_selected_enzyme(self) -> None:
        def rank_reaction(rhea_id: str, **_: object) -> dict:
            accession = "AAAAAA" if rhea_id == "RHEA:10000" else "BBBBBB"
            return {"candidates": [{"rank": 1, "candidate_id": accession, "uniprot_id": accession, "score": 1.0, "score_fraction": 1.0}], "ranking": {"route_id": "fake"}}

        profiles = {
            "AAAAAA": {"available": True, "ph_optimum": [7.5, 7.5], "ph_active": [7.0, 8.0], "cofactors": ["Mg(2+)"]},
            "BBBBBB": {"available": True, "cofactors": ["Mg(2+)"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = PathwayCompatibilityAnalyzer(
                root=Path(tmp), catalog=SimpleNamespace(protein_by_id={}), rank_reaction=rank_reaction,
                user_agent="test", cache_root=Path(tmp),
            )
            analyzer.conditions.profile = lambda accession: {"accession": accession, "source": "fixture", **profiles[accession]}
            result = analyzer.analyze(steps=[{"rhea_id": "RHEA:10000"}, {"rhea_id": "RHEA:20000"}], execution_mode="one_pot")
        self.assertIsNone(result["shared_conditions"]["ph"])
        self.assertEqual(result["shared_conditions"]["ph_coverage"], 1)
        self.assertEqual(result["shared_conditions"]["cofactors"], ["Mg(2+)"])
        self.assertEqual(result["verdict"], "partial_evidence")


if __name__ == "__main__":
    unittest.main()
