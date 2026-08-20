from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.catalyst_finder.route_design import RheaRouteDesigner
from scripts.catalyst_finder.route_feasibility import RouteFeasibilityAnalyzer


class RouteFeasibilityTests(unittest.TestCase):
    @staticmethod
    def _routes() -> list[dict]:
        return [
            {"route_id": "A", "score": 95.0, "base_route_score": 95.0, "metrics": {"step_count": 1}, "steps": []},
            {"route_id": "B", "score": 70.0, "base_route_score": 70.0, "metrics": {"step_count": 2}, "steps": []},
            {"route_id": "C", "score": 60.0, "base_route_score": 60.0, "metrics": {"step_count": 3}, "steps": []},
        ]

    def _analyzer(self, tmp: str) -> RouteFeasibilityAnalyzer:
        designer = SimpleNamespace(enrich_route_stoichiometry=lambda route: dict(route))
        return RouteFeasibilityAnalyzer(Path(tmp), designer)  # type: ignore[arg-type]

    def test_completed_zero_host_flux_is_filtered_but_unknown_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = self._analyzer(tmp)
            analyzer.status = lambda: {
                "thermodynamics": {"ready": True},
                "ecoli_fba": {"ready": True},
            }

            def worker(worker: Path, _payload: dict, **_: object) -> dict:
                if worker.name == "route_thermo_worker.py":
                    return {
                        "status": "complete",
                        "routes": [
                            {"route_id": "A", "status": "complete", "mdf_kj_mol": 20.0},
                            {"route_id": "B", "status": "complete", "mdf_kj_mol": 30.0},
                            {"route_id": "C", "status": "insufficient_evidence"},
                        ],
                    }
                return {
                    "status": "complete",
                    "routes": [
                        {"route_id": "A", "status": "complete", "max_route_flux_50pct_growth": 0.0},
                        {"route_id": "B", "status": "complete", "max_route_flux_50pct_growth": 2.0},
                        {"route_id": "C", "status": "target_unmapped"},
                    ],
                }

            analyzer._run_worker = worker  # type: ignore[method-assign]
            result = analyzer.evaluate(self._routes(), host="Escherichia coli", priority="balanced", requested_count=3)

        ids = [row["route_id"] for row in result["routes"]]
        self.assertNotIn("A", ids)
        self.assertIn("B", ids)
        self.assertIn("C", ids)
        self.assertEqual(result["summary"]["host_infeasible_filtered_count"], 1)
        c = next(row for row in result["routes"] if row["route_id"] == "C")
        self.assertEqual(c["host_feasibility_gate"], "unknown")

    def test_thermodynamic_priority_can_rerank_by_real_mdf_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = self._analyzer(tmp)
            analyzer.status = lambda: {
                "thermodynamics": {"ready": True},
                "ecoli_fba": {"ready": False},
            }

            def worker(_worker: Path, _payload: dict, **_: object) -> dict:
                return {
                    "status": "complete",
                    "routes": [
                        {"route_id": "A", "status": "complete", "mdf_kj_mol": 1.0},
                        {"route_id": "B", "status": "complete", "mdf_kj_mol": 50.0},
                        {"route_id": "C", "status": "complete", "mdf_kj_mol": 10.0},
                    ],
                }

            analyzer._run_worker = worker  # type: ignore[method-assign]
            result = analyzer.evaluate(self._routes(), priority="thermodynamic", requested_count=3)
        self.assertEqual(result["routes"][0]["route_id"], "B")
        self.assertGreater(result["routes"][0]["ranking_components"]["weights"]["thermo"], 0.5)

    def test_globally_unavailable_feasibility_layers_fall_back_to_base_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = self._analyzer(tmp)
            analyzer.status = lambda: {
                "thermodynamics": {"ready": False},
                "ecoli_fba": {"ready": False},
            }
            result = analyzer.evaluate(self._routes(), host="Escherichia coli", priority="host_flux", requested_count=3)
        self.assertEqual([row["route_id"] for row in result["routes"]], ["A", "B", "C"])
        self.assertEqual(result["summary"]["host_infeasible_filtered_count"], 0)
        self.assertEqual(result["routes"][0]["ranking_components"]["weights"]["base"], 1.0)

    def test_full_rhea_stoichiometry_uses_exact_official_structure_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            designer = RheaRouteDesigner(root, user_agent="test", cache_root=root / "cache")
            reaction = designer._asset_path("reaction_smiles")
            compounds = designer._asset_path("chebi_smiles")
            reaction.parent.mkdir(parents=True, exist_ok=True)
            reaction.write_text("10001\tCCO.N>>CC=O.[NH4+]\n", encoding="utf-8")
            compounds.write_text(
                "CHEBI:1\tCCO\nCHEBI:2\tN\nCHEBI:3\tCC=O\nCHEBI:4\t[NH4+]\n",
                encoding="utf-8",
            )
            designer._download_asset = lambda key: designer._asset_path(key)  # type: ignore[method-assign]
            result = designer.reaction_stoichiometry("RHEA:10001")
        self.assertEqual(result["status"], "complete")
        by_chebi = {row["chebi_candidates"][0]: row["coefficient"] for row in result["participants"]}
        self.assertEqual(by_chebi["CHEBI:1"], -1.0)
        self.assertEqual(by_chebi["CHEBI:2"], -1.0)
        self.assertEqual(by_chebi["CHEBI:3"], 1.0)
        self.assertEqual(by_chebi["CHEBI:4"], 1.0)

    def test_real_ecoli_pool_is_preferred_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            designer = RheaRouteDesigner(root, user_agent="test", cache_root=root / "cache")
            designer._index = {
                "chebi_smiles": {"CHEBI:10": "CC", "CHEBI:20": "CCC"},
                "names": {}, "name_to_ids": {}, "adjacency": {}, "reverse": {}, "enzyme_counts": {}, "stats": {},
            }
            pool = root / "results/catalyst_finder_runtime/route_feasibility/iML1515_cytosol_chebi.txt"
            pool.parent.mkdir(parents=True, exist_ok=True)
            pool.write_text("CHEBI:10\nCHEBI:999\n", encoding="utf-8")
            self.assertEqual(designer.ecoli_start_pool(), {"CHEBI:10"})


if __name__ == "__main__":
    unittest.main()
