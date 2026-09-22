from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.starase_navigator.route_design import RheaRouteDesigner, _biochemical_name_variants, _connectivity_key


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

    def test_parent_compound_name_matches_stereoisomers_without_derivative_substring_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = RheaRouteDesigner(Path(tmp), user_agent="test", cache_root=Path(tmp) / "cache")
            names = {
                "CHEBI:1": "(2E,6E)-farnesyl diphosphate",
                "CHEBI:2": "(2Z,6E)-farnesyl diphosphate",
                "CHEBI:3": "(2Z,6Z)-farnesyl diphosphate",
                "CHEBI:4": "(2E,6E,10E,14E)-geranylfarnesyl diphosphate",
                "CHEBI:5": "(2E,6E)-omega-hydroxy-farnesyl diphosphate",
            }
            d._index = {
                "names": names,
                "name_to_ids": {name.casefold(): [cid] for cid, name in names.items()},
                "chebi_smiles": {
                    "CHEBI:1": "CC",
                    "CHEBI:2": "CCC",
                    "CHEBI:3": "CCCC",
                    "CHEBI:4": "CCCCC",
                    "CHEBI:5": "CCCCCC",
                },
                "adjacency": {},
                "reverse": {},
                "enzyme_counts": {},
                "stats": {},
            }
            self.assertIn("farnesyl diphosphate", _biochemical_name_variants("(2E,6E)-farnesyl diphosphate"))
            self.assertIn("geranylfarnesyl diphosphate", _biochemical_name_variants("(2E,6E,10E,14E)-geranylfarnesyl diphosphate"))
            rows = d.resolve_compound(["farnesyl diphosphate"], limit=10)
        self.assertEqual(
            [row["chebi_id"] for row in rows],
            ["CHEBI:1", "CHEBI:2", "CHEBI:3"],
        )

    def test_current_official_chebi_label_can_align_an_older_local_rhea_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = RheaRouteDesigner(Path(tmp), user_agent="test", cache_root=Path(tmp) / "cache")
            d._index = {
                "names": {
                    "CHEBI:30854": "(indol-3-yl)acetate",
                    "CHEBI:188445": "2-oxindole-3-acetate",
                },
                "name_to_ids": {
                    "(indol-3-yl)acetate": ["CHEBI:30854"],
                    "2-oxindole-3-acetate": ["CHEBI:188445"],
                },
                "chebi_smiles": {
                    "CHEBI:30854": "CC",
                    "CHEBI:188445": "CCC",
                },
                "adjacency": {},
                "reverse": {},
                "enzyme_counts": {},
                "stats": {},
            }

            class Response:
                @staticmethod
                def raise_for_status() -> None:
                    return None

                @staticmethod
                def json() -> dict:
                    return {
                        "response": {
                            "docs": [
                                {"obo_id": "CHEBI:30854", "label": "indole-3-acetate"},
                                {"obo_id": "CHEBI:188445", "label": "2-oxindole-3-acetate"},
                                {"obo_id": "CHEBI:999999", "label": "indole-3-acetate"},
                            ]
                        }
                    }

            calls = []
            d.session.get = lambda *args, **kwargs: calls.append((args, kwargs)) or Response()  # type: ignore[method-assign]
            rows = d.resolve_compound(["indole-3-acetate"], limit=6)
            first_call_count = len(calls)
            cached_rows = d.resolve_compound(["indole-3-acetate"], limit=6)

        self.assertEqual([row["chebi_id"] for row in rows], ["CHEBI:30854"])
        self.assertEqual(cached_rows, rows)
        self.assertTrue(rows[0]["identity_confident"])
        self.assertEqual(rows[0]["match_type"], "official_label_equivalent")
        self.assertEqual(rows[0]["match_source"], "chebi_ols_current_label")
        self.assertGreaterEqual(first_call_count, 1)
        self.assertEqual(len(calls), first_call_count)
        queried = {str(kwargs.get("params", {}).get("q") or "") for _args, kwargs in calls}
        self.assertIn("indole-3-acetate", queried)
        self.assertIn("indole-3-acetic acid", queried)

    def test_substring_derivative_is_candidate_only_not_verified_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = RheaRouteDesigner(Path(tmp), user_agent="test", cache_root=Path(tmp) / "cache")
            d._index = {
                "names": {
                    "CHEBI:188445": "2-oxindole-3-acetate",
                    "CHEBI:777": "4-hydroxy-indole-3-acetate",
                },
                "name_to_ids": {
                    "2-oxindole-3-acetate": ["CHEBI:188445"],
                    "4-hydroxy-indole-3-acetate": ["CHEBI:777"],
                },
                "chebi_smiles": {
                    "CHEBI:188445": "CCC",
                    "CHEBI:777": "CCCC",
                },
                "adjacency": {},
                "reverse": {},
                "enzyme_counts": {},
                "stats": {},
            }

            class EmptyResponse:
                @staticmethod
                def raise_for_status() -> None:
                    return None

                @staticmethod
                def json() -> dict:
                    return {"response": {"docs": []}}

            d.session.get = lambda *args, **kwargs: EmptyResponse()  # type: ignore[method-assign]
            rows = d.resolve_compound(["indole-3-acetate"], limit=6)

        self.assertTrue(rows)
        self.assertTrue(all(row["match_type"] == "lexical_candidate" for row in rows))
        self.assertTrue(all(not row["identity_confident"] for row in rows))
        self.assertEqual({row["chebi_id"] for row in rows}, {"CHEBI:188445", "CHEBI:777"})

    def test_known_uniprot_ids_indexes_master_and_directed_rhea_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = RheaRouteDesigner(Path(tmp), user_agent="test", cache_root=Path(tmp) / "cache")
            path = d._asset_path("sprot")
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                "RHEA_ID\tDIRECTION\tMASTER_ID\tID",
                "32883\tUN\t32883\tC5H429",
                "32885\tRL\t32883\tC5H429",
                "32885\tRL\t32883\tQ9TEST",
            ]
            path.write_text("\n".join(rows) + "\n" + ("# padding\n" * 20), encoding="utf-8")
            self.assertEqual(d.known_uniprot_ids("RHEA:32883"), ["C5H429", "Q9TEST"])
            self.assertEqual(d.known_uniprot_ids("32885"), ["C5H429", "Q9TEST"])
            self.assertEqual(d.known_rhea_ids("C5H429"), ["RHEA:32883"])
            self.assertEqual(d.known_rhea_ids("q9test"), ["RHEA:32883"])

    def test_priority_changes_route_order_without_changing_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = self._designer(tmp)
            short = d.design(source_terms=["start"], target_terms=["target"], max_steps=4, limit=3, priority="short")
            enzyme = d.design(source_terms=["start"], target_terms=["target"], max_steps=4, limit=3, priority="enzyme_available")
        self.assertEqual(short["routes"][0]["compound_ids"], ["A", "D"])
        self.assertEqual(enzyme["routes"][0]["compound_ids"], ["A", "B", "D"])
        self.assertEqual(short["routes"][0]["thermodynamics"]["status"], "not_computed")
        self.assertTrue(all(step["rhea_id"].startswith("RHEA:") for route in short["routes"] for step in route["steps"]))

    def test_excluded_reaction_forces_a_true_alternative_and_materializer_preserves_normal_route_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = self._designer(tmp)
            normal = d.design(
                source_terms=["start"],
                target_terms=["target"],
                max_steps=4,
                limit=3,
                priority="short",
            )
            original = normal["routes"][0]
            rebuilt = d.materialize_route(
                original["steps"],
                max_steps=4,
                priority="short",
            )
            self.assertEqual(rebuilt["route_id"], original["route_id"])
            self.assertEqual(rebuilt["score"], original["score"])
            self.assertEqual(rebuilt["metrics"], original["metrics"])

            alternative = d.design(
                source_terms=["start"],
                target_terms=["target"],
                max_steps=4,
                limit=3,
                priority="short",
                excluded_reaction_ids=["RHEA:10000"],
            )
        self.assertTrue(alternative["routes"])
        self.assertNotEqual(alternative["routes"][0]["compound_ids"], ["A", "D"])
        self.assertTrue(
            all(
                step["rhea_id"] != "RHEA:10000"
                for route in alternative["routes"]
                for step in route["steps"]
            )
        )

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

class RouteModelFrontierTests(unittest.TestCase):
    def _service(self, calls):
        from scripts.starase_navigator.route_pathway_service import RoutePathwayService
        svc = RoutePathwayService.__new__(RoutePathwayService)
        def rank_model(command, payload):
            calls.append((command, dict(payload)))
            if command == "rank-enzymes":
                rid = str(payload["reaction_id"])
                return {
                    "query": {"route_id": f"r2e:{rid}", "score_source": "fake-r2e"},
                    "candidates": [
                        {"rank": 1, "candidate_id": f"P_{rid[-5:]}"},
                        {"rank": 2, "candidate_id": "P2"},
                        {"rank": 3, "candidate_id": "P3"},
                    ],
                }
            rid = "RHEA:" + str(payload["enzyme_id"]).split("P_", 1)[-1]
            return {
                "query": {"route_id": "e2r:fake"},
                "candidates": [{"rank": 7, "candidate_id": rid}],
            }
        svc._rank_model = rank_model
        return svc

    def test_model_enzyme_frontier_deduplicates_steps_and_is_diagnostic_only(self) -> None:
        calls = []
        svc = self._service(calls)
        routes = [
            {"route_id": "A", "score": 91.0, "steps": [
                {"rhea_id": "RHEA:10001", "evidence_type": "known_rhea"},
                {"rhea_id": "RHEA:10002", "evidence_type": "known_rhea"},
            ]},
            {"route_id": "B", "score": 88.0, "steps": [
                {"rhea_id": "RHEA:10001", "evidence_type": "known_rhea"},
            ]},
        ]
        before = [(r["route_id"], r["score"]) for r in routes]
        audit = svc._annotate_model_enzyme_frontier(routes, max_unique_steps=12)
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(audit["requested_unique_steps"], 2)
        self.assertEqual(audit["scored_steps"], 2)
        self.assertEqual(audit["reverse_checked_steps"], 2)
        self.assertEqual(audit["reverse_recovered_at_20_steps"], 2)
        self.assertEqual(len(calls), 4)  # one R2E + one E2R per unique step
        self.assertEqual(before, [(r["route_id"], r["score"]) for r in routes])
        first = routes[0]["steps"][0]["model_enzyme_frontier"]
        repeated = routes[1]["steps"][0]["model_enzyme_frontier"]
        self.assertEqual(first, repeated)
        self.assertTrue(first["reverse_recovery_at_20"])
        self.assertEqual(first["reverse_recovery_rank"], 7)

    def test_model_enzyme_frontier_honors_unique_step_cap(self) -> None:
        calls = []
        svc = self._service(calls)
        route = {"steps": [
            {"rhea_id": f"RHEA:{10000+i}", "evidence_type": "known_rhea"}
            for i in range(5)
        ]}
        audit = svc._annotate_model_enzyme_frontier([route], max_unique_steps=2)
        self.assertEqual(audit["scored_steps"], 2)
        self.assertTrue(audit["truncated"])
        self.assertEqual(len(calls), 4)
        self.assertIn("model_enzyme_frontier", route["steps"][0])
        self.assertNotIn("model_enzyme_frontier", route["steps"][4])
