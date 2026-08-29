from __future__ import annotations

import unittest

from scripts.catalyst_finder.homology import CURRENT_SEQUENCES, PRODUCTION_REGISTRY, ProteinHomologyIndex
from scripts.catalyst_finder.route_view import build_e2r_route_view, build_r2e_route_view, system_route_catalog


class RouteViewTests(unittest.TestCase):
    def test_canonical_route_catalog_is_fully_projected(self) -> None:
        catalog = system_route_catalog()
        self.assertEqual(catalog["counts"]["manifest_routes"], 12)
        self.assertTrue(catalog["coverage"]["complete"])
        keys = {item["key"] for item in catalog["overlays"]}
        self.assertIn("r2e-cross-cluster-filter-v1", keys)
        self.assertIn("r2e-discovery-known-mask-v1", keys)
        self.assertIn("r2e-mixed-zero-shot", keys)
        self.assertIn("e2r-mixed-zero-shot", keys)
        self.assertIn("r2e-tps-specialized", keys)
        self.assertIn("e2r-tps-specialized", keys)
        self.assertNotIn("r2e-manual-override-overlay", keys)
        self.assertNotIn("e2r-manual-override-overlay", keys)
        self.assertNotIn("r2e-temporary-universe-overlay", keys)
        self.assertNotIn("e2r-temporary-universe-overlay", keys)
        self.assertNotIn("r2e-known-association-mask-overlay", keys)
        self.assertNotIn("r2e-cage-rescue-overlay", keys)
        self.assertGreater(catalog["counts"]["hidden_internal_overlays"], 0)
        self.assertEqual(catalog["counts"]["public_overlays"], len(catalog["overlays"]))
        for item in [*catalog["base_routes"], *catalog["overlays"], *catalog["downstream_workflows"]]:
            self.assertTrue(item.get("flow"), item.get("key"))
            if item in catalog["base_routes"] or item in catalog["overlays"]:
                self.assertTrue(item.get("label"), item.get("key"))
                self.assertTrue(item.get("label_en"), item.get("key"))
            for step in item["flow"]:
                self.assertTrue(step.get("title"), step)
                self.assertTrue(step.get("detail"), step)

    def test_route_design_workflow_and_isolated_prediction_overlay_are_catalogued(self) -> None:
        catalog = system_route_catalog()
        workflows = {item["key"] for item in catalog["downstream_workflows"]}
        overlays = {item["key"] for item in catalog["overlays"]}
        self.assertIn("route-design-rhea-known-v1", workflows)
        self.assertIn("pathway-compatibility-v1", workflows)
        self.assertNotIn("controlled-uniprot-rescue", workflows)
        self.assertNotIn("registry-wide-discovery", workflows)
        self.assertNotIn("wetlab-panel-selection", workflows)
        self.assertIn("route-design-pickaxe-isolated", overlays)

    def test_actual_route_view_shows_route_not_just_id(self) -> None:
        view = build_r2e_route_view(
            reaction={"rhea_id": "RHEA:33983", "equation": "A = B"},
            query={
                "route_id": "r2e-current-top10-v1+fewshot",
                "scope": "current",
                "shot_mode": "few_shot",
                "ranking_objective": "top10",
                "score_source": "seed",
                "candidate_universe_size": 2085,
                "candidate_universe_pre_taxonomy_size": 2085,
                "candidate_universe_post_taxonomy_size": 2085,
                "enzyme_taxonomy_scope": "all",
            },
            routing={
                "top_k": 10,
                "known_enzyme_ids": ["C8XPS0"],
                "seed_source": "user_explicit",
                "homology_policy": "cross_cluster",
                "homology_filter": {
                    "applied": True,
                    "excluded_count": 11,
                    "anchor_count": 1,
                    "definition": "MMseqs2 50% sequence-identity cluster, coverage >= 80%",
                },
                "discovery_filter": {
                    "applied": True,
                    "recorded_association_count": 6,
                    "excluded_count": 6,
                },
            },
            candidates=[{"candidate_id": "X"} for _ in range(10)],
        )
        node_ids = [item["id"] for item in view["nodes"]]
        self.assertIn("r2e-seed", node_ids)
        self.assertIn("r2e-seed-mask", node_ids)
        self.assertIn("r2e-cross-cluster", node_ids)
        self.assertIn("r2e-known-mask", node_ids)
        self.assertEqual(node_ids[-1], "r2e-output")
        self.assertIn("r2e-cross-cluster-filter-v1", view["active_overlays"])
        self.assertIn("r2e-discovery-known-mask-v1", view["active_overlays"])

    def test_known_only_filters_are_projected_in_catalog(self) -> None:
        catalog = system_route_catalog()
        keys = {item["key"] for item in catalog["overlays"]}
        self.assertIn("r2e-known-only-filter-v1", keys)
        self.assertIn("e2r-known-only-filter-v1", keys)

    def test_r2e_known_only_actual_route_is_explicit(self) -> None:
        view = build_r2e_route_view(
            reaction={"rhea_id": "RHEA:33983", "equation": "A = B"},
            query={
                "route_id": "r2e-current-top10-v1",
                "scope": "current",
                "shot_mode": "zero_shot",
                "ranking_objective": "top10",
                "candidate_universe_size": 2085,
                "enzyme_taxonomy_scope": "all",
            },
            routing={
                "top_k": 10,
                "discovery_filter": {
                    "result_mode": "known_associations_only",
                    "recorded_association_count": 6,
                    "retained_count": 6,
                },
            },
            candidates=[{"candidate_id": "X"} for _ in range(6)],
        )
        ids = [item["id"] for item in view["nodes"]]
        self.assertIn("r2e-known-only", ids)
        self.assertNotIn("r2e-known-mask", ids)
        self.assertIn("r2e-known-only-filter-v1", view["active_overlays"])
        self.assertEqual(view["decision"]["known_association_policy"], "known_only")

    def test_e2r_known_only_actual_route_is_explicit(self) -> None:
        view = build_e2r_route_view(
            protein={"id": "C8XPS0", "name": "KSL1", "organism": "Salvia"},
            query={
                "route_id": "e2r-current-top10-v1",
                "scope": "current",
                "shot_mode": "zero_shot",
                "ranking_objective": "top10",
                "candidate_universe_size": 753,
            },
            routing={
                "top_k": 10,
                "use_known_activity_seeds": False,
                "discovery_filter": {
                    "result_mode": "known_associations_only",
                    "recorded_association_count": 2,
                    "retained_count": 2,
                },
            },
            candidates=[{"candidate_id": "RHEA:33983"}, {"candidate_id": "RHEA:54512"}],
        )
        ids = [item["id"] for item in view["nodes"]]
        self.assertIn("e2r-known-only", ids)
        self.assertNotIn("e2r-mask-only", ids)
        self.assertIn("e2r-known-only-filter-v1", view["active_overlays"])
        self.assertEqual(view["decision"]["known_association_policy"], "known_only")


    def test_route_view_uses_runtime_candidate_universe_metadata_not_legacy_counts(self) -> None:
        r2e = build_r2e_route_view(
            reaction={"rhea_id": "RHEA:1", "equation": "A = B"},
            query={
                "route_id": "r2e-external-top10-v1", "scope": "external", "shot_mode": "zero_shot",
                "ranking_objective": "top10", "candidate_universe_size": 185918,
                "candidate_universe_pre_taxonomy_size": 185918, "candidate_universe_post_taxonomy_size": 185918,
                "candidate_universe_description": "General merged enzyme universe", "enzyme_taxonomy_scope": "all",
            },
            routing={"top_k": 10, "candidate_universe": "general_merged"}, candidates=[{"candidate_id": "P1"}],
        )
        r2e_node = next(row for row in r2e["nodes"] if row["id"] == "r2e-universe")
        self.assertEqual(r2e_node["metric"], "185918 proteins")
        self.assertEqual(r2e_node["detail"], "General merged enzyme universe")
        self.assertNotIn("1,391", r2e_node["detail"])

        e2r = build_e2r_route_view(
            protein={"id": "P1", "name": "E", "organism": "O"},
            query={
                "route_id": "e2r-external-top10-neural-rrf-v1", "scope": "external", "shot_mode": "zero_shot",
                "ranking_objective": "top10", "candidate_universe_size": 11081,
                "candidate_universe_description": "General merged reaction universe",
            },
            routing={"top_k": 10, "candidate_universe": "general_merged"}, candidates=[{"candidate_id": "RHEA:1"}],
        )
        e2r_node = next(row for row in e2r["nodes"] if row["id"] == "e2r-universe")
        self.assertEqual(e2r_node["metric"], "11081 reactions")
        self.assertEqual(e2r_node["detail"], "General merged reaction universe")
        self.assertNotIn("753", e2r_node["detail"])

    def test_deployed_homology_index_matches_known_same_family_examples(self) -> None:
        if not CURRENT_SEQUENCES.is_file() or not PRODUCTION_REGISTRY.is_file():
            self.skipTest("deployed homology source assets are not provisioned in this checkout")
        index = ProteinHomologyIndex()
        excluded, meta = index.exclusion_set(["C8XPS0"])
        self.assertIn("C8XPS0", excluded)
        self.assertIn("W8QMF8", excluded)
        self.assertIn("A0A1W6QDI7", excluded)
        self.assertEqual(meta["min_sequence_identity"], 0.5)
        self.assertEqual(meta["min_coverage"], 0.8)


if __name__ == "__main__":
    unittest.main()
