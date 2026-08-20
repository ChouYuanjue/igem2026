from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.database_bridge import DatabaseBridge
from scripts.database_bridge.model_catalog import ModelDataCatalog
from projects.active.terpene_screening.core.routing import resolve_route
from projects.active.terpene_screening.core.taxonomy_scope import (
    DEFAULT_TAXONOMY_SCOPE_REGISTRY,
    filter_candidate_ids,
    taxonomy_record,
    taxonomy_summary,
    validate_seed_scope,
)
from scripts.terpene_portal.route_catalog import build_route_catalog
from scripts.terpene_portal.serve import ROOT, _safe_path


def decode(response):
    return json.loads(response.body.decode("utf-8"))


def test_database_bridge_serves_only_existing_read_views() -> None:
    bridge = DatabaseBridge(upstream_api=None)
    response = bridge.handle("GET", "/api/v1/graph")
    payload = decode(response)

    assert response.status == 200
    assert payload["success"] is True
    assert payload["meta"]["bridgeMode"] == "compatibility_snapshot"
    assert len(payload["data"]["nodes"]) >= 2
    assert len(payload["data"]["edges"]) >= 1


def test_database_bridge_rejects_write_or_unfinished_operations() -> None:
    bridge = DatabaseBridge(upstream_api=None)
    response = bridge.handle("POST", "/api/v1/download/entries", b"{}")
    payload = decode(response)

    assert response.status == 403
    assert payload["error"]["code"] == "READ_ONLY_BRIDGE"


def test_database_bridge_supports_upstream_edge_expansion() -> None:
    bridge = DatabaseBridge(upstream_api=None)
    response = bridge.handle("GET", "/api/v1/graph/edge-groups/GROUP-GPP-PIN/edges")
    payload = decode(response)

    assert response.status == 200
    assert payload["data"]["edgeGroupId"] == "GROUP-GPP-PIN"
    assert len(payload["data"]["edges"]) == 2


def test_model_data_catalog_matches_deployed_candidate_universe() -> None:
    catalog = ModelDataCatalog(ROOT)
    summary = catalog.summary()

    assert summary["proteins"] == 2085
    assert summary["reactions"] == 753
    assert summary["associations"] == 3439
    assert summary["registered_proteins"] == 694
    assert summary["registered_reactions"] == 240
    assert summary["seen_proteins"] == 1391
    assert summary["seen_reactions"] == 513
    assert summary["read_only"] is True
    assert summary["catalog_contract"] == "deployed-candidate-universe-v1"


def test_model_data_catalog_searches_and_builds_read_only_graph() -> None:
    catalog = ModelDataCatalog(ROOT)
    search = catalog.search("A0A075FBG7", kind="protein", limit=10)
    graph = catalog.graph(limit=12)

    assert any(item["id"] == "A0A075FBG7" for item in search["items"])
    assert graph["read_only"] is True
    assert graph["edge_count"] == 12
    assert graph["node_count"] >= 2
    assert {node["kind"] for node in graph["nodes"]} <= {"protein", "reaction"}
    assert all(edge["protein_id"] and edge["reaction_id"] for edge in graph["edges"])


def test_model_data_catalog_focus_preserves_provenance() -> None:
    catalog = ModelDataCatalog(ROOT)
    graph = catalog.graph(focus_id="A0A075FBG7", limit=40)

    assert any(node["id"] == "A0A075FBG7" for node in graph["nodes"])
    assert all(edge["source_file"].endswith("training_pairs.csv") for edge in graph["edges"])


def test_route_catalog_projects_every_manifest_route_and_protocol_overlay() -> None:
    catalog = build_route_catalog()

    assert catalog["route_count"] == 12
    assert catalog["overlay_count"] == 11
    assert catalog["display_path_count"] == 23
    assert catalog["coverage"]["complete"] is True
    assert catalog["coverage"]["missing_manifest_routes"] == []
    assert catalog["coverage"]["missing_runtime_modifiers"] == []
    assert catalog["coverage"]["represented_modifier_suffixes"] == [
        "eukaryote-only", "fewshot", "manual", "masked", "prokaryote-only", "temporary-universe",
    ]
    assert catalog["read_only"] is True
    assert [route["route_id"] for route in catalog["routes"][:3]] == [
        "r2e-current-top3-v1",
        "r2e-current-top10-v1",
        "r2e-current-top20-v1",
    ]
    assert {route["route_id"] for route in catalog["routes"]} == {
        "r2e-current-top3-v1",
        "r2e-current-top10-v1",
        "r2e-current-top20-v1",
        "r2e-external-top3-v1",
        "r2e-external-top10-v1",
        "r2e-external-top20-v1",
        "e2r-current-top3-v1",
        "e2r-current-top10-v1",
        "e2r-current-top20-v1",
        "e2r-external-top3-neighbor-v1",
        "e2r-external-top10-neural-rrf-v1",
        "e2r-external-top20-dual-kernel-rrf-v1",
    }
    all_paths = [*catalog["routes"], *catalog["overlays"]]
    assert len(all_paths) == catalog["display_path_count"] == 23
    assert all(item.get("use_case") for item in all_paths)
    assert all(item.get("description") for item in all_paths)

    overlays = {item["key"]: item for item in catalog["overlays"]}
    assert overlays["r2e-fewshot-seed"]["retrieval"] == "seed"
    assert overlays["e2r-fewshot-seed"]["retrieval"] == "seed"
    assert overlays["e2r-zero-shot-mask-overlay"]["shot_mode"] == "zero_shot"
    assert overlays["e2r-zero-shot-mask-overlay"]["retrieval"] == "route_preserved_then_masked"
    assert overlays["r2e-known-association-mask-overlay"]["availability"] == "batch_only"
    assert overlays["r2e-known-association-mask-overlay"]["modifier_suffix"] == "masked"
    assert overlays["r2e-temporary-universe-overlay"]["modifier_suffix"] == "temporary-universe"
    assert overlays["e2r-temporary-universe-overlay"]["availability"] == "cli_only"
    assert overlays["r2e-manual-override-overlay"]["modifier_suffix"] == "manual"
    assert overlays["e2r-manual-override-overlay"]["availability"] == "cli_only"
    assert overlays["r2e-eukaryote-only-overlay"]["modifier_suffix"] == "eukaryote-only"
    assert overlays["r2e-prokaryote-only-overlay"]["modifier_suffix"] == "prokaryote-only"
    assert overlays["r2e-cage-rescue-overlay"]["category"] == "conditional_path"
    assert catalog["taxonomy_scope"]["scope_counts"] == {
        "eukaryote": 1340, "other": 6, "prokaryote": 180, "unknown": 559,
    }


def test_route_catalog_records_distinct_top10_and_top20_e2r_fusion_modules() -> None:
    catalog = build_route_catalog()
    routes = {route["route_id"]: route for route in catalog["routes"]}

    top10 = routes["e2r-external-top10-neural-rrf-v1"]
    top20 = routes["e2r-external-top20-dual-kernel-rrf-v1"]
    assert "e2r-hardneg" in top10["modules"]
    assert "e2r-rrf10" in top10["modules"]
    assert top10["secondary_deployment"].endswith("marts_adapted_drfp_pu_e2r_hardneg128")
    assert "e2r-dualkernel" in top20["modules"]
    assert "e2r-rrf20" in top20["modules"]
    assert top20["auxiliary_deployment"].endswith("marts_dual_kernel_e2r_top20")


def test_route_catalog_covers_the_runtime_route_suffix_grammar() -> None:
    catalog = build_route_catalog()
    represented = set(catalog["coverage"]["represented_modifier_suffixes"])
    route = resolve_route(
        direction="enzyme_to_reaction",
        objective="top10",
        is_current=False,
        has_seed=True,
        masked_discovery=True,
        temporary_candidate_extension=True,
        manual_override=True,
    )

    assert route.route_id == (
        "e2r-external-top10-neural-rrf-v1"
        "+fewshot+masked+temporary-universe+manual"
    )
    assert represented == {
        "fewshot", "masked", "temporary-universe", "manual",
        "eukaryote-only", "prokaryote-only",
    }


def test_r2e_taxonomy_scope_registry_is_complete_and_conservative() -> None:
    summary = taxonomy_summary()
    assert summary == {
        "version": "terpene-enzyme-taxonomy-scope-v1",
        "total": 2085,
        "eukaryote": 1340,
        "prokaryote": 180,
        "other": 6,
        "unknown": 559,
    }
    euk = taxonomy_record("A0A075FBG7")
    prok = taxonomy_record("A0A0H5BB10")
    assert (euk.taxonomy_scope, euk.kingdom) == ("eukaryote", "Plantae")
    assert (prok.taxonomy_scope, prok.kingdom) == ("prokaryote", "Bacteria")

    ids = ["A0A075FBG7", "A0A0H5BB10", "A0A060KY90"]
    keep_euk, audit_euk = filter_candidate_ids(ids, "eukaryote")
    keep_prok, audit_prok = filter_candidate_ids(ids, "prokaryote")
    assert keep_euk == [0]
    assert keep_prok == [1]
    assert audit_euk["unknown_count"] == audit_prok["unknown_count"] == 1
    assert audit_euk["post_filter_size"] == audit_prok["post_filter_size"] == 1

    with pytest.raises(ValueError, match="incompatible"):
        validate_seed_scope(["A0A0H5BB10"], "eukaryote")
    with pytest.raises(ValueError, match="unclassified"):
        validate_seed_scope(["A0A060KY90"], "prokaryote")
    assert DEFAULT_TAXONOMY_SCOPE_REGISTRY.is_file()


def test_r2e_taxonomy_scope_route_suffix_is_explicit_and_e2r_rejects_it() -> None:
    route = resolve_route(
        direction="reaction_to_enzyme",
        objective="top10",
        is_current=False,
        has_seed=True,
        enzyme_taxonomy_scope="eukaryote",
    )
    assert route.route_id == "r2e-external-top10-v1+fewshot+eukaryote-only"
    with pytest.raises(ValueError, match="only defined for reaction_to_enzyme"):
        resolve_route(
            direction="enzyme_to_reaction",
            objective="top10",
            is_current=False,
            enzyme_taxonomy_scope="prokaryote",
        )


def test_route_atlas_source_mentions_every_catalog_path_key() -> None:
    catalog = build_route_catalog()
    source = (ROOT / "frontend/terpene_portal/src/components/RouteAtlas.tsx").read_text(encoding="utf-8")

    for route in catalog["routes"]:
        assert route["route_id"] in source
    for overlay in catalog["overlays"]:
        assert overlay["key"] in source or overlay["modules"][0] in source


def test_portal_static_path_guard_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="Path traversal"):
        _safe_path(root, "../secret.txt")
