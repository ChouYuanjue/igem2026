from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.database_bridge import DatabaseBridge
from scripts.database_bridge.model_catalog import ModelDataCatalog
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


def test_portal_static_path_guard_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="Path traversal"):
        _safe_path(root, "../secret.txt")
