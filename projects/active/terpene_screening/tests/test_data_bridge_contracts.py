from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.database_bridge import DatabaseBridge
from scripts.database_bridge.model_catalog import ModelDataCatalog


def _write_rows(path: Path, rows: list[dict[str, str]], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _catalog_root(tmp_path: Path) -> Path:
    prod = tmp_path / "results/terpene_production_models/marts_adapted_drfp_pu"
    _write_rows(prod / "protein_registry.csv", [
        {"protein_id": "P_CURRENT", "source": "current"},
        {"protein_id": "P_REGISTERED", "source": "registered"},
    ])
    _write_rows(prod / "reaction_registry.csv", [
        {"reaction_id": "RHEA:10001", "source": "current", "reaction_signature": "sig1", "reaction_smiles": "CC>>CO"},
        {"reaction_id": "RHEA:10002", "source": "registered", "reaction_signature": "sig2", "reaction_smiles": "CO>>C=O"},
    ])
    _write_rows(prod / "training_pairs.csv", [
        {"Entry": "P_CURRENT", "rhea_id": "RHEA:10001", "source": "fixture"},
        {"Entry": "P_REGISTERED", "rhea_id": "RHEA:10002", "source": "fixture"},
        {"Entry": "P_CURRENT", "rhea_id": "RHEA:10002", "source": "fixture"},
    ])
    _write_rows(tmp_path / "data/terpene_marts/marts_enzymes.tsv", [
        {"enzyme_id": "P_CURRENT", "uniprot_id": "P_CURRENT", "genbank_id": "", "sequence": "MPEPTIDE", "enzyme_name": "Current enzyme", "species": "Plant A", "kingdom": "Plantae", "terpene_type": "mono", "tps_class": "I"},
        {"enzyme_id": "P_REGISTERED", "uniprot_id": "P_REGISTERED", "genbank_id": "", "sequence": "MSEQUENCE", "enzyme_name": "Registered enzyme", "species": "Plant B", "kingdom": "Plantae", "terpene_type": "sesqui", "tps_class": "I"},
    ], delimiter="\t")
    _write_rows(tmp_path / "data/terpene_marts/marts_reactions.tsv", [
        {"reaction_signature": "sig1", "canonical_reaction": "CC>>CO", "substrate_name": "A", "product_name": "B", "terpene_type": "mono", "mechanism_marts_id": "M1"},
        {"reaction_signature": "sig2", "canonical_reaction": "CO>>C=O", "substrate_name": "C", "product_name": "D", "terpene_type": "sesqui", "mechanism_marts_id": "no_mechanism"},
    ], delimiter="\t")
    _write_rows(tmp_path / "data/terpene/all_seq_terpene_synthase.tsv", [
        {"Entry": "P_CURRENT", "Sequence": "MPEPTIDE"},
        {"Entry": "P_REGISTERED", "Sequence": "MSEQUENCE"},
    ], delimiter="\t")
    return tmp_path


def _decode(response):
    return json.loads(response.body.decode("utf-8"))


def test_database_bridge_read_only_snapshot_contract() -> None:
    bridge = DatabaseBridge(upstream_api=None)
    graph = bridge.handle("GET", "/api/v1/graph")
    payload = _decode(graph)
    assert graph.status == 200
    assert payload["success"] is True
    assert payload["meta"]["bridgeMode"] == "compatibility_snapshot"
    assert len(payload["data"]["nodes"]) >= 2
    assert len(payload["data"]["edges"]) >= 1

    denied = bridge.handle("POST", "/api/v1/download/entries", b"{}")
    denied_payload = _decode(denied)
    assert denied.status == 403
    assert denied_payload["error"]["code"] == "READ_ONLY_BRIDGE"

    expanded = _decode(bridge.handle("GET", "/api/v1/graph/edge-groups/GROUP-GPP-PIN/edges"))
    assert len(expanded["data"]["edges"]) == 2


def test_model_data_catalog_uses_only_declared_runtime_files(tmp_path: Path) -> None:
    catalog = ModelDataCatalog(_catalog_root(tmp_path))
    summary = catalog.summary()
    assert summary["proteins"] == 2
    assert summary["reactions"] == 2
    assert summary["associations"] == 3
    assert summary["registered_proteins"] == 1
    assert summary["registered_reactions"] == 1
    assert summary["seen_proteins"] == 1
    assert summary["seen_reactions"] == 1
    assert summary["read_only"] is True
    assert summary["catalog_contract"] == "deployed-candidate-universe-v1"

    search = catalog.search("P_CURRENT", kind="protein", limit=10)
    assert [item["id"] for item in search["items"]] == ["P_CURRENT"]
    graph = catalog.graph(focus_id="P_CURRENT", limit=40)
    assert graph["read_only"] is True
    assert graph["edge_count"] == 2
    assert any(node["id"] == "P_CURRENT" for node in graph["nodes"])
    assert all(edge["source_file"].endswith("training_pairs.csv") for edge in graph["edges"])


def test_model_data_catalog_fails_closed_when_runtime_assets_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing model catalog files"):
        ModelDataCatalog(tmp_path)
