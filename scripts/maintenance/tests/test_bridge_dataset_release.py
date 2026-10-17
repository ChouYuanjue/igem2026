"""Dataset publication metadata and scientific membership generator checks (source-only CI)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.maintenance import rebuild_bridge_canonical_splits as splits

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "benchmarks/bridge/canonical-v1/manifest.json"


def test_release_lock_has_exact_canonical_identity():
    lock = json.loads(LOCK.read_text())
    assert lock["schema"] == "atlas-bridge-canonical-dataset-v1"
    assert lock["publication"]["gitlab_split_files_committed"] is False
    snapshot = lock["publication"]["published_snapshot"]
    assert snapshot["repo_id"] == "ChouYuanjue/atlas-bridge"
    assert snapshot["revision"] == "6b0850a8aac6357869037912a51fbdd66dc0fca4"
    assert snapshot["files"]["validation"]["sha256"] == "60fae36dbc7ae514b98aa28a1bb3cd94faeeb4f0cad1fd2f1221261c237b3fd4"
    assert lock["publication"]["dedicated_dataset_created"] is False
    assert lock["splits"]["train.csv.gz"]["rows"] == 218537
    assert lock["splits"]["validation.csv.gz"]["rows"] == 5216
    assert lock["splits"]["test.csv.gz"]["rows"] == 21505
    assert lock["checks"]["validation_test_shared_reaction_queries"] == 516
    assert lock["checks"]["development_partition_counts"] == {"learn": 3900, "validation": 1316}
    for item in lock["splits"].values():
        assert len(item["csv_sha256"]) == 64


def test_hash_based_query_partition_is_stable():
    assert splits.parent_split("O17433") == "evaluation"
    assert splits.external_split("O17433", {}) == "gate_learn"
    assert splits.external_split("Q9AJE3", {}) == "gate_validation"
    assert splits.external_split("O17433", {"O17433": "evaluation"}) == "evaluation"
    assert splits.bucket(0) == "0"
    assert splits.bucket(16) == "16+"
    assert splits.exposure("P", "R", {"P": 0}, {"R": 0}) == ("double_cold", "double_cold|degree=0")


def test_split_rejects_positives_present_in_training():
    train = [{"protein_id": "P", "reaction_id": "R"}]
    with pytest.raises(ValueError, match="overlap"):
        splits.split(train, train, [])


def test_published_records_have_factual_membership_without_editorial_status():
    official = ROOT / "reproducibility/bridge/records"
    native = ROOT / "reproducibility/bime_rank/records"
    records = official if official.exists() else native
    canonical = json.loads((records / "BRIDGE_CANONICAL_21505_RESULT.json").read_text())
    common = json.loads((records / "BRIDGE_COMMON_5216_VALIDATION_RESULT.json").read_text())
    episodic = json.loads((records / "BRIDGE_EPISODIC_SINGLE_5216_VALIDATION_RESULT.json").read_text())
    runtime = json.loads((ROOT / "projects/active/bridge/release/runtime/final_bridge_v1/manifest.json").read_text())
    assert canonical["main_test"]["strict_train_relation_unseen_edges"] == 21505
    assert common["unified_validation"]["positive_pairs"] == 5216
    assert episodic["dev_final_test_exact_positive_overlap"] == 0
    assert "scientific_status" not in canonical
    assert "scientific_status" not in common["formal_final_test"]
    assert "validity" not in episodic
    for key in ("test_as_a_metric_selection_set", "test_result_inspected_before_model_iteration"):
        assert key not in canonical["main_test"]
    for mode in ("adaptive_relation_gate", "episodic_memory_gate"):
        assert "outer_used_for_selection" not in runtime["production_validation"][mode]


def test_published_hf_file_hashes_match_immutable_asset_registry_when_present():
    asset_lock = ROOT / "assets/full_reproduction.lock.json"
    if not asset_lock.exists():
        return  # The native research repo uses a separate asset registry.
    dataset = json.loads(LOCK.read_text())
    frozen = json.loads(asset_lock.read_text())
    source_sha = {entry["path"]: entry["sha256"] for entry in frozen["files"]}
    source_paths = {
        "train": "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv",
        "validation": "results/bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz",
        "test": "results/bridge_gate_split_v2/evaluation_pairs.csv.gz",
        "parent": "results/bridge_relation_unseen_max_v3/targets.csv",
        "rhea": "results/bridge_rhea_longitudinal_growth_v4/targets.csv",
    }
    published = dataset["publication"]["published_snapshot"]["files"]
    for role, local_path in source_paths.items():
        assert published[role]["sha256"] == source_sha[local_path]
