"""Source checkout must not re-hash partial private/runtime asset trees."""

import json
from pathlib import Path

import pytest

from scripts.maintenance import build_bridge_release_manifests as manifests


def _profile():
    return manifests.load_profiles()["starase-application"]


def test_source_only_preserves_frozen_asset_claims(monkeypatch):
    previous = json.loads(
        (manifests.OUT_DIR / "starase-application.json").read_text(encoding="utf-8")
    )["assets"]
    monkeypatch.setattr(
        manifests,
        "_application_root_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Source-only CI must not inspect runtime asset roots")
        ),
    )
    result = manifests.application_assets(_profile(), source_only=True)
    assert result["asset_integrity_verification"] == "frozen_metadata_only_not_rematerialized"
    for key in (
        "current_information_roots",
        "tps_specialist_source_roots",
        "runtime_asset_roots",
    ):
        assert result[key] == previous[key]
    assert result["generated_application_assets"] == previous["generated_application_assets"]


def test_source_only_rejects_missing_frozen_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(manifests, "OUT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="previously materialized"):
        manifests.application_assets(_profile(), source_only=True)


def test_source_only_rejects_tampered_frozen_hash(tmp_path, monkeypatch):
    payload = json.loads(
        (manifests.OUT_DIR / "starase-application.json").read_text(encoding="utf-8")
    )
    payload["assets"]["current_information_roots"][0]["root_manifest_sha256"] = "bad"
    (tmp_path / "starase-application.json").write_text(json.dumps(payload))
    monkeypatch.setattr(manifests, "OUT_DIR", tmp_path)
    with pytest.raises(ValueError, match="Incomplete frozen runtime root provenance"):
        manifests.application_assets(_profile(), source_only=True)
