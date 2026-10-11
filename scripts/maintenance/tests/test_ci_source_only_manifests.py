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


def test_release_artifact_reports_existing_source_role_fields():
    """Prevent a stale GitHub workflow from reintroducing missing count keys."""
    import yaml

    source = json.loads(
        Path("reproducibility/bime_rank/source_roles.json").read_text()
    )
    workflow = yaml.safe_load(Path(".github/workflows/research-release.yml").read_text())
    steps = workflow["jobs"]["portable-release"]["steps"]
    step = next(x for x in steps if x.get("name") == "Prepare release validation artifact")
    body = step["run"]
    assert "source['counts']['tracked_project_python']" not in body
    assert "source['counts']['historical_research_source']" not in body
    assert "source['counts']['tracked_active_and_reproduction_python']" in body
    assert "len(source['historical_research_source'])" in body
    assert int(source["counts"]["tracked_active_and_reproduction_python"]) > 0
    assert len(source["historical_research_source"]) == int(
        source["counts"]["reproduction_support_source"]
    )
