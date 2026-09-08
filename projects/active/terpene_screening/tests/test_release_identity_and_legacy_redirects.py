from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[4]


def test_current_retrieval_status_is_a_compatibility_redirect():
    text = (ROOT / "projects/active/terpene_screening/CURRENT_RETRIEVAL_STATUS.md").read_text()
    assert "Compatibility redirect" in text
    assert "canonical.json" in text and "terpene_v1.yaml" in text
    assert "BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json" in text
    assert "not" in text and "independent current-result document" in text


def test_historical_human_docs_are_redirects_not_current_authority():
    for name in (
        "CURRENT_STRONGEST_VS_ENZYMECAGE.md",
        "RETRIEVAL_CAPABILITY_SCORECARD.md",
        "RETRIEVAL_EVIDENCE_LEDGER.md",
    ):
        text = (ROOT / "projects/active/terpene_screening" / name).read_text()
        assert "canonical.json" in text
        assert "superseded" in text
        assert "历史原文" in text


def test_python_distribution_uses_public_release_identity():
    text = (ROOT / "pyproject.toml").read_text()
    assert 'name = "bime-rank-research-release"' in text
    assert 'name = "igem2026-research-workspace"' not in text
    assert 'authors = [{name = "NJU-China"}]' in text


def test_citation_names_the_release_author():
    text = (ROOT / "CITATION.cff").read_text()
    assert '  - name: "NJU-China"' in text
    assert "BiME-Rank project team" not in text


def test_public_project_readme_describes_one_bime_rank_system():
    text = (ROOT / "projects/active/terpene_screening/README.md").read_text()
    assert "BiME-Rank" in text
    assert "Bidirectional Multi-Expert Learning-to-Rank" in text
    assert "bime-rank-production-routes-v2" in text
    assert "185,918 proteins" in text and "11,081 reactions" in text
    assert "availability-aware structural evidence" in text.lower()
    assert "Known-positive context" in text
    assert "Cost-aware execution" in text
    assert "not eight competing" in text


def test_current_scorecard_keeps_claim_scopes_separate():
    score = json.loads(
        (ROOT / "projects/active/terpene_screening/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json").read_text()
    )
    assert score["method"] == "BiME-Rank"
    assert score["candidate_universes"] == {"r2e_proteins": 185918, "e2r_reactions": 11081}
    rules = " ".join(score["presentation_rules"])
    assert "different candidate universes" in rules
    assert "known-positive" in rules
    zero = score["zero_shot_external"]
    assert "clipzyme_strict_temporal_same_support" in zero
    assert "enzyme405" in zero
    assert set(zero) == {"clipzyme_strict_temporal_same_support", "enzyme405"}
    assert "tps_enzymecage_same_support" not in zero
    assert "selenzyme_author_pool" not in zero
    assert "Historical precursor results" in rules
    assert "conditional_known_positive" in score
    assert score["conditional_known_positive"]["r2e"]["claim_boundary"].startswith("one-known-positive")


def test_release_manifest_and_ci_use_master():
    manifest = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    assert manifest["release_branch"] == "master"
    workflow = (ROOT / ".github/workflows/terpene-ci.yml").read_text()
    assert "branches: [master]" in workflow
    assert "branches: [main]" not in workflow
    assert "build_research_release_manifest.py" in workflow
    assert "git diff --exit-code -- reproducibility/research_release_manifest.json" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "bime-rank-release-validation-${{ github.sha }}" in workflow

def test_source_roles_cover_project_python_and_separate_current_from_history():
    roles = json.loads((ROOT / "reproducibility/bime_rank/source_roles.json").read_text())
    assert roles["release_branch"] == "master"
    assert roles["method_identity"] == "BiME-Rank"
    current = set(roles["current_runtime"]) | set(roles["canonical_reproduction"]) | set(roles["release_regression"])
    extended = set(roles["extended_reproduction_tests"])
    history = set(roles["historical_research_source"]) | set(roles["historical_lineage_tests"])
    assert current.isdisjoint(history)
    assert extended.isdisjoint(history)
    assert roles["historical_lineage_tests"] == []
    assert "projects/active/terpene_screening/rank_open_world.py" in current
    assert "projects/active/terpene_screening/bime_rank_r2e_runtime.py" in current
    assert "projects/active/terpene_screening/bime_rank_e2r_runtime.py" in current
    assert roles["counts"]["tracked_project_python"] == len(current | extended | history)
    demotions = json.loads((ROOT / "reproducibility/bime_rank/historical_source_demotions.json").read_text())
    assert demotions["category"] == "historical_lineage_tests"
    assert demotions["count"] == len(demotions["records"]) and demotions["deleted"] == 0


def test_release_records_both_git_only_source_demotion_audits():
    manifest = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    validation = manifest["validation"]
    assert validation["historical_source_demotions"] == "reproducibility/bime_rank/historical_source_demotions.json"
    assert validation["historical_research_source_demotions"] == "reproducibility/bime_rank/historical_research_source_demotions.json"
    auxiliary = json.loads((ROOT / validation["historical_research_source_demotions"]).read_text())
    assert auxiliary["category"] == "historical_research_auxiliary_source"
    assert auxiliary["deleted"] == 0
    assert auxiliary["count"] == len(auxiliary["records"])


def test_current_model_asset_index_is_route_derived_and_complete():
    model_assets = json.loads((ROOT / "reproducibility/bime_rank/model_assets.json").read_text())
    assert model_assets["authority"] == "configs/production_routes/terpene_v1.yaml"
    assert model_assets["counts"]["production_model_bundles"] >= 1
    assert model_assets["counts"]["learned_parameters"] >= 1
    assert model_assets["counts"]["external_model_assets"] == 3
    assert all(record["sha256"] for record in model_assets["project_owned_assets"])
    assert all((ROOT / record["path"]).is_file() for record in model_assets["project_owned_assets"])


def test_derived_release_matrices_have_executable_build_contracts():
    manifest = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    for asset in manifest["rebuildable_assets"]:
        if asset["kind"] == "derived_feature_matrix":
            assert asset.get("builder")
            assert asset.get("command")
            assert asset["builder"] in asset["command"]
        elif asset["kind"] == "training_cache":
            assert asset["required_for_inference"] is False
            assert asset.get("replacement")


def test_canonical_database_release_is_portable_and_count_locked():
    database = json.loads((ROOT / "reproducibility/bime_rank/database_assets.json").read_text())
    assert database["counts"]["proteins"] == 185918
    assert database["counts"]["reactions"] == 11081
    assert database["counts"]["associations"] == 246610
    assert database["counts"]["canonical_tables"] == 7
    assert database["counts"]["historical_assembly_source_files"] == 13
    assert database["counts"]["historical_assembly_sources_not_vendored"] == 7
    assert database["counts"]["model_ready_rebuildable_assets"] == 5
    assert all((ROOT / table["path"]).is_file() for table in database["canonical_tables"])


def test_rxnmapper_precompute_closes_reaction_center_rebuild_boundary():
    mapping = json.loads((ROOT / "reproducibility/bime_rank/rxnmapper_general_merged_v1.json").read_text())
    pre = mapping["portable_precompute"]
    assert pre["reaction_count"] == 11081
    assert pre["successful_mappings"] == 10839
    assert pre["failed_mappings"] == 242
    assert (ROOT / pre["path"]).is_file()
    manifest = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    center = next(x for x in manifest["rebuildable_assets"] if x["path"].endswith("drfp_categorical_rdkitplus_center_v1/reaction_feature_matrix.npy"))
    assert pre["path"] in center["precomputed_inputs"]

def test_all_current_canonical_claims_have_final_replay_boundaries():
    canonical = json.loads((ROOT / "reproducibility/bime_rank/canonical.json").read_text())
    provenance = json.loads((ROOT / "reproducibility/bime_rank/canonical_source_provenance.json").read_text())
    assert set(provenance["claims"]) == set(canonical["claims"])
    assert len(canonical["claims"]) == 12
    assert not [claim for claim, value in provenance["claims"].items() if value.get("missing_final_generator")]


def test_publication_metadata_and_ci_artifact_are_explicit():
    manifest = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    publication = manifest["publication_metadata"]
    assert publication["citation_file"] == "CITATION.cff"
    assert publication["third_party_notices"] == "THIRD_PARTY_NOTICES.md"
    assert publication["project_license_status"] == "not_declared"
    assert publication["project_license_file"] is None
    assert (ROOT / publication["citation_file"]).is_file()
    assert (ROOT / publication["third_party_notices"]).is_file()
    workflow = (ROOT / ".github/workflows/terpene-ci.yml").read_text()
    assert "bime-rank-release-validation-${{ github.sha }}" in workflow
    assert "cp CITATION.cff" in workflow
    assert "cp THIRD_PARTY_NOTICES.md" in workflow
    assert "cp reproducibility/bime_rank/historical_source_demotions.json" in workflow
    assert "cp reproducibility/bime_rank/historical_research_source_demotions.json" in workflow
    assert "cp reproducibility/bime_rank/historical_artifact_demotions.json" in workflow
    assert "cp reproducibility/bime_rank/enzgfm_stage2_530_timing_20260907.json" in workflow
    assert "cp reproducibility/bime_rank/rxnmapper_general_merged_v1.json" in workflow
    assert "release-status.json" in workflow

def test_plain_pytest_is_scoped_to_release_maintenance_tests():
    text = (ROOT / "pyproject.toml").read_text()
    assert '"scripts/maintenance/tests"' in text
    assert 'testpaths = [\n  "projects/active/terpene_screening/tests"' not in text
    root_readme = (ROOT / "README.md").read_text()
    assert "run_bime_project_tests.py --tier release" in root_readme
    assert "run_bime_project_tests.py --tier extended" in root_readme

def test_documentation_map_describes_public_and_local_demotions_correctly():
    text = (ROOT / "docs/README.md").read_text()
    for name in (
        "historical_source_demotions.json",
        "historical_research_source_demotions.json",
        "historical_artifact_demotions.json",
    ):
        assert name in text
    assert ".git/info/exclude" in text
    assert "Exact ignored paths are documented in `.gitignore`" not in text
    assert "bime-rank-release-validation-<commit>" in text


def test_asset_audit_uses_machine_gates_instead_of_stale_test_counts():
    text = (ROOT / "docs/BIME_ASSET_AUDIT.md").read_text()
    assert "run_bime_project_tests.py --tier release|extended" in text
    assert "共 15 项" not in text
    assert "本文路径均相对仓库根目录" in text

def test_current_route_file_evidence_is_direct_and_legacy_runtime_weights_are_demoted():
    release = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    direct = {record["path"] for record in release["direct_git_assets"]}
    for path in (
        "results/bime_rank_unified_v1/r2e_seed_context_v1/development_result.json",
        "results/bime_rank_unified_v1/e2r_seed_context_v1/development_result.json",
    ):
        assert path in direct
    audit = json.loads((ROOT / "reproducibility/bime_rank/historical_runtime_asset_demotions.json").read_text())
    assert audit["count"] == len(audit["records"]) and audit["count"] > 0
    assert audit["bytes"] == sum(record["bytes"] for record in audit["records"])
    assert audit["deleted"] == 0
    model_index = json.loads((ROOT / "reproducibility/bime_rank/model_assets.json").read_text())
    current_models = {record["path"] for record in model_index["project_owned_assets"]}
    for record in audit["records"]:
        assert record["path"] not in direct
        assert record["path"] not in current_models


def test_validation_artifact_includes_runtime_demotion_audit():
    workflow = (ROOT / ".github/workflows/terpene-ci.yml").read_text()
    assert 'historical_runtime_asset_demotions.json "$OUT/"' in workflow
    assert "'historical_runtime_assets_demoted': runtime_demotions['count']" in workflow

def test_research_release_builder_uses_runtime_demotion_audit_as_single_exclusion_source():
    text = (ROOT / "scripts/maintenance/build_research_release_manifest.py").read_text()
    assert 'RUNTIME_DEMOTIONS = ROOT / "reproducibility/bime_rank/historical_runtime_asset_demotions.json"' in text
    assert "historical_runtime_direct_excludes(runtime_files)" in text
    assert "LEGACY_RUNTIME_DIRECT_EXCLUDES" not in text

def test_tracked_data_results_surface_is_direct_or_declared_historical_primary():
    import subprocess

    release = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    direct = {record["path"] for record in release["direct_git_assets"]}
    provenance = json.loads((ROOT / "reproducibility/bime_rank/canonical_source_provenance.json").read_text())
    retained = {
        entry["primary"]
        for entry in provenance.get("historical_or_supplemental", {}).values()
        if entry.get("primary", "").startswith(("data/", "results/"))
        and (entry.get("historical_result_retained") or entry.get("reproducible_historical_result"))
    }
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    outside = {p for p in tracked if p.startswith(("data/", "results/")) and p not in direct}
    assert outside == retained

def test_current_wetlab_reproduction_uses_frozen_bime_finalizer_not_legacy_tps_campaign_seeds():
    provenance = json.loads((ROOT / "reproducibility/bime_rank/canonical_source_provenance.json").read_text())
    wetlab = provenance["claims"]["wetlab_success_first"]
    sources = {source["path"] for source in wetlab["sources"]}
    assert "reproducibility/bime_rank/source_snapshots/manual_success_first_finalize.py" in sources
    roles = json.loads((ROOT / "reproducibility/bime_rank/source_roles.json").read_text())
    legacy = {
        "projects/active/terpene_screening/balance_wetlab_reactions_across_plates.py",
        "projects/active/terpene_screening/build_combined_wetlab_campaign.py",
        "projects/active/terpene_screening/build_wetlab_discovery_panels.py",
        "projects/active/terpene_screening/build_wetlab_plate_manifest.py",
        "projects/active/terpene_screening/manage_wetlab_feedback.py",
        "projects/active/terpene_screening/randomize_wetlab_candidate_positions.py",
    }
    assert legacy.isdisjoint(set(roles["reproduction_seeds"]))
    assert legacy.isdisjoint(set(roles["canonical_reproduction"]))
