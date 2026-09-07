from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reproducibility/research_release_manifest.json"
GITHUB_BLOB_LIMIT = 100_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return {item.decode("utf-8") for item in completed.stdout.split(b"\0") if item}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Git-scoped research release.")
    parser.add_argument(
        "--portable-only",
        action="store_true",
        help="Require only directly tracked assets and rebuild contracts; do not require large local/external payloads.",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"Missing release manifest: {MANIFEST}", file=sys.stderr)
        return 1
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tracked = tracked_paths()
    failures: list[str] = []

    if payload.get("release_branch") != "master":
        failures.append(
            f"research release branch must be master, got {payload.get('release_branch')!r}"
        )

    direct = payload.get("direct_git_assets", [])
    for record in direct:
        relative = str(record["path"])
        path = ROOT / relative
        if relative not in tracked:
            failures.append(f"release asset is not Git-tracked: {relative}")
            continue
        if not path.is_file():
            failures.append(f"tracked release asset missing from checkout: {relative}")
            continue
        size = path.stat().st_size
        if size != int(record["bytes"]):
            failures.append(f"size mismatch: {relative}: {size} != {record['bytes']}")
        if size >= GITHUB_BLOB_LIMIT:
            failures.append(f"release asset exceeds GitHub blob limit: {relative}: {size}")
        digest = sha256(path)
        if digest != str(record["sha256"]):
            failures.append(f"sha256 mismatch: {relative}: {digest} != {record['sha256']}")

    direct_paths = {str(record["path"]) for record in direct}
    support = payload.get("evaluation_support_assets", [])
    support_paths: list[str] = []
    seen_support: set[str] = set()
    for item in support:
        relative = str(item.get("path", ""))
        if not relative:
            failures.append("evaluation support asset missing path")
            continue
        if relative in seen_support:
            failures.append(f"duplicate evaluation support asset: {relative}")
        seen_support.add(relative)
        support_paths.append(relative)
        if ".partial" in Path(relative).name:
            failures.append(f"partial/incomplete file cannot be evaluation support: {relative}")
        if relative not in direct_paths:
            failures.append(f"evaluation support asset absent from direct release assets: {relative}")
        if relative not in tracked:
            failures.append(f"evaluation support asset is not Git-tracked: {relative}")
        support_path = ROOT / relative
        if not support_path.is_file():
            failures.append(f"evaluation support asset missing: {relative}")
        if not str(item.get("role", "")).strip():
            failures.append(f"evaluation support asset missing role: {relative}")
        direct_record = next((record for record in direct if str(record["path"]) == relative), None)
        if direct_record is not None:
            if int(item.get("bytes", -1)) != int(direct_record["bytes"]):
                failures.append(f"evaluation support bytes drift from direct asset: {relative}")
            if str(item.get("sha256", "")) != str(direct_record["sha256"]):
                failures.append(f"evaluation support sha256 drift from direct asset: {relative}")


    aggregate_support = payload.get("aggregate_support_assets", [])
    seen_aggregate_support: set[str] = set()
    for item in aggregate_support:
        relative = str(item.get("path", ""))
        if not relative:
            failures.append("aggregate support asset missing path")
            continue
        if relative in seen_aggregate_support:
            failures.append(f"duplicate aggregate support asset: {relative}")
        seen_aggregate_support.add(relative)
        if ".partial" in Path(relative).name:
            failures.append(f"partial/incomplete file cannot be aggregate support: {relative}")
        if relative not in direct_paths:
            failures.append(f"aggregate support asset absent from direct release assets: {relative}")
        if relative not in tracked:
            failures.append(f"aggregate support asset is not Git-tracked: {relative}")
        support_path = ROOT / relative
        if not support_path.is_file():
            failures.append(f"aggregate support asset missing: {relative}")
        if not str(item.get("role", "")).strip():
            failures.append(f"aggregate support asset missing role: {relative}")
        direct_record = next((record for record in direct if str(record["path"]) == relative), None)
        if direct_record is not None:
            if int(item.get("bytes", -1)) != int(direct_record["bytes"]):
                failures.append(f"aggregate support bytes drift from direct asset: {relative}")
            if str(item.get("sha256", "")) != str(direct_record["sha256"]):
                failures.append(f"aggregate support sha256 drift from direct asset: {relative}")

    direct_by_path = {str(record["path"]): record for record in direct}
    for contract in payload.get("evaluation_support_rebuilds", []):
        asset = str(contract.get("asset", ""))
        builder = str(contract.get("builder", ""))
        command = str(contract.get("command", ""))
        expected_sha = str(contract.get("expected_sha256", ""))
        if asset not in support_paths:
            failures.append(f"evaluation-support rebuild target is not declared support: {asset}")
        if not builder or builder not in tracked or not (ROOT / builder).is_file():
            failures.append(f"evaluation-support rebuild builder missing/untracked: {builder}")
        if builder and builder not in command:
            failures.append(f"evaluation-support rebuild command does not invoke builder: {asset}")
        if contract.get("model_scoring") is not False:
            failures.append(f"evaluation-support rebuild must explicitly disable model scoring: {asset}")
        if contract.get("protocol_modified") is not False:
            failures.append(f"evaluation-support rebuild must preserve frozen protocol: {asset}")
        direct_record = direct_by_path.get(asset)
        if direct_record and expected_sha != str(direct_record.get("sha256", "")):
            failures.append(f"evaluation-support rebuild SHA drift: {asset}")

    publication = payload.get("publication_metadata", {})
    citation_file = str(publication.get("citation_file", ""))
    notices_file = str(publication.get("third_party_notices", ""))
    for label, relative in (("citation", citation_file), ("third-party notices", notices_file)):
        if not relative or relative not in tracked or not (ROOT / relative).is_file():
            failures.append(f"publication {label} file missing/untracked: {relative}")
    license_status = str(publication.get("project_license_status", ""))
    license_file = publication.get("project_license_file")
    tracked_license_candidates = [x for x in ("LICENSE", "LICENSE.md", "COPYING") if x in tracked]
    if license_status == "not_declared":
        if license_file is not None:
            failures.append("project license status is not_declared but a license file is declared")
        if tracked_license_candidates:
            failures.append("project license status is not_declared but a repository-level license file is tracked")
    elif license_status == "declared":
        relative = str(license_file or "")
        if not relative or relative not in tracked or not (ROOT / relative).is_file():
            failures.append("declared project license file is missing/untracked")
    else:
        failures.append(f"invalid project license status: {license_status!r}")

    canonical = json.loads((ROOT / "reproducibility/bime_rank/canonical.json").read_text())
    allowed_release_roles = {
        "production_contract",
        "external_benchmark",
        "conditional_capability",
        "method_ablation",
        "execution_evidence",
        "application_case",
        "release_presentation",
    }
    declared = payload.get("canonical_claim_primaries", {})
    for claim_id, claim in canonical["claims"].items():
        release_role = str(claim.get("release_role", ""))
        if release_role not in allowed_release_roles:
            failures.append(
                f"canonical claim lacks a valid current-release role: {claim_id}: {release_role!r}"
            )
        relative = str(claim["primary"])
        if declared.get(claim_id) != relative:
            failures.append(f"canonical claim manifest drift: {claim_id}")
        if relative not in tracked:
            failures.append(f"canonical primary is not Git-tracked: {claim_id}: {relative}")
        if not (ROOT / relative).is_file():
            failures.append(f"canonical primary missing: {claim_id}: {relative}")

    model_index_path = ROOT / "reproducibility/bime_rank/model_assets.json"
    if not model_index_path.is_file():
        failures.append("missing current production model asset index")
    else:
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        route_relative = str(model_index.get("authority", ""))
        route_path = ROOT / route_relative
        if route_relative != "configs/production_routes/terpene_v1.yaml":
            failures.append(f"model asset authority drift: {route_relative}")
        if route_relative not in tracked or not route_path.is_file():
            failures.append(f"model asset authority missing from Git: {route_relative}")
        elif sha256(route_path) != str(model_index.get("route_sha256", "")):
            failures.append("model asset index route sha256 drift")
        direct_paths = {str(x["path"]) for x in direct}
        for record in model_index.get("project_owned_assets", []):
            relative = str(record["path"])
            path = ROOT / relative
            if relative not in tracked:
                failures.append(f"current project model asset is not Git-tracked: {relative}")
                continue
            if relative not in direct_paths:
                failures.append(f"current project model asset missing from direct release assets: {relative}")
            if not path.is_file():
                failures.append(f"current project model asset missing: {relative}")
                continue
            if path.stat().st_size != int(record.get("bytes", -1)):
                failures.append(f"current project model asset size mismatch: {relative}")
            if sha256(path) != str(record.get("sha256", "")):
                failures.append(f"current project model asset sha256 mismatch: {relative}")
            if path.stat().st_size >= GITHUB_BLOB_LIMIT:
                failures.append(f"current project model asset exceeds GitHub blob limit: {relative}")
        release_external = {str(x["target"]): x for x in payload.get("external_assets", [])}
        for record in model_index.get("external_model_assets", []):
            target = str(record["target"])
            expected = release_external.get(target)
            if expected is None:
                failures.append(f"external production model absent from release contract: {target}")
                continue
            if str(expected.get("sha256", "")) != str(record.get("sha256", "")):
                failures.append(f"external production model sha256 contract drift: {target}")
            if target in tracked:
                failures.append(f"external production model must not be vendored in normal Git: {target}")
        bundle_counts = model_index.get("bundle_asset_counts", {})
        for bundle in model_index.get("model_bundles", []):
            if int(bundle_counts.get(bundle, 0)) <= 0:
                failures.append(f"production model bundle has no indexed assets: {bundle}")

    rebuildable_paths = {str(item["path"]) for item in payload.get("rebuildable_assets", [])}
    for item in payload.get("rebuildable_assets", []):
        kind = str(item.get("kind", ""))
        if kind == "derived_feature_matrix":
            builder = str(item.get("builder", ""))
            command = str(item.get("command", ""))
            if not builder:
                failures.append(f"derived feature matrix missing builder: {item['path']}")
            if not command:
                failures.append(f"derived feature matrix missing rebuild command: {item['path']}")
            elif builder and builder not in command:
                failures.append(f"rebuild command does not invoke declared builder: {item['path']}")
        elif kind == "training_cache":
            replacement = str(item.get("replacement", ""))
            if item.get("required_for_inference") is not False:
                failures.append(f"training cache must be explicitly non-inference: {item['path']}")
            if not replacement or replacement not in rebuildable_paths:
                failures.append(f"training cache replacement is not a declared rebuildable asset: {item['path']}")
        for key in ("builder", "merge_builder", "metadata_manifest", "precompute_provenance"):
            relative = item.get(key)
            if not relative:
                continue
            if relative not in tracked:
                failures.append(f"rebuild contract {key} is not Git-tracked: {relative}")
            if not (ROOT / relative).is_file():
                failures.append(f"rebuild contract {key} missing: {relative}")
        for relative in item.get("precomputed_inputs", []):
            relative = str(relative)
            if relative not in tracked:
                failures.append(f"rebuild precomputed input is not Git-tracked: {relative}")
            if not (ROOT / relative).is_file():
                failures.append(f"rebuild precomputed input missing: {relative}")
        path = ROOT / str(item["path"])
        if args.portable_only or not path.exists():
            continue
        expected_shape = item.get("expected_shape")
        if expected_shape and path.suffix in {".npy", ".npz"}:
            try:
                import numpy as np
                if path.suffix == ".npy":
                    shape = list(np.load(path, mmap_mode="r").shape)
                else:
                    from scipy import sparse
                    shape = list(sparse.load_npz(path).shape)
                if shape != list(expected_shape):
                    failures.append(f"rebuildable shape mismatch: {item['path']}: {shape} != {expected_shape}")
            except Exception as exc:
                failures.append(f"cannot inspect rebuildable asset {item['path']}: {exc!r}")

    for asset in payload.get("external_assets", []):
        relative = str(asset["target"])
        if not asset.get("sha256"):
            failures.append(f"external asset missing sha256 contract: {relative}")
        if not (asset.get("repository") or asset.get("zenodo_doi") or asset.get("zenodo_record") or asset.get("upstream_archive_url") or asset.get("upstream_url")):
            failures.append(f"external asset missing source provenance locator: {relative}")
        if relative in tracked:
            failures.append(f"large third-party asset must not be vendored in normal Git: {relative}")
        if args.portable_only:
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"external asset missing: {relative}")
            continue
        expected_size = asset.get("bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            failures.append(f"external asset size mismatch: {relative}")
        expected_sha = asset.get("sha256")
        if expected_sha and sha256(path) != str(expected_sha):
            failures.append(f"external asset sha256 mismatch: {relative}")

    # Exact evaluator snapshots must not depend on undeclared server-local replay inputs.
    # Discover these snapshots from canonical provenance rather than maintaining a second
    # evaluator list. Literal ROOT/'...' roots are covered when the path itself or at least
    # one descendant is a direct, rebuildable, or external release asset.
    provenance_for_replay = json.loads(
        (ROOT / "reproducibility/bime_rank/canonical_source_provenance.json").read_text(encoding="utf-8")
    )
    declared_roots = set(direct_paths)
    declared_roots.update(str(item["path"]) for item in payload.get("rebuildable_assets", []))
    declared_roots.update(str(item["target"]) for item in payload.get("external_assets", []))
    exact_snapshots: list[str] = []
    replay_literal_roots: set[str] = set()
    for claim in provenance_for_replay.get("claims", {}).values():
        for source in claim.get("sources", []):
            if source.get("role") != "exact_final_external_confirmation_evaluator_snapshot":
                continue
            snapshot = str(source.get("path", ""))
            exact_snapshots.append(snapshot)
            snapshot_path = ROOT / snapshot
            if not snapshot_path.is_file():
                failures.append(f"exact evaluator snapshot missing: {snapshot}")
                continue
            text = snapshot_path.read_text(encoding="utf-8", errors="strict")
            for relative in re.findall(r"ROOT\s*/\s*['\"]([^'\"]+)['\"]", text):
                replay_literal_roots.add(relative)
                prefix = relative.rstrip("/") + "/"
                covered = relative in declared_roots or any(
                    item.startswith(prefix) for item in declared_roots
                )
                if not covered:
                    failures.append(
                        f"exact evaluator has uncovered repo-local replay input: {snapshot}: {relative}"
                    )

    database_index_path = ROOT / "reproducibility/bime_rank/database_assets.json"
    if not database_index_path.is_file():
        failures.append("missing canonical database asset index")
    else:
        database_index = json.loads(database_index_path.read_text(encoding="utf-8"))
        expected_counts = {"proteins": 185918, "reactions": 11081, "associations": 246610}
        for key, expected in expected_counts.items():
            if int(database_index.get("counts", {}).get(key, -1)) != expected:
                failures.append(f"canonical database {key} count drift")
        direct_paths = {str(x["path"]) for x in direct}
        for record in database_index.get("canonical_tables", []):
            relative = str(record["path"])
            path = ROOT / relative
            if relative not in tracked or relative not in direct_paths:
                failures.append(f"canonical database table not in direct Git release: {relative}")
                continue
            if not path.is_file():
                failures.append(f"canonical database table missing: {relative}")
                continue
            if path.stat().st_size != int(record.get("bytes", -1)):
                failures.append(f"canonical database table size mismatch: {relative}")
            if sha256(path) != str(record.get("sha256", "")):
                failures.append(f"canonical database table sha256 mismatch: {relative}")
        historical = database_index.get("historical_assembly", {})
        builder = str(historical.get("builder", ""))
        if builder not in tracked or not (ROOT / builder).is_file():
            failures.append("historical database assembly builder missing from Git")
        for source in historical.get("source_files", []):
            if source.get("portable_release_input") and str(source.get("path", "")) not in tracked:
                failures.append(f"database source marked portable but not tracked: {source.get('path')}")
        for precompute in database_index.get("portable_precomputed_assets", []):
            record = precompute.get("portable_precompute", {})
            relative = str(record.get("path", ""))
            path = ROOT / relative
            if relative not in tracked or relative not in direct_paths:
                failures.append(f"portable database precompute not in direct Git release: {relative}")
            elif not path.is_file() or sha256(path) != str(record.get("sha256", "")):
                failures.append(f"portable database precompute hash mismatch: {relative}")

    provenance_path = ROOT / "reproducibility/bime_rank/canonical_source_provenance.json"
    retained_provenance_project_sources: set[str] = set()
    if not provenance_path.is_file():
        failures.append("missing canonical source provenance")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance_claims = provenance.get("claims", {})
        if set(provenance_claims) != set(canonical["claims"]):
            failures.append("canonical source provenance claim IDs drift from canonical.json")
        allowed_status = {
            "runtime_contract",
            "direct_generator",
            "frozen_finalization_with_upstream_source",
            "curated_aggregate_with_components",
            "frozen_result_only",
            "source_snapshot",
            "curated_presentation_contract",
        }
        for claim_id, claim in canonical["claims"].items():
            entry = provenance_claims.get(claim_id, {})
            if entry.get("canonical_primary") != claim["primary"]:
                failures.append(f"canonical source provenance primary drift: {claim_id}")
            status = str(entry.get("final_generator_status", ""))
            if status not in allowed_status:
                failures.append(f"invalid canonical source provenance status: {claim_id}: {status}")
            if entry.get("missing_final_generator"):
                failures.append(f"current canonical claim lacks a final generator: {claim_id}")
            if not str(entry.get("boundary", "")).strip():
                failures.append(f"missing canonical source provenance boundary: {claim_id}")
            for source in entry.get("sources", []):
                relative = str(source.get("path", ""))
                if not relative:
                    failures.append(f"empty canonical source path: {claim_id}")
                    continue
                if relative not in tracked:
                    failures.append(f"canonical provenance source is not Git-tracked: {claim_id}: {relative}")
                if not (ROOT / relative).is_file():
                    failures.append(f"canonical provenance source missing: {claim_id}: {relative}")
                if source.get("retain_in_reproduction_source") and relative.startswith("projects/active/terpene_screening/") and relative.endswith(".py"):
                    retained_provenance_project_sources.add(relative)

    source_roles_path = ROOT / "reproducibility/bime_rank/source_roles.json"
    if not source_roles_path.is_file():
        failures.append("missing BiME-Rank source-role manifest")
    else:
        source_roles = json.loads(source_roles_path.read_text(encoding="utf-8"))
        if source_roles.get("release_branch") != "master":
            failures.append("BiME-Rank source-role manifest must target master")
        current_runtime = set(source_roles.get("current_runtime", []))
        reproduction = set(source_roles.get("canonical_reproduction", []))
        release_regression = set(source_roles.get("release_regression", []))
        extended_reproduction = set(source_roles.get("extended_reproduction_tests", []))
        historical_source = set(source_roles.get("historical_research_source", []))
        historical_lineage_tests = set(source_roles.get("historical_lineage_tests", []))
        if current_runtime & reproduction:
            failures.append("source-role current_runtime/canonical_reproduction overlap")
        current_union = current_runtime | reproduction | release_regression
        retained_test_union = release_regression | extended_reproduction
        historical_union = historical_source | historical_lineage_tests
        if current_union & historical_union or extended_reproduction & historical_union:
            failures.append("current/reproduction source is also classified historical")
        if historical_lineage_tests:
            failures.append(f"historical lineage tests remain Git-tracked: {len(historical_lineage_tests)}")
        project_python = {
            rel for rel in tracked
            if rel.startswith("projects/active/terpene_screening/") and rel.endswith(".py")
        }
        classified = current_union | extended_reproduction | historical_union
        missing_provenance_sources = sorted(retained_provenance_project_sources - (current_runtime | reproduction))
        if missing_provenance_sources:
            failures.append(f"canonical provenance project source not retained in reproduction roles: {len(missing_provenance_sources)}")
        missing_classification = sorted(project_python - classified)
        stale_classification = sorted(classified - project_python)
        if missing_classification:
            failures.append(f"unclassified tracked project Python: {len(missing_classification)}")
        if stale_classification:
            failures.append(f"source-role entries are not tracked project Python: {len(stale_classification)}")
        for relative in current_union:
            if relative not in tracked or not (ROOT / relative).is_file():
                failures.append(f"current/reproduction source missing from Git: {relative}")


    demotion_audits = [
        ROOT / "reproducibility/bime_rank/historical_source_demotions.json",
        ROOT / "reproducibility/bime_rank/historical_research_source_demotions.json",
    ]
    for demotions_path in demotion_audits:
        if not demotions_path.is_file():
            failures.append(f"missing historical source demotion audit: {demotions_path.relative_to(ROOT)}")
            continue
        demotions = json.loads(demotions_path.read_text(encoding="utf-8"))
        records = demotions.get("records", [])
        if int(demotions.get("count", -1)) != len(records):
            failures.append(f"historical source demotion count mismatch: {demotions_path.relative_to(ROOT)}")
        if int(demotions.get("deleted", -1)) != 0:
            failures.append(f"historical source demotion audit must record deleted=0: {demotions_path.relative_to(ROOT)}")
        for record in records:
            relative = str(record.get("path", ""))
            if relative in tracked:
                failures.append(f"demoted historical source is still Git-tracked: {relative}")
            local = ROOT / relative
            # A clean clone intentionally does not contain demoted files. On a development
            # server, if a local copy exists, require it to remain byte-identical to audit.
            if local.is_file():
                if local.stat().st_size != int(record.get("bytes", -1)):
                    failures.append(f"demoted local source size mismatch: {relative}")
                elif sha256(local) != str(record.get("sha256", "")):
                    failures.append(f"demoted local source sha256 mismatch: {relative}")

    private_roots = [str(value) for value in payload.get("private_roots", [])]
    for relative in tracked:
        if any(relative.startswith(root) for root in private_roots):
            failures.append(f"private/local path is Git-tracked: {relative}")

    # Unrelated local development branches may exist on the server, but must not be in a research checkout.
    for forbidden in ("src/experimental_touch_cascade/", "configs/experimental_touch/", "projects/planned/"):
        offenders = sorted(path for path in tracked if path.startswith(forbidden))
        if offenders:
            failures.append(f"non-release Git surface remains under {forbidden}: {len(offenders)} files")

    result = {
        "status": "valid" if not failures else "invalid",
        "portable_only": args.portable_only,
        "direct_git_assets": len(direct),
        "direct_git_bytes": sum(int(x["bytes"]) for x in direct),
        "canonical_claims": len(canonical["claims"]),
        "rebuildable_assets": len(payload.get("rebuildable_assets", [])),
        "external_assets": len(payload.get("external_assets", [])),
        "project_model_assets": len(json.loads((ROOT / "reproducibility/bime_rank/model_assets.json").read_text()).get("project_owned_assets", [])) if (ROOT / "reproducibility/bime_rank/model_assets.json").is_file() else 0,
        "canonical_database_tables": len(json.loads((ROOT / "reproducibility/bime_rank/database_assets.json").read_text()).get("canonical_tables", [])) if (ROOT / "reproducibility/bime_rank/database_assets.json").is_file() else 0,
        "evaluation_support_assets": len(payload.get("evaluation_support_assets", [])),
        "aggregate_support_assets": len(payload.get("aggregate_support_assets", [])),
        "exact_replay_snapshots": len(set(exact_snapshots)),
        "exact_replay_literal_roots": len(replay_literal_roots),
        "failures": len(failures),
    }
    print(json.dumps(result, indent=2))
    if failures:
        for failure in failures[:200]:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
