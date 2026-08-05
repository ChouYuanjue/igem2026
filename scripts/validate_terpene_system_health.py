from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.conformal import (
    CONFORMAL_RETRIEVAL_VERSION,
    DEFAULT_CONFORMAL_CALIBRATORS,
)
from projects.active.terpene_screening.core.provenance import identifier_set_hash
from projects.active.terpene_screening.core.registry_snapshots import (
    current_snapshot_root,
    load_snapshot_manifest,
)
from projects.active.terpene_screening.core.routing import (
    DEFAULT_ROUTE_MANIFEST,
    load_route_manifest,
    resolve_route,
)
from projects.active.terpene_screening.rank_open_world import (
    DEFAULT_E2R_DUAL_TOWER_DIR,
    DEFAULT_PROTEIN_DIR,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    DEFAULT_UNCERTAINTY_CALIBRATORS,
    load_external_reaction_rows,
    load_feature_schema,
    load_protein_library,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_check(checks: list[dict[str, object]], name: str, ok: bool, detail: object) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the complete non-pocket terpene production system.")
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "results/terpene_system_health.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    payload = load_route_manifest(str(args.route_manifest.resolve()))
    add_check(checks, "route_manifest_version", payload.get("manifest_version") == 1, payload.get("route_version"))
    for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
        for scope in ["current", "external"]:
            for objective in ["top3", "top10", "top20"]:
                route = resolve_route(
                    direction=direction,
                    objective=objective,
                    is_current=scope == "current",
                    manifest_path=args.route_manifest,
                )
                add_check(checks, f"route:{route.route_id}:deployment", route.deployment.exists(), str(route.deployment))
                if route.secondary_deployment is not None:
                    add_check(checks, f"route:{route.route_id}:secondary", route.secondary_deployment.exists(), str(route.secondary_deployment))
                if route.auxiliary_deployment is not None:
                    add_check(checks, f"route:{route.route_id}:auxiliary", route.auxiliary_deployment.exists(), str(route.auxiliary_deployment))

    registry_root = DEFAULT_REGISTERED_PROTEIN_DIR.parent
    snapshot = current_snapshot_root(registry_root)
    manifest = load_snapshot_manifest(registry_root)
    add_check(checks, "registry_snapshot_active", snapshot is not None, manifest or "legacy")
    if snapshot is not None:
        mirror_pairs = [
            (snapshot / "proteins/embeddings.npy", DEFAULT_REGISTERED_PROTEIN_DIR / "embeddings.npy"),
            (snapshot / "proteins/entries.csv", DEFAULT_REGISTERED_PROTEIN_DIR / "entries.csv"),
            (snapshot / "proteins/metadata.csv", DEFAULT_REGISTERED_PROTEIN_DIR / "metadata.csv"),
            (snapshot / "reactions.csv", DEFAULT_REGISTERED_REACTIONS),
        ]
        for current, legacy in mirror_pairs:
            ok = current.exists() and legacy.exists() and sha256(current) == sha256(legacy)
            add_check(checks, f"registry_mirror:{current.name}", ok, {"snapshot": str(current), "legacy": str(legacy)})

    _, current_ids = load_protein_library(DEFAULT_PROTEIN_DIR)
    _, registered_ids = load_protein_library(DEFAULT_REGISTERED_PROTEIN_DIR)
    current_set = set(current_ids)
    protein_ids = current_ids + [value for value in registered_ids if value not in current_set]
    schema = load_feature_schema(DEFAULT_E2R_DUAL_TOWER_DIR)
    reaction_ids = [str(value) for value in schema["reaction_ids"]]
    reaction_set = set(reaction_ids)
    registered_reactions = load_external_reaction_rows(DEFAULT_REGISTERED_REACTIONS)
    reaction_ids.extend(
        value
        for value in registered_reactions["reaction_id"].astype(str)
        if value not in reaction_set
    )
    candidate_hashes = {
        "reaction_to_enzyme": identifier_set_hash(protein_ids),
        "enzyme_to_reaction": identifier_set_hash(reaction_ids),
    }
    add_check(checks, "candidate_counts", len(protein_ids) == 2085 and len(reaction_ids) == 753, {"proteins": len(protein_ids), "reactions": len(reaction_ids)})

    calibrators = json.loads(DEFAULT_UNCERTAINTY_CALIBRATORS.read_text(encoding="utf-8"))
    for key, calibrator in calibrators.items():
        if key.startswith("_"):
            continue
        direction, objective = key.rsplit("_", 1)
        route = resolve_route(direction=direction, objective=objective, is_current=False, manifest_path=args.route_manifest)
        compatibility = calibrator.get("compatibility", {})
        ok = (
            compatibility.get("route_id") == route.route_id
            and compatibility.get("candidate_universe_hash") == candidate_hashes[direction]
            and compatibility.get("model_bundle_version") == route.model_bundle_version
        )
        add_check(checks, f"calibrator:{key}:binding", ok, compatibility)

    conformal_payload = json.loads(
        DEFAULT_CONFORMAL_CALIBRATORS.read_text(encoding="utf-8")
    )
    add_check(
        checks,
        "conformal:manifest",
        conformal_payload.get("manifest_version") == 1
        and conformal_payload.get("conformal_retrieval_version")
        == CONFORMAL_RETRIEVAL_VERSION,
        {
            "manifest_version": conformal_payload.get("manifest_version"),
            "conformal_retrieval_version": conformal_payload.get(
                "conformal_retrieval_version"
            ),
        },
    )
    conformal_calibrators = conformal_payload.get("calibrators", {})
    add_check(
        checks,
        "conformal:calibrator_count",
        len(conformal_calibrators) == 6,
        len(conformal_calibrators),
    )
    for key, calibrator in conformal_calibrators.items():
        direction, objective = key.rsplit("_", 1)
        route = resolve_route(
            direction=direction,
            objective=objective,
            is_current=False,
            manifest_path=args.route_manifest,
        )
        compatibility = calibrator.get("compatibility", {})
        binding_ok = (
            compatibility.get("route_id") == route.route_id
            and compatibility.get("candidate_universe_hash")
            == candidate_hashes[direction]
            and compatibility.get("model_bundle_version")
            == route.model_bundle_version
        )
        alpha_spec = calibrator.get("alphas", {}).get("0.10", {})
        global_spec = alpha_spec.get("global", {})
        validation = global_spec.get("validation", alpha_spec.get("validation", {}))
        validation_ok = bool(validation.get("coverage_passed", False))
        set_size = int(global_spec.get("production_set_size", 0) or 0)
        size_ok = 1 <= set_size <= len(
            protein_ids if direction == "reaction_to_enzyme" else reaction_ids
        )
        add_check(
            checks,
            f"conformal:{key}:binding",
            binding_ok,
            compatibility,
        )
        add_check(
            checks,
            f"conformal:{key}:alpha0.10",
            validation_ok and size_ok,
            {
                "validation": validation,
                "production_set_size": set_size,
            },
        )

    dual_kernel = ROOT / "results/terpene_production_models/marts_dual_kernel_e2r_top20"
    dual_proteins = set(pd.read_csv(dual_kernel / "protein_ids.csv", dtype=str)["protein_id"].astype(str))
    dual_reactions = set(pd.read_csv(dual_kernel / "reaction_ids.csv", dtype=str)["reaction_id"].astype(str))
    add_check(checks, "dual_kernel_protein_set", dual_proteins == set(protein_ids), len(dual_proteins))
    add_check(checks, "dual_kernel_reaction_set", dual_reactions == set(reaction_ids), len(dual_reactions))

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="terpene_health_") as temp:
            temp_path = Path(temp)
            commands = [
                [sys.executable, str(ROOT / "projects/active/terpene_screening/rank_open_world.py"), "rank-enzymes", "--reaction-id", "RHEA:54512", "--top-k", "3", "--output", str(temp_path / "r2e.csv")],
                [sys.executable, str(ROOT / "projects/active/terpene_screening/rank_open_world.py"), "rank-reactions", "--enzyme-id", "7S5L_A", "--top-k", "3", "--output", str(temp_path / "e2r.csv")],
            ]
            for index, command in enumerate(commands):
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
                add_check(checks, f"smoke:{index}", completed.returncode == 0, completed.stderr[-2000:])
                if completed.returncode == 0:
                    frame = pd.read_csv(temp_path / ("r2e.csv" if index == 0 else "e2r.csv"))
                    required = {
                        "route_id",
                        "registry_version",
                        "candidate_universe_hash",
                        "candidate_universe_size",
                        "evidence_passport_version",
                        "query_applicability_score",
                        "query_applicability_tier",
                        "candidate_evidence_score",
                        "candidate_evidence_tier",
                        "conformal_retrieval_version",
                        "conformal_status",
                        "conformal_binding_status",
                        "conformal_set_size",
                        "conformal_set_member",
                    }
                    add_check(
                        checks,
                        f"smoke:{index}:contract",
                        required.issubset(frame.columns),
                        list(frame.columns),
                    )
                    if required.issubset(frame.columns):
                        score_ok = frame["query_applicability_score"].between(0.0, 1.0).all()
                        candidate_ok = frame["candidate_evidence_score"].between(0.0, 1.0).all()
                        version_ok = frame["evidence_passport_version"].eq(
                            "terpene-candidate-evidence-passport-v1"
                        ).all()
                        expected_conformal = (
                            ("not_applicable_current_entity", "not_applicable")
                            if index == 0
                            else (
                                "validated_external_double_cold_transport",
                                "compatible",
                            )
                        )
                        conformal_ok = (
                            frame["conformal_retrieval_version"]
                            .eq(CONFORMAL_RETRIEVAL_VERSION)
                            .all()
                            and frame["conformal_status"]
                            .eq(expected_conformal[0])
                            .all()
                            and frame["conformal_binding_status"]
                            .eq(expected_conformal[1])
                            .all()
                            and (
                                index == 0
                                or pd.to_numeric(
                                    frame["conformal_set_size"], errors="coerce"
                                ).gt(0).all()
                            )
                        )
                        add_check(
                            checks,
                            f"smoke:{index}:evidence_passport",
                            score_ok and candidate_ok and version_ok and conformal_ok,
                            {
                                "query_score": float(frame.iloc[0]["query_applicability_score"]),
                                "query_tier": str(frame.iloc[0]["query_applicability_tier"]),
                                "candidate_score_min": float(frame["candidate_evidence_score"].min()),
                                "candidate_score_max": float(frame["candidate_evidence_score"].max()),
                                "conformal_status": str(frame.iloc[0]["conformal_status"]),
                                "conformal_binding_status": str(
                                    frame.iloc[0]["conformal_binding_status"]
                                ),
                            },
                        )

    failures = [check for check in checks if not check["ok"]]
    report = {
        "status": "healthy" if not failures else "failed",
        "route_version": payload["route_version"],
        "registry_version": (manifest or {}).get("registry_version", "legacy"),
        "candidate_hashes": candidate_hashes,
        "checks": checks,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
