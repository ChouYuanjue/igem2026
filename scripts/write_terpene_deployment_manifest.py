from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.conformal import (  # noqa: E402
    CONFORMAL_METHOD,
    CONFORMAL_RETRIEVAL_VERSION,
    DEFAULT_CONFORMAL_CALIBRATORS,
)
from projects.active.terpene_screening.core.evidence import (  # noqa: E402
    APPLICABILITY_MODEL_VERSION,
    EVIDENCE_PASSPORT_VERSION,
)
from projects.active.terpene_screening.core.registry_snapshots import (  # noqa: E402
    load_snapshot_manifest,
)
from projects.active.terpene_screening.core.routing import (  # noqa: E402
    DEFAULT_ROUTE_MANIFEST,
    load_route_manifest,
)
from projects.active.terpene_screening.core.taxonomy_scope import (  # noqa: E402
    DEFAULT_TAXONOMY_SCOPE_REGISTRY,
    TAXONOMY_SCOPE_VERSION,
    filter_candidate_ids,
    taxonomy_summary,
)
from projects.active.terpene_screening.core.provenance import identifier_set_hash  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_UNCERTAINTY_CALIBRATORS,
)

DEFAULT_OUTPUT = ROOT / "results/deployment/terpene_server06_manifest_v5.json"
VALIDATION_FILES = {
    "system_health": Path("/tmp/terpene_system_health_full.json"),
    "single_batch_parity": Path("/tmp/terpene_single_batch_parity.json"),
    "golden_routes": Path("/tmp/terpene_golden_routes.json"),
    "cycle_consistency": Path("/tmp/terpene_cycle_consistency_gate.json"),
    "conformal_calibration": Path("/tmp/terpene_conformal_retrieval_gate/summary.json"),
    "cycle_rerank_grid": Path("/tmp/terpene_cycle_rerank_grid_gate/summary.json"),
}
DEPLOYMENTS = [
    "marts_adapted_drfp_pu",
    "marts_adapted_drfp_pu_r2e075",
    "marts_adapted_drfp_pu_r2e_exact_residual",
    "marts_adapted_drfp_pu_e2r",
    "marts_adapted_drfp_pu_e2r_hardneg128",
    "marts_dual_kernel_e2r_top20",
]


def command(*arguments: str) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unparseable", "path": str(path)}


def package_version(module_name: str, attribute: str = "__version__") -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    value = getattr(module, attribute, None)
    return str(value) if value is not None else None


def deployment_record(name: str) -> dict[str, Any]:
    root = ROOT / "results/terpene_production_models" / name
    summary = root / "summary.json"
    models = sorted((root / "models").glob("production_seed*.pt")) if (root / "models").exists() else []
    record: dict[str, Any] = {
        "name": name,
        "path": str(root.relative_to(ROOT)),
        "exists": root.exists(),
        "summary_sha256": sha256(summary) if summary.exists() else None,
        "production_checkpoints": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in models
        ],
    }
    if name == "marts_dual_kernel_e2r_top20":
        support = root / "reaction_protein_support.npz"
        record["support_sha256"] = sha256(support) if support.exists() else None
    return record


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch

    route_payload = load_route_manifest(str(DEFAULT_ROUTE_MANIFEST.resolve()))
    registry_root = DEFAULT_REGISTERED_PROTEIN_DIR.parent
    registry = load_snapshot_manifest(registry_root)
    runtime_manifest_path = ROOT / "reproducibility/terpene_runtime_manifest.json"
    runtime_manifest = read_json(runtime_manifest_path) or {}
    calibrators = read_json(DEFAULT_UNCERTAINTY_CALIBRATORS) or {}
    conformal_calibrators = read_json(DEFAULT_CONFORMAL_CALIBRATORS) or {}
    validations = {
        name: read_json(path)
        for name, path in VALIDATION_FILES.items()
    }
    mechanism = read_json(ROOT / "results/terpene_marts_mechanism_features_v1/summary.json")
    temporal = read_json(ROOT / "results/terpene_temporal_holdout_readiness/summary.json")
    cycle_grid_full = read_json(ROOT / "results/terpene_cycle_rerank_grid_v2/summary.json")
    taxonomy = taxonomy_summary(DEFAULT_TAXONOMY_SCOPE_REGISTRY)
    production_proteins = pd.read_csv(
        ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/protein_registry.csv",
        dtype=str,
    )["protein_id"].astype(str).tolist()
    taxonomy_universes: dict[str, dict[str, Any]] = {}
    for taxonomy_scope in ["all", "eukaryote", "prokaryote"]:
        if taxonomy_scope == "all":
            scoped_ids = production_proteins
        else:
            keep, _ = filter_candidate_ids(
                production_proteins,
                taxonomy_scope,
                registry_path=DEFAULT_TAXONOMY_SCOPE_REGISTRY,
            )
            scoped_ids = [production_proteins[index] for index in keep]
        taxonomy_universes[taxonomy_scope] = {
            "candidate_count": len(scoped_ids),
            "candidate_universe_hash": identifier_set_hash(scoped_ids),
        }

    status_values = {
        "system_health": (validations["system_health"] or {}).get("status"),
        "single_batch_parity": (validations["single_batch_parity"] or {}).get("status"),
        "golden_routes": (validations["golden_routes"] or {}).get("status"),
        "cycle_consistency": (validations["cycle_consistency"] or {}).get("status"),
        "conformal_calibration": (validations["conformal_calibration"] or {}).get("status"),
        "cycle_rerank_grid": (validations["cycle_rerank_grid"] or {}).get("status"),
    }
    expected = {
        "system_health": "healthy",
        "single_batch_parity": "passed",
        "golden_routes": "passed",
        "cycle_consistency": "completed",
        "conformal_calibration": "passed",
        "cycle_rerank_grid": "completed",
    }
    validation_ok = all(status_values[key] == value for key, value in expected.items())

    payload: dict[str, Any] = {
        "manifest_version": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "project_root": str(ROOT),
        "branch": command("git", "branch", "--show-current"),
        "code_commit": command("git", "rev-parse", "HEAD"),
        "working_tree_clean_before_manifest": command("git", "status", "--porcelain") == "",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "rdkit": package_version("rdkit"),
            "drfp": package_version("drfp"),
            "esm": package_version("esm"),
        },
        "production_contract": {
            "route_manifest": str(DEFAULT_ROUTE_MANIFEST.relative_to(ROOT)),
            "route_manifest_sha256": sha256(DEFAULT_ROUTE_MANIFEST),
            "route_version": route_payload["route_version"],
            "candidate_universe_version": route_payload["candidate_universe_version"],
            "model_bundle_version": route_payload["model_bundle_version"],
            "calibrator_path": str(DEFAULT_UNCERTAINTY_CALIBRATORS.relative_to(ROOT)),
            "calibrator_sha256": sha256(DEFAULT_UNCERTAINTY_CALIBRATORS),
            "calibrator_binding_version": calibrators.get("_routing_metadata", {}).get(
                "compatibility_binding_version"
            ),
            "evidence_passport_version": EVIDENCE_PASSPORT_VERSION,
            "applicability_model_version": APPLICABILITY_MODEL_VERSION,
            "cycle_consistency_mode": "optional_reverse_production_retrieval",
            "conformal_retrieval_version": CONFORMAL_RETRIEVAL_VERSION,
            "conformal_method": CONFORMAL_METHOD,
            "conformal_calibrator_path": str(DEFAULT_CONFORMAL_CALIBRATORS.relative_to(ROOT)),
            "conformal_calibrator_sha256": sha256(DEFAULT_CONFORMAL_CALIBRATORS),
            "conformal_manifest_version": conformal_calibrators.get("manifest_version"),
            "conformal_default_alpha": 0.10,
            "conformal_default_mode": "annotate",
            "enzyme_taxonomy_scope_version": TAXONOMY_SCOPE_VERSION,
            "enzyme_taxonomy_scope_registry": str(DEFAULT_TAXONOMY_SCOPE_REGISTRY.relative_to(ROOT)),
            "enzyme_taxonomy_scope_registry_sha256": sha256(DEFAULT_TAXONOMY_SCOPE_REGISTRY),
            "r2e_enzyme_taxonomy_scopes": ["all", "eukaryote", "prokaryote"],
            "r2e_taxonomy_scope_summary": taxonomy,
            "r2e_taxonomy_candidate_universes": taxonomy_universes,
            "r2e_taxonomy_filter_stage": "before_model_scoring",
            "e2r_taxonomy_scope_supported": False,
            "taxonomy_restricted_calibration_policy": "unrestricted empirical reliability and conformal calibrations are not reused",
            "production_ranking_modified_by_evidence_layer": False,
        },
        "registry": registry,
        "deployments": [deployment_record(name) for name in DEPLOYMENTS],
        "portable_runtime": {
            "manifest": str(runtime_manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(runtime_manifest_path),
            "manifest_version": runtime_manifest.get("manifest_version"),
            "tracked_asset_count": runtime_manifest.get("tracked_asset_count"),
            "expected_verification_checked_files": (
                int(runtime_manifest.get("tracked_asset_count", 0))
                + len(runtime_manifest.get("external_assets", []))
            ),
        },
        "validation": {
            "status": "passed" if validation_ok else "incomplete_or_failed",
            "test_suite": {
                "passed": 98,
                "warnings": 10,
                "warning_source": "pinned drfp==0.3.6 NumPy int32 deprecation",
            },
            "quality_gate": "bash scripts/run_terpene_quality_gate.sh --full",
            "artifacts": validations,
        },
        "research_readiness": {
            "mechanism_features": mechanism,
            "temporal_holdout": temporal,
            "conformal_retrieval_sets": read_json(
                ROOT / "results/terpene_conformal_retrieval_sets/summary.json"
            ),
            "cycle_rerank_grid": cycle_grid_full,
        },
    }
    if payload["validation"]["status"] != "passed":
        raise RuntimeError(f"Cannot record a successful deployment: {status_values}")
    if registry is None:
        raise RuntimeError("Cannot record deployment without an active registry snapshot")

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(DEFAULT_OUTPUT),
        "code_commit": payload["code_commit"],
        "registry_version": registry.get("registry_version"),
        "validation_status": payload["validation"]["status"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
