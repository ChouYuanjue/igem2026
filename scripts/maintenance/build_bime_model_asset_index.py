from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROUTE = ROOT / "configs/production_routes/terpene_v1.yaml"
RELEASE_MANIFEST = ROOT / "reproducibility/research_release_manifest.json"
OUT = ROOT / "reproducibility/bime_rank/model_assets.json"

LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
SUPPORT_SUFFIXES = {".npz"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked() -> set[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {x.decode() for x in out.split(b"\0") if x}


def path_values(route: dict) -> set[str]:
    deployments = {str(k): str(v) for k, v in route.get("deployments", {}).items()}
    paths = set(deployments.values())
    path_keys = {
        "ranker_bundle",
        "ineligible_fallback_deployment",
    }
    alias_keys = {"deployment", "secondary_deployment", "auxiliary_deployment"}

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str):
                    if key in path_keys and "/" in child:
                        paths.add(child)
                    elif key in alias_keys:
                        resolved = deployments.get(child, child)
                        if "/" in resolved:
                            paths.add(resolved)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(route.get("routes", {}))
    return paths


def classify(rel: str) -> str | None:
    path = Path(rel)
    if path.suffix.lower() in LEARNED_SUFFIXES:
        return "learned_parameter"
    if path.name == "ranker.json" or ("ranker" in path.name.lower() and path.suffix.lower() == ".json"):
        return "learned_parameter"
    if path.suffix.lower() in SUPPORT_SUFFIXES:
        return "runtime_support_array"
    return None


def main() -> None:
    route = yaml.safe_load(ROUTE.read_text(encoding="utf-8"))
    tracked_paths = tracked()
    bundles = sorted(path_values(route))
    project_assets: list[dict[str, object]] = []
    bundle_counts: dict[str, int] = {}
    for bundle in bundles:
        prefix = bundle.rstrip("/") + "/"
        members = []
        for rel in sorted(tracked_paths):
            if not rel.startswith(prefix):
                continue
            role = classify(rel)
            if role is None:
                continue
            path = ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(rel)
            record = {
                "path": rel,
                "bundle": bundle,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            project_assets.append(record)
            members.append(record)
        # Every route model bundle must carry either learned parameters or an explicit
        # runtime support array; otherwise route->model inventory drift is ambiguous.
        if not members:
            raise RuntimeError(f"production route bundle has no tracked model/support asset: {bundle}")
        bundle_counts[bundle] = len(members)

    # Deduplicate assets that are reachable through multiple route aliases while retaining
    # a complete bundle membership list.
    merged: dict[str, dict[str, object]] = {}
    for record in project_assets:
        rel = str(record["path"])
        if rel not in merged:
            merged[rel] = {
                "path": rel,
                "role": record["role"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "bundles": [],
            }
        bundles_for_asset = merged[rel]["bundles"]
        assert isinstance(bundles_for_asset, list)
        if record["bundle"] not in bundles_for_asset:
            bundles_for_asset.append(record["bundle"])

    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    external_models = []
    external_data = []
    for asset in release.get("external_assets", []):
        target = str(asset["target"])
        record = dict(asset)
        if Path(target).suffix.lower() in LEARNED_SUFFIXES:
            external_models.append(record)
        else:
            external_data.append(record)

    project = sorted(merged.values(), key=lambda x: str(x["path"]))
    learned = [x for x in project if x["role"] == "learned_parameter"]
    support = [x for x in project if x["role"] == "runtime_support_array"]
    payload = {
        "schema_version": 1,
        "authority": "configs/production_routes/terpene_v1.yaml",
        "route_version": route.get("route_version"),
        "route_sha256": sha256(ROUTE),
        "policy": {
            "project_owned": "All current production-route learned parameters and compact runtime support arrays are Git-tracked and hash-locked.",
            "external": "Large third-party foundation checkpoints are not vendored; exact restore/version/hash contracts are inherited from reproducibility/research_release_manifest.json.",
            "historical": "Protected historical/dev-fold checkpoints are not current model assets merely because they remain on the development server.",
        },
        "model_bundles": bundles,
        "bundle_asset_counts": bundle_counts,
        "project_owned_assets": project,
        "external_model_assets": sorted(external_models, key=lambda x: str(x["target"])),
        "external_data_assets": sorted(external_data, key=lambda x: str(x["target"])),
        "counts": {
            "production_model_bundles": len(bundles),
            "project_owned_assets": len(project),
            "learned_parameters": len(learned),
            "runtime_support_arrays": len(support),
            "project_owned_bytes": sum(int(x["bytes"]) for x in project),
            "external_model_assets": len(external_models),
            "external_data_assets": len(external_data),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
