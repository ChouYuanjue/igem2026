from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROUTE_MANIFEST = ROOT / "configs/production_routes/terpene_v1.yaml"

@dataclass(frozen=True)
class RouteProvenance:
    route_id: str
    route_version: str
    candidate_universe_version: str
    model_bundle_version: str
    deployment: Path
    secondary_deployment: Path | None = None
    auxiliary_deployment: Path | None = None
    retrieval: str = "direct"
    settings: dict[str, Any] | None = None

@lru_cache(maxsize=4)
def load_route_manifest(path: str = str(DEFAULT_ROUTE_MANIFEST)) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("manifest_version", 0)) != 1:
        raise ValueError(f"Unsupported terpene route manifest: {manifest_path}")
    required = {"route_version", "candidate_universe_version", "model_bundle_version", "deployments", "routes"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Route manifest is missing fields: {sorted(missing)}")
    return payload

def _deployment(payload: dict[str, Any], key: str | None) -> Path | None:
    if not key:
        return None
    relative = payload["deployments"].get(key)
    if not relative:
        raise KeyError(f"Unknown deployment key in route manifest: {key}")
    return (ROOT / str(relative)).resolve()

def resolve_route(*, direction: str, objective: str, is_current: bool,
                  has_seed: bool = False, manual_override: bool = False,
                  temporary_candidate_extension: bool = False,
                  masked_discovery: bool = False,
                  enzyme_taxonomy_scope: str = "all",
                  manifest_path: Path = DEFAULT_ROUTE_MANIFEST) -> RouteProvenance:
    payload = load_route_manifest(str(manifest_path.resolve()))
    entity_scope = "current" if is_current else "external"
    try:
        spec = dict(payload["routes"][direction][entity_scope][objective])
    except KeyError as exc:
        raise ValueError(f"No route for direction={direction}, scope={entity_scope}, objective={objective}") from exc
    suffixes = []
    if has_seed: suffixes.append("fewshot")
    if masked_discovery: suffixes.append("masked")
    if temporary_candidate_extension: suffixes.append("temporary-universe")
    if enzyme_taxonomy_scope not in {"all", "eukaryote", "prokaryote"}:
        raise ValueError(f"Unsupported enzyme taxonomy scope: {enzyme_taxonomy_scope!r}")
    if enzyme_taxonomy_scope != "all":
        if direction != "reaction_to_enzyme":
            raise ValueError("Enzyme taxonomy scope is only defined for reaction_to_enzyme candidate retrieval")
        suffixes.append(f"{enzyme_taxonomy_scope}-only")
    if manual_override: suffixes.append("manual")
    route_id = str(spec["route_id"])
    if suffixes:
        route_id = f"{route_id}+{'+'.join(suffixes)}"
    settings = {k: v for k, v in spec.items() if k not in {
        "route_id", "deployment", "secondary_deployment", "auxiliary_deployment", "retrieval"
    }}
    return RouteProvenance(
        route_id=route_id,
        route_version=str(payload["route_version"]),
        candidate_universe_version=str(payload["candidate_universe_version"]),
        model_bundle_version=str(payload["model_bundle_version"]),
        deployment=_deployment(payload, str(spec["deployment"])) or ROOT,
        secondary_deployment=_deployment(payload, spec.get("secondary_deployment")),
        auxiliary_deployment=_deployment(payload, spec.get("auxiliary_deployment")),
        retrieval=str(spec.get("retrieval", "direct")),
        settings=settings,
    )
