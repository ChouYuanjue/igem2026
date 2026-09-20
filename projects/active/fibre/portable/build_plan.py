from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY=ROOT/"projects/active/fibre/release/coordinate_registry.yaml"


def build_plan(config_path: str | Path, registry_path: str | Path=DEFAULT_REGISTRY) -> dict:
    config_path=Path(config_path)
    config=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    registry=yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
    features=config.get("features") or {}
    protein=features.get("protein") or {}
    reaction=features.get("reaction") or {}
    precompute=config.get("precompute") or {}
    geometry=config.get("geometry") or {}

    fresh=[]
    provided=[]
    for factor,section in (("protein",protein),("reaction",reaction)):
        mode=str(section.get("mode") or "")
        if mode=="precomputed":
            provided.append({
                "factor":factor,"coordinate":"global","kind":"feature",
                "path":str(section.get("path") or ""),
            })
        else:
            fresh.append({
                "factor":factor,"coordinate":"global","kind":"feature",
                "builder_mode":mode,
            })
        for item in section.get("views") or []:
            provided.append({
                "factor":factor,"coordinate":str(item["name"]),"kind":"feature",
                "path":str(item["path"]),
            })
        for item in section.get("distance_views") or []:
            provided.append({
                "factor":factor,"coordinate":str(item["name"]),"kind":"distance",
                "distance":str(item["distance"]),"available":str(item["available"]),
            })

    rc=(precompute.get("reaction_center") or {})
    if bool(rc.get("enabled",False)):
        fresh.extend([
            {
                "factor":"reaction","coordinate":"center_wasserstein","kind":"distance",
                "builder":"projects.active.fibre.portable.reaction_center",
                "mapped_reactions":str(rc.get("mapped_reactions") or ""),
            },
            {
                "factor":"reaction","coordinate":"center_tokens","kind":"distance",
                "builder":"projects.active.fibre.portable.reaction_center",
                "mapped_reactions":str(rc.get("mapped_reactions") or ""),
            },
        ])

    statuses={}
    for item in (registry.get("coordinates") or {}).values():
        status=str(item.get("portability") or "unspecified")
        statuses[status]=statuses.get(status,0)+1

    optional_declared=bool(
        (protein.get("views") or []) or (reaction.get("views") or [])
        or (protein.get("distance_views") or []) or (reaction.get("distance_views") or [])
        or bool(rc.get("enabled",False))
    )
    backend=str(geometry.get("backend") or "auto")
    warnings=[]
    if backend=="faiss_hnsw" and optional_declared:
        warnings.append(
            "faiss_hnsw is supported only for one fully observed feature view; "
            "use exact mixed geometry or provide a separately built sparse affinity."
        )
    if geometry.get("protein_affinity") and (
        (protein.get("views") or []) or (protein.get("distance_views") or [])
    ):
        warnings.append(
            "protein_affinity already defines the protein factor; separate protein optional views "
            "must be incorporated by that affinity builder rather than declared again."
        )
    if geometry.get("reaction_affinity") and (
        (reaction.get("views") or []) or (reaction.get("distance_views") or [])
        or bool(rc.get("enabled",False))
    ):
        warnings.append(
            "reaction_affinity already defines the reaction factor; separate reaction optional views "
            "must be incorporated by that affinity builder rather than declared again."
        )

    return {
        "schema":"fibre-portable-build-plan-v1",
        "config":str(config_path),
        "output_dir":str(config.get("output_dir") or ""),
        "dataset":dict(config.get("dataset") or {}),
        "fresh_recompute":fresh,
        "provided_coordinates":provided,
        "geometry":{
            "backend":backend,
            "graph_k":int(geometry.get("graph_k") or 0),
            "block_size":int(geometry.get("block_size") or 1024),
            "dense_limit":int(geometry.get("dense_limit") or 5000),
            "precomputed_protein_affinity":geometry.get("protein_affinity"),
            "precomputed_reaction_affinity":geometry.get("reaction_affinity"),
        },
        "registry_status_counts":statuses,
        "warnings":warnings,
        "important_semantics":{
            "factor_geometry_labels_used":False,
            "unknown_pairs_are_negative":False,
            "runtime_few_shot_promoted_automatically":False,
            "missing_optional_coordinate":"missing_not_negative",
        },
    }


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Explain what a portable FIBRE config will recompute versus reuse."
    )
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--registry",type=Path,default=DEFAULT_REGISTRY)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    plan=build_plan(args.config,args.registry)
    text=json.dumps(plan,indent=2,ensure_ascii=False)+"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    print(text,end="")


if __name__=="__main__":
    main()
