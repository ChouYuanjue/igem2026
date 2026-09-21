from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[2]
PROFILE_DIR=ROOT/"projects/active/fibre/release/profiles"
OUT_DIR=ROOT/"projects/active/fibre/release/manifests"
MODEL_ASSETS=ROOT/"reproducibility/bime_rank/model_assets.json"
RESEARCH_RELEASE=ROOT/"reproducibility/research_release_manifest.json"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda:fh.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict:
    rel=str(path.relative_to(ROOT))
    return {
        "path":rel,
        "bytes":int(path.stat().st_size),
        "sha256":sha256_file(path),
    }


def files_under(rel: str) -> list[Path]:
    root=ROOT/rel
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return [
        p for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".snakemake" not in p.parts
        and not p.name.endswith((".pyc",".pyo"))
    ]


def load_profiles() -> dict[str,dict]:
    out={}
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        x=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out[str(x["package_id"])]=x
    return out


def collect_source(profile: dict) -> list[dict]:
    source=profile.get("source") or {}
    roots=[]
    for key in ("shared_roots","workflow_roots","test_roots","roots"):
        roots.extend(map(str,source.get(key) or []))
    roots.extend(map(str,source.get("test_files") or []))
    roots.extend(map(str,source.get("required_paths") or []))
    excludes={str(x) for x in source.get("package_exclude_paths") or []}
    paths={}
    for rel in roots:
        for path in files_under(rel):
            r=str(path.relative_to(ROOT))
            if r in excludes:
                continue
            paths[r]=path
    return [record(paths[k]) for k in sorted(paths)]


def claim_records(profile: dict) -> list[dict]:
    claim=profile.get("claim_policy") or {}
    paths=list(claim.get("claim_authorities") or [])
    paths+=list(claim.get("supplemental_negative_evidence") or [])
    paths+=list(claim.get("supplemental_audit_authorities") or [])
    return [record(ROOT/str(x)) for x in paths]


def reproduction_assets() -> dict:
    model=json.loads(MODEL_ASSETS.read_text(encoding="utf-8"))
    research=json.loads(RESEARCH_RELEASE.read_text(encoding="utf-8"))
    project=[]
    for row in model.get("project_owned_assets") or []:
        p=ROOT/str(row["path"])
        project.append({
            **record(p),
            "role":str(row.get("role") or ""),
            "bundles":list(row.get("bundles") or []),
        })
    return {
        "project_owned_weights":project,
        "external_model_restore_contracts":list(model.get("external_model_assets") or []),
        "external_data_restore_contracts":list(model.get("external_data_assets") or []),
        "research_release_authority":record(RESEARCH_RELEASE),
        "direct_git_asset_count":int(research.get("direct_git_asset_count",0)),
        "direct_git_bytes":int(research.get("direct_git_bytes",0)),
        "rebuildable_asset_count":len(research.get("rebuildable_assets") or []),
        "external_asset_count":len(research.get("external_assets") or []),
    }


def application_assets(profile: dict) -> dict:
    assets=profile.get("assets") or {}
    info=[]
    for rel in assets.get("current_information_roots") or []:
        root=ROOT/str(rel)
        if not root.exists():
            raise FileNotFoundError(root)
        files=files_under(str(rel))
        info.append({
            "root":str(rel),
            "file_count":len(files),
            "bytes":sum(int(x.stat().st_size) for x in files),
            "root_manifest_sha256":hashlib.sha256(
                "\n".join(
                    f"{x.relative_to(ROOT)}\t{sha256_file(x)}"
                    for x in sorted(files)
                ).encode()
            ).hexdigest(),
        })
    specialist=[]
    for rel in assets.get("tps_specialist_source_weights") or []:
        files=files_under(str(rel))
        specialist.append({
            "root":str(rel),
            "file_count":len(files),
            "bytes":sum(int(x.stat().st_size) for x in files),
            "root_manifest_sha256":hashlib.sha256(
                "\n".join(
                    f"{x.relative_to(ROOT)}\t{sha256_file(x)}"
                    for x in sorted(files)
                ).encode()
            ).hexdigest(),
        })
    return {
        "current_information_roots":info,
        "tps_specialist_source_roots":specialist,
        "generated_application_assets":list(assets.get("generated_application_assets") or []),
        "rebuild_command":(
            "PYTHONPATH=. python -m projects.active.fibre.application.build_full_data "
            "--output-dir results/fibre_application/full_data"
        ),
    }


def build_one(pid: str, profile: dict, profiles: dict[str,dict]) -> dict:
    payload={
        "schema":"fibre-resolved-release-manifest-v1",
        "package_id":pid,
        "display_name":profile["display_name"],
        "release_role":profile["release_role"],
        "depends_on":list(profile.get("depends_on") or []),
        "profile_sha256":sha256_file(
            PROFILE_DIR/(
                "method.yaml" if pid=="fibre-method"
                else "reproduction.yaml" if pid=="fibre-reproduction"
                else "application.yaml"
            )
        ),
        "source_files":collect_source(profile),
        "claim_policy":profile.get("claim_policy") or {},
        "data_policy":profile.get("data_policy") or {},
        "learning_policy":profile.get("learning_policy") or {},
        "network_policy":profile.get("network_policy") or {},
        "isolation":profile.get("isolation") or {},
    }
    payload["source_file_count"]=len(payload["source_files"])
    payload["source_bytes"]=sum(int(x["bytes"]) for x in payload["source_files"])
    if pid=="fibre-reproduction":
        payload["claim_authorities"]=claim_records(profile)
        payload["assets"]=reproduction_assets()
    elif pid=="starase-application":
        payload["assets"]=application_assets(profile)
    else:
        payload["assets"]={
            "project_owned_weights_included":False,
            "project_database_assets_included":False,
            "tiny_cleanroom_fixture_included":True,
        }
    return payload


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Resolve the three isolated FIBRE/Starase release profiles to hashed manifests."
    )
    ap.add_argument("--output-dir",type=Path,default=OUT_DIR)
    args=ap.parse_args()
    profiles=load_profiles()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index={
        "schema":"fibre-release-manifest-index-v1",
        "functional_packages":[],
        "asset_catalog":"projects/active/fibre/release/asset_catalog.yaml",
        "asset_catalog_is_functional_package":False,
    }
    for pid in ("fibre-method","fibre-reproduction","starase-application"):
        payload=build_one(pid,profiles[pid],profiles)
        out=args.output_dir/f"{pid}.json"
        out.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
        index["functional_packages"].append({
            "package_id":pid,
            "manifest":str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
            "sha256":sha256_file(out),
            "source_file_count":payload["source_file_count"],
            "source_bytes":payload["source_bytes"],
        })
    idx=args.output_dir/"index.json"
    idx.write_text(json.dumps(index,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(index,indent=2))


if __name__=="__main__":
    main()
