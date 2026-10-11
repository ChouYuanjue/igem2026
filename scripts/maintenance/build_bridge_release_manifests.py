from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[2]
PROFILE_DIR=ROOT/"projects/active/bridge/release/profiles"
OUT_DIR=ROOT/"projects/active/bridge/release/manifests"
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


def _application_root_records(assets: dict, key: str) -> list[dict]:
    rows=[]
    for rel in assets.get(key) or []:
        root=ROOT/str(rel)
        if not root.exists():
            raise FileNotFoundError(root)
        files=files_under(str(rel))
        rows.append({
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
    return rows


def application_assets(profile: dict, *, source_only: bool = False) -> dict:
    assets=profile.get("assets") or {}
    if source_only:
        # A source checkout does not contain every runtime database/weight.
        # Keep the previously recorded scientific asset hashes, counts and
        # bytes as FROZEN claims, explicitly mark them unverified here.
        # Never compute a partial root digest from whatever the CI checkout
        # happens to contain; that would silently replace the real inventory.
        previous=OUT_DIR/"starase-application.json"
        if not previous.is_file():
            raise FileNotFoundError(
                "Source-only release requires the committed, previously "
                "materialized application manifest"
            )
        prior=json.loads(previous.read_text(encoding="utf-8"))
        if prior.get("package_id")!="starase-application":
            raise ValueError("Frozen application manifest package ID drift")
        stored=prior.get("assets") or {}

        def preserve(profile_key: str, group_key: str) -> list[dict]:
            expected=[str(x).rstrip("/") for x in assets.get(profile_key) or []]
            rows=stored.get(group_key) or []
            indexed={str(x.get("root","")):x for x in rows}
            if len(indexed)!=len(rows) or set(indexed)!=set(expected):
                raise ValueError(
                    "Frozen application asset roots are missing or altered: "
                    + group_key
                )
            result=[]
            for rel in expected:
                row=indexed[rel]
                digest=str(row.get("root_manifest_sha256") or "")
                if (len(digest)!=64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or int(row.get("file_count") or 0)<1
                    or int(row.get("bytes") or 0)<1):
                    raise ValueError(
                        "Incomplete frozen runtime root provenance: "+rel
                    )
                result.append({
                    "root":rel,
                    "file_count":int(row["file_count"]),
                    "bytes":int(row["bytes"]),
                    "root_manifest_sha256":digest,
                })
            return result

        return {
            "current_information_roots":preserve(
                "current_information_roots","current_information_roots"
            ),
            "tps_specialist_source_roots":preserve(
                "tps_specialist_source_weights","tps_specialist_source_roots"
            ),
            "runtime_asset_roots":preserve(
                "runtime_asset_roots","runtime_asset_roots"
            ),
            "generated_application_assets":list(
                assets.get("generated_application_assets") or []
            ),
            "rebuild_command":(
                "PYTHONPATH=. python -m projects.active.bridge.application.build_full_data "
                "--output-dir results/fibre_application/full_data"
            ),
            "asset_integrity_verification":"frozen_metadata_only_not_rematerialized",
        }
    return {
        "current_information_roots":_application_root_records(
            assets,"current_information_roots"
        ),
        "tps_specialist_source_roots":_application_root_records(
            assets,"tps_specialist_source_weights"
        ),
        "runtime_asset_roots":_application_root_records(
            assets,"runtime_asset_roots"
        ),
        "generated_application_assets":list(assets.get("generated_application_assets") or []),
        "rebuild_command":(
            "PYTHONPATH=. python -m projects.active.bridge.application.build_full_data "
            "--output-dir results/fibre_application/full_data"
        ),
    }


def build_one(
    pid: str, profile: dict, profiles: dict[str,dict], *,
    source_only: bool = False,
) -> dict:
    payload={
        "schema":"bridge-resolved-release-manifest-v1",
        "package_id":pid,
        "display_name":profile["display_name"],
        "release_role":profile["release_role"],
        "depends_on":list(profile.get("depends_on") or []),
        "profile_sha256":sha256_file(
            PROFILE_DIR/(
                "method.yaml" if pid=="bridge-method"
                else "reproduction.yaml" if pid=="bridge-reproduction"
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
    if pid=="bridge-reproduction":
        payload["claim_authorities"]=claim_records(profile)
        payload["assets"]=reproduction_assets()
    elif pid=="starase-application":
        payload["assets"]=application_assets(profile,source_only=source_only)
    else:
        payload["assets"]={
            "project_owned_weights_included":False,
            "project_database_assets_included":False,
            "tiny_cleanroom_fixture_included":True,
        }
    return payload


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Resolve the three isolated BRIDGE/Starase release profiles to hashed manifests."
    )
    ap.add_argument("--output-dir",type=Path,default=OUT_DIR)
    ap.add_argument(
        "--source-only",action="store_true",
        help="Preserve existing frozen runtime asset hashes when generating "
             "manifests in a public source checkout without private assets",
    )
    args=ap.parse_args()
    profiles=load_profiles()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index={
        "schema":"bridge-release-manifest-index-v1",
        "functional_packages":[],
        "asset_catalog":"projects/active/bridge/release/asset_catalog.yaml",
        "asset_catalog_is_functional_package":False,
    }
    for pid in ("bridge-method","bridge-reproduction","starase-application"):
        payload=build_one(pid,profiles[pid],profiles,source_only=args.source_only)
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
