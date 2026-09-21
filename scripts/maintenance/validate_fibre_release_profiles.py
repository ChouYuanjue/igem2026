from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[2]
PROFILE_DIR=ROOT/"projects/active/fibre/release/profiles"
CATALOG=ROOT/"projects/active/fibre/release/asset_catalog.yaml"
MODEL_ASSETS=ROOT/"reproducibility/bime_rank/model_assets.json"
SUBMISSION_MANIFEST=ROOT/"IGEM_SUBMISSION_MANIFEST.json"
EXPECTED_IDS={"fibre-method","fibre-reproduction","starase-application"}


def source_only_roots() -> tuple[str,...]:
    if SUBMISSION_MANIFEST.is_file():
        payload=json.loads(SUBMISSION_MANIFEST.read_text(encoding="utf-8"))
        roots=tuple(str(x) for x in payload.get("excluded_scientific_asset_roots") or ())
        if roots:
            return roots
    return (
        "data/","results/","archive/","external/","external_models/",
        "external_repos/","external_runtime/",
    )


def externalized_in_source_only(rel: str) -> bool:
    return str(rel).startswith(source_only_roots())


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda:fh.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def load_profiles() -> dict[str,dict]:
    out={}
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        payload=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("schema")!="fibre-release-profile-v1":
            raise ValueError(f"{path}: wrong schema")
        pid=str(payload.get("package_id") or "")
        if not pid or pid in out:
            raise ValueError(f"{path}: missing/duplicate package_id {pid!r}")
        payload["_path"]=str(path.relative_to(ROOT))
        out[pid]=payload
    if set(out)!=EXPECTED_IDS:
        raise ValueError(
            f"release profile IDs drifted: expected={sorted(EXPECTED_IDS)} got={sorted(out)}"
        )
    return out


def dependency_order(profiles: dict[str,dict]) -> list[str]:
    order=[]; visiting=set(); done=set()
    def visit(pid: str) -> None:
        if pid in done:
            return
        if pid in visiting:
            raise ValueError(f"release profile dependency cycle at {pid}")
        visiting.add(pid)
        for dep in profiles[pid].get("depends_on") or []:
            if dep not in profiles:
                raise ValueError(f"{pid}: unknown dependency {dep}")
            visit(str(dep))
        visiting.remove(pid); done.add(pid); order.append(pid)
    for pid in sorted(profiles):
        visit(pid)
    return order


def python_files(root_rel: str) -> list[Path]:
    root=ROOT/root_rel
    if root.is_file() and root.suffix==".py":
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return [
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def imported_modules(path: Path) -> set[str]:
    try:
        tree=ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {path.relative_to(ROOT)}: {exc}") from exc
    out=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            out.update(str(x.name) for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
            out.add(str(node.module))
    return out


def scan_import_boundary(profile: dict) -> list[dict[str,str]]:
    source=profile.get("source") or {}
    roots=source.get("import_scan_roots") or []
    excludes={str(x) for x in source.get("import_scan_exclude_paths") or []}
    forbidden=[str(x) for x in source.get("forbidden_import_prefixes") or []]
    violations=[]
    seen=set()
    for root in roots:
        for path in python_files(str(root)):
            rel=str(path.relative_to(ROOT))
            if rel in excludes or rel in seen:
                continue
            seen.add(rel)
            for module in imported_modules(path):
                for prefix in forbidden:
                    if module==prefix or module.startswith(prefix+"."):
                        violations.append({
                            "profile":str(profile["package_id"]),
                            "source":rel,
                            "import":module,
                            "forbidden_prefix":prefix,
                        })
    return violations


def validate_required_paths(profile: dict, *, source_only: bool) -> dict:
    source=profile.get("source") or {}
    for rel in source.get("required_paths") or []:
        if not (ROOT/str(rel)).exists():
            raise FileNotFoundError(f"{profile['package_id']}: required path missing: {rel}")
    materialized_claims=0
    externalized_claims=[]
    for rel in ((profile.get("claim_policy") or {}).get("claim_authorities") or []):
        path=ROOT/str(rel)
        if path.is_file():
            materialized_claims+=1
        elif source_only and externalized_in_source_only(str(rel)):
            externalized_claims.append(str(rel))
        else:
            raise FileNotFoundError(f"{profile['package_id']}: claim authority missing: {rel}")
    materialized_negative=0
    externalized_negative=[]
    for rel in ((profile.get("claim_policy") or {}).get("supplemental_negative_evidence") or []):
        path=ROOT/str(rel)
        if path.is_file():
            materialized_negative+=1
        elif source_only and externalized_in_source_only(str(rel)):
            externalized_negative.append(str(rel))
        else:
            raise FileNotFoundError(f"{profile['package_id']}: negative evidence missing: {rel}")
    materialized_audits=0
    externalized_audits=[]
    for rel in ((profile.get("claim_policy") or {}).get("supplemental_audit_authorities") or []):
        path=ROOT/str(rel)
        if path.is_file():
            materialized_audits+=1
        elif source_only and externalized_in_source_only(str(rel)):
            externalized_audits.append(str(rel))
        else:
            raise FileNotFoundError(f"{profile['package_id']}: audit authority missing: {rel}")
    return {
        "materialized_claim_authorities":materialized_claims,
        "externalized_claim_authorities":externalized_claims,
        "materialized_supplemental_negative_evidence":materialized_negative,
        "externalized_supplemental_negative_evidence":externalized_negative,
        "materialized_supplemental_audit_authorities":materialized_audits,
        "externalized_supplemental_audit_authorities":externalized_audits,
    }


def validate_reproduction_weights(profile: dict, *, source_only: bool) -> dict:
    if not bool((profile.get("assets") or {}).get("require_every_project_owned_asset_present")):
        raise ValueError("fibre-reproduction must require every project-owned model asset")
    payload=json.loads(MODEL_ASSETS.read_text(encoding="utf-8"))
    assets=list(payload.get("project_owned_assets") or [])
    if not assets:
        raise ValueError("model_assets.json contains no project-owned assets")
    missing=[]
    drift=[]
    materialized=0
    for row in assets:
        rel=str(row.get("path") or "")
        expected=str(row.get("sha256") or "")
        expected_bytes=int(row.get("bytes") or 0)
        if not rel or len(expected)!=64 or expected_bytes<=0:
            raise ValueError(f"incomplete project-owned asset record: {row}")
        path=ROOT/rel
        if not path.is_file():
            if source_only and externalized_in_source_only(rel):
                missing.append(rel)
                continue
            missing.append(rel)
            continue
        materialized+=1
        if path.stat().st_size!=expected_bytes:
            drift.append(rel)
            continue
        if sha256_file(path)!=expected:
            drift.append(rel)
    if missing and not source_only:
        raise FileNotFoundError(
            f"reproduction project-owned model weights missing: {missing[:10]}"
        )
    if source_only:
        illegal=[rel for rel in missing if not externalized_in_source_only(rel)]
        if illegal:
            raise FileNotFoundError(
                f"source-only reproduction asset missing outside excluded roots: {illegal[:10]}"
            )
    if drift:
        raise ValueError(
            f"reproduction project-owned model weights hash/size drift: {drift[:10]}"
        )
    counts=payload.get("counts") or {}
    if int(counts.get("project_owned_assets",-1))!=len(assets):
        raise ValueError("model asset count metadata drift")
    return {
        "project_owned_assets":len(assets),
        "materialized_project_owned_assets":materialized,
        "externalized_project_owned_assets":len(missing),
        "project_owned_asset_materialization_complete":not missing,
        "learned_parameters":int(counts.get("learned_parameters",0)),
        "project_owned_bytes":int(counts.get("project_owned_bytes",0)),
        "external_model_assets":len(payload.get("external_model_assets") or []),
    }


def validate_catalog(profiles: dict[str,dict], *, source_only: bool) -> dict:
    catalog=yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}
    if catalog.get("schema")!="fibre-release-asset-catalog-v1":
        raise ValueError("asset catalog schema drift")
    groups=catalog.get("groups") or {}
    known=set(profiles)
    externalized=[]
    for name,row in groups.items():
        consumers=set(map(str,row.get("consumers") or []))
        unknown=consumers-known
        if unknown:
            raise ValueError(f"asset group {name} has unknown consumers: {sorted(unknown)}")
        prohibited=set(map(str,(row.get("prohibited_uses") or {}).keys()))
        conflict=consumers&prohibited
        if conflict:
            raise ValueError(f"asset group {name} both consumes and prohibits: {sorted(conflict)}")
        for rel in row.get("paths") or []:
            rel=str(rel)
            # Application-derived outputs may intentionally not exist until built.
            if rel.startswith("results/fibre_application/"):
                continue
            if not (ROOT/rel).exists():
                if source_only and externalized_in_source_only(rel):
                    # A catalog-only failed/excluded experiment is lineage, not a
                    # release asset that must later be restored for a consumer.
                    if consumers:
                        externalized.append(rel)
                    continue
                raise FileNotFoundError(f"asset catalog path missing: {rel}")
    failed=groups.get("failed_tps_foundation_exploration") or {}
    if failed.get("consumers") not in ([],None):
        raise ValueError("failed TPS foundation exploration must not enter a release profile")
    return {
        "asset_groups":len(groups),
        "externalized_catalog_paths":sorted(set(externalized)),
    }


def validate_semantics(profiles: dict[str,dict]) -> None:
    method=profiles["fibre-method"]
    repro=profiles["fibre-reproduction"]
    app=profiles["starase-application"]
    if (method.get("learning_policy") or {}).get("project_trained_weights")!="forbidden":
        raise ValueError("method package must forbid project-trained weights")
    if (repro.get("claim_policy") or {}).get("benchmark_metrics")!="required_and_frozen":
        raise ValueError("reproduction benchmark policy must remain frozen")
    if (app.get("claim_policy") or {}).get("benchmark_metrics")!="forbidden":
        raise ValueError("application package must not publish benchmark metrics")
    tps=((app.get("learning_policy") or {}).get("tps_adapted_coordinate") or {})
    if not bool(tps.get("enabled")) or bool(tps.get("use_legacy_cross_factor_score")):
        raise ValueError("application TPS coordinate must be enabled as a factor view, not legacy score")
    if "fibre-reproduction" not in set(method.get("isolation",{}).get("can_be_dependency_of") or []):
        raise ValueError("method package dependency contract drift")


def build_status(*, source_only: bool=False) -> dict:
    profiles=load_profiles()
    order=dependency_order(profiles)
    path_status={}
    for pid,profile in profiles.items():
        path_status[pid]=validate_required_paths(profile,source_only=source_only)
    violations=[]
    for profile in profiles.values():
        violations.extend(scan_import_boundary(profile))
    if violations:
        preview=json.dumps(violations[:20],indent=2)
        raise ValueError(f"release import boundary violations:\n{preview}")
    weights=validate_reproduction_weights(
        profiles["fibre-reproduction"],source_only=source_only
    )
    catalog=validate_catalog(profiles,source_only=source_only)
    validate_semantics(profiles)
    return {
        "schema":"fibre-release-profile-validation-v1",
        "validation_mode":(
            "source_only_contract" if source_only else "materialized_workspace"
        ),
        "profiles":order,
        "profile_files":{
            pid:str(profiles[pid]["_path"]) for pid in order
        },
        "path_materialization":path_status,
        "import_boundary_violations":0,
        "reproduction_weights":weights,
        **catalog,
        "functional_package_count":3,
        "asset_catalog_is_functional_package":False,
        "full_reproduction_assets_materialized":bool(
            weights.get("project_owned_asset_materialization_complete")
            and not path_status["fibre-reproduction"].get(
                "externalized_claim_authorities"
            )
            and not path_status["fibre-reproduction"].get(
                "externalized_supplemental_negative_evidence"
            )
            and not path_status["fibre-reproduction"].get(
                "externalized_supplemental_audit_authorities"
            )
        ),
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Validate isolated FIBRE release profiles.")
    ap.add_argument("--output",type=Path)
    ap.add_argument(
        "--source-only",action="store_true",
        help=(
            "validate the source/content-address contract without requiring large "
            "scientific assets to be materialized in this checkout"
        ),
    )
    args=ap.parse_args()
    source_only=bool(args.source_only or SUBMISSION_MANIFEST.is_file())
    status=build_status(source_only=source_only)
    text=json.dumps(status,indent=2)+"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    print(text,end="")


if __name__=="__main__":
    main()
