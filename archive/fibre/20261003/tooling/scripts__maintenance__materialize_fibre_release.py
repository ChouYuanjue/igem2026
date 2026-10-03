from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[2]
MANIFEST_DIR=ROOT/"projects/active/fibre/release/manifests"
PROFILE_DIR=ROOT/"projects/active/fibre/release/profiles"
RESEARCH_RELEASE=ROOT/"reproducibility/research_release_manifest.json"


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode=="hardlink":
        try:
            os.link(src,dst)
            return
        except OSError:
            shutil.copy2(src,dst)
            return
    if mode=="copy":
        shutil.copy2(src,dst)
        return
    raise ValueError(mode)


def _load_manifest(pid: str) -> dict:
    path=MANIFEST_DIR/f"{pid}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} missing; run build_fibre_release_manifests.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_profile(pid: str) -> tuple[Path,dict]:
    filename={
        "fibre-method":"method.yaml",
        "fibre-reproduction":"reproduction.yaml",
        "starase-application":"application.yaml",
    }[pid]
    path=PROFILE_DIR/filename
    return path,yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _record_paths(rows) -> set[str]:
    return {str(x["path"]) for x in rows}


def _research_direct_assets() -> set[str]:
    x=json.loads(RESEARCH_RELEASE.read_text(encoding="utf-8"))
    rows=x.get("direct_git_assets") or []
    out=set()
    for row in rows:
        if isinstance(row,str):
            out.add(row)
        elif isinstance(row,dict) and row.get("path"):
            out.add(str(row["path"]))
    return out


def _application_asset_files(profile: dict) -> set[str]:
    out=set()
    assets=profile.get("assets") or {}
    for key in ("current_information_roots","tps_specialist_source_weights"):
        for rel in assets.get(key) or []:
            root=ROOT/str(rel)
            if not root.exists():
                raise FileNotFoundError(root)
            if root.is_file():
                out.add(str(root.relative_to(ROOT)))
            else:
                for path in root.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        out.add(str(path.relative_to(ROOT)))
    # Include already-materialized application outputs when present; they are
    # rebuildable and therefore not required merely to inspect/materialize source.
    for rel in assets.get("generated_application_assets") or []:
        root=ROOT/str(rel)
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    out.add(str(path.relative_to(ROOT)))
    return out


def resolve_package_paths(pid: str) -> tuple[set[str],dict,dict]:
    manifest=_load_manifest(pid)
    profile_path,profile=_load_profile(pid)
    paths=_record_paths(manifest.get("source_files") or [])
    # Dependency source is physically present in standalone staging while
    # remaining semantically owned by the method package.
    for dep in manifest.get("depends_on") or []:
        dep_manifest=_load_manifest(str(dep))
        paths |= _record_paths(dep_manifest.get("source_files") or [])

    if pid=="fibre-reproduction":
        paths |= _record_paths(manifest.get("claim_authorities") or [])
        paths |= _record_paths(
            (manifest.get("assets") or {}).get("project_owned_weights") or []
        )
        paths |= _research_direct_assets()
        paths.add("reproducibility/research_release_manifest.json")
        paths.add("reproducibility/bime_rank/model_assets.json")
    elif pid=="starase-application":
        paths |= _application_asset_files(profile)
    elif pid=="fibre-method":
        bad=[
            p for p in paths
            if p.startswith(("data/","results/","reproducibility/bime_rank/"))
        ]
        if bad:
            raise ValueError(f"method package accidentally contains project assets: {bad[:10]}")

    if pid!="starase-application":
        bad=[p for p in paths if p.startswith("projects/active/fibre/application/")]
        if bad:
            raise ValueError(f"{pid} accidentally contains application source: {bad[:10]}")
    if pid=="starase-application":
        bad=[p for p in paths if p.startswith("reproducibility/bime_rank/")]
        if bad:
            raise ValueError(f"application package accidentally contains reproduction source: {bad[:10]}")

    # Only existing files are materializable. External third-party assets stay
    # represented by restore contracts in the resolved manifest.
    missing=[p for p in sorted(paths) if not (ROOT/p).is_file()]
    if missing:
        raise FileNotFoundError(f"{pid}: package file missing: {missing[:10]}")
    return paths,manifest,profile


def materialize(pid: str, output: Path, mode: str, *, dry_run: bool) -> dict:
    paths,manifest,profile=resolve_package_paths(pid)
    total=sum(int((ROOT/p).stat().st_size) for p in paths)
    summary={
        "schema":"fibre-materialized-release-v1",
        "package_id":pid,
        "file_count":len(paths),
        "logical_bytes":total,
        "link_mode":mode,
        "dry_run":bool(dry_run),
        "output":str(output),
    }
    if dry_run:
        return summary
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True)
    for rel in sorted(paths):
        _copy_or_link(ROOT/rel,output/rel,mode)
    # Put release metadata at the staging root as well as preserving repository paths.
    shutil.copy2(MANIFEST_DIR/f"{pid}.json",output/"PACKAGE_MANIFEST.json")
    profile_filename={
        "fibre-method":"method.yaml",
        "fibre-reproduction":"reproduction.yaml",
        "starase-application":"application.yaml",
    }[pid]
    shutil.copy2(PROFILE_DIR/profile_filename,output/"PACKAGE_PROFILE.yaml")
    shutil.copy2(
        ROOT/"projects/active/fibre/release/asset_catalog.yaml",
        output/"ASSET_CATALOG.yaml",
    )
    (output/"MATERIALIZATION.json").write_text(
        json.dumps(summary,indent=2)+"\n",encoding="utf-8"
    )
    return summary


def main() -> None:
    ap=argparse.ArgumentParser(
        description=(
            "Materialize one isolated FIBRE/Starase release profile. Hardlinks "
            "avoid duplicate storage while each staged tree remains independently auditable."
        )
    )
    ap.add_argument(
        "--profile",
        required=True,
        choices=["fibre-method","fibre-reproduction","starase-application"],
    )
    ap.add_argument("--output-dir",type=Path)
    ap.add_argument("--mode",choices=["hardlink","copy"],default="hardlink")
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    output=(
        args.output_dir.resolve()
        if args.output_dir
        else (ROOT/"dist/releases"/args.profile).resolve()
    )
    print(json.dumps(materialize(
        args.profile,output,args.mode,dry_run=args.dry_run
    ),indent=2))


if __name__=="__main__":
    main()
