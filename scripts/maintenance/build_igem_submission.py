from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
LIMIT_BYTES=50*1024*1024
MAX_SINGLE_FILE=5*1024*1024

ROOT_FILES=(
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements.lock.txt",
    "requirements-terpene-runtime.txt",
    ".gitignore",
    ".snakemake-workflow-catalog.yml",
    ".gitlab-ci.yml",
)
ROOT_DIRS=(
    "projects/active/fibre",
    "scripts",
    "frontend",
    "workflow",
    "config",
    "configs",
    "docs",
    "reproducibility",
    ".test",
    ".github",
)
EXCLUDED_PARTS={
    ".git","node_modules","__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache",".snakemake",".venv","dist",
}
EXCLUDED_SUFFIXES={".pyc",".pyo",".so",".bin"}
# Historical/project scientific assets are restored from manifests and are never
# copied into the iGEM source repository.
EXCLUDED_PREFIXES=(
    "data/",
    "results/",
    "archive/",
    "external/",
    "external_models/",
    "external_repos/",
    "external_runtime/",
)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _allowed(path: Path) -> bool:
    rel=_rel(path)
    if any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def _iter_files() -> list[Path]:
    found: dict[str,Path]={}
    for rel in ROOT_FILES:
        p=ROOT/rel
        if p.is_file():
            found[rel]=p
    for rel in ROOT_DIRS:
        base=ROOT/rel
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and _allowed(path):
                found[_rel(path)]=path
    return [found[key] for key in sorted(found)]


def _tree_bytes(root: Path, *, include_git: bool=True) -> int:
    total=0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not include_git and ".git" in path.relative_to(root).parts:
            continue
        total+=path.stat().st_size
    return total


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()


def build(output: Path, *, init_git: bool=True) -> dict:
    files=_iter_files()
    oversized=[
        (_rel(p),int(p.stat().st_size))
        for p in files if p.stat().st_size>MAX_SINGLE_FILE
    ]
    if oversized:
        raise RuntimeError(
            "iGEM source tree contains >5 MiB files; classify them as external/rebuildable assets: "
            + json.dumps(oversized[:10])
        )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for src in files:
        rel=src.relative_to(ROOT)
        dst=output/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)

    # Submission-specific ignore rules are intentionally stricter than the
    # research workspace: a future accidental local data/model file must not enter.
    with (output/".gitignore").open("a",encoding="utf-8") as fh:
        fh.write(
            "\n# iGEM official source repository hard boundary\n"
            "data/\nresults/\narchive/\nexternal/\nexternal_models/\n"
            "external_repos/\nexternal_runtime/\nnode_modules/\ndist/\n"
            "*.pt\n*.ckpt\n*.npy\n*.npz\n*.h5\n*.hdf5\n"
        )
    source_bytes=_tree_bytes(output,include_git=False)
    if source_bytes>=LIMIT_BYTES:
        raise RuntimeError(
            f"iGEM source working tree is {source_bytes/1048576:.2f} MiB; must be <50 MiB"
        )

    git_bytes=0
    commit=""
    if init_git:
        subprocess.check_call(["git","init","-b","main"],cwd=output)
        subprocess.check_call(["git","config","user.name","NJU-China iGEM"],cwd=output)
        subprocess.check_call(["git","config","user.email","software@nju-china.igem"],cwd=output)
        subprocess.check_call(["git","add","-A"],cwd=output)
        subprocess.check_call(
            ["git","commit","-m","iGEM 2026 software submission source"],
            cwd=output,
        )
        subprocess.check_call(["git","gc","--prune=now"],cwd=output)
        commit=_run(["git","rev-parse","HEAD"],output)
        git_bytes=_tree_bytes(output/".git",include_git=True)
        if git_bytes>=LIMIT_BYTES:
            raise RuntimeError(
                f"iGEM Git history is {git_bytes/1048576:.2f} MiB; must be <50 MiB"
            )
        total=_tree_bytes(output,include_git=True)
        if total>=LIMIT_BYTES:
            raise RuntimeError(
                f"iGEM checkout including .git is {total/1048576:.2f} MiB; "
                "conservative submission gate requires <50 MiB"
            )
    else:
        total=source_bytes

    manifest={
        "schema":"igem-software-submission-source-v1",
        "purpose":"official iGEM GitLab source-only repository",
        "file_count":len(files)+1,
        "source_bytes":source_bytes,
        "source_mib":round(source_bytes/1048576,3),
        "git_bytes":git_bytes,
        "git_mib":round(git_bytes/1048576,3),
        "checkout_bytes":total,
        "checkout_mib":round(total/1048576,3),
        "limit_bytes":LIMIT_BYTES,
        "under_50_mib":bool(total<LIMIT_BYTES),
        "git_commit":commit,
        "excluded_scientific_asset_roots":list(EXCLUDED_PREFIXES),
        "large_asset_policy":(
            "large data/model/result assets are content-addressed by release manifests "
            "and restored/built outside the official source Git history"
        ),
        "research_workspace_inventory_note":(
            "reproducibility/research_release_manifest.json inventories the full "
            "research workspace and its historical/direct scientific assets; its "
            "direct_git_bytes field does not describe this source-only iGEM repository"
        ),
        "reproduction_materialization":(
            "source-only contract is validated here; frozen metric reproduction "
            "requires restoration of the externalized hash-addressed scientific assets"
        ),
    }
    (output/"IGEM_SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8"
    )
    if init_git:
        subprocess.check_call(["git","add","IGEM_SUBMISSION_MANIFEST.json"],cwd=output)
        subprocess.check_call(["git","commit","-m","Record submission size manifest"],cwd=output)
        subprocess.check_call(["git","gc","--prune=now"],cwd=output)
        # Final repository metrics are returned to the caller. They are not
        # written back into the committed manifest, because doing so would make
        # the freshly created submission repository dirty by construction.
        manifest["final_git_commit"]=_run(["git","rev-parse","HEAD"],output)
        manifest["final_git_bytes"]=_tree_bytes(output/".git",include_git=True)
        manifest["final_git_mib"]=round(manifest["final_git_bytes"]/1048576,3)
        manifest["final_checkout_bytes"]=_tree_bytes(output,include_git=True)
        manifest["final_checkout_mib"]=round(
            manifest["final_checkout_bytes"]/1048576,3
        )
        manifest["under_50_mib"]=bool(manifest["final_checkout_bytes"]<LIMIT_BYTES)
        if not manifest["under_50_mib"]:
            raise RuntimeError("final iGEM repository exceeded 50 MiB")
        if _run(["git","status","--porcelain"],output):
            raise RuntimeError("generated iGEM submission repository is dirty")
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Build a clean <50 MiB source repository for the official iGEM GitLab."
    )
    ap.add_argument(
        "--output",type=Path,default=ROOT/"dist/igem-gitlab",
        help="destination; it is replaced atomically as a staging tree",
    )
    ap.add_argument("--no-init-git",action="store_true")
    args=ap.parse_args()
    result=build(args.output.resolve(),init_git=not args.no_init_git)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
