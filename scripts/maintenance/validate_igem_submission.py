from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
LIMIT=50*1024*1024


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def remotes() -> list[str]:
    try:
        raw=subprocess.check_output(
            ["git","remote","-v"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    return [line for line in raw.splitlines() if line.strip()]


def main() -> None:
    required=["README.md","LICENSE","requirements.lock.txt",".gitlab-ci.yml"]
    missing=[x for x in required if not (ROOT/x).is_file()]
    if missing:
        raise SystemExit(f"missing required iGEM files: {missing}")

    license_text=(ROOT/"LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License\n"):
        raise SystemExit("LICENSE is not the selected MIT license")

    readme=(ROOT/"README.md").read_text(encoding="utf-8").lower()
    readme_terms={
        "purpose":["what this software does","software does"],
        "audience":["who this is for","who is it for"],
        "install":["install"],
        "run":["run starase navigator","run the software"],
        "reproduce":["reproduce the main results","reproduce main results"],
        "igem_gitlab":["igem gitlab","gitlab.igem.org"],
    }
    missing_sections=[
        name for name,terms in readme_terms.items()
        if not any(term in readme for term in terms)
    ]
    if missing_sections:
        raise SystemExit(f"README award sections missing: {missing_sections}")

    bad_lock=[]
    for raw in (ROOT/"requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or any(op in line for op in (">=","<=", "~=", "!=", ">", "<")):
            bad_lock.append(line)
    if bad_lock:
        raise SystemExit(f"requirements.lock.txt contains non-exact pins: {bad_lock[:10]}")

    remote_lines=remotes()
    official=any("gitlab.igem.org" in line for line in remote_lines)
    submission_tree=(ROOT/"IGEM_SUBMISSION_MANIFEST.json").is_file()
    if official or submission_tree:
        tracked=subprocess.check_output(
            ["git","ls-files"],cwd=ROOT,text=True
        ).splitlines()
        forbidden_prefixes=(
            "data/","results/","archive/","external/","external_models/",
            "external_repos/","external_runtime/",
        )
        forbidden_suffixes=(".pt",".ckpt",".npy",".npz",".h5",".hdf5",".bin")
        forbidden=[
            path for path in tracked
            if path.startswith(forbidden_prefixes)
            or path.lower().endswith(forbidden_suffixes)
        ]
        if forbidden:
            raise SystemExit(
                "official/source-only iGEM tree tracks scientific bulk assets: "
                + ", ".join(forbidden[:20])
            )
    git_bytes=directory_bytes(ROOT/".git") if (ROOT/".git").is_dir() else 0
    if (official or submission_tree) and git_bytes>=LIMIT:
        raise SystemExit(
            f"official iGEM Git history is {git_bytes/1048576:.2f} MiB; must be <50 MiB"
        )

    status={
        "schema":"igem-software-award-check-v1",
        "license":"MIT",
        "readme_contract":"present",
        "dependency_lock":"requirements.lock.txt",
        "dependency_pins":"exact",
        "official_igem_remote_detected":official,
        "source_only_submission_tree":submission_tree,
        "forbidden_scientific_assets_tracked":0 if (official or submission_tree) else None,
        "git_history_mib":round(git_bytes/1048576,3),
        "git_history_under_50_mib":bool(git_bytes<LIMIT),
        "hosting_note":(
            "official iGEM GitLab remote detected"
            if official else
            "working research checkout is not the official iGEM GitLab; build/push dist/igem-gitlab"
        ),
    }
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
