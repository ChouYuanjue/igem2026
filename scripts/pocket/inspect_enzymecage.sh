#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENZYMECAGE_ROOT="${ROOT_DIR}/external_repos/EnzymeCAGE"
OUTPUT_JSON="${ROOT_DIR}/results/pocket/enzymecage_inspection.json"
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

"${PYTHON}" - "${ROOT_DIR}" "${ENZYMECAGE_ROOT}" "${OUTPUT_JSON}" <<'PY'
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

root_dir = Path(sys.argv[1])
enzymecage_root = Path(sys.argv[2])
output_json = Path(sys.argv[3])

required_paths = [
    "README.md",
    "infer.py",
    "feature/main.py",
    "scripts/prepare_mining_input.py",
    "scripts/extract_p2rank_pockets.py",
    "dataset/demo/reaction.csv",
    "dataset/demo/structures",
    "checkpoints/pretrain/seed_42/best_model.pth",
]

script_paths = [
    "scripts/prepare_mining_input.py",
    "scripts/extract_p2rank_pockets.py",
    "feature/main.py",
    "infer.py",
    "scripts/run_mining_pipeline.py",
]


def run_capture(command: list[str], cwd: Path | None = None, timeout: int = 20) -> dict[str, object]:
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def git_value(args: list[str]) -> str | None:
    if not enzymecage_root.exists():
        return None
    result = run_capture(["git", "-C", str(enzymecage_root), *args], timeout=10)
    if result["returncode"] == 0:
        value = str(result["stdout"]).strip()
        return value or None
    return None


def extract_demo_commands(readme_text: str) -> list[str]:
    commands = []
    for block in re.findall(r"```(?:shell|bash)?\n(.*?)```", readme_text, flags=re.S):
        if any(token in block for token in ["dataset/demo", "run_mining_pipeline", "run_demo_mining"]):
            cleaned = "\n".join(line.rstrip() for line in block.strip().splitlines())
            commands.append(cleaned)
    return commands


inspection: dict[str, object] = {
    "enzymecage_exists": enzymecage_root.exists(),
    "current_commit": git_value(["rev-parse", "HEAD"]),
    "current_branch": git_value(["branch", "--show-current"]),
    "required_files_present": {},
    "missing_files": [],
    "checkpoint_present": False,
    "checkpoint_missing": True,
    "available_demo_files": [],
    "suspected_demo_commands_from_readme": [],
    "script_help_outputs": {},
    "warnings": [],
}

if not enzymecage_root.exists():
    inspection["warnings"].append(f"EnzymeCAGE root missing: {enzymecage_root}")
    output_json.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    print(output_json)
    raise SystemExit(0)

for relative in required_paths:
    path = enzymecage_root / relative
    present = path.exists()
    inspection["required_files_present"][relative] = present
    if not present:
        inspection["missing_files"].append(relative)

checkpoint = enzymecage_root / "checkpoints/pretrain/seed_42/best_model.pth"
inspection["checkpoint_present"] = checkpoint.exists()
inspection["checkpoint_missing"] = not checkpoint.exists()
if not checkpoint.exists():
    inspection["warnings"].append("Checkpoint missing: checkpoints/pretrain/seed_42/best_model.pth")

demo_dir = enzymecage_root / "dataset/demo"
if demo_dir.exists():
    inspection["available_demo_files"] = [
        str(path.relative_to(enzymecage_root))
        for path in sorted(demo_dir.rglob("*"))
        if path.is_file()
    ]
else:
    inspection["warnings"].append("Demo data missing: dataset/demo")

readme_path = enzymecage_root / "README.md"
if readme_path.exists():
    readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
    inspection["suspected_demo_commands_from_readme"] = extract_demo_commands(readme_text)
else:
    inspection["warnings"].append("README.md missing")

for relative in script_paths:
    script = enzymecage_root / relative
    if script.exists():
        inspection["script_help_outputs"][relative] = run_capture(
            [sys.executable, str(script), "--help"],
            cwd=enzymecage_root,
        )
    else:
        inspection["script_help_outputs"][relative] = {
            "command": [sys.executable, str(script), "--help"],
            "returncode": None,
            "stdout": "",
            "stderr": "script missing",
        }

for relative, help_output in inspection["script_help_outputs"].items():
    if help_output["returncode"] not in (0, None):
        stderr = str(help_output.get("stderr", "")).strip().splitlines()
        if stderr:
            inspection["warnings"].append(f"{relative} --help failed: {stderr[-1]}")

output_json.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
print(output_json)
PY
