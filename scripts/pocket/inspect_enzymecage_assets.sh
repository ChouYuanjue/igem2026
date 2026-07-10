#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

"${PYTHON}" - <<'PY'
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

root = Path.cwd()
enz = root / "external_repos/EnzymeCAGE"
dataset = enz / "dataset"
checkpoints = enz / "checkpoints"
out_json = root / "results/pocket/enzymecage_asset_tree_summary.json"
out_md = root / "results/pocket/enzymecage_asset_tree_summary.md"
out_json.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def count_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return None


def safe_preview(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": rel(path),
        "size_bytes": path.stat().st_size,
        "n_rows": count_rows(path),
        "columns": [],
        "preview_rows": [],
        "error": None,
    }
    try:
        df = pd.read_csv(path, nrows=3)
        info["columns"] = [str(col) for col in df.columns]
        rows = []
        for row in df.fillna("").to_dict("records"):
            rows.append({str(k): str(v)[:240] for k, v in row.items()})
        info["preview_rows"] = rows
    except Exception as exc:  # noqa: BLE001 - inspection should keep going
        info["error"] = str(exc)
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                info["columns"] = next(reader, [])
                info["preview_rows"] = [
                    {f"col_{idx}": value[:240] for idx, value in enumerate(row)}
                    for _, row in zip(range(3), reader)
                ]
        except Exception as inner_exc:  # noqa: BLE001
            info["error"] = f"{exc}; fallback_csv_error={inner_exc}"
    return info


patterns = [
    "Enzyme-405",
    "Orphan-335",
    "all_enzymes",
    "mining",
    "candidate",
    "rxn2uids",
    "rhea_rxn2uids",
]
csvs: list[Path] = []
if dataset.exists():
    for path in dataset.rglob("*.csv"):
        name = path.name.lower()
        if any(pattern.lower() in name for pattern in patterns):
            csvs.append(path)
csvs = sorted(set(csvs))

pth_files = sorted(checkpoints.rglob("*.pth")) if checkpoints.exists() else []
required_dataset_paths = {
    "dataset/internal-test-set": dataset / "internal-test-set",
    "dataset/internal-test-set/Enzyme-405": dataset / "internal-test-set/Enzyme-405",
    "dataset/internal-test-set/Orphan-335": dataset / "internal-test-set/Orphan-335",
    "dataset/RHEA/2023-07-12": dataset / "RHEA/2023-07-12",
    "dataset/RHEA/2025-02-05": dataset / "RHEA/2025-02-05",
    "dataset/RHEA/2025-02-05/pockets/pocket": dataset / "RHEA/2025-02-05/pockets/pocket",
    "dataset/demo": dataset / "demo",
}

payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "enzymecage_root": rel(enz),
    "checkpoint": {
        "epoch_19_seed42_present": (checkpoints / "pretrain/seed_42/epoch_19.pth").exists(),
        "best_model_seed42_present": (checkpoints / "pretrain/seed_42/best_model.pth").exists(),
        "pth_files": [rel(path) for path in pth_files],
    },
    "dataset": {key: path.exists() for key, path in required_dataset_paths.items()},
    "candidate_input_csvs": [safe_preview(path) for path in csvs],
}
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# EnzymeCAGE Asset Tree Summary",
    "",
    f"- generated_at: {payload['timestamp']}",
    f"- enzymecage_root: `{payload['enzymecage_root']}`",
    "",
    "## Checkpoints",
    "",
    f"- seed_42 `epoch_19.pth`: {payload['checkpoint']['epoch_19_seed42_present']}",
    f"- seed_42 `best_model.pth`: {payload['checkpoint']['best_model_seed42_present']}",
    f"- total .pth files: {len(payload['checkpoint']['pth_files'])}",
]
for path in payload["checkpoint"]["pth_files"]:
    lines.append(f"  - `{path}`")

lines.extend(["", "## Dataset Paths", ""])
for key, exists in payload["dataset"].items():
    lines.append(f"- `{key}`: {exists}")

lines.extend(["", "## Candidate Input CSVs", ""])
if not payload["candidate_input_csvs"]:
    lines.append("No matching CSV files found.")
for item in payload["candidate_input_csvs"]:
    lines.extend(
        [
            f"### `{item['path']}`",
            f"- size_bytes: {item['size_bytes']}",
            f"- n_rows: {item['n_rows']}",
            f"- columns: {', '.join(item['columns']) if item['columns'] else 'unknown'}",
            "",
            "```json",
            json.dumps(item["preview_rows"], indent=2),
            "```",
            "",
        ]
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_json)
print(out_md)
PY
