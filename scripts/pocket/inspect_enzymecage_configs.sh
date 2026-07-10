#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

"${PYTHON}" - <<'PY'
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

root = Path.cwd()
enz = root / "external_repos/EnzymeCAGE"
out_json = root / "results/pocket/enzymecage_config_summary.json"
out_md = root / "results/pocket/enzymecage_config_summary.md"
out_json.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_config_path(value: str | None) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else enz / path


def list_model_names(config: dict[str, Any]) -> list[str]:
    raw = config.get("model_list") or config.get("model_name") or config.get("checkpoint_name")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def summarize_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return {"config_path": rel(path), "error": str(exc)}

    ckpt_dir = resolve_config_path(config.get("ckpt_dir") or config.get("checkpoint_dir"))
    model_names = list_model_names(config)
    referenced = {
        "data_path": resolve_config_path(config.get("data_path")),
        "checkpoint_dir": ckpt_dir,
        "result_dir": resolve_config_path(config.get("result_dir")),
        "pocket_dir": resolve_config_path(config.get("pocket_dir")),
        "rxn_fp": resolve_config_path(config.get("rxn_fp")),
        "mol_conformation": resolve_config_path(config.get("mol_conformation")),
        "reaction_center": resolve_config_path(config.get("reaction_center")),
        "protein_gvp_feat": resolve_config_path(config.get("protein_gvp_feat")),
        "esm_mean_feature": resolve_config_path(config.get("esm_mean_feature")),
        "esm_node_feature": resolve_config_path(config.get("esm_node_feature")),
    }
    checkpoint_files = []
    if ckpt_dir is not None:
        for name in model_names:
            checkpoint_files.append(ckpt_dir / name)

    path_exists = {key: (value.exists() if value is not None else None) for key, value in referenced.items()}
    checkpoint_exists = {str(path): path.exists() for path in checkpoint_files}
    status = "ok"
    if "config/demo" in rel(path) and "best_model.pth" in model_names and not checkpoint_exists.get(str(ckpt_dir / "best_model.pth"), False):
        status = "stale_demo_config_or_missing_demo_assets"
    elif any(value is False for value in path_exists.values()) or any(value is False for value in checkpoint_exists.values()):
        status = "referenced_paths_missing"

    return {
        "config_path": rel(path),
        "status": status,
        "referenced_data_path": str(referenced["data_path"]) if referenced["data_path"] else None,
        "referenced_checkpoint_dir": str(ckpt_dir) if ckpt_dir else None,
        "referenced_model_names": model_names,
        "referenced_output_path": str(referenced["result_dir"]) if referenced["result_dir"] else None,
        "referenced_pocket_dir": str(referenced["pocket_dir"]) if referenced["pocket_dir"] else None,
        "referenced_feature_paths": {
            key: str(value) for key, value in referenced.items()
            if key in {"rxn_fp", "mol_conformation", "reaction_center", "protein_gvp_feat", "esm_mean_feature", "esm_node_feature"} and value is not None
        },
        "path_exists": {key: value for key, value in path_exists.items()},
        "checkpoint_files_exist": checkpoint_exists,
    }


paths = []
for base in [enz / "config/infer", enz / "config/demo"]:
    if base.exists():
        paths.extend(sorted(base.rglob("*.yaml")))

summaries = [summarize_config(path) for path in paths]
payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "enzymecage_root": rel(enz),
    "configs": summaries,
}
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# EnzymeCAGE Config Summary",
    "",
    f"- generated_at: {payload['timestamp']}",
    f"- enzymecage_root: `{payload['enzymecage_root']}`",
    "",
]
for item in summaries:
    lines.extend(
        [
            f"## `{item.get('config_path')}`",
            "",
            f"- status: {item.get('status')}",
            f"- data_path: `{item.get('referenced_data_path')}`",
            f"- checkpoint_dir: `{item.get('referenced_checkpoint_dir')}`",
            f"- model/checkpoint names: {item.get('referenced_model_names')}",
            f"- output path: `{item.get('referenced_output_path')}`",
            f"- pocket_dir: `{item.get('referenced_pocket_dir')}`",
            "",
            "### Path Existence",
            "",
            "```json",
            json.dumps(item.get("path_exists", {}), indent=2),
            "```",
            "",
            "### Checkpoints",
            "",
            "```json",
            json.dumps(item.get("checkpoint_files_exist", {}), indent=2),
            "```",
            "",
        ]
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_json)
print(out_md)
PY
