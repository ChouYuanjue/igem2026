from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _method_from_baseline(baseline: dict[str, Any]) -> str:
    if baseline.get("data_mode") == "official_eval":
        return "official_eval"
    source = baseline["pocket_source"]
    selection = baseline["pocket_selection"]
    if source == "enzymecage_official_or_p2rank":
        return "official_or_p2rank_top1"
    if source == "p2rank+fpocket":
        return "p2rank_fpocket_union"
    return f"{source}_{selection}"


def build_config(matrix: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    defaults = matrix["defaults"]
    name = baseline["name"]
    data_mode = baseline.get("data_mode", defaults["data"].get("data_mode", "demo_mining"))
    run_id = f"demo_{name}" if data_mode == "demo_mining" else name
    source = baseline.get("pocket_source", "official")

    if source == "p2rank+fpocket":
        union_sources = ["p2rank", "fpocket"]
    elif source == "enzymecage_official_or_p2rank":
        union_sources = None
    else:
        union_sources = [source]

    pocket: dict[str, Any] = {
        "method": _method_from_baseline(baseline),
        "source": source,
        "selection": baseline.get("pocket_selection", "none"),
        "top_k": int(baseline.get("top_k", 1)),
    }
    if "top_k_each" in baseline:
        pocket["top_k_each"] = int(baseline["top_k_each"])
    if union_sources:
        pocket["union_sources"] = union_sources
    if "prior_source" in baseline:
        pocket["prior_source"] = baseline["prior_source"]

    aggregation: dict[str, Any] = {
        "method": baseline.get("aggregation", "none"),
    }
    if "temperature" in baseline:
        aggregation["temperature"] = baseline["temperature"]
    if "source_weights" in baseline:
        aggregation["source_weights"] = baseline["source_weights"]

    config = {
        "project": {
            "name": run_id,
            "run_id": run_id,
            "seed": defaults["seed"],
            "baseline_name": name,
            "purpose": baseline.get("purpose", ""),
            "status": baseline.get("status", "active"),
            "data_mode": data_mode,
        },
        "external": {
            "enzymecage_root": defaults["external"]["enzymecage_root"],
            "p2rank_home": defaults["external"]["p2rank_home"],
            "fpocket_bin": defaults["external"]["fpocket_bin"],
        },
        "data": {
            "data_mode": data_mode,
            "reaction_csv": defaults["data"]["reaction_csv"],
            "structure_dir": defaults["data"]["structure_dir"],
            "working_dir": f"data/pocket_runs/{run_id}",
            "output_dir": f"results/pocket/{run_id}",
        },
        "model": {
            "checkpoint_dir": defaults["model"]["checkpoint_dir"],
            "checkpoint_name": baseline.get("checkpoint_name", defaults["model"]["checkpoint_name"]),
        },
        "pocket": pocket,
        "aggregation": aggregation,
        "evaluation": {
            "label_csv": defaults["data"].get("label_csv"),
            "topk": defaults["evaluation"]["topk"],
        },
    }
    if "config_source" in baseline:
        config["data"]["config_source"] = baseline["config_source"]
    if "source_dataset" in baseline:
        config["data"]["source_dataset"] = baseline["source_dataset"]
    if "n_reactions" in baseline:
        config["data"]["n_reactions"] = int(baseline["n_reactions"])
    if "n_enzymes_per_reaction" in baseline:
        config["data"]["n_enzymes_per_reaction"] = int(baseline["n_enzymes_per_reaction"])
    if "allow_checkpoint_substitution" in baseline:
        config["model"]["allow_checkpoint_substitution"] = bool(baseline["allow_checkpoint_substitution"])
    return config


def generate_configs(matrix_path: Path, output_dir: Path) -> list[Path]:
    matrix = _load_yaml(matrix_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for baseline in matrix["baselines"]:
        config = build_config(matrix, baseline)
        path = output_dir / f"demo_{baseline['name']}.yaml"
        print(f"[write] {path}")
        _write_yaml(config, path)
        generated.append(path)
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-baseline configs from baseline_matrix.yaml.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generated = generate_configs(Path(args.matrix), Path(args.output_dir))
    print(f"[done] Generated {len(generated)} baseline configs.")


if __name__ == "__main__":
    main()
