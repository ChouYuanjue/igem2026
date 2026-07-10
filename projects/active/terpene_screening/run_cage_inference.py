from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.terpene_screening.common import (
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    coerce_text,
    identify_terpene_columns,
    read_table,
    safe_json_dump,
    write_table,
)


ENZYMECAGE_ROOT = PROJECT_ROOT / "external_repos" / "EnzymeCAGE"
FEATURE_SCRIPT = ENZYMECAGE_ROOT / "feature" / "main.py"
INFER_SCRIPT = ENZYMECAGE_ROOT / "infer.py"
CHECKPOINT_DIR = ENZYMECAGE_ROOT / "checkpoints" / "domain-specific-ft" / "terpene" / "seed_42"
MODEL_NAME = "epoch_9.pth"
ENZYMECAGE_PYTHON = os.environ.get("ENZYMECAGE_PYTHON", "/home/runnel/miniconda3/envs/enzymecage/bin/python")

DEFAULT_DATA_PATH = TERPENE_DATA_DIR / "terpene_candidate_pairs.csv"
DEFAULT_RESULT_DIR = TERPENE_RESULTS_DIR / "predictions"
DEFAULT_POCKET_DIR = TERPENE_DATA_DIR / "pockets"


def _copy_results_to_data_mirror(data_path: Path, fallback: Path | None = None) -> None:
    if data_path.exists():
        return
    if fallback is not None and fallback.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, data_path)
        return
    raise FileNotFoundError(f"Candidate pairs CSV not found at {data_path}")


def _write_infer_config(data_path: Path, result_dir: Path, pocket_dir: Path) -> Path:
    feature_root = data_path.parent / "feature"
    config = {
        "model": "EnzymeCAGE",
        "interaction_method": "geo-enhanced-interaction",
        "rxn_inner_interaction": True,
        "pocket_inner_interaction": True,
        "use_prods_info": False,
        "use_structure": True,
        "use_drfp": True,
        "use_esm": True,
        "esm_model": "ESM-C_600M",
        "batch_size": 256,
        "model_list": [MODEL_NAME],
        "data_path": str(data_path.resolve()),
        "ckpt_dir": str(CHECKPOINT_DIR.resolve()),
        "result_dir": str(result_dir.resolve()),
        "rxn_fp": str((feature_root / "reaction" / "drfp" / "rxn2fp.pkl").resolve()),
        "mol_conformation": str((feature_root / "reaction" / "molecule_conformation").resolve()),
        "reaction_center": str((feature_root / "reaction" / "reacting_center" / "reacting_center.pkl").resolve()),
        "protein_gvp_feat": str((feature_root / "protein" / "gvp_feature" / "gvp_protein_feature.pt").resolve()),
        "esm_mean_feature": str(
            (feature_root / "protein" / "ESM-C_600M" / "protein_level" / "seq2feature.pkl").resolve()
        ),
        "esm_node_feature": str(
            (feature_root / "protein" / "ESM-C_600M" / "pocket_node_feature" / "esm_node_feature.pt").resolve()
        ),
    }
    config_path = result_dir / "terpene_infer.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return config_path


def _run_feature_generation(data_path: Path, pocket_dir: Path) -> None:
    cmd = [
        ENZYMECAGE_PYTHON,
        str(FEATURE_SCRIPT),
        "--data_path",
        str(data_path),
        "--pocket_dir",
        str(pocket_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(FEATURE_SCRIPT.parent))


def _run_inference(config_path: Path) -> None:
    cmd = [
        ENZYMECAGE_PYTHON,
        str(INFER_SCRIPT),
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(ENZYMECAGE_ROOT))


def _postprocess_scores(raw_csv: Path, final_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(raw_csv)
    if "pred" not in df.columns:
        raise ValueError(f"Raw inference output missing `pred`: {raw_csv}")
    df = df.rename(columns={"pred": "cage_score"})

    cols = identify_terpene_columns(df)
    reaction_col = "reaction_id" if "reaction_id" in df.columns else cols["rhea_id"]["column"]
    if reaction_col is None:
        raise ValueError("Could not find a reaction identifier column in inference output.")

    enzyme_col = "enzyme_id" if "enzyme_id" in df.columns else cols["enzyme_id"]["column"]
    if enzyme_col is None:
        raise ValueError("Could not find an enzyme identifier column in inference output.")

    uid_col = "uniprot_id" if "uniprot_id" in df.columns else ("UniprotID" if "UniprotID" in df.columns else None)
    if uid_col is None:
        uid_col = enzyme_col

    df = df.sort_values(
        by=[reaction_col, "cage_score", uid_col, enzyme_col],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank_within_reaction"] = df.groupby(reaction_col).cumcount() + 1
    if reaction_col != "reaction_id":
        df["reaction_id"] = df[reaction_col]
    if "rhea_id" not in df.columns and cols["rhea_id"]["column"] in df.columns:
        df["rhea_id"] = df[cols["rhea_id"]["column"]]

    ordered_columns = [
        "reaction_id",
        "rhea_id",
        "enzyme_id",
        "uniprot_id",
        "UniprotID",
        "label",
        "Label",
        "cage_score",
        "rank_within_reaction",
    ]
    # Keep all existing columns, but make sure the requested ones are present first.
    leading = [column for column in ordered_columns if column in df.columns]
    trailing = [column for column in df.columns if column not in leading]
    df = df[leading + trailing]
    write_table(df, final_csv, sep=",")

    summary = {
        "raw_infer_csv": str(raw_csv),
        "final_scores_csv": str(final_csv),
        "n_pairs_scored": int(len(df)),
        "n_query_reactions_scored": int(df["reaction_id"].nunique()) if "reaction_id" in df.columns else 0,
        "n_positive_pairs_scored": int(df["label"].sum()) if "label" in df.columns else 0,
    }
    safe_json_dump(summary, final_csv.with_suffix(".json"))
    return summary


def run_cage_inference(
    data_path: Path = DEFAULT_DATA_PATH,
    result_dir: Path = DEFAULT_RESULT_DIR,
    pocket_dir: Path = DEFAULT_POCKET_DIR,
) -> dict[str, Any]:
    _copy_results_to_data_mirror(data_path, fallback=TERPENE_RESULTS_DIR / "terpene_candidate_pairs.csv")
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_infer_csv = result_dir / f"{data_path.stem}_{MODEL_NAME.replace('.pth', '.csv')}"
    final_scores_csv = result_dir / "all_pair_scores.csv"
    config_path = result_dir / "terpene_infer.yaml"

    if final_scores_csv.exists():
        return {
            "status": "reused",
            "final_scores_csv": str(final_scores_csv),
        }

    if not raw_infer_csv.exists():
        _run_feature_generation(data_path, pocket_dir)
        config_path = _write_infer_config(data_path, result_dir, pocket_dir)
        _run_inference(config_path)

    if not raw_infer_csv.exists():
        raise FileNotFoundError(f"Raw inference output missing: {raw_infer_csv}")

    summary = _postprocess_scores(raw_infer_csv, final_scores_csv)
    summary["status"] = "completed"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EnzymeCAGE inference for terpene screening.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--pocket_dir", type=str, default=str(DEFAULT_POCKET_DIR))
    args = parser.parse_args()
    summary = run_cage_inference(
        data_path=Path(args.data_path),
        result_dir=Path(args.result_dir),
        pocket_dir=Path(args.pocket_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
